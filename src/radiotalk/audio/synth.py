from __future__ import annotations

import hashlib
import io
import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..normalize import normalize
from .assign import assign, unique_speakers
from .encode import encode_audio
from .model import (
    NUM_TRANSITION_STEPS,
    LoadedTada,
    make_locked_inference_options,
)
from .schema import AudioBytes, AudioTurnRow, SAMPLE_RATE
from .staging import Staging, decode_tokens, encode_tokens
from .tokens import extract_token_spans
from .writer import AudioShardWriter


FAILURES_NAME = "failures.jsonl"
MAX_SYNTH_ATTEMPTS = 3
MAX_WPM = 300.0
MIN_OOM_BATCH = 4


def _generate_oom_safe(loaded, prompt, texts, opts):
    audios: list = []
    times: list = []
    pos = 0
    sub = len(texts)
    while pos < len(texts):
        end = min(pos + sub, len(texts))
        try:
            output = loaded.model.generate(
                prompt=prompt, text=texts[pos:end],
                num_transition_steps=NUM_TRANSITION_STEPS,
                inference_options=opts,
            )
            audios.extend(output.audio)
            times.extend(output.time_before)
            pos = end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if sub <= MIN_OOM_BATCH:
                for _ in range(end - pos):
                    audios.append(None)
                    times.append(torch.tensor([]))
                pos = end
            else:
                sub = max(sub // 2, MIN_OOM_BATCH)
    return audios, times


@dataclass
class SynthStats:
    voices_processed: int = 0
    turns_ok: int = 0
    turns_failed: int = 0
    turns_by_attempt: dict[int, int] | None = None

    def record_attempt(self, attempt: int) -> None:
        if self.turns_by_attempt is None:
            self.turns_by_attempt = {}
        self.turns_by_attempt[attempt] = self.turns_by_attempt.get(attempt, 0) + 1


@dataclass
class _QueueItem:
    scenario_id: str
    turn_idx: int
    text_normalized: str
    last_reason: str = ""


def synth_config_fingerprint(model_id: str, opts) -> str:
    payload = {
        "tts_model": model_id,
        "num_flow_matching_steps": opts.num_flow_matching_steps,
        "num_acoustic_candidates": opts.num_acoustic_candidates,
        "scorer": opts.scorer,
        "speed_up_factor": opts.speed_up_factor,
        "acoustic_cfg_scale": opts.acoustic_cfg_scale,
        "duration_cfg_scale": opts.duration_cfg_scale,
        "noise_temperature": opts.noise_temperature,
        "num_transition_steps": NUM_TRANSITION_STEPS,
        "sample_rate": SAMPLE_RATE,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def gather_voice_queue(
    transcripts_ds, voice_ids: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, list[_QueueItem]]]:
    assignments: dict[str, dict[str, str]] = {}
    voice_queue: dict[str, list[_QueueItem]] = defaultdict(list)
    for tr in transcripts_ds:
        sid = tr["scenario_id"]
        a = assign(sid, unique_speakers(tr["turns"]), voice_ids)
        assignments[sid] = a
        for ti, turn in enumerate(tr["turns"]):
            voice_queue[a[turn["speaker"]]].append(
                _QueueItem(
                    scenario_id=sid,
                    turn_idx=ti,
                    text_normalized=normalize(turn["text"]),
                ),
            )
    return assignments, voice_queue


def synthesize_voice_queue(
    loaded: LoadedTada,
    voice_cache: dict[str, object],
    voice_queue: dict[str, list[_QueueItem]],
    staging: Staging,
    *,
    max_batch: int = 256,
    on_voice_done: Callable[[str, int], None] | None = None,
    on_turn_ok: Callable[[int], None] | None = None,
    log_failure: Callable[[str, str], None] | None = None,
) -> SynthStats:
    stats = SynthStats()
    opts = make_locked_inference_options()
    already_done = staging.voices_done()
    tokenizer = loaded.tokenizer

    for voice_id, items in voice_queue.items():
        if voice_id in already_done:
            stats.voices_processed += 1
            continue
        if voice_id not in voice_cache:
            continue

        rows_buffer: list[tuple] = []
        pending: list[_QueueItem] = list(items)

        with torch.inference_mode():
            for attempt in range(MAX_SYNTH_ATTEMPTS):
                if not pending:
                    break
                next_pending: list[_QueueItem] = []
                for chunk_start in range(0, len(pending), max_batch):
                    chunk = pending[chunk_start : chunk_start + max_batch]
                    texts = [it.text_normalized for it in chunk]
                    audios, times = _generate_oom_safe(
                        loaded, voice_cache[voice_id], texts, opts,
                    )
                    for it, audio, time_before in zip(chunk, audios, times):
                        if audio is None:
                            it.last_reason = "no audio returned"
                            next_pending.append(it)
                            continue
                        expected = len(tokenizer.encode(
                            it.text_normalized, add_special_tokens=False,
                        ))
                        if len(time_before) < expected:
                            it.last_reason = (
                                f"token count {len(time_before)} < {expected}"
                            )
                            next_pending.append(it)
                            continue
                        arr = audio.detach().cpu().float().numpy().squeeze()
                        duration_s = arr.shape[-1] / SAMPLE_RATE
                        n_words = len(it.text_normalized.split())
                        wpm = n_words * 60.0 / duration_s if duration_s > 0 else float("inf")
                        if wpm > MAX_WPM:
                            it.last_reason = f"wpm {wpm:.0f} > {MAX_WPM:.0f}"
                            next_pending.append(it)
                            continue
                        spans = extract_token_spans(
                            it.text_normalized,
                            time_before.detach().cpu().tolist(),
                            tokenizer,
                            duration_s,
                        )
                        rows_buffer.append((
                            it.scenario_id, it.turn_idx, voice_id,
                            it.text_normalized,
                            encode_audio(arr),
                            encode_tokens(spans),
                        ))
                        stats.turns_ok += 1
                        stats.record_attempt(attempt)
                        if on_turn_ok is not None:
                            on_turn_ok(stats.turns_ok)
                    if len(rows_buffer) >= 1024:
                        staging.insert_turns(rows_buffer)
                        rows_buffer.clear()
                pending = next_pending
            if loaded.device == "cuda":
                torch.cuda.empty_cache()

        for it in pending:
            stats.turns_failed += 1
            if log_failure is not None:
                reason = it.last_reason or "validation failed"
                log_failure(
                    it.scenario_id,
                    f"turn {it.turn_idx}: {reason} after "
                    f"{MAX_SYNTH_ATTEMPTS} synthesis attempts",
                )
        if rows_buffer:
            staging.insert_turns(rows_buffer)
        staging.mark_voice_done(voice_id, _now_iso())
        stats.voices_processed += 1
        if on_voice_done is not None:
            on_voice_done(voice_id, stats.voices_processed)
    return stats


def assemble_shards(
    transcripts_ds,
    staging: Staging,
    writer: AudioShardWriter,
    tts_model_id: str,
    *,
    log_failure: Callable[[str, str], None] | None = None,
) -> tuple[int, int]:
    emitted = 0
    failed = 0
    skip = writer.total_rows
    progressed = 0
    for tr in transcripts_ds:
        sid = tr["scenario_id"]
        n_turns = len(tr["turns"])
        if progressed + n_turns <= skip:
            progressed += n_turns
            continue
        staged = staging.turns_for_scenario(sid)
        if len(staged) != n_turns:
            failed += 1
            if log_failure is not None:
                log_failure(
                    sid,
                    f"missing audio for {n_turns - len(staged)} of {n_turns} turns",
                )
            progressed += n_turns
            continue
        for st in staged:
            turn = tr["turns"][st.turn_idx]
            writer.add(AudioTurnRow(
                scenario_id=sid,
                scenario=tr["scenario"],
                turn_idx=st.turn_idx,
                speaker=turn["speaker"],
                text=turn["text"],
                text_normalized=st.text_normalized,
                voice_id=st.voice_id,
                audio=AudioBytes(bytes=st.audio_bytes),
                tokens=decode_tokens(st.tokens_json),
                model=tr["model"],
                prompt_version=tr["prompt_version"],
                taxonomy_version=tr["taxonomy_version"],
                tts_model=tts_model_id,
            ))
        emitted += 1
        progressed += n_turns
    return emitted, failed


def append_failure(out_dir: Path, scenario_id: str, reason: str) -> None:
    path = out_dir / FAILURES_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({
            "scenario_id": scenario_id,
            "reason": reason,
            "at": _now_iso(),
        }) + "\n")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def encode_voice_pool(
    loaded: LoadedTada,
    voices_ds,
    voice_ids_needed: Iterable[str],
) -> dict[str, object]:
    voice_index = {vid: i for i, vid in enumerate(voices_ds["voice_id"])}
    cache: dict[str, object] = {}
    for vid in dict.fromkeys(voice_ids_needed):
        row = voices_ds[voice_index[vid]]
        arr, sr = _decode_audio_field(row["audio"])
        cache[vid] = loaded.encode_voice(
            torch.tensor(arr, dtype=torch.float32), sample_rate=sr,
        )
    if loaded.device == "cuda":
        torch.cuda.empty_cache()
    return cache


def _decode_audio_field(audio_field: Any) -> tuple[np.ndarray, int]:
    if hasattr(audio_field, "get_all_samples"):
        samples = audio_field.get_all_samples()
        data = samples.data
        if data.ndim > 1:
            data = data.mean(dim=0)
        return data.detach().cpu().numpy().astype(np.float32), int(samples.sample_rate)
    if isinstance(audio_field, dict):
        sr = int(audio_field.get("sampling_rate", 0))
        if "array" in audio_field:
            arr = np.asarray(audio_field["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr, sr
        if "bytes" in audio_field and audio_field["bytes"]:
            import soundfile as sf
            arr, sr = sf.read(io.BytesIO(audio_field["bytes"]))
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.astype(np.float32), int(sr)
    raise TypeError(f"unrecognized audio field shape: {type(audio_field).__name__}")
