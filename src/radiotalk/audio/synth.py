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
from .model import HIGGS_MODEL_ID, HiggsInferenceOptions, LoadedHiggs
from .schema import AudioBytes, AudioTurnRow, SAMPLE_RATE
from .staging import Staging
from .writer import AudioShardWriter


FAILURES_NAME = "failures.jsonl"
MAX_SYNTH_ATTEMPTS = 3
MAX_WPM = 300.0
MIN_DUR_S = 0.30
MIN_OOM_BATCH = 4


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


@dataclass
class _VoiceRef:
    ref_path: Path
    ref_text: str


def synth_config_fingerprint(model_id: str, opts: HiggsInferenceOptions) -> str:
    payload = {
        "tts_model": model_id,
        "max_new_tokens": opts.max_new_tokens,
        "do_sample": opts.do_sample,
        "temperature": opts.temperature,
        "top_p": opts.top_p,
        "sample_rate": SAMPLE_RATE,
        "max_wpm": MAX_WPM,
        "min_dur_s": MIN_DUR_S,
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


def encode_voice_pool(
    loaded: LoadedHiggs,
    voices_ds,
    voice_ids_needed: Iterable[str],
    ref_dir: Path,
) -> dict[str, _VoiceRef]:
    """Materialize the reference WAV for each voice on disk and capture its
    transcript text. Higgs's AutoProcessor consumes the audio reference as a
    file URL, so we stage WAVs once per voice instead of carrying tensors."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    voice_index = {vid: i for i, vid in enumerate(voices_ds["voice_id"])}
    cache: dict[str, _VoiceRef] = {}
    for vid in dict.fromkeys(voice_ids_needed):
        row = voices_ds[voice_index[vid]]
        path = ref_dir / f"{vid}.wav"
        if not path.exists():
            arr, sr = _decode_audio_field(row["audio"])
            import soundfile as sf
            sf.write(path, arr, sr)
        cache[vid] = _VoiceRef(ref_path=path, ref_text=str(row["text"]))
    return cache


def _generate_oom_safe(
    loaded: LoadedHiggs,
    voice_ref: _VoiceRef,
    texts: list[str],
    opts: HiggsInferenceOptions,
) -> list[np.ndarray | None]:
    """Run model.generate over `texts` (all sharing voice_ref) with auto-halving
    batch size on CUDA OOM. Returns one float32 array (or None on failure)
    per input text, in order."""
    out: list[np.ndarray | None] = [None] * len(texts)
    pos = 0
    sub = len(texts)
    while pos < len(texts):
        end = min(pos + sub, len(texts))
        chunk = texts[pos:end]
        convs = [
            loaded.build_conversation(voice_ref.ref_text, voice_ref.ref_path, t)
            for t in chunk
        ]
        try:
            inputs = loaded.processor.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=True,
                return_dict=True, sampling_rate=loaded.sampling_rate,
                return_tensors="pt", padding=True,
            ).to(loaded.model.device)
            with torch.inference_mode():
                gen = loaded.model.generate(
                    **inputs,
                    max_new_tokens=opts.max_new_tokens,
                    do_sample=opts.do_sample,
                    temperature=opts.temperature if opts.do_sample else 1.0,
                    top_p=opts.top_p if opts.do_sample else 1.0,
                )
            decoded = loaded.processor.batch_decode(gen)
            for k, audio_obj in enumerate(decoded):
                arr = audio_obj.get("audio") if isinstance(audio_obj, dict) else audio_obj
                if arr is None:
                    continue
                out[pos + k] = np.asarray(arr, dtype=np.float32).reshape(-1)
            pos = end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if sub <= MIN_OOM_BATCH:
                # Give up on this chunk; entries stay None and the caller retries.
                pos = end
            else:
                sub = max(sub // 2, MIN_OOM_BATCH)
    return out


def synthesize_voice_queue(
    loaded: LoadedHiggs,
    voice_cache: dict[str, _VoiceRef],
    voice_queue: dict[str, list[_QueueItem]],
    staging: Staging,
    *,
    opts: HiggsInferenceOptions,
    max_batch: int = 32,
    on_voice_done: Callable[[str, int], None] | None = None,
    on_turn_ok: Callable[[int], None] | None = None,
    log_failure: Callable[[str, str], None] | None = None,
) -> SynthStats:
    stats = SynthStats()
    already_done = staging.voices_done()

    for voice_id, items in voice_queue.items():
        if voice_id in already_done:
            stats.voices_processed += 1
            continue
        if voice_id not in voice_cache:
            continue

        voice_ref = voice_cache[voice_id]
        rows_buffer: list[tuple] = []
        # Length-bucket within this voice's queue: similar-length batches mean
        # less padding waste in model.generate().
        pending: list[_QueueItem] = sorted(items, key=lambda it: len(it.text_normalized))

        for attempt in range(MAX_SYNTH_ATTEMPTS):
            if not pending:
                break
            next_pending: list[_QueueItem] = []
            for chunk_start in range(0, len(pending), max_batch):
                chunk = pending[chunk_start : chunk_start + max_batch]
                texts = [it.text_normalized for it in chunk]
                audios = _generate_oom_safe(loaded, voice_ref, texts, opts)
                for it, audio in zip(chunk, audios):
                    if audio is None or audio.size == 0:
                        it.last_reason = "no audio returned"
                        next_pending.append(it)
                        continue
                    duration_s = audio.shape[-1] / loaded.sampling_rate
                    if duration_s < MIN_DUR_S:
                        it.last_reason = f"duration {duration_s:.2f}s < {MIN_DUR_S}s"
                        next_pending.append(it)
                        continue
                    n_words = len(it.text_normalized.split())
                    wpm = n_words * 60.0 / duration_s if duration_s > 0 else float("inf")
                    if wpm > MAX_WPM:
                        it.last_reason = f"wpm {wpm:.0f} > {MAX_WPM:.0f}"
                        next_pending.append(it)
                        continue
                    rows_buffer.append((
                        it.scenario_id, it.turn_idx, voice_id,
                        it.text_normalized,
                        encode_audio(audio),
                        None,  # tokens_json — Higgs does not expose per-token timing
                    ))
                    stats.turns_ok += 1
                    stats.record_attempt(attempt)
                    if on_turn_ok is not None:
                        on_turn_ok(stats.turns_ok)
                if len(rows_buffer) >= 1024:
                    staging.insert_turns(rows_buffer)
                    rows_buffer.clear()
            pending = next_pending
        if loaded.device.startswith("cuda"):
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
                tokens=None,
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
