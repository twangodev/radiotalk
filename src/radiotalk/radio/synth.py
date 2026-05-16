from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..audio.schema import AudioBytes
from .pipeline import pipeline_fingerprint
from .presets import PIPELINE_VERSION, PROFILES, Profile
from .schema import RadioRow
from .source import iter_clean_rows
from .worker import seed_for_row, synthesize_variant
from .writer import RadioShardWriter


FAILURES_NAME = "failures.jsonl"


@dataclass
class SynthStats:
    variants_ok: int = 0
    variants_failed: int = 0


def synth_config_fingerprint(profiles: dict[str, Profile] | None = None) -> str:
    """Aggregate fingerprint across all profile configs. Detects any change
    to the registered effects/probs/SNR samplers on resume.
    """
    profiles = profiles or PROFILES
    fp_blob = "|".join(sorted(pipeline_fingerprint(p) for p in profiles.values()))
    return hashlib.sha256(fp_blob.encode("utf-8")).hexdigest()[:16]


def iter_jobs(
    shards: list[Path],
    variants: int,
    *,
    skip_clean_rows: int = 0,
    limit_clean_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Fan one clean row out into ``variants`` jobs. Yields plain dicts safe
    to ship across a process pool boundary.
    """
    for row in iter_clean_rows(
        shards, skip=skip_clean_rows, limit=limit_clean_rows,
    ):
        clean_row_id = f"{row['scenario_id']}#{row['turn_idx']}"
        raw_audio = row["audio"]["bytes"]
        passthrough = {
            "scenario_id": row["scenario_id"],
            "scenario": row["scenario"],
            "turn_idx": int(row["turn_idx"]),
            "speaker": row["speaker"],
            "text": row["text"],
            "text_normalized": row["text_normalized"],
            "voice_id": row["voice_id"],
            "tokens": row.get("tokens"),
            "model": row["model"],
            "prompt_version": row["prompt_version"],
            "taxonomy_version": row["taxonomy_version"],
            "tts_model": row["tts_model"],
        }
        for variant_idx in range(variants):
            yield {
                "raw_audio": raw_audio,
                "clean_row_id": clean_row_id,
                "variant_idx": variant_idx,
                "seed": seed_for_row(clean_row_id, variant_idx),
                **passthrough,
            }


def result_to_row(result: dict[str, Any]) -> RadioRow:
    return RadioRow(
        scenario_id=result["scenario_id"],
        scenario=result["scenario"],
        turn_idx=result["turn_idx"],
        speaker=result["speaker"],
        text=result["text"],
        text_normalized=result["text_normalized"],
        voice_id=result["voice_id"],
        audio=AudioBytes(bytes=result["audio_bytes"]),
        tokens=result.get("tokens"),
        model=result["model"],
        prompt_version=result["prompt_version"],
        taxonomy_version=result["taxonomy_version"],
        tts_model=result["tts_model"],
        clean_row_id=result["clean_row_id"],
        variant_idx=result["variant_idx"],
        seed=result["seed"],
        profile=result["profile"],
        applied_effects=result["applied_effects"],
        effective_snr_db=result["effective_snr_db"],
        pipeline_version=PIPELINE_VERSION,
        pipeline_fingerprint=result["pipeline_fingerprint"],
    )


def append_failure(out_dir: Path, clean_row_id: str, reason: str) -> None:
    path = out_dir / FAILURES_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a") as f:
        f.write(json.dumps({
            "clean_row_id": clean_row_id,
            "reason": reason,
            "at": ts,
        }) + "\n")


def run_synthesis(
    jobs: Iterator[dict[str, Any]],
    writer: RadioShardWriter,
    *,
    workers: int,
    chunksize: int,
    on_variant_ok: Callable[[int], None] | None = None,
    log_failure: Callable[[str, str], None] | None = None,
) -> SynthStats:
    """Drive the worker pool with bounded in-flight queue + background writer
    thread. Returns when ``jobs`` is exhausted and all results are flushed.
    """
    stats = SynthStats()
    max_in_flight = workers * chunksize * 2
    result_q: queue.Queue = queue.Queue(maxsize=workers * 16)
    writer_error: list[BaseException] = []

    def _writer_loop() -> None:
        try:
            while True:
                item = result_q.get()
                if item is None:
                    return
                writer.add(result_to_row(item))
        except BaseException as exc:
            writer_error.append(exc)

    writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    writer_thread.start()

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            in_flight: deque = deque()

            def _drain_one() -> None:
                if writer_error:
                    raise writer_error[0]
                fut = in_flight.popleft()
                try:
                    result = fut.result()
                except Exception as exc:
                    stats.variants_failed += 1
                    if log_failure is not None:
                        log_failure("?", f"worker exception: {exc!r}")
                    return
                result_q.put(result)
                stats.variants_ok += 1
                if on_variant_ok is not None:
                    on_variant_ok(stats.variants_ok)

            for job in jobs:
                while len(in_flight) >= max_in_flight:
                    _drain_one()
                in_flight.append(pool.submit(synthesize_variant, job))
            while in_flight:
                _drain_one()

        result_q.put(None)
        writer_thread.join()
        if writer_error:
            raise writer_error[0]
    finally:
        pass
    return stats
