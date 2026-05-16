from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from .._pa import pydantic_to_pa_schema
from .._writer import MANIFEST_NAME, ShardedParquetWriter
from ..audio.writer import _scenario_features
from .schema import RadioRow, SAMPLE_RATE


def _hf_features_metadata() -> dict[bytes, bytes]:
    features = {
        "scenario_id": {"dtype": "string", "_type": "Value"},
        "scenario": _scenario_features(),
        "turn_idx": {"dtype": "int64", "_type": "Value"},
        "speaker": {"dtype": "string", "_type": "Value"},
        "text": {"dtype": "string", "_type": "Value"},
        "text_normalized": {"dtype": "string", "_type": "Value"},
        "voice_id": {"dtype": "string", "_type": "Value"},
        "audio": {"sampling_rate": SAMPLE_RATE, "_type": "Audio"},
        "tokens": {
            "feature": {
                "text": {"dtype": "string", "_type": "Value"},
                "start_s": {"dtype": "float64", "_type": "Value"},
                "end_s": {"dtype": "float64", "_type": "Value"},
            },
            "_type": "List",
        },
        "model": {"dtype": "string", "_type": "Value"},
        "prompt_version": {"dtype": "string", "_type": "Value"},
        "taxonomy_version": {"dtype": "string", "_type": "Value"},
        "tts_model": {"dtype": "string", "_type": "Value"},
        "clean_row_id": {"dtype": "string", "_type": "Value"},
        "variant_idx": {"dtype": "int64", "_type": "Value"},
        "seed": {"dtype": "int64", "_type": "Value"},
        "profile": {"dtype": "string", "_type": "Value"},
        "applied_effects": {
            "feature": {"dtype": "string", "_type": "Value"},
            "_type": "List",
        },
        "effective_snr_db": {"dtype": "float64", "_type": "Value"},
        "pipeline_version": {"dtype": "string", "_type": "Value"},
        "pipeline_fingerprint": {"dtype": "string", "_type": "Value"},
    }
    payload = {"info": {"features": features}}
    return {b"huggingface": json.dumps(payload).encode("utf-8")}


def radio_schema() -> pa.Schema:
    return pydantic_to_pa_schema(RadioRow).with_metadata(_hf_features_metadata())


class ConfigFingerprintMismatch(RuntimeError):
    """Resume target was produced with an incompatible pipeline config."""


class RadioShardWriter:
    def __init__(self, writer: ShardedParquetWriter) -> None:
        self._writer = writer

    @classmethod
    def open(
        cls,
        out_dir: Path,
        shard_size: int,
        pipeline_fingerprint: str,
        *,
        resume: bool,
        extra_meta: dict[str, Any] | None = None,
    ) -> "RadioShardWriter":
        meta = {"pipeline_fingerprint": pipeline_fingerprint, **(extra_meta or {})}
        manifest_path = out_dir / MANIFEST_NAME
        if resume and manifest_path.exists():
            with manifest_path.open() as f:
                existing = json.load(f)
            existing_fp = existing.get("pipeline_fingerprint")
            if existing_fp and existing_fp != pipeline_fingerprint:
                raise ConfigFingerprintMismatch(
                    f"existing run fingerprint {existing_fp!r} does not match "
                    f"current {pipeline_fingerprint!r}; use a fresh --out or "
                    f"delete the existing manifest to start over"
                )
        writer = ShardedParquetWriter.open(
            out_dir, radio_schema(), shard_size, resume=resume, meta=meta,
        )
        return cls(writer)

    @property
    def total_rows(self) -> int:
        return self._writer.total_rows

    @property
    def last_shard_index(self) -> int:
        return self._writer.last_shard_index

    def add(self, row: RadioRow) -> None:
        self._writer.add_row(row.model_dump(mode="python"))

    def close(self) -> None:
        self._writer.close()
