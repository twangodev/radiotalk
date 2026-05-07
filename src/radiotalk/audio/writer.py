from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from .._pa import pydantic_to_pa_schema
from .._writer import MANIFEST_NAME, ShardedParquetWriter
from .schema import AudioTurnRow, SAMPLE_RATE


def _scenario_features() -> dict:
    return {
        "icao": {"dtype": "string", "_type": "Value"},
        "region": {"dtype": "string", "_type": "Value"},
        "phase": {"dtype": "string", "_type": "Value"},
        "aircraft": {
            "feature": {
                "callsign": {"dtype": "string", "_type": "Value"},
                "aircraft_type": {"dtype": "string", "_type": "Value"},
                "wake": {"dtype": "string", "_type": "Value"},
                "operator_class": {"dtype": "string", "_type": "Value"},
            },
            "_type": "List",
        },
        "runway": {"dtype": "string", "_type": "Value"},
        "sid_star": {"dtype": "string", "_type": "Value"},
        "squawk": {"dtype": "string", "_type": "Value"},
        "frequency_mhz": {"dtype": "float64", "_type": "Value"},
        "weather": {
            "wind_dir_deg": {"dtype": "int64", "_type": "Value"},
            "wind_kt": {"dtype": "int64", "_type": "Value"},
            "vis_sm": {"dtype": "float64", "_type": "Value"},
            "ceiling_ft": {"dtype": "int64", "_type": "Value"},
            "altimeter_inhg": {"dtype": "float64", "_type": "Value"},
            "vmc_imc": {"dtype": "string", "_type": "Value"},
        },
        "time_of_day": {"dtype": "string", "_type": "Value"},
        "traffic_density": {"dtype": "string", "_type": "Value"},
        "event": {"dtype": "string", "_type": "Value"},
        "callsign": {"dtype": "string", "_type": "Value"},
        "aircraft_type": {"dtype": "string", "_type": "Value"},
        "wake": {"dtype": "string", "_type": "Value"},
        "operator_class": {"dtype": "string", "_type": "Value"},
        "n_aircraft": {"dtype": "int64", "_type": "Value"},
        "is_emergency": {"dtype": "bool", "_type": "Value"},
        "is_towered": {"dtype": "bool", "_type": "Value"},
    }


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
    }
    payload = {"info": {"features": features}}
    return {b"huggingface": json.dumps(payload).encode("utf-8")}


def audio_schema() -> pa.Schema:
    return pydantic_to_pa_schema(AudioTurnRow).with_metadata(_hf_features_metadata())


class ConfigFingerprintMismatch(RuntimeError):
    """Resume target was produced with an incompatible synth config."""


class AudioShardWriter:
    def __init__(self, writer: ShardedParquetWriter) -> None:
        self._writer = writer

    @classmethod
    def open(
        cls,
        out_dir: Path,
        shard_size: int,
        config_fingerprint: str,
        *,
        resume: bool,
        extra_meta: dict[str, Any] | None = None,
    ) -> "AudioShardWriter":
        meta = {"config_fingerprint": config_fingerprint, **(extra_meta or {})}
        manifest_path = out_dir / MANIFEST_NAME
        if resume and manifest_path.exists():
            with manifest_path.open() as f:
                existing = json.load(f)
            existing_fp = existing.get("config_fingerprint")
            if existing_fp and existing_fp != config_fingerprint:
                raise ConfigFingerprintMismatch(
                    f"existing run fingerprint {existing_fp!r} does not match "
                    f"current {config_fingerprint!r}; use a fresh --out or "
                    f"delete the existing manifest to start over"
                )
        writer = ShardedParquetWriter.open(
            out_dir, audio_schema(), shard_size, resume=resume, meta=meta,
        )
        return cls(writer)

    @property
    def total_rows(self) -> int:
        return self._writer.total_rows

    @property
    def last_shard_index(self) -> int:
        return self._writer.last_shard_index

    def add(self, row: AudioTurnRow) -> None:
        self._writer.add_row(row.model_dump(mode="python"))

    def close(self) -> None:
        self._writer.close()
