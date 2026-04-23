from __future__ import annotations

import json
import shutil
from pathlib import Path

from .._pa import now_iso, pydantic_to_pa_schema
from .._writer import MANIFEST_NAME, ShardedParquetWriter
from .transcript import Transcript


FAILURES_NAME = "failures.jsonl"

TRANSCRIPT_SCHEMA = pydantic_to_pa_schema(Transcript)


class ConfigFingerprintMismatch(RuntimeError):
    """Raised when resuming into an out_dir produced by an incompatible run."""


class Manifest:
    """View of a data-pipeline manifest.json (compatible with shared writer)."""

    def __init__(
        self,
        *,
        seed: int,
        config_fingerprint: str,
        total_rows: int,
        last_shard_index: int,
        started_at: str,
        updated_at: str,
    ) -> None:
        self.seed = seed
        self.config_fingerprint = config_fingerprint
        self.total_rows = total_rows
        self.last_shard_index = last_shard_index
        self.started_at = started_at
        self.updated_at = updated_at

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Manifest):
            return NotImplemented
        return vars(self) == vars(other)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        with path.open() as f:
            data = json.load(f)
        return cls(
            seed=data["seed"],
            config_fingerprint=data["config_fingerprint"],
            total_rows=data["total_rows"],
            last_shard_index=data["last_shard_index"],
            started_at=data["started_at"],
            updated_at=data["updated_at"],
        )

    def dump(self, path: Path) -> None:
        import os
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(vars(self), f, indent=2)
        os.replace(tmp, path)


class ParquetShardWriter:
    """Buffers Transcript rows, flushes to numbered Parquet shards.

    Wraps :class:`ShardedParquetWriter` and adds fingerprint validation
    and failure tracking.
    """

    def __init__(self, writer: ShardedParquetWriter) -> None:
        self._writer = writer

    @classmethod
    def open(
        cls,
        out_dir: Path,
        shard_size: int,
        config_fingerprint: str,
        seed: int,
        *,
        resume: bool,
        overwrite: bool,
    ) -> ParquetShardWriter:
        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir)
        meta = {"seed": seed, "config_fingerprint": config_fingerprint}
        manifest_path = out_dir / MANIFEST_NAME
        if resume and manifest_path.exists():
            with manifest_path.open() as f:
                data = json.load(f)
            existing_fp = data.get("config_fingerprint")
            if existing_fp != config_fingerprint:
                raise ConfigFingerprintMismatch(
                    f"Existing run fingerprint {existing_fp!r} "
                    f"does not match current {config_fingerprint!r}. "
                    "Use --overwrite to start fresh, or change --out to a new path."
                )
            existing_seed = data.get("seed")
            if existing_seed != seed:
                raise ConfigFingerprintMismatch(
                    f"Existing run seed {existing_seed} does not match current {seed}."
                )
        writer = ShardedParquetWriter.open(
            out_dir, TRANSCRIPT_SCHEMA, shard_size, resume=resume, meta=meta,
        )
        return cls(writer)

    @property
    def out_dir(self) -> Path:
        return self._writer.out_dir

    @property
    def total_rows(self) -> int:
        return self._writer.total_rows

    @property
    def last_shard_index(self) -> int:
        return self._writer.last_shard_index

    def add(self, transcript: Transcript) -> None:
        self._writer.add_row(transcript.model_dump(mode="json"))

    def add_failure(self, *, scenario_id: str, scenario: dict, error: str) -> None:
        path = self._writer.out_dir / FAILURES_NAME
        with path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "scenario_id": scenario_id,
                        "scenario": scenario,
                        "error": error,
                        "at": now_iso(),
                    }
                )
                + "\n"
            )

    def close(self) -> None:
        self._writer.close()
