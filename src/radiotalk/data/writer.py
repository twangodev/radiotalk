from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from .transcript import Transcript


MANIFEST_NAME = "manifest.json"
FAILURES_NAME = "failures.jsonl"
SHARD_TEMPLATE = "shard-{idx:06d}.parquet"


_PRIMITIVE_PA: dict[type, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    datetime: pa.string(),
}


def _annotation_to_pa(annotation: Any) -> pa.DataType:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            raise TypeError(f"unsupported Union in model schema: {annotation!r}")
        return _annotation_to_pa(non_none[0])

    if origin is Literal:
        kinds = {type(a) for a in args}
        if len(kinds) != 1:
            raise TypeError(f"mixed-type Literal not supported: {annotation!r}")
        return _PRIMITIVE_PA[next(iter(kinds))]

    if origin is list:
        (inner,) = args
        return pa.list_(_annotation_to_pa(inner))

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_to_pa_struct(annotation)

    if annotation in _PRIMITIVE_PA:
        return _PRIMITIVE_PA[annotation]

    raise TypeError(f"no pyarrow mapping for annotation: {annotation!r}")


def _model_to_pa_struct(model: type[BaseModel]) -> pa.DataType:
    fields: list[tuple[str, pa.DataType]] = []
    for name, info in model.model_fields.items():
        fields.append((name, _annotation_to_pa(info.annotation)))
    for name, info in model.model_computed_fields.items():
        fields.append((name, _annotation_to_pa(info.return_type)))
    return pa.struct(fields)


def _pydantic_to_pa_schema(model: type[BaseModel]) -> pa.Schema:
    struct = _model_to_pa_struct(model)
    return pa.schema([pa.field(f.name, f.type) for f in struct])


TRANSCRIPT_SCHEMA = _pydantic_to_pa_schema(Transcript)


class ConfigFingerprintMismatch(RuntimeError):
    """Raised when resuming into an out_dir produced by an incompatible run."""


@dataclass
class Manifest:
    seed: int
    config_fingerprint: str
    total_rows: int
    last_shard_index: int
    started_at: str
    updated_at: str

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        with path.open() as f:
            return cls(**json.load(f))

    def dump(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self.__dict__, f, indent=2)
        os.replace(tmp, path)


class ParquetShardWriter:
    """Buffers Transcript rows, flushes to numbered Parquet shards.

    Atomic writes: each shard is written to `<name>.tmp` then renamed. The manifest
    is updated after each successful rename.
    """

    def __init__(
        self,
        out_dir: Path,
        shard_size: int,
        config_fingerprint: str,
        seed: int,
        *,
        manifest: Manifest | None = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._buffer: list[dict[str, Any]] = []
        if manifest is None:
            now = _now_iso()
            manifest = Manifest(
                seed=seed,
                config_fingerprint=config_fingerprint,
                total_rows=0,
                last_shard_index=-1,
                started_at=now,
                updated_at=now,
            )
        self._manifest = manifest

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
    ) -> "ParquetShardWriter":
        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir)
        manifest_path = out_dir / MANIFEST_NAME
        if resume and manifest_path.exists():
            manifest = Manifest.load(manifest_path)
            if manifest.config_fingerprint != config_fingerprint:
                raise ConfigFingerprintMismatch(
                    f"Existing run fingerprint {manifest.config_fingerprint!r} "
                    f"does not match current {config_fingerprint!r}. "
                    "Use --overwrite to start fresh, or change --out to a new path."
                )
            if manifest.seed != seed:
                raise ConfigFingerprintMismatch(
                    f"Existing run seed {manifest.seed} does not match current {seed}."
                )
            return cls(
                out_dir,
                shard_size,
                config_fingerprint,
                seed,
                manifest=manifest,
            )
        return cls(out_dir, shard_size, config_fingerprint, seed)

    @property
    def total_rows(self) -> int:
        return self._manifest.total_rows

    @property
    def last_shard_index(self) -> int:
        return self._manifest.last_shard_index

    def add(self, transcript: Transcript) -> None:
        self._buffer.append(transcript.model_dump(mode="json"))
        if len(self._buffer) >= self.shard_size:
            self._flush_shard()

    def add_failure(self, *, scenario_id: str, scenario: dict, error: str) -> None:
        path = self.out_dir / FAILURES_NAME
        with path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "scenario_id": scenario_id,
                        "scenario": scenario,
                        "error": error,
                        "at": _now_iso(),
                    }
                )
                + "\n"
            )

    def close(self) -> None:
        if self._buffer:
            self._flush_shard()
        self._manifest.updated_at = _now_iso()
        self._manifest.dump(self.out_dir / MANIFEST_NAME)

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        next_idx = self._manifest.last_shard_index + 1
        shard_path = self.out_dir / SHARD_TEMPLATE.format(idx=next_idx)
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        table = pa.Table.from_pylist(self._buffer, schema=TRANSCRIPT_SCHEMA)
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, shard_path)
        self._manifest.last_shard_index = next_idx
        self._manifest.total_rows += len(self._buffer)
        self._manifest.updated_at = _now_iso()
        self._manifest.dump(self.out_dir / MANIFEST_NAME)
        self._buffer.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
