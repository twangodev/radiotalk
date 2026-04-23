from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ._pa import now_iso

MANIFEST_NAME = "manifest.json"
SHARD_TEMPLATE = "shard-{idx:06d}.parquet"


class ShardedParquetWriter:
    """Generic sharded parquet writer with JSON manifest.

    Buffers rows in memory and flushes to numbered parquet shards when the
    buffer reaches ``shard_size``. Each shard is written atomically via
    temp-file + rename. A JSON manifest tracks progress and caller-provided
    metadata so runs can be resumed.
    """

    def __init__(
        self,
        out_dir: Path,
        schema: pa.Schema,
        shard_size: int,
        *,
        total_rows: int = 0,
        last_shard_index: int = -1,
        started_at: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.schema = schema
        self.shard_size = shard_size
        self._buffer: list[dict[str, Any]] = []
        self._total_rows = total_rows
        self._last_shard_index = last_shard_index
        self._started_at = started_at or now_iso()
        self._meta: dict[str, Any] = dict(meta or {})

    @classmethod
    def open(
        cls,
        out_dir: Path,
        schema: pa.Schema,
        shard_size: int,
        *,
        resume: bool,
        meta: dict[str, Any] | None = None,
    ) -> ShardedParquetWriter:
        manifest_path = out_dir / MANIFEST_NAME
        if resume and manifest_path.exists():
            with manifest_path.open() as f:
                data = json.load(f)
            return cls(
                out_dir,
                schema,
                shard_size,
                total_rows=data.get("total_rows", 0),
                last_shard_index=data.get("last_shard_index", -1),
                started_at=data.get("started_at"),
                meta=meta,
            )
        return cls(out_dir, schema, shard_size, meta=meta)

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def last_shard_index(self) -> int:
        return self._last_shard_index

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def add_row(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self._flush_shard()

    def close(self) -> None:
        if self._buffer:
            self._flush_shard()
        self._write_manifest()

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        next_idx = self._last_shard_index + 1
        shard_path = self.out_dir / SHARD_TEMPLATE.format(idx=next_idx)
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, shard_path)
        self._last_shard_index = next_idx
        self._total_rows += len(self._buffer)
        self._buffer.clear()
        self._write_manifest()

    def _write_manifest(self) -> None:
        data = {
            "total_rows": self._total_rows,
            "last_shard_index": self._last_shard_index,
            "started_at": self._started_at,
            "updated_at": now_iso(),
            **self._meta,
        }
        manifest_path = self.out_dir / MANIFEST_NAME
        tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, manifest_path)
