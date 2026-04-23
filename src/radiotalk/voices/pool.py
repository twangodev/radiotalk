from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .._writer import SHARD_TEMPLATE, ShardedParquetWriter
from .manifest import VOICE_SCHEMA, VoiceRecord


class VoicePoolWriter:
    """Sharded parquet writer for voice pools with embedded audio."""

    def __init__(
        self,
        writer: ShardedParquetWriter,
        *,
        existing_records: list[VoiceRecord] | None = None,
    ) -> None:
        self._writer = writer
        self._records: list[VoiceRecord] = list(existing_records or [])
        self._seen_voice_ids: set[str] = {r.voice_id for r in self._records}

    @classmethod
    def open(
        cls,
        out_dir: Path,
        seed: int,
        target: int,
        sources: tuple[str, ...],
        *,
        shard_size: int = 500,
        resume: bool,
    ) -> VoicePoolWriter:
        meta = {"seed": seed, "target": target, "sources": list(sources)}
        writer = ShardedParquetWriter.open(
            out_dir, VOICE_SCHEMA, shard_size, resume=resume, meta=meta,
        )
        existing: list[VoiceRecord] = []
        if resume:
            existing = _read_existing_records(out_dir)
        return cls(writer, existing_records=existing)

    def contains(self, voice_id: str) -> bool:
        return voice_id in self._seen_voice_ids

    @property
    def total_voices(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[VoiceRecord]:
        return list(self._records)

    def add(self, record: VoiceRecord, audio: bytes) -> None:
        if record.voice_id in self._seen_voice_ids:
            return
        row: dict[str, Any] = record.model_dump()
        row["audio"] = audio
        self._writer.add_row(row)
        self._records.append(record)
        self._seen_voice_ids.add(record.voice_id)

    def close(self) -> None:
        self._writer.close()


def _read_existing_records(out_dir: Path) -> list[VoiceRecord]:
    records: list[VoiceRecord] = []
    idx = 0
    while True:
        shard = out_dir / SHARD_TEMPLATE.format(idx=idx)
        if not shard.exists():
            break
        table = pq.read_table(shard)
        for row in table.to_pylist():
            row.pop("audio", None)
            records.append(VoiceRecord.model_validate(row))
        idx += 1
    return records
