from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .._writer import SHARD_TEMPLATE, ShardedParquetWriter
from .manifest import VOICE_SCHEMA, VoiceRecord


class VoicePoolWriter:
    """Sharded parquet writer for voice pools with embedded audio."""

    def __init__(
        self,
        writer: ShardedParquetWriter,
        *,
        existing_voice_ids: set[str] | None = None,
    ) -> None:
        self._writer = writer
        self._records: list[VoiceRecord] = []
        self._seen_voice_ids: set[str] = set(existing_voice_ids or set())

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
        existing: set[str] = set()
        if resume:
            existing = _read_existing_voice_ids(out_dir)
        return cls(writer, existing_voice_ids=existing)

    def contains(self, voice_id: str) -> bool:
        return voice_id in self._seen_voice_ids

    @property
    def total_voices(self) -> int:
        return len(self._seen_voice_ids)

    @property
    def records(self) -> list[VoiceRecord]:
        return list(self._records)

    def add(self, record: VoiceRecord, audio: bytes) -> None:
        if record.voice_id in self._seen_voice_ids:
            return
        row = {
            "voice_id": record.voice_id,
            "audio": {"bytes": audio, "path": None},
            "text": record.text,
            "source_clip_id": record.source_clip_id,
        }
        self._writer.add_row(row)
        self._records.append(record)
        self._seen_voice_ids.add(record.voice_id)

    def close(self) -> None:
        self._writer.close()


def _read_existing_voice_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    idx = 0
    while True:
        shard = out_dir / SHARD_TEMPLATE.format(idx=idx)
        if not shard.exists():
            break
        table = pq.read_table(shard, columns=["voice_id"])
        ids.update(table.column("voice_id").to_pylist())
        idx += 1
    return ids
