from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from radiotalk.voices.encode import encode_flac
from radiotalk.voices.manifest import VoiceRecord, voice_id_for
from radiotalk.voices.pool import VoicePoolWriter


def _tone(seconds: float, sr: int = 24000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _candidate_record(speaker: str) -> tuple[VoiceRecord, bytes]:
    vid = voice_id_for(source="libritts-r", source_speaker_id=speaker)
    audio = _tone(12.0)
    blob = encode_flac(audio, sr=24000)
    rec = VoiceRecord(
        voice_id=vid,
        source="libritts-r",
        source_speaker_id=speaker,
        source_clip_id=f"{speaker}_x_001",
        duration_s=12.0,
        sample_rate=24000,
        license="CC-BY-4.0",
        attribution="LibriTTS-R, CC BY 4.0",
        selected_at="2026-04-22T15:00:00+00:00",
    )
    return rec, blob


def test_writer_writes_sharded_parquet_with_audio(tmp_path: Path) -> None:
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=2,
        sources=("libritts-r",), shard_size=10, resume=False,
    )
    for spk in ("1034", "1035"):
        rec, blob = _candidate_record(spk)
        writer.add(rec, blob)
    writer.close()

    shard = tmp_path / "shard-000000.parquet"
    assert shard.exists()
    table = pq.read_table(shard)
    assert table.num_rows == 2
    speakers = set(table.column("source_speaker_id").to_pylist())
    assert speakers == {"1034", "1035"}

    for row in table.to_pylist():
        assert isinstance(row["audio"], bytes)
        assert len(row["audio"]) > 0

    manifest = tmp_path / "manifest.json"
    assert manifest.exists()


def test_resume_skips_already_written_voice_ids(tmp_path: Path) -> None:
    w1 = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=2,
        sources=("libritts-r",), shard_size=10, resume=False,
    )
    rec, blob = _candidate_record("1034")
    w1.add(rec, blob)
    w1.close()

    w2 = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=2,
        sources=("libritts-r",), shard_size=10, resume=True,
    )
    assert w2.contains(rec.voice_id) is True
    assert w2.total_voices == 1
    rec2, blob2 = _candidate_record("1035")
    w2.add(rec2, blob2)
    w2.close()

    shards = sorted(tmp_path.glob("shard-*.parquet"))
    total = sum(pq.read_table(s).num_rows for s in shards)
    assert total == 2


def test_records_property_returns_typed_records(tmp_path: Path) -> None:
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=2,
        sources=("libritts-r",), shard_size=10, resume=False,
    )
    rec, blob = _candidate_record("1034")
    writer.add(rec, blob)
    writer.close()
    records = writer.records
    assert len(records) == 1
    assert isinstance(records[0], VoiceRecord)
    assert records[0].voice_id == rec.voice_id
