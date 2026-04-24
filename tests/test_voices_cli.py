from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from typer.testing import CliRunner

from radiotalk.voices.cli import voices_app
from radiotalk.voices.libritts_r import LibriTTSR, _BestClip
from radiotalk.voices.source import SourceInfo, _REGISTRY


def _clip(speaker: str, clip_id: str, seconds: float = 22.0, sr: int = 24000) -> _BestClip:
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    arr = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return _BestClip(
        speaker_id=speaker,
        clip_id=clip_id,
        duration_s=seconds,
        sample_rate=sr,
        audio_bytes=buf.getvalue(),
    )


def test_build_writes_pool_with_unique_speakers(tmp_path: Path, monkeypatch) -> None:
    clips = [
        _clip("1034", "a"),
        _clip("1035", "c"),
        _clip("1036", "d", seconds=0.5),  # too short — filtered out
        _clip("1037", "e"),
    ]

    monkeypatch.setitem(
        _REGISTRY,
        "libritts-r",
        SourceInfo(
            name="libritts-r",
            sub_pool="controller",
            factory=lambda: LibriTTSR(scan_factory=lambda: iter(clips)),
        ),
    )

    out = tmp_path / "voices"
    runner = CliRunner()
    result = runner.invoke(
        voices_app,
        ["build", "--out", str(out), "--target", "3", "--sources", "libritts-r"],
    )
    assert result.exit_code == 0, result.output

    shards = sorted(out.glob("shard-*.parquet"))
    assert len(shards) >= 1
    table = pq.read_table(shards[0])
    assert table.num_rows == 3
    assert set(table.column_names) == {"voice_id", "audio", "text", "source_clip_id"}
    clip_ids = sorted(set(table.column("source_clip_id").to_pylist()))
    assert clip_ids == ["a", "c", "e"]

    for row in table.to_pylist():
        assert isinstance(row["audio"], dict)
        assert isinstance(row["audio"]["bytes"], bytes)
        assert len(row["audio"]["bytes"]) > 0

    notice = (out / "NOTICE.md").read_text()
    assert "libritts-r" in notice


def test_build_unknown_source_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        voices_app,
        ["build", "--out", str(tmp_path / "v"), "--target", "1", "--sources", "nope"],
    )
    assert result.exit_code != 0
    assert "unknown source" in result.output.lower()
