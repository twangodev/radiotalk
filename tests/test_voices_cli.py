from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from typer.testing import CliRunner

from radiotalk.voices.cli import voices_app
from radiotalk.voices.libritts_r import LibriTTSR
from radiotalk.voices.source import SourceInfo, _REGISTRY


def _row(speaker: str, clip_id: str, seconds: float = 12.0, sr: int = 24000) -> dict:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    arr = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return {
        "id": clip_id,
        "speaker_id": speaker,
        "audio": {"array": arr, "sampling_rate": sr, "path": f"{clip_id}.wav"},
    }


def test_build_writes_pool_with_unique_speakers(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _row("1034", "a"),
        _row("1034", "b"),  # same speaker — should be skipped after first
        _row("1035", "c"),
        _row("1036", "d", seconds=0.5),  # too short — filtered out
        _row("1037", "e"),
    ]

    monkeypatch.setitem(
        _REGISTRY,
        "libritts-r",
        SourceInfo(
            name="libritts-r",
            sub_pool="controller",
            factory=lambda: LibriTTSR(stream_factory=lambda: iter(rows)),
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
    speakers = sorted(set(table.column("source_speaker_id").to_pylist()))
    assert speakers == ["1034", "1035", "1037"]

    for row in table.to_pylist():
        assert isinstance(row["audio"], bytes)
        assert len(row["audio"]) > 0

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
