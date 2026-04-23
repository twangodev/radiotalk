from __future__ import annotations

from pathlib import Path

import numpy as np

from radiotalk.voices.builder import BuildStats, build
from radiotalk.voices.filter import FilterParams
from radiotalk.voices.libritts_r import LibriTTSR
from radiotalk.voices.pool import VoicePoolWriter


def _row(speaker: str, clip_id: str, seconds: float = 12.0, sr: int = 24000) -> dict:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    arr = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return {
        "id": clip_id,
        "speaker_id": speaker,
        "audio": {"array": arr, "sampling_rate": sr, "path": f"{clip_id}.wav"},
    }


def test_build_collects_unique_speakers(tmp_path: Path) -> None:
    rows = [
        _row("1034", "a"),
        _row("1034", "b"),
        _row("1035", "c"),
        _row("1037", "d"),
    ]
    src = LibriTTSR(stream_factory=lambda: iter(rows))
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=3,
        sources=("libritts-r",), shard_size=100, resume=False,
    )
    stats = build(
        sources=[src], writer=writer, params=FilterParams(), target=3,
    )
    assert stats.kept == 3
    assert stats.skipped >= 1


def test_build_filters_short_clips(tmp_path: Path) -> None:
    rows = [
        _row("1034", "a", seconds=0.5),
        _row("1035", "b", seconds=12.0),
    ]
    src = LibriTTSR(stream_factory=lambda: iter(rows))
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=10,
        sources=("libritts-r",), shard_size=100, resume=False,
    )
    stats = build(
        sources=[src], writer=writer, params=FilterParams(), target=10,
    )
    assert stats.kept == 1
    assert stats.skipped == 1


def test_build_calls_on_progress(tmp_path: Path) -> None:
    rows = [_row("1034", "a"), _row("1035", "b")]
    src = LibriTTSR(stream_factory=lambda: iter(rows))
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=2,
        sources=("libritts-r",), shard_size=100, resume=False,
    )
    calls: list[BuildStats] = []
    build(
        sources=[src], writer=writer, params=FilterParams(), target=2,
        on_progress=lambda s: calls.append(BuildStats(kept=s.kept, skipped=s.skipped)),
    )
    assert len(calls) >= 2
    assert calls[-1].kept == 2
