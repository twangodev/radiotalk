from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from radiotalk.voices.builder import BuildStats, build
from radiotalk.voices.filter import FilterParams
from radiotalk.voices.libritts_r import LibriTTSR, _BestClip
from radiotalk.voices.pool import VoicePoolWriter


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


def test_build_collects_unique_speakers(tmp_path: Path) -> None:
    clips = [
        _clip("1034", "a"),
        _clip("1035", "c"),
        _clip("1037", "d"),
    ]
    src = LibriTTSR(scan_factory=lambda: iter(clips))
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=3,
        sources=("libritts-r",), shard_size=100, resume=False,
    )
    stats = build(
        sources=[src], writer=writer, params=FilterParams(), target=3,
    )
    assert stats.kept == 3


def test_build_filters_short_clips(tmp_path: Path) -> None:
    clips = [
        _clip("1034", "a", seconds=0.5),
        _clip("1035", "b", seconds=22.0),
    ]
    src = LibriTTSR(scan_factory=lambda: iter(clips))
    writer = VoicePoolWriter.open(
        out_dir=tmp_path, seed=42, target=10,
        sources=("libritts-r",), shard_size=100, resume=False,
    )
    stats = build(
        sources=[src], writer=writer, params=FilterParams(), target=10,
    )
    assert stats.kept == 1


def test_build_calls_on_progress(tmp_path: Path) -> None:
    clips = [_clip("1034", "a"), _clip("1035", "b")]
    src = LibriTTSR(scan_factory=lambda: iter(clips))
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
