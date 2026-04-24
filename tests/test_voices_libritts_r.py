from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from radiotalk.voices.libritts_r import (
    LibriTTSR,
    _BestClip,
    _probe_wav,
    _scan_longest,
    _to_candidate,
)


def _wav_bytes(seconds: float, sr: int = 24000) -> bytes:
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    arr = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _clip(speaker: str, clip_id: str, seconds: float = 22.0, sr: int = 24000) -> _BestClip:
    b = _wav_bytes(seconds, sr)
    return _BestClip(
        speaker_id=speaker,
        clip_id=clip_id,
        duration_s=seconds,
        sample_rate=sr,
        audio_bytes=b,
    )


def test_probe_wav_reads_duration_without_decoding() -> None:
    b = _wav_bytes(3.0)
    dur, sr = _probe_wav(b)
    assert sr == 24000
    assert abs(dur - 3.0) < 0.01


def test_to_candidate_maps_fields() -> None:
    cand = _to_candidate(_clip("1034", "1034_x_001", seconds=4.0))
    assert cand.source == "libritts-r"
    assert cand.source_speaker_id == "1034"
    assert cand.source_clip_id == "1034_x_001"
    assert cand.sample_rate == 24000
    assert cand.audio.shape == (24000 * 4,)
    assert cand.license == "CC-BY-4.0"
    assert "LibriTTS-R" in cand.attribution
    assert cand.gender_hint is None


def test_libritts_r_candidates_uses_injected_scan() -> None:
    clips = [
        _clip("1034", "a", seconds=8.0),
        _clip("1035", "b", seconds=12.0),
    ]
    src = LibriTTSR(scan_factory=lambda: iter(clips))
    out = list(src.candidates())
    assert [c.source_speaker_id for c in out] == ["1034", "1035"]
    assert all(c.source == "libritts-r" for c in out)


def test_scan_longest_picks_longest_per_speaker(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    def row(speaker: str, clip_id: str, seconds: float) -> dict:
        return {
            "audio": {"bytes": _wav_bytes(seconds), "path": f"{clip_id}.wav"},
            "speaker_id": speaker,
            "id": clip_id,
            "text_normalized": f"transcript for {clip_id}",
        }

    rows = [
        row("1034", "a", 5.0),
        row("1034", "b", 22.0),  # longest for 1034
        row("1034", "c", 12.0),
        row("1035", "d", 18.0),
        row("1036", "e", 45.0),  # outside SCAN_MAX_S (30)
    ]
    tbl = pa.Table.from_pylist(rows)
    shard = tmp_path / "shard.parquet"
    pq.write_table(tbl, shard)

    best = _scan_longest([shard])
    assert set(best.keys()) == {"1034", "1035"}
    assert best["1034"].clip_id == "b"
    assert best["1035"].clip_id == "d"
    assert best["1034"].duration_s > 21.9
