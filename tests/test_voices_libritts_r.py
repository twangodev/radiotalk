from __future__ import annotations

import numpy as np

from radiotalk.voices.libritts_r import LibriTTSR, _to_candidate


def _row(speaker: str, clip_id: str, sr: int = 24000) -> dict:
    return {
        "id": clip_id,
        "speaker_id": speaker,
        "audio": {
            "array": np.zeros(sr * 4, dtype=np.float32),
            "sampling_rate": sr,
            "path": f"{clip_id}.wav",
        },
    }


def test_to_candidate_maps_fields() -> None:
    cand = _to_candidate(_row("1034", "1034_x_001"))
    assert cand.source == "libritts-r"
    assert cand.source_speaker_id == "1034"
    assert cand.source_clip_id == "1034_x_001"
    assert cand.sample_rate == 24000
    assert cand.audio.shape == (24000 * 4,)
    assert cand.license == "CC-BY-4.0"
    assert "LibriTTS-R" in cand.attribution
    assert cand.gender_hint is None


def test_libritts_r_candidates_uses_injected_stream() -> None:
    rows = [_row("1034", "a"), _row("1035", "b"), _row("1034", "c")]
    src = LibriTTSR(stream_factory=lambda: iter(rows))
    out = list(src.candidates())
    assert [c.source_speaker_id for c in out] == ["1034", "1035", "1034"]
    assert all(c.source == "libritts-r" for c in out)