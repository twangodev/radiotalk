from __future__ import annotations

import numpy as np

from radiotalk.voices.source import SourceCandidate


def test_source_candidate_holds_audio_and_metadata() -> None:
    audio = np.zeros(24000, dtype=np.float32)
    cand = SourceCandidate(
        source="libritts-r",
        source_speaker_id="1034",
        source_clip_id="1034_x_001",
        audio=audio,
        sample_rate=24000,
        license="CC-BY-4.0",
        attribution="LibriTTS-R, CC BY 4.0",
        gender_hint="F",
        language="en",
    )
    assert cand.source == "libritts-r"
    assert cand.audio.shape == (24000,)
    assert cand.gender_hint == "F"