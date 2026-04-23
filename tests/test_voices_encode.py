from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from radiotalk.voices.encode import encode_flac


def _tone(seconds: float, sr: int = 24000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_encode_flac_returns_decodable_bytes() -> None:
    audio = _tone(2.0, sr=24000)
    blob = encode_flac(audio, sr=24000)
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    decoded, sr = sf.read(io.BytesIO(blob), dtype="float32")
    assert sr == 24000
    assert decoded.shape == audio.shape
    np.testing.assert_allclose(decoded, audio, atol=1e-3)


def test_encode_rejects_non_mono() -> None:
    stereo = np.zeros((24000, 2), dtype=np.float32)
    try:
        encode_flac(stereo, sr=24000)
    except ValueError:
        return
    raise AssertionError("expected ValueError for stereo input")