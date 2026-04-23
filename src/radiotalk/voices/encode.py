from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def encode_flac(audio: np.ndarray, sr: int) -> bytes:
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()
