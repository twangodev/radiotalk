from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from .schema import SAMPLE_RATE


def encode_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32, copy=False), sample_rate, format="FLAC")
    return buf.getvalue()
