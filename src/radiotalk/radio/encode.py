from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from .schema import SAMPLE_RATE


def encode_wav_pcm16(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32, copy=False), sample_rate,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()
