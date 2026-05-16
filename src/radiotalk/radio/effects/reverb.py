from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def apply_room_ir(samples: np.ndarray, ir: np.ndarray, mix: float = 1.0) -> np.ndarray:
    """Convolve with a room impulse response. ``mix`` blends dry/wet."""
    if ir.ndim > 1:
        ir = ir.mean(axis=-1)
    wet = fftconvolve(samples, ir, mode="full")[: samples.shape[0]]
    wet_rms = float(np.sqrt(np.mean(wet.astype(np.float64) ** 2) + 1e-10))
    dry_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2) + 1e-10))
    if wet_rms > 0:
        wet = wet * (dry_rms / wet_rms)
    return ((1.0 - mix) * samples + mix * wet).astype(np.float32, copy=False)
