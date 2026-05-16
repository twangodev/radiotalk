from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


BANDPASS_LOW_HZ = 300.0
BANDPASS_HIGH_HZ = 3400.0
BANDPASS_ORDER = 6


@lru_cache(maxsize=8)
def _design_sos(
    sample_rate: int, low_hz: float, high_hz: float, order: int,
) -> np.ndarray:
    nyq = 0.5 * sample_rate
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def apply_bandpass(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
    zero_phase: bool = True,
) -> np.ndarray:
    """ITU-R M.1084 / DO-186B aero voice passband. Re-apply after any
    pitch/stretch/noise step (Su & Haq 2026 recipe). ``zero_phase=False``
    uses forward-only ``sosfilt`` — about 2× faster and the phase shift
    is realistic for the post-noise bandpass.
    """
    sos = _design_sos(sample_rate, low_hz, high_hz, order)
    out = sosfiltfilt(sos, samples) if zero_phase else sosfilt(sos, samples)
    return out.astype(np.float32, copy=False)
