from __future__ import annotations

import numpy as np


def apply_heterodyne(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    freq_hz_range: tuple[float, float] = (400.0, 2000.0),
    level_db_range: tuple[float, float] = (-24.0, -12.0),
    coverage_range: tuple[float, float] = (0.3, 1.0),
) -> np.ndarray:
    """Pure-tone beat between two co-channel carriers — the iconic squeal
    when two stations transmit simultaneously.
    """
    freq_hz = rng.uniform(*freq_hz_range)
    level_db = rng.uniform(*level_db_range)
    coverage = rng.uniform(*coverage_range)

    n_total = samples.shape[0]
    n_tone = int(n_total * coverage)
    start = rng.integers(0, max(1, n_total - n_tone + 1))

    t = np.arange(n_tone, dtype=np.float32) / sample_rate
    phase = rng.uniform(0.0, 2.0 * np.pi)
    amp = 10.0 ** (level_db / 20.0)
    tone = amp * np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)

    ramp = min(n_tone, int(0.020 * sample_rate))
    if ramp > 0:
        tone[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        tone[-ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)

    out = samples.copy()
    out[start : start + n_tone] += tone
    return out.astype(np.float32, copy=False)
