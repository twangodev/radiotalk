from __future__ import annotations

import numpy as np


def apply_multipath_flutter(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    rate_hz_range: tuple[float, float] = (3.0, 8.0),
    depth_range: tuple[float, float] = (0.10, 0.30),
) -> np.ndarray:
    """Fast amplitude flutter from reflections off the moving airframe.
    Characteristic of airborne transmissions ("airplane flutter").
    """
    rate_hz = rng.uniform(*rate_hz_range)
    depth = rng.uniform(*depth_range)

    t = np.arange(samples.shape[0], dtype=np.float32) / sample_rate
    phase = rng.uniform(0.0, 2.0 * np.pi)
    env = 1.0 - depth * (0.5 - 0.5 * np.cos(2.0 * np.pi * rate_hz * t + phase))
    return (samples * env).astype(np.float32, copy=False)
