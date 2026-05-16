from __future__ import annotations

import numpy as np


def apply_tanh(samples: np.ndarray, drive_db: float) -> np.ndarray:
    """Soft saturation modeling VOGAD-style aviation voice limiter."""
    gain = 10.0 ** (drive_db / 20.0)
    return np.tanh(samples * gain).astype(np.float32, copy=False)


def apply_hard_clip(samples: np.ndarray, threshold_percentile: float) -> np.ndarray:
    """Hot-mic over-modulation. ``threshold_percentile`` in (0, 100)."""
    thresh = float(np.percentile(np.abs(samples), threshold_percentile))
    return np.clip(samples, -thresh, thresh).astype(np.float32, copy=False)


def apply_agc_pump(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    rate_hz_range: tuple[float, float] = (0.5, 2.0),
    depth_db_range: tuple[float, float] = (1.5, 4.0),
) -> np.ndarray:
    """Slow LFO amplitude modulation simulating receiver AGC chasing signal."""
    rate_hz = rng.uniform(*rate_hz_range)
    depth_db = rng.uniform(*depth_db_range)
    depth_lin = 10.0 ** (depth_db / 20.0) - 1.0

    t = np.arange(samples.shape[0], dtype=np.float32) / sample_rate
    phase = rng.uniform(0.0, 2.0 * np.pi)
    lfo = 1.0 + depth_lin * 0.5 * np.sin(2.0 * np.pi * rate_hz * t + phase)
    return (samples * lfo).astype(np.float32, copy=False)
