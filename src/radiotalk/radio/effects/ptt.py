from __future__ import annotations

import numpy as np

from .bandpass import apply_bandpass
from .noise import generate_pink_noise


def apply_key_click(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    duration_ms_range: tuple[float, float] = (5.0, 15.0),
    amplitude_range: tuple[float, float] = (0.2, 0.6),
) -> np.ndarray:
    """Short exponential transient prepended to simulate PTT key-up."""
    duration_ms = rng.uniform(*duration_ms_range)
    amp = rng.uniform(*amplitude_range)
    n = max(1, int(duration_ms * 1e-3 * sample_rate))
    t = np.arange(n, dtype=np.float32)
    decay = np.exp(-t / max(n * 0.25, 1.0)).astype(np.float32)
    click = (rng.standard_normal(n).astype(np.float32) * decay * amp)
    return np.concatenate([click, samples]).astype(np.float32, copy=False)


def apply_squelch_tail(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    duration_ms_range: tuple[float, float] = (100.0, 300.0),
    level_db_range: tuple[float, float] = (-30.0, -20.0),
) -> np.ndarray:
    """Append band-limited noise burst — the iconic ksshhk at unkey."""
    duration_ms = rng.uniform(*duration_ms_range)
    level_db = rng.uniform(*level_db_range)
    n = max(1, int(duration_ms * 1e-3 * sample_rate))

    noise = generate_pink_noise(n, rng)
    noise = apply_bandpass(noise, sample_rate)

    ramp_in = min(n, int(0.005 * sample_rate))
    ramp_out = min(n, int(0.030 * sample_rate))
    env = np.ones(n, dtype=np.float32)
    if ramp_in > 0:
        env[:ramp_in] = np.linspace(0.0, 1.0, ramp_in, dtype=np.float32)
    if ramp_out > 0:
        env[-ramp_out:] = np.linspace(1.0, 0.0, ramp_out, dtype=np.float32)
    noise = noise * env

    target_rms = 10.0 ** (level_db / 20.0)
    cur_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1e-10))
    noise = noise * (target_rms / (cur_rms + 1e-10))

    return np.concatenate([samples, noise.astype(np.float32, copy=False)])
