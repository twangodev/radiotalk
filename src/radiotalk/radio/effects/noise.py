from __future__ import annotations

import numpy as np


_EPS = 1e-10


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + _EPS))


def measure_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    sig_rms = _rms(signal)
    noise_rms = _rms(noise)
    return 20.0 * np.log10(sig_rms / (noise_rms + _EPS) + _EPS)


def generate_pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via FFT spectral shaping. Faster than Voss-McCartney
    cumsum stacks and gives an exact 1/sqrt(f) magnitude response.
    """
    n_freq = n_samples // 2 + 1
    real = rng.standard_normal(n_freq).astype(np.float32)
    imag = rng.standard_normal(n_freq).astype(np.float32)
    spectrum = (real + 1j * imag).astype(np.complex64)
    freqs = np.arange(n_freq, dtype=np.float32)
    freqs[0] = 1.0
    spectrum /= np.sqrt(freqs)
    spectrum[0] = 0.0
    pink = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    pink -= pink.mean()
    pink /= (pink.std() + _EPS)
    return pink


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> tuple[np.ndarray, float]:
    """Scale ``noise`` so that signal:noise ratio equals ``target_snr_db``,
    then add. Returns (mixed, effective_snr_db).
    """
    if noise.shape[0] < signal.shape[0]:
        reps = signal.shape[0] // noise.shape[0] + 1
        noise = np.tile(noise, reps)
    noise = noise[: signal.shape[0]]

    sig_rms = _rms(signal)
    noise_rms = _rms(noise)
    target_noise_rms = sig_rms / (10.0 ** (target_snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + _EPS)
    scaled_noise = noise * scale

    mixed = signal + scaled_noise
    effective = measure_snr_db(signal, scaled_noise)
    return mixed.astype(np.float32, copy=False), effective


def add_pink_noise_at_snr(
    samples: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    noise = generate_pink_noise(samples.shape[0], rng)
    return mix_at_snr(samples, noise, target_snr_db)
