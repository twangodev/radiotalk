from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .effects import (
    add_pink_noise_at_snr,
    apply_agc_pump,
    apply_bandpass,
    apply_codec_roundtrip,
    apply_hard_clip,
    apply_heterodyne,
    apply_key_click,
    apply_multipath_flutter,
    apply_squelch_tail,
    apply_tanh,
)
from .presets import PIPELINE_VERSION, Profile


@dataclass(frozen=True)
class PipelineResult:
    samples: np.ndarray
    applied_effects: list[str]
    effective_snr_db: float


def _roll(rng: np.random.Generator, prob: float) -> bool:
    return bool(rng.random() < prob)


def apply_pipeline(
    samples: np.ndarray,
    sample_rate: int,
    profile: Profile,
    rng: np.random.Generator,
) -> PipelineResult:
    """Probabilistic channel chain. Order is fixed; per-effect gates come
    from ``profile``. Bandpass is re-applied after any disruptive step
    per Su & Haq 2026.
    """
    out = samples.astype(np.float32, copy=True)
    applied: list[str] = []

    if _roll(rng, profile.effects["agc_pump"].prob):
        out = apply_agc_pump(out, sample_rate, rng)
        applied.append("agc_pump")

    cfg = profile.effects["tanh"]
    if _roll(rng, cfg.prob):
        drive_db = float(rng.uniform(*cfg.params["drive_db_range"]))
        out = apply_tanh(out, drive_db)
        applied.append(f"tanh(drive_db={drive_db:.1f})")

    cfg = profile.effects["hard_clip"]
    if _roll(rng, cfg.prob):
        pct = float(rng.uniform(*cfg.params["threshold_percentile_range"]))
        out = apply_hard_clip(out, pct)
        applied.append(f"hard_clip(p={pct:.1f})")

    out = apply_bandpass(out, sample_rate)
    applied.append("bandpass(300-3400,6)")

    if _roll(rng, profile.effects["multipath_flutter"].prob):
        out = apply_multipath_flutter(out, sample_rate, rng)
        applied.append("multipath_flutter")

    target_snr = profile.snr.sample(rng)
    out, effective_snr = add_pink_noise_at_snr(out, target_snr, rng)
    applied.append(f"pink_noise(snr={target_snr:.1f})")

    out = apply_bandpass(out, sample_rate, zero_phase=False)
    applied.append("bandpass(300-3400,6,fwd)")

    if _roll(rng, profile.effects["heterodyne"].prob):
        out = apply_heterodyne(out, sample_rate, rng)
        applied.append("heterodyne")

    if _roll(rng, profile.effects["key_click"].prob):
        out = apply_key_click(out, sample_rate, rng)
        applied.append("key_click")
    if _roll(rng, profile.effects["squelch_tail"].prob):
        out = apply_squelch_tail(out, sample_rate, rng)
        applied.append("squelch_tail")

    cfg = profile.effects["codec_roundtrip"]
    if _roll(rng, cfg.prob):
        codec = cfg.params["codec"]
        out = apply_codec_roundtrip(out, sample_rate, codec)
        applied.append(f"codec_roundtrip({codec})")

    peak = float(np.max(np.abs(out)) + 1e-10)
    if peak > 0.99:
        out = out * (0.99 / peak)

    return PipelineResult(
        samples=out.astype(np.float32, copy=False),
        applied_effects=applied,
        effective_snr_db=effective_snr,
    )


def pipeline_fingerprint(profile: Profile) -> str:
    from .schema import SAMPLE_RATE
    payload = {
        "version": PIPELINE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "profile": profile.name,
        "snr": {
            "mean_db": profile.snr.mean_db,
            "std_db": profile.snr.std_db,
            "low_db": profile.snr.low_db,
            "high_db": profile.snr.high_db,
        },
        "effects": {
            name: {"prob": cfg.prob, "params": cfg.params}
            for name, cfg in profile.effects.items()
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
