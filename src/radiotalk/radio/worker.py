from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
import soundfile as sf

from .encode import encode_wav_pcm16
from .pipeline import apply_pipeline, pipeline_fingerprint
from .presets import pick_profile
from .schema import SAMPLE_RATE


def seed_for_row(clean_row_id: str, variant_idx: int) -> int:
    """Deterministic per-(row, variant) seed. Masked to 63 bits to fit int64."""
    blob = f"{clean_row_id}|{variant_idx}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(blob).digest()[:8], "big", signed=False,
    ) & ((1 << 63) - 1)


def _decode_audio_bytes_to_mono_float32(
    raw: bytes, target_sr: int,
) -> np.ndarray:
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    if sr != target_sr:
        try:
            import soxr
            samples = soxr.resample(samples, sr, target_sr).astype(np.float32)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr)
            samples = resample_poly(samples, target_sr // g, sr // g).astype(np.float32)
    return samples


def synthesize_variant(job: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: take a job dict, return a result dict. All inputs
    and outputs are pickle-friendly plain types.
    """
    samples = _decode_audio_bytes_to_mono_float32(job["raw_audio"], SAMPLE_RATE)
    seed = job["seed"]
    rng = np.random.default_rng(seed)
    profile = pick_profile(rng)
    result = apply_pipeline(samples, SAMPLE_RATE, profile, rng)
    wav_bytes = encode_wav_pcm16(result.samples, SAMPLE_RATE)
    return {
        "scenario_id": job["scenario_id"],
        "scenario": job["scenario"],
        "turn_idx": job["turn_idx"],
        "speaker": job["speaker"],
        "text": job["text"],
        "text_normalized": job["text_normalized"],
        "voice_id": job["voice_id"],
        "audio_bytes": wav_bytes,
        "tokens": job.get("tokens"),
        "model": job["model"],
        "prompt_version": job["prompt_version"],
        "taxonomy_version": job["taxonomy_version"],
        "tts_model": job["tts_model"],
        "clean_row_id": job["clean_row_id"],
        "variant_idx": job["variant_idx"],
        "seed": seed,
        "profile": profile.name,
        "applied_effects": result.applied_effects,
        "effective_snr_db": result.effective_snr_db,
        "pipeline_fingerprint": pipeline_fingerprint(profile),
    }
