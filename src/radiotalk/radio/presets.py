from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PIPELINE_VERSION = "0.1.0"


@dataclass(frozen=True)
class SnrSampler:
    """ATCO2-calibrated SNR sampler. Gaussian, clipped to [low, high]."""
    mean_db: float
    std_db: float
    low_db: float = -5.0
    high_db: float = 30.0

    def sample(self, rng: np.random.Generator) -> float:
        for _ in range(8):
            v = float(rng.normal(self.mean_db, self.std_db))
            if self.low_db <= v <= self.high_db:
                return v
        return float(np.clip(v, self.low_db, self.high_db))


SNR_AGGREGATE = SnrSampler(mean_db=8.0, std_db=9.0)
SNR_CLEAN_TOWER = SnrSampler(mean_db=14.0, std_db=9.0)
SNR_NOISY_UPLINK = SnrSampler(mean_db=4.0, std_db=8.0)


@dataclass(frozen=True)
class EffectConfig:
    name: str
    prob: float
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    """A per-utterance configuration: which effects to consider, each with
    its own probability and parameter ranges. Effects are applied in the
    fixed order declared in the pipeline; this just gates them.
    """
    name: str
    snr: SnrSampler
    effects: dict[str, EffectConfig]


def _eff(name: str, prob: float, **params) -> EffectConfig:
    return EffectConfig(name=name, prob=prob, params=dict(params))


PILOT_SIDE = Profile(
    name="pilot",
    snr=SNR_NOISY_UPLINK,
    effects={
        "agc_pump": _eff("agc_pump", 0.15),
        "tanh": _eff("tanh", 0.40, drive_db_range=(3.0, 15.0)),
        "hard_clip": _eff("hard_clip", 0.15, threshold_percentile_range=(95.0, 99.0)),
        "multipath_flutter": _eff("multipath_flutter", 0.25),
        "heterodyne": _eff("heterodyne", 0.03),
        "key_click": _eff("key_click", 0.70),
        "squelch_tail": _eff("squelch_tail", 0.80),
        "codec_roundtrip": _eff("codec_roundtrip", 0.30, codec="ogg"),
    },
)


CONTROLLER_SIDE = Profile(
    name="controller",
    snr=SNR_CLEAN_TOWER,
    effects={
        "agc_pump": _eff("agc_pump", 0.10),
        "tanh": _eff("tanh", 0.30, drive_db_range=(3.0, 10.0)),
        "hard_clip": _eff("hard_clip", 0.08, threshold_percentile_range=(96.0, 99.0)),
        "multipath_flutter": _eff("multipath_flutter", 0.0),
        "heterodyne": _eff("heterodyne", 0.03),
        "key_click": _eff("key_click", 0.70),
        "squelch_tail": _eff("squelch_tail", 0.75),
        "codec_roundtrip": _eff("codec_roundtrip", 0.30, codec="ogg"),
    },
)


PROFILES: dict[str, Profile] = {
    PILOT_SIDE.name: PILOT_SIDE,
    CONTROLLER_SIDE.name: CONTROLLER_SIDE,
}


def pick_profile(rng: np.random.Generator) -> Profile:
    return PILOT_SIDE if rng.random() < 0.5 else CONTROLLER_SIDE
