from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterParams:
    min_duration_s: float = 10.0
    max_duration_s: float = 30.0
    min_rms_dbfs: float = -40.0


def accept_clip(audio: np.ndarray, sr: int, params: FilterParams) -> bool:
    if audio.ndim != 1:
        return False
    duration = audio.shape[0] / sr
    if not (params.min_duration_s <= duration <= params.max_duration_s):
        return False
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    if rms <= 0:
        return False
    rms_dbfs = 20.0 * math.log10(rms)
    if rms_dbfs < params.min_rms_dbfs:
        return False
    return True
