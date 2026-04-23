from __future__ import annotations

import numpy as np

from radiotalk.voices.filter import FilterParams, accept_clip


def _tone(seconds: float, sr: int = 24000, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_accept_clip_within_bounds() -> None:
    audio = _tone(15.0)
    assert accept_clip(audio, sr=24000, params=FilterParams()) is True


def test_reject_too_short() -> None:
    audio = _tone(5.0)
    assert accept_clip(audio, sr=24000, params=FilterParams()) is False


def test_reject_too_long() -> None:
    audio = _tone(35.0)
    assert accept_clip(audio, sr=24000, params=FilterParams()) is False


def test_reject_silent_clip() -> None:
    audio = np.zeros(24000 * 15, dtype=np.float32)
    assert accept_clip(audio, sr=24000, params=FilterParams()) is False


def test_reject_too_quiet() -> None:
    audio = _tone(15.0, amp=0.0005)
    assert accept_clip(audio, sr=24000, params=FilterParams()) is False


def test_custom_params_can_widen_bounds() -> None:
    audio = _tone(2.0)
    params = FilterParams(min_duration_s=1.0, max_duration_s=10.0)
    assert accept_clip(audio, sr=24000, params=params) is True
