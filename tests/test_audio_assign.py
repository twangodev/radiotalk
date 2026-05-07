from __future__ import annotations

import pytest

from radiotalk.audio.assign import VoicePoolTooSmall, assign, unique_speakers


VOICE_POOL = [f"v{i:04d}" for i in range(20)]


def test_unique_speakers_dedupes_and_sorts():
    turns = [
        {"speaker": "B", "text": "..."},
        {"speaker": "A", "text": "..."},
        {"speaker": "B", "text": "..."},
        {"speaker": "C", "text": "..."},
    ]
    assert unique_speakers(turns) == ["A", "B", "C"]


def test_assign_unique_per_speaker():
    a = assign("scenario-foo", ["A", "B", "C"], VOICE_POOL)
    assert sorted(a.keys()) == ["A", "B", "C"]
    assert len(set(a.values())) == 3, "tower and pilot must be different voices"


def test_assign_deterministic():
    a1 = assign("scenario-foo", ["A", "B", "C"], VOICE_POOL)
    a2 = assign("scenario-foo", ["A", "B", "C"], VOICE_POOL)
    assert a1 == a2


def test_assign_independent_across_scenarios():
    a1 = assign("scenario-A", ["X", "Y"], VOICE_POOL)
    a2 = assign("scenario-B", ["X", "Y"], VOICE_POOL)
    # Different scenario_id → different RNG → different voices (with very
    # high probability for a 20-voice pool sampled 2-at-a-time).
    assert a1 != a2


def test_assign_invariant_to_input_order():
    a1 = assign("scenario-foo", ["A", "B", "C"], VOICE_POOL)
    a2 = assign("scenario-foo", ["C", "A", "B"], VOICE_POOL)
    assert a1 == a2


def test_assign_pool_too_small():
    with pytest.raises(VoicePoolTooSmall):
        assign("scenario-foo", ["A", "B", "C"], ["only_one"])
