from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from radiotalk.data.scenario import ScenarioSampler
from radiotalk.data.transcript import ModelTranscript, Transcript, Turn


def _scenario():
    return next(iter(ScenarioSampler(seed=42).iter(1)))


def test_model_transcript_rejects_under_min_turns():
    # Schema floor is 4 turns.
    with pytest.raises(ValidationError):
        ModelTranscript.model_validate({"turns": []})
    with pytest.raises(ValidationError):
        ModelTranscript.model_validate(
            {"turns": [{"speaker": "ATC", "callsign": "x", "facility": "x",
                        "text": "x", "intent": "x"}] * 3}
        )


def test_model_transcript_accepts_4_turns():
    turn = {"speaker": "ATC", "callsign": "x", "facility": "x",
            "text": "x", "intent": "x"}
    ModelTranscript.model_validate({"turns": [turn] * 4})


def test_turn_roundtrip():
    t = Turn(
        speaker="ATC",
        callsign="KSFO_TWR",
        facility="KSFO_TWR",
        text="Delta one two three, cleared for takeoff runway 28R.",
        intent="takeoff_clearance",
    )
    assert Turn.model_validate_json(t.model_dump_json()) == t


def test_transcript_has_all_version_columns():
    sc = _scenario()
    t = Transcript(
        scenario_id=sc.scenario_id,
        scenario=sc,
        turns=[
            Turn(
                speaker="ATC",
                callsign="KSFO_TWR",
                facility="KSFO_TWR",
                text="hi",
                intent="greeting",
            )
        ],
        model="Qwen/Qwen3-32B-NVFP4",
        generated_at=datetime.now(timezone.utc),
        prompt_version="p1",
        taxonomy_version="t1",
        decoding="json_schema",
    )
    dumped = t.model_dump()
    assert dumped["prompt_version"] == "p1"
    assert dumped["taxonomy_version"] == "t1"
    assert dumped["decoding"] == "json_schema"
    assert dumped["scenario_id"] == sc.scenario_id
