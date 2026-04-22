from __future__ import annotations

from datetime import datetime, timezone

import pytest

from radiotalk.data.scenario import ScenarioSampler
from radiotalk.data.transcript import (
    MIN_TURNS,
    Transcript,
    TranscriptParseError,
    Turn,
    parse_turns,
    validate_turns,
)


def _scenario():
    return next(iter(ScenarioSampler(seed=42).iter(1)))


def test_parse_basic_two_speakers():
    raw = "UAL1080: sup tower\nATC: sup"
    turns = parse_turns(raw)
    assert turns == [
        Turn(speaker="UAL1080", text="sup tower"),
        Turn(speaker="ATC", text="sup"),
    ]


def test_parse_skips_blank_and_garbage_lines():
    raw = (
        "\n"
        "random preamble with no colon\n"
        "ATC: one\n"
        "\n"
        "UAL1: two\n"
        "   \n"
        "KSFO_TWR: three four\n"
    )
    turns = parse_turns(raw)
    assert [t.speaker for t in turns] == ["ATC", "UAL1", "KSFO_TWR"]
    assert turns[2].text == "three four"


def test_parse_handles_hyphenated_callsign():
    turns = parse_turns("D-AIBC: guten tag\nATC: hallo")
    assert turns[0].speaker == "D-AIBC"


def test_validate_rejects_too_few_turns():
    sc = _scenario()
    too_few = [Turn(speaker="ATC", text="x")] * (MIN_TURNS - 1)
    with pytest.raises(TranscriptParseError):
        validate_turns(too_few, sc)


def test_validate_rejects_missing_focal_callsign():
    sc = _scenario()
    turns = [Turn(speaker="ATC", text="x")] * MIN_TURNS
    with pytest.raises(TranscriptParseError):
        validate_turns(turns, sc)


def test_validate_accepts_when_focal_present():
    sc = _scenario()
    focal = sc.aircraft[0].callsign
    turns = [
        Turn(speaker="ATC", text="a"),
        Turn(speaker=focal, text="b"),
        Turn(speaker="ATC", text="c"),
        Turn(speaker=focal, text="d"),
    ]
    validate_turns(turns, sc)  # no raise


def test_transcript_roundtrip():
    sc = _scenario()
    raw = f"ATC: hi\n{sc.aircraft[0].callsign}: hello\nATC: ok\n{sc.aircraft[0].callsign}: bye"
    turns = parse_turns(raw)
    t = Transcript(
        scenario_id=sc.scenario_id,
        scenario=sc,
        raw_text=raw,
        turns=turns,
        model="test",
        generated_at=datetime.now(timezone.utc),
        prompt_version="p2",
        taxonomy_version="t1",
    )
    dumped = t.model_dump()
    assert dumped["raw_text"] == raw
    assert dumped["prompt_version"] == "p2"
    assert dumped["scenario_id"] == sc.scenario_id
    assert len(dumped["turns"]) == 4
