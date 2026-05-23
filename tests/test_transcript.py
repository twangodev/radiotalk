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
        Turn(speaker=f"{sc.icao}_TWR", text="a"),
        Turn(speaker=focal, text="b"),
        Turn(speaker=f"{sc.icao}_TWR", text="c"),
        Turn(speaker=focal, text="d"),
    ]
    validate_turns(turns, sc)  # no raise


# ---------------------------------------------------------------------------
# v6.1 realism validators — post-gen rejection forces retry rather than
# shipping a known-bad transcript. One test per failure pattern.
# ---------------------------------------------------------------------------


def _focal_turns_with(text: str) -> tuple[list[Turn], object]:
    sc = _scenario()
    focal = sc.aircraft[0].callsign
    # Use a Tower speaker — most legitimate clearances are Tower-side, so the
    # role-discipline validator won't reject these clean fixtures. Avoid
    # referencing any runway in the surrounding turns so the runway-realism
    # validator doesn't trip on fixture text that's about other rules.
    twr = f"{sc.icao}_TWR"
    turns = [
        Turn(speaker=twr, text="standby"),
        Turn(speaker=focal, text=text),
        Turn(speaker=twr, text="contact tower one one eight decimal one"),
        Turn(speaker=focal, text="tower one one eight decimal one"),
    ]
    return turns, sc


@pytest.mark.parametrize("offending_text", [
    "contact tower one two one decimal niner, frequency check complete",
    "contact tower one two one decimal niner, frequency check required",
    "departure one three five decimal three, good night",
    "have a nice flight, copy",
    "have a nice landing, copy",
    "have a smooth flight, copy",
    "QNH three zero one two",
    "position and hold runway two seven left",
])
def test_validate_rejects_forbidden_phrases(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="forbidden phrase"):
        validate_turns(turns, sc)


def test_validate_rejects_point_in_frequency():
    turns, sc = _focal_turns_with("contact tower one two one point niner")
    with pytest.raises(TranscriptParseError, match="point"):
        validate_turns(turns, sc)


def test_validate_rejects_runway_as_arabic_digits():
    turns, sc = _focal_turns_with("runway 22 cleared for takeoff")
    with pytest.raises(TranscriptParseError, match="Arabic digits"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("offending_text", [
    "runway thirty two cleared for takeoff",
    "runway twenty two right line up and wait",
    "runway thirty-five left cleared to land",
    "runways thirteen right and seventeen left are closed",
])
def test_validate_rejects_runway_as_compound_words(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="compound number"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("offending_text", [
    "KSFO Tower, ready for departure",
    "KLAX Approach on one one eight decimal one",
    "PHNL Ground, taxi clearance",
])
def test_validate_rejects_icao_leak(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="ICAO"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("offending_text", [
    "Catbird seven one six five medium, cleared to land",
    "Acme one two three four light, line up and wait",
])
def test_validate_rejects_wake_suffix(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="medium/light"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("offending_text", [
    "squawk seven five zero zero",
    "squawk seven six zero zero",
    "squawk seven seven zero zero",
])
def test_validate_rejects_hijack_squawk(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="hijack"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("offending_text", [
    "Acme one two three four, [no response]",
    "ATC: Acme [pauses] cleared for takeoff",
    "[inaudible] runway zero five",
])
def test_validate_rejects_stage_directions(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="stage direction"):
        validate_turns(turns, sc)


# ---------------------------------------------------------------------------
# v6.2 role-discipline + structural validators
# ---------------------------------------------------------------------------


def _turns_with_speaker(speaker: str, text: str, sc):
    focal = sc.aircraft[0].callsign
    return [
        Turn(speaker=speaker, text=text),
        Turn(speaker=focal, text="roger"),
        Turn(speaker=speaker, text="standby"),
        Turn(speaker=focal, text="standby"),
    ]


@pytest.mark.parametrize("speaker,text,label", [
    ("KSFO_RAMP", "Acme one two three four, cleared for takeoff", "Ramp"),
    ("KSFO_RAMP", "Acme one two three four, taxi to runway two two right", "Ramp"),
    ("KSFO_RAMP", "Acme one two three four, line up and wait runway two seven", "Ramp"),
    ("KSFO_GND", "Acme one two three four, cleared for takeoff runway zero six", "Ground"),
    ("KSFO_GND", "Acme one two three four, line up and wait runway zero six", "Ground"),
    ("KSFO_APP", "Acme one two three four, cleared to land runway two two right", "Approach"),
    ("KSFO_APP", "Acme one two three four, taxi to runway two two right", "Approach"),
    ("KSFO_CTR", "Acme one two three four, cleared to land runway one six", "Center"),
    ("KSFO_CTR", "Acme one two three four, taxi to runway one six", "Center"),
    ("KSFO_CTR", "Acme one two three four, line up and wait runway one six", "Center"),
    ("KSFO_TWR", "Acme one two three four, taxi to runway two two right via Alpha", "Tower"),
    # v6.3 additions — alternative phrasings the model used to escape v6.2.
    ("KSFO_GND", "Acme one two three four, cleared to enter runway zero six", "Ground"),
    ("KSFO_GND", "Acme one two three four, cleared to enter the runway when ready", "Ground"),
    ("KSFO_TWR", "Acme one two three four, cleared to taxi runway two two right", "Tower"),
    ("KSFO_TWR", "Acme one two three four, ILS runway zero six left approved", "Tower"),
    ("KSFO_RAMP", "Acme one two three four, cleared for landing runway two two", "Ramp"),
    ("KSFO_APP", "Acme one two three four, cleared for an immediate landing", "Approach"),
])
def test_validate_rejects_role_violations(speaker: str, text: str, label: str):
    sc = _scenario()
    turns = _turns_with_speaker(speaker, text, sc)
    with pytest.raises(TranscriptParseError, match=label):
        validate_turns(turns, sc)


def test_validate_rejects_no_controller_voice():
    sc = _scenario()
    focal = sc.aircraft[0].callsign
    turns = [
        Turn(speaker=focal, text="cleared for takeoff runway two two right"),
        Turn(speaker=focal, text="climbing to four thousand"),
        Turn(speaker=focal, text="contact departure one one nine decimal one"),
        Turn(speaker=focal, text="departure one one nine decimal one"),
    ]
    with pytest.raises(TranscriptParseError, match="no controller voice"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("artcc_text", [
    "Pleasanton Center, November three zero four zero eight, ident",
    "Iowa County Center, climb and maintain one one thousand",
    "Scammon Bay Center, radar contact",
    "San Francisco Center, contact Approach one two zero decimal one",
])
def test_validate_rejects_invented_artcc(artcc_text: str):
    sc = _scenario()
    turns = _turns_with_speaker("KSFO_CTR", artcc_text, sc)
    with pytest.raises(TranscriptParseError, match="invented ARTCC"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("artcc_text", [
    "Oakland Center, November three zero four zero eight, ident",
    "Salt Lake Center, climb and maintain one one thousand",
    "Fort Worth Center, radar contact",
    "Kansas City Center, contact Approach one two zero decimal one",
    "Los Angeles Center, descend and maintain six thousand",
    "New York Center, climb and maintain one zero thousand",
])
def test_validate_accepts_real_artcc(artcc_text: str):
    sc = _scenario()
    turns = _turns_with_speaker("KSFO_CTR", artcc_text, sc)
    validate_turns(turns, sc)  # must not raise


@pytest.mark.parametrize("clean_text", [
    "contact departure one three five decimal one, good day",
    "altimeter two niner niner two, wind two five zero at one zero",
    "squawk four three two one, climb and maintain one one thousand",
])
def test_validate_accepts_clean_phraseology(clean_text: str):
    """Sanity check: the validator does not over-reject legitimate ATC.
    Runway-containing phrases are tested separately via fixtures that pick
    a real runway for the scenario's airport."""
    turns, sc = _focal_turns_with(clean_text)
    validate_turns(turns, sc)  # must not raise


def test_validate_accepts_real_runway_for_airport():
    """The runway-realism validator must accept any runway in the airport's
    actual runway list, using the focal scenario.runway as the canonical
    one."""
    from radiotalk.data.runways import runways_for
    sc = _scenario()
    real = runways_for(sc.icao)
    assert real, f"scenario airport {sc.icao} should have runway data"
    rw_digits = " ".join({"0": "zero", "1": "one", "2": "two", "3": "three",
                          "4": "four", "5": "five", "6": "six", "7": "seven",
                          "8": "eight", "9": "nine"}[c] for c in real[0][:2])
    side_word = {"L": " left", "R": " right", "C": " center", "": ""}[real[0][2:]]
    text = f"cleared for takeoff runway {rw_digits}{side_word}"
    turns, _ = _focal_turns_with(text)
    validate_turns(turns, sc)  # must not raise


@pytest.mark.parametrize("offending_text", [
    "contact tower one one seven decimal zero",       # 117.0 navaid
    "contact approach one one two decimal five",       # 112.5 navaid
    "contact approach one one four decimal two",       # 114.2 navaid
    "contact ground one zero eight decimal three",     # 108.3 navaid
    "contact tower one three seven decimal five",      # 137.5 above band
])
def test_validate_rejects_navaid_band_frequency(offending_text: str):
    turns, sc = _focal_turns_with(offending_text)
    with pytest.raises(TranscriptParseError, match="VHF voice band"):
        validate_turns(turns, sc)


@pytest.mark.parametrize("clean_text", [
    "contact tower one one eight decimal three",       # 118.3 valid
    "contact departure one three five decimal niner",  # 135.9 valid
    "contact center one two seven decimal five",       # 127.5 valid
    "contact ground one two one decimal niner",        # 121.9 valid
])
def test_validate_accepts_voice_band_frequency(clean_text: str):
    turns, sc = _focal_turns_with(clean_text)
    validate_turns(turns, sc)


def test_validate_rejects_off_airport_runway():
    """Spoken runway must exist at the named airport."""
    from radiotalk.data.runways import runways_for
    sc = _scenario()
    real = runways_for(sc.icao)
    # Pick a runway number that doesn't exist at this airport.
    digit_words = {"0": "zero", "1": "one", "2": "two", "3": "three",
                   "4": "four", "5": "five", "6": "six", "7": "seven",
                   "8": "eight", "9": "nine"}
    fake_num = None
    real_prefixes = {rw[:2] for rw in real}
    for candidate in range(1, 37):
        if f"{candidate:02d}" not in real_prefixes:
            fake_num = f"{candidate:02d}"
            break
    assert fake_num, f"no fake runway possible at {sc.icao}"
    fake_spoken = f"{digit_words[fake_num[0]]} {digit_words[fake_num[1]]}"
    focal = sc.aircraft[0].callsign
    twr = f"{sc.icao}_TWR"
    turns = [
        Turn(speaker=twr, text="standby"),
        Turn(speaker=focal, text="contact tower"),
        Turn(speaker=twr, text=f"cleared for takeoff runway {fake_spoken}"),
        Turn(speaker=focal, text=f"cleared for takeoff {fake_spoken}"),
    ]
    with pytest.raises(TranscriptParseError, match="not at"):
        validate_turns(turns, sc)


def test_transcript_roundtrip():
    sc = _scenario()
    raw = f"ATC: hi\n{sc.aircraft[0].callsign}: hello\nATC: ok\n{sc.aircraft[0].callsign}: bye"
    turns = parse_turns(raw)
    t = Transcript(
        scenario_id=sc.scenario_id,
        scenario=sc,
        turns=turns,
        model="test",
        generated_at=datetime.now(timezone.utc),
        prompt_version="p2",
        taxonomy_version="t1",
    )
    dumped = t.model_dump()
    assert dumped["prompt_version"] == "p2"
    assert dumped["scenario_id"] == sc.scenario_id
    assert len(dumped["turns"]) == 4
