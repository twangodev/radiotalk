from __future__ import annotations

from radiotalk.data.prompt import PROMPT_VERSION, build
from radiotalk.data.scenario import ScenarioSampler
from radiotalk.data.spoken_names import airport_spoken_name, spoken_callsign


def test_build_returns_system_then_user():
    scenario = next(iter(ScenarioSampler(seed=1).iter(1)))
    msgs = build(scenario)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_system_specifies_plaintext_contract():
    scenario = next(iter(ScenarioSampler(seed=1).iter(1)))
    sys_msg = build(scenario)[0]["content"]
    assert "Plaintext" in sys_msg or "plaintext" in sys_msg
    assert "SPEAKER: utterance" in sys_msg
    # No JSON schema leaking into the system prompt.
    assert "json_schema" not in sys_msg.lower()
    assert "{" not in sys_msg  # no embedded JSON


def test_user_briefing_contains_scenario_fields():
    scenario = next(iter(ScenarioSampler(seed=3).iter(1)))
    user_msg = build(scenario)[1]["content"]
    assert scenario.icao in user_msg
    assert scenario.callsign in user_msg
    assert scenario.aircraft_type in user_msg
    assert scenario.squawk in user_msg
    assert scenario.runway in user_msg


def test_user_briefing_includes_spoken_forms_and_facility_tags():
    """v2 briefing must give the model the spoken airport name, spoken focal
    callsign, and a closed set of valid facility speaker tags — these are the
    fields that eliminate city-name and tag-invention hallucinations."""
    scenario = next(iter(ScenarioSampler(seed=7).iter(1)))
    user_msg = build(scenario)[1]["content"]
    assert "Airport spoken name:" in user_msg
    assert airport_spoken_name(scenario.icao) in user_msg
    assert "Focal aircraft spoken callsign:" in user_msg
    assert spoken_callsign(scenario.callsign) in user_msg
    assert "Facility speaker tags" in user_msg
    for suffix in ("GND", "TWR", "APP", "DEP", "CTR", "RAMP"):
        assert f"{scenario.icao}_{suffix}" in user_msg


def test_prompt_version_constant():
    assert PROMPT_VERSION == "p2"


def test_center_briefing_includes_artcc_line():
    """For phase=center scenarios, the briefing must inject the resolved
    ARTCC name so the model never has to pick one from a list."""
    # KSFO_CTR → Oakland Center per the artcc lookup.
    for s in ScenarioSampler(seed=11).iter(500):
        if s.phase == "center" and s.icao == "KSFO":
            user_msg = build(s)[1]["content"]
            assert "ARTCC" in user_msg
            assert "Oakland Center" in user_msg
            return
    # If we didn't land on KSFO/center in 500 draws, at least verify *some*
    # center scenario got an ARTCC line.
    for s in ScenarioSampler(seed=11).iter(500):
        if s.phase == "center" and s.artcc:
            user_msg = build(s)[1]["content"]
            assert "ARTCC" in user_msg
            assert s.artcc in user_msg
            return
    raise AssertionError("no center scenarios sampled in 500 draws")


def test_non_center_briefing_omits_artcc_line():
    """Ground/tower/approach/ramp briefings should not include the ARTCC
    line — it's only relevant when the controller actually self-identifies
    as a Center."""
    for s in ScenarioSampler(seed=13).iter(50):
        if s.phase != "center":
            user_msg = build(s)[1]["content"]
            assert "ARTCC" not in user_msg
            return
    raise AssertionError("no non-center scenarios sampled in 50 draws")
