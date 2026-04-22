from __future__ import annotations

import json

from radiotalk.data.prompt import PROMPT_VERSION, build
from radiotalk.data.scenario import ScenarioSampler
from radiotalk.data.transcript import model_transcript_json_schema


def test_build_returns_system_then_user():
    scenario = next(iter(ScenarioSampler(seed=1).iter(1)))
    msgs = build(scenario)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_system_contains_output_schema():
    scenario = next(iter(ScenarioSampler(seed=1).iter(1)))
    sys_msg = build(scenario)[0]["content"]
    schema_json = json.dumps(model_transcript_json_schema(), separators=(",", ":"))
    assert schema_json in sys_msg


def test_user_briefing_contains_scenario_fields():
    scenario = next(iter(ScenarioSampler(seed=3).iter(1)))
    user_msg = build(scenario)[1]["content"]
    assert scenario.icao in user_msg
    assert scenario.callsign in user_msg
    assert scenario.aircraft_type in user_msg
    assert scenario.squawk in user_msg
    assert scenario.runway in user_msg


def test_prompt_version_constant():
    assert PROMPT_VERSION
