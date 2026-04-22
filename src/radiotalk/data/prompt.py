from __future__ import annotations

import json
from typing import TypedDict

from .scenario import Scenario
from .transcript import model_transcript_json_schema

PROMPT_VERSION = "p1"

_WAKE_NAMES = {"L": "light", "M": "medium", "H": "heavy", "J": "super"}


class ChatMessage(TypedDict):
    role: str
    content: str


_SYSTEM_TEMPLATE = """\
You are a transcript generator for realistic air-traffic-control (ATC) radio exchanges.
You will be given a scenario briefing; produce a plausible voice-radio exchange between
ATC and the pilot(s).

Follow ICAO/FAA phraseology conventions:
- Pilots read back clearances, headings, altitudes, runway assignments, and squawk codes.
- Controllers begin transmissions with the callsign being addressed.
- Wake category drives the callsign suffix: H ("heavy") aircraft append "heavy" to the
  callsign in every transmission; J ("super") aircraft append "super"; L and M append
  nothing. The scenario briefing tells you the wake category for each aircraft — use it.
- Use standard phrases: "cleared for takeoff", "line up and wait", "contact",
  "report reaching", "maintain", "expect", "descend via", "say altitude", "traffic in sight",
  etc.
- Use numeric readback conventions (e.g., "one two thousand" only in spoken text;
  raw digits are fine since this is text).
- Keep each turn short and radio-realistic (1-3 sentences).
- The briefing lists EVERY aircraft on this frequency; one is marked [FOCAL]. The event
  is centered on the focal aircraft. Background traffic (the other aircraft listed) may
  receive routine transmissions — handoffs, taxi instructions, traffic advisories — that
  add realism. For light traffic, you may keep the exchange focal-only. For moderate or
  heavy traffic, weave in 1-3 short transmissions to/from the background aircraft.
- Use ONLY the callsigns provided in the briefing. Do not invent additional aircraft.
- Produce AT LEAST 4 turns and at most 60. Routine exchanges should be 4-8 turns;
  abnormals 6-15; emergencies 12-30. The schema enforces a hard minimum of 4.

OUTPUT CONTRACT: Respond with a single JSON object and nothing else. The object MUST match
this JSON schema exactly:

{schema}

Every turn must have speaker, callsign, facility (null for PILOT), text, intent. Intents
are short snake_case tags such as: takeoff_clearance, landing_clearance, readback,
taxi_instruction, handoff, frequency_change, squawk_assignment, altitude_change,
heading_vector, traffic_advisory, wx_advisory, pirep, emergency_declaration,
intent_statement, acknowledgement, go_around_instruction, missed_approach.
"""


def _render_aircraft_roster(scenario: Scenario) -> str:
    lines = ["Aircraft on this frequency:"]
    for i, ac in enumerate(scenario.aircraft):
        marker = "[FOCAL]" if i == 0 else "       "
        lines.append(
            f"  {marker} {ac.callsign} "
            f"({ac.aircraft_type}, {_WAKE_NAMES[ac.wake]}, {ac.operator_class})"
        )
    return "\n".join(lines)


def _render_scenario_briefing(scenario: Scenario) -> str:
    w = scenario.weather
    ceiling = (
        f"ceiling {w.ceiling_ft} ft" if w.ceiling_ft is not None else "ceiling unlimited"
    )
    focal = scenario.aircraft[0]
    lines = [
        f"Airport (ICAO): {scenario.icao}",
        f"Phase: {scenario.phase}",
        f"Frequency: {scenario.frequency_mhz:.2f} MHz",
        f"Time of day: {scenario.time_of_day}",
        f"Traffic density: {scenario.traffic_density}",
        (
            "Weather: "
            f"{w.vmc_imc}, wind {w.wind_dir_deg:03d} at {w.wind_kt} kt, "
            f"visibility {w.vis_sm} SM, {ceiling}, altimeter {w.altimeter_inhg:.2f}"
        ),
        "",
        _render_aircraft_roster(scenario),
        "",
        f"Focal aircraft: {focal.callsign} ({focal.aircraft_type})",
        f"Focal runway in use: {scenario.runway}",
        f"Focal SID/STAR: {scenario.sid_star or 'none'}",
        f"Focal assigned squawk: {scenario.squawk}",
        f"Event (centered on focal aircraft): {scenario.event}",
    ]
    return "\n".join(lines)


def build(scenario: Scenario) -> list[ChatMessage]:
    schema = json.dumps(model_transcript_json_schema(), separators=(",", ":"))
    system = _SYSTEM_TEMPLATE.format(schema=schema)
    user = (
        "Generate an ATC exchange for this scenario. Respond with JSON only.\n\n"
        f"{_render_scenario_briefing(scenario)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
