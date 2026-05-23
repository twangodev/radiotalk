from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from .runways import runways_for
from .scenario import Scenario
from .spoken_names import airport_spoken_name, spoken_callsign

PROMPT_VERSION = "p2"

_WAKE_NAMES = {"L": "light", "M": "medium", "H": "heavy", "J": "super"}

_FACILITY_SUFFIXES = ("GND", "TWR", "APP", "DEP", "CTR", "RAMP")


class ChatMessage(TypedDict):
    role: str
    content: str


_SYSTEM_TEMPLATE = (Path(__file__).parent / "prompt_system.txt").read_text()


def _render_facility_tags(icao: str) -> str:
    tags = ", ".join(f"{icao}_{s}" for s in _FACILITY_SUFFIXES)
    return (
        "Facility speaker tags (use EXACTLY one of these; do not invent others): "
        f"{tags}"
    )


def _render_aircraft_roster(scenario: Scenario) -> str:
    lines = ["Aircraft on this frequency:"]
    for i, ac in enumerate(scenario.aircraft):
        marker = "[FOCAL]" if i == 0 else "       "
        lines.append(
            f'  {marker} ICAO={ac.callsign}  spoken="{spoken_callsign(ac.callsign)}"  '
            f"type={ac.aircraft_type}  wake={_WAKE_NAMES[ac.wake]}  "
            f"operator={ac.operator_class}"
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
        f"Airport spoken name: {airport_spoken_name(scenario.icao)}",
        _render_facility_tags(scenario.icao),
    ]
    if scenario.phase == "center" and scenario.artcc:
        lines.append(
            f"ARTCC (use this exact name when controller self-identifies): "
            f"{scenario.artcc} Center"
        )
    real_rws = runways_for(scenario.icao)
    if real_rws:
        lines.append(
            f"Runways in use at this airport (any runway you mention — "
            f"focal, background, hold-short — MUST be one of these): "
            f"{', '.join(real_rws)}"
        )
    lines += [
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
        f"Focal aircraft ICAO callsign: {focal.callsign}",
        f"Focal aircraft spoken callsign: {spoken_callsign(focal.callsign)}",
        f"Focal aircraft type: {focal.aircraft_type}",
        f"Focal runway in use: {scenario.runway}",
        f"Focal SID/STAR: {scenario.sid_star or 'none'}",
        f"Focal assigned squawk: {scenario.squawk}",
        f"Event (centered on focal aircraft): {scenario.event}",
    ]
    return "\n".join(lines)


def build(scenario: Scenario) -> list[ChatMessage]:
    user = (
        "Generate an ATC exchange for this scenario. Plaintext only, one "
        "`SPEAKER: utterance` per line.\n\n"
        f"{_render_scenario_briefing(scenario)}"
    )
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": user},
    ]
