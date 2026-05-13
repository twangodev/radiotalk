from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from .scenario import Scenario

PROMPT_VERSION = "p1"

_WAKE_NAMES = {"L": "light", "M": "medium", "H": "heavy", "J": "super"}


class ChatMessage(TypedDict):
    role: str
    content: str


_SYSTEM_TEMPLATE = (Path(__file__).parent / "prompt_system.txt").read_text()


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
    user = (
        "Generate an ATC exchange for this scenario. Plaintext only, one "
        "`SPEAKER: utterance` per line.\n\n"
        f"{_render_scenario_briefing(scenario)}"
    )
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": user},
    ]
