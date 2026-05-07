from __future__ import annotations

from ..pipeline import Rule
from .airports import airport_rules
from .callsigns import callsign_rules
from .case import case_rules
from .numbers import number_rules
from .runways import runway_rules


def default_rules() -> list[Rule]:
    return [
        *callsign_rules(),
        *airport_rules(),
        *runway_rules(),
        *number_rules(),
        *case_rules(),
    ]


__all__ = [
    "default_rules",
    "callsign_rules",
    "airport_rules",
    "runway_rules",
    "number_rules",
    "case_rules",
]
