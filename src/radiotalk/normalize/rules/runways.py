from __future__ import annotations

import re

from ..phonetic import spell_digits
from ..pipeline import Rule


_SIDE = {"L": "left", "R": "right", "C": "center"}

_RE_RUNWAY = re.compile(
    r"\b(runway|rwy)\s+(\d{1,2})(?:\s*([LRClrc]))?\b",
    re.IGNORECASE,
)
_RE_BARE_RUNWAY = re.compile(r"\b(\d{1,2})([LRC])\b")


def _apply_runway(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        digits = spell_digits(m.group(2))
        side = m.group(3)
        suffix = f" {_SIDE[side.upper()]}" if side else ""
        return f"{m.group(1).lower()} {digits}{suffix}"

    return _RE_RUNWAY.sub(repl, text)


def _apply_bare_runway(text: str) -> str:
    return _RE_BARE_RUNWAY.sub(
        lambda m: f"{spell_digits(m.group(1))} {_SIDE[m.group(2)]}", text,
    )


def runway_rules() -> list[Rule]:
    return [
        Rule("runway.designator",      50, _apply_runway),
        Rule("runway.bare_designator", 55, _apply_bare_runway),
    ]
