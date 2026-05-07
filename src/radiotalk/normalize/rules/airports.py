from __future__ import annotations

import re

from ..data._sources import iata_airports, icao_airports
from ..pipeline import Rule


_FACILITY_WORDS = (
    "Tower|Ground|Approach|Departure|Arrival|Arrivals|Center|Centre|Apron|Tracon|Radio"
)
_RE_IATA_FACILITY = re.compile(rf"\b([A-Z]{{3}})\s+({_FACILITY_WORDS})\b")
_RE_ICAO = re.compile(r"\b([A-Z]{4})\b")


def _apply_iata_facility(text: str) -> str:
    db = iata_airports()

    def repl(m: re.Match[str]) -> str:
        iata, facility = m.group(1), m.group(2)
        rec = db.get(iata)
        if rec and rec.get("city"):
            return f"{rec['city']} {facility}"
        return m.group(0)

    return _RE_IATA_FACILITY.sub(repl, text)


def _apply_icao(text: str) -> str:
    db = icao_airports()

    def repl(m: re.Match[str]) -> str:
        code = m.group(1)
        rec = db.get(code)
        if rec and rec.get("city"):
            return rec["city"]
        return code

    return _RE_ICAO.sub(repl, text)


def airport_rules() -> list[Rule]:
    return [
        Rule("airport.iata_facility", 40, _apply_iata_facility),
        Rule("airport.icao",          45, _apply_icao),
    ]
