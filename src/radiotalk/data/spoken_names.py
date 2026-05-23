"""Spoken-form lookups for ICAO airport + operator callsigns.

Airport names come from the `airportsdata` package (already a project dep).
Operator telephony comes from data/seed/operators.csv, sourced from
FAA JO 7340.2 Chapter 3 §3 plus a small supplementary list. See
radiotalk.data.operators.
"""
from __future__ import annotations

import airportsdata

from .operators import telephony_for

_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
           "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def _spell_digits(s: str) -> str:
    return " ".join(_DIGITS.get(c, c) for c in s if c.isdigit())


def spoken_callsign(icao_callsign: str) -> str:
    """Radio-spoken form of a callsign like 'UAL1234' or 'N12345'.

    Looks up the operator prefix in the FAA-sourced registry; falls back to
    spelling the prefix letter-by-letter if the prefix isn't recognized.
    GA N-numbers (callsigns matching ``N`` + digits) get the standard FAA
    "November <digits>" form.
    """
    cs = icao_callsign.strip().upper()
    if cs.startswith("N") and cs[1:].isdigit():
        return "November " + _spell_digits(cs[1:])
    # Find the longest matching FAA operator prefix.
    for plen in (9, 8, 6, 5, 4, 3, 2):
        head = cs[:plen]
        spoken = telephony_for(head)
        if spoken is not None:
            tail = cs[plen:]
            spoken_tail = _spell_digits(tail) if tail.isdigit() else tail
            return f"{spoken} {spoken_tail}".strip()
    # Unknown prefix — fall back to verbatim (uppercase).
    return cs


_AIRPORTS = airportsdata.load("ICAO")


def airport_spoken_name(icao: str) -> str:
    """Spoken city/airport name for an ICAO code (strips 'International',
    'Airport', etc. so the result sounds like radio)."""
    info = _AIRPORTS.get(icao.upper())
    if not info:
        return icao
    name = info.get("name", icao)
    for noise in (" International Airport", " International", " Regional Airport",
                  " Municipal Airport", " Memorial Airport", " Airport"):
        if name.endswith(noise):
            name = name[: -len(noise)]
            break
    if not name.strip():
        name = info.get("city", icao)
    return name.strip()


if __name__ == "__main__":
    for cs in ("UAL1234", "SWA1520", "N28327", "LAN8936", "REACH99", "FDX5",
               "SKW1234", "ASH4567", "EDV2345", "RPA1122", "LIFEGUARD7",
               "KLM55", "UPS123"):
        print(f"{cs:14s} -> {spoken_callsign(cs)}")
    for ic in ("KSFO", "KORD", "KCLT", "EGLL", "RJTT"):
        print(f"{ic} -> {airport_spoken_name(ic)}")
