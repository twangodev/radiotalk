"""Loader for the FAA-sourced operator registry at data/seed/operators.csv.

This replaces the hand-curated `operator_prefixes` table that used to live in
us.yaml. The CSV is produced by `scripts/build_operators_csv.py` from FAA
JO 7340.2 Chapter 3 §3 plus a small supplementary list of pseudo-operators
(NAVY, ARMY, LIFEGUARD, etc.) that controllers use but that don't have an
ICAO three-letter designator.

Schema: icao,company,country,telephony,operator_class

Exports:
    prefixes_by_class()       — {OperatorClass: [icao_prefix, ...]} for the sampler.
    telephony_for(icao)       — spoken telephony string for an ICAO, title-cased.
    operator_class_for(icao)  — class of an ICAO operator (rarely needed).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
from importlib import resources
from typing import Mapping, Sequence


@lru_cache(maxsize=1)
def _load_rows() -> tuple[dict, ...]:
    with resources.files("radiotalk.data.seed").joinpath("operators.csv").open() as f:
        return tuple(csv.DictReader(f))


@lru_cache(maxsize=1)
def prefixes_by_class() -> Mapping[str, Sequence[str]]:
    """All operator ICAO prefixes grouped by operator_class."""
    out: dict[str, list[str]] = defaultdict(list)
    for row in _load_rows():
        out[row["operator_class"]].append(row["icao"])
    return {k: tuple(v) for k, v in out.items()}


@lru_cache(maxsize=1)
def _telephony_map() -> Mapping[str, str]:
    """ICAO → spoken telephony. CSV already stores the radio-pronounceable form
    (normalization done at build time in scripts/build_operators_csv.py)."""
    return {row["icao"]: row["telephony"] for row in _load_rows()}


@lru_cache(maxsize=1)
def _class_map() -> Mapping[str, str]:
    return {row["icao"]: row["operator_class"] for row in _load_rows()}


def telephony_for(icao: str) -> str | None:
    """Return the title-cased spoken telephony for an ICAO operator code, or None."""
    return _telephony_map().get(icao.upper())


def operator_class_for(icao: str) -> str | None:
    return _class_map().get(icao.upper())


