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


# ---------------------------------------------------------------------------
# Per-operator activity weights derived from OpenSky ADS-B archive.
# See scripts/build_operator_weights.py — derived from 30 days (2022-09)
# of US-touching flights, ~1.5M flights total. Used by the sampler so
# within-class operator selection follows real-world activity (American /
# Delta / United / Southwest dominate commercial; the special "N" entry
# triggers random N-number synthesis for GA).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_weights() -> tuple[dict, ...]:
    """Load operator_weights.csv as a tuple of dict rows."""
    try:
        with resources.files("radiotalk.data.seed").joinpath("operator_weights.csv").open() as f:
            return tuple(csv.DictReader(f))
    except FileNotFoundError:
        return ()


@lru_cache(maxsize=1)
def weighted_prefixes_by_class() -> Mapping[str, tuple[tuple[str, float], ...]]:
    """{operator_class: ((icao, weight), ...)} for activity-weighted sampling.

    Falls back to uniform (weight 1.0 each) if operator_weights.csv is missing
    so the sampler still works on a registry-only install.
    """
    weights = _load_weights()
    if not weights:
        # Uniform fallback derived from operators.csv alone.
        out: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in _load_rows():
            out[row["operator_class"]].append((row["icao"], 1.0))
        return {k: tuple(v) for k, v in out.items()}
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in weights:
        cls = row["operator_class"]
        out[cls].append((row["icao"], float(row["weight"])))
    return {k: tuple(v) for k, v in out.items()}


