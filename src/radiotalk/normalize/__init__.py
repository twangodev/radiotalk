from __future__ import annotations

from functools import lru_cache

from .pipeline import Pipeline, Rule
from .rules import (
    airport_rules,
    callsign_rules,
    case_rules,
    default_rules,
    number_rules,
    runway_rules,
)


@lru_cache(maxsize=1)
def default_pipeline() -> Pipeline:
    return Pipeline(default_rules())


def normalize(text: str) -> str:
    return default_pipeline()(text)


@lru_cache(maxsize=1)
def _callsigns_pipeline() -> Pipeline:
    return Pipeline(callsign_rules())


@lru_cache(maxsize=1)
def _airports_pipeline() -> Pipeline:
    return Pipeline(airport_rules())


@lru_cache(maxsize=1)
def _runways_pipeline() -> Pipeline:
    return Pipeline(runway_rules())


@lru_cache(maxsize=1)
def _numbers_pipeline() -> Pipeline:
    return Pipeline(number_rules())


def expand_callsigns(text: str) -> str:
    return _callsigns_pipeline()(text)


def expand_airport_codes(text: str) -> str:
    return _airports_pipeline()(text)


def expand_runways(text: str) -> str:
    return _runways_pipeline()(text)


def normalize_numbers(text: str) -> str:
    return _numbers_pipeline()(text)


__all__ = [
    "Pipeline",
    "Rule",
    "default_pipeline",
    "default_rules",
    "normalize",
    "expand_callsigns",
    "expand_airport_codes",
    "expand_runways",
    "normalize_numbers",
]
