from __future__ import annotations

from functools import lru_cache

from . import normalize_numbers
from .pipeline import Pipeline, Rule
from .rules.numbers import number_rules


@lru_cache(maxsize=1)
def _by_name() -> dict[str, Rule]:
    return {r.name: r for r in number_rules()}


def _single(name: str):
    return Pipeline([_by_name()[name]])


def expand_altimeters(text: str) -> str:
    pipe = Pipeline(
        [_by_name()["number.altimeter_decimal"], _by_name()["number.altimeter_bare"]]
    )
    return pipe(text)


def expand_flight_levels(text: str) -> str:
    return _single("number.flight_level")(text)


def expand_frequencies(text: str) -> str:
    return _single("number.frequency")(text)


def expand_squawks(text: str) -> str:
    return _single("number.squawk")(text)


def expand_headings(text: str) -> str:
    return _single("number.heading")(text)


def expand_winds(text: str) -> str:
    return _single("number.wind")(text)


def expand_decimals(text: str) -> str:
    return _single("number.decimal")(text)


def expand_remaining_numbers(text: str) -> str:
    return _single("number.bare")(text)


__all__ = [
    "normalize_numbers",
    "expand_altimeters",
    "expand_flight_levels",
    "expand_frequencies",
    "expand_squawks",
    "expand_headings",
    "expand_winds",
    "expand_decimals",
    "expand_remaining_numbers",
]
