from __future__ import annotations

import re

from num2words import num2words

from ..phonetic import spell_digits
from ..pipeline import Rule


_SMALL_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _altitude_phrase(n: int) -> str:
    if n == 0:
        return "zero"
    thousands = n // 1000
    hundreds = (n % 1000) // 100
    rest = n % 100
    parts: list[str] = []
    if thousands:
        if thousands < 10:
            parts.append(f"{_SMALL_WORDS[thousands]} thousand")
        else:
            parts.append(f"{spell_digits(str(thousands))} thousand")
    if hundreds:
        parts.append(f"{_SMALL_WORDS[hundreds]} hundred")
    if rest:
        parts.append(spell_digits(f"{rest:02d}"))
    return " ".join(parts)


_RE_THOUSANDS_SEP = re.compile(r"(\d),(\d{3})\b")
_RE_ALTIMETER_DECIMAL = re.compile(r"\baltimeter\s+(\d{2})\.(\d{2})\b", re.IGNORECASE)
_RE_ALTIMETER_BARE = re.compile(
    r"\b(altimeter|QNH|setting)\s+(\d{4})\b", re.IGNORECASE,
)
_RE_FLIGHT_LEVEL = re.compile(r"\bFL\s*(\d{2,3})\b", re.IGNORECASE)
_RE_FREQUENCY = re.compile(r"\b(1[0-3]\d)\.(\d{1,3})\b")
_RE_SQUAWK = re.compile(r"\bsquawk(?:ing)?\s+(\d{4})\b", re.IGNORECASE)
_RE_HEADING = re.compile(r"\b(?:heading|course)\s+(\d{1,3})\b", re.IGNORECASE)
_RE_WIND = re.compile(r"\bwinds?\s+(\d{1,3})\s+at\s+(\d{1,2})\b", re.IGNORECASE)
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_BARE_NUMBER = re.compile(r"\b\d+\b")


def _apply_thousands_separator(text: str) -> str:
    return _RE_THOUSANDS_SEP.sub(r"\1\2", text)


def _apply_altimeter_decimal(text: str) -> str:
    return _RE_ALTIMETER_DECIMAL.sub(
        lambda m: f"altimeter {spell_digits(m.group(1) + m.group(2))}", text,
    )


def _apply_altimeter_bare(text: str) -> str:
    return _RE_ALTIMETER_BARE.sub(
        lambda m: f"{m.group(1).lower()} {spell_digits(m.group(2))}", text,
    )


def _apply_flight_level(text: str) -> str:
    return _RE_FLIGHT_LEVEL.sub(
        lambda m: f"flight level {spell_digits(m.group(1))}", text,
    )


def _apply_frequency(text: str) -> str:
    return _RE_FREQUENCY.sub(
        lambda m: f"{spell_digits(m.group(1))} point {spell_digits(m.group(2))}",
        text,
    )


def _apply_squawk(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return m.group(0).replace(m.group(1), spell_digits(m.group(1)))
    return _RE_SQUAWK.sub(repl, text)


def _apply_heading(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        digits = m.group(1).zfill(3)
        return m.group(0).replace(m.group(1), spell_digits(digits))
    return _RE_HEADING.sub(repl, text)


def _apply_wind(text: str) -> str:
    return _RE_WIND.sub(
        lambda m: m.group(0)
        .replace(m.group(1), spell_digits(m.group(1).zfill(3)), 1)
        .replace(f" at {m.group(2)}", f" at {spell_digits(m.group(2).zfill(2))}", 1),
        text,
    )


def _apply_decimal(text: str) -> str:
    return _RE_DECIMAL.sub(
        lambda m: f"{spell_digits(m.group(1))} point {spell_digits(m.group(2))}",
        text,
    )


def _apply_bare_number(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        n = int(s)
        if n >= 10000:
            return _altitude_phrase(n)
        if n >= 1000:
            if n % 100 == 0:
                return _altitude_phrase(n)
            return spell_digits(s)
        if n >= 100:
            if n % 100 == 0:
                return f"{_SMALL_WORDS[n // 100]} hundred"
            return spell_digits(s)
        return num2words(n)

    return _RE_BARE_NUMBER.sub(repl, text)


def number_rules() -> list[Rule]:
    return [
        Rule("number.thousands_separator", 60, _apply_thousands_separator),
        Rule("number.altimeter_decimal",   62, _apply_altimeter_decimal),
        Rule("number.altimeter_bare",      63, _apply_altimeter_bare),
        Rule("number.flight_level",        65, _apply_flight_level),
        Rule("number.frequency",           70, _apply_frequency),
        Rule("number.squawk",              75, _apply_squawk),
        Rule("number.heading",             80, _apply_heading),
        Rule("number.wind",                85, _apply_wind),
        Rule("number.decimal",             87, _apply_decimal),
        Rule("number.bare",                90, _apply_bare_number),
    ]
