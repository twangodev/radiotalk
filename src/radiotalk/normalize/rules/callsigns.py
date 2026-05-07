from __future__ import annotations

import re

from ..data._sources import airlines
from ..phonetic import spell_digits, spell_letters
from ..pipeline import Rule


_NON_CALLSIGN_PREFIXES = frozenset({"FL"})


_RE_N_NUMBER = re.compile(r"\bN(\d{1,5}[A-Z]?)\b")
_RE_CALLSIGN_CONCAT = re.compile(r"\b([A-Za-z]{2,6})(\d{1,5})\b")
_RE_CALLSIGN_SPACED = re.compile(r"\b([A-Z]{2,6})\s+(\d{1,5})\b")
_DIGIT_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
_RE_CALLSIGN_WORD_DIGITS = re.compile(
    rf"\b([A-Z]{{2,6}})((?:\s+{_DIGIT_WORD}){{1,5}})\b"
)


def _expand_n_number(suffix: str) -> str:
    digit_part = "".join(c for c in suffix if c.isdigit())
    letter_part = "".join(c for c in suffix if c.isalpha())
    spoken = f"november {spell_digits(digit_part)}"
    if letter_part:
        spoken += f" {spell_letters(letter_part)}"
    return spoken


def _apply_n_number(text: str) -> str:
    return _RE_N_NUMBER.sub(lambda m: _expand_n_number(m.group(1)), text)


def _expand_concat(prefix: str, digits: str) -> str:
    pu = prefix.upper()
    if pu in _NON_CALLSIGN_PREFIXES:
        return f"{prefix}{digits}"
    table = airlines()
    if pu in table:
        return f"{table[pu]} {spell_digits(digits)}"
    return f"{spell_letters(prefix)} {spell_digits(digits)}"


def _apply_concat(text: str) -> str:
    return _RE_CALLSIGN_CONCAT.sub(
        lambda m: _expand_concat(m.group(1), m.group(2)), text,
    )


def _apply_spaced(text: str) -> str:
    table = airlines()

    def repl(m: re.Match[str]) -> str:
        prefix, digits = m.group(1), m.group(2)
        if prefix in table:
            return f"{table[prefix]} {spell_digits(digits)}"
        return m.group(0)

    return _RE_CALLSIGN_SPACED.sub(repl, text)


def _apply_word_digits(text: str) -> str:
    table = airlines()

    def repl(m: re.Match[str]) -> str:
        prefix = m.group(1)
        digits_part = m.group(2)
        if prefix in table:
            return f"{table[prefix]}{digits_part}"
        return m.group(0)

    return _RE_CALLSIGN_WORD_DIGITS.sub(repl, text)


def callsign_rules() -> list[Rule]:
    return [
        Rule("callsign.n_number",    10, _apply_n_number),
        Rule("callsign.concat",      20, _apply_concat),
        Rule("callsign.spaced",      30, _apply_spaced),
        Rule("callsign.word_digits", 35, _apply_word_digits),
    ]
