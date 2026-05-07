from __future__ import annotations

NATO: dict[str, str] = {
    "A": "alpha", "B": "bravo", "C": "charlie", "D": "delta", "E": "echo",
    "F": "foxtrot", "G": "golf", "H": "hotel", "I": "india", "J": "juliet",
    "K": "kilo", "L": "lima", "M": "mike", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo", "S": "sierra", "T": "tango",
    "U": "uniform", "V": "victor", "W": "whiskey", "X": "x-ray",
    "Y": "yankee", "Z": "zulu",
}

DIGITS: dict[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def spell_digits(s: str) -> str:
    return " ".join(DIGITS[c] for c in s if c in DIGITS)


def spell_letters(s: str) -> str:
    return " ".join(NATO[c.upper()] for c in s if c.upper() in NATO)
