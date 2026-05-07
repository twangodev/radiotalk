from __future__ import annotations

import re

from ..pipeline import Rule


_RE_ALL_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")


def _apply_lowercase_caps_words(text: str) -> str:
    return _RE_ALL_CAPS_WORD.sub(lambda m: m.group(0).lower(), text)


def case_rules() -> list[Rule]:
    return [Rule("case.lowercase_caps", 99, _apply_lowercase_caps_words)]
