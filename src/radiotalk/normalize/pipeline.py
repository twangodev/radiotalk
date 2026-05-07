from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    priority: int
    apply: Callable[[str], str]


class Pipeline:
    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules: list[Rule] = sorted(rules, key=lambda r: r.priority)

    def __call__(self, text: str) -> str:
        for rule in self._rules:
            text = rule.apply(text)
        return text

    def trace(self, text: str) -> list[tuple[str, str, str]]:
        steps: list[tuple[str, str, str]] = []
        for rule in self._rules:
            before = text
            text = rule.apply(text)
            if text != before:
                steps.append((rule.name, before, text))
        return steps

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    def excluding(self, *names: str) -> "Pipeline":
        excluded = set(names)
        return Pipeline(
            r for r in self._rules
            if r.name not in excluded
            and not any(r.name.startswith(p + ".") for p in excluded)
        )
