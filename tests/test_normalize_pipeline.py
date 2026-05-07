from __future__ import annotations

import pytest

from radiotalk.normalize import (
    Pipeline,
    Rule,
    default_pipeline,
    default_rules,
    normalize,
)
from radiotalk.normalize.rules import (
    airport_rules,
    callsign_rules,
    case_rules,
    number_rules,
    runway_rules,
)


class TestPipelineBasics:
    def test_runs_in_priority_order(self):
        order: list[str] = []

        def make(name: str, prio: int):
            def fn(text: str) -> str:
                order.append(name)
                return text
            return Rule(name, prio, fn)

        Pipeline([make("c", 30), make("a", 10), make("b", 20)])("hi")
        assert order == ["a", "b", "c"]

    def test_call_applies_rules(self):
        upper = Rule("upper", 10, str.upper)
        suffix = Rule("suffix", 20, lambda s: s + "!")
        assert Pipeline([upper, suffix])("hi") == "HI!"

    def test_trace_records_changes(self):
        upper = Rule("upper", 10, str.upper)
        noop = Rule("noop", 15, lambda s: s)
        suffix = Rule("suffix", 20, lambda s: s + "!")
        steps = Pipeline([upper, noop, suffix]).trace("hi")
        names = [s[0] for s in steps]
        assert names == ["upper", "suffix"]
        assert steps[0] == ("upper", "hi", "HI")
        assert steps[1] == ("suffix", "HI", "HI!")

    def test_excluding_drops_named_rule(self):
        upper = Rule("upper", 10, str.upper)
        suffix = Rule("suffix", 20, lambda s: s + "!")
        pipe = Pipeline([upper, suffix]).excluding("suffix")
        assert pipe("hi") == "HI"

    def test_excluding_by_prefix(self):
        a1 = Rule("foo.first", 10, lambda s: s + "1")
        a2 = Rule("foo.second", 20, lambda s: s + "2")
        b = Rule("bar.x", 30, lambda s: s + "X")
        pipe = Pipeline([a1, a2, b]).excluding("foo")
        assert pipe("") == "X"


class TestDefaultPipeline:
    def test_default_pipeline_matches_normalize(self):
        text = "DLH9462, contact tower 118.7, runway 27R"
        assert default_pipeline()(text) == normalize(text)

    def test_default_pipeline_is_cached(self):
        assert default_pipeline() is default_pipeline()

    def test_rule_set_is_complete(self):
        names = {r.name for r in default_rules()}
        # Spot-check coverage across categories.
        assert "callsign.concat" in names
        assert "callsign.word_digits" in names
        assert "airport.icao" in names
        assert "airport.iata_facility" in names
        assert "runway.designator" in names
        assert "runway.bare_designator" in names
        assert "number.frequency" in names
        assert "number.squawk" in names
        assert "number.bare" in names

    def test_rule_priorities_are_unique(self):
        priorities = [r.priority for r in default_rules()]
        assert len(priorities) == len(set(priorities)), \
            "two rules share a priority — order would be ambiguous"

    def test_rule_priorities_match_category_grouping(self):
        # Callsigns < airports < runways < numbers.
        groups = {}
        for r in default_rules():
            cat = r.name.split(".")[0]
            groups.setdefault(cat, []).append(r.priority)
        max_callsign = max(groups["callsign"])
        min_airport = min(groups["airport"])
        max_airport = max(groups["airport"])
        min_runway = min(groups["runway"])
        max_runway = max(groups["runway"])
        min_number = min(groups["number"])
        assert max_callsign < min_airport
        assert max_airport < min_runway
        assert max_runway < min_number


class TestRuleFactories:
    @pytest.mark.parametrize(
        "factory, expected_prefix",
        [
            (callsign_rules, "callsign."),
            (airport_rules, "airport."),
            (runway_rules, "runway."),
            (number_rules, "number."),
            (case_rules, "case."),
        ],
    )
    def test_factory_returns_namespaced_rules(self, factory, expected_prefix):
        rules = factory()
        assert rules
        assert all(r.name.startswith(expected_prefix) for r in rules)


class TestTraceForDebugging:
    def test_trace_explains_real_input(self):
        text = "DLH9462, contact tower 118.7, squawk 3560"
        steps = default_pipeline().trace(text)
        rules_that_fired = {s[0] for s in steps}
        assert "callsign.concat" in rules_that_fired
        assert "number.frequency" in rules_that_fired
        assert "number.squawk" in rules_that_fired
