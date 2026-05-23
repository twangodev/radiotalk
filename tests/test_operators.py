from __future__ import annotations

from radiotalk.data.operators import (
    operator_class_for,
    prefixes_by_class,
    telephony_for,
)


def test_registry_has_all_expected_classes():
    """The FAA-sourced registry covers every class the sampler uses except
    the N-number-fallback classes (ga / training)."""
    classes = set(prefixes_by_class())
    assert "commercial" in classes
    assert "cargo" in classes
    assert "military" in classes
    assert "medevac" in classes
    assert "business" in classes
    assert "rotorcraft" in classes


def test_registry_size_reasonable():
    """v2 corpus depends on a much larger operator pool than the 50-entry
    hand-curated list it replaced. Sanity-check the order of magnitude."""
    pbc = prefixes_by_class()
    total = sum(len(v) for v in pbc.values())
    assert total >= 1000, f"only {total} operators registered; expected >= 1000"
    assert len(pbc["commercial"]) >= 500
    assert len(pbc["cargo"]) >= 20


def test_known_major_carriers_resolve():
    """Spot-check a mix of US majors, regionals, and cargo to confirm both
    that they're in the registry and their telephony is radio-pronounceable."""
    cases = {
        "AAL": "American",
        "DAL": "Delta",
        "UAL": "United",
        "SWA": "Southwest",
        "SKW": "Skywest",     # major US regional, was already in old list
        "ASH": "Air Shuttle", # Mesa, was MISSING from old list
        "EDV": "Endeavor",    # Delta Connection, was MISSING from old list
        "RPA": "Brickyard",   # Republic, was MISSING from old list
        "FDX": "Fedex",
        "RCH": "Reach",       # Air Mobility Command military
    }
    for icao, expected in cases.items():
        actual = telephony_for(icao)
        assert actual == expected, f"{icao}: expected {expected!r}, got {actual!r}"


def test_spelled_acronym_telephonies():
    """Some FAA telephonies are spelled letter-by-letter on the radio even
    though FAA writes them as single uppercase words. Build script applies
    overrides; verify a few stuck."""
    assert telephony_for("KLM") == "K L M"
    assert telephony_for("UPS") == "U P S"


def test_supplementary_pseudo_operators_present():
    """Pseudo-operators (NAVY, ARMY, LIFEGUARD, etc.) don't have FAA ICAO
    designators but are real telephonies. Build script appends them."""
    assert telephony_for("NAVY") == "Navy"
    assert telephony_for("ARMY") == "Army"
    assert telephony_for("LIFEGUARD") == "Lifeguard"
    assert telephony_for("SHELL") == "Shell"
    assert operator_class_for("NAVY") == "military"
    assert operator_class_for("LIFEGUARD") == "medevac"


def test_unknown_icao_returns_none():
    assert telephony_for("ZZZZZZ") is None
    assert operator_class_for("ZZZZZZ") is None
