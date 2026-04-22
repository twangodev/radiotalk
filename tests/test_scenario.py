from __future__ import annotations

from pathlib import Path

from radiotalk.data.scenario import (
    CustomWeighter,
    Scenario,
    ScenarioSampler,
    UniformWeighter,
    region_for_icao,
)


def test_sampler_is_deterministic_given_seed():
    a = list(ScenarioSampler(seed=123).iter(20))
    b = list(ScenarioSampler(seed=123).iter(20))
    assert a == b


def test_different_seeds_produce_different_sequences():
    a = list(ScenarioSampler(seed=1).iter(20))
    b = list(ScenarioSampler(seed=2).iter(20))
    assert a != b


def test_scenario_id_is_stable_and_deterministic():
    s = next(iter(ScenarioSampler(seed=7).iter(1)))
    sid = s.scenario_id
    again = Scenario.model_validate_json(s.model_dump_json()).scenario_id
    assert sid == again


def test_fast_forward_matches_drop_in_resume():
    s1 = ScenarioSampler(seed=99)
    full = list(s1.iter(50))

    s2 = ScenarioSampler(seed=99)
    s2.fast_forward(30)
    resumed = list(s2.iter(20))
    assert full[30:] == resumed


def test_uniform_weighter_distribution_more_even_than_tier(monkeypatch):
    # Draw a lot of samples; the config-driven default tier weighter should be
    # heavily skewed to tier-1 airports, uniform should spread across many more.
    tier_icaos = {s.icao for s in ScenarioSampler(seed=0).iter(500)}
    uniform_icaos = {
        s.icao for s in ScenarioSampler(seed=0, weighter=UniformWeighter()).iter(500)
    }
    assert len(uniform_icaos) > len(tier_icaos)


def test_custom_weighter_only_samples_listed_airports(tmp_path: Path):
    csv = tmp_path / "w.csv"
    csv.write_text("icao,weight\nKSFO,1\nKJFK,1\n")
    weighter = CustomWeighter(csv)
    seen = {s.icao for s in ScenarioSampler(seed=11, weighter=weighter).iter(200)}
    assert seen <= {"KSFO", "KJFK"}


def test_squawk_avoids_reserved_codes():
    for s in ScenarioSampler(seed=2026).iter(2000):
        assert s.squawk not in {"7500", "7600", "7700"}
        assert len(s.squawk) == 4
        assert all(c in "01234567" for c in s.squawk)


def test_frequency_band_matches_phase():
    for s in ScenarioSampler(seed=5).iter(500):
        assert 118.0 <= s.frequency_mhz <= 135.9


def test_weather_vmc_imc_and_fields():
    for s in ScenarioSampler(seed=8).iter(200):
        w = s.weather
        assert w.vmc_imc in ("VMC", "IMC")
        assert 0 <= w.wind_dir_deg <= 360
        if w.vmc_imc == "VMC":
            assert w.vis_sm >= 5.0
        else:
            assert w.ceiling_ft is not None and w.ceiling_ft <= 2000


def test_region_classification():
    assert region_for_icao("KSFO") == "us"
    assert region_for_icao("PHNL") == "us"
    assert region_for_icao("CYYZ") == "canada"
    assert region_for_icao("EGLL") == "uk"
    assert region_for_icao("EIDW") == "uk"
    assert region_for_icao("EDDF") == "europe_west"
    assert region_for_icao("LFPG") == "europe_west"
    assert region_for_icao("EPWA") == "europe_east"
    assert region_for_icao("LTFM") == "me"
    assert region_for_icao("RJAA") == "asia_east"
    assert region_for_icao("ZBAA") == "asia_east"
    assert region_for_icao("VTBS") == "asia_se"
    assert region_for_icao("WSSS") == "asia_se"
    assert region_for_icao("YSSY") == "oceanic"
    assert region_for_icao("OMDB") == "me"
    assert region_for_icao("SBGR") == "lat_am"
    assert region_for_icao("MMMX") == "lat_am"
    assert region_for_icao("FAOR") == "africa"


def test_scenario_carries_region_and_is_derived_from_icao():
    for s in ScenarioSampler(seed=13).iter(50):
        assert s.region == region_for_icao(s.icao)


def test_default_allowed_regions_is_us_only():
    import airportsdata
    ap = airportsdata.load("ICAO")
    for s in ScenarioSampler(seed=26).iter(200):
        assert s.region == "us"
        # All US airports per airportsdata's country code.
        assert ap[s.icao]["country"] == "US"


def test_allowed_regions_override_filters_pool():
    # With only canada allowed, every scenario should be a Canadian airport.
    sampler = ScenarioSampler(seed=27, allowed_regions=frozenset({"canada"}))
    for s in sampler.iter(50):
        assert s.region == "canada"
        assert s.icao.startswith("C")


def test_empty_allowed_regions_after_filter_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        # No region called "atlantis" exists, so the pool is empty.
        ScenarioSampler(seed=28, allowed_regions=frozenset({"atlantis"}))  # type: ignore[arg-type]


def test_operator_class_is_sampled():
    seen_classes = {s.operator_class for s in ScenarioSampler(seed=14).iter(500)}
    # Should see multiple classes across 500 draws.
    assert len(seen_classes) >= 4


def test_aircraft_matches_operator_class():
    # Each aircraft entry should be a valid type for its operator_class.
    military_types = {
        "C30J", "C17", "K35R", "E3CF", "P8",
        "F16", "F18", "F35", "H60", "H47", "V22",
    }
    ga_types_no_overlap = {"C172", "PA28", "SR22", "DA40"}
    for s in ScenarioSampler(seed=15).iter(300):
        for ac in s.aircraft:
            if ac.operator_class == "commercial":
                assert ac.aircraft_type not in ga_types_no_overlap
            if ac.operator_class == "military":
                assert ac.aircraft_type in military_types


def test_aircraft_list_size_scales_with_traffic_density():
    by_density: dict[str, list[int]] = {"light": [], "moderate": [], "heavy": []}
    for s in ScenarioSampler(seed=24).iter(500):
        by_density[s.traffic_density].append(s.n_aircraft)
    # All scenarios have at least 1 aircraft.
    assert all(n >= 1 for ns in by_density.values() for n in ns)
    # Density categories observe their bounds.
    assert all(1 <= n <= 2 for n in by_density["light"])
    assert all(2 <= n <= 4 for n in by_density["moderate"])
    assert all(5 <= n <= 8 for n in by_density["heavy"])
    # Heavy on average has more than light.
    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert avg(by_density["heavy"]) > avg(by_density["light"])


def test_focal_accessors_match_aircraft_zero():
    for s in ScenarioSampler(seed=25).iter(50):
        assert s.callsign == s.aircraft[0].callsign
        assert s.aircraft_type == s.aircraft[0].aircraft_type
        assert s.wake == s.aircraft[0].wake
        assert s.operator_class == s.aircraft[0].operator_class
        assert s.n_aircraft == len(s.aircraft)


def test_is_emergency_derived_field():
    for s in ScenarioSampler(seed=16).iter(200):
        assert s.is_emergency == s.event.startswith("emergency_")


def test_is_towered_derived_field():
    for s in ScenarioSampler(seed=17).iter(200):
        assert s.is_towered == (s.phase in ("ground", "tower", "approach", "center"))


def test_is_emergency_and_is_towered_serialize_to_model_dump():
    s = next(iter(ScenarioSampler(seed=18).iter(1)))
    dumped = s.model_dump()
    assert "is_emergency" in dumped
    assert "is_towered" in dumped


def test_scenario_id_excludes_computed_fields():
    # scenario_id is computed over real fields only; round-tripping through
    # model_dump (which includes computed fields) should reconstitute the same
    # scenario and yield the same hash.
    s = next(iter(ScenarioSampler(seed=19).iter(1)))
    sid = s.scenario_id
    again = Scenario.model_validate_json(s.model_dump_json()).scenario_id
    assert sid == again


def test_event_weights_override_biases_distribution():
    emergency_only = {"emergency_medical": 1.0}
    events = [
        s.event for s in ScenarioSampler(seed=20, event_weights=emergency_only).iter(50)
    ]
    assert all(e == "emergency_medical" for e in events)


def test_wake_is_propagated_from_aircraft_csv():
    seen = {s.wake for s in ScenarioSampler(seed=22).iter(500)}
    assert seen.issubset({"L", "M", "H", "J"})
    # Across 500 samples we should see at least light + heavy.
    assert "H" in seen


def test_wake_matches_aircraft_type_lookup(tmp_path):
    # Sanity check a couple of known mappings from data/seed/aircraft.csv.
    expected = {"B748": "H", "A388": "J", "C172": "L", "B738": "M"}
    for s in ScenarioSampler(seed=23).iter(2000):
        for ac in s.aircraft:
            if ac.aircraft_type in expected:
                assert ac.wake == expected[ac.aircraft_type]


def test_operator_weights_override_biases_distribution():
    military_only = {"military": 1.0}
    classes = [
        s.operator_class
        for s in ScenarioSampler(seed=21, operator_weights=military_only).iter(50)
    ]
    assert all(c == "military" for c in classes)
