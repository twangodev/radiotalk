from __future__ import annotations

import pytest

from radiotalk.data.config import (
    ConfigNotFound,
    RegionConfig,
    available,
    load,
)


def test_us_yaml_is_available():
    assert "us" in available()


def test_load_us_returns_validated_region_config():
    cfg = load("us")
    assert isinstance(cfg, RegionConfig)
    assert cfg.name == "us"
    assert cfg.allowed_regions == ["us"]


def test_load_unknown_raises():
    with pytest.raises(ConfigNotFound):
        load("atlantis")


def test_us_config_has_all_required_sections():
    cfg = load("us")
    # Tier weights + at least one tier-1 hub
    assert cfg.tier_weights.tiers[1] > cfg.tier_weights.tiers[2]
    assert cfg.airport_tiers["KSFO"] == 1
    # All operator classes present in both weights and prefixes
    assert set(cfg.operator_weights.keys()) == set(cfg.operator_prefixes.keys())
    # Event weights cover the expected buckets
    assert cfg.event_weights["routine"] > 50.0
    assert "emergency_medical" in cfg.event_weights
    # Traffic ranges scale up with density
    assert cfg.traffic_aircraft_range["heavy"][0] > cfg.traffic_aircraft_range["light"][1]
    # Phase frequency bands sane
    for band in cfg.phase_frequency_bands.values():
        assert 117.0 < band[0] < band[1] < 138.0
    # Weather knobs
    assert cfg.weather.vmc_imc_weights["VMC"] > cfg.weather.vmc_imc_weights["IMC"]
    assert len(cfg.weather.wind_kt_buckets) >= 2


def test_changing_config_changes_sample_distribution(tmp_path):
    """A custom config with different operator weights should change samples."""
    from radiotalk.data.scenario import ScenarioSampler
    cfg = load("us")
    # Force the sampler to military-only via runtime override
    sampler = ScenarioSampler(
        seed=99, config=cfg, operator_weights={"military": 1.0},
    )
    classes = {s.operator_class for s in sampler.iter(50)}
    assert classes == {"military"}
