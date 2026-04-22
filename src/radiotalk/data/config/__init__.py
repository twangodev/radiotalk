"""Region-level generation configs.

Each region (currently only ``us``) is described by a YAML file shipped alongside
this module. Code that needs per-region tunables (weights, prefixes, tiers,
ranges) should load a ``RegionConfig`` via :func:`load` rather than reading raw
constants.

Layout::

    radiotalk/data/config/
      __init__.py        # this module (RegionConfig + load)
      us.yaml            # default config
      <other>.yaml       # future regions

Add a new region by dropping a YAML file here; ``load("<name>")`` will pick it up.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# These Literals must stay in lock-step with the ones in ``..scenario``. They
# live here so that config parsing is self-contained and catches typos early.
Phase = Literal["ground", "tower", "approach", "center", "ramp"]
TimeOfDay = Literal["day", "night", "dawn", "dusk"]
Density = Literal["light", "moderate", "heavy"]
OperatorClass = Literal[
    "commercial", "cargo", "ga", "business",
    "military", "training", "rotorcraft", "medevac",
]
Event = Literal[
    "routine",
    "go_around", "missed_approach", "runway_change", "hold",
    "traffic_advisory", "weather_deviation", "pirep",
    "equipment_issue", "navigation_issue",
    "emergency_medical", "emergency_mechanical", "emergency_fuel",
    "emergency_bird_strike", "emergency_fire", "emergency_pressurization",
    "emergency_hydraulic", "emergency_electrical", "emergency_flight_control",
    "nordo", "minimum_fuel_advisory", "diversion", "priority_handling",
]
Region = Literal[
    "us", "canada", "uk", "europe_west", "europe_east",
    "asia_east", "asia_se", "oceanic", "me", "lat_am", "africa", "other",
]


class TierWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    default: float
    tiers: dict[int, float] = Field(default_factory=dict)

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "TierWeights":
        default = float(d.pop("default"))
        tiers = {int(k): float(v) for k, v in d.items()}
        return cls(default=default, tiers=tiers)


class WindBucket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    range: tuple[int, int]
    weight: float


class WeatherConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vmc_imc_weights: dict[Literal["VMC", "IMC"], float]
    vmc_vis_sm: tuple[float, float]
    imc_vis_sm: tuple[float, float]
    vmc_ceiling_chance: float = Field(ge=0.0, le=1.0)
    vmc_ceiling_ft: tuple[int, int]
    imc_ceiling_ft: tuple[int, int]
    altimeter_inhg: tuple[float, float]
    wind_dir_deg: tuple[int, int]
    wind_kt_buckets: list[WindBucket]


class RegionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    allowed_regions: list[Region]

    tier_weights: TierWeights
    airport_tiers: dict[str, int]

    operator_weights: dict[OperatorClass, float]
    operator_prefixes: dict[OperatorClass, list[str]]

    event_weights: dict[Event, float]

    traffic_density_weights: dict[Density, float]
    traffic_aircraft_range: dict[Density, tuple[int, int]]

    phase_weights: dict[Phase, float]
    phase_frequency_bands: dict[Phase, tuple[float, float]]

    time_of_day_weights: dict[TimeOfDay, float]

    runway_suffix_weights: dict[str, float]
    sid_star_none_chance: float = Field(ge=0.0, le=1.0)

    weather: WeatherConfig


class ConfigNotFound(FileNotFoundError):
    pass


def available() -> list[str]:
    """Return the names of config YAMLs shipped with the package."""
    return sorted(
        p.name[:-5]
        for p in resources.files(__name__).iterdir()
        if p.name.endswith(".yaml")
    )


def load(name: str = "us") -> RegionConfig:
    """Load and validate ``{name}.yaml`` into a :class:`RegionConfig`."""
    fname = f"{name}.yaml"
    try:
        text = resources.files(__name__).joinpath(fname).read_text()
    except (FileNotFoundError, IsADirectoryError) as e:
        raise ConfigNotFound(
            f"region config {fname!r} not found. Available: {available()}"
        ) from e
    raw = yaml.safe_load(text)
    # Normalize tier_weights (YAML has keys "1"/"2"/"default" — we split).
    raw_tw = raw.get("tier_weights", {})
    raw["tier_weights"] = TierWeights.from_yaml_dict(dict(raw_tw)).model_dump()
    return RegionConfig.model_validate(raw)


DEFAULT = "us"
