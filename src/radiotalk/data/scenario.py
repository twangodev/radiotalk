from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Protocol

if TYPE_CHECKING:
    from .config import RegionConfig

import airportsdata
from pydantic import BaseModel, ConfigDict, Field, computed_field

TAXONOMY_VERSION = "t1"

Phase = Literal["ground", "tower", "approach", "center", "ramp"]
TimeOfDay = Literal["day", "night", "dawn", "dusk"]
Density = Literal["light", "moderate", "heavy"]
OperatorClass = Literal[
    "commercial",
    "cargo",
    "ga",
    "business",
    "military",
    "training",
    "rotorcraft",
    "medevac",
]
Region = Literal[
    "us",
    "canada",
    "uk",
    "europe_west",
    "europe_east",
    "asia_east",
    "asia_se",
    "oceanic",
    "me",
    "lat_am",
    "africa",
    "other",
]
Event = Literal[
    # Routine
    "routine",
    # Abnormals
    "go_around",
    "missed_approach",
    "runway_change",
    "hold",
    "traffic_advisory",
    "weather_deviation",
    "pirep",
    "equipment_issue",
    "navigation_issue",
    # Emergencies
    "emergency_medical",
    "emergency_mechanical",
    "emergency_fuel",
    "emergency_bird_strike",
    "emergency_fire",
    "emergency_pressurization",
    "emergency_hydraulic",
    "emergency_electrical",
    "emergency_flight_control",
    # Rare
    "nordo",
    "minimum_fuel_advisory",
    "diversion",
    "priority_handling",
]
VmcImc = Literal["VMC", "IMC"]
Wake = Literal["L", "M", "H", "J"]

TOWERED_PHASES: tuple[Phase, ...] = ("ground", "tower", "approach", "center")


class Weather(BaseModel):
    model_config = ConfigDict(frozen=True)
    wind_dir_deg: int = Field(ge=0, le=360)
    wind_kt: int = Field(ge=0, le=80)
    vis_sm: float = Field(ge=0.0, le=10.0)
    ceiling_ft: int | None = Field(ge=0, le=50000, default=None)
    altimeter_inhg: float = Field(ge=28.0, le=31.5)
    vmc_imc: VmcImc


class Aircraft(BaseModel):
    """One aircraft on the frequency."""

    model_config = ConfigDict(frozen=True)
    callsign: str
    aircraft_type: str
    wake: Wake
    operator_class: OperatorClass


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)
    icao: str
    region: Region
    phase: Phase
    aircraft: list[Aircraft]   # focal aircraft is aircraft[0]
    runway: str
    sid_star: str | None
    squawk: str
    frequency_mhz: float
    weather: Weather
    time_of_day: TimeOfDay
    traffic_density: Density
    event: Event

    # ---- Focal-aircraft accessors (computed → ergonomic Parquet columns) ----

    @computed_field  # type: ignore[prop-decorator]
    @property
    def callsign(self) -> str:
        return self.aircraft[0].callsign

    @computed_field  # type: ignore[prop-decorator]
    @property
    def aircraft_type(self) -> str:
        return self.aircraft[0].aircraft_type

    @computed_field  # type: ignore[prop-decorator]
    @property
    def wake(self) -> Wake:
        return self.aircraft[0].wake

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operator_class(self) -> OperatorClass:
        return self.aircraft[0].operator_class

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_aircraft(self) -> int:
        return len(self.aircraft)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_emergency(self) -> bool:
        return self.event.startswith("emergency_")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_towered(self) -> bool:
        return self.phase in TOWERED_PHASES

    _COMPUTED = frozenset({
        "callsign", "aircraft_type", "wake", "operator_class",
        "n_aircraft", "is_emergency", "is_towered",
    })

    @property
    def scenario_id(self) -> str:
        payload = self.model_dump(mode="json", exclude=self._COMPUTED)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------

# ISO 3166-1 alpha-2 country code → region bucket. Anything not in this map
# (or any ICAO not in airportsdata) falls through to "other".
_COUNTRY_TO_REGION: dict[str, Region] = {
    # United States + territories
    "US": "us", "PR": "us", "VI": "us", "GU": "us",
    "MP": "us", "AS": "us", "UM": "us",
    # Canada
    "CA": "canada",
    # UK + Ireland (shared phraseology dialect)
    "GB": "uk", "IE": "uk",
    # Western Europe
    "AT": "europe_west", "BE": "europe_west", "CH": "europe_west",
    "DE": "europe_west", "DK": "europe_west", "ES": "europe_west",
    "FI": "europe_west", "FO": "europe_west", "FR": "europe_west",
    "GI": "europe_west", "GL": "europe_west", "GR": "europe_west",
    "IS": "europe_west", "IT": "europe_west", "LI": "europe_west",
    "LU": "europe_west", "MC": "europe_west", "MT": "europe_west",
    "NL": "europe_west", "NO": "europe_west", "PT": "europe_west",
    "SE": "europe_west", "SM": "europe_west", "VA": "europe_west",
    "AD": "europe_west",
    # Eastern Europe + former USSR
    "AL": "europe_east", "BA": "europe_east", "BG": "europe_east",
    "BY": "europe_east", "CZ": "europe_east", "EE": "europe_east",
    "HR": "europe_east", "HU": "europe_east", "LT": "europe_east",
    "LV": "europe_east", "MD": "europe_east", "ME": "europe_east",
    "MK": "europe_east", "PL": "europe_east", "RO": "europe_east",
    "RS": "europe_east", "RU": "europe_east", "SI": "europe_east",
    "SK": "europe_east", "UA": "europe_east", "XK": "europe_east",
    # Middle East + Caucasus + Central Asia
    "AE": "me", "BH": "me", "IL": "me", "IQ": "me", "IR": "me",
    "JO": "me", "KW": "me", "LB": "me", "OM": "me", "PS": "me",
    "QA": "me", "SA": "me", "SY": "me", "TR": "me", "YE": "me",
    "AM": "me", "AZ": "me", "GE": "me", "KZ": "me", "KG": "me",
    "TJ": "me", "TM": "me", "UZ": "me", "AF": "me",
    # East Asia
    "CN": "asia_east", "JP": "asia_east", "KP": "asia_east",
    "KR": "asia_east", "MO": "asia_east", "MN": "asia_east",
    "TW": "asia_east", "HK": "asia_east",
    # South + Southeast Asia
    "BN": "asia_se", "ID": "asia_se", "KH": "asia_se",
    "LA": "asia_se", "MM": "asia_se", "MY": "asia_se",
    "PH": "asia_se", "SG": "asia_se", "TH": "asia_se",
    "TL": "asia_se", "VN": "asia_se",
    "IN": "asia_se", "BD": "asia_se", "BT": "asia_se",
    "LK": "asia_se", "MV": "asia_se", "NP": "asia_se", "PK": "asia_se",
    # Oceania / Pacific
    "AU": "oceanic", "NZ": "oceanic", "FJ": "oceanic", "FM": "oceanic",
    "KI": "oceanic", "MH": "oceanic", "NR": "oceanic", "NU": "oceanic",
    "NC": "oceanic", "PG": "oceanic", "PF": "oceanic", "PW": "oceanic",
    "SB": "oceanic", "TK": "oceanic", "TO": "oceanic", "TV": "oceanic",
    "VU": "oceanic", "WS": "oceanic", "CK": "oceanic",
    # Latin America + Caribbean
    "AG": "lat_am", "AI": "lat_am", "AR": "lat_am", "AW": "lat_am",
    "BB": "lat_am", "BL": "lat_am", "BM": "lat_am", "BO": "lat_am",
    "BQ": "lat_am", "BR": "lat_am", "BS": "lat_am", "BZ": "lat_am",
    "CL": "lat_am", "CO": "lat_am", "CR": "lat_am", "CU": "lat_am",
    "CW": "lat_am", "DM": "lat_am", "DO": "lat_am", "EC": "lat_am",
    "FK": "lat_am", "GD": "lat_am", "GF": "lat_am", "GP": "lat_am",
    "GT": "lat_am", "GY": "lat_am", "HN": "lat_am", "HT": "lat_am",
    "JM": "lat_am", "KN": "lat_am", "KY": "lat_am", "LC": "lat_am",
    "MF": "lat_am", "MQ": "lat_am", "MS": "lat_am", "MX": "lat_am",
    "NI": "lat_am", "PA": "lat_am", "PE": "lat_am", "PM": "lat_am",
    "PY": "lat_am", "SR": "lat_am", "SV": "lat_am", "SX": "lat_am",
    "TC": "lat_am", "TT": "lat_am", "UY": "lat_am", "VC": "lat_am",
    "VE": "lat_am", "VG": "lat_am",
    # Africa
    "DZ": "africa", "EG": "africa", "LY": "africa", "MA": "africa",
    "TN": "africa", "SD": "africa", "SS": "africa", "EH": "africa",
    "ER": "africa", "DJ": "africa", "ET": "africa", "KE": "africa",
    "SO": "africa", "UG": "africa", "RW": "africa", "BI": "africa",
    "TZ": "africa", "MW": "africa", "MZ": "africa", "ZW": "africa",
    "ZM": "africa", "CD": "africa", "CG": "africa", "GA": "africa",
    "CM": "africa", "CF": "africa", "TD": "africa", "AO": "africa",
    "NA": "africa", "BW": "africa", "ZA": "africa", "LS": "africa",
    "SZ": "africa", "MG": "africa", "MU": "africa", "SC": "africa",
    "KM": "africa", "RE": "africa", "YT": "africa",
    "NG": "africa", "BJ": "africa", "BF": "africa", "CV": "africa",
    "CI": "africa", "GH": "africa", "GW": "africa", "GN": "africa",
    "LR": "africa", "ML": "africa", "MR": "africa", "NE": "africa",
    "SL": "africa", "SN": "africa", "GM": "africa", "TG": "africa",
    "ST": "africa",
}


_AIRPORTS_CACHE: dict | None = None

_ICAO_RE = re.compile(r"^[A-Z]{4}$")


def _airports() -> dict:
    """Real ICAO-coded airports only (excludes FAA LIDs like '00AA')."""
    global _AIRPORTS_CACHE
    if _AIRPORTS_CACHE is None:
        raw = airportsdata.load("ICAO")
        _AIRPORTS_CACHE = {k: v for k, v in raw.items() if _ICAO_RE.match(k)}
    return _AIRPORTS_CACHE


def region_for_icao(icao: str) -> Region:
    """Look up the region for an ICAO via airportsdata's country code."""
    info = _airports().get(icao)
    if info is None:
        return "other"
    return _COUNTRY_TO_REGION.get(info.get("country", ""), "other")


# ---------------------------------------------------------------------------
# Airport weighters
# ---------------------------------------------------------------------------


class AirportWeighter(Protocol):
    name: str

    def icaos_and_weights(self) -> tuple[list[str], list[float]]: ...


class UniformWeighter:
    name = "uniform"

    def __init__(self) -> None:
        self._airports = _airports()

    def icaos_and_weights(self) -> tuple[list[str], list[float]]:
        icaos = list(self._airports.keys())
        return icaos, [1.0] * len(icaos)


class TierWeighter:
    """Weights airports according to a RegionConfig's tier assignments."""

    def __init__(
        self,
        tiers: dict[str, int],
        tier_weights: dict[int, float],
        default_weight: float,
        *,
        name: str = "tier_v1",
    ) -> None:
        self.name = name
        self._airports = _airports()
        self._tiers = tiers
        self._tier_weights = tier_weights
        self._default = default_weight

    def icaos_and_weights(self) -> tuple[list[str], list[float]]:
        icaos = list(self._airports.keys())
        weights = [
            self._tier_weights.get(self._tiers.get(ic, -1), self._default)
            for ic in icaos
        ]
        return icaos, weights


class CustomWeighter:
    def __init__(self, path: Path, name: str | None = None) -> None:
        self.name = name or f"custom:{path.name}"
        self._weights = self._load(path)

    @staticmethod
    def _load(path: Path) -> dict[str, float]:
        with path.open() as f:
            reader = csv.DictReader(f)
            return {row["icao"]: float(row["weight"]) for row in reader}

    def icaos_and_weights(self) -> tuple[list[str], list[float]]:
        icaos = list(self._weights.keys())
        return icaos, [self._weights[ic] for ic in icaos]


# ---------------------------------------------------------------------------
# Scenario sampling
# ---------------------------------------------------------------------------


def _load_aircraft() -> list[dict]:
    with resources.files("radiotalk.data.seed").joinpath("aircraft.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["operator_classes"] = tuple(r["operator_classes"].split("|"))
    return rows


_RESERVED_SQUAWKS = {"7500", "7600", "7700"}


class ScenarioSampler:
    """Draws Scenarios deterministically, driven by a RegionConfig.

    Weighter override: by default the sampler builds a TierWeighter from the
    config's tier assignments. Pass ``weighter=`` to swap in Uniform or Custom.
    """

    def __init__(
        self,
        seed: int,
        config: "RegionConfig | None" = None,
        weighter: AirportWeighter | None = None,
        *,
        event_weights: dict[Event, float] | None = None,
        operator_weights: dict[OperatorClass, float] | None = None,
        allowed_regions: frozenset[Region] | None = None,
    ) -> None:
        if config is None:
            from .config import load as _load_config
            config = _load_config()
        self._cfg = config
        self._seed = seed
        self._rng = random.Random(seed)

        self._weighter = weighter or TierWeighter(
            tiers=config.airport_tiers,
            tier_weights=config.tier_weights.tiers,
            default_weight=config.tier_weights.default,
        )

        self._allowed_regions = allowed_regions or frozenset(config.allowed_regions)
        self._icaos, self._icao_weights = self._filter_to_regions(
            *self._weighter.icaos_and_weights()
        )
        if not self._icaos:
            raise ValueError(
                f"No airports left after filtering to regions={sorted(self._allowed_regions)}."
            )

        aircraft = _load_aircraft()
        self._aircraft_by_class: dict[OperatorClass, list[dict]] = {}
        for ac in aircraft:
            for oc in ac["operator_classes"]:
                self._aircraft_by_class.setdefault(oc, []).append(ac)

        ew = event_weights or config.event_weights
        self._events = list(ew.keys())
        self._event_weights = list(ew.values())

        ow = operator_weights or config.operator_weights
        # Drop classes with no aircraft to avoid sampling impossibilities.
        ow = {k: v for k, v in ow.items() if self._aircraft_by_class.get(k)}
        self._operators = list(ow.keys())
        self._operator_weights = list(ow.values())

        self._phases = list(config.phase_weights.keys())
        self._phase_weights = list(config.phase_weights.values())
        self._densities = list(config.traffic_density_weights.keys())
        self._density_weights = list(config.traffic_density_weights.values())
        self._tod = list(config.time_of_day_weights.keys())
        self._tod_weights = list(config.time_of_day_weights.values())
        self._rw_suffixes = list(config.runway_suffix_weights.keys())
        self._rw_suffix_weights = list(config.runway_suffix_weights.values())

    @property
    def weighter_name(self) -> str:
        return self._weighter.name

    @property
    def allowed_regions(self) -> frozenset[Region]:
        return self._allowed_regions

    @property
    def config(self) -> "RegionConfig":
        return self._cfg

    def _filter_to_regions(
        self, icaos: list[str], weights: list[float]
    ) -> tuple[list[str], list[float]]:
        out_i: list[str] = []
        out_w: list[float] = []
        for ic, w in zip(icaos, weights):
            if region_for_icao(ic) in self._allowed_regions:
                out_i.append(ic)
                out_w.append(w)
        return out_i, out_w

    def fast_forward(self, n: int) -> None:
        for _ in range(n):
            self.sample()

    def sample_batch(self, n: int) -> list[Scenario]:
        return [self.sample() for _ in range(n)]

    def iter(self, count: int) -> Iterator[Scenario]:
        for _ in range(count):
            yield self.sample()

    def sample(self) -> Scenario:
        rng = self._rng
        cfg = self._cfg
        icao = rng.choices(self._icaos, weights=self._icao_weights, k=1)[0]
        region = region_for_icao(icao)
        phase: Phase = rng.choices(self._phases, weights=self._phase_weights, k=1)[0]
        traffic_density: Density = rng.choices(
            self._densities, weights=self._density_weights, k=1
        )[0]
        n_lo, n_hi = cfg.traffic_aircraft_range[traffic_density]
        n_aircraft = rng.randint(n_lo, n_hi)
        aircraft_list = [self._sample_aircraft(rng) for _ in range(n_aircraft)]

        runway = self._sample_runway(rng)
        sid_star = self._sample_sid_star(rng, phase)
        squawk = self._sample_squawk(rng)
        freq_lo, freq_hi = cfg.phase_frequency_bands[phase]
        frequency_mhz = round(rng.uniform(freq_lo, freq_hi), 2)
        weather = self._sample_weather(rng)
        time_of_day: TimeOfDay = rng.choices(
            self._tod, weights=self._tod_weights, k=1
        )[0]
        event: Event = rng.choices(self._events, weights=self._event_weights, k=1)[0]
        return Scenario(
            icao=icao,
            region=region,
            phase=phase,
            aircraft=aircraft_list,
            runway=runway,
            sid_star=sid_star,
            squawk=squawk,
            frequency_mhz=frequency_mhz,
            weather=weather,
            time_of_day=time_of_day,
            traffic_density=traffic_density,
            event=event,
        )

    def _sample_aircraft(self, rng: random.Random) -> Aircraft:
        operator_class: OperatorClass = rng.choices(
            self._operators, weights=self._operator_weights, k=1
        )[0]
        aircraft = rng.choice(self._aircraft_by_class[operator_class])
        callsign = self._sample_callsign(rng, operator_class)
        return Aircraft(
            callsign=callsign,
            aircraft_type=aircraft["icao_type"],
            wake=aircraft["wake"],
            operator_class=operator_class,
        )

    def _sample_callsign(
        self, rng: random.Random, operator_class: OperatorClass
    ) -> str:
        prefixes = self._cfg.operator_prefixes.get(operator_class, [])
        if not prefixes:
            # GA-style US N-number.
            digits = rng.randint(1, 5)
            n = "".join(str(rng.randint(0, 9)) for _ in range(digits))
            return f"N{n}"
        prefix = rng.choice(prefixes)
        suffix = str(rng.randint(1, 9999))
        return f"{prefix}{suffix}"

    def _sample_runway(self, rng: random.Random) -> str:
        heading = rng.randint(1, 36)
        side = rng.choices(self._rw_suffixes, weights=self._rw_suffix_weights, k=1)[0]
        return f"{heading:02d}{side}"

    def _sample_sid_star(self, rng: random.Random, phase: Phase) -> str | None:
        if phase in ("ground", "ramp"):
            return None
        if rng.random() < self._cfg.sid_star_none_chance:
            return None
        letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4))
        num = rng.randint(1, 9)
        return f"{letters}{num}"

    @staticmethod
    def _sample_squawk(rng: random.Random) -> str:
        while True:
            digits = "".join(str(rng.randint(0, 7)) for _ in range(4))
            if digits not in _RESERVED_SQUAWKS:
                return digits

    def _sample_weather(self, rng: random.Random) -> Weather:
        w = self._cfg.weather
        vmc_imc_keys = list(w.vmc_imc_weights.keys())
        vmc_imc_w = list(w.vmc_imc_weights.values())
        vmc_imc: VmcImc = rng.choices(vmc_imc_keys, weights=vmc_imc_w, k=1)[0]
        if vmc_imc == "VMC":
            vis_sm = round(rng.uniform(*w.vmc_vis_sm), 1)
            ceiling_ft: int | None = (
                rng.randint(*w.vmc_ceiling_ft) if rng.random() < w.vmc_ceiling_chance else None
            )
        else:
            vis_sm = round(rng.uniform(*w.imc_vis_sm), 2)
            ceiling_ft = rng.randint(*w.imc_ceiling_ft)
        bucket_ranges = [b.range for b in w.wind_kt_buckets]
        bucket_weights = [b.weight for b in w.wind_kt_buckets]
        lo, hi = rng.choices(bucket_ranges, weights=bucket_weights, k=1)[0]
        wind_kt = rng.randint(lo, hi)
        return Weather(
            wind_dir_deg=rng.randint(*w.wind_dir_deg),
            wind_kt=wind_kt,
            vis_sm=vis_sm,
            ceiling_ft=ceiling_ft,
            altimeter_inhg=round(rng.uniform(*w.altimeter_inhg), 2),
            vmc_imc=vmc_imc,
        )


def iter_scenarios(
    count: int,
    seed: int,
    config: "RegionConfig | None" = None,
    weighter: AirportWeighter | None = None,
    **kwargs,
) -> Iterable[Scenario]:
    sampler = ScenarioSampler(seed=seed, config=config, weighter=weighter, **kwargs)
    yield from sampler.iter(count)
