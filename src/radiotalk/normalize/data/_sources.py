from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def airlines() -> dict[str, str]:
    raw = json.loads(
        files("radiotalk.normalize.data").joinpath("airlines.json").read_text()
    )
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def icao_airports() -> dict[str, dict]:
    import airportsdata
    return airportsdata.load("ICAO")


@lru_cache(maxsize=1)
def iata_airports() -> dict[str, dict]:
    import airportsdata
    return airportsdata.load("IATA")
