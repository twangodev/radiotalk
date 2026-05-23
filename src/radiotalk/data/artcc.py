"""Airport → ARTCC (Air Route Traffic Control Center) lookup.

v5 review found the model invents Center names like "Pleasanton Center" or
"Iowa County Center" when asked to pick from a list. v6 strategy: inject the
ARTCC name directly into the prompt briefing so the model never guesses.

State defaults catch ~95% of US airports. Per-airport overrides handle
airports whose ARTCC boundary crosses state lines (TX is split four ways;
California is split Oakland/Los Angeles; etc.).
"""
from __future__ import annotations

import airportsdata

_AIRPORTS_CACHE: dict | None = None


def _airports() -> dict:
    global _AIRPORTS_CACHE
    if _AIRPORTS_CACHE is None:
        _AIRPORTS_CACHE = airportsdata.load("ICAO")
    return _AIRPORTS_CACHE

_STATE_TO_ARTCC: dict[str, str] = {
    "Alabama": "Atlanta",
    "Alaska": "Anchorage",
    "Arizona": "Albuquerque",
    "Arkansas": "Memphis",
    "California": "Los Angeles",
    "Colorado": "Denver",
    "Connecticut": "Boston",
    "Delaware": "Washington",
    "District of Columbia": "Washington",
    "Florida": "Jacksonville",
    "Georgia": "Atlanta",
    "Hawaii": "Honolulu",
    "Idaho": "Salt Lake",
    "Illinois": "Chicago",
    "Indiana": "Chicago",
    "Iowa": "Minneapolis",
    "Kansas": "Kansas City",
    "Kentucky": "Indianapolis",
    "Louisiana": "Houston",
    "Maine": "Boston",
    "Maryland": "Washington",
    "Massachusetts": "Boston",
    "Michigan": "Cleveland",
    "Minnesota": "Minneapolis",
    "Mississippi": "Memphis",
    "Missouri": "Kansas City",
    "Montana": "Salt Lake",
    "Nebraska": "Minneapolis",
    "Nevada": "Los Angeles",
    "New Hampshire": "Boston",
    "New Jersey": "New York",
    "New Mexico": "Albuquerque",
    "New York": "New York",
    "North Carolina": "Washington",
    "North Dakota": "Minneapolis",
    "Ohio": "Cleveland",
    "Oklahoma": "Fort Worth",
    "Oregon": "Seattle",
    "Pennsylvania": "Cleveland",
    "Rhode Island": "Boston",
    "South Carolina": "Atlanta",
    "South Dakota": "Minneapolis",
    "Tennessee": "Memphis",
    "Texas": "Fort Worth",
    "Utah": "Salt Lake",
    "Vermont": "Boston",
    "Virginia": "Washington",
    "Washington": "Seattle",
    "West Virginia": "Washington",
    "Wisconsin": "Minneapolis",
    "Wyoming": "Denver",
    "Puerto Rico": "San Juan",
    "U.S. Virgin Islands": "San Juan",
    "Guam": "Honolulu",
    "Northern Mariana Islands": "Honolulu",
    "American Samoa": "Honolulu",
}

_AIRPORT_OVERRIDES: dict[str, str] = {
    # Northern California / northern Nevada → Oakland (ZOA)
    "KSFO": "Oakland", "KOAK": "Oakland", "KSJC": "Oakland",
    "KSMF": "Oakland", "KSCK": "Oakland", "KFAT": "Oakland",
    "KMRY": "Oakland", "KMOD": "Oakland", "KAPC": "Oakland",
    "KSTS": "Oakland", "KCCR": "Oakland", "KHWD": "Oakland",
    "KPAO": "Oakland", "KSQL": "Oakland", "KSNS": "Oakland",
    "KSAC": "Oakland", "KMHR": "Oakland", "KLVK": "Oakland",
    "KNUQ": "Oakland", "KOAR": "Oakland", "KMCC": "Oakland",
    "KRNO": "Oakland", "KCXP": "Oakland", "KMEV": "Oakland",
    # Indiana — Indianapolis ARTCC covers central/south Indiana
    "KIND": "Indianapolis", "KEVV": "Indianapolis", "KBMG": "Indianapolis",
    "KHUF": "Indianapolis", "KLAF": "Indianapolis",
    # Kentucky — covered by Indianapolis
    "KSDF": "Indianapolis", "KLEX": "Indianapolis", "KCVG": "Indianapolis",
    # Florida south → Miami (ZMA)
    "KMIA": "Miami", "KFLL": "Miami", "KPBI": "Miami",
    "KFXE": "Miami", "KOPF": "Miami", "KTMB": "Miami",
    "KAPF": "Miami", "KRSW": "Miami", "KEYW": "Miami",
    "KMTH": "Miami", "KHWO": "Miami", "KBCT": "Miami",
    # Florida panhandle → handled by Jacksonville default
    # Houston ARTCC (ZHU)
    "KIAH": "Houston", "KHOU": "Houston", "KSAT": "Houston",
    "KAUS": "Houston", "KCRP": "Houston", "KEFD": "Houston",
    "KSGR": "Houston", "KSSF": "Houston", "KHRL": "Houston",
    "KBRO": "Houston", "KMFE": "Houston", "KLRD": "Houston",
    # Louisiana under Houston ARTCC already by state default
    # Memphis ARTCC airports in MS/TN/AR — state defaults cover
    "KBNA": "Memphis", "KMEM": "Memphis",
    "KTYS": "Atlanta",  # eastern TN
    "KCHA": "Atlanta",
    # Western NY / western PA → Cleveland
    "KBUF": "Cleveland", "KROC": "Cleveland", "KSYR": "Cleveland",
    "KPIT": "Cleveland", "KERI": "Cleveland", "KIAG": "Cleveland",
    "KELM": "Cleveland",
    # Eastern PA / DE / southern NJ → New York
    "KPHL": "New York", "KABE": "New York", "KAVP": "New York",
    "KMDT": "New York", "KLNS": "New York", "KILG": "New York",
    "KACY": "New York", "KTTN": "New York",
    # Western TX → Albuquerque
    "KELP": "Albuquerque", "KAMA": "Albuquerque",
    "KLBB": "Albuquerque", "KROW": "Albuquerque", "KMAF": "Fort Worth",
    "KSAF": "Albuquerque",
    # Arizona — KPHX/KTUS in Albuquerque ARTCC (ZAB)
    "KPHX": "Albuquerque", "KTUS": "Albuquerque", "KIWA": "Albuquerque",
    "KDVT": "Albuquerque", "KSDL": "Albuquerque", "KAVQ": "Albuquerque",
    "KFFZ": "Albuquerque", "KCHD": "Albuquerque",
    # Las Vegas → LA
    "KLAS": "Los Angeles", "KHND": "Los Angeles", "KVGT": "Los Angeles",
    "KSAN": "Los Angeles",
    # KMEM and KMCO area
    "KMCO": "Jacksonville", "KTPA": "Jacksonville", "KJAX": "Jacksonville",
    "KSFB": "Jacksonville", "KISM": "Jacksonville", "KORL": "Jacksonville",
    "KDAB": "Jacksonville",
    # SC coast
    "KCHS": "Jacksonville", "KMYR": "Atlanta",
    # NC south/central → Atlanta; coastal NC → Washington
    "KCLT": "Atlanta", "KGSP": "Atlanta", "KAVL": "Atlanta",
    "KGSO": "Washington", "KRDU": "Washington", "KORF": "Washington",
    # NY airports
    "KJFK": "New York", "KLGA": "New York", "KEWR": "New York",
    "KISP": "New York", "KHPN": "New York", "KSWF": "New York",
    "KFRG": "New York", "KFOK": "New York", "KMSS": "Boston",
    "KALB": "Boston",
    # Connecticut → New York / Boston split
    "KHFD": "New York", "KBDR": "New York", "KBDL": "Boston",
    # Northern MI → Cleveland default; central MI → Cleveland
    "KDTW": "Cleveland", "KFNT": "Cleveland", "KGRR": "Chicago",
    "KAZO": "Chicago", "KLAN": "Cleveland",
    # WI east → Chicago
    "KMKE": "Chicago", "KMSN": "Chicago", "KGRB": "Minneapolis",
    "KOSH": "Minneapolis",
    # MO south Memphis-adjacent
    "KSGF": "Kansas City", "KSTL": "Kansas City", "KMCI": "Kansas City",
    # IA → split; KDSM → KC
    "KDSM": "Kansas City",
    # IL south → KC
    "KBLV": "Kansas City",
    "KSPI": "Kansas City",
    # Western NE → Denver
    "KCDR": "Denver", "KBFF": "Denver", "KSNY": "Denver", "KLBF": "Denver",
    # Southern AL panhandle → handled by Atlanta default
    # KLEX, KSDF already overridden to Indianapolis
    # KSPI is closer to KC ARTCC boundary; KPBF Pine Bluff AR → Memphis default fine
    # PR/USVI → San Juan CERAP (handled by state default)
    # KAAT (Buchanan, MI) → Cleveland default
    # KCPF (Madisonville TN per LID code) — likely Atlanta default fine
    # KOMH (Orange VA) → Washington default fine
    # KAVK (Alva OK) → Fort Worth default fine
    # KLPC (Lompoc CA) → Los Angeles default fine
    # OK → Fort Worth default
    "KOKC": "Fort Worth", "KTUL": "Fort Worth",
    # KS → KC default
    "KICT": "Kansas City",
    # ID → Salt Lake
    "KBOI": "Salt Lake",
    # Eastern WA → Salt Lake (eastern WA boundary)
    "KGEG": "Salt Lake", "KSFF": "Salt Lake",
    # Montana split
    "KBIL": "Salt Lake", "KGTF": "Salt Lake", "KMSO": "Salt Lake",
    "KBZN": "Salt Lake",
    # Northeast PA
    "KAOO": "Cleveland",
    # WV south
    "KCRW": "Indianapolis", "KHTS": "Indianapolis",
    # MA/RI/NH already covered by Boston
    # VT/ME → Boston
    "KBTV": "Boston", "KPWM": "Boston", "KBGR": "Boston",
}


def artcc_for_icao(icao: str) -> str | None:
    """Return the spoken ARTCC name for a US airport, or None for non-US."""
    if icao in _AIRPORT_OVERRIDES:
        return _AIRPORT_OVERRIDES[icao]
    info = _airports().get(icao)
    if info is None:
        return None
    state = info.get("subd", "")
    return _STATE_TO_ARTCC.get(state)
