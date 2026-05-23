"""Real-runway lookup for the 86 tier 1+2 US airports radiotalk samples from.

v6.4 fix for the "runway 35 at KLAX" problem: in v6.3, the scenario sampler
picked random runway headings (01-36) independent of the airport's actual
runway set. Reviewers flagged ~84% of transcripts as containing spoken
clearances on runways the named airport doesn't have.

This module ships a hand-curated lookup, generated from OurAirports
runways.csv filtered to the tier 1+2 ICAO set in config/us.yaml. The
sampler's _sample_runway() uses runways_for(icao) to draw from actual
runway designators (e.g. KLAX → 06L, 06R, 07L, 07R, 24L, 24R, 25L, 25R).

If an airport is not in this table (e.g. when the config adds more tier
entries), the sampler falls back to the previous random 01-36 generator.

v6.6 regen: rebuilt from OurAirports CSV. Earlier v6.4 table had manual
typing errors at ~15% of airports (KTUL had 01L/19L that don't exist;
PAFA had pre-2009 renumbered designators; KLAS had nonexistent 08C/26C
runways; etc).
"""
from __future__ import annotations


_AIRPORT_RUNWAYS: dict[str, tuple[str, ...]] = {
    'KABQ': ('03', '08', '12', '21', '26', '30'),
    'KALB': ('01', '10', '19', '28'),
    'KAMA': ('04', '13', '22', '31'),
    'KATL': ('08L', '08R', '09L', '09R', '10', '26L', '26R', '27L', '27R', '28'),
    'KAUS': ('18L', '18R', '36L', '36R'),
    'KBDL': ('06', '15', '24', '33'),
    'KBHM': ('06', '18', '24', '36'),
    'KBIL': ('07', '10L', '10R', '25', '28L', '28R'),
    'KBNA': ('02C', '02L', '02R', '13', '20C', '20L', '20R', '31'),
    'KBOI': ('10L', '10R', '28L', '28R'),
    'KBOS': ('04L', '04R', '09', '14', '15L', '15R', '22L', '22R', '27', '32', '33L', '33R'),
    'KBUF': ('05', '14', '23', '32'),
    'KBUR': ('08', '15', '26', '33'),
    'KBWI': ('10', '15L', '15R', '28', '33L', '33R'),
    'KCHS': ('03', '15', '21', '33'),
    'KCLE': ('06L', '06R', '10', '24L', '24R', '28'),
    'KCLT': ('18C', '18L', '18R', '36C', '36L', '36R'),
    'KCMH': ('10L', '10R', '28L', '28R'),
    'KCOS': ('12', '17L', '17R', '30', '35L', '35R'),
    'KCVG': ('09', '18C', '18L', '18R', '27', '36C', '36L', '36R'),
    'KDCA': ('01', '04', '15', '19', '22', '33'),
    'KDEN': ('07', '08', '16L', '16R', '17L', '17R', '25', '26', '34L', '34R', '35L', '35R'),
    'KDFW': ('13L', '13R', '17C', '17L', '17R', '18L', '18R', '31L', '31R', '35C', '35L', '35R', '36L', '36R'),
    'KDTW': ('03L', '03R', '04L', '04R', '09L', '09R', '21L', '21R', '22L', '22R', '27L', '27R'),
    'KELP': ('04', '08L', '08R', '22', '26L', '26R'),
    'KEWR': ('04L', '04R', '11', '22L', '22R', '29'),
    'KFAT': ('11L', '11R', '29L', '29R'),
    'KFLL': ('10L', '10R', '28L', '28R'),
    'KGSP': ('04', '22'),
    'KHSV': ('18L', '18R', '36L', '36R'),
    'KIAD': ('01C', '01L', '01R', '12', '19C', '19L', '19R', '30'),
    'KIAH': ('08L', '08R', '09', '15L', '15R', '26L', '26R', '27', '33L', '33R'),
    'KIND': ('05L', '05R', '14', '23L', '23R', '32'),
    'KJAX': ('08', '14', '26', '32'),
    'KJFK': ('04L', '04R', '13L', '13R', '22L', '22R', '31L', '31R'),
    'KLAS': ('01L', '01R', '08L', '08R', '19L', '19R', '26L', '26R'),
    'KLAX': ('06L', '06R', '07L', '07R', '24L', '24R', '25L', '25R'),
    'KLBB': ('08', '17L', '17R', '26', '35L', '35R'),
    'KLEX': ('04', '09', '22', '27'),
    'KLGA': ('04', '13', '22', '31'),
    'KLGB': ('08L', '08R', '12', '26L', '26R', '30'),
    'KMCI': ('01L', '01R', '09', '19L', '19R', '27'),
    'KMCO': ('17L', '17R', '18L', '18R', '35L', '35R', '36L', '36R'),
    # KMDW: 04L/22R was decommissioned in 2018. Active runways per current
    # FAA charts: 04R/22L, 13C/31C, 13L/31R, 13R/31L. OurAirports still
    # lists the decommissioned pair.
    'KMDW': ('04R', '13C', '13L', '13R', '22L', '31C', '31L', '31R'),
    'KMEM': ('09', '18C', '18L', '18R', '27', '36C', '36L', '36R'),
    'KMHT': ('06', '17', '24', '35'),
    'KMIA': ('08L', '08R', '09', '12', '26L', '26R', '27', '30'),
    'KMKE': ('01L', '01R', '07L', '07R', '13', '19L', '19R', '25L', '25R', '31'),
    'KMSP': ('04', '12L', '12R', '17', '22', '30L', '30R', '35'),
    'KMSY': ('02', '11', '20', '29'),
    'KMYR': ('18', '36'),
    'KOAK': ('10L', '10R', '12', '15', '28L', '28R', '30', '33'),
    'KOKC': ('13', '17L', '17R', '18', '31', '35L', '35R', '36'),
    'KOMA': ('14L', '14R', '18', '32L', '32R', '36'),
    'KORD': ('04L', '04R', '09C', '09L', '09R', '10C', '10L', '10R', '22L', '22R', '27C', '27L', '27R', '28C', '28L', '28R'),
    'KPBI': ('10L', '10R', '14', '28L', '28R', '32'),
    'KPDX': ('03', '10L', '10R', '21', '28L', '28R'),
    'KPHL': ('08', '09L', '09R', '17', '26', '27L', '27R', '35'),
    'KPHX': ('07L', '07R', '08', '25L', '25R', '26'),
    'KPIT': ('10C', '10L', '10R', '14', '28C', '28L', '28R', '32'),
    'KPVD': ('05', '16', '23', '34'),
    'KPWM': ('11', '18', '29', '36'),
    'KRDU': ('05L', '05R', '14', '23L', '23R', '32'),
    'KRIC': ('02', '16', '20', '34'),
    'KRNO': ('08', '17L', '17R', '26', '35L', '35R'),
    'KROC': ('04', '07', '10', '22', '25', '28'),
    'KRSW': ('06', '24'),
    'KSAN': ('09', '27'),
    'KSAV': ('01', '10', '19', '28'),
    'KSBA': ('07', '15L', '15R', '25', '33L', '33R'),
    'KSDF': ('11', '17L', '17R', '29', '35L', '35R'),
    'KSEA': ('16C', '16L', '16R', '34C', '34L', '34R'),
    'KSFO': ('01L', '01R', '10L', '10R', '19L', '19R', '28L', '28R'),
    'KSHV': ('06', '14', '24', '32'),
    'KSJC': ('12L', '12R', '30L', '30R'),
    'KSLC': ('14', '16L', '16R', '17', '32', '34L', '34R', '35'),
    'KSMF': ('17L', '17R', '35L', '35R'),
    'KSNA': ('02L', '02R', '20L', '20R'),
    'KSTL': ('06', '11', '12L', '12R', '24', '29', '30L', '30R'),
    'KSYR': ('10', '15', '28', '33'),
    'KTPA': ('01L', '01R', '10', '19L', '19R', '28'),
    'KTUL': ('08', '18L', '18R', '26', '36L', '36R'),
    # KTUS: OurAirports CSV mislabeled designations (04/22 + 12/30 don't
    # exist). Real KTUS runways per current FAA charts: 03/21, 11L/29R,
    # 11R/29L. (Likely a mag-vs-true heading conversion error in source.)
    'KTUS': ('03', '11L', '11R', '21', '29L', '29R'),
    'PAFA': ('02', '02L', '02R', '20', '20L', '20R'),
    'PANC': ('07L', '07R', '15', '25L', '25R', '33'),
    'PHNL': ('04L', '04R', '08L', '08R', '22L', '22R', '26L', '26R'),
}


def runways_for(icao: str) -> tuple[str, ...] | None:
    """Return the tuple of real runway designators for an airport, or None
    if the airport isn't in the tier 1+2 lookup. Designators include the
    side letter (L/R/C) when one applies — e.g. ('06L', '06R', '07L', ...).
    """
    return _AIRPORT_RUNWAYS.get(icao.upper())
