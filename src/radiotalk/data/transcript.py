from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from .scenario import Scenario


class Turn(BaseModel):
    """One utterance parsed from a plaintext transcript line `SPEAKER: text`."""

    model_config = ConfigDict(frozen=True)
    speaker: str
    text: str


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)
    scenario_id: str
    scenario: Scenario
    turns: list[Turn]
    model: str
    generated_at: datetime
    prompt_version: str
    taxonomy_version: str


# Speaker tag: uppercase alnum + `_ - /` (e.g. ATC, KSFO_TWR, UAL1080, D-AIBC, N12345).
# Anchored to line start; colon terminates the tag.
_LINE_RE = re.compile(r"^\s*([A-Z0-9][A-Z0-9 _\-/]{0,31}?)\s*:\s*(.+?)\s*$")


class TranscriptParseError(ValueError):
    """Raised when a raw plaintext transcript cannot be parsed into turns."""


def parse_turns(raw: str) -> list[Turn]:
    """Parse `SPEAKER: utterance` lines into Turns. Skips blank/unparseable lines."""
    turns: list[Turn] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        turns.append(Turn(speaker=m.group(1).strip(), text=m.group(2).strip()))
    return turns


MIN_TURNS = 4


# ---------------------------------------------------------------------------
# v6 realism validators — deterministic post-gen filters for the failure
# patterns that the prompt could not reliably suppress in v5/v6 review. When
# any pattern matches, the generator retries (max_parse_retries=5), so these
# turn a 5-20%-per-attempt failure rate into a 0.01-0.1% shipped rate.
# ---------------------------------------------------------------------------

_FORBIDDEN_PHRASES = [
    (re.compile(r"\bfrequency check\b", re.IGNORECASE), "frequency check"),
    (re.compile(r"\bgood night\b", re.IGNORECASE), "good night"),
    (re.compile(r"\bhave a nice (flight|landing|day)\b", re.IGNORECASE),
     "have a nice"),
    (re.compile(r"\bhave a (smooth flight|great day)\b", re.IGNORECASE),
     "have a smooth/great"),
    (re.compile(r"\bposition and hold\b", re.IGNORECASE),
     "position and hold (pre-2010)"),
    (re.compile(r"\bQNH\b"), "QNH (non-US)"),
]

_POINT_FREQ_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|niner)\s+point\s+"
    r"(zero|one|two|three|four|five|six|seven|eight|nine|niner)\b",
    re.IGNORECASE,
)

_RUNWAY_DIGITS_RE = re.compile(r"\brunways?\s+\d", re.IGNORECASE)

# Compound number words used as runway designations are wrong; ATC uses
# digit-by-digit only ("runway two two", never "runway twenty two").
_RUNWAY_WORD_RE = re.compile(
    r"\brunways?\s+("
    r"twenty|thirty|forty|fifty|"
    r"eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|"
    r"ten"
    r")(\s+|-|$|\.|,)",
    re.IGNORECASE,
)

# ICAO airport code (Kxxx/Pxxx) immediately followed by a facility role —
# e.g. "KSFO Tower", "PHNL Approach". The speaker tag may contain ICAO, but
# the spoken text must use the airport's English name.
_ICAO_LEAK_RE = re.compile(
    r"\b[KP][A-Z]{3}\s+(Tower|Ground|Approach|Center|Centre|Departure|Ramp|Clearance)\b"
)

# "medium" / "light" as a wake suffix — only "heavy" and "super" are spoken.
# Pattern: digit-word followed by "medium" or "light".
_WAKE_SUFFIX_RE = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|niner)\s+"
    r"(medium|light)\b",
    re.IGNORECASE,
)

# Hijack / lost-comm / emergency squawk codes — the sampler excludes these,
# so the model must never assign them in transcripts.
_HIJACK_SQUAWK_RE = re.compile(
    r"\bsquawk\s+seven\s+(five|six|seven)\s+zero\s+zero\b",
    re.IGNORECASE,
)

# Stage directions / editorial bracketed content.
_STAGE_DIR_RE = re.compile(
    r"\[(no response|pauses?|sic|silent|inaudible|garbled|static|click|"
    r"radio static|long pause|transmission cut|crosstalk)\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# v6.2 role-discipline validators — all 4 v6.1 reviewers independently called
# out the same remaining failures: Ramp issuing taxi-to-runway / LUAW / squawk;
# Ground issuing LUAW or takeoff; Approach issuing landing clearance; Center
# scenarios with no controller voice at all; ARTCC name drift past header.
# Regex-matchable, so they belong in Layer 1 not the prompt.
# ---------------------------------------------------------------------------

# Speaker suffix → facility role. The validator inspects each turn's speaker
# tag suffix and checks the utterance against role-specific forbidden phrases.
_FACILITY_SUFFIXES = frozenset({"TWR", "GND", "APP", "DEP", "CTR", "RAMP", "CLR"})

_SPEAKER_SUFFIX_RE = re.compile(r"_(TWR|GND|APP|DEP|CTR|RAMP|CLR)$")

_TAKEOFF_OR_LAND_RE = re.compile(
    r"\b(cleared (for |to )?(an )?(immediate )?(takeoff|land(ing)?)|"
    r"cleared (immediate )?(takeoff|land(ing)?)|"
    r"line up and wait|"
    r"cleared to enter (the )?runway)\b",     # "cleared to enter runway"
    re.IGNORECASE,                             # is LUAW-equivalent
)
# v6.3: Tower-specific extras — Tower never issues "cleared to taxi" (that
# prefix is Ground; Tower says just "taxi" if anything) or "ILS X approved"
# (ILS approach clearance is Approach's job).
_TWR_EXTRA_FORBIDDEN_RE = re.compile(
    r"\bcleared to taxi\b|"
    r"\bI\s?L\s?S\b[^.]{0,60}\bapproved\b",
    re.IGNORECASE,
)
# Catches both "taxi to runway X" and "taxi to <spoken-digit>" (model often
# drops the literal word "runway" — "taxi to two two right via Alpha").
_TAXI_TO_RUNWAY_RE = re.compile(
    r"\btaxi(?:ing)? to (?:runway |"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)\b)",
    re.IGNORECASE,
)
_PUSHBACK_RE = re.compile(r"\bpushback\b", re.IGNORECASE)
_SQUAWK_ASSIGN_RE = re.compile(
    r"\bsquawk\s+(zero|one|two|three|four|five|six|seven|eight|nine|niner)",
    re.IGNORECASE,
)
_ALTIMETER_ASSIGN_RE = re.compile(
    r"\baltimeter\s+(two|three)\s+(zero|one|two|three|four|five|six|seven|eight|nine|niner)",
    re.IGNORECASE,
)
_FLY_HEADING_RE = re.compile(r"\bfly heading\b", re.IGNORECASE)


# Per-role forbidden actions. Each entry is (suffix, pattern, label) — if the
# turn's speaker matches the suffix AND the utterance matches the pattern, the
# transcript is rejected. Squawk + altimeter from Ramp are stylistic
# overreaches but not safety-critical, so the validator allows them — only
# rejects role violations that would teach an ASR model wrong speaker priors
# for takeoff/landing/taxi clearance types.
_ROLE_FORBIDDEN: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # RAMP handles pushback and gate movement only.
    ("RAMP", _TAKEOFF_OR_LAND_RE, "Ramp issuing takeoff/landing/LUAW"),
    ("RAMP", _TAXI_TO_RUNWAY_RE, "Ramp issuing taxi-to-runway"),
    ("RAMP", _FLY_HEADING_RE, "Ramp issuing heading"),
    # GROUND handles taxi/squawk/altimeter, never runway-entry or air work.
    ("GND", _TAKEOFF_OR_LAND_RE, "Ground issuing takeoff/landing/LUAW"),
    ("GND", _FLY_HEADING_RE, "Ground issuing heading"),
    # APPROACH handles vectors/altitudes/ILS clearance, not surface or landing.
    ("APP", _TAKEOFF_OR_LAND_RE, "Approach issuing takeoff/landing/LUAW"),
    ("APP", _TAXI_TO_RUNWAY_RE, "Approach issuing taxi-to-runway"),
    ("APP", _PUSHBACK_RE, "Approach issuing pushback"),
    # CENTER handles en-route vectors/altitudes, never surface or landing.
    ("CTR", _TAKEOFF_OR_LAND_RE, "Center issuing takeoff/landing/LUAW"),
    ("CTR", _TAXI_TO_RUNWAY_RE, "Center issuing taxi-to-runway"),
    ("CTR", _PUSHBACK_RE, "Center issuing pushback"),
    # TOWER owns runway entry/exit; full surface taxi is Ground's job.
    ("TWR", _TAXI_TO_RUNWAY_RE, "Tower issuing taxi-to-runway"),
    ("TWR", _TWR_EXTRA_FORBIDDEN_RE,
     "Tower issuing 'cleared to taxi' / 'ILS approved'"),
)


# US ARTCC name allowlist (spoken form, sans "Center"). When a _CTR speaker
# emits "<X> Center" in their utterance, X must be one of these.
_VALID_ARTCC_NAMES = frozenset({
    "Albuquerque", "Anchorage", "Atlanta", "Boston", "Chicago",
    "Cleveland", "Denver", "Fort Worth", "Houston", "Indianapolis",
    "Jacksonville", "Kansas City", "Los Angeles", "Memphis", "Miami",
    "Minneapolis", "New York", "Oakland", "Salt Lake", "Seattle",
    "Washington", "Honolulu", "San Juan",
})
_ARTCC_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)\s+Center\b"
)


def validate_turns(turns: list[Turn], scenario: Scenario) -> None:
    """Raise TranscriptParseError if the parsed turns don't meet minimum bar."""
    if len(turns) < MIN_TURNS:
        raise TranscriptParseError(
            f"only {len(turns)} turns parsed (min {MIN_TURNS})"
        )
    focal = scenario.aircraft[0].callsign.upper()
    speakers = {t.speaker.upper() for t in turns}
    if focal not in speakers:
        raise TranscriptParseError(
            f"focal callsign {focal!r} not found among speakers {sorted(speakers)}"
        )
    _validate_realism(turns, scenario)


def _validate_realism(turns: list[Turn], scenario: Scenario) -> None:
    """Reject transcripts containing v5/v6 reviewer-identified failure patterns.

    These rules are applied across the joined transcript text. The generator
    retries on TranscriptParseError, so a hit here forces resampling rather
    than shipping a known-bad transcript.
    """
    text = " ".join(t.text for t in turns)

    for pat, label in _FORBIDDEN_PHRASES:
        if m := pat.search(text):
            raise TranscriptParseError(f"forbidden phrase {label!r}: {m.group()!r}")

    if m := _POINT_FREQ_RE.search(text):
        raise TranscriptParseError(f"'point' in frequency: {m.group()!r}")

    if m := _RUNWAY_DIGITS_RE.search(text):
        raise TranscriptParseError(f"runway as Arabic digits: {m.group()!r}")

    if m := _RUNWAY_WORD_RE.search(text):
        raise TranscriptParseError(f"runway as compound number: {m.group()!r}")

    if m := _ICAO_LEAK_RE.search(text):
        raise TranscriptParseError(f"ICAO airport code in facility name: {m.group()!r}")

    if m := _WAKE_SUFFIX_RE.search(text):
        raise TranscriptParseError(f"medium/light wake suffix: {m.group()!r}")

    if m := _HIJACK_SQUAWK_RE.search(text):
        raise TranscriptParseError(f"hijack/reserved squawk assigned: {m.group()!r}")

    if m := _STAGE_DIR_RE.search(text):
        raise TranscriptParseError(f"stage direction in transcript: {m.group()!r}")

    _validate_role_discipline(turns)
    _validate_controller_present(turns)
    _validate_artcc_names(turns)
    _validate_runway_realism(turns, scenario)
    _validate_frequency_band(turns)


def _facility_suffix(speaker: str) -> str | None:
    m = _SPEAKER_SUFFIX_RE.search(speaker.upper())
    return m.group(1) if m else None


def _validate_role_discipline(turns: list[Turn]) -> None:
    """Reject when a facility speaker emits a clearance type outside its role.

    Tower-only: takeoff, landing, LUAW. Ground-only: full taxi-to-runway,
    squawk, altimeter. Ramp-only: pushback. Approach/Center never issue
    surface or landing clearances.
    """
    for t in turns:
        suffix = _facility_suffix(t.speaker)
        if suffix is None:
            continue
        for forb_suffix, pat, label in _ROLE_FORBIDDEN:
            if suffix != forb_suffix:
                continue
            if m := pat.search(t.text):
                raise TranscriptParseError(
                    f"role violation: {label} (speaker {t.speaker!r}, "
                    f"text {m.group()!r})"
                )


def _validate_controller_present(turns: list[Turn]) -> None:
    """Reject transcripts with no controller voice at all.

    A pilot-only transcript teaches an ASR model wrong speaker priors. The
    reviewer-flagged failure mode: phase=center scenarios where the model
    drops the controller entirely and only emits pilot readbacks of
    instructions never issued.
    """
    for t in turns:
        if _facility_suffix(t.speaker) is not None:
            return
    raise TranscriptParseError(
        "no controller voice — every turn is a pilot speaker"
    )


# v6.5 — frequency band validator. US VHF aviation voice band is
# 118.000-136.975 MHz. The model sometimes invents handoff frequencies in
# the navaid band (108.000-117.975 MHz: VOR/ILS/LOC) — reviewer batch 3
# v6.4 caught 4 instances. The sampler always picks the focal frequency in
# the right band; only the model's spoken handoff frequencies leak.
_FREQ_SPOKEN_RE = re.compile(
    r"\b("
    r"one\s+zero\s+\w+|"          # 10X.XX (navaid)
    r"one\s+one\s+(\w+\s+){0,3}?decimal"  # 11X.XX
    r")",
    re.IGNORECASE,
)
# Match patterns like "one one seven decimal three" — first three digit-words
# form the integer part of the frequency.
_FREQ_PARSE_RE = re.compile(
    r"\b(one\s+(?:zero|one|two|three)\s+"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner))"
    r"\s+decimal\s+"
    r"((?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
    r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)){0,2})",
    re.IGNORECASE,
)
_DIGIT_WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "niner": 9,
}


_RWY_SPOKEN_RE = re.compile(
    r"\brunway\s+("
    r"(?:zero|one|two|three)\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
    r")"
    r"(?:\s+(left|right|center))?",
    re.IGNORECASE,
)
_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "niner": "9",
}
_SIDE_TO_LETTER = {"left": "L", "right": "R", "center": "C"}


def _validate_runway_realism(turns: list[Turn], scenario: Scenario) -> None:
    """Reject transcripts that reference runways not at the named airport.

    The v6.3 sampler restricted airports to tier 1+2 hubs, but the model
    still freelances on background-traffic runway mentions (~25% of v6.4
    transcripts referenced a runway the airport doesn't have). The v6.4
    briefing now lists the airport's actual runways; this validator enforces
    that no spoken "runway X Y" mention is off-airport.

    Spoken "runway zero six" matches if the airport has 06 in any form
    (06L/06R/06) — the side is sometimes elided in informal traffic calls.
    """
    from .runways import runways_for
    real = runways_for(scenario.icao)
    if not real:
        return
    real_set = set(real)
    # Build a set of numeric prefixes (without side) for partial matches.
    real_prefixes = {rw[:2] for rw in real}
    for t in turns:
        for m in _RWY_SPOKEN_RE.finditer(t.text):
            d1, d2 = m.group(1).split()
            num = _WORD_TO_DIGIT[d1.lower()] + _WORD_TO_DIGIT[d2.lower()]
            num = f"{int(num):02d}"
            side = _SIDE_TO_LETTER.get((m.group(2) or "").lower(), "")
            spoken = f"{num}{side}"
            # Accept: exact match, OR side-less mention when the numeric
            # prefix exists at the airport (some controllers omit L/R/C).
            if spoken in real_set:
                continue
            if not side and num in real_prefixes:
                continue
            raise TranscriptParseError(
                f"runway {spoken!r} not at {scenario.icao} (real: {sorted(real)})"
            )


def _validate_frequency_band(turns: list[Turn]) -> None:
    """Reject spoken frequencies outside the US VHF aviation voice band
    (118.000-136.975 MHz). Anything in 108.000-117.975 is navaid (VOR/ILS),
    not a comm frequency. Reviewer-caught failure mode in v6.4 batch 3.
    """
    for t in turns:
        for m in _FREQ_PARSE_RE.finditer(t.text):
            int_words = m.group(1).split()
            int_part = (
                _DIGIT_WORD_TO_INT[int_words[0].lower()] * 100 +
                _DIGIT_WORD_TO_INT[int_words[1].lower()] * 10 +
                _DIGIT_WORD_TO_INT[int_words[2].lower()]
            )
            if int_part < 118 or int_part > 136:
                raise TranscriptParseError(
                    f"frequency {int_part} MHz out of VHF voice band 118-136: "
                    f"{m.group()!r}"
                )


def _validate_artcc_names(turns: list[Turn]) -> None:
    """Reject when a _CTR speaker invents an ARTCC name not in the allowlist.

    The briefing already injects the correct ARTCC name; this catches the
    drift cases where the model uses a city/airport-derived name instead
    ("Pleasanton Center", "Scammon Bay Center", "Metro Field Center").
    """
    for t in turns:
        if _facility_suffix(t.speaker) != "CTR":
            continue
        for m in _ARTCC_NAME_RE.finditer(t.text):
            name = m.group(1)
            if name not in _VALID_ARTCC_NAMES:
                raise TranscriptParseError(
                    f"invented ARTCC name {name + ' Center'!r} from {t.speaker!r}"
                )
