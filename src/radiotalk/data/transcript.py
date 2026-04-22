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
    raw_text: str
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
