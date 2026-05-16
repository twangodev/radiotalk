from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..audio.schema import AudioBytes, TokenSpan
from ..data.scenario import Scenario

SAMPLE_RATE = 8000


class RadioRow(BaseModel):
    """Channel-degraded variant of an AudioTurnRow.

    Mirrors the clean ``AudioTurnRow`` schema 1:1 (so any code that reads
    ``tada-clean`` can read this dataset unchanged), then appends radio-
    specific provenance.
    """
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    scenario: Scenario
    turn_idx: int
    speaker: str
    text: str
    text_normalized: str
    voice_id: str
    audio: AudioBytes
    tokens: list[TokenSpan] | None = None
    model: str
    prompt_version: str
    taxonomy_version: str
    tts_model: str

    clean_row_id: str
    variant_idx: int
    seed: int
    profile: str
    applied_effects: list[str]
    effective_snr_db: float
    pipeline_version: str
    pipeline_fingerprint: str
