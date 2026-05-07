from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..data.scenario import Scenario


SAMPLE_RATE = 24000


class AudioBytes(BaseModel):
    model_config = ConfigDict(frozen=True)
    bytes: bytes
    path: str | None = None


class TokenSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str
    start_s: float
    end_s: float


class AudioTurnRow(BaseModel):
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
