from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .scenario import Scenario

Decoding = Literal["json_schema", "free"]


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    speaker: Literal["ATC", "PILOT"]
    callsign: str = Field(min_length=1, max_length=32)
    facility: str | None = None
    text: str = Field(min_length=1, max_length=2000)
    intent: str = Field(min_length=1, max_length=64)


class ModelTranscript(BaseModel):
    """The JSON shape the LLM is asked to emit. Subset of Transcript.

    The min/max bounds on ``turns`` are *enforced by the SGLang server* when
    the run uses ``--decoding json_schema`` (the default). Under ``--decoding
    free`` the schema is only validated post-hoc.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    turns: list[Turn] = Field(min_length=4, max_length=60)


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)
    scenario_id: str
    scenario: Scenario
    turns: list[Turn]
    model: str
    generated_at: datetime
    prompt_version: str
    taxonomy_version: str
    decoding: Decoding


def model_transcript_json_schema() -> dict:
    """Schema shipped to the LLM via guided_json / response_format."""
    return ModelTranscript.model_json_schema()
