from __future__ import annotations

import hashlib

import pyarrow as pa
from pydantic import BaseModel

from .._pa import now_iso, pydantic_to_pa_schema


class VoiceRecord(BaseModel):
    voice_id: str
    source: str
    source_speaker_id: str
    source_clip_id: str
    duration_s: float
    sample_rate: int
    license: str
    attribution: str
    selected_at: str
    sub_pool: str | None = None
    gender_hint: str | None = None
    language: str = "en"


_META_SCHEMA = pydantic_to_pa_schema(VoiceRecord)
VOICE_SCHEMA = pa.schema(list(_META_SCHEMA) + [pa.field("audio", pa.binary())])


def voice_id_for(*, source: str, source_speaker_id: str) -> str:
    """Deterministic 12-hex-char voice identifier.

    Stable for a given (source, source_speaker_id). Independent of which clip
    was selected for the speaker, so re-running with different filter params
    that pick a different clip leaves voice_id unchanged.
    """
    payload = f"{source}:{source_speaker_id}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]
