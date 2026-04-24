from __future__ import annotations

import hashlib
import json

import pyarrow as pa
from pydantic import BaseModel


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
    text: str | None = None


_AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

VOICE_COLUMNS: tuple[str, ...] = ("voice_id", "audio", "text", "source_clip_id")


def _hf_features_metadata() -> dict[bytes, bytes]:
    features = {
        "voice_id": {"dtype": "string", "_type": "Value"},
        "audio": {"sampling_rate": 24000, "_type": "Audio"},
        "text": {"dtype": "string", "_type": "Value"},
        "source_clip_id": {"dtype": "string", "_type": "Value"},
    }
    payload = json.dumps({"info": {"features": features}})
    return {b"huggingface": payload.encode("utf-8")}


VOICE_SCHEMA = pa.schema(
    [
        pa.field("voice_id", pa.string()),
        pa.field("audio", _AUDIO_TYPE),
        pa.field("text", pa.string()),
        pa.field("source_clip_id", pa.string()),
    ],
    metadata=_hf_features_metadata(),
)


def voice_id_for(*, source: str, source_speaker_id: str) -> str:
    """Deterministic 12-hex-char voice identifier.

    Stable for a given (source, source_speaker_id). Independent of which clip
    was selected for the speaker, so re-running with different filter params
    that pick a different clip leaves voice_id unchanged.
    """
    payload = f"{source}:{source_speaker_id}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]
