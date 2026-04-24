from __future__ import annotations

import pyarrow as pa

from radiotalk.voices.manifest import VOICE_SCHEMA, VoiceRecord


def _record() -> VoiceRecord:
    return VoiceRecord(
        voice_id="abc123def456",
        source="libritts-r",
        source_speaker_id="1034",
        source_clip_id="1034_121119_000001_000001",
        duration_s=4.231,
        sample_rate=24000,
        license="CC-BY-4.0",
        attribution="LibriTTS-R (Koizumi et al., 2023), CC BY 4.0",
        sub_pool="controller",
        gender_hint="F",
        language="en",
        selected_at="2026-04-22T15:00:00+00:00",
    )


def test_voice_schema_has_expected_columns() -> None:
    assert VOICE_SCHEMA.names == ["voice_id", "audio", "text", "source_clip_id"]


def test_voice_record_optional_fields_default_to_none() -> None:
    rec = VoiceRecord(
        voice_id="abc123def456",
        source="libritts-r",
        source_speaker_id="1034",
        source_clip_id="x",
        duration_s=4.0,
        sample_rate=24000,
        license="CC-BY-4.0",
        attribution="x",
        selected_at="2026-04-22T15:00:00+00:00",
    )
    assert rec.sub_pool is None
    assert rec.gender_hint is None
    assert rec.language == "en"


from radiotalk.voices.manifest import voice_id_for


def test_voice_id_is_stable_across_calls() -> None:
    a = voice_id_for(source="libritts-r", source_speaker_id="1034")
    b = voice_id_for(source="libritts-r", source_speaker_id="1034")
    assert a == b


def test_voice_id_differs_by_source_or_speaker() -> None:
    a = voice_id_for(source="libritts-r", source_speaker_id="1034")
    b = voice_id_for(source="libritts-r", source_speaker_id="1035")
    c = voice_id_for(source="vctk", source_speaker_id="1034")
    assert len({a, b, c}) == 3


def test_voice_id_format() -> None:
    vid = voice_id_for(source="libritts-r", source_speaker_id="1034")
    assert len(vid) == 12
    assert all(ch in "0123456789abcdef" for ch in vid)
