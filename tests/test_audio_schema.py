from __future__ import annotations

import pyarrow as pa

from radiotalk._pa import pydantic_to_pa_schema
from radiotalk.audio.schema import (
    AudioBytes,
    AudioTurnRow,
    SAMPLE_RATE,
    TokenSpan,
)


SCENARIO_FIXTURE = {
    "icao": "KSFO",
    "region": "us",
    "phase": "ground",
    "aircraft": [
        {"callsign": "DLH9462", "aircraft_type": "B748", "wake": "H", "operator_class": "commercial"},
    ],
    "runway": "28L",
    "sid_star": "SID4",
    "squawk": "4402",
    "frequency_mhz": 121.65,
    "weather": {
        "wind_dir_deg": 270, "wind_kt": 8, "vis_sm": 10.0,
        "ceiling_ft": 5000, "altimeter_inhg": 30.02, "vmc_imc": "VMC",
    },
    "time_of_day": "day",
    "traffic_density": "moderate",
    "event": "routine",
    "callsign": "DLH9462",
    "aircraft_type": "B748",
    "wake": "H",
    "operator_class": "commercial",
    "n_aircraft": 1,
    "is_emergency": False,
    "is_towered": True,
}


def test_schema_top_level_field_names():
    schema = pydantic_to_pa_schema(AudioTurnRow)
    expected = {
        "scenario_id", "scenario", "turn_idx", "speaker", "text",
        "text_normalized", "voice_id", "audio", "tokens",
        "model", "prompt_version", "taxonomy_version", "tts_model",
    }
    assert set(schema.names) == expected


def test_audio_is_top_level_struct():
    """Audio at the top level so the HF Hub viewer renders it inline.
    Nested-Audio (inside list<struct>) breaks the viewer."""
    schema = pydantic_to_pa_schema(AudioTurnRow)
    audio = schema.field("audio").type
    assert pa.types.is_struct(audio)
    fields = {audio.field(i).name for i in range(audio.num_fields)}
    assert fields == {"bytes", "path"}
    assert audio.field("bytes").type == pa.binary()


def test_tokens_field_is_nullable_list_of_struct():
    schema = pydantic_to_pa_schema(AudioTurnRow)
    tokens = schema.field("tokens").type
    assert pa.types.is_list(tokens)
    span = tokens.value_type
    fields = {span.field(i).name for i in range(span.num_fields)}
    assert fields == {"text", "start_s", "end_s"}


def test_tokens_can_be_none():
    row = AudioTurnRow(
        scenario_id="s1",
        scenario=SCENARIO_FIXTURE,
        turn_idx=0,
        speaker="DLH9462",
        text="DLH9462, ready for push",
        text_normalized="lufthansa nine four six two, ready for push",
        voice_id="v0001",
        audio=AudioBytes(bytes=b"opus"),
        tokens=None,
        model="qwen-32b",
        prompt_version="v1",
        taxonomy_version="v1",
        tts_model="HumeAI/tada-3b-ml",
    )
    assert row.tokens is None


def test_row_round_trip_with_tokens():
    row = AudioTurnRow(
        scenario_id="s1",
        scenario=SCENARIO_FIXTURE,
        turn_idx=2,
        speaker="DLH9462",
        text="DLH9462",
        text_normalized="lufthansa nine four six two",
        voice_id="v0007",
        audio=AudioBytes(bytes=b"opus_bytes"),
        tokens=[
            TokenSpan(text="lufth", start_s=0.0, end_s=0.2),
            TokenSpan(text="ansa", start_s=0.2, end_s=0.5),
        ],
        model="qwen-32b",
        prompt_version="v1",
        taxonomy_version="v1",
        tts_model="HumeAI/tada-3b-ml",
    )
    dumped = row.model_dump(mode="python")
    assert dumped["audio"]["bytes"] == b"opus_bytes"
    assert dumped["turn_idx"] == 2
    assert len(dumped["tokens"]) == 2


def test_no_generated_at_field():
    """`generated_at` was dropped — provenance comes from the dataset
    publish/commit time, not row-level timestamps."""
    schema = pydantic_to_pa_schema(AudioTurnRow)
    assert "generated_at" not in schema.names


def test_sample_rate_constant():
    assert SAMPLE_RATE == 24000