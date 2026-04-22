from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from openai import AsyncOpenAI

from radiotalk.data.runtime import RuntimeConfig
from radiotalk.data.generator import run
from radiotalk.data.scenario import ScenarioSampler
from radiotalk.data.writer import ParquetShardWriter

BASE_URL = "http://test-server/v1"


def _chat_completion_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _valid_transcript_json() -> str:
    base = [
        {"speaker": "ATC", "callsign": "KSFO_TWR", "facility": "KSFO_TWR",
         "text": "hello", "intent": "greeting"},
        {"speaker": "PILOT", "callsign": "DAL123", "facility": None,
         "text": "hi", "intent": "ack"},
        {"speaker": "ATC", "callsign": "KSFO_TWR", "facility": "KSFO_TWR",
         "text": "cleared for takeoff", "intent": "clearance"},
        {"speaker": "PILOT", "callsign": "DAL123", "facility": None,
         "text": "roger cleared for takeoff", "intent": "readback"},
    ]
    return json.dumps({"turns": base})


def _cfg(tmp_path: Path, *, decoding: str = "json_schema") -> RuntimeConfig:
    return RuntimeConfig(
        base_url=BASE_URL,
        api_key="x",
        model="test-model",
        count=3,
        out_dir=tmp_path,
        concurrency=2,
        max_retries=3,
        decoding=decoding,  # type: ignore[arg-type]
    )


def _writer(tmp_path: Path, cfg: RuntimeConfig) -> ParquetShardWriter:
    return ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=cfg.shard_size,
        config_fingerprint=cfg.fingerprint(),
        seed=cfg.seed,
        resume=False,
        overwrite=True,
    )


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_json_schema(tmp_path: Path):
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_response(_valid_transcript_json()))
    )
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=7)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=list(sampler.iter(3)), writer=writer, client=client)
    await client.close()
    assert stats.ok == 3 and stats.failed == 0
    assert route.call_count == 3
    payload = json.loads(route.calls[0].request.content.decode())
    rf = payload.get("response_format")
    assert rf and rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "ModelTranscript"
    assert rf["json_schema"]["schema"]["type"] == "object"
    assert "guided_json" not in payload


@pytest.mark.asyncio
@respx.mock
async def test_free_mode_omits_response_format(tmp_path: Path):
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_response(_valid_transcript_json()))
    )
    cfg = _cfg(tmp_path, decoding="free")
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=8)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    await run(cfg=cfg, scenarios=list(sampler.iter(1)), writer=writer, client=client)
    await client.close()
    payload = json.loads(route.calls[0].request.content.decode())
    assert "response_format" not in payload
    assert "extra_body" not in payload


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(tmp_path: Path):
    responses = [
        httpx.Response(500, json={"error": {"message": "oops"}}),
        httpx.Response(200, json=_chat_completion_response(_valid_transcript_json())),
    ]
    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=responses)
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=9)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=list(sampler.iter(1)), writer=writer, client=client)
    await client.close()
    assert stats.ok == 1 and stats.failed == 0


@pytest.mark.asyncio
@respx.mock
async def test_retry_exhaustion_records_failure(tmp_path: Path):
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "down"}})
    )
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=10)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=list(sampler.iter(1)), writer=writer, client=client)
    await client.close()
    assert stats.ok == 0 and stats.failed == 1
    assert (tmp_path / "failures.jsonl").exists()


@pytest.mark.asyncio
@respx.mock
async def test_invalid_json_under_free_mode_records_failure(tmp_path: Path):
    # Under SGLang's json_schema (default) the engine guarantees valid output;
    # under --decoding free a malformed response is a hard failure (no nudge).
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_response("not json"))
    )
    cfg = _cfg(tmp_path, decoding="free")
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=11)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=list(sampler.iter(1)), writer=writer, client=client)
    await client.close()
    assert stats.ok == 0 and stats.failed == 1


@pytest.mark.asyncio
@respx.mock
async def test_concurrency_does_not_exceed_limit(tmp_path: Path):
    import asyncio

    in_flight = 0
    peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json=_chat_completion_response(_valid_transcript_json()))

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=handler)
    cfg = _cfg(tmp_path)  # concurrency=2
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=12)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    await run(cfg=cfg, scenarios=list(sampler.iter(8)), writer=writer, client=client)
    await client.close()
    assert peak <= cfg.concurrency
