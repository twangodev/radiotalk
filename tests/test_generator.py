from __future__ import annotations

import json
import re
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


_FOCAL_RE = re.compile(r"Focal aircraft:\s*(\S+)")


def _focal_from_request(request: httpx.Request) -> str:
    body = json.loads(request.content.decode())
    user_msg = body["messages"][1]["content"]
    m = _FOCAL_RE.search(user_msg)
    assert m, "focal aircraft line missing from user prompt"
    return m.group(1)


def _plaintext_for_callsign(cs: str) -> str:
    return (
        f"ATC: {cs}, contact tower one two one point niner\n"
        f"{cs}: tower one two one point niner, {cs}\n"
        f"ATC: {cs}, cleared for takeoff\n"
        f"{cs}: cleared for takeoff, {cs}\n"
    )


def _cfg(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        base_url=BASE_URL,
        api_key="x",
        model="test-model",
        count=3,
        out_dir=tmp_path,
        concurrency=2,
        max_retries=3,
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


async def _plaintext_handler(request):
    cs = _focal_from_request(request)
    return httpx.Response(200, json=_chat_completion_response(_plaintext_for_callsign(cs)))


@pytest.mark.asyncio
@respx.mock
async def test_happy_path_plaintext(tmp_path: Path):
    route = respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=_plaintext_handler)
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    scenarios = list(ScenarioSampler(seed=7).iter(3))
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=scenarios, writer=writer, client=client)
    await client.close()
    assert stats.ok == 3 and stats.failed == 0
    assert route.call_count == 3
    payload = json.loads(route.calls[0].request.content.decode())
    assert "response_format" not in payload
    assert "extra_body" not in payload


@pytest.mark.asyncio
@respx.mock
async def test_unparseable_output_records_failure(tmp_path: Path):
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_response("no colons here just prose"))
    )
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=11)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=list(sampler.iter(1)), writer=writer, client=client)
    await client.close()
    assert stats.ok == 0 and stats.failed == 1
    assert (tmp_path / "failures.jsonl").exists()


@pytest.mark.asyncio
@respx.mock
async def test_missing_focal_callsign_records_failure(tmp_path: Path):
    # Output has enough turns but never names the focal aircraft — rejected.
    bad = (
        "ATC: ground, runway two eight right\n"
        "UNKNOWN: roger\n"
        "ATC: taxi via alpha\n"
        "UNKNOWN: taxi alpha\n"
    )
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_response(bad))
    )
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    sampler = ScenarioSampler(seed=13)
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=list(sampler.iter(1)), writer=writer, client=client)
    await client.close()
    assert stats.failed == 1


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success(tmp_path: Path):
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": {"message": "oops"}})
        return await _plaintext_handler(request)

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=handler)
    cfg = _cfg(tmp_path)
    writer = _writer(tmp_path, cfg)
    scenarios = list(ScenarioSampler(seed=9).iter(1))
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    stats = await run(cfg=cfg, scenarios=scenarios, writer=writer, client=client)
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
        return await _plaintext_handler(request)

    respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=handler)
    cfg = _cfg(tmp_path)  # concurrency=2
    writer = _writer(tmp_path, cfg)
    scenarios = list(ScenarioSampler(seed=12).iter(8))
    client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    await run(cfg=cfg, scenarios=scenarios, writer=writer, client=client)
    await client.close()
    assert peak <= cfg.concurrency
