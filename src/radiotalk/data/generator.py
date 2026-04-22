from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Iterable

from openai import AsyncOpenAI
from openai import APIStatusError, RateLimitError, APIConnectionError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .runtime import RuntimeConfig
from .prompt import build as build_prompt
from .scenario import Scenario
from .transcript import (
    ModelTranscript,
    Transcript,
    model_transcript_json_schema,
)
from .writer import ParquetShardWriter

log = logging.getLogger("radiotalk.generator")


@dataclass
class GenStats:
    ok: int = 0
    failed: int = 0


ProgressCallback = Callable[[GenStats], None]


class _HttpError(Exception):
    """Wraps transient HTTP errors we want tenacity to retry."""


def _request_kwargs(cfg: RuntimeConfig) -> dict:
    """Build chat-completion kwargs for an SGLang OpenAI-compatible endpoint."""
    kwargs: dict = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.decoding == "json_schema":
        # SGLang structured outputs (OpenAI spec). The engine enforces the
        # schema during decoding, including min_length on `turns`.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ModelTranscript",
                "schema": model_transcript_json_schema(),
            },
        }
    # "free" adds nothing — no constraints, validated post-hoc.
    return kwargs


async def _one_call(
    client: AsyncOpenAI,
    cfg: RuntimeConfig,
    messages: list[dict],
) -> str:
    try:
        resp = await client.chat.completions.create(messages=messages, **_request_kwargs(cfg))
    except (RateLimitError, APIConnectionError) as e:
        raise _HttpError(str(e)) from e
    except APIStatusError as e:
        if 500 <= e.status_code < 600:
            raise _HttpError(str(e)) from e
        raise
    return resp.choices[0].message.content or ""


async def _call_with_retries(
    client: AsyncOpenAI, cfg: RuntimeConfig, messages: list[dict]
) -> str:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_HttpError),
        reraise=True,
    ):
        with attempt:
            return await _one_call(client, cfg, messages)
    raise RuntimeError("unreachable")  # pragma: no cover


def _parse(content: str) -> ModelTranscript:
    data = json.loads(content)
    return ModelTranscript.model_validate(data)


async def _generate_one(
    client: AsyncOpenAI, cfg: RuntimeConfig, scenario: Scenario
) -> Transcript:
    messages = build_prompt(scenario)
    content = await _call_with_retries(client, cfg, messages)
    parsed = _parse(content)
    return Transcript(
        scenario_id=scenario.scenario_id,
        scenario=scenario,
        turns=parsed.turns,
        model=cfg.model,
        generated_at=datetime.now(timezone.utc),
        prompt_version=cfg.prompt_version,
        taxonomy_version=cfg.taxonomy_version,
        decoding=cfg.decoding,
    )


async def run(
    cfg: RuntimeConfig,
    scenarios: Iterable[Scenario] | AsyncIterator[Scenario],
    writer: ParquetShardWriter,
    *,
    on_progress: ProgressCallback | None = None,
    client: AsyncOpenAI | None = None,
) -> GenStats:
    stats = GenStats()
    writer_lock = asyncio.Lock()
    queue: asyncio.Queue[Scenario | None] = asyncio.Queue(maxsize=cfg.concurrency * 2)
    owns_client = client is None
    if client is None:
        client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "not-needed")

    async def feeder() -> None:
        if hasattr(scenarios, "__aiter__"):
            async for s in scenarios:  # type: ignore[union-attr]
                await queue.put(s)
        else:
            for s in scenarios:  # type: ignore[assignment]
                await queue.put(s)
        for _ in range(cfg.concurrency):
            await queue.put(None)

    async def worker() -> None:
        while True:
            scenario = await queue.get()
            if scenario is None:
                return
            try:
                transcript = await _generate_one(client, cfg, scenario)
            except Exception as e:  # noqa: BLE001 — any failure -> failures.jsonl
                async with writer_lock:
                    writer.add_failure(
                        scenario_id=scenario.scenario_id,
                        scenario=scenario.model_dump(mode="json"),
                        error=f"{type(e).__name__}: {e}",
                    )
                    stats.failed += 1
                    if on_progress:
                        on_progress(stats)
                continue
            async with writer_lock:
                writer.add(transcript)
                stats.ok += 1
                if on_progress:
                    on_progress(stats)

    try:
        await asyncio.gather(feeder(), *(worker() for _ in range(cfg.concurrency)))
    finally:
        async with writer_lock:
            writer.close()
        if owns_client:
            await client.close()

    return stats
