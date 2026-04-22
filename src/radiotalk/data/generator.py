from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Iterable

from openai import AsyncOpenAI
from openai import APIStatusError, RateLimitError, APIConnectionError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .runtime import RuntimeConfig
from .prompt import build as build_prompt
from .scenario import Scenario
from .transcript import Transcript, TranscriptParseError, parse_turns, validate_turns
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
    """Build chat-completion kwargs for an OpenAI-compatible endpoint.

    Plaintext output — no response_format, no grammar backend. The model is
    prompted to emit `SPEAKER: utterance` lines and we parse them post-hoc.
    Qwen3's reasoning mode is disabled: the dataset wants the transcript, not
    the CoT, and disabling it ~5xs throughput.
    """
    return {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


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


async def _generate_one(
    client: AsyncOpenAI, cfg: RuntimeConfig, scenario: Scenario
) -> Transcript:
    messages = build_prompt(scenario)
    last_err: TranscriptParseError | None = None
    for _ in range(max(1, cfg.max_parse_retries)):
        raw = await _call_with_retries(client, cfg, messages)
        try:
            turns = parse_turns(raw)
            validate_turns(turns, scenario)
        except TranscriptParseError as e:
            last_err = e
            continue
        return Transcript(
            scenario_id=scenario.scenario_id,
            scenario=scenario,
            turns=turns,
            model=cfg.model,
            generated_at=datetime.now(timezone.utc),
            prompt_version=cfg.prompt_version,
            taxonomy_version=cfg.taxonomy_version,
        )
    assert last_err is not None
    raise last_err


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
