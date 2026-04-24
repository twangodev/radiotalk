from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class SourceCandidate:
    source: str
    source_speaker_id: str
    source_clip_id: str
    audio: np.ndarray
    sample_rate: int
    license: str
    attribution: str
    gender_hint: str | None = None
    language: str = "en"
    text: str | None = None


class Source(Protocol):
    """Iterates candidate clips from an upstream dataset.

    Implementations are expected to stream (not download) and to yield clips
    in the upstream's natural order. The caller is responsible for filtering,
    per-speaker capping, and stopping early when target counts are met.
    """

    name: str

    def candidates(self) -> Iterator[SourceCandidate]: ...


@dataclass(frozen=True)
class SourceInfo:
    name: str
    sub_pool: str
    factory: Callable[[], Source]


_REGISTRY: dict[str, SourceInfo] = {}


def register_source(
    name: str, *, sub_pool: str, factory: Callable[[], Source],
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"source {name!r} already registered")
    _REGISTRY[name] = SourceInfo(name=name, sub_pool=sub_pool, factory=factory)


def get_source(name: str) -> Source:
    info = _REGISTRY.get(name)
    if info is None:
        raise KeyError(f"unknown source: {name!r}. Available: {sorted(_REGISTRY)}")
    return info.factory()


def get_sub_pool(name: str) -> str:
    return _REGISTRY[name].sub_pool


def available_sources() -> list[str]:
    return sorted(_REGISTRY)