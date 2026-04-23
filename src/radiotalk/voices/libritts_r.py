from __future__ import annotations

from typing import Callable, Iterable, Iterator

import numpy as np

from .source import SourceCandidate, register_source


SOURCE_NAME = "libritts-r"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus "
    "(Koizumi et al., 2023). Licensed CC BY 4.0."
)
DEFAULT_HF_DATASET = "mythicinfinity/libritts_r"
# All three training splits, streamed in order. Clean first (higher quality),
# then "other" (more variable but cleaned by LibriTTS-R restoration).
DEFAULT_HF_SPLITS: tuple[tuple[str, str], ...] = (
    ("clean", "train.clean.100"),
    ("clean", "train.clean.360"),
    ("other", "train.other.500"),
)
DEFAULT_HF_REVISION = "main"


def _to_candidate(row: dict) -> SourceCandidate:
    audio = row["audio"]
    arr = np.asarray(audio["array"], dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32)
    return SourceCandidate(
        source=SOURCE_NAME,
        source_speaker_id=str(row["speaker_id"]),
        source_clip_id=str(row["id"]),
        audio=arr,
        sample_rate=int(audio["sampling_rate"]),
        license=LICENSE,
        attribution=ATTRIBUTION,
        gender_hint=None,
        language="en",
    )


def _default_stream() -> Iterable[dict]:
    from datasets import load_dataset

    for config, split in DEFAULT_HF_SPLITS:
        ds = load_dataset(
            DEFAULT_HF_DATASET,
            name=config,
            split=split,
            streaming=True,
            revision=DEFAULT_HF_REVISION,
        )
        yield from ds  # type: ignore[misc]


class LibriTTSR:
    name = SOURCE_NAME

    def __init__(
        self,
        *,
        stream_factory: Callable[[], Iterable[dict]] = _default_stream,
    ) -> None:
        self._stream_factory = stream_factory

    def candidates(self) -> Iterator[SourceCandidate]:
        for row in self._stream_factory():
            yield _to_candidate(row)


register_source("libritts-r", sub_pool="controller", factory=LibriTTSR)