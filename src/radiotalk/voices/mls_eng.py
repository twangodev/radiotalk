from __future__ import annotations

import io
from typing import Callable, Iterable, Iterator

import numpy as np
import soundfile as sf

from .source import SourceCandidate, register_source


SOURCE_NAME = "mls-eng"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "Multilingual LibriSpeech (MLS) English split (Pratap et al., 2020). "
    "Parler-TTS distribution. Licensed CC BY 4.0."
)
DEFAULT_HF_DATASET = "parler-tts/mls_eng"
DEFAULT_HF_SPLIT = "train"
DEFAULT_HF_REVISION = "main"

# Conservative pre-filter on the cheap audio_duration metadata column so we
# skip audio decoding for clips that can't pass the main duration filter.
# Slightly wider than the default FilterParams (10-30s) to avoid dropping
# clips that are borderline after re-measurement from the audio array.
PREFILTER_MIN_S = 8.0
PREFILTER_MAX_S = 35.0


def _to_candidate(row: dict) -> SourceCandidate:
    audio = row["audio"]
    # With decode=False the audio column yields {"path": str, "bytes": bytes}.
    arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32)
    return SourceCandidate(
        source=SOURCE_NAME,
        source_speaker_id=str(row["speaker_id"]),
        source_clip_id=str(row["original_path"]),
        audio=arr,
        sample_rate=int(sr),
        license=LICENSE,
        attribution=ATTRIBUTION,
        gender_hint=None,
        language="en",
    )


def _default_stream() -> Iterable[dict]:
    from datasets import Audio, load_dataset

    ds = load_dataset(
        DEFAULT_HF_DATASET,
        split=DEFAULT_HF_SPLIT,
        streaming=True,
        revision=DEFAULT_HF_REVISION,
    )
    # Don't decode audio during iteration — we pre-filter on metadata first
    # and only manually decode (via soundfile) for clips we actually yield.
    ds = ds.cast_column("audio", Audio(decode=False))
    yield from ds  # type: ignore[misc]


class MLSEng:
    name = SOURCE_NAME

    def __init__(
        self,
        *,
        stream_factory: Callable[[], Iterable[dict]] = _default_stream,
    ) -> None:
        self._stream_factory = stream_factory

    def candidates(self) -> Iterator[SourceCandidate]:
        # Dedup by speaker and duration on the cheap metadata columns before
        # accessing row["audio"] (which triggers a torchcodec decode). MLS has
        # many clips per speaker and a wide duration spread, so filtering on
        # metadata is a 100×+ speedup over the naive yield-everything loop.
        seen: set[str] = set()
        for row in self._stream_factory():
            spk = str(row["speaker_id"])
            if spk in seen:
                continue
            dur = row.get("audio_duration")
            if dur is not None and not (PREFILTER_MIN_S <= dur <= PREFILTER_MAX_S):
                continue
            seen.add(spk)
            yield _to_candidate(row)


register_source("mls-eng", sub_pool="controller", factory=MLSEng)
