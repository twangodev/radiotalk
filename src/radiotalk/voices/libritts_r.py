from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .source import SourceCandidate, register_source


SOURCE_NAME = "libritts-r"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = (
    "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus "
    "(Koizumi et al., 2023). Licensed CC BY 4.0."
)
DEFAULT_HF_DATASET = "mythicinfinity/libritts_r"
DEFAULT_HF_REVISION = "main"
DEFAULT_TRAIN_SPLITS: tuple[str, ...] = (
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
)

SCAN_MIN_S = 1.0
SCAN_MAX_S = 30.0


@dataclass(frozen=True)
class _BestClip:
    speaker_id: str
    clip_id: str
    duration_s: float
    sample_rate: int
    audio_bytes: bytes
    text: str | None = None


def _probe_wav(audio_bytes: bytes) -> tuple[float, int] | None:
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception:
        return None
    if info.samplerate <= 0:
        return None
    return info.frames / info.samplerate, info.samplerate


def _to_candidate(clip: _BestClip) -> SourceCandidate:
    arr, sr = sf.read(io.BytesIO(clip.audio_bytes), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32)
    return SourceCandidate(
        source=SOURCE_NAME,
        source_speaker_id=clip.speaker_id,
        source_clip_id=clip.clip_id,
        audio=arr,
        sample_rate=int(sr),
        license=LICENSE,
        attribution=ATTRIBUTION,
        gender_hint=None,
        language="en",
        text=clip.text,
    )


def _download_shards(splits: Iterable[str]) -> list[Path]:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    patterns = [f"data/{split}/*.parquet" for split in splits]
    local = Path(
        snapshot_download(
            repo_id=DEFAULT_HF_DATASET,
            repo_type="dataset",
            revision=DEFAULT_HF_REVISION,
            allow_patterns=patterns,
        )
    )
    shards: list[Path] = []
    for split in splits:
        shards.extend(sorted((local / "data" / split).glob("*.parquet")))
    return shards


ProgressFn = Callable[[str, int, int], None]


def _scan_longest(
    shards: Iterable[Path],
    *,
    min_s: float = SCAN_MIN_S,
    max_s: float = SCAN_MAX_S,
    on_progress: ProgressFn | None = None,
) -> dict[str, _BestClip]:
    best: dict[str, _BestClip] = {}
    rows_seen = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for g in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(
                g, columns=["audio", "speaker_id", "id", "text_normalized"],
            )
            audios = tbl.column("audio").to_pylist()
            speakers = tbl.column("speaker_id").to_pylist()
            ids = tbl.column("id").to_pylist()
            texts = tbl.column("text_normalized").to_pylist()
            for a, spk, cid, txt in zip(audios, speakers, ids, texts):
                rows_seen += 1
                if not a:
                    continue
                b = a.get("bytes")
                if not b:
                    continue
                probe = _probe_wav(b)
                if probe is None:
                    continue
                dur, sr = probe
                if dur < min_s or dur > max_s:
                    continue
                s_id = str(spk)
                prev = best.get(s_id)
                if prev is None or dur > prev.duration_s:
                    best[s_id] = _BestClip(
                        speaker_id=s_id,
                        clip_id=str(cid),
                        duration_s=dur,
                        sample_rate=int(sr),
                        audio_bytes=b,
                        text=str(txt) if txt is not None else None,
                    )
            if on_progress is not None:
                on_progress(str(shard.name), rows_seen, len(best))
    return best


def _default_scan() -> Iterable[_BestClip]:
    from rich.console import Console

    console = Console()
    console.print(f"[dim]libritts-r:[/] downloading {len(DEFAULT_TRAIN_SPLITS)} train splits...")
    shards = _download_shards(DEFAULT_TRAIN_SPLITS)
    console.print(f"[dim]libritts-r:[/] scanning {len(shards)} parquet shards for longest clip per speaker...")

    def _log(shard: str, rows: int, speakers: int) -> None:
        console.log(f"  {shard}: rows_seen={rows:,} speakers={speakers}")

    best = _scan_longest(shards, on_progress=_log)
    console.print(f"[green]libritts-r:[/] scan done. {len(best)} speakers, yielding longest clip each.")
    return best.values()


class LibriTTSR:
    name = SOURCE_NAME

    def __init__(
        self,
        *,
        scan_factory: Callable[[], Iterable[_BestClip]] = _default_scan,
    ) -> None:
        self._scan_factory = scan_factory

    def candidates(self) -> Iterator[SourceCandidate]:
        for clip in self._scan_factory():
            yield _to_candidate(clip)


register_source("libritts-r", sub_pool="controller", factory=LibriTTSR)
