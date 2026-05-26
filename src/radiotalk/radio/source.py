from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


_CLEAN_COLS = [
    "scenario_id", "scenario", "turn_idx", "speaker", "text", "text_normalized",
    "voice_id", "audio", "tokens",
    "model", "prompt_version", "taxonomy_version", "tts_model",
]


def snapshot_clean_repo(repo_id_or_path: str) -> Path:
    """Resolve the local directory holding shard-*.parquet files.

    Accepts either a HuggingFace dataset repo id (downloads if missing,
    no-op if already cached) or a local directory path. Local paths let
    us run the radio pipeline against an in-progress audio build before
    it has been published to the Hub.
    """
    local = Path(repo_id_or_path)
    if local.exists() and local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id=repo_id_or_path,
        repo_type="dataset",
        allow_patterns=["shard-*.parquet"],
    )
    return Path(p)


def list_clean_shards(snap_dir: Path) -> list[Path]:
    shards = sorted(snap_dir.glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.parquet under {snap_dir}")
    return shards


def count_clean_rows(shards: list[Path]) -> int:
    return sum(pq.read_metadata(s).num_rows for s in shards)


def iter_clean_rows(
    shards: list[Path],
    *,
    skip: int = 0,
    limit: int | None = None,
    batch_size: int = 512,
) -> Iterator[dict]:
    """Stream rows from the cached parquet shards. Memory ceiling is one
    record batch (~50 MB at default batch_size). Honors ``skip`` (resume)
    and ``limit``.
    """
    emitted = 0
    seen = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        shard_rows = pf.metadata.num_rows
        if seen + shard_rows <= skip:
            seen += shard_rows
            continue
        for batch in pf.iter_batches(batch_size=batch_size, columns=_CLEAN_COLS):
            n = batch.num_rows
            cols = {c: batch.column(c).to_pylist() for c in _CLEAN_COLS}
            for i in range(n):
                if seen < skip:
                    seen += 1
                    continue
                yield {c: cols[c][i] for c in _CLEAN_COLS}
                seen += 1
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
