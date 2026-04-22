from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .prompt import PROMPT_VERSION
from .scenario import TAXONOMY_VERSION


@dataclass(frozen=True)
class RuntimeConfig:
    base_url: str
    api_key: str
    model: str
    count: int
    out_dir: Path
    concurrency: int = 32
    temperature: float = 0.9
    max_tokens: int = 1024
    shard_size: int = 10_000
    seed: int = 42
    weighter_name: str = "tier_v1"
    regions: tuple[str, ...] = ("us",)
    config_name: str = "us"
    prompt_version: str = PROMPT_VERSION
    taxonomy_version: str = TAXONOMY_VERSION
    max_retries: int = 5

    def fingerprint(self) -> str:
        """Hash of fields that must match for a resume to be valid."""
        payload = {
            "model": self.model,
            "seed": self.seed,
            "prompt_version": self.prompt_version,
            "taxonomy_version": self.taxonomy_version,
            "weighter_name": self.weighter_name,
            "regions": sorted(self.regions),
            "config_name": self.config_name,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def dump(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d
