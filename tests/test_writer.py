from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from radiotalk.data.scenario import ScenarioSampler
from radiotalk.data.transcript import Transcript, Turn
from radiotalk.data.writer import (
    ConfigFingerprintMismatch,
    Manifest,
    ParquetShardWriter,
)


def _make_transcript(scenario, i: int) -> Transcript:
    return Transcript(
        scenario_id=scenario.scenario_id,
        scenario=scenario,
        turns=[
            Turn(
                speaker="ATC",
                callsign="KSFO_TWR",
                facility="KSFO_TWR",
                text=f"turn {i}",
                intent="ack",
            )
        ],
        model="test-model",
        generated_at=datetime.now(timezone.utc),
        prompt_version="p1",
        taxonomy_version="t1",
        decoding="json_schema",
    )


def test_shard_rotation_and_atomicity(tmp_path: Path):
    sampler = ScenarioSampler(seed=1)
    writer = ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=5,
        config_fingerprint="fp",
        seed=1,
        resume=False,
        overwrite=False,
    )
    for i, sc in enumerate(sampler.iter(12)):
        writer.add(_make_transcript(sc, i))
    writer.close()
    shards = sorted(tmp_path.glob("shard-*.parquet"))
    assert [p.name for p in shards] == [
        "shard-000000.parquet",
        "shard-000001.parquet",
        "shard-000002.parquet",
    ]
    assert not list(tmp_path.glob("*.tmp"))
    total = sum(pq.read_table(p).num_rows for p in shards)
    assert total == 12


def test_resume_continues_from_manifest(tmp_path: Path):
    sampler = ScenarioSampler(seed=2)
    w1 = ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=3,
        config_fingerprint="fp",
        seed=2,
        resume=False,
        overwrite=False,
    )
    first = list(sampler.iter(3))
    for i, sc in enumerate(first):
        w1.add(_make_transcript(sc, i))
    w1.close()

    w2 = ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=3,
        config_fingerprint="fp",
        seed=2,
        resume=True,
        overwrite=False,
    )
    assert w2.total_rows == 3
    assert w2.last_shard_index == 0

    sampler2 = ScenarioSampler(seed=2)
    sampler2.fast_forward(3)
    for i, sc in enumerate(sampler2.iter(3)):
        w2.add(_make_transcript(sc, i + 10))
    w2.close()

    shards = sorted(tmp_path.glob("shard-*.parquet"))
    assert len(shards) == 2
    total = sum(pq.read_table(p).num_rows for p in shards)
    assert total == 6


def test_resume_refuses_mismatched_fingerprint(tmp_path: Path):
    sampler = ScenarioSampler(seed=3)
    w = ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=2,
        config_fingerprint="orig",
        seed=3,
        resume=False,
        overwrite=False,
    )
    for i, sc in enumerate(sampler.iter(2)):
        w.add(_make_transcript(sc, i))
    w.close()

    with pytest.raises(ConfigFingerprintMismatch):
        ParquetShardWriter.open(
            out_dir=tmp_path,
            shard_size=2,
            config_fingerprint="different",
            seed=3,
            resume=True,
            overwrite=False,
        )


def test_overwrite_clears_out_dir(tmp_path: Path):
    sampler = ScenarioSampler(seed=4)
    w = ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=2,
        config_fingerprint="fp",
        seed=4,
        resume=False,
        overwrite=False,
    )
    for i, sc in enumerate(sampler.iter(2)):
        w.add(_make_transcript(sc, i))
    w.close()
    assert list(tmp_path.glob("shard-*.parquet"))

    ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=2,
        config_fingerprint="fp",
        seed=4,
        resume=False,
        overwrite=True,
    )
    assert not list(tmp_path.glob("shard-*.parquet"))


def test_add_failure_appends_jsonl(tmp_path: Path):
    sc = next(iter(ScenarioSampler(seed=5).iter(1)))
    w = ParquetShardWriter.open(
        out_dir=tmp_path,
        shard_size=10,
        config_fingerprint="fp",
        seed=5,
        resume=False,
        overwrite=False,
    )
    w.add_failure(
        scenario_id=sc.scenario_id, scenario=sc.model_dump(mode="json"), error="boom"
    )
    w.add_failure(
        scenario_id=sc.scenario_id, scenario=sc.model_dump(mode="json"), error="kaboom"
    )
    lines = (tmp_path / "failures.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_manifest_roundtrip(tmp_path: Path):
    m = Manifest(
        seed=1,
        config_fingerprint="fp",
        total_rows=5,
        last_shard_index=0,
        started_at="2026-04-21T00:00:00+00:00",
        updated_at="2026-04-21T00:00:01+00:00",
    )
    p = tmp_path / "manifest.json"
    m.dump(p)
    loaded = Manifest.load(p)
    assert loaded == m
