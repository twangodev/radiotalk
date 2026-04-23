from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from radiotalk._writer import MANIFEST_NAME, SHARD_TEMPLATE, ShardedParquetWriter

SCHEMA = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])


def test_writes_shard_and_manifest(tmp_path: Path) -> None:
    w = ShardedParquetWriter.open(
        tmp_path, SCHEMA, shard_size=3, resume=False, meta={"purpose": "test"},
    )
    for i in range(5):
        w.add_row({"id": i, "name": f"row-{i}"})
    w.close()

    shard0 = tmp_path / SHARD_TEMPLATE.format(idx=0)
    shard1 = tmp_path / SHARD_TEMPLATE.format(idx=1)
    assert shard0.exists()
    assert shard1.exists()
    assert pq.read_table(shard0).num_rows == 3
    assert pq.read_table(shard1).num_rows == 2

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["total_rows"] == 5
    assert manifest["last_shard_index"] == 1
    assert manifest["purpose"] == "test"
    assert "started_at" in manifest
    assert "updated_at" in manifest


def test_resume_continues_from_last_shard(tmp_path: Path) -> None:
    w1 = ShardedParquetWriter.open(
        tmp_path, SCHEMA, shard_size=10, resume=False, meta={"k": "v"},
    )
    for i in range(3):
        w1.add_row({"id": i, "name": f"row-{i}"})
    w1.close()

    w2 = ShardedParquetWriter.open(
        tmp_path, SCHEMA, shard_size=10, resume=True, meta={"k": "v"},
    )
    assert w2.total_rows == 3
    assert w2.last_shard_index == 0
    for i in range(3, 5):
        w2.add_row({"id": i, "name": f"row-{i}"})
    w2.close()

    assert w2.total_rows == 5
    shard1 = tmp_path / SHARD_TEMPLATE.format(idx=1)
    assert shard1.exists()
    assert pq.read_table(shard1).num_rows == 2


def test_empty_close_writes_manifest_only(tmp_path: Path) -> None:
    w = ShardedParquetWriter.open(tmp_path, SCHEMA, shard_size=10, resume=False)
    w.close()
    assert (tmp_path / MANIFEST_NAME).exists()
    assert w.total_rows == 0
    assert not (tmp_path / SHARD_TEMPLATE.format(idx=0)).exists()
