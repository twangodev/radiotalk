from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    scenario_id      TEXT    NOT NULL,
    turn_idx         INTEGER NOT NULL,
    voice_id         TEXT    NOT NULL,
    text_normalized  TEXT    NOT NULL,
    audio_bytes      BLOB    NOT NULL,
    tokens_json      TEXT,
    PRIMARY KEY (scenario_id, turn_idx)
);

CREATE TABLE IF NOT EXISTS voice_progress (
    voice_id    TEXT PRIMARY KEY,
    finished_at TEXT NOT NULL
);
"""


@dataclass
class StagedTurn:
    voice_id: str
    text_normalized: str
    audio_bytes: bytes
    tokens_json: str | None
    turn_idx: int


class Staging:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA page_size=65536")
        self._conn.executescript(_SCHEMA)

    def insert_turns(self, rows: list[tuple]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO turns "
                "(scenario_id, turn_idx, voice_id, text_normalized, audio_bytes, tokens_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def mark_voice_done(self, voice_id: str, when: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO voice_progress (voice_id, finished_at) VALUES (?, ?)",
                (voice_id, when),
            )

    def voices_done(self) -> set[str]:
        return {row[0] for row in self._conn.execute(
            "SELECT voice_id FROM voice_progress"
        )}

    def turns_for_scenario(self, scenario_id: str) -> list[StagedTurn]:
        cur = self._conn.execute(
            "SELECT turn_idx, voice_id, text_normalized, audio_bytes, tokens_json "
            "FROM turns WHERE scenario_id = ? ORDER BY turn_idx",
            (scenario_id,),
        )
        return [
            StagedTurn(
                turn_idx=row[0],
                voice_id=row[1],
                text_normalized=row[2],
                audio_bytes=row[3],
                tokens_json=row[4],
            )
            for row in cur.fetchall()
        ]

    def total_turns(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def __call__(self) -> Iterator["Staging"]:
        try:
            yield self
        finally:
            self.close()


def encode_tokens(spans) -> str | None:
    if spans is None:
        return None
    return json.dumps([
        {"text": s.text, "start_s": s.start_s, "end_s": s.end_s} for s in spans
    ])


def decode_tokens(tokens_json: str | None):
    if tokens_json is None:
        return None
    return json.loads(tokens_json)
