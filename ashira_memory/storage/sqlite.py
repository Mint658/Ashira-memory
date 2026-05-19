"""SQLite storage backend — the default for v0.1.

Why SQLite first:

* Zero-config. `Memory()` works out of the box on any machine.
* Companion app devs are mostly shipping personal / small-team tools
  where SQLite is plenty for the first 100k memories per user.
* Embeddings live in the same row as the text, so there's exactly
  one persistence target — no "your vector store and your metadata
  store drifted apart" failure mode (a real mem0 pain — see #5160).

Postgres / pgvector / Qdrant adapters ship as separate packages.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..types import Hit, MemoryEntry, Relationship


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id                      TEXT PRIMARY KEY,
    character_id            TEXT NOT NULL,
    user_id                 TEXT NOT NULL,
    text                    TEXT NOT NULL,
    created_at              REAL NOT NULL,
    last_accessed_at        REAL NOT NULL,
    access_count            INTEGER NOT NULL DEFAULT 0,
    importance              REAL NOT NULL DEFAULT 0.5,
    salience                REAL NOT NULL DEFAULT 0.5,
    reconsolidation_count   INTEGER NOT NULL DEFAULT 0,
    decay_half_life_hours   REAL NOT NULL DEFAULT 72.0,
    trust                   REAL NOT NULL DEFAULT 1.0,
    tags                    TEXT NOT NULL DEFAULT '[]',
    emotional               TEXT NOT NULL DEFAULT '{}',
    embedding               TEXT,
    deleted                 INTEGER NOT NULL DEFAULT 0,
    extra                   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_mem_char_user
    ON memories(character_id, user_id, deleted);

CREATE TABLE IF NOT EXISTS relationships (
    character_id            TEXT NOT NULL,
    user_id                 TEXT NOT NULL,
    trust                   REAL NOT NULL DEFAULT 0.5,
    familiarity             REAL NOT NULL DEFAULT 0.0,
    warmth                  REAL NOT NULL DEFAULT 0.5,
    interaction_count       INTEGER NOT NULL DEFAULT 0,
    last_interaction_at     REAL NOT NULL DEFAULT 0,
    shared_themes           TEXT NOT NULL DEFAULT '[]',
    callbacks               TEXT NOT NULL DEFAULT '[]',
    open_promises           TEXT NOT NULL DEFAULT '[]',
    broken_promises         TEXT NOT NULL DEFAULT '[]',
    notes                   TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (character_id, user_id)
);
"""


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    emb_raw = row["embedding"]
    return MemoryEntry(
        id=row["id"],
        character_id=row["character_id"],
        user_id=row["user_id"],
        text=row["text"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        access_count=row["access_count"],
        importance=row["importance"],
        salience=row["salience"],
        reconsolidation_count=row["reconsolidation_count"],
        decay_half_life_hours=row["decay_half_life_hours"],
        trust=row["trust"],
        tags=json.loads(row["tags"]),
        emotional=json.loads(row["emotional"]),
        embedding=json.loads(emb_raw) if emb_raw else None,
        deleted=bool(row["deleted"]),
        extra=json.loads(row["extra"]),
    )


def _row_to_relationship(row: sqlite3.Row) -> Relationship:
    return Relationship(
        character_id=row["character_id"],
        user_id=row["user_id"],
        trust=row["trust"],
        familiarity=row["familiarity"],
        warmth=row["warmth"],
        interaction_count=row["interaction_count"],
        last_interaction_at=row["last_interaction_at"],
        shared_themes=json.loads(row["shared_themes"]),
        callbacks=json.loads(row["callbacks"]),
        open_promises=json.loads(row["open_promises"]),
        broken_promises=json.loads(row["broken_promises"]),
        notes=json.loads(row["notes"]),
    )


class SQLiteStorage:
    """Default storage backend.

    All disk I/O is wrapped in ``asyncio.to_thread`` so the rest of
    the library can stay async-first without paying for an async DB
    driver in v0.1.
    """

    def __init__(self, path: str | Path = "ashira_memory.db") -> None:
        self._path = str(path)
        self._lock = asyncio.Lock()
        self._init_schema()

    # ----- internals -----

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as c:
            c.executescript(_SCHEMA)

    # ----- Storage protocol -----

    async def insert(self, entry: MemoryEntry) -> None:
        async with self._lock:
            await asyncio.to_thread(self._insert_sync, entry)

    def _insert_sync(self, e: MemoryEntry) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, character_id, user_id, text,
                    created_at, last_accessed_at, access_count,
                    importance, salience, reconsolidation_count,
                    decay_half_life_hours, trust,
                    tags, emotional, embedding, deleted, extra
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    e.id, e.character_id, e.user_id, e.text,
                    e.created_at, e.last_accessed_at, e.access_count,
                    e.importance, e.salience, e.reconsolidation_count,
                    e.decay_half_life_hours, e.trust,
                    json.dumps(e.tags), json.dumps(e.emotional),
                    json.dumps(e.embedding) if e.embedding is not None else None,
                    int(e.deleted), json.dumps(e.extra),
                ),
            )

    async def get(self, memory_id: str) -> MemoryEntry | None:
        return await asyncio.to_thread(self._get_sync, memory_id)

    def _get_sync(self, memory_id: str) -> MemoryEntry | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT * FROM memories WHERE id=? AND deleted=0", (memory_id,)
            ).fetchone()
        return _row_to_entry(row) if row else None

    async def search(
        self,
        *,
        character_id: str,
        user_id: str,
        embedding: list[float] | None = None,
        keywords: list[str] | None = None,
        k: int = 5,
    ) -> list[Hit]:
        return await asyncio.to_thread(
            self._search_sync,
            character_id, user_id, embedding, keywords, k,
        )

    def _search_sync(
        self,
        character_id: str,
        user_id: str,
        embedding: list[float] | None,
        keywords: list[str] | None,
        k: int,
    ) -> list[Hit]:
        # CRITICAL: every query is scoped by (character_id, user_id).
        # That's the isolation guarantee the README sells. Never
        # relax this filter without bumping the major version.
        with self._connect() as c:
            rows = c.execute(
                """
                SELECT * FROM memories
                 WHERE character_id=? AND user_id=? AND deleted=0
                """,
                (character_id, user_id),
            ).fetchall()

        entries = [_row_to_entry(r) for r in rows]
        if not entries:
            return []

        scored: list[tuple[float, str, MemoryEntry]] = []
        kw_set = {k.lower() for k in (keywords or []) if k}

        for e in entries:
            score = 0.0
            reason_bits: list[str] = []

            if embedding is not None and e.embedding is not None:
                sim = _cosine(embedding, e.embedding)
                # base relevance from semantic similarity
                score += 0.7 * max(sim, 0.0)
                if sim > 0:
                    reason_bits.append(f"sim={sim:.2f}")

            if kw_set:
                text_low = e.text.lower()
                hits = sum(1 for k in kw_set if k in text_low)
                if hits:
                    kw_score = hits / max(len(kw_set), 1)
                    score += 0.2 * kw_score
                    reason_bits.append(f"kw={hits}/{len(kw_set)}")

            # importance + reconsolidation lift, gently
            score += 0.1 * e.importance + 0.05 * min(e.reconsolidation_count, 5) / 5

            scored.append((score, "; ".join(reason_bits) or "default", e))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [Hit(entry=e, score=s, reason=r) for s, r, e in scored[:k]]

    async def soft_delete(self, memory_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._soft_delete_sync, memory_id)

    def _soft_delete_sync(self, memory_id: str) -> None:
        with self._connect() as c:
            c.execute("UPDATE memories SET deleted=1 WHERE id=?", (memory_id,))

    async def relationship_get(
        self, character_id: str, user_id: str
    ) -> Relationship:
        return await asyncio.to_thread(
            self._relationship_get_sync, character_id, user_id
        )

    def _relationship_get_sync(
        self, character_id: str, user_id: str
    ) -> Relationship:
        with self._connect() as c:
            row = c.execute(
                "SELECT * FROM relationships WHERE character_id=? AND user_id=?",
                (character_id, user_id),
            ).fetchone()
        if row is None:
            return Relationship(
                character_id=character_id,
                user_id=user_id,
                last_interaction_at=time.time(),
            )
        return _row_to_relationship(row)

    async def relationship_set(self, rel: Relationship) -> None:
        async with self._lock:
            await asyncio.to_thread(self._relationship_set_sync, rel)

    def _relationship_set_sync(self, r: Relationship) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO relationships (
                    character_id, user_id, trust, familiarity, warmth,
                    interaction_count, last_interaction_at,
                    shared_themes, callbacks,
                    open_promises, broken_promises, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    r.character_id, r.user_id, r.trust, r.familiarity, r.warmth,
                    r.interaction_count, r.last_interaction_at,
                    json.dumps(r.shared_themes), json.dumps(r.callbacks),
                    json.dumps(r.open_promises), json.dumps(r.broken_promises),
                    json.dumps(r.notes),
                ),
            )
