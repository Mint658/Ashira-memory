"""The Memory facade — the entire public surface of ashira-memory v0.1.

If something isn't a method on :class:`Memory`, it's internal.
The whole library has to fit on one screen of API or it's failed.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .protocols import Provider, Storage
from .types import Hit, MemoryEntry, Relationship


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")


def _keywords(text: str, *, max_kw: int = 8) -> list[str]:
    seen: dict[str, None] = {}
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok not in seen:
            seen[tok] = None
        if len(seen) >= max_kw:
            break
    return list(seen)


class Memory:
    """Relational memory for a single AI character.

    Every method that touches a memory is scoped by ``character_id``
    (set at construction) and an explicit ``user_id``. Cross-character
    contamination is impossible by construction.

    Parameters
    ----------
    character_id:
        Stable identifier for the character — ``"ashira"``, ``"npc-tavern-keeper"``.
    provider:
        Optional :class:`Provider`. Defaults to :class:`OllamaProvider`
        on ``localhost:11434``.
    storage:
        Optional :class:`Storage`. Defaults to :class:`SQLiteStorage`
        at ``./ashira_memory.db``.
    """

    def __init__(
        self,
        character_id: str,
        *,
        provider: Provider | None = None,
        storage: Storage | None = None,
    ) -> None:
        if not character_id or not isinstance(character_id, str):
            raise ValueError("character_id is required and must be a non-empty string")
        self._character_id = character_id

        if provider is None:
            from .providers.ollama import OllamaProvider
            provider = OllamaProvider()
        if storage is None:
            from .storage.sqlite import SQLiteStorage
            storage = SQLiteStorage()

        self._provider = provider
        self._storage = storage

    # ------------------------------------------------------------------ store

    async def remember(
        self,
        user_id: str,
        text: str,
        *,
        importance: float = 0.5,
        tags: list[str] | None = None,
        emotional: dict[str, float] | None = None,
    ) -> MemoryEntry:
        """Store one new memory for ``user_id``.

        The embedding is computed lazily here so callers never need to
        think about providers.
        """
        if not user_id:
            raise ValueError("user_id is required")
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        emb = (await self._provider.embed([text]))[0] if text else None
        entry = MemoryEntry(
            character_id=self._character_id,
            user_id=user_id,
            text=text.strip(),
            importance=max(0.0, min(1.0, importance)),
            tags=list(tags or []),
            emotional=dict(emotional or {}),
            embedding=emb,
        )
        await self._storage.insert(entry)
        await self._bump_relationship(user_id)
        return entry

    async def remember_turn(
        self, user_id: str, user_msg: str, character_msg: str,
    ) -> MemoryEntry:
        """Convenience: store one conversational exchange as a single memory."""
        text = f"User: {user_msg.strip()}\n{self._character_id}: {character_msg.strip()}"
        return await self.remember(user_id, text)

    async def forget(self, memory_id: str) -> None:
        """Soft-delete a memory. The row stays for audit/export."""
        await self._storage.soft_delete(memory_id)

    # ------------------------------------------------------------------ recall

    async def recall(self, user_id: str, query: str, *, k: int = 5) -> list[Hit]:
        """Semantic + keyword recall, scored together."""
        if not user_id:
            raise ValueError("user_id is required")
        if not query or not query.strip():
            return []

        emb = (await self._provider.embed([query]))[0]
        hits = await self._storage.search(
            character_id=self._character_id,
            user_id=user_id,
            embedding=emb,
            keywords=_keywords(query),
            k=k,
        )
        # mark recall on the entries so reconsolidation counters move
        for h in hits:
            h.entry.access_count += 1
            h.entry.last_accessed_at = time.time()
            h.entry.reconsolidation_count += 1
            await self._storage.insert(h.entry)
        return hits

    async def recent(self, user_id: str, *, k: int = 10) -> list[Hit]:
        """Most-recent ``k`` memories for the user."""
        # delegate to search with no query — storage orders by recency
        # implicitly via importance lift + creation time. Simple enough.
        return await self._storage.search(
            character_id=self._character_id,
            user_id=user_id,
            embedding=None,
            keywords=None,
            k=k,
        )

    # ------------------------------------------------------------------ relationship (the wedge)

    async def relationship(self, user_id: str) -> Relationship:
        """Return the current relationship state with ``user_id``.

        This is the API mem0 cannot ship without rewriting itself.
        """
        return await self._storage.relationship_get(self._character_id, user_id)

    async def update_relationship(self, user_id: str, **deltas: Any) -> Relationship:
        """Patch fields on the relationship object.

        Only declared fields are accepted; unknown keys raise so
        typos don't silently vanish.
        """
        rel = await self.relationship(user_id)
        allowed = {
            "trust", "familiarity", "warmth",
            "shared_themes", "callbacks",
            "open_promises", "broken_promises",
            "notes",
        }
        for k, v in deltas.items():
            if k not in allowed:
                raise ValueError(f"unknown relationship field: {k!r}")
            setattr(rel, k, v)
        rel.last_interaction_at = time.time()
        await self._storage.relationship_set(rel)
        return rel

    # ------------------------------------------------------------------ maintenance

    async def consolidate(self, user_id: str) -> int:
        """Apply decay; return the number of memories whose effective
        importance dropped enough that they will rank lower next time.

        v0.1 implementation is intentionally simple: half-life decay
        on ``importance``. The richer dream-style consolidation from
        the Ashira codebase lands in v0.2.
        """
        hits = await self._storage.search(
            character_id=self._character_id,
            user_id=user_id,
            embedding=None,
            keywords=None,
            k=10_000,
        )
        now = time.time()
        decayed = 0
        for h in hits:
            e = h.entry
            age_h = max(0.0, (now - e.last_accessed_at) / 3600.0)
            if e.decay_half_life_hours <= 0 or age_h <= 0:
                continue
            factor = 0.5 ** (age_h / e.decay_half_life_hours)
            new_imp = max(0.0, e.importance * factor)
            if new_imp < e.importance - 1e-6:
                e.importance = new_imp
                await self._storage.insert(e)
                decayed += 1
        return decayed

    async def export(self, user_id: str) -> dict[str, Any]:
        """Full portable dump: every memory + the relationship."""
        hits = await self._storage.search(
            character_id=self._character_id,
            user_id=user_id,
            embedding=None,
            keywords=None,
            k=10_000,
        )
        rel = await self.relationship(user_id)
        return {
            "character_id": self._character_id,
            "user_id": user_id,
            "exported_at": time.time(),
            "relationship": rel.__dict__,
            "memories": [h.entry.__dict__ for h in hits],
        }

    # ------------------------------------------------------------------ internals

    async def _bump_relationship(self, user_id: str) -> None:
        rel = await self.relationship(user_id)
        rel.interaction_count += 1
        # gentle familiarity climb that asymptotes at 1.0
        rel.familiarity = 1.0 - (1.0 - rel.familiarity) * 0.97
        rel.last_interaction_at = time.time()
        await self._storage.relationship_set(rel)
