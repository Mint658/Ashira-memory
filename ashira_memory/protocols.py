"""Provider and Storage protocols.

Two protocols. That's the entire extensibility surface of v0.1.
If you want to plug in your own LLM/embedding stack or a different
database, you implement these and inject them at construction time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Hit, MemoryEntry, Relationship


@runtime_checkable
class Provider(Protocol):
    """LLM + embeddings. Two methods, intentionally."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    async def complete(self, prompt: str, *, json: bool = False) -> str:
        """Return a single completion. When ``json=True`` the provider
        must return a string that's a valid JSON document."""


@runtime_checkable
class Storage(Protocol):
    """Persistent backend. SQLite ships in core; everything else
    (Postgres, pgvector, Chroma, Qdrant) is a separate package."""

    async def insert(self, entry: MemoryEntry) -> None: ...

    async def get(self, memory_id: str) -> MemoryEntry | None: ...

    async def search(
        self,
        *,
        character_id: str,
        user_id: str,
        embedding: list[float] | None = None,
        keywords: list[str] | None = None,
        k: int = 5,
    ) -> list[Hit]: ...

    async def soft_delete(self, memory_id: str) -> None: ...

    async def relationship_get(
        self, character_id: str, user_id: str
    ) -> Relationship: ...

    async def relationship_set(self, rel: Relationship) -> None: ...
