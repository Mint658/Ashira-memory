"""End-to-end smoke test: store, recall, relationship, isolation."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from ashira_memory import Memory
from ashira_memory.storage.sqlite import SQLiteStorage


class _FakeProvider:
    """Deterministic embedder so tests don't need a model server.

    Hashes each input into a small fixed-width vector — enough for
    cosine similarity to be meaningfully ordered for identical /
    overlapping text without bringing in an embedding model.
    """

    DIM = 16

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            vec = [b / 255.0 for b in h[: self.DIM]]
            out.append(vec)
        return out

    async def complete(self, prompt: str, *, json: bool = False) -> str:
        return "{}" if json else ""


@pytest.fixture
def mem(tmp_path: Path) -> Memory:
    storage = SQLiteStorage(tmp_path / "mem.db")
    return Memory("ashira", provider=_FakeProvider(), storage=storage)


async def test_remember_and_recall(mem: Memory) -> None:
    await mem.remember("alice", "I'm allergic to peanuts")
    await mem.remember("alice", "I love anime, especially Studio Ghibli")
    hits = await mem.recall("alice", "I love anime, especially Studio Ghibli", k=3)
    assert hits, "expected at least one hit"
    assert "anime" in hits[0].entry.text.lower()


async def test_character_isolation_by_construction(tmp_path: Path) -> None:
    """Two characters sharing one DB must never see each other's memories.

    This is the wedge against mem0 bug #5121.
    """
    storage = SQLiteStorage(tmp_path / "shared.db")
    mem_a = Memory("ashira", provider=_FakeProvider(), storage=storage)
    mem_b = Memory("npc_villain", provider=_FakeProvider(), storage=storage)

    await mem_a.remember("alice", "Ashira knows Alice loves jazz")
    await mem_b.remember("alice", "the villain knows Alice fears the dark")

    hits_a = await mem_a.recall("alice", "what does she love", k=5)
    hits_b = await mem_b.recall("alice", "what does she fear", k=5)

    a_texts = " ".join(h.entry.text for h in hits_a)
    b_texts = " ".join(h.entry.text for h in hits_b)

    assert "jazz" in a_texts and "villain" not in a_texts
    assert "fear" in b_texts and "Ashira" not in b_texts


async def test_user_isolation(mem: Memory) -> None:
    await mem.remember("alice", "Alice loves ramen")
    await mem.remember("bob", "Bob loves pizza")
    hits = await mem.recall("alice", "favourite food", k=5)
    joined = " ".join(h.entry.text for h in hits)
    assert "Alice" in joined and "Bob" not in joined


async def test_relationship_first_class(mem: Memory) -> None:
    rel = await mem.relationship("alice")
    assert rel.interaction_count == 0
    await mem.remember("alice", "first chat")
    rel = await mem.relationship("alice")
    assert rel.interaction_count == 1
    assert rel.familiarity > 0.0


async def test_update_relationship_rejects_typos(mem: Memory) -> None:
    with pytest.raises(ValueError):
        await mem.update_relationship("alice", trsut=0.9)  # typo


async def test_update_relationship_persists(mem: Memory) -> None:
    await mem.update_relationship(
        "alice",
        trust=0.8,
        callbacks=["the joke about the rubber duck"],
        open_promises=["I'll remember your birthday"],
    )
    rel = await mem.relationship("alice")
    assert rel.trust == 0.8
    assert "rubber duck" in rel.callbacks[0]
    assert rel.open_promises == ["I'll remember your birthday"]


async def test_forget_hides_memory(mem: Memory) -> None:
    e = await mem.remember("alice", "secret I want forgotten")
    await mem.forget(e.id)
    hits = await mem.recall("alice", "secret", k=5)
    assert all("forgotten" not in h.entry.text for h in hits)


async def test_export_round_trip(mem: Memory) -> None:
    await mem.remember("alice", "fact one")
    await mem.remember("alice", "fact two")
    dump = await mem.export("alice")
    assert dump["character_id"] == "ashira"
    assert dump["user_id"] == "alice"
    assert len(dump["memories"]) == 2
    assert "relationship" in dump
