"""Real Ollama end-to-end test. Requires Ollama running locally."""

import asyncio
import tempfile
from pathlib import Path

from ashira_memory import Memory
from ashira_memory.providers import OllamaProvider
from ashira_memory.storage import SQLiteStorage


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "real.db"
        mem = Memory(
            "ashira",
            provider=OllamaProvider(completion_model="llama3.3:latest"),
            storage=SQLiteStorage(db_path),
        )

        print("=== Storing 3 memories about Alice ===")
        await mem.remember("alice", "I'm allergic to peanuts and tree nuts")
        await mem.remember("alice", "I love Studio Ghibli, especially Spirited Away")
        await mem.remember("alice", "I work as a software engineer on memory systems")
        print("  stored.\n")

        print("=== Recall: 'what food should I avoid?' ===")
        hits = await mem.recall("alice", "what food should I avoid?", k=3)
        for h in hits:
            print(f"  [score={h.score:.3f}] {h.entry.text}")
            print(f"    reason: {h.reason}")
        print()

        print("=== Recall: 'tell me about her work' ===")
        hits = await mem.recall("alice", "tell me about her work", k=3)
        for h in hits:
            print(f"  [score={h.score:.3f}] {h.entry.text}")
        print()

        print("=== Relationship state ===")
        await mem.update_relationship(
            "alice",
            trust=0.7,
            callbacks=["the rubber duck joke"],
            open_promises=["I'll remember your birthday"],
        )
        rel = await mem.relationship("alice")
        print(f"  interactions: {rel.interaction_count}")
        print(f"  familiarity:  {rel.familiarity:.3f}")
        print(f"  trust:        {rel.trust}")
        print(f"  callbacks:    {rel.callbacks}")
        print(f"  promises:     {rel.open_promises}")
        print()

        print("=== Embedding sanity check ===")
        # confirm embeddings actually got stored
        recent = await mem.recent("alice", k=3)
        for h in recent:
            entry = h.entry
            emb_len = len(entry.embedding) if entry.embedding else 0
            print(f"  '{entry.text[:40]}...' -> embedding dim = {emb_len}")

        print("\n=== ALL GREEN ===")


if __name__ == "__main__":
    asyncio.run(main())
