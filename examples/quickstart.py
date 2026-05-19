"""5-minute demo. Requires Ollama running locally with
`nomic-embed-text` and `llama3.2`."""

import asyncio

from ashira_memory import Memory


async def main() -> None:
    mem = Memory("ashira")

    await mem.remember("alice", "I'm allergic to peanuts")
    await mem.remember("alice", "I love Studio Ghibli, especially Spirited Away")
    await mem.remember("alice", "I'm a software engineer working on memory systems")

    hits = await mem.recall("alice", "what food should I avoid?", k=3)
    print("=== recall: 'what food should I avoid?' ===")
    for h in hits:
        print(f"  [{h.score:.2f}] {h.entry.text}  ({h.reason})")

    await mem.update_relationship(
        "alice",
        callbacks=["the joke about the rubber duck"],
        open_promises=["I will remember your birthday"],
    )

    rel = await mem.relationship("alice")
    print("\n=== relationship ===")
    print(f"  interactions: {rel.interaction_count}")
    print(f"  familiarity:  {rel.familiarity:.2f}")
    print(f"  callbacks:    {rel.callbacks}")
    print(f"  promises:     {rel.open_promises}")


if __name__ == "__main__":
    asyncio.run(main())
