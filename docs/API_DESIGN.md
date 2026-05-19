# ashira-memory — API Design (v0.1 draft)

> The memory layer for AI **characters and companions**.
> Models relationships, not just facts. Local-first. OpenAI-optional.

This document is the contract. Code follows this; if a feature isn't
in here, it doesn't belong in v0.1.

---

## Design rules

1. **Public surface is tiny.** Target ≤ 15 public methods on `Memory`.
2. **Providers and storage are abstract.** Default to Ollama + SQLite. Drop-in
   for anything else.
3. **`character_id` is mandatory.** Cross-character contamination is impossible
   by construction (direct counter to mem0 #5121).
4. **Async-first.** All I/O is `async`. Sync wrappers come later.
5. **Type-clean.** `py.typed`, mypy strict, no `Any` in the public surface.
6. **Zero hidden network calls.** If you didn't configure a provider, nothing
   leaves the box. Privacy is the wedge.

---

## Mental model

| Concept | What it is | Why it exists |
|---|---|---|
| **Character** | The AI persona — Ashira, your D&D NPC, your tutor | Companion apps have many; mem0 has no such concept |
| **User** | The human talking to the character | Standard |
| **Episode** | One stored conversational moment | Things that happened |
| **Fact** | Extracted semantic claim | Things that are true |
| **Relationship** | First-class state between a (character, user) pair | The wedge |

A memory always belongs to exactly one `(character_id, user_id)` pair.
This is non-negotiable and enforced at the storage layer.

---

## Public API — `Memory` facade

```python
from ashira_memory import Memory

mem = Memory(
    character_id="ashira",
    provider=...,    # OllamaProvider() default
    storage=...,     # SQLiteStorage("./memory.db") default
)

# ---- storing ----
await mem.remember(user_id, text, *, importance=0.5, tags=None, emotional=None)
await mem.remember_turn(user_id, user_msg, character_msg)   # convenience
await mem.forget(memory_id)                                 # soft-delete

# ---- recalling ----
hits = await mem.recall(user_id, query, k=5)
hits = await mem.recent(user_id, k=10)
linked = await mem.related(memory_id, depth=1)

# ---- relationship state ----
rel = await mem.relationship(user_id)
# rel.trust, rel.familiarity, rel.shared_themes, rel.callbacks, rel.broken_promises
await mem.update_relationship(user_id, **deltas)

# ---- maintenance (run periodically; safe to call often) ----
await mem.consolidate(user_id)    # decay + merge weekly summaries
await mem.export(user_id)         # full portable dump
```

That's the entire v0.1 surface. 12 methods. Anything beyond this is v0.2.

---

## Provider protocol

```python
class Provider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def complete(self, prompt: str, *, json: bool = False) -> str: ...
```

Two methods. That's it. `OllamaProvider` and `OpenAIProvider` both
implement this. Anyone can write their own in 30 lines.

---

## Storage protocol

```python
class Storage(Protocol):
    async def insert(self, entry: MemoryEntry) -> None: ...
    async def get(self, memory_id: str) -> MemoryEntry | None: ...
    async def search(
        self, *, character_id: str, user_id: str,
        embedding: list[float] | None = None,
        keywords: list[str] | None = None,
        k: int = 5,
    ) -> list[Hit]: ...
    async def soft_delete(self, memory_id: str) -> None: ...
    async def relationship_get(self, character_id: str, user_id: str) -> Relationship: ...
    async def relationship_set(self, rel: Relationship) -> None: ...
```

Six methods. SQLite ships in core. Postgres/pgvector/Chroma are
separate packages.

---

## Why this beats mem0 for companion apps

| Pain point | mem0 | ashira-memory |
|---|---|---|
| Cross-character contamination (#5121) | Open bug | Impossible by construction |
| Cloud-first default | OpenAI required | Ollama + SQLite default; OpenAI optional |
| Privacy guarantees for intimate companion data | Cloud-by-default | Zero hidden network calls |
| Relationship as first-class object | No equivalent | `mem.relationship(user_id)` |
| Local-only deployment story | Possible but fiddly | One line, zero config |

---

## What v0.1 will NOT have

- Graph memory (mem0 has it; we don't need it for companions)
- Hosted SaaS (Phase 4)
- Multi-agent orchestration (out of scope)
- Anything that requires an external service to boot

Keep the surface small. Win the wedge. Then expand.
