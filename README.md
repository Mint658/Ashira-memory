# ashira-memory

> **The memory layer for AI characters and companions.**
> Models relationships, not just facts. Local-first. OpenAI-optional.

[![PyPI](https://img.shields.io/pypi/v/ashira-memory.svg)](https://pypi.org/project/ashira-memory/)
[![Python](https://img.shields.io/pypi/pyversions/ashira-memory.svg)](https://pypi.org/project/ashira-memory/)
[![License](https://img.shields.io/pypi/l/ashira-memory.svg)](https://github.com/Mint658/Ashira-memory/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/ashira-memory.svg)](https://pypi.org/project/ashira-memory/)
[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)](https://github.com/Mint658/Ashira-memory/tree/main/tests)

---

`ashira-memory` is built for one specific job: giving an **AI character or
companion** a memory that feels personal. Not "I retrieved chunk 4 from the
vector store" — *"I remember you mentioned your sister last week, how is she?"*

Existing memory libraries (mem0, Letta, Zep) are built for agent / dev-tool
use cases. Their primitives are `User → Session → Agent`. None of them
model a *relationship between a character and a user* as a first-class object.

This one does.

```bash
pip install ashira-memory
```

## What you get out of the box

```python
import asyncio
from ashira_memory import Memory

async def demo():
    mem = Memory("ashira")  # local Ollama + local SQLite. zero config.

    await mem.remember("alice", "I'm allergic to peanuts")
    await mem.remember("alice", "I love Studio Ghibli — Spirited Away is my favourite")

    hits = await mem.recall("alice", "what food should I avoid?", k=3)
    for h in hits:
        print(h.score, h.entry.text)

    # Relationship is a first-class object — the bit nobody else has
    rel = await mem.relationship("alice")
    print(f"interactions: {rel.interaction_count}, familiarity: {rel.familiarity:.2f}")

    # Track in-jokes, open promises, broken promises, shared themes
    await mem.update_relationship(
        "alice",
        callbacks=["the joke about the rubber duck"],
        open_promises=["I'll remember your birthday"],
    )

asyncio.run(demo())
```

That's the entire quickstart.

## Why this, not mem0?

| | mem0 | ashira-memory |
|---|---|---|
| Built for | AI agents, dev tools | AI characters, companions |
| Default LLM | OpenAI `gpt-5-mini` | Whatever Ollama you've got |
| Default embedding | OpenAI `text-embedding-3-small` | `nomic-embed-text` (local) |
| Cross-character isolation | Application's problem (see [#5121](https://github.com/mem0ai/mem0/issues/5121)) | Impossible by construction |
| Relationship as object | No equivalent | `mem.relationship(user_id)` |
| Hidden network calls | Yes (by default) | None |
| Lines of dependencies | Big | `httpx`. That's it. |

We are not trying to beat mem0 at being mem0. We're winning a market they left.

## Install

```bash
pip install ashira-memory          # core + Ollama
pip install ashira-memory[openai]  # add OpenAI provider
```

Then either:

```bash
# local (default)
ollama pull nomic-embed-text
ollama pull llama3.2
```

or:

```python
# cloud — bring your own API key
import os
from ashira_memory import Memory
from ashira_memory.providers import OpenAIProvider

os.environ["OPENAI_API_KEY"] = "sk-..."
mem = Memory("ashira", provider=OpenAIProvider())
```

## The mental model in 60 seconds

- **Character** — the AI persona. You set this once per `Memory` instance.
- **User** — the human. Every method takes `user_id`. Always.
- **Episode** — one stored memory (`remember`, `recall`, `forget`).
- **Relationship** — first-class state between `(character, user)`: trust,
  familiarity, warmth, in-jokes, open promises, broken promises.

Every memory belongs to exactly one `(character_id, user_id)` pair. The
storage layer enforces this. Two characters sharing one database **cannot**
see each other's memories, no matter what your application code does.

## API surface

12 methods. The whole library:

```python
# storing
await mem.remember(user_id, text, *, importance=0.5, tags=None, emotional=None)
await mem.remember_turn(user_id, user_msg, character_msg)
await mem.forget(memory_id)

# recalling
await mem.recall(user_id, query, k=5)
await mem.recent(user_id, k=10)

# relationship — the wedge
await mem.relationship(user_id)
await mem.update_relationship(user_id, trust=..., callbacks=..., open_promises=...)

# maintenance
await mem.consolidate(user_id)
await mem.export(user_id)
```

That's it. If you're reaching for something that's not here, open an issue.

## Status

Alpha. The API is stable enough to build against; expect minor breaking changes
through `0.1.x`. Tests pass on Python 3.10–3.12.

## License

Apache-2.0.

## Roadmap

- `0.1` — OpenAI provider, Postgres/pgvector adapter
- `0.2` — synthesis (cross-memory reflections), dream-style consolidation
- `0.3` — `RelMemBench` public benchmark + scorecards against mem0/Letta/Zep
- `0.4` — agent skills for Claude Code / Cursor / Codex
