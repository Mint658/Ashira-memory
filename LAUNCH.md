# Launch posts — copy/paste ready

These are drafts for each channel. Edit lightly, then post. Don't post all at once — stagger by a few hours so you can respond to comments.

**Order I'd suggest:**
1. r/LocalLLaMA (most receptive crowd, low-stakes warm-up)
2. r/Python (broader, gives you karma)
3. Hacker News (peak audience, post around 8-10am US Pacific = ~1-3am AEST)
4. dev.to / X / LinkedIn (longer tail)

After posting, **stay on the thread for 2 hours** to answer questions. That's where conversions happen.

---

## 1. r/LocalLLaMA

**Title:** `[Project] ashira-memory — local-first relational memory for AI characters (sqlite + ollama, no cloud required)`

**Body:**

I built a small Python library for a problem I kept hitting with mem0/Letta/Zep: they're designed for **agents**, but I needed memory for an **AI character** — something that models the *relationship* between the character and the user, not just a flat list of facts.

`ashira-memory` is what I wish existed:

- **Local-first by default** — SQLite + Ollama (`nomic-embed-text` + `llama3.2`). No API keys, no network calls.
- **`(character_id, user_id)` isolation enforced at the storage layer.** Two characters sharing one database physically cannot see each other's memories. (mem0 has [a known bug](https://github.com/mem0ai/mem0/issues/5121) where this leaks.)
- **`Relationship` is a first-class object.** Trust, familiarity, in-jokes, open promises, broken promises — not just "facts about user."
- **One dependency:** `httpx`. OpenAI is opt-in via `pip install ashira-memory[openai]`.

```python
from ashira_memory import Memory

mem = Memory("ashira")
await mem.remember("alice", "I'm allergic to peanuts")
hits = await mem.recall("alice", "what food should I avoid?")

rel = await mem.relationship("alice")
print(rel.familiarity, rel.open_promises)
```

12 methods total. The whole API.

**v0.1.0 just hit PyPI:** https://pypi.org/project/ashira-memory/
**Code + roadmap:** https://github.com/Mint658/Ashira-memory

Honest about the status: alpha. Tests pass on 3.10–3.12. API stable enough to build against, expect minor breaks through 0.1.x. Roadmap includes `RelMemBench` to publicly benchmark against mem0/Letta/Zep on relationship-modeling tasks.

Built this while building a much larger AI companion project. Splitting the memory layer out felt right.

Feedback / issues / "you missed this prior art" all welcome.

---

## 2. r/Python

**Title:** `Show: ashira-memory — a tiny memory library for AI characters (one dep, local-first)`

**Body:**

Released v0.1.0 of [`ashira-memory`](https://pypi.org/project/ashira-memory/) — a Python library for giving AI characters/companions persistent memory.

**Design choices that might be interesting:**

- **One dependency** (`httpx`). Optional `openai` extra for the cloud provider.
- **Provider protocol** — swap embeddings/completions without touching the rest. Default is Ollama (local), `OpenAIProvider` is opt-in.
- **Storage protocol** — currently SQLite with WAL mode + `(character_id, user_id)` scoping at every query. Postgres adapter is on the 0.1.x roadmap.
- **Async-first** — every public method is `await`-able. No sync shim layer to maintain.
- **12 public methods, full stop.** If you reach for something that isn't there, that's an issue, not an extension point.

```python
from ashira_memory import Memory
from ashira_memory.providers import OpenAIProvider  # optional

mem = Memory("ashira", provider=OpenAIProvider())  # or omit for Ollama
await mem.remember("alice", "I'm allergic to peanuts")
hits = await mem.recall("alice", "what food should I avoid?")
```

PyPI: https://pypi.org/project/ashira-memory/
GitHub: https://github.com/Mint658/Ashira-memory
License: Apache-2.0

Built it because mem0/Letta/Zep are great for agent use-cases but felt heavy for "give my LLM character a memory." Different problem, different shape. Comparison table in the README.

15 tests passing, 3.10–3.12. Genuine alpha — feedback welcome.

---

## 3. Hacker News

**Title:** `Show HN: Ashira-memory – Local-first relational memory for AI characters`

(HN title limit is 80 chars. The above is 65.)

**Text post (HN allows ~2000 chars for Show HN):**

Hi HN — I built `ashira-memory`, a small Python library that gives AI characters (companions, NPCs, persistent chatbot personas) a memory that models *relationships*, not just facts.

The gap I kept hitting: mem0, Letta, and Zep are great for **agents** — their primitives are `User → Session → Agent`. But for an AI **character**, the most important object is the relationship between *that character* and *that user* — trust level, in-jokes, open promises, things you've already explained. None of the existing libraries model this as a first-class object.

What ashira-memory does differently:

1. `Relationship(character_id, user_id)` is a first-class object with `trust`, `familiarity`, `warmth`, `callbacks`, `open_promises`, `broken_promises`.

2. `(character_id, user_id)` scoping is enforced at the storage layer — every SQL query. Two characters sharing one database physically cannot see each other's memories. (Cross-user leakage is a known mem0 issue: github.com/mem0ai/mem0/issues/5121.)

3. Local-first by default: SQLite + Ollama. Zero API keys, zero network calls. OpenAI is opt-in via `pip install ashira-memory[openai]`.

4. One real dependency (`httpx`). 12 public methods. Apache-2.0.

Quickstart:

    from ashira_memory import Memory
    mem = Memory("ashira")
    await mem.remember("alice", "I'm allergic to peanuts")
    hits = await mem.recall("alice", "what food should I avoid?")
    rel = await mem.relationship("alice")

Status: alpha (v0.1.0), 15 tests passing on 3.10–3.12. API stable enough to build against, expect minor breaks through 0.1.x. Roadmap includes a public benchmark (`RelMemBench`) to score against mem0/Letta/Zep on relationship-modeling tasks specifically.

I'm building a larger AI companion project and splitting the memory layer out felt overdue. Genuinely curious whether this resonates with people working on character/companion-style products — and where it falls short.

PyPI: https://pypi.org/project/ashira-memory/
GitHub: https://github.com/Mint658/Ashira-memory

---

## 4. X / Twitter (thread)

**Tweet 1:**
> Shipped a thing.
>
> `ashira-memory` — a Python library for giving AI characters persistent, relationship-aware memory. Local-first (SQLite + Ollama), one real dependency, 12 methods total.
>
> pip install ashira-memory
>
> 🧵 why this exists ↓

**Tweet 2:**
> mem0/Letta/Zep are great for **agents**. Their world is `User → Session → Agent`.
>
> But for an AI **character** (companion, NPC, persistent persona) the most important object isn't a session — it's the *relationship* between that character and that user.
>
> Nobody models that as a first-class thing.

**Tweet 3:**
> ```python
> rel = await mem.relationship("alice")
> rel.trust        # 0.0 – 1.0
> rel.familiarity  # 0.0 – 1.0
> rel.callbacks    # in-jokes you've shared
> rel.open_promises
> rel.broken_promises
> ```
>
> A character that *remembers it owes you something* feels alive in a way that flat fact-recall doesn't.

**Tweet 4:**
> Other design choices:
>
> • `(character_id, user_id)` isolation enforced at storage layer. Two characters sharing one DB physically cannot leak. (mem0 has a known bug here: github.com/mem0ai/mem0/issues/5121)
>
> • SQLite + Ollama default. Zero cloud. OpenAI opt-in.
>
> • One dep: httpx.

**Tweet 5:**
> v0.1.0 is live on PyPI today. Apache-2.0.
>
> Alpha but tested (15/15 passing on 3.10–3.12). API stable enough to build against.
>
> Code: github.com/Mint658/Ashira-memory
> PyPI: pypi.org/project/ashira-memory/
>
> If you're building AI companions or characters, I want your feedback.

---

## 5. LinkedIn (single post)

I shipped my first open-source library today.

**ashira-memory** — a Python library that gives AI characters persistent, relationship-aware memory.

The gap: existing memory libraries (mem0, Letta, Zep) are built for AI **agents**. They model `User → Session → Agent`. But when you're building an AI **character** — a companion, an NPC, a persistent persona — the most important object isn't a session, it's the relationship between *that character* and *that user*. Trust. Familiarity. In-jokes. Promises you've made.

None of the existing libraries model this as a first-class concept. So I built one that does.

Design choices I'm proud of:
• Local-first by default (SQLite + Ollama). Zero API keys required.
• `(character_id, user_id)` isolation enforced at the storage layer.
• One real dependency.
• 12 public methods. Full stop.
• Apache-2.0.

It's part of a larger AI companion project I'm building, and splitting the memory layer out into its own thing felt overdue.

v0.1.0 is on PyPI today: pypi.org/project/ashira-memory/
Code: github.com/Mint658/Ashira-memory

If you build with this — or break it, or hate it — I want to hear from you.

#Python #OpenSource #AI #MachineLearning

---

## Post-launch checklist

- [ ] PyPI live (✅ already done — v0.1.0)
- [ ] GitHub repo description updated to one-liner
- [ ] GitHub topics: `memory`, `llm`, `ai-agents`, `ai-companion`, `ollama`, `openai`, `sqlite`, `python`
- [ ] Pin v0.1.0 release on GitHub
- [ ] Post to r/LocalLLaMA
- [ ] (2-4 hrs later) Post to r/Python
- [ ] (next morning, US time) Post to Hacker News
- [ ] X thread
- [ ] LinkedIn post
- [ ] Reply to every comment within first 4 hours of each post
- [ ] Track installs at https://pypistats.org/packages/ashira-memory after 72hrs
