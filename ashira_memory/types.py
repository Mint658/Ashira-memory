"""Public types for ashira-memory.

Deliberately small. If a field isn't useful to a companion-app
developer in the first 30 minutes, it doesn't belong here.

The richness lives in two places only:

* :class:`MemoryEntry` carries the cognitive depth (importance,
  emotional context, reconsolidation, decay) that separates this
  library from "vector DB with timestamps".
* :class:`Relationship` is the *wedge* — a first-class object that
  tracks the state between a (character, user) pair. mem0 has no
  equivalent.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single thing the character remembers about a user.

    Every entry is scoped to exactly one ``(character_id, user_id)``
    pair. This is enforced at the storage layer so cross-character
    contamination is impossible by construction (direct counter to
    mem0 bug #5121).
    """

    character_id: str
    user_id: str
    text: str

    # Identity / timing
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0

    # Cognitive depth — the bit that makes recall feel human
    importance: float = 0.5            # 0..1, drives decay & ranking
    salience: float = 0.5              # attention-weighted memorability
    reconsolidation_count: int = 0     # how many times the memory was reinforced
    decay_half_life_hours: float = 72.0
    trust: float = 1.0                 # confidence in the content

    # Relational / affective
    tags: list[str] = field(default_factory=list)
    emotional: dict[str, float] = field(default_factory=dict)

    # Retrieval
    embedding: list[float] | None = None

    # Bookkeeping
    deleted: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """A single retrieval result."""

    entry: MemoryEntry
    score: float
    reason: str = ""  # short human-readable explanation


# ---------------------------------------------------------------------------
# Relationship — the wedge
# ---------------------------------------------------------------------------


@dataclass
class Relationship:
    """First-class state between a character and a user.

    This is what mem0 cannot model. A character's relationship with
    a user is not just a bag of facts — it has texture: trust that
    grows or breaks, themes the two keep returning to, jokes only
    they share, promises the character made.

    Companion-app developers care about this. Agent-tool developers
    don't. That's why mem0 won't build it.
    """

    character_id: str
    user_id: str

    # State that evolves
    trust: float = 0.5            # 0..1
    familiarity: float = 0.0      # 0..1, grows with interaction count
    warmth: float = 0.5           # 0..1, current affective baseline
    interaction_count: int = 0
    last_interaction_at: float = field(default_factory=time.time)

    # Texture — the bits that make recall feel personal
    shared_themes: list[str] = field(default_factory=list)
    callbacks: list[str] = field(default_factory=list)   # in-jokes, references
    open_promises: list[str] = field(default_factory=list)
    broken_promises: list[str] = field(default_factory=list)

    # Free-form notes the character chose to keep
    notes: dict[str, Any] = field(default_factory=dict)
