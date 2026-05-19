"""ashira-memory — relational memory for AI characters and companions.

>>> from ashira_memory import Memory
>>> mem = Memory("ashira")
>>> # await mem.remember("alice", "I'm allergic to peanuts")
>>> # hits = await mem.recall("alice", "food I should avoid")
"""

from .memory import Memory
from .protocols import Provider, Storage
from .types import Hit, MemoryEntry, Relationship

__all__ = [
    "Memory",
    "MemoryEntry",
    "Hit",
    "Relationship",
    "Provider",
    "Storage",
]

__version__ = "0.1.0"
