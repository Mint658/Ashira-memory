"""LLM + embedding providers.

``OllamaProvider`` ships with core. ``OpenAIProvider`` requires the
``[openai]`` extra and is lazy-imported so missing the dep doesn't
break ``from ashira_memory.providers import OllamaProvider``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ollama import OllamaProvider

if TYPE_CHECKING:
    from .openai import OpenAIProvider

__all__ = ["OllamaProvider", "OpenAIProvider"]


def __getattr__(name: str) -> Any:
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider as _OpenAIProvider

        return _OpenAIProvider
    raise AttributeError(f"module 'ashira_memory.providers' has no attribute {name!r}")
