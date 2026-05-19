"""OpenAI provider — opt-in. Bring your own ``OPENAI_API_KEY``.

Install with ``pip install 'ashira-memory[openai]'``.

This provider deliberately uses the OpenAI SDK (rather than raw httpx)
because the SDK handles streaming, retries, and Azure/compatible
endpoints in ways we don't want to reimplement.
"""

from __future__ import annotations

import os
from typing import Any


class OpenAIProvider:
    """OpenAI embeddings + completions, configurable per-call.

    Defaults (May 2026):

    - embedding model: ``text-embedding-3-small`` (1536 dims, cheap, strong)
    - completion model: ``gpt-4o-mini`` (cheap, JSON-mode capable)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        completion_model: str = "gpt-4o-mini",
        base_url: str | None = None,
        organization: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            # caller supplied a pre-configured AsyncOpenAI (e.g. for Azure,
            # an OpenAI-compatible proxy, or a mock in tests)
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIProvider requires the 'openai' extra. "
                    "Install with: pip install 'ashira-memory[openai]'"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url,
                organization=organization,
            )
        self._embed_model = embedding_model
        self._chat_model = completion_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        # OpenAI returns results in the input order, but be explicit
        return [list(d.embedding) for d in resp.data]

    async def complete(self, prompt: str, *, json: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self._chat_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
