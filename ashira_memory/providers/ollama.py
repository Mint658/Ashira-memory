"""Ollama provider — the default. Zero hidden network calls.

If a developer doesn't override this, embeddings and completions
go to ``localhost:11434`` and nowhere else. That's the entire
privacy story in one paragraph.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaProvider:
    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        completion_model: str = "llama3.2",
        timeout: float = 60.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._embed_model = embedding_model
        self._chat_model = completion_model
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for t in texts:
                r = await client.post(
                    f"{self._host}/api/embeddings",
                    json={"model": self._embed_model, "prompt": t},
                )
                r.raise_for_status()
                data = r.json()
                out.append(list(data.get("embedding") or []))
        return out

    async def complete(self, prompt: str, *, json: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self._chat_model,
            "prompt": prompt,
            "stream": False,
        }
        if json:
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._host}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        return str(data.get("response", ""))
