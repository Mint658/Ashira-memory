"""OpenAIProvider — tests use an injected fake client so no API key
or network is required."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest

from ashira_memory import Memory
from ashira_memory.providers import OpenAIProvider
from ashira_memory.storage import SQLiteStorage


# ---------- mock OpenAI client ----------------------------------------------


@dataclass
class _EmbeddingItem:
    embedding: list[float]


@dataclass
class _EmbeddingResponse:
    data: list[_EmbeddingItem]


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _ChatResponse:
    choices: list[_Choice]


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model: str, input: list[str]) -> _EmbeddingResponse:
        self.calls.append({"model": model, "input": input})
        data = []
        for t in input:
            h = hashlib.sha256(t.lower().encode()).digest()
            data.append(_EmbeddingItem(embedding=[b / 255.0 for b in h[:16]]))
        return _EmbeddingResponse(data=data)


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _ChatResponse:
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if kwargs.get("response_format", {}).get("type") == "json_object":
            return _ChatResponse(choices=[_Choice(message=_Message(content='{"ok": true}'))])
        return _ChatResponse(choices=[_Choice(message=_Message(content=f"echo: {prompt[:30]}"))])


class _FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions()})()


# ---------- tests -----------------------------------------------------------


async def test_embed_returns_one_vector_per_input() -> None:
    fake = _FakeOpenAI()
    p = OpenAIProvider(client=fake)
    vecs = await p.embed(["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == 16 for v in vecs)
    # input order preserved
    assert vecs[0] != vecs[1]


async def test_embed_empty_input_no_call() -> None:
    fake = _FakeOpenAI()
    p = OpenAIProvider(client=fake)
    out = await p.embed([])
    assert out == []
    assert fake.embeddings.calls == []


async def test_complete_passes_model_and_messages() -> None:
    fake = _FakeOpenAI()
    p = OpenAIProvider(client=fake, completion_model="gpt-4o-mini")
    out = await p.complete("hi there")
    assert out.startswith("echo: hi there")
    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["messages"][0]["content"] == "hi there"
    assert "response_format" not in call


async def test_complete_json_mode_sets_response_format() -> None:
    fake = _FakeOpenAI()
    p = OpenAIProvider(client=fake)
    out = await p.complete("give me json", json=True)
    assert out == '{"ok": true}'
    call = fake.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}


async def test_provider_satisfies_protocol() -> None:
    from ashira_memory.protocols import Provider

    fake = _FakeOpenAI()
    p = OpenAIProvider(client=fake)
    assert isinstance(p, Provider)


async def test_end_to_end_with_memory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The whole library should work with the OpenAI provider injected."""
    fake = _FakeOpenAI()
    provider = OpenAIProvider(client=fake)
    storage = SQLiteStorage(tmp_path / "openai.db")
    mem = Memory("ashira", provider=provider, storage=storage)

    await mem.remember("alice", "I'm allergic to peanuts")
    await mem.remember("alice", "I love anime")

    hits = await mem.recall("alice", "anime favourites", k=2)
    assert hits
    assert any("anime" in h.entry.text for h in hits)


async def test_missing_openai_package_raises_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If openai isn't installed, instantiating without a client gives a
    useful pip-install hint, not a bare ImportError stack trace."""
    import builtins

    real_import = builtins.__import__

    def _fail_openai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_openai)
    with pytest.raises(ImportError, match=r"ashira-memory\[openai\]"):
        OpenAIProvider()
