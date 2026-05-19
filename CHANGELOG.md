# Changelog

All notable changes to `ashira-memory` will be documented here. This project
follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-19

### Added
- **`OpenAIProvider`** — opt-in OpenAI backend. Install with
  `pip install ashira-memory[openai]`, then
  `Memory("ashira", provider=OpenAIProvider())`. Supports Azure / OpenAI-
  compatible proxies via the `base_url` and `client` parameters.
- Friendly `ImportError` when the `openai` extra is missing.
- 7 new tests covering the OpenAI provider with a fully-mocked client
  (no API key or network required).

### Changed
- `providers.__init__` now lazy-imports `OpenAIProvider` via `__getattr__`,
  so importing `OllamaProvider` doesn't fail when the OpenAI SDK is absent.

## [0.0.1] — 2026-05-19

Initial release.

- `Memory` facade with 12-method public surface.
- First-class `Relationship` object for AI companions.
- `OllamaProvider` (local-first default).
- `SQLiteStorage` with `(character_id, user_id)` isolation enforced at
  every query.
- 8 smoke tests + verified end-to-end against real Ollama
  (`nomic-embed-text` + `llama3.3`).
