"""Provider-agnostic LLM interface (same pattern as Sift, on purpose).

Pick the backend with ``YARDSTICK_LLM_PROVIDER`` ("anthropic" | "openai" | "offline"). Never import a
vendor SDK outside its adapter. When a schema is passed, return a dict conforming to it; otherwise return
text. The offline client keeps tests and keyless runs working end to end.
"""

from __future__ import annotations

import os
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None
    ) -> dict[str, Any] | str:
        ...


class AnthropicAdapter:
    """Worker: implement via the anthropic SDK (tool/structured output when schema is given)."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    def complete(self, messages, schema=None):
        raise NotImplementedError


class OpenAIAdapter:
    """Worker: implement via the openai SDK (response_format / function calling)."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model

    def complete(self, messages, schema=None):
        raise NotImplementedError


class OfflineClient:
    """Deterministic, no-network client for tests and keyless runs.

    Worker: return schema-valid stub output so the whole pipeline runs without an API key.
    """

    def complete(self, messages, schema=None):
        raise NotImplementedError


def get_llm_client() -> LLMClient:
    provider = os.environ.get("YARDSTICK_LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider == "openai":
        return OpenAIAdapter()
    if provider == "offline":
        return OfflineClient()
    raise ValueError(f"Unknown YARDSTICK_LLM_PROVIDER: {provider!r}")
