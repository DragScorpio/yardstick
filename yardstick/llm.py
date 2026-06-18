"""Provider-agnostic LLM interface (same pattern as Sift, on purpose).

Rule (initiative-wide): never import a vendor SDK outside its adapter. Choose the backend with
``YARDSTICK_LLM_PROVIDER`` ("anthropic" | "openai" | "offline"); the default ``auto`` uses Anthropic if
``ANTHROPIC_API_KEY`` is set, else OpenAI if ``OPENAI_API_KEY`` is set, else the deterministic
:class:`OfflineClient`. When a schema is passed, an adapter returns a dict conforming to it; otherwise it
returns text. Responses are cached on disk so re-runs are free and ``run`` is reproducible.

The offline client is a real baseline, not a stub that peeks at the answer. As an *answerer* it sees only
the question and makes a knowledge-free guess (so an offline leaderboard is honestly weak). As a *judge* it
sees the reference and candidate (which the prompt legitimately provides) and decides correctness with a
token-F1 threshold — a defensible heuristic judge that genuinely tolerates paraphrase better than strict
exact match. Plug a real provider and the LLM judge replaces it; the eval scores whichever judge runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Protocol

from .metrics import token_f1

DEFAULT_CACHE_DIR = "data/.llm_cache"
# The offline client reads a fenced JSON payload the prompt builders embed for it; real models read the
# surrounding natural language and ignore the fence.
_PAYLOAD_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
# An answer is judged correct offline when its token-F1 against the reference clears this threshold.
OFFLINE_JUDGE_F1_THRESHOLD = 0.5


class LLMClient(Protocol):
    """The only contract the rest of Yardstick depends on."""

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None
    ) -> dict[str, Any] | str: ...


# --------------------------------------------------------------------------- response cache


def _cache_key(provider: str, model: str, messages: list[dict], schema: dict | None) -> str:
    """Hash the whole request into one stable filename, so identical calls reuse the same cached answer."""
    blob = json.dumps(
        {"provider": provider, "model": model, "messages": messages, "schema": schema},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cached(
    cache_dir: str | None,
    provider: str,
    model: str,
    messages: list[dict],
    schema: dict | None,
    call: Callable[[], dict | str],
) -> dict | str:
    """Return a cached response if present, else call the model and cache the result."""
    if not cache_dir:
        return call()
    key = _cache_key(provider, model, messages, schema)
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    result = call()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"response": result}, indent=2), encoding="utf-8")
    return result


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic takes ``system`` as a top-level argument, not a message role."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


def _extract_payload(messages: list[dict[str, str]]) -> dict:
    """Recover the fenced JSON payload the prompt builders embed for the offline client."""
    for m in reversed(messages):
        match = _PAYLOAD_FENCE.search(m.get("content", ""))
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return {}


def _is_judge_schema(schema: dict | None) -> bool:
    """True when the caller is asking for a judge verdict (the schema has a "correct" field)."""
    return bool(schema) and "correct" in schema.get("properties", {})


# --------------------------------------------------------------------------- adapters

_TOOL_NAME = "emit_verdict"


class AnthropicAdapter:
    """Structured output via a forced tool call whose ``input_schema`` is our JSON schema."""

    def __init__(self, model: str = "claude-sonnet-4-6", cache_dir: str | None = DEFAULT_CACHE_DIR):
        self.model = model
        self.cache_dir = cache_dir

    def complete(self, messages, schema=None):
        """Send the messages to Anthropic and return the structured verdict (or plain text); cached on disk."""

        def call() -> dict | str:
            import anthropic  # imported only inside the adapter

            client = anthropic.Anthropic()
            system, convo = _split_system(messages)
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 1024,
                "system": system or anthropic.NOT_GIVEN,
                "messages": convo,
            }
            if schema is not None:
                kwargs["tools"] = [
                    {
                        "name": _TOOL_NAME,
                        "description": "Emit the structured result.",
                        "input_schema": schema,
                    }
                ]
                kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}
            resp = client.messages.create(**kwargs)
            if schema is not None:
                for block in resp.content:
                    if block.type == "tool_use" and block.name == _TOOL_NAME:
                        return dict(block.input)
                raise RuntimeError("Anthropic response contained no tool_use block")
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        return _cached(self.cache_dir, "anthropic", self.model, messages, schema, call)


class OpenAIAdapter:
    """Structured output via the ``json_schema`` response format."""

    def __init__(self, model: str = "gpt-4o", cache_dir: str | None = DEFAULT_CACHE_DIR):
        self.model = model
        self.cache_dir = cache_dir

    def complete(self, messages, schema=None):
        """Send the messages to OpenAI and return the structured verdict (or plain text); cached on disk."""

        def call() -> dict | str:
            import openai  # imported only inside the adapter

            client = openai.OpenAI()
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "verdict", "schema": schema},
                }
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return json.loads(content) if schema is not None else content.strip()

        return _cached(self.cache_dir, "openai", self.model, messages, schema, call)


class OfflineClient:
    """Deterministic, no-network client. A real baseline, not a peeking stub (see module docstring)."""

    model = "offline-deterministic"

    def complete(self, messages, schema=None):
        """Route to the offline judge when a verdict is asked for, otherwise to the knowledge-free answerer."""
        if _is_judge_schema(schema):
            return self._judge(messages)
        return self._answer(messages)

    @staticmethod
    def _judge(messages: list[dict[str, str]]) -> dict:
        """Grade an answer offline by its token-F1 overlap with the reference (a real heuristic, threshold 0.5)."""
        payload = _extract_payload(messages)
        reference = str(payload.get("reference", ""))
        candidate = str(payload.get("candidate", ""))
        f1 = token_f1(candidate, reference)
        correct = f1 >= OFFLINE_JUDGE_F1_THRESHOLD
        return {
            "correct": correct,
            "reason": (
                f"Offline heuristic judge: token-F1 vs reference = {f1:.2f} "
                f"({'>=' if correct else '<'} {OFFLINE_JUDGE_F1_THRESHOLD} threshold)."
            ),
        }

    @staticmethod
    def _answer(messages: list[dict[str, str]]) -> str:
        """The offline answerer has no knowledge, so it honestly says "I don't know"."""
        # Knowledge-free: the offline answerer sees only the question, so it cannot really answer.
        # Honest by design — an offline leaderboard is weak, and that is the point of needing a key.
        return "I don't know."


# --------------------------------------------------------------------------- factory


def get_llm_client() -> LLMClient:
    """Factory selected by YARDSTICK_LLM_PROVIDER (default: auto-detect from available keys)."""
    provider = os.environ.get("YARDSTICK_LLM_PROVIDER", "auto").lower()
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "offline"
    if provider == "anthropic":
        return AnthropicAdapter()
    if provider == "openai":
        return OpenAIAdapter()
    if provider == "offline":
        return OfflineClient()
    raise ValueError(
        f"Unknown YARDSTICK_LLM_PROVIDER: {provider!r} (expected 'anthropic', 'openai', or 'offline')"
    )
