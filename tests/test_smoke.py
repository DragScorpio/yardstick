"""Smoke tests: the package imports and the core wiring is present. Real coverage comes with the logic."""

from __future__ import annotations

import os


def test_package_imports():
    import yardstick

    assert yardstick.__version__


def test_llm_factory_is_provider_agnostic():
    os.environ["YARDSTICK_LLM_PROVIDER"] = "offline"
    from yardstick.llm import get_llm_client

    client = get_llm_client()
    assert hasattr(client, "complete")


def test_judge_schema_present():
    from yardstick.judge import JUDGE_SCHEMA

    assert JUDGE_SCHEMA["required"]
