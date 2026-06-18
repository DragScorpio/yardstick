"""Unit tests for the provider-agnostic LLM layer and the judge wiring (offline backend)."""

from __future__ import annotations

from yardstick.judge import JUDGE_SCHEMA, build_messages, judge_answer
from yardstick.llm import OfflineClient, get_llm_client


def test_factory_selects_offline(monkeypatch):
    monkeypatch.setenv("YARDSTICK_LLM_PROVIDER", "offline")
    assert isinstance(get_llm_client(), OfflineClient)


def test_factory_auto_falls_back_to_offline_without_keys(monkeypatch):
    monkeypatch.setenv("YARDSTICK_LLM_PROVIDER", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(get_llm_client(), OfflineClient)


def test_offline_answerer_is_knowledge_free():
    client = OfflineClient()
    out = client.complete([{"role": "user", "content": "What is the capital of France?"}])
    assert isinstance(out, str)
    assert "know" in out.lower()  # honest "I don't know."


def test_offline_judge_accepts_paraphrase_rejects_mismatch():
    client = OfflineClient()
    accept = judge_answer(client, "Which Pope?", "Pope Leo X", "Leo X")
    reject = judge_answer(client, "Which Pope?", "Pope Leo X", "Pope Pius XII")
    assert accept["correct"] is True
    assert reject["correct"] is False
    assert "F1" in accept["reason"]


def test_offline_judge_is_a_real_classifier_not_a_peek():
    # The offline judge decides from token overlap, so an empty candidate is rejected
    # even though the prompt also contains the reference. It does not just echo "correct".
    client = OfflineClient()
    verdict = judge_answer(client, "When?", "1968", "")
    assert verdict["correct"] is False


def test_build_messages_embeds_payload_for_offline_and_text_for_models():
    msgs = build_messages("Q?", "ref answer", "cand answer")
    joined = msgs[-1]["content"]
    assert "ref answer" in joined and "cand answer" in joined  # readable for real models
    assert '"candidate": "cand answer"' in joined  # fenced payload for the offline client


def test_judge_schema_shape():
    assert JUDGE_SCHEMA["required"] == ["correct", "reason"]


def test_cache_roundtrip(tmp_path, monkeypatch):
    from yardstick import llm

    calls = {"n": 0}

    def fake_call():
        calls["n"] += 1
        return {"correct": True, "reason": "x"}

    cache_dir = str(tmp_path / "cache")
    msgs = [{"role": "user", "content": "hi"}]
    first = llm._cached(cache_dir, "p", "m", msgs, JUDGE_SCHEMA, fake_call)
    second = llm._cached(cache_dir, "p", "m", msgs, JUDGE_SCHEMA, fake_call)
    assert first == second == {"correct": True, "reason": "x"}
    assert calls["n"] == 1  # second call served from disk cache
