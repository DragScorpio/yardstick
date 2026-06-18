"""LLM-as-judge: decide whether a candidate answer is correct given the question + reference.

The judge is itself evaluated (calibrated against golden labels in calibrate.py). Watch the known judge
pitfalls and surface them in the report: verbosity bias, position bias, self-preference (prefer a judge
model different from the model under test).
"""

from __future__ import annotations

import json
from typing import Any

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
}

RUBRIC = (
    "Mark the answer correct if it conveys the reference answer's key fact, allowing paraphrase and "
    "extra detail. Mark it incorrect if it contradicts or omits that fact."
)

SYSTEM_PROMPT = (
    "You are a careful grader of short factual answers. "
    + RUBRIC
    + " Judge only the factual content; do not reward verbosity or longer answers. "
    "Respond with the structured verdict only."
)


def build_messages(question: str, reference: str, candidate: str) -> list[dict[str, str]]:
    """Prompt = rubric + the three fields, plus a fenced JSON payload for the offline judge.

    Real models read the natural-language fields; the offline client parses the fence. Both see the same
    information, so the prompt is honest for either backend.
    """
    payload = json.dumps({"question": question, "reference": reference, "candidate": candidate})
    user = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Decide whether the candidate is correct per the rubric.\n\n"
        f"```json\n{payload}\n```"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def judge_answer(client, question: str, reference: str, candidate: str) -> dict:
    """Return {correct: bool, reason: str}."""
    result = client.complete(build_messages(question, reference, candidate), JUDGE_SCHEMA)
    if not isinstance(result, dict):
        raise RuntimeError("Judge backend did not return a structured verdict")
    return {"correct": bool(result["correct"]), "reason": str(result.get("reason", ""))}
