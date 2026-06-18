"""LLM-as-judge: decide whether a candidate answer is correct given the question + reference.

The judge is itself evaluated (calibrated against golden labels in calibrate.py). Watch the known judge
pitfalls and surface them in the report: verbosity bias, position bias, self-preference (prefer a judge
model different from the model under test).
"""

from __future__ import annotations

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


def judge_answer(client, question: str, reference: str, candidate: str) -> dict:
    """Return {correct: bool, reason: str}. Worker: build the prompt from RUBRIC + inputs."""
    raise NotImplementedError
