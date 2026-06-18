"""Calibrate the judge against hand-labeled golden truth, and quantify the gap vs exact match.

This is the heart of the project: it tells you how much to trust the judge, and exposes how strict the
deterministic metric is. Golden labels are treated as ground truth; we report how well each rater (the
judge, and plain exact match) agrees with them.
"""

from __future__ import annotations

from typing import Any

from .metrics import cohen_kappa


def _agreement_stats(predicted: list[bool], golden: list[bool]) -> dict[str, Any]:
    """Agreement of a binary rater (``predicted``) against truth (``golden``).

    "Correct" is the positive class. Reports accuracy, precision/recall/F1 for that class, Cohen's kappa
    (agreement beyond chance), and the confusion counts.
    """
    n = len(golden)
    if n == 0 or n != len(predicted):
        raise ValueError("predicted and golden must be non-empty and the same length")

    tp = sum(1 for p, g in zip(predicted, golden) if p and g)
    fp = sum(1 for p, g in zip(predicted, golden) if p and not g)
    fn = sum(1 for p, g in zip(predicted, golden) if not p and g)
    tn = sum(1 for p, g in zip(predicted, golden) if not p and not g)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": cohen_kappa(predicted, golden),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def judge_vs_golden(judge_correct: list[bool], golden_correct: list[bool]) -> dict:
    """Accuracy, precision/recall/F1 for the 'correct' class, and Cohen's kappa (judge vs golden)."""
    return _agreement_stats(judge_correct, golden_correct)


def em_vs_golden(em_correct: list[bool], golden_correct: list[bool]) -> dict:
    """Same agreement stats for plain exact match, to expose how often EM disagrees with truth."""
    return _agreement_stats(em_correct, golden_correct)
