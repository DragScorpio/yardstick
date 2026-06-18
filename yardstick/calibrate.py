"""Calibrate the judge against hand-labeled golden truth, and quantify the gap vs exact match.

This is the heart of the project: it tells you how much to trust the judge, and exposes how strict the
deterministic metric is.
"""

from __future__ import annotations


def judge_vs_golden(judge_correct: list[bool], golden_correct: list[bool]) -> dict:
    """Accuracy, precision/recall/F1 for the 'correct' class, and Cohen's kappa (judge vs golden)."""
    raise NotImplementedError


def em_vs_golden(em_correct: list[bool], golden_correct: list[bool]) -> dict:
    """Same agreement stats for plain exact match, to expose how often EM disagrees with truth."""
    raise NotImplementedError
