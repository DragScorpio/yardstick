"""Deterministic answer metrics (pure functions; unit-test these on known inputs).

SQuAD-style normalization then exact match, plus token-level F1. Cohen's kappa for rater agreement.
These are the "cheap and strict" baseline the judge gets compared against.
"""

from __future__ import annotations


def normalize(text: str) -> str:
    """Lowercase, strip articles, drop punctuation, collapse whitespace (SQuAD-style)."""
    raise NotImplementedError


def exact_match(prediction: str, reference: str) -> bool:
    """True if normalized prediction == normalized reference."""
    raise NotImplementedError


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between normalized prediction and reference (partial credit)."""
    raise NotImplementedError


def cohen_kappa(rater_a: list[bool], rater_b: list[bool]) -> float:
    """Agreement beyond chance between two binary raters (e.g. judge vs golden)."""
    raise NotImplementedError
