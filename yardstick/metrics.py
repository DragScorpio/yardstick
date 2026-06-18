"""Deterministic answer metrics (pure functions; unit-tested on known inputs).

SQuAD-style normalization then exact match, plus token-level F1. Cohen's kappa for rater agreement.
These are the "cheap and strict" baseline the judge gets compared against.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# Compiled once: SQuAD-style normalization pieces.
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip articles, drop punctuation, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    """True if normalized prediction == normalized reference."""
    return normalize(prediction) == normalize(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between normalized prediction and reference (partial credit)."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    # Two empties agree perfectly; one empty against a non-empty has zero overlap.
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def cohen_kappa(rater_a: list[bool], rater_b: list[bool]) -> float:
    """Agreement beyond chance between two binary raters (e.g. judge vs golden).

    kappa = (po - pe) / (1 - pe), where po is observed agreement and pe is the agreement
    expected by chance from each rater's marginal rate of "True". Returns 1.0 when both raters
    are constant and identical (no disagreement possible), 0.0 when they are constant but differ.
    """
    n = len(rater_a)
    if n == 0 or n != len(rater_b):
        raise ValueError("raters must be non-empty and the same length")
    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    pa = sum(1 for a in rater_a if a) / n
    pb = sum(1 for b in rater_b if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        # Both raters are constant. kappa is undefined by the formula; report perfect agreement
        # if they match on that constant, otherwise zero.
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)
