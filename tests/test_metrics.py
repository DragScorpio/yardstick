"""Unit tests for the deterministic metrics — the pure, teachable core."""

from __future__ import annotations

import math

import pytest

from yardstick.metrics import cohen_kappa, exact_match, normalize, token_f1


def test_normalize_strips_articles_punctuation_and_case():
    assert normalize("The Denver Broncos!") == "denver broncos"
    assert normalize("  a   CAT,  an apple. ") == "cat apple"
    assert normalize("1,858") == "1858"


def test_exact_match_is_paraphrase_strict():
    assert exact_match("the Court of Justice", "Court of Justice")  # only article differs
    assert not exact_match("The European Court of Justice", "the Court of Justice")
    assert not exact_match("9", "nine")  # numerals it cannot reconcile


def test_token_f1_partial_credit():
    assert token_f1("Pope Leo X", "Leo X") == pytest.approx(0.8)  # subset, high overlap
    assert token_f1("direction and magnitude", "magnitude and direction") == 1.0  # reorder
    assert token_f1("apples", "oranges") == 0.0


def test_token_f1_empty_cases():
    assert token_f1("", "") == 1.0
    assert token_f1("", "something") == 0.0
    assert token_f1("something", "") == 0.0


def test_cohen_kappa_perfect_and_chance():
    a = [True, False, True, False]
    assert cohen_kappa(a, a) == 1.0
    # Anti-correlated raters: agreement below chance -> negative kappa.
    b = [False, True, False, True]
    assert cohen_kappa(a, b) == pytest.approx(-1.0)


def test_cohen_kappa_known_value():
    # 2x2: both-yes=20, both-no=15, a-yes-b-no=5, a-no-b-yes=10 (n=50).
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    # po = 35/50 = 0.70; pa=0.5, pb=0.6; pe=0.5 -> kappa=(0.7-0.5)/0.5=0.4
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_constant_raters():
    assert cohen_kappa([True, True], [True, True]) == 1.0
    assert cohen_kappa([True, True], [False, False]) == 0.0


def test_cohen_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def test_kappa_is_finite_for_typical_inputs():
    assert math.isfinite(cohen_kappa([True, False, True], [True, True, False]))
