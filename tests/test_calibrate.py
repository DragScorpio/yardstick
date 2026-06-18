"""Unit tests for the calibration agreement stats."""

from __future__ import annotations

import pytest

from yardstick.calibrate import em_vs_golden, judge_vs_golden


def test_agreement_stats_known_confusion():
    # predicted vs golden: tp=2, fp=1, fn=1, tn=1 (n=5)
    predicted = [True, True, True, False, False]
    golden = [True, True, False, True, False]
    s = judge_vs_golden(predicted, golden)
    assert s["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)


def test_em_recall_collapses_on_paraphrases():
    # EM accepts only the one exact item; golden marks all five correct.
    em = [True, False, False, False, False]
    golden = [True, True, True, True, True]
    s = em_vs_golden(em, golden)
    assert s["precision"] == 1.0  # never wrongly accepts
    assert s["recall"] == pytest.approx(1 / 5)  # but rejects the paraphrases


def test_zero_division_guards():
    # No positive predictions -> precision defined as 0.0, not a crash.
    s = judge_vs_golden([False, False], [True, False])
    assert s["precision"] == 0.0
    assert s["recall"] == 0.0
    assert s["f1"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        judge_vs_golden([True], [True, False])
