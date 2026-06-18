"""Report rendering + an integrity check on the committed golden file."""

from __future__ import annotations

from pathlib import Path

from yardstick.data import load_golden
from yardstick.report import render_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_render_report_includes_sections():
    results = {
        "meta": {"dataset": "SQuAD v1.1 dev", "answer_provider": "x", "judge_provider": "y"},
        "leaderboard": {
            "model": "x",
            "n": 10,
            "exact_match_rate": 0.2,
            "mean_token_f1": 0.5,
            "judge_correct_rate": 0.6,
            "em_judge_disagreement_rate": 0.4,
        },
        "calibration": {
            "n": 50,
            "golden_correct_rate": 0.62,
            "judge_vs_golden": {
                "cohen_kappa": 0.71,
                "accuracy": 0.86,
                "precision": 0.9,
                "recall": 0.87,
                "f1": 0.885,
            },
            "em_vs_golden": {
                "cohen_kappa": 0.32,
                "accuracy": 0.62,
                "precision": 1.0,
                "recall": 0.39,
                "f1": 0.56,
            },
            "headline": "judge beats EM",
        },
        "divergence_examples": [
            {
                "question": "Q",
                "reference": "r",
                "candidate": "c",
                "em": False,
                "judge": True,
                "golden": True,
            }
        ],
    }
    md = render_report(results)
    assert "# Yardstick evaluation report" in md
    assert "Leaderboard" in md
    assert "Calibration against hand-labeled gold" in md
    assert "Cohen's κ" in md
    assert "judge beats EM" in md
    assert "Judge pitfalls" in md


def test_committed_golden_file_is_well_formed():
    rows = load_golden(str(REPO_ROOT / "data" / "golden_labels.csv"))
    assert len(rows) == 50
    labels = [r["golden_correct"] for r in rows]
    # Both classes are represented, so calibration kappa is meaningful.
    assert any(labels) and not all(labels)
    for r in rows:
        assert r["question"] and r["reference_answer"] and r["candidate_answer"]
