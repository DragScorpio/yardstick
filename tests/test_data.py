"""Unit tests for dataset flattening and golden-label IO."""

from __future__ import annotations

from yardstick.data import _canonical_reference, _flatten_squad, load_golden


def test_canonical_reference_picks_most_common():
    assert _canonical_reference(["Paris", "Paris", "paris city"]) == "Paris"
    assert _canonical_reference([]) == ""
    # Tie -> first in dataset order.
    assert _canonical_reference(["A", "B"]) == "A"


def test_flatten_squad_extracts_and_dedupes():
    raw = {
        "data": [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {"id": "1", "question": "Q1?", "answers": [{"text": "a1"}]},
                            {"id": "2", "question": "Q1?", "answers": [{"text": "dup"}]},
                            {"id": "3", "question": "Q2?", "answers": []},  # no answer -> dropped
                            {"id": "4", "question": "Q3?", "answers": [{"text": "a3"}]},
                        ]
                    }
                ]
            }
        ]
    }
    rows = _flatten_squad(raw)
    questions = [r["question"] for r in rows]
    assert questions == ["Q1?", "Q3?"]  # Q1 deduped, Q2 dropped (no answer)
    assert rows[0]["reference_answer"] == "a1"


def test_load_golden_parses_labels_and_skips_blanks(tmp_path):
    csv_path = tmp_path / "golden.csv"
    csv_path.write_text(
        "id,question,reference_answer,candidate_answer,golden_correct\n"
        "1,Q1,ref1,cand1,correct\n"
        "2,Q2,ref2,cand2,incorrect\n"
        "3,Q3,ref3,,\n",  # unlabeled template row -> skipped
        encoding="utf-8",
    )
    rows = load_golden(str(csv_path))
    assert len(rows) == 2
    assert rows[0]["golden_correct"] is True
    assert rows[1]["golden_correct"] is False
