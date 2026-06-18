"""Dataset prep + golden-label IO.

``prepare`` downloads SQuAD v1.1 dev (a public factual-QA set, CC BY-SA 4.0), slices it to ``n``
``(question, reference_answer)`` rows, and writes a golden-label template to hand-fill. The dataset bulk
is NOT committed (``data/`` is gitignored); the small hand-labeled golden file IS committed. Public data
only.

The golden file is self-contained on purpose: each row pins ``(question, reference_answer,
candidate_answer, golden_correct)``, so calibration is reproducible without re-running a model — the human
label is attached to the exact candidate text it judged.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

SLICE_FILENAME = "slice.jsonl"
GOLDEN_TEMPLATE_FILENAME = "golden_template.csv"
GOLDEN_FILENAME = "golden_labels.csv"
GOLDEN_FIELDS = ["id", "question", "reference_answer", "candidate_answer", "golden_correct"]


def _canonical_reference(answers: list[str]) -> str:
    """SQuAD lists several acceptable answers; use the most common (ties broken by first seen)."""
    counts = Counter(a.strip() for a in answers if a and a.strip())
    if not counts:
        return ""
    top = max(counts.values())
    for a in answers:  # preserve dataset order among ties
        if a.strip() and counts[a.strip()] == top:
            return a.strip()
    return ""


def _flatten_squad(raw: dict) -> list[dict[str, str]]:
    """Pull (id, question, reference_answer) rows out of the SQuAD JSON structure."""
    rows: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for article in raw.get("data", []):
        for para in article.get("paragraphs", []):
            for qa in para.get("qas", []):
                question = qa.get("question", "").strip()
                reference = _canonical_reference([a["text"] for a in qa.get("answers", [])])
                if not question or not reference or question.lower() in seen_questions:
                    continue
                seen_questions.add(question.lower())
                rows.append(
                    {"id": qa.get("id", ""), "question": question, "reference_answer": reference}
                )
    return rows


def prepare(out_dir: str = "data", n: int = 200, seed: int = 7) -> dict[str, Any]:
    """Download, slice to ``n`` items, and emit a golden-label template to hand-fill.

    Returns a small summary dict (counts + paths). Deterministic given ``seed``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(
        SQUAD_DEV_URL, timeout=60
    ) as resp:  # noqa: S310 (trusted public URL)
        raw = json.loads(resp.read().decode("utf-8"))

    rows = _flatten_squad(raw)
    random.Random(seed).shuffle(rows)
    sliced = rows[:n]

    slice_path = out / SLICE_FILENAME
    with slice_path.open("w", encoding="utf-8", newline="") as fh:
        for r in sliced:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Golden template: the first ~50 sliced items, with candidate + label left blank for a human.
    template_path = out / GOLDEN_TEMPLATE_FILENAME
    with template_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=GOLDEN_FIELDS)
        writer.writeheader()
        for r in sliced[: min(50, len(sliced))]:
            writer.writerow(
                {
                    "id": r["id"],
                    "question": r["question"],
                    "reference_answer": r["reference_answer"],
                    "candidate_answer": "",
                    "golden_correct": "",
                }
            )

    return {
        "dataset": "SQuAD v1.1 dev",
        "total_available": len(rows),
        "sliced": len(sliced),
        "slice_path": str(slice_path),
        "golden_template_path": str(template_path),
    }


def load_slice(path: str = "data/slice.jsonl") -> list[dict[str, str]]:
    """Yield {id, question, reference_answer} rows."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_bool(value: str) -> bool:
    """Read a human-written label ("correct", "yes", "true", ...) as a boolean."""
    return str(value).strip().lower() in {"1", "true", "yes", "y", "correct", "t"}


def load_golden(path: str = "data/golden_labels.csv") -> list[dict[str, Any]]:
    """Load the hand-labeled correct/incorrect gold used to calibrate the judge."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            label = (r.get("golden_correct") or "").strip()
            if label == "":
                continue  # skip un-labeled template rows
            rows.append(
                {
                    "id": r.get("id", ""),
                    "question": r["question"],
                    "reference_answer": r["reference_answer"],
                    "candidate_answer": r.get("candidate_answer", ""),
                    "golden_correct": _parse_bool(label),
                }
            )
    return rows
