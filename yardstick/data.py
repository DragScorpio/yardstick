"""Dataset prep + golden-label IO.

prepare(): download/slice a public factual-QA set into (question, reference_answer) rows and write a
golden-label template for hand labeling. The dataset bulk is NOT committed (data/ is gitignored); the
small hand-labeled golden file IS committed. Public data only.
"""

from __future__ import annotations


def prepare(out_dir: str = "data", n: int = 200) -> None:
    """Download, slice to ``n`` items, and emit a golden-label template to hand-fill."""
    raise NotImplementedError


def load_slice(path: str):
    """Yield {question, reference_answer} rows."""
    raise NotImplementedError


def load_golden(path: str = "data/golden_labels.csv"):
    """Load the hand-labeled correct/incorrect gold used to calibrate the judge."""
    raise NotImplementedError
