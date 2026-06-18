"""Render the markdown evaluation report: leaderboard, exact-match-vs-judge divergence, calibration.

``render_report`` takes the combined results dict the CLI assembles (meta + optional leaderboard +
optional calibration + divergence examples) and returns markdown. Sections are rendered only when present,
so a report can be produced after ``score`` alone or after the full pipeline.
"""

from __future__ import annotations

from typing import Any


def _pct(x: float) -> str:
    """Show a 0–1 fraction as a friendly percentage like "86.0%"."""
    return f"{x * 100:.1f}%"


def _fmt(x: float) -> str:
    """Show a float to three decimals — used for kappa and F1 values."""
    return f"{x:.3f}"


def _stats_row(label: str, s: dict[str, Any]) -> str:
    """Lay out one rater's agreement numbers as a single markdown table row."""
    return (
        f"| {label} | {_fmt(s['cohen_kappa'])} | {_pct(s['accuracy'])} | "
        f"{_pct(s['precision'])} | {_pct(s['recall'])} | {_fmt(s['f1'])} |"
    )


def render_report(results: dict) -> str:
    """Turn the combined results dict into the full markdown report.

    Builds the page section by section — metadata, leaderboard, calibration, the "where exact match
    lies" examples, and the judge-pitfalls note — including each section only when its data is present,
    so a partial run still produces a sensible report.
    """
    meta = results.get("meta", {})
    lines: list[str] = []
    lines.append("# Yardstick evaluation report")
    lines.append("")
    if meta:
        lines.append(
            f"- **Dataset:** {meta.get('dataset', '?')}  "
            f"\n- **Answer model:** `{meta.get('answer_provider', '?')}`  "
            f"\n- **Judge:** `{meta.get('judge_provider', '?')}`  "
            f"\n- **Slice size:** {meta.get('n_slice', '?')}  "
            f"\n- **Golden labels:** {meta.get('n_golden', '?')}  "
            f"\n- **Generated:** {meta.get('generated_at', '?')}"
        )
        lines.append("")

    lb = results.get("leaderboard")
    if lb:
        lines.append("## Leaderboard (full slice)")
        lines.append("")
        lines.append("| Model | Items | Exact match | Mean token-F1 | Judge says correct |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| `{lb['model']}` | {lb['n']} | {_pct(lb['exact_match_rate'])} | "
            f"{_fmt(lb['mean_token_f1'])} | {_pct(lb['judge_correct_rate'])} |"
        )
        lines.append("")
        lines.append(
            f"Exact match and the judge disagree on **{_pct(lb['em_judge_disagreement_rate'])}** "
            "of items — the cheap metric and the judge are not measuring the same thing."
        )
        lines.append("")

    cal = results.get("calibration")
    if cal:
        jvg = cal["judge_vs_golden"]
        evg = cal["em_vs_golden"]
        lines.append("## Calibration against hand-labeled gold")
        lines.append("")
        lines.append(
            f"Treating {cal['n']} hand-labeled answers as truth "
            f"({_pct(cal['golden_correct_rate'])} of them labeled correct):"
        )
        lines.append("")
        lines.append("| Rater | Cohen's κ | Accuracy | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(_stats_row("LLM-as-judge", jvg))
        lines.append(_stats_row("Exact match", evg))
        lines.append("")
        lines.append(f"**Headline:** {cal['headline']}")
        lines.append("")

    examples = results.get("divergence_examples") or []
    if examples:
        lines.append("## Where exact match lies (judge & humans agree, EM disagrees)")
        lines.append("")
        lines.append("| Question | Reference | Candidate | EM | Judge | Gold |")
        lines.append("|---|---|---|:--:|:--:|:--:|")
        for ex in examples:
            q = _truncate(ex["question"], 60)
            ref = _truncate(ex["reference"], 30)
            cand = _truncate(ex["candidate"], 40)
            lines.append(
                f"| {q} | {ref} | {cand} | "
                f"{_tick(ex['em'])} | {_tick(ex['judge'])} | {_tick(ex['golden'])} |"
            )
        lines.append("")

    lines.append("## Judge pitfalls to keep in mind")
    lines.append("")
    lines.append(
        "An LLM judge is not free of bias. Known failure modes this harness is built to keep honest:\n"
        "- **Verbosity bias** — preferring longer answers; the rubric grades factual content only.\n"
        "- **Position bias** — order effects in pairwise setups (not used here; single-answer grading).\n"
        "- **Self-preference** — a model favoring its own style; prefer a judge model different from the "
        "model under test. Calibration against gold is what tells you the residual bias that remains."
    )
    lines.append("")
    return "\n".join(lines)


def _truncate(text: str, width: int) -> str:
    """Squeeze text down to fit a table cell, adding an ellipsis if it had to be cut."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _tick(value: bool) -> str:
    """Render a yes/no verdict as a green check or red cross for the report tables."""
    return "✅" if value else "❌"
