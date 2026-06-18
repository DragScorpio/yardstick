"""Yardstick CLI: prepare, run, score, calibrate, report.

A small, scriptable surface. Each subcommand reads/writes plain files under ``data/`` and ``results/`` so
the stages compose and the run is reproducible:

    prepare  -> data/slice.jsonl, data/golden_template.csv
    run      -> results/run.jsonl                 (model answers; responses cached on disk)
    score    -> results/scored.jsonl, results/leaderboard.json
    calibrate-> results/calibration.json          (judge & EM agreement vs hand-labeled gold)
    report   -> results/report.md, results/latest.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from . import data as data_mod
from .judge import judge_answer
from .llm import LLMClient, get_llm_client
from .metrics import exact_match, token_f1
from .report import render_report

RESULTS_DIR = "results"


# --------------------------------------------------------------------------- helpers


def _client_label(client: LLMClient) -> str:
    return f"{type(client).__name__}:{getattr(client, 'model', 'unknown')}"


def _answer_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Answer the question with the single fact requested, as briefly as possible.",
        },
        {"role": "user", "content": question},
    ]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# --------------------------------------------------------------------------- commands


def cmd_prepare(args: argparse.Namespace) -> int:
    summary = data_mod.prepare(out_dir=args.out, n=args.n)
    print(
        f"Prepared {summary['sliced']} items from {summary['dataset']} "
        f"(of {summary['total_available']} available).\n"
        f"  slice:          {summary['slice_path']}\n"
        f"  golden template: {summary['golden_template_path']}\n"
        f"Hand-label candidate answers in {data_mod.GOLDEN_FILENAME} to build the calibration gold."
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    client = get_llm_client()
    rows = data_mod.load_slice(args.slice)
    if args.limit:
        rows = rows[: args.limit]
    out_rows = []
    for r in rows:
        answer = client.complete(_answer_messages(r["question"]))
        out_rows.append({**r, "candidate_answer": str(answer).strip()})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Generated {len(out_rows)} answers with {_client_label(client)} -> {out_path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    client = get_llm_client()
    rows = data_mod.load_slice(args.run)  # run.jsonl is jsonl too
    scored = []
    em_hits = 0
    f1_sum = 0.0
    judge_hits = 0
    disagreements = 0
    for r in rows:
        ref = r["reference_answer"]
        cand = r.get("candidate_answer", "")
        em = exact_match(cand, ref)
        f1 = token_f1(cand, ref)
        verdict = judge_answer(client, r["question"], ref, cand)
        scored.append({**r, "exact_match": em, "token_f1": f1, "judge": verdict})
        em_hits += int(em)
        f1_sum += f1
        judge_hits += int(verdict["correct"])
        disagreements += int(em != verdict["correct"])

    n = len(rows) or 1
    leaderboard = {
        "model": _client_label(client),
        "n": len(rows),
        "exact_match_rate": em_hits / n,
        "mean_token_f1": f1_sum / n,
        "judge_correct_rate": judge_hits / n,
        "em_judge_disagreement_rate": disagreements / n,
    }

    scored_path = Path(args.out)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_path.open("w", encoding="utf-8", newline="") as fh:
        for r in scored:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _write_json(Path(RESULTS_DIR) / "leaderboard.json", leaderboard)
    print(
        f"Scored {len(rows)} items: EM {leaderboard['exact_match_rate']:.1%}, "
        f"mean F1 {leaderboard['mean_token_f1']:.3f}, "
        f"judge-correct {leaderboard['judge_correct_rate']:.1%}, "
        f"EM/judge disagree {leaderboard['em_judge_disagreement_rate']:.1%}"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import em_vs_golden, judge_vs_golden

    client = get_llm_client()
    golden = data_mod.load_golden(args.golden)
    if not golden:
        print(f"No labeled rows in {args.golden}; nothing to calibrate.")
        return 1

    golden_correct: list[bool] = []
    em_correct: list[bool] = []
    judge_correct: list[bool] = []
    divergence: list[dict[str, Any]] = []

    for r in golden:
        ref = r["reference_answer"]
        cand = r["candidate_answer"]
        g = r["golden_correct"]
        em = exact_match(cand, ref)
        verdict = judge_answer(client, r["question"], ref, cand)
        golden_correct.append(g)
        em_correct.append(em)
        judge_correct.append(verdict["correct"])
        # Headline illustration: the human says correct and the judge agrees, but EM marks it wrong.
        if g and verdict["correct"] and not em:
            divergence.append(
                {
                    "question": r["question"],
                    "reference": ref,
                    "candidate": cand,
                    "em": em,
                    "judge": verdict["correct"],
                    "golden": g,
                }
            )

    jvg = judge_vs_golden(judge_correct, golden_correct)
    evg = em_vs_golden(em_correct, golden_correct)
    headline = (
        f"the judge agrees with hand labels at kappa={jvg['cohen_kappa']:.2f}, "
        f"exact match only at kappa={evg['cohen_kappa']:.2f} — "
        "exact match marks correct paraphrases wrong, the judge does not."
    )
    calibration = {
        "n": len(golden),
        "golden_correct_rate": sum(golden_correct) / len(golden),
        "judge_provider": _client_label(client),
        "judge_vs_golden": jvg,
        "em_vs_golden": evg,
        "headline": headline,
        "divergence_examples": divergence[: args.max_examples],
    }
    _write_json(Path(RESULTS_DIR) / "calibration.json", calibration)
    print(
        f"Calibrated on {len(golden)} gold labels. "
        f"Judge kappa={jvg['cohen_kappa']:.3f} (acc {jvg['accuracy']:.1%}); "
        f"EM kappa={evg['cohen_kappa']:.3f} (acc {evg['accuracy']:.1%})."
    )
    print(f"  {headline}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    leaderboard_path = Path(RESULTS_DIR) / "leaderboard.json"
    calibration_path = Path(RESULTS_DIR) / "calibration.json"
    leaderboard = (
        json.loads(leaderboard_path.read_text(encoding="utf-8"))
        if leaderboard_path.exists()
        else None
    )
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.exists()
        else None
    )

    n_slice = leaderboard["n"] if leaderboard else None
    results = {
        "meta": {
            "dataset": "SQuAD v1.1 dev",
            "answer_provider": leaderboard["model"] if leaderboard else None,
            "judge_provider": (calibration or {}).get("judge_provider")
            or (leaderboard["model"] if leaderboard else None),
            "n_slice": n_slice,
            "n_golden": calibration["n"] if calibration else None,
            "generated_at": _timestamp(),
        },
        "leaderboard": leaderboard,
        "calibration": calibration,
        "divergence_examples": (calibration or {}).get("divergence_examples", []),
    }
    markdown = render_report(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(markdown, encoding="utf-8")
    _write_json(Path(RESULTS_DIR) / "latest.json", results)
    print(f"Wrote report -> {args.out} (and {RESULTS_DIR}/latest.json)")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yardstick", description="Calibrated LLM evaluation")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser(
        "prepare", help="download/slice the dataset + write a golden-label template"
    )
    sp.add_argument("--out", default="data")
    sp.add_argument("--n", type=int, default=200)
    sp.set_defaults(func=cmd_prepare)

    sr = sub.add_parser("run", help="generate model answers for the slice (provider-agnostic)")
    sr.add_argument("--slice", default="data/slice.jsonl")
    sr.add_argument("--out", default=f"{RESULTS_DIR}/run.jsonl")
    sr.add_argument("--limit", type=int, default=0, help="cap items (0 = all)")
    sr.set_defaults(func=cmd_run)

    ss = sub.add_parser("score", help="deterministic metrics + LLM-judge verdicts")
    ss.add_argument("--run", default=f"{RESULTS_DIR}/run.jsonl")
    ss.add_argument("--out", default=f"{RESULTS_DIR}/scored.jsonl")
    ss.set_defaults(func=cmd_score)

    sc = sub.add_parser(
        "calibrate", help="judge-vs-golden agreement (accuracy, kappa) + the EM gap"
    )
    sc.add_argument("--golden", default=f"data/{data_mod.GOLDEN_FILENAME}")
    sc.add_argument("--max-examples", type=int, default=8)
    sc.set_defaults(func=cmd_calibrate)

    srep = sub.add_parser("report", help="render the markdown report")
    srep.add_argument("--out", default=f"{RESULTS_DIR}/report.md")
    srep.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
