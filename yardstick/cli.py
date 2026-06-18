"""Yardstick CLI: prepare, run, score, calibrate, report.

Worker: wire each subcommand to the real implementation. Keep the surface small and scriptable.
"""

from __future__ import annotations

import argparse


def cmd_prepare(args: argparse.Namespace) -> int:
    """Download/slice the dataset and write a golden-label template."""
    raise NotImplementedError


def cmd_run(args: argparse.Namespace) -> int:
    """Generate model answers for the slice (provider-agnostic), caching responses."""
    raise NotImplementedError


def cmd_score(args: argparse.Namespace) -> int:
    """Compute deterministic metrics (exact match, token-F1) and LLM-judge verdicts."""
    raise NotImplementedError


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Report judge-vs-golden agreement (accuracy, kappa) and the exact-match gap."""
    raise NotImplementedError


def cmd_report(args: argparse.Namespace) -> int:
    """Render the markdown evaluation report."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yardstick", description="Calibrated LLM evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    for name, func, help_text in [
        ("prepare", cmd_prepare, "download/slice the dataset + write a golden-label template"),
        ("run", cmd_run, "generate model answers for the slice (provider-agnostic)"),
        ("score", cmd_score, "deterministic metrics + LLM-judge verdicts"),
        ("calibrate", cmd_calibrate, "judge-vs-golden agreement (accuracy, kappa) + the EM gap"),
        ("report", cmd_report, "render the markdown report"),
    ]:
        s = sub.add_parser(name, help=help_text)
        s.set_defaults(func=func)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
