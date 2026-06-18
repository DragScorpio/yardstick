# Yardstick — calibrated LLM evaluation

A small, focused, end-to-end evaluation of an LLM on one task (short-answer factual QA). Yardstick scores
model answers two ways, a strict deterministic metric and an LLM-as-judge, then does the move most eval
demos skip: it **calibrates the judge against hand-labeled ground truth** and shows exactly where the
cheap string match lies to you.

> Status: scaffolded skeleton. Implementation in progress. See
> [`../blueprints/02-yardstick.md`](../blueprints/02-yardstick.md) for the full spec and acceptance
> criteria, and [`../PORTFOLIO.md`](../PORTFOLIO.md) for the initiative rules.

## Why it's not a prompt wrapper

Remove the LLMs and there is still real software: dataset tooling, normalized exact-match and token-F1
metrics, calibration statistics (Cohen's kappa), regression diff, and a report generator. The AI usage
here, the judge, is itself evaluated (calibrated against gold), which is the whole point.

## Quickstart (once implemented)

```bash
uv venv && uv pip install -e ".[dev,anthropic]"
yardstick prepare                 # download/slice a public QA set + golden-label template
# (hand-label ~50 rows in data/golden_labels.csv)
yardstick run                     # generate model answers (provider-agnostic)
yardstick score                   # exact-match + token-F1 + LLM-judge verdicts
yardstick calibrate               # judge-vs-golden agreement (accuracy, kappa) + the exact-match gap
yardstick report                  # markdown report
```

Pick the provider with `YARDSTICK_LLM_PROVIDER=anthropic` (default), `openai`, or `offline`, plus that
provider's API key. Any capable model works; the LLM sits behind a provider-agnostic interface.

## License

MIT (intended). Public, unrestricted libraries and public datasets only.
