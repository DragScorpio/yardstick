# Yardstick — Architecture

A reading companion for the codebase. It explains the mental model, how the modules depend on each other,
what each one owns, the end-to-end data flow, and a concrete trace of a single item from CSV to report
cell. Read this alongside the source; it is meant to make the code obvious rather than to replace it.

## 1. The one idea

Every AI feature eventually has to answer "how good is the model at this task?" People usually answer it
badly — either with a strict string match (which wrongly fails correct paraphrases) or with an
un-calibrated LLM judge (which you have no reason to trust). Yardstick answers it honestly: score answers
**two ways** (a cheap deterministic metric and an LLM-as-judge), then **calibrate the judge against
hand-labeled human truth** so you know exactly how much to trust it — and exactly where the cheap metric
lies to you.

Everything in the repo serves that one idea. If you remember nothing else: **the deliverable is the
*measurement of the measurer*, not the model scores.**

## 2. Mental model

```
                 ┌────────────┐   deterministic   ┌──────────────┐
   question ───► │  a model   │ ───► candidate ──►│ exact match  │──┐
   reference     │ (answerer) │      answer       │  + token-F1  │  │
                 └────────────┘                   └──────────────┘  │
                                                  ┌──────────────┐  ├─► compare both
                                                  │ LLM-as-judge │──┤   to HUMAN gold
                                                  └──────────────┘  │   = calibration
                                   hand-labeled "correct/incorrect" ─┘   (the product)
```

Two raters (exact match, the judge) each give a correct/incorrect verdict on every candidate answer. A
human has *also* labeled ~50 of those answers. Calibration is just: how well does each rater agree with the
human? The judge wins (κ 0.71 vs 0.32), and calibration is what *proves* it rather than asserting it.

## 3. Module dependency graph

Read bottom-up — every file only uses ones below it, so nothing is forward-referenced.

```
metrics.py   pure math: normalize, exact_match, token_f1, cohen_kappa   (no internal deps)
   ▲   ▲
   │   └────────────── calibrate.py   confusion matrix + agreement stats (uses cohen_kappa)
   │                        ▲
llm.py   provider-agnostic client + disk cache + offline baseline        (uses token_f1)
   ▲   (offline judge reuses token_f1)
   │
judge.py   rubric + JSON schema + build_messages + judge_answer          (drives a client)
   ▲
   │        data.py    SQuAD download/slice + golden CSV IO               (stdlib only)
   │        report.py  results dict → markdown                            (pure)
   │            ▲
   └────────────┴──── cli.py   the conductor: prepare|run|score|calibrate|report
```

`cli.py` is the only file that imports everything; it owns orchestration and file I/O. Every other module
is a self-contained, independently testable unit.

## 4. Module by module

| Module | Owns | Key functions |
|---|---|---|
| **metrics.py** | The deterministic, LLM-free core. Pure functions, exhaustively unit-tested. | `normalize` (SQuAD-style cleanup), `exact_match`, `token_f1` (partial-credit overlap), `cohen_kappa` (agreement beyond chance, hand-rolled) |
| **llm.py** | The provider-agnostic boundary. One `complete(messages, schema)` contract; vendor SDKs imported *only inside* their adapters; on-disk response cache. | `LLMClient` protocol, `AnthropicAdapter`, `OpenAIAdapter`, `OfflineClient`, `get_llm_client` (factory), `_cached` |
| **judge.py** | The LLM-as-judge: the rubric, the structured-output schema, and the prompt. | `RUBRIC`, `JUDGE_SCHEMA`, `build_messages`, `judge_answer` |
| **data.py** | Dataset tooling and golden-label IO. Downloads/slices SQuAD; reads/writes the golden CSV. | `prepare`, `load_slice`, `load_golden`, `_flatten_squad`, `_canonical_reference` |
| **calibrate.py** | The heart: how well each rater agrees with human truth. | `judge_vs_golden`, `em_vs_golden`, `_agreement_stats` (builds the confusion matrix → accuracy/precision/recall/F1/κ) |
| **report.py** | Presentation only: a results dict becomes a markdown report. Sections render only if present. | `render_report` + small formatting helpers |
| **cli.py** | Orchestration. Each subcommand reads/writes plain files so stages compose and a run is reproducible. | `cmd_prepare`, `cmd_run`, `cmd_score`, `cmd_calibrate`, `cmd_report`, `build_parser`, `main` |

## 5. End-to-end data flow

Each command is a pure function of files in → files out. That is what makes the pipeline resumable and a
run reproducible.

```
prepare    (SQuAD over the network)        ─► data/slice.jsonl              (≈200 {id,question,reference})
                                            ─► data/golden_template.csv     (50 blank rows to hand-label)
                                               data/golden_labels.csv       (committed; hand-labeled gold)

run        data/slice.jsonl                ─► results/run.jsonl             (+ candidate_answer per row)
score      results/run.jsonl               ─► results/scored.jsonl          (+ em, token_f1, judge per row)
                                            ─► results/leaderboard.json      (aggregate rates)
calibrate  data/golden_labels.csv          ─► results/calibration.json      (judge_vs_golden, em_vs_golden,
                                                                              divergence examples, headline)
report     leaderboard.json + calibration.json ─► results/report.md
                                                ─► results/latest.json       (the full combined snapshot)
```

Note the split: **`score` runs over the model's slice answers (the leaderboard); `calibrate` runs over the
hand-labeled golden file (the trust measurement).** They are deliberately separate inputs — the leaderboard
tells you how the model did, calibration tells you how much to believe the grader.

## 6. Trace one item end to end (the digestion exercise)

Take this committed golden row:

```
question:          "Which Pope sought to undermine Luther's theories?"
reference_answer:  "Pope Leo X"
candidate_answer:  "Leo X"
golden_correct:    correct        ← a human said this answer is right
```

1. **`cli.cmd_calibrate`** loads it via **`data.load_golden`** (which parses `golden_correct` → `True`).
2. Deterministic rater: **`metrics.exact_match("Leo X", "Pope Leo X")`** → `normalize` makes `"leo x"` vs
   `"pope leo x"` → **not equal → `False`**. Exact match calls this answer *wrong*.
3. Judge rater: **`judge.judge_answer`** builds the prompt (**`build_messages`**) and calls the client.
   Offline, **`llm.OfflineClient._judge`** reads the fenced payload and computes
   **`token_f1("Leo X", "Pope Leo X") = 0.8 ≥ 0.5` → `correct: True`**. The judge calls it *right*.
4. The three verdicts land in parallel lists: `em_correct=[…,False,…]`, `judge_correct=[…,True,…]`,
   `golden_correct=[…,True,…]`. Because gold=True, judge=True, EM=False, this row is also appended to
   `divergence_examples` (the "where EM lies" table).
5. **`calibrate._agreement_stats`** tallies it: for the judge it's a **true positive**; for exact match it's
   a **false negative** (correct answer wrongly rejected). Across all 50 rows that is why the judge's recall
   is 87% and EM's is only 39%.
6. **`report.render_report`** turns the two stat dicts into the calibration table, and this row shows up in
   the divergence table as `Pope Leo X | Leo X | ❌ | ✅ | ✅`.

When you can narrate that path from memory, you understand Yardstick.

## 7. Why the offline client is honest (not a cheat)

The offline backend exists so the whole pipeline — and the headline calibration — runs with **no API key
and no network**, which is what makes the result reproducible for anyone. It is a *real baseline*, not a
stub that peeks at the answer:

- As an **answerer** it sees only the question and returns "I don't know." It has no knowledge, so an
  offline *leaderboard* is honestly weak — plug in a real provider for a real leaderboard.
- As a **judge** it decides from `token_f1(candidate, reference) ≥ 0.5`. That is a legitimate, common
  heuristic grader that genuinely tolerates paraphrase better than strict exact match. So the
  judge-beats-EM result is a true property of the method, not a rigged number — and the same calibration
  even exposes *this* judge's own errors (it misfires on reversed facts like "more NADPH than ATP", and
  misses numeral/synonym paraphrases like "9" for "nine").

Swap in `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) and the real LLM judge replaces the heuristic behind the
exact same `LLMClient` interface; the harness then calibrates *that* judge instead. Nothing else changes.

## 8. Key design decisions (and where to read more)

- **Cohen's kappa is hand-rolled** (~10 lines in `metrics.py`), not pulled from scikit-learn — more
  teachable, fewer deps. The blueprint's Decisions log records this.
- **The golden CSV is self-contained**: each human label is pinned to the exact candidate text it judged
  (`id,question,reference,candidate,golden_correct`). That dissolves the chicken-and-egg between `run` and
  the gold, so `calibrate` reproduces without re-running any model.
- **Structured output, two ways**: the judge always returns `{correct, reason}` validated against
  `JUDGE_SCHEMA` — via a forced tool-call on Anthropic, via `json_schema` response format on OpenAI.

## 9. Known cleanups / notes for the reviewer

- **`numpy` is a declared-but-unused dependency** in `pyproject.toml`. Kappa ended up hand-rolled, so
  nothing imports numpy. Safe to remove from `dependencies` (ask me and I'll do it).
- The offline **leaderboard is intentionally all-zero** (the answerer is knowledge-free). That is expected;
  the leaderboard only becomes interesting with a real provider. The *calibration* section is the part that
  carries the v0.1 story.

## 10. Where v0.2 would extend this

The shape is built to grow without rework: a multi-model / multi-prompt leaderboard ranked by the
*calibrated* judge, a `yardstick diff <runA> <runB>` that flags metric regressions between two versioned
runs, and an optional HTML render of the same report dict. Each reuses the existing metric/judge/calibrate/
report spine untouched.
