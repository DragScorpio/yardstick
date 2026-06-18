# Data

Holds the dataset slice and the golden labels. The dataset **bulk is not committed** (`data/` is
gitignored); `yardstick prepare` downloads it. The small **hand-labeled golden file**
(`golden_labels.csv`) IS committed, because it is the evaluation gold the calibration rests on.

## Source dataset

**SQuAD v1.1** (Stanford Question Answering Dataset), dev split, downloaded from
`https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json`.

- License: **CC BY-SA 4.0** (Rajpurkar, Zhang, Lopyrev, Liang, 2016).
- `prepare` flattens it to `(id, question, reference_answer)` rows (the canonical reference is the most
  common of SQuAD's acceptable answers), dedupes by question, deterministically shuffles with `seed=7`,
  and slices the first `n` (default 200) into `slice.jsonl`. Public data only, never work data.

## Golden labels (`golden_labels.csv`)

50 rows, each `id,question,reference_answer,candidate_answer,golden_correct`. The questions and references
are authentic SQuAD items; the **candidate answers and correct/incorrect labels are hand-authored** — this
is the human ground truth the judge and exact-match are calibrated against. The file is self-contained:
each label is pinned to the exact candidate text it judged, so `yardstick calibrate` is reproducible
without re-running a model.

The candidates were authored to span the realistic distribution that makes the calibration lesson land:
exact matches, correct paraphrases/supersets (which exact match wrongly rejects), a few hard paraphrases
(numerals/synonyms/morphology that even a token-overlap judge misses), clean wrong answers, and a few
lexical-overlap-but-wrong answers (reversed or shuffled facts that fool a bag-of-words judge). Labeling is
itself the exercise: it is how you learn what "correct" means for the task.
