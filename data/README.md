# Data

Holds the dataset slice and golden labels. The dataset **bulk is not committed** (`data/` is gitignored);
a `prepare` step downloads it. The small **hand-labeled golden file** (`golden_labels.csv`) IS committed,
because it is your own evaluation gold.

## Source

A small slice (~200 items) of a public factual-QA dataset with reference answers (default: TriviaQA
no-context, or SQuAD v1.1 dev). Record the exact dataset and its license here once chosen. Public data
only, never work data.
