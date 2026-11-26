# Experiments of the second paper

Scripts behind the paper "A Cost-Aware Evaluation of Lexicon, Linear, and Transformer Sentiment
Classifiers on Kazakh Customer Reviews". They evaluate the lexicon baseline of the toolkit, a linear
model and a fine-tuned XLM-R on KazSAnDRA, a public corpus of Kazakh reviews with star ratings.

Unlike `experiments/run_experiments.py`, these scripts need third-party packages, the corpus and, for
the transformer, a pretrained model. None of that is stored in the repository.

## What you need

- **Corpus.** KazSAnDRA (R. Yeshpanov and H. A. Varol, LREC-COLING 2024), licence CC BY 4.0, from
  <https://github.com/IS2AI/KazSAnDRA>. The scripts read the six official files of the unbalanced
  partitions: `01_pc_train_ib`, `04_pc_valid`, `05_pc_test` (polarity) and `06_sc_train_ib`,
  `09_sc_valid`, `10_sc_test` (ratings), each in a directory of its own under `data/kazsandra/`.
- **Model.** `xlm-roberta-base` in `models/xlm-roberta-base/`.
- **Packages.** Python 3.12 with numpy, pandas, scikit-learn, torch, transformers, joblib and
  threadpoolctl. The versions of the reported runs are recorded in every result file (`environment`).
- **Root for data, models and run outputs.** `~/Desktop/projects/aitu/smm-ml` (`ML_ROOT` in
  `common.py`); `run_queue.sh` also accepts the environment variable `ML_ROOT`.

Downloaded files are data: run the scripts as `python -I -B <script>.py`.

## Tasks

- `p3`, the task of the paper: the rating partition with one and two stars as negative, three stars
  as neutral, four and five stars as positive.
- `pc`: the official polarity partition, used only to compare with the published scores.

The two partitions were drawn independently and overlap, so a model trained on one task is never
evaluated on the test set of the other.

## Scripts, in the order of a full reproduction

| Script | What it does | Output |
|---|---|---|
| `data_profile.py` | Sizes, class shares, scripts, repeated texts, overlap of the partitions | `results/data_profile.json` |
| `baselines.py` | Majority class, lexicon, TF-IDF + logistic regression (C and class weights chosen on validation data); test scores, bootstrap intervals, subgroups, suite, fixed-string identifier test | `results/baselines_<task>_text.json`, model and predictions in `runs/` |
| `linear_curve_weighting.py` | Linear model with and without class weights on the training subsets of the learning curve, with a search for C that widens its range while the best value lies at an end | `results/linear_curve_by_weighting.json` |
| `finetune.py` | One fine-tuning run of XLM-R (seed, training size, loss with or without class weights); validation after every epoch, the best epoch is kept | `runs/<run>/metrics.json`, `predictions.npz`, optionally the model |
| `run_queue.sh`, `watchdog.sh` | All fine-tuning runs of the paper in order, with the probes of the saved models; a guard for temperature and memory | `runs/queue.log`, `runs/watchdog.log` |
| `baseline_timing.py` | Training time and throughput of the baselines on an idle machine, one thread | `results/baseline_timing.json` |
| `collect.py` | Metrics of all finished runs, paired bootstrap against the linear model, scores averaged over seeds | `results/transformer_runs.json` |
| `suite_breakdown.py` | Behavioural suite by language with the scores of the constant answers; refits the linear model without class weights and checks that the refit repeats the stored test predictions | `results/suite_breakdown.json` |
| `test_subsets.py` | From stored test predictions: per-class scores, reviews with a Kazakh negation word, reviews by script and by source, repeated texts | `results/test_subsets.json` |
| `probe_strings.py` | Label changes under appended strings: five random fictional identifiers per format, controls (one neutral word, three neutral words, digit groups), placeholders; `--summarise` pools the seeds and adds bootstrap intervals | `runs/appended_strings/<model>.npz`, `results/appended_strings.json` |
| `linear_negation.py` | Feature contributions of the linear model on the negation cases of the suite | `results/linear_negation.json` |
| `probe_transitions.py` | The fixed-string identifier test of the toolkit with the direction of the label changes and a neutral-word control. Kept for the record: the paper reports `probe_strings.py`, because the effect of a fixed string on the linear model depends on single tokens such as "123" | `results/identifier_transitions.json` |

`common.py` holds what the scripts share: loading of the partitions, metrics, bootstrap, the suite
runner and the fixed-string identifier test.

## Protocol of the fine-tuning runs

Fixed number of epochs by training size (40 for 250 reviews, 10 for 1,000, 6 for 4,000, 5 for 16,000
and more), validation after every epoch, the epoch with the best validation macro-F1 is kept, no
early stopping. Both losses (plain and class-weighted cross-entropy) at every size, three seeds; the
same seed gives the same training subset to the linear model and to XLM-R. Below 64,000 reviews,
models are selected on a 2,000-review subset of the validation set. The numbers of epochs were fixed
after pilot runs; three pilot runs with other budgets are kept in `runs/superseded/` and are not
collected. Training on the Metal backend is not deterministic: two runs with the same seed and
settings give slightly different models.

## Reading the result files

Every number of the paper is filled in from `results/*.json` by the build script of the manuscript,
which also checks each qualitative statement against the data. The result files do not contain
reviews of the corpus: text appears only as the constructed cases of the suite, as single words and
n-grams of the vocabulary (the strongest features of the linear model, the control words) and as the
fictional identifier strings.
