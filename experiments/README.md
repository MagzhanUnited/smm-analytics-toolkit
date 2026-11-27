# Experiments

Scripts that reproduce the evaluation of the toolkit. They use only the synthetic data in
`data/sample` and the package itself: no network access and no third-party packages.

```bash
pip install -e .
python experiments/run_experiments.py           # print the tables and rewrite results.json
python experiments/run_experiments.py --check   # fail if results.json would change
```

| File | Content |
|---|---|
| `behaviour_suite.py` | 239 behavioural test cases for sentiment classifiers in 14 capabilities: vocabulary, Russian and Kazakh negation, code-switching, mixed opinions, emoji |
| `run_experiments.py` | Evaluation on the labelled sample (bootstrap intervals, ablation of the negation rules), the behavioural suite, privacy checks, the planted-pattern check for the metrics, determinism and timing |
| `results.json` | Output of the last run; `--check` compares a fresh run with it (timing excluded) |

## What is measured

- **Labelled sample.** Accuracy and macro-F1 of the lexicon baseline on the 46 labelled comments,
  with percentile bootstrap 95 % intervals (10 000 resamples, fixed seed), and the same metrics
  with the pre-negation rule, the post-negation rule or both switched off.
- **Behavioural suite.** Pass rate per capability, in the spirit of CheckList (Ribeiro et al., 2020).
  A case passes when the predicted label is in the set of accepted labels.
- **Privacy.** Seven formats of fictional identifiers (handle, three phone formats, e-mail, two URL
  formats) are added before and after every sample comment. The script counts identifiers that
  survive `scrub_text` and predictions that change. It also probes identifier types that the
  toolkit does not cover; these are expected to survive and are reported as known gaps.
- **Analytics.** The mean engagement rate by format must reproduce the ranking planted by
  `scripts/make_sample_data.py`.
- **Determinism.** `smmkit report` is run under three hash seeds; the reports must be byte-identical.

## How to read the numbers

The sample and the suite are synthetic and were written together with the lexicon. The results
validate the pipeline and describe what the baseline can and cannot do; they are **not** estimates
of accuracy on real comments. The suite keeps the vocabulary inside the lexicon on purpose, so a
failed capability points to a missing rule (for example, negation separated from the polar word by
an intensifier, or Kazakh verbal negation), not to a missing word.

To evaluate another classifier, give it a `predict(texts) -> list[str]` method and pass its
predictions to `smmkit.sentiment.evaluate` and to the cases returned by `build_cases()`.

## Second paper

`paper2/` holds the evaluation on KazSAnDRA, a public corpus of real Kazakh reviews: the lexicon
baseline, a linear model and a fine-tuned XLM-R, with learning curves, the behavioural suite by
language and label changes under appended identifiers. Unlike the scripts above, it needs third-party
packages, the corpus and a pretrained model; see `paper2/README.md`.

## Third paper

`paper3/` holds the engagement-prediction study on the public UCI "Facebook metrics" data: what the attributes
known before publication predict when models are evaluated forward in time; see `paper3/README.md`.
