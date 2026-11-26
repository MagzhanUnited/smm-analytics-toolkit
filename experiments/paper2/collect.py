"""Copy the metrics of finished fine-tuning runs into the repository as one small JSON file.

    python -I collect.py     ->  results/transformer_runs.json

Run outputs (weights, logits) stay outside the repository; this file holds what the paper reports.
It also compares the full-data transformer with the linear baseline on the same test reviews.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402


def stored_predictions(run: dict, logits, y, k: int):
    """Test predictions of a run, rebuilt from its saved logits.

    The logits are saved as float16, so two classes that were almost equal can become exactly equal and
    the arg-max of that review can differ from the one the run scored. The confusion matrix in
    ``metrics.json`` was computed from the unrounded logits and decides between the tied classes.
    """
    logits = np.asarray(logits, dtype=np.float32)
    target = np.asarray(run["test"]["confusion"])
    order = np.argsort(-logits, axis=1, kind="stable")
    pred = order[:, 0].copy()
    if (C.confusion(y, pred, k) == target).all():
        return pred
    rows = np.arange(len(logits))
    ties = np.flatnonzero(logits[rows, order[:, 0]] == logits[rows, order[:, 1]])
    assert 0 < len(ties) <= 12, (run["name"], len(ties))
    for mask in range(1, 2 ** len(ties)):
        trial = pred.copy()
        for bit, index in enumerate(ties):
            if mask >> bit & 1:
                trial[index] = order[index, 1]
        if (C.confusion(y, trial, k) == target).all():
            print(
                f"{run['name']}: {bin(mask).count('1')} tied review(s) resolved by the confusion matrix of the run"
            )
            return trial
    raise AssertionError(f"{run['name']}: the saved logits do not reproduce the confusion matrix of the run")


def seed_mean_bootstrap(
    y, preds: list, k: int, baselines: dict, resamples: int = 2000, seed: int = C.SEED
) -> dict:
    """Scores of one configuration averaged over its seeds, with percentile intervals over resampled test reviews.

    Training is not deterministic, so the reported score of the transformer is the mean over its seeds;
    the interval describes the sampling of the test set, as for the other models. The same resamples give
    the interval of the difference between that mean and each linear model. The resamples are those of
    ``common.bootstrap`` (same generator, same order), so one seed alone reproduces the interval of its run.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    acc, f1 = np.empty(resamples), np.empty(resamples)
    deltas = {name: np.empty(resamples) for name in baselines}
    for b in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        rows = [C.scores(C.confusion(y[idx], pred[idx], k)) for pred in preds]
        acc[b], f1[b] = np.mean([row["accuracy"] for row in rows]), np.mean([row["macro_f1"] for row in rows])
        for name, base in baselines.items():
            deltas[name][b] = f1[b] - C.scores(C.confusion(y[idx], base[idx], k))["macro_f1"]
    observed = [C.scores(C.confusion(y, pred, k)) for pred in preds]
    mean_f1 = float(np.mean([row["macro_f1"] for row in observed]))
    return {
        "seeds": len(preds),
        "resamples": resamples,
        "accuracy": float(np.mean([row["accuracy"] for row in observed])),
        "macro_f1": mean_f1,
        "accuracy_by_seed": [float(row["accuracy"]) for row in observed],
        "macro_f1_by_seed": [float(row["macro_f1"]) for row in observed],
        "accuracy_ci95": [round(float(np.percentile(acc, q)), 4) for q in (2.5, 97.5)],
        "macro_f1_ci95": [round(float(np.percentile(f1, q)), 4) for q in (2.5, 97.5)],
        "against": {
            name: {
                "delta_macro_f1": mean_f1 - float(C.scores(C.confusion(y, base, k))["macro_f1"]),
                "ci95": [round(float(np.percentile(deltas[name], q)), 4) for q in (2.5, 97.5)],
                "share_of_resamples_with_mean_better": round(float((deltas[name] > 0).mean()), 4),
            }
            for name, base in baselines.items()
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    args = parser.parse_args(argv)

    runs = {}
    for metrics in sorted(args.runs_dir.glob("*/metrics.json")):
        run = json.loads(metrics.read_text(encoding="utf-8"))
        if "suite" in run:
            run["suite"].pop("predictions", None)  # kept in the run directory
        runs[run["name"]] = run
    out = {"runs": runs, "comparisons": {}}
    # the watchdog pauses training on a serious thermal state; such a pause is counted in the training time of the run
    guard = args.runs_dir / "watchdog.log"
    out["watchdog_pauses"] = (
        [line.strip() for line in guard.read_text(encoding="utf-8").splitlines() if "training paused" in line]
        if guard.exists()
        else []
    )

    # paired comparison with the linear baseline, per task and text column
    for name, run in runs.items():
        if run["train_size"] < 100000:
            continue
        tag = f"{run['task']}_{run['text_col']}"
        baseline = args.runs_dir / f"baselines_{tag}_predictions.npz"
        if not baseline.exists():
            continue
        base = np.load(baseline)
        mine = np.load(args.runs_dir / name / "predictions.npz")
        y = base["y_test"].astype(int)
        assert (y == mine["y_test"]).all(), name
        k = len(run["labels"])
        pred = stored_predictions(run, mine["test_logits"], y, k)
        linear = base["linear"].astype(int)
        both = (pred == y) & (linear == y)
        out["comparisons"][name] = {
            "against": "linear",
            "paired_bootstrap": C.paired_bootstrap(y, linear, pred, k),
            "transformer_only_correct": int(((pred == y) & (linear != y)).sum()),
            "linear_only_correct": int(((pred != y) & (linear == y)).sum()),
            "both_correct": int(both.sum()),
            "both_wrong": int(((pred != y) & (linear != y)).sum()),
            # like with like: the linear model trained with the same treatment of class weights
            "against_linear_by_weighting": {
                weighting: C.paired_bootstrap(y, base[f"linear_weight_{weighting}"].astype(int), pred, k)
                for weighting in ("none", "balanced")
            },
        }
    # the two loss variants of the three-class task: means over the seeds that have finished
    out["seed_means"] = {}
    baseline = args.runs_dir / "baselines_p3_text_predictions.npz"
    if baseline.exists():
        base = np.load(baseline)
        y = base["y_test"].astype(int)
        linear = {
            name: base[name].astype(int)
            for name in ("linear", "linear_weight_none", "linear_weight_balanced")
        }
        for variant, tag in (("plain", ""), ("weighted", "_weighted")):
            names = [name for seed in (1, 2, 3) if (name := f"xlmr_p3_full{tag}_s{seed}") in runs]
            if not names:
                continue
            preds = []
            for name in names:
                saved = np.load(args.runs_dir / name / "predictions.npz")
                assert (saved["y_test"].astype(int) == y).all(), name
                preds.append(stored_predictions(runs[name], saved["test_logits"], y, 3))
            row = seed_mean_bootstrap(y, preds, 3, linear)
            row["runs"] = names
            row["best_valid_macro_f1_by_seed"] = [runs[name]["best_valid_macro_f1"] for name in names]
            # one seed alone must reproduce the interval that its run computed with common.bootstrap
            alone = seed_mean_bootstrap(y, preds[:1], 3, {})
            assert alone["macro_f1_ci95"] == runs[names[0]]["bootstrap"]["macro_f1_ci95"], (
                names[0],
                alone["macro_f1_ci95"],
            )
            out["seed_means"][variant] = row
    C.write_json(C.RESULTS / "transformer_runs.json", out)
    for name, run in runs.items():
        extra = f" suite {run['suite']['passed']}/{run['suite']['n']}" if "suite" in run else ""
        print(f"{name:<28} n={run['train_size']:>6} best_epoch={run['best_epoch']} acc={run['test']['accuracy']:.4f} "
              f"macro_f1={run['test']['macro_f1']:.4f} train={run['train_seconds']:.0f}s{extra}")  # fmt: skip
    for name, comparison in out["comparisons"].items():
        print("vs linear:", name, comparison["paired_bootstrap"])
    for variant, row in out["seed_means"].items():
        print(
            f"seed mean ({variant}, {row['seeds']} seed(s)): acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} ci={row['macro_f1_ci95']} "
            f"vs selected linear {row['against']['linear']['delta_macro_f1']:+.4f} {row['against']['linear']['ci95']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
