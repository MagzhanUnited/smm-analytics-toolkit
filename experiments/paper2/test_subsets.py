"""Scores on subsets of the test set, from the stored test predictions (no model is run).

    python -I test_subsets.py

For the four trained configurations of the quality table (linear model and XLM-R, without and with class weights; for
XLM-R every finished seed) this script computes, at full precision:

* per-class precision, recall and F1 on the whole test set;
* reviews with a Kazakh negation: those that contain "жаман емес" or "нашар емес" ('not bad'), and those that contain
  the words "емес" or "жоқ" at all, against the other reviews;
* reviews by script (letters of the Kazakh alphabet, Cyrillic without them, Latin only) and reviews whose cleaned text
  also occurs in the training set.

Scores of XLM-R are means over its seeds. Intervals are percentile bootstrap intervals over the reviews of the subset
(the trained models stay fixed).

Output: results/test_subsets.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect  # noqa: E402
import common as C  # noqa: E402
import numpy as np  # noqa: E402

LABELS = C.LABELS["p3"]
K = len(LABELS)
NOT_BAD = re.compile(r"(?<!\w)(?:жаман|нашар)\s+емес(?!\w)")
NEGATION = re.compile(r"(?<!\w)(?:емес|жоқ)(?!\w)")
RESAMPLES = 2000


def exact(y, pred) -> dict:
    s = C.scores(C.confusion(y, pred, K), present_only=True)
    return {
        "accuracy": float(s["accuracy"]),
        "macro_f1": float(s["macro_f1"]),
        "per_class": {
            label: {key: float(s[key][i]) for key in ("precision", "recall", "f1")}
            for i, label in enumerate(LABELS)
        },
    }


def mean_scores(y, preds: list) -> dict:
    """Scores averaged over the runs of one configuration (one run for the linear model)."""
    rows = [exact(y, p) for p in preds]
    out = {
        "runs": len(rows),
        "n": int(len(y)),
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
        "macro_f1_by_run": [r["macro_f1"] for r in rows],
        "accuracy_by_run": [r["accuracy"] for r in rows],
        "per_class": {
            label: {
                key: float(np.mean([r["per_class"][label][key] for r in rows]))
                for key in ("precision", "recall", "f1")
            }
            for label in LABELS
        },
    }
    out["predicted"] = {
        label: float(np.mean([(p == i).sum() for p in preds])) for i, label in enumerate(LABELS)
    }
    return out


def interval(values) -> list:
    return [float(np.percentile(values, q)) for q in (2.5, 97.5)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    args = parser.parse_args(argv)

    frames = C.load_task(args.data_dir, "p3", "text")
    test = frames["test"]
    y = test["y"].to_numpy()
    base = np.load(args.runs_dir / "baselines_p3_text_predictions.npz")
    assert (base["y_test"].astype(int) == y).all()
    configs = {
        "linear": [base["linear_weight_none"].astype(int)],
        "linear_weighted": [base["linear_weight_balanced"].astype(int)],
        "xlmr": [],
        "xlmr_weighted": [],
    }
    runs = {"xlmr": [], "xlmr_weighted": []}
    for tag, key in (("", "xlmr"), ("_weighted", "xlmr_weighted")):
        for seed in (1, 2, 3):
            path = args.runs_dir / f"xlmr_p3_full{tag}_s{seed}"
            if not (path / "metrics.json").exists():
                continue
            run = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            stored = np.load(path / "predictions.npz")
            assert (stored["y_test"].astype(int) == y).all()
            configs[key].append(collect.stored_predictions(run, stored["test_logits"], y, K))
            runs[key].append(run["name"])
    out = {
        "test_reviews": int(len(y)),
        "runs": runs,
        "resamples": RESAMPLES,
        "whole": {name: mean_scores(y, preds) for name, preds in configs.items() if preds},
    }
    for name, row in out["whole"].items():  # the stored scores of the same models
        if name.startswith("linear"):
            stored = json.loads((C.RESULTS / "baselines_p3_text.json").read_text(encoding="utf-8"))[
                "linear_by_weighting"
            ]["balanced" if name.endswith("weighted") else "none"]["test"]
            assert (
                abs(stored["macro_f1"] - row["macro_f1"]) < 1e-4
                and abs(stored["accuracy"] - row["accuracy"]) < 1e-4
            )

    text = test["x"].astype(str).str.lower()
    stars = (test["label"].astype(int) + 1).to_numpy()
    masks = {
        "not_bad": text.str.contains(NOT_BAD).to_numpy(),
        "negation": text.str.contains(NEGATION).to_numpy(),
    }
    masks["no_negation"] = ~masks["negation"]
    script = test["text"].astype(str).map(C.script_type)
    for group in ("kazakh_letters", "cyrillic_without_kazakh_letters", "latin"):
        masks[group] = (script == group).to_numpy()
    train_cleaned = set(frames["train"]["text_cleaned"].astype(str))
    masks["text_also_in_train"] = test["text_cleaned"].astype(str).isin(train_cleaned).to_numpy()
    masks["unseen_text"] = ~masks["text_also_in_train"]
    domains = sorted(test["domain"].astype(str).unique())
    for domain in domains:  # the four sources of the corpus
        masks[f"domain_{domain}"] = (test["domain"].astype(str) == domain).to_numpy()

    out["subsets"] = {}
    for name, mask in masks.items():
        row = {
            "n": int(mask.sum()),
            "by_class": {label: int((y[mask] == i).sum()) for i, label in enumerate(LABELS)},
            "models": {
                config: mean_scores(y[mask], [p[mask] for p in preds])
                for config, preds in configs.items()
                if preds
            },
        }
        out["subsets"][name] = row
    out["subsets"]["not_bad"]["by_stars"] = {
        str(s): int((stars[masks["not_bad"]] == s).sum()) for s in range(1, 6)
    }

    # bootstrap over the reviews of a subset: the lead of XLM-R in accuracy over the linear model with the same class treatment
    rng = np.random.default_rng(C.SEED)

    def lead(mask, a: str, b: str, metric: str = "accuracy") -> dict:
        idx_all = np.flatnonzero(mask)
        value = lambda idx, preds: float(np.mean([exact(y[idx], p[idx])[metric] for p in preds]))  # noqa: E731
        observed = value(idx_all, configs[b]) - value(idx_all, configs[a])
        draws = np.empty(RESAMPLES)
        for r in range(RESAMPLES):
            idx = rng.choice(idx_all, len(idx_all))
            draws[r] = value(idx, configs[b]) - value(idx, configs[a])
        return {"metric": metric, "of": b, "over": a, "delta": observed, "ci95": interval(draws)}

    # the lead of the selected XLM-R variant candidates over the selected linear model in macro-F1, source by source
    out["domain_leads"] = {
        f"{b}_over_{a}": {domain: lead(masks[f"domain_{domain}"], a, b, "macro_f1") for domain in domains}
        for a, b in (("linear_weighted", "xlmr"), ("linear_weighted", "xlmr_weighted"))
        if configs[b]
    }
    out["leads"] = {}
    for a, b in (("linear", "xlmr"), ("linear_weighted", "xlmr_weighted"), ("linear_weighted", "xlmr")):
        if configs[b]:
            out["leads"][f"{b}_over_{a}"] = {
                subset: lead(masks[subset], a, b) for subset in ("negation", "no_negation")
            }

    # the drop on reviews in another script, with its interval: macro-F1 on the subset minus macro-F1 on reviews with Kazakh letters
    def gap(config: str, subset: str) -> dict:
        a_idx, b_idx = np.flatnonzero(masks[subset]), np.flatnonzero(masks["kazakh_letters"])
        value = lambda idx: float(np.mean([exact(y[idx], p[idx])["macro_f1"] for p in configs[config]]))  # noqa: E731
        draws, own = np.empty(RESAMPLES), np.empty(RESAMPLES)
        for r in range(RESAMPLES):
            a, b = rng.choice(a_idx, len(a_idx)), rng.choice(b_idx, len(b_idx))
            own[r] = value(a)
            draws[r] = value(b) - own[r]
        return {
            "macro_f1": value(a_idx),
            "macro_f1_ci95": interval(own),
            "below_kazakh_letters": value(b_idx) - value(a_idx),
            "ci95": interval(draws),
        }

    out["script_gaps"] = {
        config: {subset: gap(config, subset) for subset in ("latin", "cyrillic_without_kazakh_letters")}
        for config in configs
        if configs[config]
    }
    C.write_json(C.RESULTS / "test_subsets.json", out)

    for name, row in out["whole"].items():
        print(
            f"{name:<16} runs {row['runs']} acc {row['accuracy']:.4f} macro-F1 {row['macro_f1']:.4f} | F1 "
            + " ".join(f"{label} {row['per_class'][label]['f1']:.3f}" for label in LABELS)
            + f" | negative precision {row['per_class']['negative']['precision']:.3f} recall {row['per_class']['negative']['recall']:.3f}"
        )
    nb = out["subsets"]["not_bad"]
    print(f"reviews with жаман/нашар емес: {nb['n']}, by stars {nb['by_stars']}")
    for name, row in nb["models"].items():
        print(
            f"   {name:<16} predicted { ({k: round(v, 1) for k, v in row['predicted'].items()}) } accuracy {row['accuracy']:.3f}"
        )
    for subset in (
        "negation",
        "no_negation",
        "kazakh_letters",
        "cyrillic_without_kazakh_letters",
        "latin",
        "text_also_in_train",
        "unseen_text",
    ):
        row = out["subsets"][subset]
        print(
            f"{subset} (n={row['n']}, {row['by_class']}): "
            + "; ".join(
                f"{m} acc {r['accuracy']:.3f} F1 {r['macro_f1']:.3f}" for m, r in row["models"].items()
            )
        )
    for name, row in out["leads"].items():
        print(
            f"lead in accuracy, {name}: "
            + "; ".join(
                f"{s} {100 * r['delta']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"
                for s, r in row.items()
            )
        )
    for name, row in out["domain_leads"].items():
        print(
            f"lead in macro-F1 by source, {name}: "
            + "; ".join(
                f"{d} (n={out['subsets']['domain_' + d]['n']}) {100 * r['delta']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"
                for d, r in row.items()
            )
        )
    for config, row in out["script_gaps"].items():
        print(
            f"script gap, {config}: "
            + "; ".join(
                f"{s} F1 {r['macro_f1']:.3f} [{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}], below Kazakh letters by {r['below_kazakh_letters']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
                for s, r in row.items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
