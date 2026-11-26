"""Learning curve of the linear model for each treatment of the classes, with a bracketed choice of C.

    python -I linear_curve_weighting.py [--sizes 250 1000 4000 16000 64000 0] [--seeds 1 2 3]

For every training size and seed of the learning curve (0 stands for the full training set), the
model is fitted without class weights and with balanced weights. For each treatment C starts from the
grid of ``baselines.py``; while the value with the best validation macro-F1 lies at an end of the
range tried so far, the range is extended (by a factor of two downward, of four upward), so that no
reported point is limited by the grid. Both treatments are scored on the test set; the one with the
higher validation macro-F1 is the "selected" configuration of that subset.

Why this script exists: ``baselines.py --curve`` stores only the selected configuration and uses the
fixed grid, in which the unweighted model on small subsets picks the largest C (an audit of
26.11.2026). On the full training set the bracketed procedure must select exactly the model of
``baselines.py``; the script checks it. Results go to ``results/linear_curve_by_weighting.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines  # noqa: E402
import common as C  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

LOW_STEP, HIGH_STEP = 0.5, 4.0
LOW_CAP, HIGH_CAP = 2.0**-8, 4.0**9  # the search stops here even if the best value is still at the edge
WEIGHTINGS = ("none", "balanced")  # the order of baselines.py: at equal validation scores the first one wins


def fit_bracketed(train, valid, k: int) -> dict:
    """Fit both treatments of the classes; for each, extend the range of C until its best value is inside it."""
    start = time.perf_counter()
    union = baselines.features()
    x_train = union.fit_transform(train["x"])
    vectoriser_seconds = time.perf_counter() - start
    x_valid = union.transform(valid["x"])
    out = {
        "union": union,
        "vectoriser_seconds": round(vectoriser_seconds, 2),
        "features": int(x_train.shape[1]),
        "by_weighting": {},
    }
    for weighting in WEIGHTINGS:
        tried = {}

        def fit(c: float) -> None:
            begin = time.perf_counter()
            clf = LogisticRegression(
                C=c, class_weight=None if weighting == "none" else "balanced", max_iter=1000
            )
            clf.fit(x_train, train["y"])
            seconds = time.perf_counter() - begin
            f1 = C.scores(C.confusion(valid["y"], clf.predict(x_valid), k))["macro_f1"]
            tried[c] = {
                "valid_macro_f1": f1,
                "clf": clf,
                "fit_seconds": round(seconds, 2),
                "iterations": int(clf.n_iter_.max()),
            }

        for c in baselines.GRID_C:
            fit(c)
        while True:
            best = min(
                tried, key=lambda c: (-tried[c]["valid_macro_f1"], c)
            )  # at equal scores the smaller C, as in baselines.py
            if best == min(tried) and best > LOW_CAP:
                fit(best * LOW_STEP)
            elif best == max(tried) and best < HIGH_CAP:
                fit(best * HIGH_STEP)
            else:
                break
        out["by_weighting"][weighting] = {
            "C": best,
            "clf": tried[best]["clf"],
            "valid_macro_f1": tried[best]["valid_macro_f1"],
            "fit_seconds": tried[best]["fit_seconds"],
            "iterations": tried[best]["iterations"],
            "tried": {str(c): round(tried[c]["valid_macro_f1"], 4) for c in sorted(tried)},
            "fit_seconds_all": round(sum(t["fit_seconds"] for t in tried.values()), 2),
            "at_cap": best in (min(tried), max(tried)),
        }
    # the treatment with the higher validation score; at equal scores the smaller C, then no class weights
    out["selected"] = min(
        WEIGHTINGS,
        key=lambda w: (
            -out["by_weighting"][w]["valid_macro_f1"],
            out["by_weighting"][w]["C"],
            WEIGHTINGS.index(w),
        ),
    )
    out["configurations"] = sum(len(v["tried"]) for v in out["by_weighting"].values())
    out["grid_seconds"] = round(time.perf_counter() - start, 2)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=[250, 1000, 4000, 16000, 64000, 0],
        help="0: the full training set",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=C.RESULTS / "linear_curve_by_weighting.json")
    args = parser.parse_args(argv)

    frames = C.load_task(args.data_dir, "p3", "text")
    train, valid, test = frames["train"], frames["valid"], frames["test"]
    labels = C.LABELS["p3"]
    y_test, test_texts = test["y"].to_numpy(), test["x"].tolist()
    stored = json.loads((C.RESULTS / "baselines_p3_text.json").read_text(encoding="utf-8"))

    rows = []
    for size in args.sizes:
        for seed in (
            args.seeds if size else [0]
        ):  # the full set is fitted once: the linear model has no random seed
            subset = C.stratified_sample(train, size, seed) if size else train
            fit = fit_bracketed(subset, C.valid_for_size(valid, len(subset)), len(labels))
            x_test = fit["union"].transform(test_texts)
            row = {"size": len(subset), "seed": seed, "full": not size, "selected": fit["selected"], "features": fit["features"],
                   "configurations": fit["configurations"], "grid_seconds": fit["grid_seconds"], "vectoriser_seconds": fit["vectoriser_seconds"],
                   "by_weighting": {}}  # fmt: skip
            for weighting, result in fit["by_weighting"].items():
                s = C.summary(y_test, result.pop("clf").predict(x_test), labels)
                result["valid_macro_f1"] = round(result["valid_macro_f1"], 4)
                row["by_weighting"][weighting] = {**result, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                                                  "per_class_f1": {label: s["per_class"][label]["f1"] for label in labels}}  # fmt: skip
            if not size:  # on the full set the procedure must give the model that baselines.py reports
                main_model = stored["models"]["linear"]
                chosen = row["by_weighting"][row["selected"]]
                assert (
                    row["selected"] == main_model["fit"]["selected"]["class_weight"]
                    and chosen["C"] == main_model["fit"]["selected"]["C"]
                ), row
                assert abs(chosen["macro_f1"] - main_model["test"]["macro_f1"]) < 1e-4, row
                for weighting, entry in stored["linear_by_weighting"].items():
                    now = row["by_weighting"][weighting]
                    assert (
                        now["C"] == entry["selected"]["C"]
                        and abs(now["macro_f1"] - entry["test"]["macro_f1"]) < 1e-4
                    ), (weighting, now, entry)
            rows.append(row)
            cells = "  ".join(
                f"{w}: acc={v['accuracy']:.4f} f1={v['macro_f1']:.4f} (C={v['C']:g}{', at the cap' if v['at_cap'] else ''})"
                for w, v in row["by_weighting"].items()
            )
            print(
                f"n={row['size']:>6} seed={seed} selected={row['selected']:>8}  {cells}  [{row['configurations']} fits, {row['grid_seconds']:.0f} s]",
                flush=True,
            )

    C.write_json(
        args.out,
        {
            "task": "p3",
            "text_col": "text",
            "sizes": args.sizes,
            "seeds": args.seeds,
            "procedure": "bracketed",
            "base_grid": list(baselines.GRID_C),
            "steps": [LOW_STEP, HIGH_STEP],
            "caps": [LOW_CAP, HIGH_CAP],
            "curve": rows,
        },
    )
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
