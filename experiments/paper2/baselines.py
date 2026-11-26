"""Baselines on KazSAnDRA: majority class, the toolkit's lexicon classifier, TF-IDF with logistic regression.

    python -I baselines.py --task p3 [--text-col text] [--curve]

Model selection uses the validation set only; the test set is used once per reported model.
Results go to ``results/baselines_<task>_<text-col>.json``; fitted models and test predictions go to
the runs directory outside the repository.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from run_experiments import VARIANTS  # noqa: E402
from run_experiments import classifier as lexicon_variant  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402

from smmkit.sentiment import LexiconClassifier  # noqa: E402

GRID_C = (0.25, 0.5, 1.0, 4.0, 16.0)
GRID_WEIGHT = (None, "balanced")
NAME_TO_P3 = {"negative": 0, "neutral": 1, "positive": 2}


def features(kind: str = "both") -> FeatureUnion:
    """Word uni- and bigrams (punctuation marks and emoji are tokens) plus character 2-5-grams."""
    parts = {
        "word": TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\w+|[^\w\s]", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        ),
        "char": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    }
    return FeatureUnion([(name, part) for name, part in parts.items() if kind in ("both", name)])


def fit_linear(train, valid, k: int, kind: str = "both") -> tuple:
    """Fit the vectoriser on ``train`` and choose C and class weighting by validation macro-F1."""
    start = time.perf_counter()
    union = features(kind)
    x_train = union.fit_transform(train["x"])
    vectoriser_seconds = time.perf_counter() - start
    x_valid = union.transform(valid["x"])
    best, trials, by_weight = None, [], {}
    for c in GRID_C:
        for weight in GRID_WEIGHT:
            start = time.perf_counter()
            clf = LogisticRegression(C=c, class_weight=weight, max_iter=1000)
            clf.fit(x_train, train["y"])
            seconds = time.perf_counter() - start
            f1 = C.scores(C.confusion(valid["y"], clf.predict(x_valid), k))["macro_f1"]
            trial = {
                "C": c,
                "class_weight": weight or "none",
                "valid_macro_f1": round(f1, 4),
                "fit_seconds": round(seconds, 2),
                "iterations": int(np.max(clf.n_iter_)),
            }
            trials.append(trial)
            if best is None or f1 > best[0]:
                best = (f1, clf, trial)
            name = weight or "none"
            if name not in by_weight or f1 > by_weight[name][0]:
                by_weight[name] = (f1, clf, trial)
    info = {
        "selected": best[2],
        "trials": trials,
        "vectoriser_seconds": round(vectoriser_seconds, 2),
        "features": int(x_train.shape[1]),
        "by_weight": {name: (clf, trial) for name, (_, clf, trial) in by_weight.items()},
    }
    return union, best[1], info


def top_features(union: FeatureUnion, clf: LogisticRegression, labels: tuple, n: int = 20) -> dict:
    """Word n-grams with the largest coefficients for the negative and the positive class."""
    names = union.get_feature_names_out()
    words = np.array([name.startswith("word__") for name in names])
    out = {}
    if len(labels) == 2:  # one coefficient vector: positive class against negative
        coef = np.where(words, clf.coef_[0], 0.0)
        out["negative"] = [names[i][6:] for i in np.argsort(coef)[:n]]
        out["positive"] = [names[i][6:] for i in np.argsort(-coef)[:n]]
        return out
    for index, label in enumerate(labels):
        coef = np.where(words, clf.coef_[index], -np.inf)
        out[label] = [names[i][6:] for i in np.argsort(-coef)[:n]]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=("pc", "p3"), required=True)
    parser.add_argument("--text-col", choices=("text", "text_cleaned"), default="text")
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    parser.add_argument(
        "--curve", action="store_true", help="also fit the linear model on subsets of the training set"
    )
    parser.add_argument(
        "--feature-ablation", action="store_true", help="also fit word-only and character-only models"
    )
    parser.add_argument("--sizes", type=int, nargs="*", default=[250, 1000, 4000, 16000, 64000])
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument(
        "--probe", type=int, default=1000, help="test reviews used in the identifier-insertion probe"
    )
    args = parser.parse_args(argv)

    frames = C.load_task(args.data_dir, args.task, args.text_col)
    train, valid, test = frames["train"], frames["valid"], frames["test"]
    labels = C.LABELS[args.task]
    k = len(labels)
    y_test, test_texts = test["y"].to_numpy(), test["x"].tolist()
    train_cleaned = set(train["text_cleaned"].astype(str))
    probe_texts = C.stratified_sample(test, args.probe)["x"].tolist()
    tag = f"{args.task}_{args.text_col}"
    out = {
        "task": args.task,
        "text_col": args.text_col,
        "labels": list(labels),
        "sizes": {split: len(df) for split, df in frames.items()},
        "class_counts": {split: np.bincount(df["y"], minlength=k).tolist() for split, df in frames.items()},
        "environment": C.environment(),
        "models": {},
    }
    predictions = {"y_test": y_test.astype(np.int8)}

    def battery(name: str, predict, extra: dict | None = None) -> None:
        pred = np.asarray(predict(test_texts))
        entry = {
            "test": C.summary(y_test, pred, labels),
            "bootstrap": C.bootstrap(y_test, pred, k),
            "subgroups": C.subgroups(test, pred, labels, train_cleaned),
            "privacy": C.privacy_probe(predict, probe_texts),
            "throughput_cpu": C.throughput(predict, test_texts),
        }
        if args.task == "p3":
            entry["suite"] = C.run_suite(lambda texts: [labels[i] for i in predict(texts)])
        entry.update(extra or {})
        out["models"][name] = entry
        predictions[name] = pred.astype(np.int8)
        suite = f" | suite {entry['suite']['passed']}/{entry['suite']['n']}" if "suite" in entry else ""
        print(f"{name:<10} acc={entry['test']['accuracy']:.4f} macro_f1={entry['test']['macro_f1']:.4f}"
              f" ci={entry['bootstrap']['macro_f1_ci95']}{suite} | {entry['throughput_cpu']['texts_per_second']:.0f} texts/s", flush=True)  # fmt: skip

    # 1. majority class of the training set
    majority = int(np.bincount(train["y"], minlength=k).argmax())
    pred = np.full(len(y_test), majority)
    out["models"]["majority"] = {"label": labels[majority], "test": C.summary(y_test, pred, labels)}
    predictions["majority"] = pred.astype(np.int8)
    print(f"majority   acc={out['models']['majority']['test']['accuracy']:.4f} macro_f1={out['models']['majority']['test']['macro_f1']:.4f}", flush=True)  # fmt: skip

    # 2. the toolkit's lexicon classifier, unchanged
    lexicon = LexiconClassifier()

    def predict_lexicon(texts):
        names = lexicon.predict(texts)
        if args.task == "p3":
            return np.array([NAME_TO_P3[name] for name in names])
        # the binary task has no neutral class: undecided reviews get the majority class
        return np.array(
            [0 if name == "negative" else 1 if name == "positive" else majority for name in names]
        )

    decided = float(np.mean([lexicon.score(text) != 0 for text in test_texts]))
    battery("lexicon", predict_lexicon, {"decided_share": round(decided, 4)})

    # the negation rules of the lexicon classifier on real reviews
    out["lexicon_ablation"] = {}
    for variant, flags in VARIANTS.items():
        with lexicon_variant(**flags) as lexicon:
            pred = predict_lexicon(test_texts)
        s = C.summary(y_test, pred, labels)
        out["lexicon_ablation"][variant] = {
            "accuracy": s["accuracy"],
            "macro_f1": s["macro_f1"],
            "correct": int((pred == y_test).sum()),
        }
    lexicon = LexiconClassifier()
    print(
        "lexicon ablation:",
        {name: row["macro_f1"] for name, row in out["lexicon_ablation"].items()},
        flush=True,
    )

    # 3. TF-IDF with logistic regression on the full training set
    start = time.perf_counter()
    union, clf, info = fit_linear(train, valid, k)
    info["grid_seconds"] = round(time.perf_counter() - start, 2)
    info["train_seconds"] = round(info["vectoriser_seconds"] + info["selected"]["fit_seconds"], 2)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.runs_dir / f"linear_{tag}.joblib"
    joblib.dump((union, clf), model_path, compress=0)
    info["model_megabytes"] = round(model_path.stat().st_size / 1e6, 1)
    info["top_word_features"] = top_features(union, clf, labels)

    def predict_linear(texts):
        return clf.predict(union.transform(texts))

    # the best model of each class-weighting scheme: plain training favours accuracy, balanced weights favour macro-F1
    out["linear_by_weighting"] = {}
    for name, (weight_clf, trial) in info.pop("by_weight").items():
        pred = weight_clf.predict(union.transform(test_texts))
        out["linear_by_weighting"][name] = {
            "selected": trial,
            "test": C.summary(y_test, pred, labels),
            "bootstrap": C.bootstrap(y_test, pred, k),
        }
        predictions[f"linear_weight_{name}"] = pred.astype(np.int8)
        print(
            f"linear ({name:<8}) acc={out['linear_by_weighting'][name]['test']['accuracy']:.4f} macro_f1={out['linear_by_weighting'][name]['test']['macro_f1']:.4f}",
            flush=True,
        )
    battery("linear", predict_linear, {"fit": info})
    predictions["linear_proba"] = clf.predict_proba(union.transform(test_texts)).astype(np.float16)

    # word-only and character-only variants of the linear model
    if args.feature_ablation:
        out["feature_ablation"] = {}
        for kind in ("word", "char"):
            kind_union, kind_clf, kind_info = fit_linear(train, valid, k, kind)
            pred = kind_clf.predict(kind_union.transform(test_texts))
            kind_info.pop("by_weight")
            row = {"test": C.summary(y_test, pred, labels), "selected": kind_info["selected"], "features": kind_info["features"],
                   "subgroups": C.subgroups(test, pred, labels, train_cleaned)}  # fmt: skip
            if args.task == "p3":
                row["suite"] = C.run_suite(
                    lambda texts, u=kind_union, c=kind_clf: [labels[i] for i in c.predict(u.transform(texts))]
                )
            out["feature_ablation"][kind] = row
            predictions[f"linear_{kind}"] = pred.astype(np.int8)
            print(
                f"linear {kind}-only macro_f1={row['test']['macro_f1']:.4f} acc={row['test']['accuracy']:.4f}",
                flush=True,
            )

    # 4. learning curve of the linear model
    if args.curve:
        out["curve"] = []
        for size in args.sizes:
            for seed in args.seeds:
                subset = C.stratified_sample(train, size, seed)
                start = time.perf_counter()
                sub_union, sub_clf, sub_info = fit_linear(subset, C.valid_for_size(valid, size), k)
                sub_info.pop("by_weight")
                seconds = time.perf_counter() - start
                pred = sub_clf.predict(sub_union.transform(test_texts))
                s = C.summary(y_test, pred, labels)
                row = {
                    "size": len(subset),
                    "seed": seed,
                    "accuracy": s["accuracy"],
                    "macro_f1": s["macro_f1"],
                    "selected": sub_info["selected"],
                    "features": sub_info["features"],
                    "grid_seconds": round(seconds, 2),
                }
                out["curve"].append(row)
                print(
                    f"curve n={row['size']:>6} seed={seed} macro_f1={row['macro_f1']:.4f} acc={row['accuracy']:.4f}",
                    flush=True,
                )

    C.write_json(C.RESULTS / f"baselines_{tag}.json", out)
    np.savez_compressed(args.runs_dir / f"baselines_{tag}_predictions.npz", **predictions)
    print("written", C.RESULTS / f"baselines_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
