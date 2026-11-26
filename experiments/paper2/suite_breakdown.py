"""Behavioural suite by language, with constant answers as reference points, for every evaluated model.

    python -I suite_breakdown.py

The suite of the toolkit has 239 constructed cases: 150 in Russian, 60 in Kazakh and 29 that mix the languages or
consist of an emoji. The corpus of this study is Kazakh, so the Russian cases test another language than the one the
models were trained on, and a model that always gives one answer already passes a large part of the suite. This script
therefore reports, for every model, the cases passed by language and by group of capabilities, next to the scores of
the three constant answers, and counts how often the words of the suite occur in the training set.

The suite predictions are read from where the evaluation stored them (results/baselines_p3_text.json for the lexicon
and the selected linear model, runs/<run>/metrics.json for XLM-R). The linear model without class weights is the one
trained configuration of the quality table whose suite predictions were not stored: it is refitted here with the stored
value of C on the stored features, and the refit must repeat its stored test predictions.

Output: results/suite_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # common.py, and the behavioural suite in experiments/

import behaviour_suite as S  # noqa: E402
import common as C  # noqa: E402
import numpy as np  # noqa: E402

GROUPS = {  # groups of capabilities reported in the paper
    "kazakh": ("V-KK", "N1-KK", "N2-KK", "N3-KK", "N4-KK"),
    "russian": ("V-RU", "N1-RU", "N2-RU", "N3-RU", "N4-RU", "N5-RU"),
    "other": ("CS", "MIX", "EMO"),
    "kazakh_negation": ("N1-KK", "N2-KK", "N3-KK", "N4-KK"),
    "kazakh_adjacent_negation": ("N1-KK", "N2-KK", "N3-KK"),
    "kazakh_negated_negative": ("N2-KK", "N3-KK"),
    "russian_negation": ("N1-RU", "N2-RU", "N3-RU", "N4-RU", "N5-RU"),
    "russian_negated_negative": ("N2-RU", "N3-RU"),
}
LABELS = C.LABELS["p3"]
RU_WORDS = sorted(
    set(S.RU_POSITIVE)
    | set(S.RU_NEGATIVE)
    | set(S.RU_GRADABLE)
    | set(S.RU_NEGATIVE_NOUNS)
    | {"не", "нет", "без"}
)
KK_WORDS = (
    "жақсы",
    "керемет",
    "тамаша",
    "жылдам",
    "мықты",
    "дәмді",
    "ұнады",
    "ұнайды",
    "рахмет",
    "нашар",
    "жаман",
    "баяу",
    "сапасыз",
    "емес",
    "жоқ",
)


def passed_by(cases, predictions, keys=None, expected_not_negative=False) -> tuple:
    """(cases passed, number of cases) among the capabilities ``keys`` (all when None)."""
    rows = [
        (c, p)
        for c, p in zip(cases, predictions, strict=True)
        if (keys is None or c.capability in keys)
        and (not expected_not_negative or c.expected == S.NOT_NEGATIVE)
    ]
    return sum(p in c.expected for c, p in rows), len(rows)


def breakdown(cases, predictions) -> dict:
    assert len(predictions) == len(cases) and set(predictions) <= set(LABELS), set(predictions)
    out = {"all": passed_by(cases, predictions)}
    out.update({name: passed_by(cases, predictions, keys) for name, keys in GROUPS.items()})
    out["expected_not_negative"] = passed_by(cases, predictions, expected_not_negative=True)
    out.update({cap.key: passed_by(cases, predictions, (cap.key,)) for cap in S.CAPABILITIES})
    return {
        "passed": {name: value[0] for name, value in out.items()},
        "answers": {label: predictions.count(label) for label in LABELS},
        "answers_russian": {
            label: sum(
                p == label
                for c, p in zip(cases, predictions, strict=True)
                if c.capability in GROUPS["russian"]
            )
            for label in LABELS
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    args = parser.parse_args(argv)

    cases = S.build_cases()
    sizes = {
        "all": len(cases),
        **{name: sum(c.capability in keys for c in cases) for name, keys in GROUPS.items()},
        "expected_not_negative": sum(c.expected == S.NOT_NEGATIVE for c in cases),
        **{cap.key: sum(c.capability == cap.key for c in cases) for cap in S.CAPABILITIES},
    }
    out = {
        "cases": len(cases),
        "groups": {name: list(keys) for name, keys in GROUPS.items()},
        "sizes": sizes,
        "constant": {label: breakdown(cases, [label] * len(cases)) for label in LABELS},
        "models": {},
    }
    # the best constant answer of every row: the reference below which a score says nothing
    out["best_constant"] = {
        name: max(out["constant"][label]["passed"][name] for label in LABELS) for name in sizes
    }

    base = json.loads((C.RESULTS / "baselines_p3_text.json").read_text(encoding="utf-8"))
    for name in ("lexicon", "linear"):
        suite = base["models"][name]["suite"]
        row = breakdown(cases, suite["predictions"])
        assert row["passed"]["all"] == suite["passed"] and all(
            row["passed"][c["key"]] == c["passed"] for c in suite["capabilities"]
        )
        out["models"]["linear_weight_balanced" if name == "linear" else name] = row
    assert base["models"]["linear"]["fit"]["selected"]["class_weight"] == "balanced"

    # the linear model without class weights: refitted as baselines.py fitted it (features fitted and transformed in one
    # step, the stored C); the refit must repeat the stored test predictions of that model
    import baselines as B
    import joblib
    from sklearn.linear_model import LogisticRegression

    frames = C.load_task(args.data_dir, "p3", "text")
    setting = base["linear_by_weighting"]["none"]["selected"]
    assert setting["class_weight"] == "none"
    stored = np.load(args.runs_dir / "baselines_p3_text_predictions.npz")
    saved = args.runs_dir / "linear_p3_text_weight_none.joblib"
    if saved.exists():  # an earlier refit: reused when it still repeats the stored test predictions
        union, plain = joblib.load(saved)
    if (
        not saved.exists()
        or (plain.predict(union.transform(frames["test"]["x"])) != stored["linear_weight_none"]).any()
    ):
        union = B.features("both")
        plain = LogisticRegression(C=setting["C"], class_weight=None, max_iter=1000).fit(
            union.fit_transform(frames["train"]["x"]), frames["train"]["y"]
        )
    differing = int(
        (plain.predict(union.transform(frames["test"]["x"])) != stored["linear_weight_none"]).sum()
    )
    assert differing == 0, f"the refit differs from the stored test predictions in {differing} reviews"
    texts = [c.text for c in cases]
    out["models"]["linear_weight_none"] = breakdown(
        cases, [LABELS[i] for i in plain.predict(union.transform(texts))]
    )
    out["linear_weight_none_refit"] = {
        "C": setting["C"],
        "iterations": int(np.max(plain.n_iter_)),
        "test_predictions_differing_from_stored": differing,
    }
    joblib.dump((union, plain), args.runs_dir / "linear_p3_text_weight_none.joblib", compress=0)

    # XLM-R: every finished full run of the two loss variants, and the superseded run that repeats one of them
    for path in sorted(args.runs_dir.glob("xlmr_p3_full*/metrics.json")) + sorted(
        (args.runs_dir / "superseded").glob("xlmr_p3_full*/metrics.json")
    ):
        run = json.loads(path.read_text(encoding="utf-8"))
        if "suite" not in run or "predictions" not in run["suite"]:
            continue
        row = breakdown(cases, run["suite"]["predictions"])
        assert row["passed"]["all"] == run["suite"]["passed"]
        confusion = np.asarray(run["test"]["confusion"])
        scores = C.scores(confusion)
        row.update(
            seed=run["seed"],
            class_weight=run["hyperparameters"].get("class_weight", "none"),
            epochs=run["hyperparameters"]["epochs"],
            best_epoch=run["best_epoch"],
            superseded=path.parent.parent.name == "superseded",
            predictions=run["suite"]["predictions"],
            test={"accuracy": float(scores["accuracy"]), "macro_f1": float(scores["macro_f1"])},
            probe_label_changes=run.get("privacy", {}).get("label_changes"),
        )
        out["models"][path.parent.name] = row

    # how often the words of the suite occur in the training set (reviews that contain the word as a token)
    train = frames["train"]["x"].astype(str).str.lower()
    count = lambda word: int(train.str.contains(rf"(?<!\w){re.escape(word)}(?!\w)", regex=True).sum())  # noqa: E731
    out["train_reviews"] = len(train)
    out["train_reviews_with_word"] = {
        "russian": {w: count(w) for w in RU_WORDS},
        "kazakh": {w: count(w) for w in KK_WORDS},
    }
    C.write_json(C.RESULTS / "suite_breakdown.json", out)

    show = (
        "all",
        "kazakh",
        "russian",
        "other",
        "kazakh_negation",
        "kazakh_adjacent_negation",
        "kazakh_negated_negative",
        "russian_negation",
        "russian_negated_negative",
        "expected_not_negative",
    )
    print("sizes:", {name: sizes[name] for name in show})
    for label in LABELS:
        print(f"always {label:<9}", {name: out["constant"][label]["passed"][name] for name in show})
    for name, row in out["models"].items():
        print(
            f"{name:<38}",
            {key: row["passed"][key] for key in show},
            "| answers",
            row["answers"],
            "| on Russian",
            row["answers_russian"],
        )
    print("train reviews with the word:", out["train_reviews_with_word"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
