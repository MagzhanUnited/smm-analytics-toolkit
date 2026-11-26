"""Why the linear model labels a negated negative word as negative.

    python -I linear_negation.py     ->  results/linear_negation.json

For every case of the behavioural suite in which a negated negative word must not be labelled
negative (capabilities N2 and N3), the score margin of the negative class over the best other class
is split into the contributions of three groups of features: word n-grams that contain the negation
word, the other word n-grams, and character n-grams. The model is the one saved by baselines.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from common import CAPABILITIES, build_cases  # noqa: E402

NEGATIONS = {"не", "нет", "без", "емес", "жоқ"}


def main() -> int:
    labels = C.LABELS["p3"]
    union, clf = joblib.load(C.ML_ROOT / "runs" / "linear_p3_text.joblib")
    names = np.asarray(union.get_feature_names_out())
    is_word = np.char.startswith(names.astype(str), "word__")
    has_negation = np.array([n.startswith("word__") and bool(NEGATIONS & set(n[6:].split())) for n in names])
    is_bigram = np.array([n.startswith("word__") and " " in n[6:] for n in names])
    groups = {
        "negation_bigrams": has_negation & is_bigram,
        "negation_word": has_negation & ~is_bigram,
        "other_words": is_word & ~has_negation,
        "character_ngrams": ~is_word,
    }
    keys = [c.key for c in CAPABILITIES if c.key.startswith(("N2", "N3"))]
    cases = [c for c in build_cases() if c.capability in keys]
    rows = []
    for case in cases:
        x = union.transform([case.text])
        scores = np.asarray(x @ clf.coef_.T).ravel() + clf.intercept_
        other = 1 + int(np.argmax(scores[1:]))  # the best class that is not negative
        direction = clf.coef_[0] - clf.coef_[other]
        dense = np.asarray(x.todense()).ravel()
        parts = {name: float(dense[mask] @ direction[mask]) for name, mask in groups.items()}
        margin = float(scores[0] - scores[other])
        assert abs(sum(parts.values()) + clf.intercept_[0] - clf.intercept_[other] - margin) < 1e-6
        present = [names[i][6:] for i in np.flatnonzero((dense != 0) & groups["negation_bigrams"])]
        rows.append({
            "capability": case.capability, "text": case.text, "label": labels[int(np.argmax(scores))],
            "margin_of_negative": round(margin, 3), "contributions": {k: round(v, 3) for k, v in parts.items()},
            "negation_bigrams_present": present,
        })  # fmt: skip
    summary = {}
    for key in keys:
        sub = [r for r in rows if r["capability"] == key]
        summary[key] = {
            "cases": len(sub),
            "labelled_negative": sum(r["label"] == "negative" for r in sub),
            "with_a_known_negation_bigram": sum(bool(r["negation_bigrams_present"]) for r in sub),
            "negation_bigrams_vote_against_negative": sum(
                r["contributions"]["negation_bigrams"] < 0 for r in sub
            ),
            "mean_contribution": {
                g: round(float(np.mean([r["contributions"][g] for r in sub])), 3) for g in groups
            },
        }
    C.write_json(
        C.RESULTS / "linear_negation.json",
        {"negation_words": sorted(NEGATIONS), "summary": summary, "cases": rows},
    )
    for key, row in summary.items():
        print(key, row)
    for r in rows:
        if r["capability"] == "N2-KK":
            print(
                "  ",
                r["text"],
                "->",
                r["label"],
                r["margin_of_negative"],
                r["contributions"],
                r["negation_bigrams_present"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
