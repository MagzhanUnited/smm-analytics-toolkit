"""Which labels change when an identifier is inserted, and in which direction.

    python -I probe_transitions.py --model linear
    python -I probe_transitions.py --model xlmr_p3_full_s1      # a run saved with --save-model, on the processor
    python -I probe_transitions.py --model xlmr_p3_full_s2 --device mps   # the same on the graphics processor (the queue does this between runs)
    python -I probe_transitions.py --model linear --control     # the same insertion with neutral words instead of identifiers

The counts of changed labels repeat ``privacy_probe`` (the script checks that they are equal); this
script adds what the probe does not record: the label before and after the change, the rate per
label of the unperturbed text, and the rate per identifier format. Results are merged into
``results/identifier_transitions.json``.

``--control`` answers what an inserted word that is not an identifier does. The control words are the
three words with the most class-neutral distribution among the 300 most frequent words of the training
set (the smallest Kullback-Leibler divergence between the classes of the reviews that contain the word
and the classes of all training reviews); the test set plays no part in the choice.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from common import PII_INJECTIONS, scrub_text  # noqa: E402


def control_words(train, k: int, frequent: int = 300, n: int = 3) -> list:
    """The ``n`` most class-neutral words among the ``frequent`` most frequent words of the training set."""
    token = re.compile(r"(?u)\w+")
    prior = np.bincount(train["y"], minlength=k) / len(train)
    reviews, by_class = Counter(), {}
    for text, y in zip(train["x"], train["y"], strict=True):
        for word in set(token.findall(text.lower())):
            reviews[word] += 1
            by_class.setdefault(word, [0] * k)[y] += 1
    rows = []
    for word, count in reviews.most_common(frequent):
        if word.isalpha() and len(word) > 1:
            share = np.array(by_class[word]) / count
            divergence = float(
                sum(share[i] * math.log(share[i] / prior[i]) for i in range(k) if share[i] > 0)
            )
            rows.append(
                {
                    "word": word,
                    "reviews": count,
                    "divergence": round(divergence, 5),
                    "class_shares": [round(float(s), 3) for s in share],
                }
            )
    return sorted(rows, key=lambda row: (row["divergence"], row["word"]))[:n]


def transitions(predict, texts: list, labels: tuple, injections=PII_INJECTIONS, treatments=None) -> dict:
    out = {}
    for treatment, treat in treatments or (("left_in_text", lambda t: t), ("placeholder", scrub_text)):
        base = np.asarray(predict([treat(t) for t in texts]))
        moves = np.zeros((len(labels), len(labels)), dtype=int)
        formats = []
        for kind, pii, _ in injections:
            changed_n = 0
            for candidates in ([f"{pii} {t}" for t in texts], [f"{t} {pii}" for t in texts]):
                pred = np.asarray(predict([treat(c) for c in candidates]))
                np.add.at(moves, (base, pred), 1)
                changed_n += int((pred != base).sum())
            alone = labels[
                int(np.asarray(predict([treat(pii)]))[0])
            ]  # label of the identifier (or its placeholder) without any review
            formats.append(
                {
                    "kind": kind,
                    "identifier": pii,
                    "changed": changed_n,
                    "perturbed": 2 * len(texts),
                    "label_alone": alone,
                }
            )
        perturbed_per_label = moves.sum(1)
        out[treatment] = {
            "changed": int(moves.sum() - np.trace(moves)),
            "moves": {
                f"{labels[a]}>{labels[b]}": int(moves[a, b])
                for a in range(len(labels))
                for b in range(len(labels))
                if a != b and moves[a, b]
            },
            "by_label_before": {
                labels[a]: {
                    "perturbed": int(perturbed_per_label[a]),
                    "changed": int(perturbed_per_label[a] - moves[a, a]),
                }
                for a in range(len(labels))
            },
            "by_format": formats,
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help="'lexicon', 'linear' or the name of a run with a saved model"
    )
    parser.add_argument("--task", default="p3")
    parser.add_argument("--probe", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    parser.add_argument("--control", action="store_true", help="insert neutral words instead of identifiers")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device for a fine-tuned model: cpu (default, usable while a run trains) or mps",
    )
    args = parser.parse_args(argv)

    labels = C.LABELS[args.task]
    frames = C.load_task(args.data_dir, args.task, "text")
    texts = [t.strip() for t in C.stratified_sample(frames["test"], args.probe)["x"].tolist()]
    if args.model == "lexicon":
        from baselines import NAME_TO_P3

        from smmkit.sentiment import LexiconClassifier

        assert args.task == "p3"
        lexicon = LexiconClassifier()
        predict = lambda batch: np.array([NAME_TO_P3[name] for name in lexicon.predict(batch)])  # noqa: E731
        recorded = json.loads((C.RESULTS / f"baselines_{args.task}_text.json").read_text(encoding="utf-8"))[
            "models"
        ]["lexicon"]["privacy"]
    elif args.model == "linear":
        import joblib

        union, clf = joblib.load(args.runs_dir / f"linear_{args.task}_text.joblib")
        predict = lambda batch: clf.predict(union.transform(batch))  # noqa: E731
        recorded = json.loads((C.RESULTS / f"baselines_{args.task}_text.json").read_text(encoding="utf-8"))[
            "models"
        ]["linear"]["privacy"]
    else:
        import finetune as F

        model = F.Classifier(args.runs_dir / args.model / "model", len(labels), 128, args.device)
        predict = model.predict
        recorded = json.loads((args.runs_dir / args.model / "metrics.json").read_text(encoding="utf-8"))[
            "privacy"
        ]
    path = C.RESULTS / "identifier_transitions.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"probe": {}, "models": {}}
    if args.control:
        words = control_words(frames["train"], len(labels))
        injections = [("word", row["word"], row["word"]) for row in words]
        perturbed = [c for _, word, _ in injections for t in texts for c in (f"{word} {t}", f"{t} {word}")]
        # the scrubber leaves a control word alone (it changes a perturbed text only where it changes the review itself),
        # so the placeholder treatment would repeat this one
        assert all(
            scrub_text(f"{word} {t}") == f"{word} {scrub_text(t)}"
            and scrub_text(f"{t} {word}") == f"{scrub_text(t)} {word}"
            for _, word, _ in injections
            for t in texts
        )
        result = transitions(predict, texts, labels, injections, (("left_in_text", lambda t: t),))[
            "left_in_text"
        ]
        result["words"] = words
        result["perturbed"] = len(perturbed)
        out.setdefault("controls", {})[args.model] = result
        C.write_json(path, out)
        print(
            f"{args.model} control words {[row['word'] for row in words]}: changed {result['changed']} of {result['perturbed']}; moves {result['moves']}"
        )
        print(
            "   by label before:",
            {key: f"{v['changed']}/{v['perturbed']}" for key, v in result["by_label_before"].items()},
        )
        print("   by word:", [(f["identifier"], f["changed"], f["label_alone"]) for f in result["by_format"]])
        return 0
    result = transitions(predict, texts, labels)
    for treatment in ("left_in_text", "placeholder"):
        assert result[treatment]["changed"] == recorded["label_changes"][treatment], (
            treatment,
            result[treatment]["changed"],
            recorded["label_changes"],
        )
    # the third treatment of the probe (scrub, then drop the placeholder) is a property of the scrubber, not of a model:
    # when the classifier input equals the input of the unperturbed review, no label can change
    removed = lambda t: C.drop_placeholders(scrub_text(t))  # noqa: E731
    restored = sum(
        removed(c) == removed(t)
        for _, pii, _ in PII_INJECTIONS
        for t in texts
        for c in (f"{pii} {t}", f"{t} {pii}")
    )
    placeholder_inputs = len(
        {scrub_text(c) for _, pii, _ in PII_INJECTIONS for t in texts for c in (f"{pii} {t}", f"{t} {pii}")}
    )
    out["probe"] = {
        "texts": len(texts),
        "identifier_formats": len(PII_INJECTIONS),
        "perturbed": 2 * len(texts) * len(PII_INJECTIONS),
        "restored_by_removal": int(restored),
        "distinct_inputs_with_placeholder": placeholder_inputs,
    }
    out["models"][args.model] = result
    C.write_json(path, out)
    for treatment, row in result.items():
        print(f"{args.model} {treatment}: changed {row['changed']}; moves {row['moves']}")
        print(
            "   by label before:",
            {k: f"{v['changed']}/{v['perturbed']}" for k, v in row["by_label_before"].items()},
        )
        print("   by format:", [(f["identifier"], f["changed"], f["label_alone"]) for f in row["by_format"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
