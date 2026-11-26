"""Label changes under appended strings: random fictional identifiers against controls.

    python -I probe_strings.py --model linear                        # the selected linear model (class weights)
    python -I probe_strings.py --model linear_weight_none            # saved by suite_breakdown.py
    python -I probe_strings.py --model xlmr_p3_full_s2 --device mps  # a run saved with --save-model
    python -I probe_strings.py --summarise                           # results/appended_strings.json

Why this probe replaces the fixed strings of probe_transitions.py in the paper. That probe inserts one fixed string per
identifier format. For a bag-of-n-grams model the effect of such a string is the sum of the weights of its tokens, and
the three fixed strings with a large effect share the token "123": in this corpus reviewers write their rating into the
text, so single digits are rating words. One string per format therefore measures that string, not the format. Here
every format is drawn five times with random digits and letters, and controls separate what an identifier is from what
it is made of:

  word        one frequent word whose reviews have the class distribution of the whole training set (five words)
  phrase      three such words in a row (a control with several tokens)
  digits      the digit groups of a phone number without its prefix, e.g. "952 86 41"; the phone formats use the
              same digits, so a phone number and its control differ in the prefix and the punctuation only

Every string is put once before and once after each of 1,000 test reviews (the sample of the other probes). The label of
the perturbed text is compared with the label that the same model gives to the review itself. For the identifiers a
second treatment replaces the string by the placeholder of the toolkit (``scrub_text``).

Per model the predictions are stored outside the repository (runs/appended_strings/<model>.npz). ``--summarise`` turns
them into results/appended_strings.json: shares of changed labels per group of strings for the linear models and for the
two XLM-R variants (all finished seeds of a variant together), by the label before the insertion, with the range over
the strings of a group and percentile bootstrap intervals over the 1,000 reviews.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from common import scrub_text  # noqa: E402
from probe_transitions import control_words  # noqa: E402

INSTANCES = 5
RESAMPLES = 2000
LETTERS = "abcdefghijklmnopqrstuvwxyz"
PLACEHOLDER = {
    "handle": "<USER>",
    "email": "<EMAIL>",
    "url_plain": "<URL>",
    "url_digits": "<URL>",
    "phone_spaced": "<PHONE>",
    "phone_brackets": "<PHONE>",
    "phone_joined": "<PHONE>",
}
IDENTIFIER_GROUPS = tuple(PLACEHOLDER)
CONTROL_GROUPS = ("word", "phrase", "digits")
# groups reported together
POOLS = {
    "phone": ("phone_spaced", "phone_brackets", "phone_joined"),
    "address": ("handle", "email", "url_plain", "url_digits"),
    "identifier": IDENTIFIER_GROUPS,
}


def make_strings(train, k: int) -> list:
    """The strings of the probe: [{"group", "text"}], the same for every model (fixed random generator)."""
    rng = np.random.default_rng(C.SEED)
    name = lambda low, high: "".join(rng.choice(list(LETTERS), int(rng.integers(low, high + 1))))  # noqa: E731
    digits = lambda n: "".join(str(d) for d in rng.integers(0, 10, n))  # noqa: E731
    neutral = [row["word"] for row in control_words(train, k, frequent=300, n=10)]
    rows = [{"group": "word", "text": word} for word in neutral[:INSTANCES]]
    rows += [
        {"group": "phrase", "text": " ".join(rng.choice(neutral, 3, replace=False))} for _ in range(INSTANCES)
    ]
    numbers = [(digits(3), digits(2), digits(2)) for _ in range(INSTANCES)]
    rows += [{"group": "digits", "text": f"{a} {b} {c}"} for a, b, c in numbers]
    rows += [{"group": "phone_spaced", "text": f"+7 700 {a} {b} {c}"} for a, b, c in numbers]
    rows += [{"group": "phone_brackets", "text": f"8 (700) {a}-{b}-{c}"} for a, b, c in numbers]
    rows += [{"group": "phone_joined", "text": f"+7700{a}{b}{c}"} for a, b, c in numbers]
    rows += [{"group": "handle", "text": f"@{name(5, 8)}_{digits(2)}"} for _ in range(INSTANCES)]
    rows += [{"group": "email", "text": f"{name(4, 7)}.{name(4, 7)}@example.com"} for _ in range(INSTANCES)]
    rows += [{"group": "url_plain", "text": f"www.example.org/{name(5, 8)}"} for _ in range(INSTANCES)]
    rows += [
        {"group": "url_digits", "text": f"https://example.org/{name(5, 8)}/{digits(3)}"}
        for _ in range(INSTANCES)
    ]
    for row in rows:  # the toolkit must recognise every identifier and leave every control alone
        expected = PLACEHOLDER.get(row["group"], row["text"])
        assert scrub_text(row["text"]) == expected, (row, scrub_text(row["text"]))
    assert len({row["text"] for row in rows}) == len(rows)
    return rows


def probe_texts(args) -> tuple:
    frames = C.load_task(args.data_dir, "p3", "text")
    texts = [t.strip() for t in C.stratified_sample(frames["test"], args.probe)["x"].tolist()]
    return frames, texts


def load_model(args, labels):
    if args.model in ("linear", "linear_weight_none"):
        import joblib

        name = "linear_p3_text.joblib" if args.model == "linear" else "linear_p3_text_weight_none.joblib"
        union, clf = joblib.load(args.runs_dir / name)
        return lambda batch: clf.predict(union.transform(batch))
    import finetune as F

    return F.Classifier(args.runs_dir / args.model / "model", len(labels), 128, args.device).predict


def run_model(args) -> int:
    labels = C.LABELS["p3"]
    frames, texts = probe_texts(args)
    strings = make_strings(frames["train"], len(labels))
    predict = load_model(args, labels)
    identifiers = [i for i, row in enumerate(strings) if row["group"] in PLACEHOLDER]
    perturbed = {
        (i, pos): [f"{row['text']} {t}" if pos == 0 else f"{t} {row['text']}" for t in texts]
        for i, row in enumerate(strings)
        for pos in (0, 1)
    }
    scrubbed = {key: [scrub_text(c) for c in perturbed[key]] for key in perturbed if key[0] in identifiers}
    alone = [row["text"] for row in strings] + sorted(set(PLACEHOLDER.values()))
    unique = sorted(
        {c for batch in perturbed.values() for c in batch}
        | {c for batch in scrubbed.values() for c in batch}
        | set(texts)
        | set(alone)
    )
    index = {c: n for n, c in enumerate(unique)}
    answers = np.asarray(predict(unique)).astype(np.int8)
    base = answers[[index[t] for t in texts]]
    pred = np.stack(
        [
            np.stack([answers[[index[c] for c in perturbed[(i, pos)]]] for pos in (0, 1)])
            for i in range(len(strings))
        ]
    )
    pred_placeholder = np.stack(
        [np.stack([answers[[index[c] for c in scrubbed[(i, pos)]]] for pos in (0, 1)]) for i in identifiers]
    )
    out_dir = args.runs_dir / "appended_strings"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / f"{args.model}.npz",
        base=base,
        pred=pred,
        pred_placeholder=pred_placeholder,
        identifiers=np.asarray(identifiers),
        alone=answers[[index[c] for c in alone]],
        strings=np.asarray([row["text"] for row in strings]),
        groups=np.asarray([row["group"] for row in strings]),
    )
    changed = (pred != base).mean(axis=(1, 2))
    print(
        f"{args.model}: {len(unique)} distinct inputs; labels of the reviews {np.bincount(base, minlength=len(labels)).tolist()}"
    )
    for group in CONTROL_GROUPS + IDENTIFIER_GROUPS:
        rows = [i for i, row in enumerate(strings) if row["group"] == group]
        print(
            f"   {group:<15} changed {100 * changed[rows].mean():5.1f} %  per string: "
            + ", ".join(f"{strings[i]['text']} {100 * changed[i]:.1f}" for i in rows)
        )
    print(f"   placeholder     changed {100 * (pred_placeholder != base).mean():5.1f} %")
    return 0


def interval(values) -> list:
    return [float(np.percentile(values, q)) for q in (2.5, 97.5)]


def summarise(args) -> int:
    labels = C.LABELS["p3"]
    frames, texts = probe_texts(args)
    strings = make_strings(frames["train"], len(labels))
    groups = np.asarray([row["group"] for row in strings])
    store = args.runs_dir / "appended_strings"
    configs = {
        "linear_weighted": ["linear"],
        "linear": ["linear_weight_none"],
        "xlmr": [f"xlmr_p3_full_s{seed}" for seed in (1, 2, 3)],
        "xlmr_weighted": [f"xlmr_p3_full_weighted_s{seed}" for seed in (1, 2, 3)],
    }
    words = control_words(frames["train"], len(labels), frequent=300, n=10)
    out = {
        "reviews": len(texts),
        "instances_per_group": INSTANCES,
        "resamples": RESAMPLES,
        "strings": strings,
        "neutral_words": words,
        "placeholders": sorted(set(PLACEHOLDER.values())),
        "configurations": {},
    }
    rng = np.random.default_rng(C.SEED)
    resamples = rng.integers(0, len(texts), (RESAMPLES, len(texts)))
    for config, names in configs.items():
        runs = [n for n in names if (store / f"{n}.npz").exists()]
        if not runs:
            continue
        data = [np.load(store / f"{n}.npz") for n in runs]
        for d in data:  # the predictions belong to this set of strings
            assert d["strings"].tolist() == [row["text"] for row in strings] and len(d["base"]) == len(texts)
        base = np.stack([d["base"] for d in data])  # runs x reviews
        changed = np.stack([d["pred"] != d["base"] for d in data])  # runs x strings x 2 x reviews
        moved_to = np.stack([d["pred"] for d in data])
        identifiers = data[0]["identifiers"]
        changed_ph = np.stack(
            [d["pred_placeholder"] != d["base"] for d in data]
        )  # runs x identifier strings x 2 x reviews
        moved_ph = np.stack([d["pred_placeholder"] for d in data])

        def stats(mask_strings, ch, moved) -> dict:
            """Shares for the strings ``mask_strings`` of the array ``ch`` (runs x strings x 2 x reviews)."""
            sub, to = ch[:, mask_strings], moved[:, mask_strings]
            per_review = sub.sum(axis=(1, 2)).astype(
                float
            )  # runs x reviews: changed insertions of each review
            n_ins = sub.shape[1] * sub.shape[2]
            row = {
                "strings": int(sub.shape[1]),
                "runs": int(sub.shape[0]),
                "insertions": int(sub.shape[0] * n_ins * sub.shape[3]),
                "changed": float(sub.mean()),
                "ci95": interval(
                    per_review[:, resamples].sum(axis=(0, 2)) / (sub.shape[0] * n_ins * sub.shape[3])
                ),
                "per_string": [float(x) for x in sub.mean(axis=(0, 2, 3))],
                "per_run": [float(x) for x in sub.mean(axis=(1, 2, 3))],
                "by_label_before": {},
            }
            for i, label in enumerate(labels):
                own = base == i  # runs x reviews
                total = own.sum() * n_ins
                part = {
                    "reviews": float(own.sum() / own.shape[0]),
                    "changed": float((per_review * own).sum() / total) if total else None,
                }
                if total:
                    numerators = (per_review * own)[:, resamples].sum(axis=(0, 2))
                    denominators = own[:, resamples].sum(axis=(0, 2)) * n_ins
                    part["ci95"] = interval(numerators[denominators > 0] / denominators[denominators > 0])
                    part["per_run"] = [
                        float((per_review[r] * own[r]).sum() / (own[r].sum() * n_ins))
                        if own[r].sum()
                        else None
                        for r in range(own.shape[0])
                    ]
                    part["to"] = {
                        labels[j]: float(((to == j) & sub & own[:, None, None, :]).sum() / total)
                        for j in range(len(labels))
                        if j != i
                    }
                row["by_label_before"][label] = part
            return row, per_review / n_ins

        entry = {
            "runs": runs,
            "reviews_by_label": {
                label: float((base == i).sum() / base.shape[0]) for i, label in enumerate(labels)
            },
            "groups": {},
        }
        per_review_share = {}
        named = {g: groups == g for g in CONTROL_GROUPS + IDENTIFIER_GROUPS}
        named.update({pool: np.isin(groups, members) for pool, members in POOLS.items()})
        for name, mask in named.items():
            entry["groups"][name], per_review_share[name] = stats(mask, changed, moved_to)
        ident_groups = groups[identifiers]
        entry["groups"]["placeholder"], per_review_share["placeholder"] = stats(
            np.ones(len(identifiers), dtype=bool), changed_ph, moved_ph
        )
        for kind in sorted(set(PLACEHOLDER.values())):
            members = [g for g, p in PLACEHOLDER.items() if p == kind]
            entry["groups"][f"placeholder_{kind.strip('<>').lower()}"], _ = stats(
                np.isin(ident_groups, members), changed_ph, moved_ph
            )
        # differences to the word control on the same resampled reviews: overall and for the reviews labelled negative
        entry["against_word"] = {}
        negative = base == labels.index("negative")
        for name in per_review_share:
            if name == "word":
                continue
            diff = per_review_share[name] - per_review_share["word"]  # runs x reviews
            row = {"delta": float(diff.mean()), "ci95": interval(diff[:, resamples].mean(axis=(0, 2)))}
            num = (diff * negative)[:, resamples].sum(axis=(0, 2))
            den = negative[:, resamples].sum(axis=(0, 2))
            row["negative"] = {
                "delta": float((diff * negative).sum() / negative.sum()),
                "ci95": interval(num[den > 0] / den[den > 0]),
            }
            entry["against_word"][name] = row
        entry["label_alone"] = {
            str(s): labels[int(a)]
            for s, a in zip(
                list(data[0]["strings"]) + sorted(set(PLACEHOLDER.values())), data[0]["alone"], strict=True
            )
        }
        out["configurations"][config] = entry
    C.write_json(C.RESULTS / "appended_strings.json", out)
    for config, entry in out["configurations"].items():
        print(
            f"{config} ({len(entry['runs'])} run(s)); reviews by label { ({k: round(v) for k, v in entry['reviews_by_label'].items()}) }"
        )
        for name, row in entry["groups"].items():
            by = row["by_label_before"]
            cells = " ".join(
                f"{label[:3]} {100 * by[label]['changed']:.1f}"
                if by[label]["changed"] is not None
                else f"{label[:3]} -"
                for label in labels
            )
            against = entry["against_word"].get(name)
            extra = (
                f" | vs word {100 * against['delta']:+.1f} [{100 * against['ci95'][0]:+.1f}, {100 * against['ci95'][1]:+.1f}], negative {100 * against['negative']['delta']:+.1f} [{100 * against['negative']['ci95'][0]:+.1f}, {100 * against['negative']['ci95'][1]:+.1f}]"
                if against
                else ""
            )
            print(
                f"   {name:<18} {100 * row['changed']:5.1f} [{100 * row['ci95'][0]:.1f}, {100 * row['ci95'][1]:.1f}] strings {100 * min(row['per_string']):.1f}-{100 * max(row['per_string']):.1f} | {cells}{extra}"
            )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", help="'linear', 'linear_weight_none' or the name of a run with a saved model"
    )
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="write results/appended_strings.json from the stored predictions",
    )
    parser.add_argument("--probe", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    parser.add_argument("--runs-dir", type=Path, default=C.ML_ROOT / "runs")
    args = parser.parse_args(argv)
    assert bool(args.model) != args.summarise, "give --model or --summarise"
    return summarise(args) if args.summarise else run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
