"""Statistics of KazSAnDRA that the paper reports. No review text is written to the output.

python -I data_profile.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smmkit.anonymise import _PATTERNS  # noqa: E402

EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿❤️]")


def distribution(series: pd.Series) -> dict:
    counts = series.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=C.ML_ROOT / "data" / "kazsandra")
    args = parser.parse_args(argv)

    out = {"tasks": {}}
    frames = {}
    for task in ("pc", "p3"):
        frames[task] = C.load_task(args.data_dir, task)
        out["tasks"][task] = {
            "labels": list(C.LABELS[task]),
            "sizes": {split: len(df) for split, df in frames[task].items()},
            "class_counts": {
                split: np.bincount(df["y"], minlength=len(C.LABELS[task])).tolist()
                for split, df in frames[task].items()
            },
        }
    sc = frames["p3"]
    everything = pd.concat(sc.values(), ignore_index=True)
    text = everything["text"].astype(str)
    out["reviews"] = len(everything)
    out["stars"] = {str(int(k) + 1): v for k, v in distribution(everything["label"]).items()}
    out["domains"] = distribution(everything["domain"])
    out["stars_by_domain"] = {
        domain: {str(int(k) + 1): v for k, v in distribution(group["label"]).items()}
        for domain, group in everything.groupby("domain")
    }
    chars, words = text.str.len(), text.str.split().str.len()
    out["length"] = {
        "characters": {
            "mean": round(float(chars.mean()), 1),
            "median": float(chars.median()),
            "p95": float(chars.quantile(0.95)),
            "max": int(chars.max()),
        },
        "words": {
            "mean": round(float(words.mean()), 1),
            "median": float(words.median()),
            "p95": float(words.quantile(0.95)),
            "max": int(words.max()),
        },
    }
    out["script"] = {
        key: round(value, 4) for key, value in text.map(C.script_type).value_counts(normalize=True).items()
    }
    out["share_with_emoji"] = round(float(text.str.contains(EMOJI).mean()), 4)

    # the two official partitions are independent
    out["partition_overlap"] = {
        "pc_test_reviews_in_sc_train": int(
            frames["pc"]["test"]["custom_id"].isin(set(sc["train"]["custom_id"])).sum()
        ),
        "pc_test": len(frames["pc"]["test"]),
        "sc_test_reviews_in_pc_train": int(
            sc["test"]["custom_id"].isin(set(frames["pc"]["train"]["custom_id"])).sum()
        ),
        "sc_test": len(sc["test"]),
    }

    # repeated texts: the corpus keeps one review per (text, score) pair
    cleaned = everything["text_cleaned"].astype(str)
    group_size = cleaned.map(cleaned.value_counts())
    repeated = everything[group_size > 1]
    train_cleaned = set(sc["train"]["text_cleaned"].astype(str))
    test_in_train = sc["test"][sc["test"]["text_cleaned"].astype(str).isin(train_cleaned)]
    out["repeated_texts"] = {
        "reviews_with_a_text_that_occurs_more_than_once": len(repeated),
        "distinct_repeated_texts": int(cleaned[group_size > 1].nunique()),
        "stars_of_reviews_with_repeated_text": {
            str(int(k) + 1): v for k, v in distribution(repeated["label"]).items()
        },
        "duplicate_text_and_score_pairs": int(everything.duplicated(["text_cleaned", "label"]).sum()),
        "test_reviews_whose_text_is_in_train": len(test_in_train),
        "stars_of_those_test_reviews": {
            str(int(k) + 1): v for k, v in distribution(test_in_train["label"]).items()
        },
        "median_words_of_repeated_texts": float(
            repeated["text_cleaned"].astype(str).str.split().str.len().median()
        ),
    }

    # direct identifiers that the toolkit recognises, in real reviews
    hits = {name.strip("<>").lower(): text.str.contains(pattern) for name, pattern in _PATTERNS}
    any_hit = np.logical_or.reduce([h.to_numpy() for h in hits.values()])
    out["identifiers_in_reviews"] = {name: int(h.sum()) for name, h in hits.items()}
    out["identifiers_in_reviews"]["reviews_with_any"] = int(any_hit.sum())
    C.write_json(C.RESULTS / "data_profile.json", out)
    for key, value in out.items():
        print(key, "->", value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
