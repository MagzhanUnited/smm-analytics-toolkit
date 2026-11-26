"""Reproduce the evaluation of the toolkit: every number is computed from the repository alone.

    python experiments/run_experiments.py           # print the tables and rewrite results.json
    python experiments/run_experiments.py --check   # fail if results.json would change

Sections of the output:

* ``sample``      the labelled sample: accuracy and macro-F1 with bootstrap intervals,
                  ablation of the negation rules, per-language breakdown, the errors;
* ``behaviour``   pass rates of the behavioural suite per capability and classifier variant;
* ``privacy``     label invariance and leak check under injected personal identifiers;
* ``analytics``   recovery of the pattern planted in the synthetic posts;
* ``determinism`` identical reports under different hash seeds;
* ``timing``      throughput on the current machine (informative, not checked).

The sample and the suite are synthetic, so the numbers validate the pipeline and
describe the behaviour of the baseline; they are not estimates of accuracy on real comments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behaviour_suite import CAPABILITIES, build_cases  # noqa: E402

from smmkit import sentiment  # noqa: E402
from smmkit.anonymise import pseudonymise, scrub_text  # noqa: E402
from smmkit.io import load_comments, load_posts  # noqa: E402
from smmkit.metrics import engagement_rate, summarise  # noqa: E402
from smmkit.sentiment import LABELS, LexiconClassifier, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "data" / "sample" / "posts.csv"
COMMENTS = ROOT / "data" / "sample" / "comments_labelled.csv"
RESULTS = Path(__file__).resolve().parent / "results.json"

SEED = 20261117
BOOTSTRAP_RESAMPLES = 10_000
SALT = "experiments"
KAZAKH_LETTERS = set("әғқңөұүһі")  # letters of the Kazakh Cyrillic alphabet that Russian does not use

VARIANTS = {
    "full": {"pre": True, "post": True},
    "no_pre_negation": {"pre": False, "post": True},
    "no_post_negation": {"pre": True, "post": False},
    "no_negation": {"pre": False, "post": False},
}

# Fictional identifiers: (kind, text to inject, fragment that must not survive scrubbing)
PII_INJECTIONS = (
    ("handle", "@test_user01", "test_user01"),
    ("phone", "+7 700 123 45 67", "123 45 67"),
    ("phone", "8 (700) 123-45-67", "123-45-67"),
    ("phone", "+77001234567", "7001234567"),
    ("email", "test.user@example.com", "test.user"),
    ("url", "https://example.org/profile/123", "example.org"),
    ("url", "www.example.org/profile", "example.org"),
)
# Identifiers outside the four supported patterns: documented gaps, expected to survive.
PII_GAP_PROBES = (
    ("profile link without scheme", "t.me/test_user01", "test_user01"),
    ("profile link without scheme", "instagram.com/test_user01", "test_user01"),
    ("12-digit national identification number", "900101300123", "900101300123"),
    ("personal name", "Айгерим Нурланова", "Нурланова"),
)


@contextmanager
def classifier(pre: bool = True, post: bool = True):
    """The lexicon baseline with the pre- or post-negation rule switched off."""
    saved = sentiment.PRE_NEGATIONS, sentiment.POST_NEGATIONS
    if not pre:
        sentiment.PRE_NEGATIONS = ()
    if not post:
        sentiment.POST_NEGATIONS = ()
    try:
        yield LexiconClassifier()
    finally:
        sentiment.PRE_NEGATIONS, sentiment.POST_NEGATIONS = saved


def percentile(sorted_values: list, q: float) -> float:
    """Linear-interpolation percentile of an already sorted list, q in [0, 1]."""
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap(y_true: list, y_pred: list) -> dict:
    """Percentile bootstrap 95 % intervals for accuracy and macro-F1."""
    rng = random.Random(SEED)
    n = len(y_true)
    acc, f1 = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        ev = evaluate([y_true[i] for i in idx], [y_pred[i] for i in idx])
        acc.append(ev.accuracy)
        f1.append(ev.macro_f1)
    acc.sort()
    f1.sort()
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
        "accuracy_ci95": [round(percentile(acc, 0.025), 3), round(percentile(acc, 0.975), 3)],
        "macro_f1_ci95": [round(percentile(f1, 0.025), 3), round(percentile(f1, 0.975), 3)],
    }


def language(text: str) -> str:
    return "kk" if KAZAKH_LETTERS & set(text.lower()) else "ru"


def run_sample() -> dict:
    comments = [c for c in load_comments(COMMENTS, SALT) if c.label]
    y_true = [c.label for c in comments]
    out = {
        "n": len(comments),
        "class_counts": {label: y_true.count(label) for label in LABELS},
        "variants": {},
    }
    for name, flags in VARIANTS.items():
        with classifier(**flags) as clf:
            y_pred = clf.predict(c.text for c in comments)
        ev = evaluate(y_true, y_pred)
        entry = {
            "accuracy": round(ev.accuracy, 3),
            "macro_f1": round(ev.macro_f1, 3),
            "correct": sum(t == p for t, p in zip(y_true, y_pred, strict=True)),
        }
        if name == "full":
            entry["per_class"] = {
                label: {k: (round(v, 3) if k != "support" else v) for k, v in m.items()}
                for label, m in ev.per_class.items()
            }
            entry["confusion"] = ev.confusion
            entry["bootstrap"] = bootstrap(y_true, y_pred)
            entry["errors"] = [
                {"text": c.text, "label": c.label, "predicted": p, "score": clf.score(c.text)}
                for c, p in zip(comments, y_pred, strict=True)
                if c.label != p
            ]
            by_language = {}
            for lang in ("ru", "kk"):
                pairs = [
                    (t, p)
                    for c, t, p in zip(comments, y_true, y_pred, strict=True)
                    if language(c.text) == lang
                ]
                by_language[lang] = {"n": len(pairs), "correct": sum(t == p for t, p in pairs)}
            entry["by_language"] = by_language
        out["variants"][name] = entry
    majority = max(LABELS, key=y_true.count)
    ev = evaluate(y_true, [majority] * len(y_true))
    out["majority_class"] = {
        "label": majority,
        "accuracy": round(ev.accuracy, 3),
        "macro_f1": round(ev.macro_f1, 3),
    }
    return out


def run_behaviour() -> dict:
    cases = build_cases()
    out = {"n": len(cases), "capabilities": [], "totals": {}, "failures_full": {}}
    predictions = {}
    for name, flags in VARIANTS.items():
        with classifier(**flags) as clf:
            predictions[name] = clf.predict(c.text for c in cases)
    for cap in CAPABILITIES:
        idx = [i for i, c in enumerate(cases) if c.capability == cap.key]
        row = {
            "key": cap.key,
            "language": cap.language,
            "description": cap.description,
            "n": len(idx),
            "passed": {},
        }
        for name in VARIANTS:
            row["passed"][name] = sum(predictions[name][i] in cases[i].expected for i in idx)
        out["capabilities"].append(row)
        failed = [
            {"text": cases[i].text, "predicted": predictions["full"][i]}
            for i in idx
            if predictions["full"][i] not in cases[i].expected
        ]
        if failed:
            out["failures_full"][cap.key] = failed[:3]
    for name in VARIANTS:
        passed = sum(p in c.expected for p, c in zip(predictions[name], cases, strict=True))
        out["totals"][name] = {"passed": passed, "pass_rate": round(passed / len(cases), 3)}
    return out


def run_privacy() -> dict:
    with COMMENTS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    raw = [r["text"] for r in rows]
    clf = LexiconClassifier()
    perturbed = leaks = label_changes = 0
    for text in raw:
        base = clf.predict_one(scrub_text(text))
        for _, pii, fragment in PII_INJECTIONS:
            for candidate in (f"{pii} {text}", f"{text} {pii}"):
                clean = scrub_text(candidate)
                perturbed += 1
                leaks += fragment in clean
                label_changes += clf.predict_one(clean) != base
    in_sample = sum(scrub_text(t) != t for t in raw)
    authors = {r["author"] for r in rows}
    pseudonyms = {pseudonymise(a, SALT) for a in authors}
    loaded = load_comments(COMMENTS, SALT)
    gaps = [
        {"kind": kind, "example": pii, "removed": fragment not in scrub_text(f"Пишите {pii}")}
        for kind, pii, fragment in PII_GAP_PROBES
    ]
    return {
        "sample_comments_with_identifiers": in_sample,
        "sample_identifiers_left_after_loading": sum(
            any(f in c.text for f in ("@aliya_k", "example.com", "701 000", "example.kz")) for c in loaded
        ),
        "authors": len(authors),
        "pseudonyms": len(pseudonyms),
        "raw_author_ids_left_after_loading": sum(c.author in authors for c in loaded),
        "injection_formats": len(PII_INJECTIONS),
        "perturbed_comments": perturbed,
        "leaks": leaks,
        "label_changes": label_changes,
        "gap_probes": gaps,
    }


def run_analytics() -> dict:
    posts = load_posts(POSTS)
    by_format = {s.key: round(s.mean_er, 4) for s in summarise(posts, by="format")}
    planted = {
        "reels": 0.062,
        "carousel": 0.048,
        "photo": 0.031,
        "text": 0.022,
    }  # scripts/make_sample_data.py
    evening = [engagement_rate(p) for p in posts if p.published_at.hour >= 18]
    daytime = [engagement_rate(p) for p in posts if p.published_at.hour < 18]
    return {
        "posts": len(posts),
        "mean_er_by_format": by_format,
        "planted_er_by_format": planted,
        "format_ranking_recovered": sorted(by_format, key=by_format.get) == sorted(planted, key=planted.get),
        "evening_posts": len(evening),
        "daytime_posts": len(daytime),
        "evening_to_daytime_ratio": round(mean(evening) / mean(daytime), 2),
        "planted_evening_multiplier": 1.35,
    }


def run_determinism() -> dict:
    digests = []
    for seed in ("0", "1", "12345"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "smmkit.cli",
                    "report",
                    "--posts",
                    str(POSTS),
                    "--comments",
                    str(COMMENTS),
                    "--out",
                    tmp,
                ],
                check=True,
                capture_output=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            digests.append(hashlib.sha256((Path(tmp) / "report.md").read_bytes()).hexdigest())
    return {"runs": len(digests), "identical_reports": len(set(digests)) == 1, "report_sha256": digests[0]}


def run_timing() -> dict:
    with COMMENTS.open(encoding="utf-8", newline="") as fh:
        texts = [r["text"] for r in csv.DictReader(fh)] * 2000
    clf = LexiconClassifier()
    start = time.perf_counter()
    for text in texts:
        clf.predict_one(scrub_text(text))
    elapsed = time.perf_counter() - start
    return {
        "comments": len(texts),
        "seconds": round(elapsed, 2),
        "comments_per_second": int(round(len(texts) / elapsed, -3)),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
    }


def print_tables(results: dict) -> None:
    sample, behaviour = results["sample"], results["behaviour"]
    print(f"Labelled sample (n = {sample['n']})")
    for name, v in sample["variants"].items():
        print(
            f"  {name:<18} accuracy={v['accuracy']:.3f} macro_f1={v['macro_f1']:.3f} correct={v['correct']}"
        )
    full = sample["variants"]["full"]
    ci = full["bootstrap"]
    print(f"  95 % CI: accuracy {ci['accuracy_ci95']}, macro-F1 {ci['macro_f1_ci95']}")
    print(f"  majority class: {sample['majority_class']}")
    print(f"\nBehavioural suite (n = {behaviour['n']})")
    print(f"  {'capability':<8}{'lang':<7}{'n':>4}  " + "  ".join(f"{n:>16}" for n in VARIANTS))
    for row in behaviour["capabilities"]:
        cells = "  ".join(f"{row['passed'][n]:>16}" for n in VARIANTS)
        print(f"  {row['key']:<8}{row['language']:<7}{row['n']:>4}  {cells}")
    print(
        "  totals: "
        + ", ".join(f"{n} {t['passed']} ({t['pass_rate']:.1%})" for n, t in behaviour["totals"].items())
    )
    print(f"\nPrivacy: {results['privacy']}")
    print(f"Analytics: {results['analytics']}")
    print(f"Determinism: {results['determinism']}")
    print(f"Timing: {results['timing']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if results.json would change")
    args = parser.parse_args(argv)
    results = {
        "sample": run_sample(),
        "behaviour": run_behaviour(),
        "privacy": run_privacy(),
        "analytics": run_analytics(),
        "determinism": run_determinism(),
    }
    if args.check:
        stored = json.loads(RESULTS.read_text(encoding="utf-8"))
        stored.pop("timing", None)
        if stored != json.loads(json.dumps(results)):
            print("results.json is out of date: run experiments/run_experiments.py", file=sys.stderr)
            return 1
        print("results.json is up to date")
        return 0
    results["timing"] = run_timing()
    print_tables(results)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
