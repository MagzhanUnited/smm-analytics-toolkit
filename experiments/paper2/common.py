"""Shared code for the experiments of the second paper. Only open data are used.

Heavy artefacts (the Python environment, the corpus, model weights, run outputs) live outside the
repository, by default in ``~/Desktop/projects/aitu/smm-ml``. Every script takes their paths as
arguments and is run in isolated mode, because the corpus is downloaded data::

    ~/Desktop/projects/aitu/smm-ml/venv/bin/python -I experiments/paper2/baselines.py --help

Tasks on KazSAnDRA (Yeshpanov and Varol, 2024), labels are the star ratings given by the reviewers:

* ``pc``  the official polarity task: 1-2 stars negative, 4-5 stars positive, official partition;
* ``p3``  three classes derived from the official score-classification partition:
          1-2 stars negative, 3 stars neutral, 4-5 stars positive.

The two official partitions were drawn independently: most ``pc`` test reviews are in the ``sc``
training set. A model trained for one task is therefore never evaluated on the test set of the other.
"""

from __future__ import annotations

import contextlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for extra in (HERE.parent, REPO / "src"):  # experiments/ (behavioural suite) and the toolkit package
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from behaviour_suite import CAPABILITIES, build_cases  # noqa: E402
from run_experiments import PII_INJECTIONS  # noqa: E402

from smmkit.anonymise import scrub_text  # noqa: E402

ML_ROOT = Path.home() / "Desktop/projects/aitu/smm-ml"
RESULTS = HERE / "results"
SEED = 20261126
FILES = {
    "pc": ("01_pc_train_ib", "04_pc_valid", "05_pc_test"),
    "sc": ("06_sc_train_ib", "09_sc_valid", "10_sc_test"),
}
LABELS = {"pc": ("negative", "positive"), "p3": ("negative", "neutral", "positive")}
SC_TO_P3 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}  # stars 1-2, 3, 4-5

_KAZAKH = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)


def load_task(data_dir: Path, task: str, text_col: str = "text") -> dict:
    """Return ``{"train": df, "valid": df, "test": df}`` with columns ``x`` (text) and ``y`` (class index)."""
    source = "pc" if task == "pc" else "sc"
    frames = {}
    for split, name in zip(("train", "valid", "test"), FILES[source], strict=True):
        df = pd.read_csv(Path(data_dir) / name / f"{name}.csv")
        df["x"] = df[text_col].astype(str)
        df["y"] = (df["label"] if task == "pc" else df["label"].map(SC_TO_P3)).astype(int)
        frames[split] = df
    return frames


def script_type(text: str) -> str:
    """How a review is written: the corpus mixes scripts and languages."""
    if _KAZAKH.search(text):
        return "kazakh_letters"  # contains letters that only the Kazakh alphabet has
    if _CYRILLIC.search(text):
        return "cyrillic_without_kazakh_letters"  # Russian, or Kazakh typed without its letters
    if _LATIN.search(text):
        return "latin"
    return "no_letters"


# ------------------------------------------------------------------ metrics
def confusion(y_true, y_pred, k: int) -> np.ndarray:
    return np.bincount(np.asarray(y_true) * k + np.asarray(y_pred), minlength=k * k).reshape(k, k)


def scores(cm: np.ndarray, present_only: bool = False) -> dict:
    """Accuracy and macro-averaged precision, recall and F1 from a confusion matrix (rows: true)."""
    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(0).astype(float), cm.sum(1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    keep = actual > 0 if present_only else np.ones_like(tp, dtype=bool)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision[keep].mean()),
        "macro_recall": float(recall[keep].mean()),
        "macro_f1": float(f1[keep].mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summary(y_true, y_pred, labels: tuple, present_only: bool = False) -> dict:
    k = len(labels)
    cm = confusion(y_true, y_pred, k)
    s = scores(cm, present_only)
    out = {key: round(s[key], 4) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    out["n"] = int(cm.sum())
    out["per_class"] = {
        label: {
            "precision": round(float(s["precision"][i]), 4),
            "recall": round(float(s["recall"][i]), 4),
            "f1": round(float(s["f1"][i]), 4),
            "support": int(cm[i].sum()),
        }
        for i, label in enumerate(labels)
    }
    out["confusion"] = cm.tolist()
    return out


def bootstrap(y_true, y_pred, k: int, resamples: int = 2000, seed: int = SEED) -> dict:
    """Percentile bootstrap 95 % intervals over test reviews."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    acc, f1 = np.empty(resamples), np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        s = scores(confusion(y_true[idx], y_pred[idx], k))
        acc[b], f1[b] = s["accuracy"], s["macro_f1"]
    return {
        "resamples": resamples,
        "seed": seed,
        "accuracy_ci95": [round(float(np.percentile(acc, q)), 4) for q in (2.5, 97.5)],
        "macro_f1_ci95": [round(float(np.percentile(f1, q)), 4) for q in (2.5, 97.5)],
    }


def paired_bootstrap(y_true, pred_a, pred_b, k: int, resamples: int = 2000, seed: int = SEED) -> dict:
    """Difference in macro-F1 (b minus a) on the same resampled test reviews."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    delta = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        delta[b] = (
            scores(confusion(y_true[idx], pred_b[idx], k))["macro_f1"]
            - scores(confusion(y_true[idx], pred_a[idx], k))["macro_f1"]
        )
    observed = (
        scores(confusion(y_true, pred_b, k))["macro_f1"] - scores(confusion(y_true, pred_a, k))["macro_f1"]
    )
    return {
        "delta_macro_f1": round(float(observed), 4),
        "ci95": [round(float(np.percentile(delta, q)), 4) for q in (2.5, 97.5)],
        "share_of_resamples_with_b_better": round(float((delta > 0).mean()), 4),
        "resamples": resamples,
    }


# ------------------------------------------------------------------ evaluation battery
def subgroups(test: pd.DataFrame, pred, labels: tuple, train_cleaned: set) -> dict:
    """Test scores by source domain, by script, and for reviews whose cleaned text also occurs in training."""
    y, pred = test["y"].to_numpy(), np.asarray(pred)
    columns = {
        "domain": test["domain"].astype(str),
        "script": test["text"].astype(str).map(script_type),
        "overlap": test["text_cleaned"]
        .astype(str)
        .isin(train_cleaned)
        .map({True: "text_also_in_train", False: "unseen_text"}),
    }
    out = {}
    for name, column in columns.items():
        out[name] = {}
        for group in sorted(column.unique()):
            mask = (column == group).to_numpy()
            s = summary(y[mask], pred[mask], labels, present_only=True)
            out[name][group] = {key: s[key] for key in ("n", "accuracy", "macro_f1")}
    return out


def run_suite(predict_labels) -> dict:
    """Behavioural suite of the toolkit. ``predict_labels`` maps a list of texts to label names."""
    cases = build_cases()
    predictions = list(predict_labels([c.text for c in cases]))
    rows = []
    for cap in CAPABILITIES:
        idx = [i for i, c in enumerate(cases) if c.capability == cap.key]
        rows.append(
            {
                "key": cap.key,
                "language": cap.language,
                "n": len(idx),
                "passed": sum(predictions[i] in cases[i].expected for i in idx),
            }
        )
    passed = sum(r["passed"] for r in rows)
    return {
        "n": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 4),
        "capabilities": rows,
        "predictions": predictions,
    }


_PLACEHOLDER = re.compile(r"\s*<(?:URL|EMAIL|PHONE|USER)>\s*")


def drop_placeholders(text: str) -> str:
    """Remove the placeholders that :func:`smmkit.anonymise.scrub_text` leaves in a text."""
    return _PLACEHOLDER.sub(" ", text).strip()


def privacy_probe(predict, texts: list) -> dict:
    """Label invariance under inserted fictional identifiers for three ways of handling them.

    ``predict`` maps a list of texts to an array of class indices. The seven identifier formats are
    those of the toolkit's own privacy test; each is inserted before and after every text. The label
    of the perturbed text is compared with the label of the same text without the identifier when

    * ``left_in_text``  the identifier stays in the classifier input (no privacy boundary);
    * ``placeholder``   it is replaced by ``<URL>``, ``<EMAIL>``, ``<PHONE>`` or ``<USER>``;
    * ``removed``       it is replaced and the placeholder is then dropped from the classifier input.
    """
    texts = [t.strip() for t in texts]
    treatments = {
        "left_in_text": lambda t: t,
        "placeholder": scrub_text,
        "removed": lambda t: drop_placeholders(scrub_text(t)),
    }
    candidates, owners, fragments, kinds = [], [], [], []
    for i, text in enumerate(texts):
        for kind, pii, fragment in PII_INJECTIONS:
            for candidate in (f"{pii} {text}", f"{text} {pii}"):
                candidates.append(candidate)
                owners.append(i)
                fragments.append(fragment)
                kinds.append(kind)
    owners, kinds = np.asarray(owners), np.asarray(kinds)
    base_raw = np.asarray(predict(texts))
    out = {
        "texts": len(texts),
        "perturbed": len(candidates),
        "identifier_formats": len(PII_INJECTIONS),
        "leaks_after_scrubbing": int(
            sum(f in scrub_text(c) for f, c in zip(fragments, candidates, strict=True))
        ),
        "label_changes": {},
        "label_changes_by_kind": {},
        "original_texts_relabelled": {},
    }
    for name, treat in treatments.items():
        base = np.asarray(predict([treat(t) for t in texts]))
        changed = np.asarray(predict([treat(c) for c in candidates])) != base[owners]
        out["label_changes"][name] = int(changed.sum())
        out["label_changes_by_kind"][name] = {
            kind: int(changed[kinds == kind].sum()) for kind in sorted(set(kinds))
        }
        out["original_texts_relabelled"][name] = int((base != base_raw).sum())
    out["perturbed_by_kind"] = {kind: int((kinds == kind).sum()) for kind in sorted(set(kinds))}
    return out


def valid_for_size(
    valid: pd.DataFrame, train_size: int, limit: int = 64000, subset: int = 2000
) -> pd.DataFrame:
    """Validation data for a learning-curve point: a 2,000-review subset below 64,000 training reviews.

    A user with a few hundred labelled comments does not have 17,000 more for model selection.
    """
    return valid if train_size >= limit else stratified_sample(valid, subset, SEED)


def throughput(predict, texts: list, repeats: int = 3) -> dict:
    """Median wall-clock time of ``predict`` over ``texts``; the machine is a laptop in normal use."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(texts)
        times.append(time.perf_counter() - start)
    median = sorted(times)[len(times) // 2]
    return {
        "texts": len(texts),
        "seconds": round(median, 3),
        "texts_per_second": round(len(texts) / median, 1),
    }


def stratified_sample(frame: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """About ``n`` rows with the class proportions of ``frame``."""
    if n >= len(frame):
        return frame
    share = n / len(frame)
    parts = [
        group.sample(max(1, round(len(group) * share)), random_state=seed) for _, group in frame.groupby("y")
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def environment() -> dict:
    info = {"python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}"}
    for module in ("numpy", "pandas", "sklearn", "torch", "transformers"):
        with contextlib.suppress(ImportError):
            info[module] = __import__(module).__version__
    return info


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
