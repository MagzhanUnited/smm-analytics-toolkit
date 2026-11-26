"""Training time and throughput of the lexicon and of the linear model, measured again on an idle machine.

    python -I baseline_timing.py [--tasks p3 pc] [--out results/baseline_timing.json]

``baselines.py`` took its timings while a transformer was training on the graphics processor of the
same laptop. This script repeats only the timed steps: the search for C of the linear model, with and
without class weights (the bracketed procedure of ``linear_curve_weighting.py``, which on the full
training set must select the configuration that ``baselines.py`` reports), and the throughput of both
baselines on the test set. It records whether a fine-tuning process was running. Numerical libraries
are limited to one thread, so the times are those of one thread of the processor, as the paper says.
Results go to ``results/baseline_timing.json``; the test set is used for throughput only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"  # before the numerical libraries are loaded
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import numpy as np  # noqa: E402
from baselines import NAME_TO_P3  # noqa: E402
from linear_curve_weighting import fit_bracketed  # noqa: E402

from smmkit.sentiment import LexiconClassifier  # noqa: E402


def training_is_running() -> bool:
    return subprocess.run(["pgrep", "-f", "finetune.py --name"], capture_output=True).returncode == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="*", default=["p3", "pc"])
    parser.add_argument("--out", type=Path, default=C.RESULTS / "baseline_timing.json")
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_info, threadpool_limits

    threadpool_limits(limits=1)
    out = {
        "environment": C.environment(),
        "training_running_at_start": training_is_running(),
        "tasks": {},
        "threads": {
            "limit": 1,
            "pools": [
                {key: pool.get(key) for key in ("user_api", "internal_api", "num_threads")}
                for pool in threadpool_info()
            ],
        },
    }
    for task in args.tasks:
        frames = C.load_task(C.ML_ROOT / "data" / "kazsandra", task, "text")
        train, valid, test = frames["train"], frames["valid"], frames["test"]
        k, test_texts = len(C.LABELS[task]), test["x"].tolist()
        stored = json.loads((C.RESULTS / f"baselines_{task}_text.json").read_text(encoding="utf-8"))[
            "models"
        ]["linear"]["fit"]

        fit = fit_bracketed(train, valid, k)
        union, chosen = fit["union"], fit["by_weighting"][fit["selected"]]
        clf = chosen["clf"]
        selected = {
            "C": chosen["C"],
            "class_weight": fit["selected"],
            "valid_macro_f1": round(chosen["valid_macro_f1"], 4),
            "fit_seconds": chosen["fit_seconds"],
            "iterations": chosen["iterations"],
        }
        assert all(
            selected[key] == stored["selected"][key] for key in ("C", "class_weight", "valid_macro_f1")
        ), (task, selected, stored["selected"])

        lexicon = LexiconClassifier()
        majority = int(np.bincount(train["y"], minlength=k).argmax())

        def predict_lexicon(texts):
            names = lexicon.predict(texts)
            if task == "p3":
                return np.array([NAME_TO_P3[name] for name in names])
            return np.array(
                [0 if name == "negative" else 1 if name == "positive" else majority for name in names]
            )

        out["tasks"][task] = {
            "selected": selected,
            "configurations": fit["configurations"],
            "tried": {weighting: result["tried"] for weighting, result in fit["by_weighting"].items()},
            "grid_seconds": fit["grid_seconds"],
            "train_seconds": round(fit["vectoriser_seconds"] + selected["fit_seconds"], 2),
            "vectoriser_seconds": fit["vectoriser_seconds"],
            "throughput_linear": C.throughput(lambda texts: clf.predict(union.transform(texts)), test_texts),
            "throughput_lexicon": C.throughput(predict_lexicon, test_texts),
            "stored": {"grid_seconds": stored["grid_seconds"], "train_seconds": stored["train_seconds"]},
        }
        print(task, json.dumps(out["tasks"][task], ensure_ascii=False), flush=True)
    out["training_running_at_end"] = training_is_running()
    out["measured"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.write_json(args.out, out)
    print(
        "written",
        args.out,
        "| training running:",
        out["training_running_at_start"],
        out["training_running_at_end"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
