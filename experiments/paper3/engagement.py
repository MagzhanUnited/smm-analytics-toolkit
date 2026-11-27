"""What can be predicted about a brand post before it is published? Experiments of the third paper.

    python -I -B engagement.py            ->  results/engagement.json

Data: UCI "Facebook metrics" (Moro, Rita and Vala, 2016), 500 posts published in 2014 on the Facebook page of a
cosmetics brand, licence CC BY 4.0, in ~/Desktop/projects/aitu/smm-ml/data/uci_facebook_metrics/dataset_Facebook.csv.
Seven attributes are known before publication (page likes at publication, type, category, month, weekday, hour,
paid); twelve measure the impact of the post after publication.

Two sets of inputs: the content of the plan (type, category, weekday, hour, paid) and the plan with the date
(month and page likes, which grow together through the year). Baselines: the median of all earlier posts, the median
within the post type, and the median within the type over the last three months, which follows the level of the page.
Evaluations: forward in time (train on earlier months, predict the next one), leave one month out, and repeated random
cross-validation. Hyperparameters are always chosen inside the training data of a split. Intervals: bootstrap over
test posts and, because posts of one month share its level, over test months.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "Desktop/projects/aitu/smm-ml/data/uci_facebook_metrics/dataset_Facebook.csv"
SEED = 20261127
RESAMPLES = 2000

CONTENT = ["Type", "Category", "Post Weekday", "Post Hour", "Paid"]
DATE = ["Post Month", "Page total likes"]
FEATURE_SETS = {"content": CONTENT, "content_date": CONTENT + DATE}
# metrics of reach and impressions, known only after publication (engaged users and consumers are left out: they
# contain the interactions almost by definition)
LEAKED = [
    "Lifetime Post Total Reach",
    "Lifetime Post Total Impressions",
    "Lifetime Post Impressions by people who have liked your Page",
    "Lifetime Post reach by people who like your Page",
]
TARGETS = {
    "interactions": "Total Interactions",
    "engaged": "Lifetime Engaged Users",
    "reach": "Lifetime Post Total Reach",
}
CATEGORICAL = ["Type", "Category", "Post Weekday"]
MODELS = ["median", "type_median", "recent_type_median", "ridge", "forest", "boosting"]
TRAINED = ["ridge", "forest", "boosting"]
BASE = "recent_type_median"  # the strongest baseline; differences of the trained models are taken against it


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, sep=";")
    assert df.shape == (500, 19), df.shape
    assert (df["Total Interactions"] == df[["like", "comment", "share"]].sum(axis=1)).all()
    df = df[df["Paid"].notna()].copy()  # one post has no value for Paid
    df["Paid"] = df["Paid"].astype(int)
    # The file lists the posts from the newest to the oldest, and the page likes grow through the year:
    # sorting by page likes, ties broken by the position in the file, gives the order of publication.
    df["file_row"] = np.arange(len(df))
    df = df.sort_values(["Page total likes", "file_row"], ascending=[True, False]).reset_index(drop=True)
    assert df["Post Month"].is_monotonic_increasing
    return df


def preprocess(columns: list, scale: bool) -> ColumnTransformer:
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c not in CATEGORICAL]
    parts = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]
    if num:
        parts.append(("num", StandardScaler() if scale else "passthrough", num))
    return ColumnTransformer(parts)  # columns not listed are dropped


class Median:
    """Median of the training target: overall, within the post type, or within the type over the last ``months``
    months of the training data (falling back to all months of the type, then to all training posts)."""

    def __init__(self, by_type: bool, months: int | None = None):
        self.by_type, self.months = by_type, months

    def fit(self, x, y):
        y = pd.Series(np.asarray(y, dtype=float), index=x.index)
        self.overall = float(y.median())
        self.by = y.groupby(x["Type"]).median().to_dict() if self.by_type else {}
        if self.months:
            recent = x["Post Month"] > x["Post Month"].max() - self.months
            self.recent = y[recent].groupby(x.loc[recent, "Type"]).median().to_dict()
            self.recent_overall = float(y[recent].median())
        return self

    def predict(self, x):
        if not self.by_type:
            return np.full(len(x), self.overall)
        if self.months:
            return np.array(
                [self.recent.get(t, self.by.get(t, self.recent_overall)) for t in x["Type"]], dtype=float
            )
        return x["Type"].map(self.by).fillna(self.overall).to_numpy(dtype=float)


def make_model(name: str, columns: list, seed: int):
    """A fresh model; hyperparameters are chosen by a 5-fold cross-validation inside the training data."""
    inner = KFold(5, shuffle=True, random_state=seed)
    if name == "median":
        return Median(False)
    if name == "type_median":
        return Median(True)
    if name == "recent_type_median":
        return Median(True, months=3)
    if name == "ridge":
        return Pipeline(
            [
                ("prep", preprocess(columns, scale=True)),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 21), cv=inner)),
            ]
        )
    if name == "forest":
        model = Pipeline(
            [
                ("prep", preprocess(columns, scale=False)),
                ("model", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)),
            ]
        )
        grid = {"model__min_samples_leaf": [1, 5, 10, 20], "model__max_features": [0.33, 0.66, 1.0]}
        return GridSearchCV(model, grid, cv=inner, scoring="neg_mean_absolute_error", n_jobs=-1)
    if name == "boosting":
        model = Pipeline(
            [
                ("prep", preprocess(columns, scale=False)),
                ("model", GradientBoostingRegressor(random_state=seed, subsample=0.8)),
            ]
        )
        grid = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.02, 0.05],
            "model__max_depth": [2, 3],
        }
        return GridSearchCV(model, grid, cv=inner, scoring="neg_mean_absolute_error", n_jobs=-1)
    raise ValueError(name)


def auc(y_top, pred) -> float:
    return float(roc_auc_score(y_top, pred)) if np.ptp(pred) > 0 and 0 < y_top.sum() < len(y_top) else 0.5


def scores(y, pred, months) -> dict:
    """MAE and R² on the log scale, Spearman's rho, and the AUC for the top quarter of posts: pooled over the test posts,
    and within each month (top quarter of that month), weighted by the posts of the month."""
    y, pred, months = np.asarray(y, dtype=float), np.asarray(pred, dtype=float), np.asarray(months)
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    within = []
    for m in np.unique(months):
        sel = months == m
        within.append((sel.sum(), auc(y[sel] >= np.quantile(y[sel], 0.75), pred[sel])))
    return {
        "mae": float(np.abs(y - pred).mean()),
        "r2": 1 - ss_res / ss_tot,
        "spearman": float(spearmanr(y, pred).statistic) if np.ptp(pred) > 0 else 0.0,
        "auc_top25": auc(y >= np.quantile(y, 0.75), pred),
        "auc_within_month": float(sum(n * a for n, a in within) / sum(n for n, _ in within)),
    }


def intervals(y, preds: dict, months, rng) -> dict:
    """Bootstrap intervals for every model and for its difference to the strongest baseline: resampling posts, and
    resampling whole test months (posts of one month share its level)."""
    y, months = np.asarray(y, dtype=float), np.asarray(months)
    unique = np.unique(months)
    members = {m: np.flatnonzero(months == m) for m in unique}
    out = {name: {} for name in preds}
    for kind in ("posts", "months"):
        draws = {name: [] for name in preds}
        for _ in range(RESAMPLES):
            if kind == "posts":
                idx = rng.integers(0, len(y), len(y))
            else:
                idx = np.concatenate([members[m] for m in rng.choice(unique, len(unique))])
            for name, p in preds.items():
                draws[name].append(scores(y[idx], np.asarray(p)[idx], months[idx]))
        for name in preds:
            row = {}
            for metric in ("mae", "r2", "spearman", "auc_within_month"):
                values = np.array([d[metric] for d in draws[name]])
                diff = values - np.array([d[metric] for d in draws[BASE]])
                row[metric] = {
                    "ci95": [float(np.percentile(values, q)) for q in (2.5, 97.5)],
                    "diff_to_base_ci95": [float(np.percentile(diff, q)) for q in (2.5, 97.5)],
                }
            out[name][kind] = row
    return out


def forward_chaining(df, target, columns, models, first_test_month=7, seed=SEED) -> dict:
    """Train on all posts of the months before m, predict the posts of month m, for m = first_test_month..12."""
    y = np.log1p(df[target].to_numpy(dtype=float))
    test = (df["Post Month"] >= first_test_month).to_numpy()
    preds = {name: np.full(len(df), np.nan) for name in models}
    for month in range(first_test_month, 13):
        train, now = (df["Post Month"] < month).to_numpy(), (df["Post Month"] == month).to_numpy()
        for name in models:
            model = make_model(name, columns, seed).fit(df[train], y[train])
            preds[name][now] = model.predict(df[now])
    return {
        "y": y[test],
        "months": df["Post Month"].to_numpy()[test],
        "preds": {n: p[test] for n, p in preds.items()},
    }


def month_out(df, target, columns, models, seed=SEED) -> dict:
    """Leave one month out: train on the other eleven months, earlier and later, predict the month left out; scored on
    July to December, the months of the forward evaluation."""
    y = np.log1p(df[target].to_numpy(dtype=float))
    preds = {name: np.zeros(len(df)) for name in models}
    for month in range(1, 13):
        now = (df["Post Month"] == month).to_numpy()
        for name in models:
            preds[name][now] = make_model(name, columns, seed).fit(df[~now], y[~now]).predict(df[now])
    test = (df["Post Month"] >= 7).to_numpy()
    return {name: scores(y[test], p[test], df["Post Month"].to_numpy()[test]) for name, p in preds.items()}


def random_cv(df, target, columns, models, seed=SEED, repeats=3) -> dict:
    """Repeated 10-fold cross-validation with shuffled folds; scores on July to December within each repetition
    (like for like with the forward evaluation) and on all posts, averaged over the repetitions."""
    y = np.log1p(df[target].to_numpy(dtype=float))
    months = df["Post Month"].to_numpy()
    test = months >= 7
    rows = {name: {"all": [], "july_to_december": []} for name in models}
    preds = {name: np.zeros(len(df)) for name in models}
    for fold, (train, held) in enumerate(
        RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=seed).split(df)
    ):
        for name in models:
            preds[name][held] = (
                make_model(name, columns, seed + fold).fit(df.iloc[train], y[train]).predict(df.iloc[held])
            )
        if (fold + 1) % 10 == 0:
            for name in models:
                rows[name]["all"].append(scores(y, preds[name], months))
                rows[name]["july_to_december"].append(scores(y[test], preds[name][test], months[test]))
    return {
        name: {
            part: {metric: float(np.mean([r[metric] for r in values])) for metric in values[0]}
            for part, values in parts.items()
        }
        for name, parts in rows.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "results" / "engagement.json")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    df = load()
    monthly = df.groupby("Post Month")[[TARGETS[k] for k in TARGETS]].median()
    out = {
        "data": {
            "posts": int(len(df)),
            "posts_in_file": 500,
            "dropped_missing_paid": 500 - int(len(df)),
            "types": df["Type"].value_counts().to_dict(),
            "paid_share": float(df["Paid"].mean()),
            "posts_by_month": df["Post Month"].value_counts().sort_index().to_dict(),
            "page_likes": [int(df["Page total likes"].min()), int(df["Page total likes"].max())],
            "spearman_page_likes_month": float(spearmanr(df["Page total likes"], df["Post Month"]).statistic),
            "monthly_median": {
                key: {int(m): float(v) for m, v in monthly[col].items()} for key, col in TARGETS.items()
            },
            "targets": {
                key: {
                    "median": float(df[col].median()),
                    "mean": float(df[col].mean()),
                    "max": float(df[col].max()),
                }
                for key, col in TARGETS.items()
            },
            "engaged_at_least_interactions": float(
                (df["Lifetime Engaged Users"] >= df["Total Interactions"]).mean()
            ),
        },
        "feature_sets": FEATURE_SETS,
        "leaked": LEAKED,
        "models": MODELS,
        "base": BASE,
        "seed": SEED,
        "resamples": RESAMPLES,
    }
    out["forward"], out["random_cv"], out["month_out"] = {}, {}, {}
    for key, target in TARGETS.items():
        out["forward"][key], out["random_cv"][key], out["month_out"][key] = {}, {}, {}
        for fs, columns in FEATURE_SETS.items():
            fc = forward_chaining(df, target, columns, MODELS)
            out["forward"][key][fs] = {
                "n_test": int(len(fc["y"])),
                "scores": {n: scores(fc["y"], p, fc["months"]) for n, p in fc["preds"].items()},
                "intervals": intervals(fc["y"], fc["preds"], fc["months"], rng),
            }
            out["month_out"][key][fs] = month_out(df, target, columns, MODELS)
            out["random_cv"][key][fs] = random_cv(
                df, target, columns, [m for m in MODELS if m != "recent_type_median"]
            )
            print(
                f"{key:<12} {fs:<12} forward: "
                + " | ".join(
                    f"{n} {s['mae']:.3f}/{s['r2']:+.2f}/{s['auc_within_month']:.2f}"
                    for n, s in out["forward"][key][fs]["scores"].items()
                ),
                flush=True,
            )

    leak_columns = CONTENT + DATE + LEAKED
    fc = forward_chaining(df, TARGETS["interactions"], leak_columns, TRAINED)
    out["leak"] = {
        "forward": {n: scores(fc["y"], p, fc["months"]) for n, p in fc["preds"].items()},
        "random_cv": random_cv(df, TARGETS["interactions"], leak_columns, TRAINED),
    }

    groups = {
        "Type": ["Type"],
        "Category": ["Category"],
        "Weekday": ["Post Weekday"],
        "Hour": ["Post Hour"],
        "Paid": ["Paid"],
        "Date": DATE,
    }
    out["drop_group"] = {}
    for key in ("interactions", "engaged"):
        out["drop_group"][key] = {}
        for model_name in ("ridge", "boosting"):
            full = CONTENT + DATE
            base = forward_chaining(df, TARGETS[key], full, [model_name])
            base_mae = scores(base["y"], base["preds"][model_name], base["months"])["mae"]
            row = {"all": base_mae}
            for group, cols in groups.items():
                reduced = forward_chaining(df, TARGETS[key], [c for c in full if c not in cols], [model_name])
                row[group] = (
                    scores(reduced["y"], reduced["preds"][model_name], reduced["months"])["mae"] - base_mae
                )
            out["drop_group"][key][model_name] = row

    out["paid"] = {}
    for key, target in TARGETS.items():
        y = np.log1p(df[target].to_numpy(dtype=float))

        def coefficient(rows, y=y):
            sub = df.iloc[rows]
            x = pd.get_dummies(
                sub[["Type", "Category", "Post Weekday"]].astype(str), drop_first=True, dtype=float
            )
            x["month"], x["hour"], x["paid"] = (
                sub["Post Month"].to_numpy(),
                sub["Post Hour"].to_numpy(),
                sub["Paid"].to_numpy(),
            )
            x.insert(0, "const", 1.0)
            beta, *_ = np.linalg.lstsq(x.to_numpy(dtype=float), y[rows], rcond=None)
            return float(beta[list(x.columns).index("paid")])

        draws = [coefficient(rng.integers(0, len(df), len(df))) for _ in range(RESAMPLES)]
        out["paid"][key] = {
            "median_paid": float(df.loc[df["Paid"] == 1, target].median()),
            "median_unpaid": float(df.loc[df["Paid"] == 0, target].median()),
            "adjusted_ratio": float(np.exp(coefficient(np.arange(len(df))))),
            "adjusted_ratio_ci95": [float(np.exp(np.percentile(draws, q))) for q in (2.5, 97.5)],
        }

    out["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    out["seconds"] = round(time.perf_counter() - start, 1)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("leak forward:", {n: round(s["r2"], 3) for n, s in out["leak"]["forward"].items()})
    for key, row in out["drop_group"].items():
        print("drop", key, {m: {g: round(v, 3) for g, v in r.items()} for m, r in row.items()})
    print(
        "paid:",
        {
            k: (round(v["adjusted_ratio"], 2), [round(x, 2) for x in v["adjusted_ratio_ci95"]])
            for k, v in out["paid"].items()
        },
    )
    print("written", args.out, "in", out["seconds"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
