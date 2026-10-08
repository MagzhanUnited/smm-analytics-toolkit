"""Optional plots (``pip install smm-analytics-toolkit[plot]``)."""

from __future__ import annotations

from pathlib import Path

from smmkit.metrics import WEEKDAYS, engagement_rate


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("plots need matplotlib: pip install 'smm-analytics-toolkit[plot]'") from exc
    return plt


def engagement_heatmap(posts, path) -> Path:
    """Mean ER by weekday (rows) and three-hour block (columns)."""
    plt = _plt()
    blocks = list(range(0, 24, 3))
    sums = [[0.0] * len(blocks) for _ in WEEKDAYS]
    counts = [[0] * len(blocks) for _ in WEEKDAYS]
    for p in posts:
        r, c = p.published_at.weekday(), p.published_at.hour // 3
        sums[r][c] += engagement_rate(p)
        counts[r][c] += 1
    grid = [[(s / n * 100 if n else float("nan")) for s, n in zip(sr, cr)] for sr, cr in zip(sums, counts)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(blocks)), [f"{b:02d}-{b + 3:02d}" for b in blocks])
    ax.set_yticks(range(len(WEEKDAYS)), WEEKDAYS)
    top = max((v for row in grid for v in row if v == v), default=0.0)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == v:  # not NaN
                colour = "white" if v > 0.6 * top else "black"  # keep labels readable on dark cells
                ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8, color=colour)
    fig.colorbar(im, ax=ax, label="mean ER, %")
    ax.set_title("Engagement rate by weekday and time of day")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def confusion_matrix(evaluation, path) -> Path:
    plt = _plt()
    labels = list(evaluation.confusion)
    grid = [[evaluation.confusion[t][p] for p in labels] for t in labels]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    top = max(max(row) for row in grid) or 1
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            ax.text(c, r, str(v), ha="center", va="center", color="white" if v > 0.6 * top else "black")
    ax.set_title(f"Confusion matrix (macro-F1 {evaluation.macro_f1:.2f})")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
