"""Markdown report combining engagement metrics and classifier evaluation."""

from __future__ import annotations

from smmkit.metrics import best_slots, summarise


def _pct(x: float) -> str:
    return f"{x * 100:.2f} %"


def render(posts, evaluation=None, classifier_name: str = "lexicon baseline") -> str:
    lines = ["# SMM analytics report", ""]
    lines += [f"Posts analysed: **{len(posts)}**, total reach: **{sum(p.reach for p in posts)}**.", ""]
    for by in ("channel", "format", "weekday"):
        lines += [f"## Engagement rate by {by}", "", "| Group | Posts | Mean ER | Median ER | Reach |"]
        lines += ["|---|---:|---:|---:|---:|"]
        for s in summarise(posts, by=by):
            lines.append(
                f"| {s.key} | {s.posts} | {_pct(s.mean_er)} | {_pct(s.median_er)} | {s.total_reach} |"
            )
        lines.append("")
    lines += ["## Recommended publication slots", "", "| Weekday | Hours | Mean ER | Posts |"]
    lines += ["|---|---|---:|---:|"]
    for slot in best_slots(posts):
        lines.append(f"| {slot.weekday} | {slot.hours} | {_pct(slot.mean_er)} | {slot.posts} |")
    lines.append("")
    if evaluation is not None:
        lines += [f"## Sentiment classifier: {classifier_name}", ""]
        lines += [
            f"Accuracy **{evaluation.accuracy:.3f}**, macro-F1 **{evaluation.macro_f1:.3f}**.",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, p, r, f1, n in evaluation.as_rows():
            lines.append(f"| {label} | {p:.3f} | {r:.3f} | {f1:.3f} | {n} |")
        lines.append("")
    return "\n".join(lines)
