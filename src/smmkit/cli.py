"""Command-line interface: ``smmkit report``, ``smmkit metrics``, ``smmkit evaluate``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smmkit import __version__
from smmkit.io import DataError, load_comments, load_posts
from smmkit.metrics import GROUPINGS, summarise
from smmkit.report import render
from smmkit.sentiment import LexiconClassifier, evaluate

SALT_ENV = "SMMKIT_SALT"


def _salt(args) -> str:
    return args.salt or os.environ.get(SALT_ENV) or "local-dev-salt"


def _evaluate(comments):
    labelled = [c for c in comments if c.label]
    if not labelled:
        raise DataError("no labelled comments: add a 'label' column to evaluate a classifier")
    predictions = LexiconClassifier().predict(c.text for c in labelled)
    return evaluate([c.label for c in labelled], predictions)


def cmd_metrics(args) -> int:
    for s in summarise(load_posts(args.posts), by=args.by):
        print(f"{s.key:<12} posts={s.posts:<4} mean_er={s.mean_er:.4f} median_er={s.median_er:.4f}")
    return 0


def cmd_evaluate(args) -> int:
    ev = _evaluate(load_comments(args.comments, _salt(args)))
    print(f"accuracy={ev.accuracy:.3f} macro_f1={ev.macro_f1:.3f}")
    for label, p, r, f1, n in ev.as_rows():
        print(f"{label:<9} precision={p:.3f} recall={r:.3f} f1={f1:.3f} support={n}")
    return 0


def cmd_report(args) -> int:
    posts = load_posts(args.posts)
    ev = _evaluate(load_comments(args.comments, _salt(args))) if args.comments else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render(posts, ev), encoding="utf-8")
    if args.plots:
        from smmkit import viz

        viz.engagement_heatmap(posts, out / "engagement_heatmap.png")
        if ev is not None:
            viz.confusion_matrix(ev, out / "confusion_matrix.png")
    print(f"report written to {out / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smmkit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("metrics", help="engagement rate grouped by a dimension")
    m.add_argument("posts")
    m.add_argument("--by", choices=GROUPINGS, default="channel")
    m.set_defaults(func=cmd_metrics)

    e = sub.add_parser("evaluate", help="evaluate the sentiment baseline on labelled comments")
    e.add_argument("comments")
    e.add_argument("--salt", help=f"pseudonymisation salt (default: ${SALT_ENV})")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="write a markdown report (and optional plots)")
    r.add_argument("--posts", required=True)
    r.add_argument("--comments")
    r.add_argument("--out", default="out")
    r.add_argument("--plots", action="store_true", help="also write PNG plots (needs matplotlib)")
    r.add_argument("--salt", help=f"pseudonymisation salt (default: ${SALT_ENV})")
    r.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
