"""Loading of post and comment exports with schema validation.

Errors name the file, the row and the column, so a broken export is fixed at
the source instead of silently skewing the metrics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

from smmkit.anonymise import pseudonymise, scrub_text

PathLike = Union[str, Path]

POST_COLUMNS = (
    "post_id",
    "channel",
    "published_at",
    "format",
    "reach",
    "likes",
    "comments",
    "shares",
    "saves",
)
COMMENT_COLUMNS = ("comment_id", "post_id", "author", "text")
VALID_LABELS = ("positive", "neutral", "negative")


class DataError(ValueError):
    """Raised when an input file does not match the expected schema."""


@dataclass(frozen=True)
class Post:
    post_id: str
    channel: str
    published_at: datetime
    format: str
    reach: int
    likes: int
    comments: int
    shares: int
    saves: int

    @property
    def interactions(self) -> int:
        return self.likes + self.comments + self.shares + self.saves


@dataclass(frozen=True)
class Comment:
    comment_id: str
    post_id: str
    author: str  # pseudonym, never the original handle
    text: str  # scrubbed text
    label: str | None = None


def _read_rows(path: PathLike, required: tuple) -> list:
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path}: file not found")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"{path}: missing columns: {', '.join(missing)}")
        return [(i, row) for i, row in enumerate(reader, start=2)]


def _int(path: PathLike, line: int, row: dict, column: str) -> int:
    raw = (row.get(column) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        raise DataError(f"{path}:{line}: column '{column}' must be an integer, got {raw!r}") from None
    if value < 0:
        raise DataError(f"{path}:{line}: column '{column}' must be non-negative, got {value}")
    return value


def load_posts(path: PathLike) -> list:
    """Load posts from a CSV export. ``published_at`` is ISO 8601 (``2026-09-14T18:30``)."""
    posts = []
    for line, row in _read_rows(path, POST_COLUMNS):
        try:
            published = datetime.fromisoformat(row["published_at"].strip())
        except ValueError:
            raise DataError(
                f"{path}:{line}: column 'published_at' is not ISO 8601: {row['published_at']!r}"
            ) from None
        posts.append(
            Post(
                post_id=row["post_id"].strip(),
                channel=row["channel"].strip().lower(),
                published_at=published,
                format=row["format"].strip().lower(),
                reach=_int(path, line, row, "reach"),
                likes=_int(path, line, row, "likes"),
                comments=_int(path, line, row, "comments"),
                shares=_int(path, line, row, "shares"),
                saves=_int(path, line, row, "saves"),
            )
        )
    return posts


def load_comments(path: PathLike, salt: str) -> list:
    """Load comments, scrub personal data and pseudonymise authors.

    An optional ``label`` column (positive / neutral / negative) is kept for
    evaluation of classifiers.
    """
    comments = []
    for line, row in _read_rows(path, COMMENT_COLUMNS):
        label = (row.get("label") or "").strip().lower() or None
        if label is not None and label not in VALID_LABELS:
            raise DataError(f"{path}:{line}: unknown label {label!r}")
        comments.append(
            Comment(
                comment_id=row["comment_id"].strip(),
                post_id=row["post_id"].strip(),
                author=pseudonymise(row["author"].strip(), salt),
                text=scrub_text(row["text"].strip()),
                label=label,
            )
        )
    return comments
