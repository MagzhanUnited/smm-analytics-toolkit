"""Engagement metrics and publication-slot recommendations.

Engagement rate (ER) is interactions divided by reach, the definition used by
Instagram and Telegram analytics exports. A post with zero reach has ER 0.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
GROUPINGS = ("channel", "format", "weekday", "hour")


@dataclass(frozen=True)
class GroupStat:
    key: str
    posts: int
    mean_er: float
    median_er: float
    total_reach: int


@dataclass(frozen=True)
class Slot:
    weekday: str
    hours: str  # three-hour block, e.g. "18-21"
    mean_er: float
    posts: int


def engagement_rate(post) -> float:
    return post.interactions / post.reach if post.reach else 0.0


def _key(post, by: str) -> str:
    if by == "channel":
        return post.channel
    if by == "format":
        return post.format
    if by == "weekday":
        return WEEKDAYS[post.published_at.weekday()]
    if by == "hour":
        return f"{post.published_at.hour:02d}"
    raise ValueError(f"unknown grouping {by!r}; expected one of {', '.join(GROUPINGS)}")


def summarise(posts, by: str = "channel") -> list:
    """Aggregate ER per group, sorted by mean ER descending."""
    groups = defaultdict(list)
    for post in posts:
        groups[_key(post, by)].append(post)
    stats = [
        GroupStat(
            key=key,
            posts=len(items),
            mean_er=mean(engagement_rate(p) for p in items),
            median_er=median(engagement_rate(p) for p in items),
            total_reach=sum(p.reach for p in items),
        )
        for key, items in groups.items()
    ]
    return sorted(stats, key=lambda s: (-s.mean_er, s.key))


def best_slots(posts, top: int = 3, min_posts: int = 2) -> list:
    """Recommend weekday x three-hour slots with the highest mean ER.

    Slots with fewer than ``min_posts`` posts are ignored: one lucky post is
    not evidence.
    """
    groups = defaultdict(list)
    for post in posts:
        start = post.published_at.hour // 3 * 3
        key = (post.published_at.weekday(), f"{start:02d}-{start + 3:02d}")
        groups[key].append(engagement_rate(post))
    slots = [
        Slot(weekday=WEEKDAYS[wd], hours=hours, mean_er=mean(ers), posts=len(ers))
        for (wd, hours), ers in groups.items()
        if len(ers) >= min_posts
    ]
    return sorted(slots, key=lambda s: (-s.mean_er, s.weekday, s.hours))[:top]
