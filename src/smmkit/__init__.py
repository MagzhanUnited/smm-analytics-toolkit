"""smmkit: reproducible engagement metrics and a sentiment baseline for SMM research."""

from smmkit.anonymise import pseudonymise, scrub_text
from smmkit.io import Comment, DataError, Post, load_comments, load_posts
from smmkit.metrics import GroupStat, Slot, best_slots, engagement_rate, summarise

__version__ = "0.1.0"

__all__ = [
    "Comment",
    "DataError",
    "GroupStat",
    "Post",
    "Slot",
    "best_slots",
    "engagement_rate",
    "load_comments",
    "load_posts",
    "pseudonymise",
    "scrub_text",
    "summarise",
]
