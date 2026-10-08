"""smmkit: reproducible engagement metrics and a sentiment baseline for SMM research."""

from smmkit.anonymise import pseudonymise, scrub_text
from smmkit.io import Comment, DataError, Post, load_comments, load_posts
from smmkit.metrics import GroupStat, Slot, best_slots, engagement_rate, summarise
from smmkit.sentiment import LABELS, EvaluationReport, LexiconClassifier, evaluate

__version__ = "0.1.0"

__all__ = [
    "LABELS",
    "Comment",
    "DataError",
    "EvaluationReport",
    "GroupStat",
    "LexiconClassifier",
    "Post",
    "Slot",
    "best_slots",
    "engagement_rate",
    "evaluate",
    "load_comments",
    "load_posts",
    "pseudonymise",
    "scrub_text",
    "summarise",
]
