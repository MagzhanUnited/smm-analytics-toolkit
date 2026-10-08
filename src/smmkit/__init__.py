"""smmkit: reproducible engagement metrics and a sentiment baseline for SMM research."""

from smmkit.anonymise import pseudonymise, scrub_text
from smmkit.io import Comment, DataError, Post, load_comments, load_posts

__version__ = "0.1.0"

__all__ = [
    "Comment",
    "DataError",
    "Post",
    "load_comments",
    "load_posts",
    "pseudonymise",
    "scrub_text",
]
