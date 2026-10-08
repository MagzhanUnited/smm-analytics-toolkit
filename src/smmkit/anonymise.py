"""Removal of personal data before anything is stored or analysed.

Comments under social-media posts contain e-mail addresses, phone numbers and
user handles. Research data must not keep them, so every text passes through
:func:`scrub_text`, and author identifiers are replaced with a salted hash by
:func:`pseudonymise` (the same author keeps the same pseudonym within one
dataset, which is enough for per-author statistics).
"""

from __future__ import annotations

import hashlib
import re

# Order matters: e-mails contain "@", so they are replaced before handles.
_PATTERNS = [
    ("<URL>", re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)),
    ("<EMAIL>", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("<PHONE>", re.compile(r"(?<!\d)(?:\+?7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")),
    ("<USER>", re.compile(r"(?<![\w.])@[A-Za-z0-9_.]{2,}")),
]


def scrub_text(text: str) -> str:
    """Replace URLs, e-mails, Kazakhstani phone numbers and @handles with placeholders."""
    for placeholder, pattern in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def pseudonymise(identifier: str, salt: str) -> str:
    """Return a stable 12-character pseudonym for an author id.

    The salt is kept outside the dataset; without it the pseudonym cannot be
    reversed by hashing a list of known usernames.
    """
    if not salt:
        raise ValueError("salt must be a non-empty string")
    digest = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return digest[:12]
