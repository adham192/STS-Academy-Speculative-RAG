"""Question normalisation and hashing, shared by Redis and PostgreSQL."""

import hashlib
import unicodedata

import settings

# Trailing characters that never change what a question means.
_TRAILING = " ?!.\u061f"  # \u061f = Arabic question mark


def normalize_question(question: str) -> str:
    """Unicode-normalise, case-fold, collapse whitespace, drop trailing punctuation."""
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(text.split()).rstrip(_TRAILING)


def clip_for_log(text, limit: int | None = None) -> str:
    """Shorten long text for a log line; the %r in the log call keeps it on one line."""
    limit = limit or settings.LOG_TEXT_MAX_CHARS
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}... [+{len(text) - limit} chars]"


def question_hash(normalized_question: str, scope: str = "") -> str:
    """SHA-256 hex digest of the normalized question.

    `scope` (the PDF a question was restricted to, "" for all documents) is part of the
    digest because the same question gives a different answer per document scope.
    """
    return hashlib.sha256(f"{normalized_question}\x1f{scope}".encode("utf-8")).hexdigest()