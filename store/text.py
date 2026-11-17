"""Text helpers shared by the P1 indexers and stores (architecture §6.4, §6.5)."""

from __future__ import annotations

import hashlib
import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOPWORDS = frozenset(
    (  # noqa: SIM905
        "a an and are as at be been but by can cannot could did do does for from had has have i "
        "i'm im if in into is it its it's me my no not of on or our so some than that the their "
        "them then there these they this to too us was we were what when where which while who "
        "will with would you your please hi hello thanks help"
    ).split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25. Keeps codes like ``ora-01013`` and ``0x800f0922`` whole."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def norm_code(code: str) -> str:
    """Error-code normalisation (§6.4): lowercase, punctuation stripped, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", code.lower())).strip()


def embedding_text(title: str | None, description: str | None) -> str:
    """§6.5: ``"{title}. {description}"`` unless the title repeats the description."""
    title = (title or "").strip()
    description = (description or "").strip()
    if not title or title == description:
        return description
    if not description:
        return title
    return f"{title.rstrip('.')}. {description}"


def text_hash(text: str) -> str:
    """Normalised text hash (lowercase, collapsed whitespace); same rule as ``ai.ticket_search``."""
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
