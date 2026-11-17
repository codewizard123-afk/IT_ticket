"""Issue splitter (architecture §7.2): at most 2 issues per ticket.

The split is normally requested inside Groq call #1 (`ai.classify.llm_classify`). This module
decides whether a split is warranted and provides the deterministic regex fallback.
"""

from __future__ import annotations

import re
from typing import Any

from ai.config import ai_config, rules_config
from ai.signals import Signals
from ai.text import word_count


def should_split(
    signals: Signals,
    label_scores: dict[tuple[str, str], float] | None = None,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Trigger: multi-issue markers, or the top-2 categories differ and both score ≥ 0.35."""
    if signals.multi_issue_markers:
        return True
    if not label_scores:
        return False
    cfg = cfg if cfg is not None else ai_config()
    min_score = float(cfg.get("classifier", {}).get("split_min_score", 0.35))
    by_cat: dict[str, float] = {}
    for (cat, _sub), s in label_scores.items():
        by_cat[cat] = by_cat.get(cat, 0.0) + s
    top = sorted(by_cat.values(), reverse=True)[:2]
    return len(top) == 2 and top[1] >= min_score and top[0] >= min_score


def _clean(span: str) -> str:
    return " ".join(span.strip(" \n\t-:;,").split())


def regex_split(text: str, products: dict[str, list[str]] | None = None) -> list[str]:
    """Deterministic fallback: `(1)/(2)` markers, numbered lines, or the sentence boundary
    where the mentioned product changes. Returns 1 or 2 spans."""
    rcfg = rules_config().get("split", {})
    min_words = int(rcfg.get("min_span_words", 4))

    for key in ("enumerated", "numbered_line"):
        pat = rcfg.get(key)
        if not pat:
            continue
        parts = re.split(pat, text)
        # re.split with a capture group -> [pre, "1", body1, "2", body2, ...]
        bodies = [_clean(parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
        bodies = [b for b in bodies if word_count(b) >= min_words]
        if len(bodies) >= 2:
            return [bodies[0], " ".join(bodies[1:])]

    if products:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        tagged = []
        for s in sentences:
            low = s.lower()
            tags = {p for p, kws in products.items() if any(k in low for k in kws)}
            tagged.append((s, tags))
        first_tags: set[str] = set()
        for idx, (_s, tags) in enumerate(tagged):
            if not tags:
                continue
            if not first_tags:
                first_tags = tags
                continue
            if not (tags & first_tags):
                a = _clean(" ".join(s for s, _ in tagged[:idx]))
                b = _clean(" ".join(s for s, _ in tagged[idx:]))
                if word_count(a) >= min_words and word_count(b) >= min_words:
                    return [a, b]
                break
    return [_clean(text)]


def split(
    text: str,
    signals: Signals,
    llm_spans: list[str] | None,
    *,
    label_scores: dict[tuple[str, str], float] | None = None,
    products: dict[str, list[str]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[str], bool]:
    """Return `(spans, used_fallback)`. `llm_spans` are the issue spans from Groq call #1
    (None when the LLM was unavailable)."""
    cfg = cfg if cfg is not None else ai_config()
    max_issues = int(cfg.get("classifier", {}).get("max_issues", 2))
    if not should_split(signals, label_scores, cfg):
        return [_clean(text)], False
    if llm_spans:
        spans = [_clean(s) for s in llm_spans if s and _clean(s)]
        if len(spans) >= 2:
            return spans[:max_issues], False
        if len(spans) == 1:
            return spans, False
    return regex_split(text, products)[:max_issues], True
