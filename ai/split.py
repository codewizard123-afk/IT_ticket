"""Issue splitter (architecture §7.2). Owner: P2.

Trigger: ``signals.multi_issue_markers``, or the top-2 fused categories differ and both score
≥ ``classifier.split_min`` (0.35). Method: the split proposed by Groq call #1; fallback is a regex
split on ``(1)/(2)``, numbered lists, or the sentence boundary where the product changes.
At most 2 issues. Ordering by priority (primary = more severe) is done by the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ai.classify import ClassifyOutcome, Label, rule_vote, top_categories
from ai.config import ai_config
from shared.constants import MAX_ISSUES
from shared.schemas import Signals

_PAREN = re.compile(r"\(\s*([12])\s*\)")
_NUMBERED = re.compile(r"(?m)^\s*([12])[.)]\s+")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")


@dataclass
class IssueSpan:
    text: str
    llm_scores: dict[Label, float] = field(default_factory=dict)
    intent: str | None = None


def should_split(signals: Signals, outcome: ClassifyOutcome, cfg: dict[str, Any] | None = None):
    if signals.multi_issue_markers:
        return True
    split_min = float((cfg or ai_config())["classifier"]["split_min"])
    cats = top_categories(outcome.fused)
    return len(cats) >= 2 and cats[0][1] >= split_min and cats[1][1] >= split_min


def _rule_top(text: str) -> Label | None:
    votes, _ = rule_vote(text)
    if not votes:
        return None
    return max(votes, key=lambda k: votes[k])


def _rule_categories(text: str) -> set[str]:
    votes, _ = rule_vote(text)
    return {cat for cat, _ in votes}


def _marker_split(text: str) -> list[str] | None:
    for pattern in (_PAREN, _NUMBERED):
        marks = [(m.start(), m.end(), m.group(1)) for m in pattern.finditer(text)]
        ones = [m for m in marks if m[2] == "1"]
        twos = [m for m in marks if m[2] == "2"]
        if ones and twos and ones[0][0] < twos[0][0]:
            first = text[ones[0][1] : twos[0][0]].strip()
            second = text[twos[0][1] :].strip()
            if first and second:
                return [first, second]
    return None


def _sentence_split(text: str) -> list[str] | None:
    """Split where single-category sentences change category; mixed sentences are skipped."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]
    tagged = [(s, _rule_categories(s)) for s in sentences]
    single = [(s, next(iter(c))) for s, c in tagged if len(c) == 1]
    if not single:
        return None
    first_cat = single[0][1]
    first = [s for s, c in single if c == first_cat]
    rest = [s for s, c in single if c != first_cat]
    if not rest:
        return None
    second_cat = rest[0][1]
    second = [s for s, c in rest if c == second_cat]
    return [" ".join(first), " ".join(second)]


def _distinct(spans: list[str]) -> bool:
    labels = [_rule_top(s) for s in spans]
    return not (labels[0] is not None and labels[0] == labels[1])


def split_issues(
    text: str,
    signals: Signals,
    outcome: ClassifyOutcome,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[IssueSpan], list[str]]:
    """Return 1–2 issue spans (in order of mention) and any degraded flags."""
    primary_llm = outcome.llm_issues[0] if outcome.llm_issues else None
    whole = IssueSpan(
        text=text,
        llm_scores=primary_llm.scores if primary_llm else {},
        intent=primary_llm.intent if primary_llm else None,
    )
    if not should_split(signals, outcome, cfg):
        return [whole], []

    llm_spans = [i for i in outcome.llm_issues if i.text_span]
    if len(llm_spans) >= 2:
        spans = [IssueSpan(i.text_span, i.scores, i.intent) for i in llm_spans[:MAX_ISSUES]]
        return spans, []
    if outcome.llm_issues:  # the LLM ran and saw a single issue: trust it
        return [whole], []

    degraded = ["llm_split_fallback"]
    parts = _marker_split(text) or _sentence_split(text)
    if not parts or len(parts) < 2 or not _distinct(parts):
        return [whole], degraded
    return [IssueSpan(p) for p in parts[:MAX_ISSUES]], degraded
