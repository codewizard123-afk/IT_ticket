"""Small text utilities shared by the synthesiser and the validator (no dependencies)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*")
_STOP_WORDS = (
    "a an and are as at be but by for from has have i in is it its my me of on or our so that "
    "the this to was we with your you if then than when it's into onto via any all"
)
STOP = frozenset(_STOP_WORDS.split())


def stem(word: str) -> str:
    """Tiny suffix stripper: tried→try, restarted→restart, cables→cable, cleared→clear."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ied"):
        return w[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]  # stopped → stopp → stop
    return w


def content_tokens(text: str) -> list[str]:
    """Lowercased, stop-word-free, stemmed tokens in order."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP]


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def grounding(step: str, cited: str) -> float:
    """ROUGE-L of the step against its cited text, as the share of the step's content tokens
    recovered in order from the cited text (1.0 = fully supported)."""
    s = content_tokens(step)
    if not s:
        return 0.0
    return lcs_len(s, content_tokens(cited)) / len(s)


def token_overlap(a: str, b: str) -> float:
    """Share of ``a``'s content tokens present in ``b``."""
    ta = set(content_tokens(a))
    if not ta:
        return 0.0
    return len(ta & set(content_tokens(b))) / len(ta)
