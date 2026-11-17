"""Small deterministic text helpers shared by the AI modules (no external deps)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

_STOPWORDS_TEXT = """a an and are as at be been but by can cannot could did do does doing for from had has have
    having he her here him his how i if in into is it its itself just me more most my no nor not
    now of off on once only or other our out over own same she should so some such than that the
    their them then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours also again still
    since already every each all any am get got im ive dont cant wont being one two after before"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())


def tokenize(
    text: str, *, drop_stopwords: bool = True, do_stem: bool = True
) -> list[str]:
    """Lowercase word tokens; keeps codes like `ora-01013` and `0xc0000005` intact."""
    toks = _TOKEN.findall(text.lower())
    out: list[str] = []
    for t in toks:
        if drop_stopwords and t in STOPWORDS:
            continue
        out.append(stem(t) if do_stem else t)
    return out


def stem(tok: str) -> str:
    """Very light suffix stripping so `drops`/`dropping`/`dropped` meet."""
    if len(tok) <= 4 or not tok.isalpha():
        return tok
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            base = tok[: -len(suf)]
            if suf in ("ing", "ed") and len(base) >= 3 and base[-1] == base[-2]:
                base = base[:-1]  # dropping -> dropp -> drop
            return base
    return tok


def normalize_code(code: str) -> str:
    """Error-code index normalisation (architecture §6.4): lowercase, collapse spaces,
    strip punctuation."""
    s = re.sub(r"[^\w\s-]", " ", code.lower())
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


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


def rouge_l_recall(candidate: str, reference: str) -> float:
    """ROUGE-L recall of `candidate` against `reference` tokens: how much of the step is
    supported by the cited text, measured as LCS / len(candidate)."""
    c = tokenize(candidate)
    r = tokenize(reference)
    if not c:
        return 0.0
    return lcs_len(c, r) / len(c)


def display_terms(text: str, wanted_stems: set[str] | None = None) -> list[str]:
    """Original (unstemmed) content words of `text`, optionally only those whose stem is in
    `wanted_stems`; order of first appearance, no duplicates."""
    out: dict[str, str] = {}
    for word in tokenize(text, do_stem=False):
        st = stem(word)
        if st in out or (wanted_stems is not None and st not in wanted_stems):
            continue
        out[st] = word
    return list(out.values())


def truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: max(0, n - 1)].rstrip() + "…"


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))
