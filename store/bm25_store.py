"""Pure-Python BM25 (Okapi) index used for tickets and KEDB chunks (architecture §6.4, §6.5).

No numpy, so it also runs in CI where only the dev requirements are installed. Indexes are
persisted as pickles of plain Python data under ``indexes/`` (gitignored, rebuilt by
``make data``).
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from store.text import tokenize


class BM25Index:
    """Incremental BM25Okapi. ``add``/``remove`` keep document frequencies consistent."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._total_len = 0

    # --- building ------------------------------------------------------------------------
    @classmethod
    def build(cls, ids: Sequence[str], texts: Sequence[str], **kw: float) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        idx = cls(**kw)
        for doc_id, text in zip(ids, texts, strict=True):
            idx.add(doc_id, text)
        return idx

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._tf:
            self.remove(doc_id)
        toks = tokenize(text)
        tf = Counter(toks)
        self._tf[doc_id] = tf
        self._len[doc_id] = len(toks)
        self._total_len += len(toks)
        self._df.update(tf.keys())

    def remove(self, doc_id: str) -> None:
        tf = self._tf.pop(doc_id, None)
        if tf is None:
            return
        self._total_len -= self._len.pop(doc_id)
        self._df.subtract(tf.keys())
        for term in tf:
            if self._df[term] <= 0:
                del self._df[term]

    def remove_where(self, predicate) -> None:  # noqa: ANN001 - simple callable(doc_id) -> bool
        for doc_id in [d for d in self._tf if predicate(d)]:
            self.remove(doc_id)

    # --- querying ------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tf)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._tf

    @property
    def ids(self) -> list[str]:
        return list(self._tf)

    def _idf(self, term: str) -> float:
        n = len(self._tf)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, text: str, ids: Iterable[str] | None = None) -> dict[str, float]:
        q = tokenize(text)
        if not q or not self._tf:
            return {}
        avgdl = self._total_len / len(self._tf) or 1.0
        idf = {t: self._idf(t) for t in set(q)}
        out: dict[str, float] = {}
        for doc_id in ids if ids is not None else self._tf:
            tf = self._tf.get(doc_id)
            if not tf:
                continue
            norm = self.k1 * (1 - self.b + self.b * self._len[doc_id] / avgdl)
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += idf[term] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                out[doc_id] = s
        return out

    def search(
        self, text: str, k: int, ids: Iterable[str] | None = None
    ) -> list[tuple[str, float]]:
        """Top-``k`` ``(doc_id, score)`` pairs, best first; ties broken by id for determinism."""
        ranked = sorted(self.scores(text, ids).items(), key=lambda kv: (-kv[1], kv[0]))
        return [(d, round(s, 4)) for d, s in ranked[: max(k, 0)]]

    # --- persistence ---------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self.k1,
            "b": self.b,
            "tf": {d: dict(tf) for d, tf in self._tf.items()},
            "len": self._len,
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        # Only ever loads files this module wrote itself under indexes/ (gitignored).
        with path.open("rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        idx = cls(k1=state["k1"], b=state["b"])
        for doc_id, tf in state["tf"].items():
            counter = Counter(tf)
            idx._tf[doc_id] = counter
            idx._len[doc_id] = state["len"][doc_id]
            idx._total_len += state["len"][doc_id]
            idx._df.update(counter.keys())
        return idx
