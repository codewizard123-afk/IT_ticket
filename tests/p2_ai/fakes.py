"""In-memory fakes for P2 tests (stand-ins for `shared/stubs.py` until Phase 0 lands).

- `FakeKBStore`: field-level chunks built exactly like architecture §6.4, error-code index,
  token-cosine "dense" search and a small BM25.
- `FakeTicketStore`: historical/verified/live collections with Chroma-like `where` filters.
- `FakeLLM`: canned JSON per purpose, or always unavailable.

Dense scores are a token-cosine mapped into the range a normalised sentence embedding
produces (`0.3 + 0.7*sqrt(cos)`, 0 when no tokens overlap), so the thresholds in
`config/ai.yaml` behave sensibly. Real scores come from P1's stores.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai.text import normalize_code, tokenize
from ai.types import ChunkHit, KBArticle, LLMBadOutput, LLMUnavailable, TicketHit

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- scoring


def _vec(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def dense_like(cos: float) -> float:
    return 0.0 if cos <= 0 else round(min(1.0, 0.3 + 0.7 * math.sqrt(cos)), 4)


class MiniBM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = list(docs)
        self.toks = {i: tokenize(t) for i, t in docs.items()}
        self.k1, self.b = k1, b
        n = len(self.ids) or 1
        self.avgdl = sum(len(t) for t in self.toks.values()) / n
        df: Counter[str] = Counter()
        for t in self.toks.values():
            df.update(set(t))
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def top(self, text: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(text)
        out: list[tuple[str, float]] = []
        for i in self.ids:
            tf = Counter(self.toks[i])
            dl = len(self.toks[i]) or 1
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += (
                        self.idf.get(w, 0)
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    )
            if s > 0:
                out.append((i, s))
        out.sort(key=lambda x: -x[1])
        return out[:k]


def _match_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_match_where(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_match_where(meta, w) for w in where["$or"])
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, ref in cond.items():
                if op == "$eq" and val != ref:
                    return False
                if op == "$ne" and val == ref:
                    return False
                if op == "$in" and val not in ref:
                    return False
                if op == "$nin" and val in ref:
                    return False
                if op == "$gte" and (val is None or val < ref):
                    return False
                if op == "$lte" and (val is None or val > ref):
                    return False
        elif val != cond:
            return False
    return True


# --------------------------------------------------------------------------- KB


class FakeKBStore:
    def __init__(self, articles: list[dict[str, Any]] | None = None) -> None:
        raw = articles if articles is not None else load_fixture("kb_articles.json")
        self.articles = {a["kb_id"]: KBArticle.from_dict(a) for a in raw}
        self.chunks: dict[str, tuple[str, str, str]] = {}  # id -> (kb_id, type, text)
        self.code_index: dict[str, list[str]] = {}
        for art in self.articles.values():
            self._chunk(art)
        searchable = {
            cid: text
            for cid, (_, ctype, text) in self.chunks.items()
            if ctype != "resolution"
        }
        self._vecs = {cid: _vec(t) for cid, t in searchable.items()}
        self._bm25 = MiniBM25(searchable)
        self.query_log: list[str] = []

    def _chunk(self, a: KBArticle) -> None:
        kid = a.kb_id
        self.chunks[f"{kid}#title"] = (kid, "title_root", f"{a.title}. {a.root_cause}")
        for i, s in enumerate(a.symptoms, 1):
            self.chunks[f"{kid}#sym.{i}"] = (kid, "symptom", s)
        if a.common_error_codes:
            self.chunks[f"{kid}#err"] = (kid, "errors", "; ".join(a.common_error_codes))
        for i, s in enumerate(a.diagnostic_steps, 1):
            self.chunks[f"{kid}#diag.{i}"] = (kid, "diagnostic", s)
        for i, s in enumerate(a.resolution.steps, 1):
            self.chunks[f"{kid}#res.{i}"] = (kid, "resolution", s)
        if a.escalation_criteria:
            self.chunks[f"{kid}#esc"] = (
                kid,
                "escalation",
                "; ".join(a.escalation_criteria),
            )
        for code in a.common_error_codes:
            self.code_index.setdefault(normalize_code(code), [])
            if kid not in self.code_index[normalize_code(code)]:
                self.code_index[normalize_code(code)].append(kid)

    # protocol ---------------------------------------------------------------
    def query_chunks(
        self, text: str, k: int, where: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        self.query_log.append(text)
        q = _vec(text)
        hits: list[ChunkHit] = []
        for cid, v in self._vecs.items():
            kid, ctype, ctext = self.chunks[cid]
            meta = {
                "kb_id": kid,
                "chunk_type": ctype,
                "status": self.articles[kid].status,
            }
            if not _match_where(meta, where):
                continue
            s = dense_like(cosine(q, v))
            if s > 0:
                hits.append(
                    ChunkHit(
                        chunk_id=cid, kb_id=kid, chunk_type=ctype, score=s, text=ctext
                    )
                )
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return self._bm25.top(text, k)

    def lookup_codes(self, codes: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for c in codes:
            ids = self.code_index.get(normalize_code(c))
            if ids:
                out[c] = list(ids)
        return out

    def get_article(self, kb_id: str) -> KBArticle:
        return self.articles[kb_id]

    def chunk_text(self, chunk_id: str) -> str:
        return self.chunks[chunk_id][2]

    def all_error_phrases(self) -> list[str]:
        return sorted({c for a in self.articles.values() for c in a.common_error_codes})


# --------------------------------------------------------------------------- tickets


class FakeTicketStore:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        recs = records if records is not None else load_fixture("mini_tickets.json")
        self.records: dict[str, dict[str, Any]] = {}
        for r in recs:
            self.add(r)
        self.where_log: list[dict[str, Any] | None] = []

    def add(self, r: dict[str, Any]) -> None:
        r = dict(r)
        r.setdefault("collection", "tickets_hist")
        text = r.get("text") or (
            f"{r['title']}. {r['description']}"
            if r.get("title") != r.get("description")
            else r["description"]
        )
        r["text"] = text
        self.records[r["ticket_id"]] = r
        docs = {i: rec["text"] for i, rec in self.records.items()}
        self._vecs = {i: _vec(t) for i, t in docs.items()}
        self._bm25 = MiniBM25(docs)

    def _hit(self, rec: dict[str, Any], score: float) -> TicketHit:
        meta = {k: v for k, v in rec.items() if k not in ("text", "collection")}
        return TicketHit(
            ticket_id=rec["ticket_id"],
            score=score,
            text=rec["text"],
            metadata=meta,
            collection=rec["collection"],
        )

    def query(
        self,
        text: str,
        k: int,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = (
            "tickets_hist",
            "tickets_verified",
            "tickets_live",
        ),
    ) -> list[TicketHit]:
        self.where_log.append(where)
        q = _vec(text)
        hits = []
        for tid, rec in self.records.items():
            if rec["collection"] not in collections or not _match_where(rec, where):
                continue
            s = dense_like(cosine(q, self._vecs[tid])) if text.strip() else 0.0
            if s > 0 or not text.strip():
                hits.append(self._hit(rec, s))
        hits.sort(key=lambda h: (-h.score, h.ticket_id))
        return hits[:k]

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return self._bm25.top(text, k)

    def get(self, ids: list[str]) -> list[TicketHit]:
        return [self._hit(self.records[i], 0.0) for i in ids if i in self.records]


# --------------------------------------------------------------------------- LLM


class FakeLLM:
    """`responses[purpose]` is a dict, or a callable(messages) -> dict. `None` -> offline."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any] | Callable[[list[dict[str, str]]], Any]]
        | None = None,
        *,
        available: bool = True,
    ) -> None:
        self.responses = responses or {}
        self._available = available
        self.calls: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self._available

    def complete_json(
        self,
        *,
        purpose: str,
        model: str,
        messages: list[dict[str, str]],
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"purpose": purpose, "model": model, "messages": messages})
        if not self._available:
            raise LLMUnavailable("offline")
        resp = self.responses.get(purpose)
        if resp is None:
            raise LLMUnavailable(f"no canned response for {purpose}")
        out = resp(messages) if callable(resp) else resp
        if not isinstance(out, dict):
            raise LLMBadOutput("canned response is not a dict")
        return json.loads(json.dumps(out))


def offline_llm() -> FakeLLM:
    return FakeLLM(available=False)
