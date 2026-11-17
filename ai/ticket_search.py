"""Similar-ticket retrieval and duplicate detection (architecture §7.5, FR-02, FR-07).

1. Dense search per collection (hist, verified, live) + BM25, fused with RRF (k=60).
   The RRF score is normalised to [0, 1] (divided by its maximum, 2/(k+1)) before bonuses.
2. Bonuses: +0.10 same category, +0.05 same subcategory, +0.05 verified resolution.
3. `where` filters (FR-07) and role-based `restricted` filtering.
4. Keyword-trap guard: "related" needs cosine ≥ related_min AND (same predicted category, or
   cross-encoder ≥ rerank_min when a reranker is injected). BM25-only hits → `keyword_only`.
5. Tiers: DUPLICATE (same text hash) · LIKELY DUPLICATE (cos ≥ 0.92, same subcategory) ·
   SIMILAR (cos ≥ 0.75) · RELATED (passes the guard).
6. Live clustering: ≥ 3 similar live tickets within 72 h in status new/in_review →
   `possible_major_incident`.
7. Pattern summary over the retrieved pool.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from typing import Any

from ai.config import ai_config
from ai.text import display_terms, tokenize, truncate
from ai.types import (
    VERIFIED_ORIGINS,
    ClassificationResult,
    SimilarIncident,
    TicketHit,
    TicketSearchResult,
    TicketStore,
)

COLLECTIONS = ("tickets_hist", "tickets_verified", "tickets_live")
Reranker = Callable[[str, str], float]

# FR-07 filter keys that map 1:1 onto Chroma metadata equality
_EQ_FILTERS = (
    "category",
    "subcategory",
    "priority",
    "status",
    "source",
    "assignment_group",
    "application",
    "environment",
    "resolution_origin",
)


def text_hash(text: str) -> str:
    """Normalised text hash (lowercase, collapsed whitespace)."""
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def _same_hash(a: str, b: str | None) -> bool:
    if not b:
        return False
    n = min(len(a), len(b))
    return n >= 8 and a[:n] == b[:n]


def role_where(role: str | None, group: str | None = None) -> dict[str, Any] | None:
    """Restricted-ticket rule (architecture §8.6): l1/kb_owner never see restricted tickets,
    l2 only their own group's, admin all. Unknown role → most restrictive."""
    if role == "admin":
        return None
    if role == "l2" and group:
        return {"$or": [{"restricted": False}, {"assignment_group": group}]}
    return {"restricted": False}


def matches_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a Chroma-style `where` clause against one metadata dict (used for BM25
    hits, which do not go through Chroma)."""
    if not where:
        return True
    if "$and" in where:
        return all(matches_where(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(matches_where(meta, w) for w in where["$or"])
    for key, cond in where.items():
        val = meta.get(key)
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, ref in cond.items():
            ok = {
                "$eq": lambda v, r: v == r,
                "$ne": lambda v, r: v != r,
                "$in": lambda v, r: v in r,
                "$nin": lambda v, r: v not in r,
                "$gte": lambda v, r: v is not None and v >= r,
                "$lte": lambda v, r: v is not None and v <= r,
                "$gt": lambda v, r: v is not None and v > r,
                "$lt": lambda v, r: v is not None and v < r,
            }.get(op)
            if ok is None or not ok(val, ref):
                return False
    return True


def build_where(
    filters: dict[str, Any] | None, role: str | None = None, group: str | None = None
) -> dict[str, Any] | None:
    """Translate FR-07 filters + role into a Chroma `where` clause."""
    clauses: list[dict[str, Any]] = []
    for key, val in (filters or {}).items():
        if val in (None, "", []):
            continue
        if key in _EQ_FILTERS:
            clauses.append(
                {key: {"$in": list(val)}}
                if isinstance(val, list | tuple)
                else {key: val}
            )
        elif key == "opened_from":
            clauses.append({"opened_ts": {"$gte": int(val)}})
        elif key == "opened_to":
            clauses.append({"opened_ts": {"$lte": int(val)}})
        elif key == "restricted" and val is False:
            clauses.append({"restricted": False})
        elif key == "where" and isinstance(val, dict):
            clauses.append(val)  # pre-built clause from the backend
    rw = role_where(role, group) if role is not None else None
    if rw:
        clauses.append(rw)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _why(
    query: str, hit: TicketHit, cls: ClassificationResult | None, verified: bool
) -> str:
    shared = [t for t in display_terms(hit.text, set(tokenize(query))) if len(t) > 2]
    parts = []
    if shared:
        parts.append("shared terms: " + ", ".join(shared[:4]))
    if cls is not None:
        if hit.metadata.get("subcategory") == cls.subcategory:
            parts.append("same subcategory")
        elif hit.metadata.get("category") == cls.category:
            parts.append("same category")
    if verified:
        parts.append("verified resolution")
    return "; ".join(parts) or f"semantic similarity {hit.score:.2f}"


def _to_incident(
    hit: TicketHit, tier: str, fused: float, why: str, excerpt_chars: int
) -> SimilarIncident:
    md = hit.metadata
    res = md.get("resolution")
    return SimilarIncident(
        incident_id=hit.ticket_id,
        similarity_score=round(float(hit.score), 2),
        resolution=res if res else None,
        tier=tier,
        excerpt=truncate(hit.text, excerpt_chars),
        why=why,
        resolution_origin=md.get("resolution_origin") or "none",
        category=md.get("category"),
        subcategory=md.get("subcategory"),
        collection=hit.collection,
        fused_score=round(fused, 4),
    )


def search_similar(
    text: str,
    store: TicketStore,
    classification: ClassificationResult | None = None,
    *,
    filters: dict[str, Any] | None = None,
    role: str | None = None,
    group: str | None = None,
    now_ts: int | None = None,
    reranker: Reranker | None = None,
    cfg: dict[str, Any] | None = None,
) -> TicketSearchResult:
    cfg = cfg if cfg is not None else ai_config()
    tcfg = cfg.get("retrieval", {}).get("tickets", {})
    dcfg = cfg.get("duplicates", {})
    k_dense = int(tcfg.get("k_dense", 30))
    k_bm25 = int(tcfg.get("k_bm25", 30))
    rrf_k = float(tcfg.get("rrf_k", 60))
    top_n = int(tcfg.get("top_n", 5))
    related_min = float(tcfg.get("related_min", 0.55))
    bonus = tcfg.get("bonus", {})
    excerpt_chars = int(tcfg.get("excerpt_chars", 160))
    near = float(dcfg.get("near", 0.92))
    similar_min = float(dcfg.get("similar", 0.75))

    where = build_where(filters, role, group)

    # 1. dense per collection (keep the best score per id) + BM25
    dense: dict[str, TicketHit] = {}
    for coll in COLLECTIONS:
        for h in store.query(text, k_dense, where=where, collections=(coll,)):
            if h.ticket_id not in dense or h.score > dense[h.ticket_id].score:
                dense[h.ticket_id] = h
    dense_rank = {
        tid: r for r, tid in enumerate(sorted(dense, key=lambda t: -dense[t].score), 1)
    }
    bm25 = store.bm25(text, k_bm25)
    bm25_rank = {tid: r for r, (tid, _s) in enumerate(bm25, 1)}

    # BM25-only ids need their records (and must still respect the filters)
    bm25_only_ids = [tid for tid in bm25_rank if tid not in dense]
    bm25_only: dict[str, TicketHit] = {}
    if bm25_only_ids:
        for h in store.get(bm25_only_ids):
            if matches_where(h.metadata, where):
                h.score = 0.0
                bm25_only[h.ticket_id] = h

    # 2. RRF + bonuses
    max_rrf = 2.0 / (rrf_k + 1)
    fused: dict[str, float] = {}
    for tid in set(dense) | set(bm25_only):
        r = 0.0
        if tid in dense_rank:
            r += 1.0 / (rrf_k + dense_rank[tid])
        if tid in bm25_rank:
            r += 1.0 / (rrf_k + bm25_rank[tid])
        s = r / max_rrf
        hit = dense.get(tid) or bm25_only[tid]
        md = hit.metadata
        if classification is not None:
            if md.get("category") == classification.category:
                s += float(bonus.get("same_category", 0.10))
            if md.get("subcategory") == classification.subcategory:
                s += float(bonus.get("same_subcategory", 0.05))
        if md.get("resolution_origin") in VERIFIED_ORIGINS:
            s += float(bonus.get("verified_resolution", 0.05))
        fused[tid] = s

    # 4/5. guard + tiers
    qhash = text_hash(text)
    qnorm = " ".join(text.lower().split())
    related: list[SimilarIncident] = []
    keyword_only: list[SimilarIncident] = []
    for tid in sorted(fused, key=lambda t: -fused[t]):
        hit = dense.get(tid) or bm25_only[tid]
        md = hit.metadata
        verified = md.get("resolution_origin") in VERIFIED_ORIGINS
        why = _why(text, hit, classification, verified)
        if tid in bm25_only:
            keyword_only.append(
                _to_incident(hit, "KEYWORD", fused[tid], why, excerpt_chars)
            )
            continue
        same_cat = (
            classification is None or md.get("category") == classification.category
        )
        if not same_cat and reranker is not None:
            same_cat = reranker(text, hit.text) >= float(tcfg.get("rerank_min", 0.5))
        exact = (
            _same_hash(qhash, md.get("text_hash"))
            or " ".join(hit.text.lower().split()) == qnorm
        )
        if exact:
            tier = "DUPLICATE"
        elif hit.score < related_min or not same_cat:
            if tid in bm25_rank:
                keyword_only.append(
                    _to_incident(hit, "KEYWORD", fused[tid], why, excerpt_chars)
                )
            continue
        elif (
            hit.score >= near
            and classification is not None
            and md.get("subcategory") == classification.subcategory
        ):
            tier = "LIKELY DUPLICATE"
        elif hit.score >= similar_min:
            tier = "SIMILAR"
        else:
            tier = "RELATED"
        related.append(_to_incident(hit, tier, fused[tid], why, excerpt_chars))

    top = sorted(related[:top_n], key=lambda s: (-s.similarity_score, -s.fused_score))

    # 6. live clustering
    possible_major = False
    if now_ts is not None:
        window = int(float(dcfg.get("live_window_h", 72)) * 3600)
        statuses = set(dcfg.get("cluster_statuses", ["new", "in_review"]))
        cmin = float(dcfg.get("cluster_min_cos", similar_min))
        live = [
            h
            for h in dense.values()
            if h.collection == "tickets_live"
            and h.score >= cmin
            and h.metadata.get("live_status", h.metadata.get("status")) in statuses
            and now_ts
            - int(h.metadata.get("created_ts") or h.metadata.get("opened_ts") or 0)
            <= window
        ]
        possible_major = len(live) >= int(dcfg.get("cluster_min", 3))

    # 7. pattern summary over the retrieved pool
    pattern: dict[str, Any] = {}
    if classification is not None:
        pool = [
            h
            for h in dense.values()
            if h.collection == "tickets_hist"
            and h.metadata.get("subcategory") == classification.subcategory
        ]
        terms: Counter[str] = Counter()
        for h in pool:
            terms.update(
                t for t in display_terms(h.text) if len(t) > 2 and not t.isdigit()
            )
        pattern = {
            "subcategory": classification.subcategory,
            "historical_count": len(pool),
            "top_terms": [
                t for t, _ in terms.most_common(int(tcfg.get("pattern_terms", 3)))
            ],
            "scope": "retrieved pool",
        }

    return TicketSearchResult(
        similar=top,
        keyword_only=keyword_only[:top_n],
        possible_major_incident=possible_major,
        pattern=pattern,
    )


def search_tickets(
    q: str,
    filters: dict[str, Any] | None,
    k: int,
    role: str | None,
    store: TicketStore,
    *,
    group: str | None = None,
) -> list[TicketHit]:
    """FR-07 search. Empty `q` → filter-only listing sorted by `opened_ts` desc."""
    where = build_where(filters, role, group)
    if not q.strip():
        hits = store.query("", max(k, 1) * 50, where=where)
        hits.sort(key=lambda h: -int(h.metadata.get("opened_ts") or 0))
        return hits[:k]
    hits = store.query(q, k, where=where)
    hits.sort(key=lambda h: -h.score)
    return hits[:k]
