"""Similar-ticket retrieval and duplicate detection (architecture §7.5, FR-02, FR-07). Owner: P2.

1. Dense search per collection (``tickets_hist``, ``tickets_verified``, ``tickets_live``) + BM25,
   fused with reciprocal-rank fusion (RRF, ``k=60``).
2. Bonuses: ``+0.10·same_category +0.05·same_subcategory +0.05·verified_resolution``.
3. FR-07 ``where`` filters (incl. ``restricted``, set by role).
4. Keyword-trap guard: a hit is "related" only if ``cosine ≥ 0.55`` and it is in the predicted
   category. BM25-only hits go to ``keyword_only``.
5. Tiers: DUPLICATE (same text hash) · LIKELY DUPLICATE (cos ≥ 0.92, same subcategory) ·
   SIMILAR (cos ≥ 0.75) · RELATED (passes the guard).
6. Live clustering → ``possible_major_incident``.  7. Pattern summary.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ai.config import ai_config
from shared.constants import CITABLE_ORIGINS, TICKET_COLLECTIONS
from shared.interfaces import TicketStore
from shared.schemas import ClassificationResult, TicketHit, TicketRecord

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9]")
_STOP_WORDS = (
    "a an and are as at be but by can cannot could did do does for from has have i in is it "
    "its my me of on or our so that the this to was we when with not no after again all also "
    "any been being every few get got had just keep keeps more since some still than then "
    "there they very what which while will would you your time times"
)
_STOP = frozenset(_STOP_WORDS.split())
_EQ_FIELDS = (
    "category",
    "subcategory",
    "priority",
    "status",
    "source",
    "assignment_group",
    "application",
    "environment",
    "restricted",
    "resolution_origin",
    "collection",
)
_LIVE_OPEN = frozenset({"new", "open", "in_review", "in review", "in_progress"})


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def text_hash(text: str) -> str:
    """Same normalisation as P1's ``text_hash``: lowercase, collapsed whitespace, sha1."""
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# --- filters (FR-07) ---------------------------------------------------------------------------
def build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Turn UI/API filters into a Chroma ``where``.

    Accepts plain fields (``category="Network"``, lists → ``$in``), ``date_from``/``date_to``
    (epoch seconds → ``opened_ts`` range) and passes through keys that are already operators.
    """
    if not filters:
        return None
    conds: list[dict[str, Any]] = []
    for key, val in filters.items():
        if val is None or val == "" or val == []:
            continue
        if key.startswith("$"):
            conds.append({key: val})
        elif key == "date_from":
            conds.append({"opened_ts": {"$gte": int(val)}})
        elif key == "date_to":
            conds.append({"opened_ts": {"$lte": int(val)}})
        elif isinstance(val, dict):
            conds.append({key: val})
        elif isinstance(val, (list, tuple, set)):
            conds.append({key: {"$in": list(val)}})
        elif key in _EQ_FIELDS or key in ("opened_ts", "ticket_id"):
            conds.append({key: {"$eq": val}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def matches_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a Chroma-style ``where`` locally (used for BM25 hits and record lists)."""
    if not where:
        return True
    ops = {
        "$eq": lambda v, r: v == r,
        "$ne": lambda v, r: v != r,
        "$in": lambda v, r: v in r,
        "$nin": lambda v, r: v not in r,
        "$gte": lambda v, r: v is not None and v >= r,
        "$lte": lambda v, r: v is not None and v <= r,
        "$gt": lambda v, r: v is not None and v > r,
        "$lt": lambda v, r: v is not None and v < r,
    }
    for key, cond in where.items():
        if key == "$and":
            if not all(matches_where(meta, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(matches_where(meta, c) for c in cond):
                return False
            continue
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, ref in cond.items():
            if op in ops and not ops[op](meta.get(key), ref):
                return False
    return True


def role_where(filters: dict[str, Any] | None, role: str | None) -> dict[str, Any]:
    """L1 users never see restricted tickets (architecture §8.6), whatever the filters say."""
    out = dict(filters or {})
    if role == "l1":
        out["restricted"] = False
    return out


# --- helpers -----------------------------------------------------------------------------------
def _hit_text(h: TicketHit) -> str:
    return " ".join(p for p in (h.title, h.excerpt) if p)


def _record_to_hit(rec: TicketRecord, score: float) -> TicketHit:
    return TicketHit(
        ticket_id=rec.ticket_id,
        score=score,
        title=rec.title,
        excerpt=rec.description[:200],
        category=rec.category,
        subcategory=rec.subcategory,
        priority=rec.priority,
        status=rec.status,
        resolution=rec.resolution,
        resolution_origin=rec.resolution_origin,
        collection=rec.collection,
        metadata={
            "text_hash": rec.text_hash,
            "opened_ts": rec.opened_ts,
            "restricted": rec.restricted,
            "assignment_group": rec.assignment_group,
            "application": rec.application,
            "environment": rec.environment,
        },
    )


def _hit_meta(h: TicketHit) -> dict[str, Any]:
    meta = {
        "category": h.category,
        "subcategory": h.subcategory,
        "priority": h.priority,
        "status": h.status,
        "resolution_origin": h.resolution_origin,
        "collection": h.collection,
        "ticket_id": h.ticket_id,
    }
    meta.update({k: v for k, v in (h.metadata or {}).items() if k not in meta or v is not None})
    meta.setdefault("restricted", False)
    if meta.get("restricted") is None:
        meta["restricted"] = False
    return meta


def shared_terms(query: str, h: TicketHit, limit: int = 3) -> list[str]:
    q = tokens(query)
    doc = set(tokens(_hit_text(h)))
    out: list[str] = []
    for t in q:
        if t in doc and t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _why(query: str, h: TicketHit, pred: ClassificationResult | None, tier: str | None) -> str:
    parts: list[str] = []
    if tier == "DUPLICATE":
        parts.append("identical ticket text")
    if pred is not None:
        if h.subcategory == pred.subcategory and h.category == pred.category:
            parts.append("same subcategory")
        elif h.category == pred.category:
            parts.append("same category")
    terms = shared_terms(query, h)
    if terms:
        parts.append("shared terms: " + ", ".join(terms))
    if h.resolution_origin in CITABLE_ORIGINS:
        parts.append("verified resolution")
    if h.keyword_only:
        parts.append("keyword match only")
    return "; ".join(parts) or "semantic similarity"


def assign_tier(
    h: TicketHit, cos: float, pred: ClassificationResult | None, qhash: str | None, cfg
) -> str:
    dup = cfg["duplicates"]
    if qhash and (h.metadata or {}).get("text_hash") == qhash:
        return "DUPLICATE"
    same_sub = pred is not None and h.subcategory == pred.subcategory
    if cos >= float(dup["near"]) and same_sub:
        return "LIKELY DUPLICATE"
    if cos >= float(dup["similar"]):
        return "SIMILAR"
    return "RELATED"


# --- core retrieval ----------------------------------------------------------------------------
@dataclass
class _Cand:
    hit: TicketHit
    cos: float | None = None  # dense similarity, None for BM25-only
    rrf: float = 0.0


def _retrieve(
    text: str, ticket_store: TicketStore, where: dict[str, Any] | None, cfg: dict[str, Any]
) -> dict[str, _Cand]:
    tcfg = cfg["retrieval"]["tickets"]
    rrf_k = float(tcfg["rrf_k"])
    cands: dict[str, _Cand] = {}

    dense: list[TicketHit] = []
    for coll in TICKET_COLLECTIONS:
        try:
            dense += ticket_store.query(text, int(tcfg["k_dense"]), where, collections=(coll,))
        except Exception:  # noqa: BLE001 - one collection failing must not stop the search
            log.warning("dense ticket query failed for %s", coll)
    best: dict[str, TicketHit] = {}
    for h in dense:
        if h.ticket_id not in best or h.score > best[h.ticket_id].score:
            best[h.ticket_id] = h
    for rank, h in enumerate(sorted(best.values(), key=lambda h: (-h.score, h.ticket_id)), 1):
        cands[h.ticket_id] = _Cand(hit=h, cos=max(0.0, min(1.0, h.score)), rrf=1 / (rrf_k + rank))

    try:
        bm = ticket_store.bm25(text, int(tcfg["k_bm25"]))
    except Exception:  # noqa: BLE001
        log.warning("ticket BM25 failed")
        bm = []
    missing = [tid for tid, _ in bm if tid not in cands]
    records = {}
    if missing:
        try:
            records = {r.ticket_id: r for r in ticket_store.get(missing)}
        except Exception:  # noqa: BLE001
            records = {}
    for rank, (tid, _score) in enumerate(bm, 1):
        if tid in cands:
            cands[tid].rrf += 1 / (rrf_k + rank)
            continue
        rec = records.get(tid)
        if rec is None:
            continue
        hit = _record_to_hit(rec, 0.0)
        if not matches_where(_hit_meta(hit), where):
            continue
        cands[tid] = _Cand(hit=hit, cos=None, rrf=1 / (rrf_k + rank))
    return cands


@dataclass
class TicketSearchResult:
    similar: list[TicketHit] = field(default_factory=list)
    keyword_only: list[TicketHit] = field(default_factory=list)
    possible_major_incident: bool = False
    pattern: dict[str, Any] = field(default_factory=dict)


def _rank_score(c: _Cand, pred: ClassificationResult | None, cfg: dict[str, Any]) -> float:
    rrf_k = float(cfg["retrieval"]["tickets"]["rrf_k"])
    bonus = cfg["retrieval"]["ticket_bonus"]
    score = c.rrf / (2 / (rrf_k + 1))  # normalise RRF to [0, 1]
    h = c.hit
    if pred is not None and h.category == pred.category:
        score += float(bonus["same_category"])
        if h.subcategory == pred.subcategory:
            score += float(bonus["same_subcategory"])
    if h.resolution_origin in CITABLE_ORIGINS:
        score += float(bonus["verified_resolution"])
    return score


def _live_cluster(
    text: str, ticket_store: TicketStore, cfg: dict[str, Any], now: float | None
) -> bool:
    dup = cfg["duplicates"]
    now = time.time() if now is None else now
    window = float(dup["live_window_h"]) * 3600
    try:
        live = ticket_store.query(
            text, int(cfg["retrieval"]["tickets"]["k_dense"]), None, collections=("tickets_live",)
        )
    except Exception:  # noqa: BLE001
        return False
    near = 0
    for h in live:
        ts = (h.metadata or {}).get("opened_ts")
        status = (h.status or "").lower()
        if h.score < float(dup["similar"]) or status not in _LIVE_OPEN:
            continue
        if ts is not None and now - float(ts) > window:
            continue
        near += 1
    return near >= int(dup["cluster_min"])


def _pattern(
    text: str, pred: ClassificationResult | None, ticket_store: TicketStore
) -> dict[str, Any]:
    if pred is None:
        return {}
    try:
        hist = ticket_store.query(
            text, 100, {"subcategory": {"$eq": pred.subcategory}}, collections=("tickets_hist",)
        )
    except Exception:  # noqa: BLE001
        return {}
    counts = Counter(t for h in hist for t in set(tokens(_hit_text(h))))
    return {
        "subcategory": pred.subcategory,
        "historical_count": len(hist),
        "top_terms": [t for t, _ in counts.most_common(3)],
    }


def find_similar(
    text: str,
    classification: ClassificationResult | None,
    ticket_store: TicketStore,
    *,
    filters: dict[str, Any] | None = None,
    role: str | None = None,
    query_hash: str | None = None,
    exclude_ids: tuple[str, ...] = (),
    now: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> TicketSearchResult:
    """Similar incidents for one issue: guarded, tiered and explained (``similar_incidents``)."""
    cfg = cfg or ai_config()
    tcfg = cfg["retrieval"]["tickets"]
    where = build_where(role_where(filters, role))
    qhash = query_hash or text_hash(text)
    cands = _retrieve(text, ticket_store, where, cfg)
    for tid in exclude_ids:
        cands.pop(tid, None)

    ranked = sorted(
        cands.values(), key=lambda c: (-_rank_score(c, classification, cfg), c.hit.ticket_id)
    )
    similar: list[TicketHit] = []
    keyword_only: list[TicketHit] = []
    related_min = float(tcfg["related_min"])
    for c in ranked:
        h = c.hit
        if c.cos is None:
            kw = h.model_copy(update={"keyword_only": True, "score": 0.0, "tier": None})
            kw.why = _why(text, kw, classification, None)
            keyword_only.append(kw)
            continue
        same_cat = classification is None or h.category == classification.category
        is_dup = qhash and (h.metadata or {}).get("text_hash") == qhash
        if not is_dup and (c.cos < related_min or not same_cat):
            continue  # keyword-trap guard
        tier = assign_tier(h, c.cos, classification, qhash, cfg)
        similar.append(
            h.model_copy(
                update={
                    "score": round(c.cos, 2),
                    "tier": tier,
                    "excerpt": h.excerpt or (h.title or "")[:200],
                    "why": _why(text, h, classification, tier),
                    "keyword_only": False,
                }
            )
        )
    top_n = int(tcfg["top_n"])
    similar = sorted(similar[: max(top_n * 2, top_n)], key=lambda h: -h.score)[:top_n]
    return TicketSearchResult(
        similar=similar,
        keyword_only=keyword_only[:top_n],
        possible_major_incident=_live_cluster(text, ticket_store, cfg, now),
        pattern=_pattern(text, classification, ticket_store),
    )


def search_tickets(
    q: str,
    filters: dict[str, Any] | None,
    k: int,
    role: str | None,
    ticket_store: TicketStore,
    *,
    cfg: dict[str, Any] | None = None,
) -> list[TicketHit]:
    """FR-07 search used by ``/tickets/search``.

    With a query: hybrid search, dense hits first (with ``why``), then keyword-only hits flagged
    ``keyword_only=True``. With an empty query: a filter-only listing sorted by ``opened_ts`` desc
    (needs the store's optional ``list(where, k)`` method; returns ``[]`` if it has none).
    """
    cfg = cfg or ai_config()
    where = build_where(role_where(filters, role))
    if not q.strip():
        lister = getattr(ticket_store, "list", None)
        if lister is None:
            return []
        recs: list[TicketRecord] = list(lister(where, k))
        recs.sort(key=lambda r: (-(r.opened_ts or 0), r.ticket_id))
        return [_record_to_hit(r, 0.0) for r in recs[:k]]

    cands = _retrieve(q, ticket_store, where, cfg)
    ranked = sorted(cands.values(), key=lambda c: (-_rank_score(c, None, cfg), c.hit.ticket_id))
    dense, kw = [], []
    for c in ranked:
        if c.cos is None:
            h = c.hit.model_copy(update={"keyword_only": True, "score": 0.0})
            h.why = _why(q, h, None, None)
            kw.append(h)
        else:
            h = c.hit.model_copy(update={"score": round(c.cos, 2)})
            h.why = _why(q, h, None, None)
            dense.append(h)
    return (dense + kw)[:k]
