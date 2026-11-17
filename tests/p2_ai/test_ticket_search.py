"""P2-T5: hybrid ticket search, guard, tiers, filters, clustering (architecture §7.5)."""

from __future__ import annotations

from typing import Any

from ai.ticket_search import (
    build_where,
    find_similar,
    matches_where,
    search_tickets,
    text_hash,
)
from shared.interfaces import DEFAULT_TICKET_COLLECTIONS
from shared.schemas import ClassificationResult, TicketHit, TicketRecord
from shared.stubs import FakeTicketStore

NOW = 1_800_000_000.0


class ScoredStore(FakeTicketStore):
    """FakeTicketStore whose dense scores are preset cosines (the real store returns cosine)."""

    def __init__(
        self,
        cos: dict[str, float],
        bm25_ids: list[str] | None = None,
        extra: list[TicketRecord] | None = None,
    ) -> None:
        super().__init__()
        for rec in extra or []:
            self._recs[rec.ticket_id] = rec
        self.cos = cos
        self.bm25_ids = bm25_ids or []
        self.wheres: list[Any] = []

    def query(
        self,
        text: str,
        k: int,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = DEFAULT_TICKET_COLLECTIONS,
    ) -> list[TicketHit]:
        self.wheres.append(where)
        out = []
        for tid, score in self.cos.items():
            rec = self._recs.get(tid)
            if rec is None or rec.collection not in collections:
                continue
            if not matches_where(rec.model_dump(), where):
                continue
            hit = self._hit(rec, score)
            hit.metadata = {"text_hash": rec.text_hash, "opened_ts": rec.opened_ts}
            out.append(hit)
        out.sort(key=lambda h: -h.score)
        return out[:k]

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return [(tid, 10.0 - i) for i, tid in enumerate(self.bm25_ids[:k])]


VPN = ClassificationResult(
    category="Network", subcategory="VPN", confidence=0.9, band="high", intent="incident"
)
QUERY = "VPN gateway not responding when working from home"


def test_hits_carry_tier_excerpt_why_origin() -> None:
    store = ScoredStore({"INC-H0001": 0.93, "TKT-0001": 0.80, "TKT-0002": 0.60})
    res = find_similar(QUERY, VPN, store, now=NOW)
    assert [h.ticket_id for h in res.similar] == ["INC-H0001", "TKT-0001", "TKT-0002"]
    assert [h.tier for h in res.similar] == ["LIKELY DUPLICATE", "SIMILAR", "RELATED"]
    for h in res.similar:
        assert h.excerpt and h.why and h.resolution_origin != "none"
    assert "same subcategory" in res.similar[0].why
    assert "verified resolution" in res.similar[0].why
    assert "shared terms: vpn, gateway" in res.similar[0].why
    assert res.similar[0].score == 0.93


def test_keyword_trap_guard_and_keyword_only() -> None:
    # TKT-0013 is a SQL ticket with high cosine but the wrong category → dropped.
    # TKT-0029 is BM25-only → keyword_only. TKT-0003 below related_min → dropped.
    store = ScoredStore(
        {"TKT-0001": 0.82, "TKT-0013": 0.90, "TKT-0003": 0.50},
        bm25_ids=["TKT-0001", "TKT-0029"],
    )
    res = find_similar(QUERY, VPN, store, now=NOW)
    assert [h.ticket_id for h in res.similar] == ["TKT-0001"]
    assert [h.ticket_id for h in res.keyword_only] == ["TKT-0029"]
    kw = res.keyword_only[0]
    assert kw.keyword_only is True and "keyword match only" in kw.why


def test_duplicate_tier_by_text_hash() -> None:
    dup = TicketRecord(
        ticket_id="TKT-9000",
        title=QUERY,
        description=QUERY,
        category="Network",
        subcategory="VPN",
        resolution_origin="synthetic_kb_derived",
        text_hash=text_hash(QUERY),
    )
    store = ScoredStore({"TKT-9000": 0.99}, extra=[dup])
    res = find_similar(QUERY, VPN, store, now=NOW)
    assert res.similar[0].tier == "DUPLICATE"
    assert "identical ticket text" in res.similar[0].why


def test_top_n_and_sorted_by_similarity() -> None:
    ids = ["INC-H0001", "INC-H0002", "TKT-0001", "TKT-0002"]
    store = ScoredStore({tid: 0.6 + i * 0.05 for i, tid in enumerate(ids)})
    res = find_similar(QUERY, VPN, store, now=NOW, cfg=None)
    scores = [h.score for h in res.similar]
    assert scores == sorted(scores, reverse=True)
    assert len(res.similar) <= 5


def test_l1_role_filters_restricted_tickets() -> None:
    secret = TicketRecord(
        ticket_id="TKT-SECRET",
        title="VPN gateway for the board",
        description="VPN gateway not responding for the board",
        category="Network",
        subcategory="VPN",
        restricted=True,
    )
    store = ScoredStore({"TKT-SECRET": 0.95, "TKT-0001": 0.8}, ["TKT-SECRET"], extra=[secret])
    l1 = find_similar(QUERY, VPN, store, role="l1", now=NOW)
    l2 = find_similar(QUERY, VPN, store, role="l2", now=NOW)
    assert "TKT-SECRET" not in {h.ticket_id for h in l1.similar + l1.keyword_only}
    assert "TKT-SECRET" in {h.ticket_id for h in l2.similar}


def test_fr07_filters_build_chroma_where() -> None:
    where = build_where(
        {
            "category": "Network",
            "priority": ["P1", "P2"],
            "environment": "",
            "date_from": 100,
            "date_to": 200,
        }
    )
    assert where == {
        "$and": [
            {"category": {"$eq": "Network"}},
            {"priority": {"$in": ["P1", "P2"]}},
            {"opened_ts": {"$gte": 100}},
            {"opened_ts": {"$lte": 200}},
        ]
    }
    assert build_where({}) is None
    assert matches_where({"category": "Network", "opened_ts": 150, "priority": "P2"}, where)
    assert not matches_where({"category": "Network", "opened_ts": 250, "priority": "P2"}, where)


def test_filters_reach_the_store() -> None:
    store = ScoredStore({"TKT-0001": 0.8, "TKT-0003": 0.8})
    find_similar(QUERY, VPN, store, filters={"subcategory": "VPN"}, now=NOW)
    assert all(w == {"subcategory": {"$eq": "VPN"}} for w in store.wheres[:3])


def _live(i: int, ts: float, status: str = "new") -> TicketRecord:
    return TicketRecord(
        ticket_id=f"LIVE-{i}",
        title="VPN gateway down",
        description="VPN gateway not responding",
        category="Network",
        subcategory="VPN",
        status=status,
        collection="tickets_live",
        opened_ts=int(ts),
    )


def test_live_cluster_sets_possible_major_incident() -> None:
    lives = [_live(i, NOW - 3600 * i) for i in range(1, 4)]
    store = ScoredStore({r.ticket_id: 0.9 for r in lives}, extra=lives)
    assert find_similar(QUERY, VPN, store, now=NOW).possible_major_incident is True


def test_old_or_closed_live_tickets_do_not_cluster() -> None:
    lives = [_live(1, NOW - 3600), _live(2, NOW - 100 * 3600), _live(3, NOW, status="closed")]
    store = ScoredStore({r.ticket_id: 0.9 for r in lives}, extra=lives)
    assert find_similar(QUERY, VPN, store, now=NOW).possible_major_incident is False


def test_pattern_summary() -> None:
    res = find_similar(QUERY, VPN, FakeTicketStore(), now=NOW)
    assert res.pattern["subcategory"] == "VPN"
    assert res.pattern["historical_count"] >= 1
    assert "vpn" in res.pattern["top_terms"]


def test_search_tickets_query_and_role() -> None:
    store = FakeTicketStore()
    hits = search_tickets("VPN gateway", {"category": "Network"}, 10, "l2", store)
    assert hits and all(h.category == "Network" for h in hits)
    assert all(h.why for h in hits)


def test_search_tickets_empty_query_lists_by_date() -> None:
    class Listing(FakeTicketStore):
        def list(self, where, k):
            return [r for r in self._recs.values() if matches_where(r.model_dump(), where)][:k]

    hits = search_tickets("", {"subcategory": "VPN"}, 10, "l1", Listing())
    assert hits and all(h.subcategory == "VPN" for h in hits)
    assert search_tickets("", {}, 10, "l1", FakeTicketStore()) == []
