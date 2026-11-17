"""P2-T5: hybrid ticket search, guard, tiers, filters, live clustering, FR-07 search."""

from __future__ import annotations

import pytest

from ai.ticket_search import (
    build_where,
    matches_where,
    role_where,
    search_similar,
    search_tickets,
)
from ai.types import ClassificationResult
from tests.p2_ai.fakes import FakeTicketStore, load_fixture

DEMOS = {d["id"]: d for d in load_fixture("demo_tickets.json")}
NOW = 1767225600 + 200 * 86400


def C(cat: str, sub: str) -> ClassificationResult:
    return ClassificationResult(cat, sub, 0.9, "high")


@pytest.fixture
def store():
    return FakeTicketStore()


def _text(demo_id):
    d = DEMOS[demo_id]
    return f"{d['title']}. {d['description']}"


def test_every_hit_has_required_fields(store):
    res = search_similar(_text("DEMO-001"), store, C("Network", "VPN"))
    assert 1 <= len(res.similar) <= 5
    for s in res.similar + res.keyword_only:
        assert s.tier and s.excerpt and s.why and s.resolution_origin
    scores = [s.similarity_score for s in res.similar]
    assert scores == sorted(scores, reverse=True)
    assert all(s.category == "Network" for s in res.similar)


def test_similar_sorted_and_rounded(store):
    res = search_similar(_text("DEMO-007"), store, C("Database", "MS SQL Server"))
    for s in res.similar:
        assert s.similarity_score == round(s.similarity_score, 2)
    assert res.similar[0].subcategory == "MS SQL Server"


def test_keyword_trap_guard_separates_keyword_only(store):
    # "Windows update" shares words with an OS ticket, but the predicted category is Hardware
    res = search_similar(_text("DEMO-003"), store, C("Hardware", "Monitor"))
    assert all(s.category == "Hardware" for s in res.similar)
    kw_ids = {s.incident_id for s in res.keyword_only}
    assert "TKT-0042" in kw_ids  # "Windows update failed to install" (Software/OS)
    assert all(s.tier == "KEYWORD" for s in res.keyword_only)


def test_duplicate_tier_on_identical_text(store):
    res = search_similar("VPN does not connect from home", store, C("Network", "VPN"))
    dups = [s for s in res.similar if s.tier == "DUPLICATE"]
    assert {d.incident_id for d in dups} == {"TKT-0001", "TKT-0051"}


def test_similar_and_related_tiers(store):
    res = search_similar(_text("DEMO-003"), store, C("Hardware", "Monitor"))
    tiers = {s.incident_id: s.tier for s in res.similar}
    assert tiers["INC-H0003"] == "SIMILAR"
    assert "RELATED" in tiers.values()


def test_likely_duplicate_tier():
    recs = load_fixture("mini_tickets.json")
    store = FakeTicketStore(recs)
    text = "External monitor goes black after a Windows update today"
    res = search_similar(text, store, C("Hardware", "Monitor"))
    assert res.similar[0].incident_id == "INC-H0003"
    assert res.similar[0].tier == "LIKELY DUPLICATE"


def test_verified_resolution_origin_is_reported(store):
    res = search_similar(_text("DEMO-003"), store, C("Hardware", "Monitor"))
    top = res.similar[0]
    assert (
        top.incident_id == "INC-H0003" and top.resolution_origin == "curated_verified"
    )
    assert "verified resolution" in top.why
    assert any(s.resolution_origin == "synthetic_kb_derived" for s in res.similar)


def test_why_mentions_shared_terms_unstemmed(store):
    res = search_similar(_text("DEMO-006"), store, C("Network", "Wireless"))
    assert any("wireless" in s.why for s in res.similar)
    assert not any("wireles," in s.why for s in res.similar)


def test_restricted_hidden_for_l1_visible_for_admin(store):
    text = "Antivirus quarantined a file I need"
    l1 = search_similar(text, store, C("Inquiry/Help", "Antivirus"), role="l1")
    admin = search_similar(text, store, C("Inquiry/Help", "Antivirus"), role="admin")
    ids_l1 = {s.incident_id for s in l1.similar + l1.keyword_only}
    ids_admin = {s.incident_id for s in admin.similar}
    assert "TKT-0045" not in ids_l1
    assert "TKT-0045" in ids_admin


def test_restricted_visible_for_own_group_l2(store):
    text = "Antivirus quarantined a file I need"
    own = search_similar(
        text, store, C("Inquiry/Help", "Antivirus"), role="l2", group="SERVICE-DESK"
    )
    other = search_similar(
        text, store, C("Inquiry/Help", "Antivirus"), role="l2", group="NET-OPS"
    )
    assert "TKT-0045" in {s.incident_id for s in own.similar}
    assert "TKT-0045" not in {s.incident_id for s in other.similar + other.keyword_only}


def test_fr07_filters_passed_to_store(store):
    search_similar(
        _text("DEMO-001"),
        store,
        C("Network", "VPN"),
        filters={
            "category": "Network",
            "opened_from": 1767225600,
            "assignment_group": "NET-OPS",
        },
        role="l1",
    )
    where = store.where_log[-1]
    assert {"category": "Network"} in where["$and"]
    assert {"opened_ts": {"$gte": 1767225600}} in where["$and"]
    assert {"restricted": False} in where["$and"]


def test_build_where_variants():
    assert build_where(None) is None
    assert build_where({"status": ["closed", "new"]}) == {
        "status": {"$in": ["closed", "new"]}
    }
    assert role_where("admin") is None
    assert role_where("kb_owner") == {"restricted": False}
    assert role_where(None) == {"restricted": False}


def test_matches_where():
    meta = {"category": "Network", "opened_ts": 10, "restricted": False}
    assert matches_where(
        meta, {"$and": [{"category": "Network"}, {"opened_ts": {"$gte": 5}}]}
    )
    assert not matches_where(meta, {"opened_ts": {"$lte": 5}})
    assert matches_where(meta, {"$or": [{"restricted": True}, {"category": "Network"}]})


def _live(
    i: int, status: str, ts: int, text: str = "VPN gateway not responding for everyone"
):
    return {
        "ticket_id": f"LIVE-{i}",
        "title": text,
        "description": text,
        "category": "Network",
        "subcategory": "VPN",
        "status": status,
        "live_status": status,
        "created_ts": ts,
        "opened_ts": ts,
        "restricted": False,
        "resolution": None,
        "resolution_origin": "none",
        "collection": "tickets_live",
        "text_hash": f"live{i}",
    }


def test_live_cluster_sets_possible_major_incident():
    recs = load_fixture("mini_tickets.json") + [
        _live(i, "new", NOW - 3600 * i) for i in range(1, 4)
    ]
    res = search_similar(
        "VPN gateway not responding for everyone",
        FakeTicketStore(recs),
        C("Network", "VPN"),
        now_ts=NOW,
    )
    assert res.possible_major_incident


def test_live_cluster_ignores_old_or_closed():
    recs = load_fixture("mini_tickets.json") + [
        _live(1, "new", NOW - 3600),
        _live(2, "closed", NOW - 3600),
        _live(3, "new", NOW - 100 * 3600),
    ]
    res = search_similar(
        "VPN gateway not responding for everyone",
        FakeTicketStore(recs),
        C("Network", "VPN"),
        now_ts=NOW,
    )
    assert not res.possible_major_incident


def test_pattern_summary(store):
    res = search_similar(_text("DEMO-001"), store, C("Network", "VPN"))
    assert res.pattern["subcategory"] == "VPN"
    assert res.pattern["historical_count"] >= 3
    assert "vpn" in res.pattern["top_terms"]


def test_reranker_can_admit_cross_category_hit(store):
    text = "Windows update failed and now my screen is black"
    without = search_similar(text, store, C("Hardware", "Monitor"))
    with_rr = search_similar(
        text, store, C("Hardware", "Monitor"), reranker=lambda q, d: 0.9
    )
    assert len(with_rr.similar) >= len(without.similar)
    assert any(s.category == "Software" for s in with_rr.similar)


def test_search_tickets_filter_only_sorted_by_date(store):
    hits = search_tickets("", {"category": "Network"}, 5, "l1", store)
    assert len(hits) == 5
    ts = [h.metadata["opened_ts"] for h in hits]
    assert ts == sorted(ts, reverse=True)
    assert all(h.metadata["category"] == "Network" for h in hits)


def test_search_tickets_query(store):
    hits = search_tickets("outlook crashes", {}, 3, "l1", store)
    assert hits[0].metadata["subcategory"] == "Email"


def test_search_tickets_hides_restricted_for_l1(store):
    hits = search_tickets("", {"subcategory": "Antivirus"}, 10, "l1", store)
    assert "TKT-0045" not in {h.ticket_id for h in hits}
    hits_admin = search_tickets("", {"subcategory": "Antivirus"}, 10, "admin", store)
    assert "TKT-0045" in {h.ticket_id for h in hits_admin}
