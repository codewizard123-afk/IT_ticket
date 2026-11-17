"""P3-T9: review queue, ticket search/detail, escalation bundles, /health and the OpenAPI surface.

Done when: the OpenAPI docs list every endpoint in architecture §10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shared.schemas import TicketRecord
from shared.stubs import FakeTicketStore
from tests.p3_backend.conftest import DEMO_IDS, make_client, make_services, role

ARCH_10 = [
    ("get", "/health"),
    ("post", "/auth/login"),
    ("post", "/analyze"),
    ("post", "/tickets/{ticket_id}/clarify"),
    ("get", "/tickets/search"),
    ("get", "/tickets/{ticket_id}"),
    ("get", "/kb/search"),
    ("get", "/kb/{kb_id}"),
    ("post", "/feedback"),
    ("post", "/decisions"),
    ("post", "/events"),
    ("get", "/review-queue"),
    ("get", "/escalations/{bundle_id}"),
    ("get", "/audit/{analysis_id}"),
    ("get", "/audit/export"),
    ("get", "/audit/verify"),
    ("get", "/metrics/mttr"),
    ("get", "/metrics/summary"),
    ("post", "/kb/drafts"),
    ("post", "/kb/drafts/{draft_id}/approve"),
    ("get", "/trial/tickets"),
]


def test_openapi_lists_every_architecture_endpoint(client: Any) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    missing = [(m, p) for m, p in ARCH_10 if m not in paths.get(f"/api/v1{p}", {})]
    assert missing == []
    assert client.get("/docs").status_code == 200


def test_health_needs_no_auth(client: Any) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["index_stale"] is False
    assert body["llm_available"] is False and body["stores"] == "stub"
    assert body["versions"]["pipeline"] == "2.0.0"


def test_health_degraded_without_stores(tmp_path: Path) -> None:
    services = make_services(tmp_path, ticket_store=None, index_error="run make data")
    services.settings = services.settings.__class__(
        **{**services.settings.__dict__, "use_stub_stores": False}
    )
    with make_client(services) as c:
        body = c.get("/api/v1/health").json()
    assert body["status"] == "degraded" and body["index_stale"] is True


# --- ticket search ------------------------------------------------------------------------------
def _search(c: Any, headers: dict[str, str] | None = None, **params: Any) -> dict[str, Any]:
    r = c.get("/api/v1/tickets/search", params=params, headers=headers or role("l1"))
    assert r.status_code == 200, r.text
    return r.json()


def test_search_with_filters_and_paging(client: Any) -> None:
    body = _search(client, q="VPN gateway not responding", size=2)
    assert body["total"] >= 3 and len(body["items"]) == 2 and body["page"] == 1
    page2 = _search(client, q="VPN gateway not responding", size=2, page=2)
    assert {i["ticket_id"] for i in page2["items"]}.isdisjoint(
        i["ticket_id"] for i in body["items"]
    )
    net = _search(client, q="not working", category="Network")
    assert net["items"] and all(i["category"] == "Network" for i in net["items"])
    high = _search(client, q="slow", priority="High")
    assert high["items"] and all(i["priority"] == "P2" for i in high["items"])


def test_search_hides_restricted_tickets_from_l1(client: Any) -> None:
    q = "Antivirus threat detected popup"
    assert "TKT-0034" not in {i["ticket_id"] for i in _search(client, q=q)["items"]}
    assert "TKT-0034" in {i["ticket_id"] for i in _search(client, role("admin"), q=q)["items"]}
    sd = role("l2", "SERVICE-DESK")
    assert "TKT-0034" in {i["ticket_id"] for i in _search(client, sd, q=q)["items"]}
    other = role("l2", "DBA")
    assert "TKT-0034" not in {i["ticket_id"] for i in _search(client, other, q=q)["items"]}


class ListingStore(FakeTicketStore):
    """A fake with the Chroma store's ``list(where=, collections=, limit=)`` signature."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def list(self, where=None, collections=None, limit=None) -> list[TicketRecord]:  # noqa: ANN001
        from shared.stubs import _matches_where

        self.calls.append({"where": where, "collections": collections, "limit": limit})
        recs = [r for r in self.get([f"TKT-{i:04d}" for i in range(1, 60)])]
        recs = [r for r in recs if _matches_where(r.model_dump(), where)]
        return sorted(recs, key=lambda r: -(r.opened_ts or 0))[:limit]


def test_filter_only_listing_uses_keyword_args(tmp_path: Path) -> None:
    store = ListingStore()
    services = make_services(tmp_path, ticket_store=store)
    with make_client(services) as c:
        body = _search(c, category="Database", date_from="2026-01-01")
    cutoff = 1767225600  # 2026-01-01T00:00:00Z
    expected = [
        r for r in store.get([f"TKT-{i:04d}" for i in range(1, 60)])
        if r.category == "Database" and (r.opened_ts or 0) >= cutoff and not r.restricted
    ]  # fmt: skip
    assert body["total"] == len(expected) > 0
    assert all(i["category"] == "Database" for i in body["items"])
    assert store.calls[0]["collections"] is None and store.calls[0]["limit"] == 1000


def test_search_bad_date_is_400(client: Any) -> None:
    r = client.get("/api/v1/tickets/search", params={"date_from": "yesterday"}, headers=role("l1"))
    assert r.status_code == 400


def test_ticket_detail(client: Any, demo_tickets: dict[str, Any]) -> None:
    assert (
        client.get("/api/v1/tickets/TKT-0001", headers=role("l1")).json()["category"] == "Network"
    )
    assert client.get("/api/v1/tickets/TKT-0034", headers=role("l1")).status_code == 403
    assert client.get("/api/v1/tickets/TKT-0034", headers=role("admin")).status_code == 200
    assert client.get("/api/v1/tickets/NOPE", headers=role("l1")).status_code == 404
    client.post(
        "/api/v1/analyze",
        json={"title": "Printer jammed", "description": "Printer on 3rd floor jammed"},
        headers=role("l1"),
    )
    live = client.get("/api/v1/tickets/search", params={"q": "printer jammed"}, headers=role("l1"))
    assert live.status_code == 200


# --- review queue + bundles ---------------------------------------------------------------------
@pytest.fixture
def analysed(client: Any, demo_tickets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        tid: client.post("/api/v1/analyze", json=demo_tickets[tid], headers=role("l1")).json()
        for tid in DEMO_IDS
    }


def test_review_queue(client: Any, analysed: dict[str, dict[str, Any]]) -> None:
    q = client.get("/api/v1/review-queue", headers=role("l1")).json()
    ids = [i["ticket_id"] for i in q["items"]]
    assert "DEMO-003" not in ids  # recommend + no approval needed
    assert set(ids) == set(DEMO_IDS) - {"DEMO-003"} and q["total"] == 6
    assert ids[0] == "DEMO-002"  # Critical first
    esc = client.get(
        "/api/v1/review-queue", params={"status": "escalate"}, headers=role("l1")
    ).json()
    assert {i["ticket_id"] for i in esc["items"]} == {"DEMO-002", "DEMO-005", "DEMO-006"}
    aid = analysed["DEMO-001"]["details"]["analysis_id"]
    client.post(
        "/api/v1/decisions",
        json={"analysis_id": aid, "final_decision": "resolved"},
        headers=role("l1"),
    )
    open_ids = {
        i["ticket_id"]
        for i in client.get("/api/v1/review-queue", headers=role("l1")).json()["items"]
    }
    assert "DEMO-001" not in open_ids
    # DBA sees its group only; kb_owner sees nothing
    dba = client.get("/api/v1/review-queue", headers=role("l2", "DBA")).json()
    assert {i["ticket_id"] for i in dba["items"]} == {"DEMO-002", "DEMO-007"}
    assert client.get("/api/v1/review-queue", headers=role("kb_owner")).json()["total"] == 0


def test_escalation_bundle(client: Any, analysed: dict[str, dict[str, Any]]) -> None:
    bundle_id = analysed["DEMO-002"]["details"]["escalation"]["bundle_id"]
    url = f"/api/v1/escalations/{bundle_id}"
    b = client.get(url, headers=role("l2", "DBA")).json()
    assert b["assignment_group"] == "DBA" and b["tier"] == "L3"
    assert [r["rule"] for r in b["reasons"]][0] == "E4"
    assert b["description_redacted"] and b["kb_matches"][0]["kb_id"] == "KB-0001"
    assert b["remaining_steps"] and b["masked"] is False
    masked = client.get(url, headers=role("l1")).json()
    assert masked["masked"] is True and all(
        s == "Handled by DBA" for s in masked["remaining_steps"]
    )
    md = client.get(url, params={"fmt": "md"}, headers=role("l2", "DBA"))
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith(f"# Escalation bundle {bundle_id}")
    assert client.get(url, headers=role("l2", "NET-OPS")).status_code == 403
    assert client.get("/api/v1/escalations/nope", headers=role("admin")).status_code == 404


def test_secondary_issue_bundle(client: Any, analysed: dict[str, dict[str, Any]]) -> None:
    esc = analysed["DEMO-005"]["details"]["escalation"]
    b = client.get(f"/api/v1/escalations/{esc['bundle_id']}", headers=role("admin")).json()
    assert b["classification"]["subcategory"] == "CPU" and b["assignment_group"] == "EUC-HW"
