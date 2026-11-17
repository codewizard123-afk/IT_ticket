"""P4-T1: typed client + mock API (golden fixtures, no backend)."""

from __future__ import annotations

import json
import threading

import httpx
import jsonschema
import pytest

from shared.schemas import AnalysisResponse
from ui.client import ApiClient, ApiError, http_transport, mock_transport
from ui.mock_api import make_server
from ui.mock_backend import MockBackend

CONTRACT_KEYS = {
    "ticket_id",
    "classification",
    "priority",
    "similar_incidents",
    "kedb_match",
    "recommended_solution",
    "escalation_required",
    "human_approval_required",
}


@pytest.mark.parametrize("demo_id", [f"DEMO-00{i}" for i in range(1, 8)])
def test_every_demo_matches_golden_and_schema(
    demo_id, make_client, demo_tickets, golden, contract_schema
):
    resp = make_client("l2").analyze(demo_tickets[demo_id])
    gold = golden[demo_id]
    assert resp.ticket_id == demo_id
    assert resp.classification.category == gold["classification"]["category"]
    assert resp.kedb_match.error_code == gold["kedb_match"]["error_code"]
    assert resp.details.decision == gold["details"]["decision"]
    assert resp.escalation_required == gold["escalation_required"]
    assert resp.human_approval_required == gold["human_approval_required"]
    jsonschema.validate(resp.model_dump(mode="json"), contract_schema)


def test_contract_view_has_exactly_eight_keys(make_client, demo_tickets, contract_schema):
    contract = make_client("l2").analyze_contract(demo_tickets["DEMO-001"])
    data = contract.model_dump(mode="json")
    assert set(data) == CONTRACT_KEYS
    jsonschema.validate(data, contract_schema)


def test_each_analysis_gets_a_new_id(make_client, demo_tickets):
    client = make_client("l2")
    a = client.analyze(demo_tickets["DEMO-001"]).details.analysis_id
    b = client.analyze(demo_tickets["DEMO-001"]).details.analysis_id
    assert a != b


def test_l1_never_sees_l2_step_text(make_client, demo_tickets, golden):
    l1 = make_client("l1").analyze(demo_tickets["DEMO-002"])
    l2_texts = {s["text"] for s in golden["DEMO-002"]["details"]["steps"] if s["role"] == "L2"}
    assert l2_texts
    shown = {s.text for s in l1.details.steps} | set(l1.recommended_solution.steps)
    assert not shown & l2_texts
    assert all(s.text == "Handled by DBA" for s in l1.details.steps if s.role == "L2")


def test_gold_labels_are_stripped(make_client, backend, demo_tickets):
    ticket = {**demo_tickets["DEMO-001"], "metadata": {"source": "demo", "expected_category": "X"}}
    resp = make_client("l2").analyze(ticket)
    stored = backend.analyses[resp.details.analysis_id]["ticket"]
    assert not any(k.startswith("expected_") for k in stored["metadata"])


def test_unknown_ticket_gets_id_and_clarify_then_terminates(make_client):
    client = make_client("l1")
    first = client.analyze({"title": "VPN broken", "description": "VPN broken"})
    assert first.ticket_id.startswith("INC-")
    assert first.details.decision == "clarify"
    assert first.details.clarifying_questions
    second = client.clarify(
        first.ticket_id,
        first.details.analysis_id,
        "VPN client says gateway not responding when working from home",
    )
    assert second.details.decision in ("recommend", "escalate")
    with pytest.raises(ApiError) as err:
        client.clarify(first.ticket_id, second.details.analysis_id, "again")
    assert err.value.status == 400


def test_missing_auth_is_401(backend):
    client = ApiClient(mock_transport(backend), mock=True)
    with pytest.raises(ApiError) as err:
        client.analyze({"title": "x", "description": "y"})
    assert err.value.status == 401


def test_login_sets_role(make_client):
    client = make_client(None)
    out = client.login("dba_carol", "pw")
    assert out["role"] == "l2" and client.assignment_group == "DBA"
    assert client.health()["status"] == "ok"


def test_search_filters_combine(make_client):
    client = make_client("admin")
    everything = client.search_tickets(size=200)["total"]
    net = client.search_tickets(category="Network", size=200)
    vpn = client.search_tickets(category="Network", subcategory="VPN", size=200)
    assert everything > net["total"] > vpn["total"] > 0
    assert all(i["subcategory"] == "VPN" for i in vpn["items"])
    high_net = client.search_tickets(category="Network", priority="High", size=200)
    assert high_net["total"] > 0
    assert all(i["priority"] == "P2" for i in high_net["items"])
    assert client.search_tickets(category="Network", subcategory="Oracle")["total"] == 0


def test_search_pagination(make_client):
    client = make_client("admin")
    p1 = client.search_tickets(page=1, size=10)
    p2 = client.search_tickets(page=2, size=10)
    assert len(p1["items"]) == 10
    assert {i["ticket_id"] for i in p1["items"]}.isdisjoint({i["ticket_id"] for i in p2["items"]})


def test_restricted_rows_hidden_for_l1(make_client, backend):
    restricted = [t for t, r in backend.ticket_store._recs.items() if r.restricted]
    assert restricted
    rid = restricted[0]
    l1_ids = {i["ticket_id"] for i in make_client("l1").search_tickets(size=200)["items"]}
    admin_ids = {i["ticket_id"] for i in make_client("admin").search_tickets(size=200)["items"]}
    assert rid in admin_ids and rid not in l1_ids
    with pytest.raises(ApiError) as err:
        make_client("l1").get_ticket(rid)
    assert err.value.status == 403


def test_kb_search_and_get(make_client):
    client = make_client("l1")
    hits = client.search_kb("vpn gateway not responding")["items"]
    assert hits[0]["kb_id"] == "KB-0018"
    assert client.get_kb("KB-0018").title


def test_audit_chain_detects_tampering(make_client, backend, demo_tickets):
    admin = make_client("admin")
    admin.analyze(demo_tickets["DEMO-003"])
    assert admin.audit_verify()["ok"] is True
    backend.tamper(1, {"forged": True})
    result = admin.audit_verify()
    assert result["ok"] is False and result["broken_at"] == 1


def test_audit_export_is_masked_for_l2(make_client, demo_tickets):
    l2 = make_client("l2")
    l2.analyze(demo_tickets["DEMO-001"])
    text = l2.audit_export("jsonl")
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows and all(r["actor"].startswith("user-") for r in rows)
    assert "demo-l2" not in text
    csv_text = make_client("admin").audit_export("csv")
    assert csv_text.startswith("id,") and "demo-l2" in csv_text


def test_ttd_metrics_from_events(make_client, clock, demo_tickets):
    client = make_client("l2")
    resp = client.analyze(demo_tickets["DEMO-001"])
    aid = resp.details.analysis_id
    client.event(resp.ticket_id, "ui.viewed", aid)
    clock.tick(90)
    client.decide({"analysis_id": aid, "final_decision": "resolved", "mode": "assisted"})
    ttd = client.metrics_mttr(group_by="mode", metric="ttd")
    assert ttd["groups"] == [
        {"key": "assisted", "n": 1, "mean_s": 90.0, "median_s": 90.0, "p90_s": 90.0}
    ]
    mttr = client.metrics_mttr(group_by="category")
    assert mttr["groups"][0]["key"] == "Network"


def test_ui_cannot_post_backend_events(make_client):
    with pytest.raises(ApiError) as err:
        make_client("l1").event("T01", "analysis.completed")
    assert err.value.status == 400


def test_trial_tickets_have_no_gold(make_client):
    items = make_client("l1").trial_tickets()
    assert len(items) == 12
    for t in items:
        assert not any(k.startswith("expected_") for k in t.get("metadata", {}))
        assert "gold" not in json.dumps(t).lower()


def test_escalation_bundle_md(make_client, demo_tickets):
    client = make_client("l2")
    resp = client.analyze(demo_tickets["DEMO-002"])
    md = client.escalation_bundle(resp.details.escalation.bundle_id, fmt="md")
    assert md.startswith("# Escalation bundle") and "DBA" in md


def test_http_mock_api_round_trip(demo_tickets, contract_schema):
    server = make_server("127.0.0.1", 0, MockBackend())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/api/v1"
        client = ApiClient(http_transport(base, 5), demo_role="l2")
        resp = client.analyze(demo_tickets["DEMO-004"])
        assert isinstance(resp, AnalysisResponse)
        jsonschema.validate(resp.model_dump(mode="json"), contract_schema)
        r = httpx.get(f"{base}/nope", headers={"X-Demo-Role": "l1"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_api_raises_api_error():
    client = ApiClient(http_transport("http://127.0.0.1:9/api/v1", 0.5), demo_role="l1")
    with pytest.raises(ApiError) as err:
        client.health()
    assert err.value.code == "unreachable"


def test_from_env_uses_mock(monkeypatch):
    monkeypatch.setenv("MOCK_API", "1")
    client = ApiClient.from_env(demo_role="l1")
    assert client.mock is True
    assert client.health()["mock"] is True
