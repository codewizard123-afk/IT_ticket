"""P3-T5: auth + RBAC (architecture §8.6).

Done when: an L1 user sees no L2 step text and no restricted tickets; export is masked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from backend import access, audit, auth
from backend.db.sqlite import open_db
from shared.schemas import TicketRecord, User
from shared.stubs import FakeTicketStore
from tests.p3_backend.conftest import make_client, make_services, role

URL = "/api/v1/analyze"
PASSWORD = "demo-pass-2026"


def _login(client: Any, username: str) -> dict[str, str]:
    r = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


# --- authentication ------------------------------------------------------------------------------
def test_login_issues_a_jwt_for_the_configured_user(client: Any) -> None:
    r = client.post("/api/v1/auth/login", json={"username": "dba_carol", "password": PASSWORD})
    body = r.json()
    assert body["role"] == "l2" and body["assignment_group"] == "DBA"
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['token']}"}).json()
    assert me == {"username": "dba_carol", "role": "l2", "assignment_group": "DBA"}


@pytest.mark.parametrize("password", ["wrong", ""])
def test_bad_password_is_401(client: Any, password: str) -> None:
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": password})
    assert r.status_code == 401


def test_tampered_and_expired_tokens_are_rejected(client: Any, services: Any) -> None:
    secret = auth.signing_secret(services.settings.app_secret)
    user = User(username="admin", role="admin")
    old = auth.issue_token(user, secret, datetime.now(UTC) - timedelta(hours=9))
    forged = auth.issue_token(user, "another-secret-that-is-long-enough-123", datetime.now(UTC))
    for token in (old, forged, "garbage"):
        r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401


def test_demo_header_ignored_when_demo_auth_is_off(tmp_path: Path) -> None:
    from dataclasses import replace

    services = make_services(tmp_path)
    services.settings = replace(services.settings, demo_auth=False)
    with make_client(services) as c:
        assert c.get("/api/v1/auth/me", headers=role("admin")).status_code == 401


def test_demo_role_gets_default_group(client: Any) -> None:
    assert client.get("/api/v1/auth/me", headers=role("l2")).json()["assignment_group"] == "NET-OPS"


# --- display: L2 step text ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ticket_id", "group"), [("DEMO-004", "EUC-SW"), ("DEMO-002", "DBA"), ("DEMO-007", "DBA")]
)
def test_l1_never_receives_l2_step_text(
    client: Any, demo_tickets: dict[str, Any], ticket_id: str, group: str
) -> None:
    l2 = client.post(URL, json=demo_tickets[ticket_id], headers=role("l2")).json()
    l2_texts = [s["text"] for s in l2["details"]["steps"] if s["role"] == "L2"]
    assert l2_texts
    l1 = client.post(URL, json=demo_tickets[ticket_id], headers=role("l1")).json()
    dump = json.dumps(l1)
    assert not any(t in dump for t in l2_texts)
    masked = [s for s in l1["details"]["steps"] if s["role"] == "L2"]
    assert all(s["text"] == f"Handled by {group}" and s["masked"] for s in masked)
    assert f"Handled by {group}" in l1["recommended_solution"]["steps"]


def test_l1_contract_view_is_masked_too(client: Any, demo_tickets: dict[str, Any]) -> None:
    r = client.post(
        URL, params={"view": "contract"}, json=demo_tickets["DEMO-004"], headers=role("l1")
    )
    assert "Handled by EUC-SW" in r.json()["recommended_solution"]["steps"]


# --- retrieval: restricted tickets --------------------------------------------------------------
VPN = "VPN not connecting from home, client shows Gateway not responding after restart"


def _restricted_store() -> FakeTicketStore:
    def rec(tid: str, restricted: bool, extra: str) -> TicketRecord:
        return TicketRecord(
            ticket_id=tid, title="VPN gateway not responding", description=f"{VPN} {extra}",
            category="Network", subcategory="VPN", priority="P2", resolution="Re-imported profile",
            resolution_origin="curated_verified", restricted=restricted,
            assignment_group="NET-OPS", collection="tickets_verified",
        )  # fmt: skip

    return FakeTicketStore(
        [rec("PUB-1", False, "on laptop"), rec("SECRET-1", True, "for the CFO laptop")]
    )


@pytest.fixture
def real_engine_client(tmp_path: Path):  # noqa: ANN201
    from ai.engine import AIEngine

    services = make_services(tmp_path, engine=AIEngine(), ticket_store=_restricted_store())
    with make_client(services) as c:
        yield c


def _ids(body: dict[str, Any]) -> set[str]:
    ids = {s["incident_id"] for s in body["similar_incidents"]}
    return ids | {s["incident_id"] for s in body["details"]["keyword_only"]}


@pytest.mark.parametrize(
    ("headers", "visible"),
    [
        (role("l1"), False),
        (role("kb_owner"), False),
        (role("l2", "DBA"), False),
        (role("l2", "NET-OPS"), True),
        (role("admin"), True),
    ],
)
def test_restricted_tickets_only_reach_allowed_roles(
    real_engine_client: Any, headers: dict[str, str], visible: bool
) -> None:
    body = real_engine_client.post(
        URL, json={"title": "VPN gateway not responding", "description": VPN}, headers=headers
    ).json()
    assert "PUB-1" in _ids(body)
    assert ("SECRET-1" in _ids(body)) is visible
    assert ("SECRET-1" in json.dumps(body)) is visible


def test_retrieval_filters_per_role() -> None:
    assert access.retrieval_filters(User(username="a", role="l1")) == {"restricted": False}
    assert access.retrieval_filters(User(username="a", role="admin")) == {}
    l2 = access.retrieval_filters(User(username="a", role="l2", assignment_group="DBA"))
    assert l2 == {"$or": [{"restricted": False}, {"assignment_group": "DBA"}]}


# --- export -------------------------------------------------------------------------------------
def test_export_is_masked_for_l1_and_l2_and_watermarked(
    client: Any, services: Any, demo_tickets: dict[str, Any]
) -> None:
    client.post(URL, json=demo_tickets["DEMO-001"], headers=role("l1"))
    for r in (role("l1"), role("l2")):
        resp = client.get("/api/v1/audit/export", params={"format": "jsonl"}, headers=r)
        assert resp.status_code == 200 and resp.headers["X-Export-Masked"] == "true"
        rows = [json.loads(line) for line in resp.text.splitlines()]
        created = next(x for x in rows if x["action"] == "analysis.created")
        assert created["actor"].startswith("user-") and "demo-l1" not in resp.text
        payload = json.loads(created["payload_json"])
        assert payload["input_redacted"] == "***" and payload["issues"] == "***"
        assert payload["decision"] == "recommend"
        assert "Gateway not responding" not in created["payload_json"]
    admin = client.get("/api/v1/audit/export", params={"format": "csv"}, headers=role("admin"))
    assert admin.headers["X-Export-Masked"] == "false" and "demo-l1" in admin.text
    with open_db(services.settings.db_path) as conn:
        assert sum(e.action == "audit.exported" for e in audit.list_entries(conn)) == 3
        assert audit.verify_chain(conn).ok


def test_kb_owner_cannot_export(client: Any, services: Any) -> None:
    r = client.get("/api/v1/audit/export", headers=role("kb_owner"))
    assert r.status_code == 403
    with open_db(services.settings.db_path) as conn:
        assert audit.list_entries(conn)[-1].action == "access.denied"


# --- audit view ---------------------------------------------------------------------------------
def _analysis(client: Any, ticket: dict[str, Any], headers: dict[str, str]) -> str:
    return client.post(URL, json=ticket, headers=headers).json()["details"]["analysis_id"]


def test_audit_view_permissions(client: Any, demo_tickets: dict[str, Any]) -> None:
    alice = _login(client, "l1_alice")
    aid = _analysis(client, demo_tickets["DEMO-001"], alice)  # Network → NET-OPS
    get = lambda h: client.get(f"/api/v1/audit/{aid}", headers=h).status_code  # noqa: E731
    assert get(alice) == 200
    assert get(role("l1")) == 403  # another l1 user
    assert get(_login(client, "l2_bob")) == 200  # NET-OPS group
    assert get(_login(client, "dba_carol")) == 403  # other group
    assert get(role("kb_owner")) == 403
    assert get(role("admin")) == 200


def test_l1_audit_record_hides_l2_text(client: Any, demo_tickets: dict[str, Any]) -> None:
    l2 = client.post(URL, json=demo_tickets["DEMO-004"], headers=role("l2")).json()
    secret = next(s["text"] for s in l2["details"]["steps"] if s["role"] == "L2")
    aid = _analysis(client, demo_tickets["DEMO-004"], role("l1"))
    rec = client.get(f"/api/v1/audit/{aid}", headers=role("l1")).json()
    assert secret not in json.dumps(rec)
    assert rec["chain"]["ok"] is True and rec["filtered_for_role"] == "l1"
    assert [e["event"] for e in rec["timeline"]] == ["ticket.submitted", "analysis.completed"]
    full = client.get(f"/api/v1/audit/{aid}", headers=role("admin")).json()
    assert secret in json.dumps(full)


def test_clarify_on_someone_elses_analysis_is_denied(client: Any) -> None:
    first = client.post(
        URL, json={"title": "PC broken", "description": "it does not work"}, headers=role("l1")
    ).json()
    r = client.post(
        f"/api/v1/tickets/{first['ticket_id']}/clarify",
        json={"analysis_id": first["details"]["analysis_id"], "answer": "blue screen"},
        headers=role("kb_owner"),
    )
    assert r.status_code == 403
