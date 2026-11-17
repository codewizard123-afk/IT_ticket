"""P3-T2: POST /analyze returns a schema-valid response using FakeEngine; intake and pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from backend import audit
from backend.db.sqlite import open_db
from shared.constants import CONTRACT_KEYS
from shared.schemas import EngineContext, EngineResult, SanitizedTicket
from shared.stubs import FakeEngine
from tests.p3_backend.conftest import (
    DEMO_IDS,
    EXPECTED,
    make_client,
    make_services,
    role,
)

URL = "/api/v1/analyze"


def _db(services: Any) -> sqlite3.Connection:
    return open_db(services.settings.db_path)


@pytest.mark.parametrize("ticket_id", DEMO_IDS)
def test_demo_tickets_end_to_end(
    ticket_id: str, client: Any, demo_tickets: dict[str, Any], contract_schema: dict
) -> None:
    r = client.post(URL, json=demo_tickets[ticket_id], headers=role("l2"))
    assert r.status_code == 200, r.text
    body = r.json()
    jsonschema.validate(body, contract_schema)
    exp = EXPECTED[ticket_id]
    assert body["ticket_id"] == ticket_id
    assert body["escalation_required"] is exp["esc"]
    assert body["human_approval_required"] is exp["approval"]
    if "group" in exp:
        assert body["details"]["escalation"]["assignment_group"] == exp["group"]


def test_contract_view_has_exactly_the_8_keys(
    client: Any, demo_tickets: dict[str, Any], contract_schema: dict
) -> None:
    r = client.post(
        URL, params={"view": "contract"}, json=demo_tickets["DEMO-001"], headers=role("l1")
    )
    assert r.status_code == 200
    assert tuple(r.json()) == CONTRACT_KEYS
    jsonschema.validate(r.json(), contract_schema)


def test_ticket_ids_are_assigned_in_sequence(client: Any) -> None:
    body = {"title": "Printer jammed again", "description": "The 3rd floor printer is jammed."}
    a = client.post(URL, json=body, headers=role("l1")).json()["ticket_id"]
    b = client.post(URL, json=body, headers=role("l1")).json()["ticket_id"]
    assert a.startswith("INC-") and a.endswith("-0001") and b.endswith("-0002")


class SpyEngine:
    def __init__(self) -> None:
        self.inner = FakeEngine()
        self.tickets: list[SanitizedTicket] = []
        self.contexts: list[EngineContext] = []

    def run(self, ticket: SanitizedTicket, ctx: EngineContext) -> EngineResult:
        self.tickets.append(ticket)
        self.contexts.append(ctx)
        return self.inner.run(ticket, ctx)


def test_gold_labels_never_reach_the_engine(tmp_path: Path, demo_tickets: dict[str, Any]) -> None:
    spy = SpyEngine()
    services = make_services(tmp_path, engine=spy)
    body = dict(demo_tickets["DEMO-002"])
    body["expected_decision"] = "escalate"  # top-level leak attempt too
    with make_client(services) as c:
        assert c.post(URL, json=body, headers=role("l1")).status_code == 200
    seen = spy.tickets[0].model_dump_json()
    assert "expected" not in seen and "Critical" not in seen
    with _db(services) as conn:
        row = dict(conn.execute("SELECT * FROM tickets").fetchone())
        assert "expected" not in json.dumps(row)
        # reporter fields are kept as reporter_* only
        assert row["reporter_category"] is None


def test_raw_pii_is_redacted_before_engine_and_storage(tmp_path: Path) -> None:
    spy = SpyEngine()
    services = make_services(tmp_path, engine=spy)
    body = {
        "title": "Outlook keeps crashing",
        "description": "Please call me back, my mail is jane.doe@example.com and it crashes.",
    }
    with make_client(services) as c:
        resp = c.post(URL, json=body, headers=role("l1")).json()
    assert "jane.doe@example.com" not in spy.tickets[0].description
    assert resp["details"]["redactions"] == {"count": 1, "types": ["EMAIL"]}
    with _db(services) as conn:
        dump = json.dumps([dict(r) for r in conn.execute("SELECT * FROM tickets")])
        dump += json.dumps([dict(r) for r in conn.execute("SELECT * FROM audit_log")])
        dump += json.dumps([dict(r) for r in conn.execute("SELECT * FROM analyses")])
    assert "jane.doe@example.com" not in dump


def test_secret_in_ticket_escalates_to_security(client: Any) -> None:
    body = {
        "title": "Cannot log in to VPN",
        "description": "My VPN login fails, my password=Hunter2! please fix it today.",
    }
    resp = client.post(URL, json=body, headers=role("l2")).json()
    assert resp["escalation_required"] is True
    esc = resp["details"]["escalation"]
    assert esc["assignment_group"] == "SEC-OPS" and esc["tier"] == "L3"
    assert esc["reasons"][0]["rule"] == "E5"


def test_analysis_is_persisted_audited_and_timed(
    client: Any, services: Any, demo_tickets: dict[str, Any]
) -> None:
    aid = client.post(URL, json=demo_tickets["DEMO-001"], headers=role("l1")).json()["details"][
        "analysis_id"
    ]
    with _db(services) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE analysis_id=?", (aid,)).fetchone()
        assert row["decision"] == "recommend" and row["role"] == "l1"
        kinds = {
            r["kind"] for r in conn.execute("SELECT kind FROM evidence WHERE analysis_id=?", (aid,))
        }
        assert {"similar", "kedb", "citation"} <= kinds
        events = [r["event"] for r in conn.execute("SELECT event FROM ticket_events ORDER BY id")]
        assert events == ["ticket.submitted", "analysis.completed"]
        entries = audit.list_entries(conn, entity="analysis", entity_id=aid)
        assert entries[0].action == "analysis.created"
        assert entries[0].payload["decision"] == "recommend"
        assert audit.verify_chain(conn).ok


def test_clarify_loop_over_http_terminates(client: Any, services: Any) -> None:
    body = {"title": "PC broken", "description": "it does not work"}
    first = client.post(URL, json=body, headers=role("l1")).json()
    assert first["details"]["decision"] == "clarify"
    assert 1 <= len(first["details"]["clarifying_questions"]) <= 3
    tid, aid = first["ticket_id"], first["details"]["analysis_id"]
    second = client.post(
        f"/api/v1/tickets/{tid}/clarify",
        json={"analysis_id": aid, "answer": "Laptop shows a blue screen at boot"},
        headers=role("l1"),
    )
    assert second.status_code == 200
    assert second.json()["details"]["decision"] == "escalate"
    again = client.post(
        f"/api/v1/tickets/{tid}/clarify",
        json={"analysis_id": second.json()["details"]["analysis_id"], "answer": "more"},
        headers=role("l1"),
    )
    assert again.status_code == 400
    with _db(services) as conn:
        events = [r["event"] for r in conn.execute("SELECT event FROM ticket_events ORDER BY id")]
        desc = conn.execute("SELECT description_redacted FROM tickets").fetchone()[0]
    assert "clarify.requested" in events and "clarify.answered" in events
    assert "[Clarification] Laptop shows a blue screen" in desc


def test_clarify_unknown_analysis_is_404(client: Any) -> None:
    r = client.post(
        "/api/v1/tickets/X/clarify", json={"analysis_id": "nope", "answer": "a"}, headers=role("l1")
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"title": "VPN", "description": "VPN is down for me since today"},
        {"title": "VPN is down", "description": "short"},
        {"title": "VPN is down", "description": "x" * 5001},
        {"description": "no title at all here"},
    ],
)
def test_validation_errors_use_the_error_model(client: Any, body: dict[str, Any]) -> None:
    r = client.post(URL, json=body, headers=role("l1"))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "validation" and err["request_id"]
    assert r.headers["X-Request-ID"] == err["request_id"]


def test_unauthenticated_is_401(client: Any, demo_tickets: dict[str, Any]) -> None:
    r = client.post(URL, json=demo_tickets["DEMO-001"])
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthenticated"


class BrokenEngine:
    def run(self, ticket: SanitizedTicket, ctx: EngineContext) -> EngineResult:
        raise RuntimeError("boom with ticket text inside")


def test_engine_failure_still_answers_and_escalates(
    tmp_path: Path, demo_tickets: dict[str, Any], contract_schema: dict
) -> None:
    services = make_services(tmp_path, engine=BrokenEngine())
    with make_client(services) as c:
        r = c.post(URL, json=demo_tickets["DEMO-001"], headers=role("l1"))
    assert r.status_code == 200
    body = r.json()
    jsonschema.validate(body, contract_schema)
    assert body["escalation_required"] is True
    assert body["details"]["degraded"] == ["engine_error"]


def test_missing_stores_return_503(tmp_path: Path, demo_tickets: dict[str, Any]) -> None:
    services = make_services(tmp_path, ticket_store=None, index_error="run make data")
    with make_client(services) as c:
        r = c.post(URL, json=demo_tickets["DEMO-001"], headers=role("l1"))
    assert r.status_code == 503 and r.json()["error"]["code"] == "index_missing"
