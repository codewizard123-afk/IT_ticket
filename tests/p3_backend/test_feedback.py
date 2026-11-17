"""P3-T6: feedback, decisions, KB prior and mark_verified (architecture §8.8).

Done when: a "useful" vote changes the KB ranking on the next run; a resolved ticket becomes a
Historical Incident source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai.kb_search import search_kb
from backend import feedback
from backend.db.sqlite import open_db
from shared.stubs import FakeKBStore
from tests.p3_backend.conftest import make_client, make_services, role
from tests.p3_backend.test_api_analyze import SpyEngine

URL = "/api/v1/analyze"


def _analyze(client: Any, ticket: dict[str, Any], headers: dict[str, str]) -> str:
    r = client.post(URL, json=ticket, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["details"]["analysis_id"]


def _vote(client: Any, aid: str, ref: str, verdict: str = "useful", **kw: Any) -> Any:
    headers = kw.pop("headers", role("l1"))
    body = {"analysis_id": aid, "target_type": "kedb_match", "target_ref": ref, "verdict": verdict}
    return client.post("/api/v1/feedback", json=body | kw, headers=headers)


# --- kb_prior -----------------------------------------------------------------------------------
def test_kb_prior_formula_and_clip(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    rows = [("KB-0001", "useful")] * 3 + [("KB-0001", "incorrect")] + [("KB-0002", "incorrect")]
    rows += [("KB-0003#res.1", "useful")] + [("KB-0009", "unresolved")]
    rows += [("KB-0050", "useful")] * 200
    for n, (ref, verdict) in enumerate(rows):
        conn.execute(
            "INSERT INTO feedback (feedback_id, target_type, target_ref, verdict) VALUES (?,?,?,?)",
            (str(n), "kedb_match" if "#" not in ref else "step", ref, verdict),
        )
    conn.execute(
        "INSERT INTO feedback (feedback_id, target_type, target_ref, verdict) "
        "VALUES ('x', 'similar_incident', 'KB-0007', 'useful')"
    )
    prior = feedback.kb_prior(conn)
    assert prior["KB-0001"] == round(2 / 9, 4)
    assert prior["KB-0002"] == round(-1 / 6, 4)
    assert prior["KB-0003"] == round(1 / 6, 4)  # step citation counts for its article
    assert prior["KB-0050"] == 0.5  # clipped
    assert "KB-0009" not in prior and "KB-0007" not in prior


def test_useful_vote_reaches_the_engine_on_the_next_run(
    tmp_path: Path, demo_tickets: dict[str, Any]
) -> None:
    spy = SpyEngine()
    services = make_services(tmp_path, engine=spy)
    with make_client(services) as c:
        aid = _analyze(c, demo_tickets["DEMO-001"], role("l1"))
        assert spy.contexts[0].feedback_prior == {}
        assert _vote(c, aid, "KB-0019").status_code == 200
        _analyze(c, demo_tickets["DEMO-001"], role("l1"))
    assert spy.contexts[1].feedback_prior == {"KB-0019": round(1 / 6, 4)}


def test_useful_vote_raises_the_kb_score_in_p2_ranking(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    kb = FakeKBStore()
    q = "VPN client says Gateway not responding and keeps disconnecting"
    before = {
        h.kb_id: h.score for h in search_kb(q, None, 10, kb, feedback_prior=feedback.kb_prior(conn))
    }
    target = sorted(before, key=lambda k: -before[k])[1]  # the runner-up
    for n in range(10):
        conn.execute(
            "INSERT INTO feedback (feedback_id, target_type, target_ref, verdict) VALUES (?,?,?,?)",
            (str(n), "kedb_match", target, "useful"),
        )
    after = {
        h.kb_id: h.score for h in search_kb(q, None, 10, kb, feedback_prior=feedback.kb_prior(conn))
    }
    assert after[target] > before[target]
    others = [k for k in before if k != target]
    assert all(after[k] == before[k] for k in others)


# --- decisions ----------------------------------------------------------------------------------
def test_resolved_decision_makes_a_verified_historical_incident(
    tmp_path: Path, demo_tickets: dict[str, Any]
) -> None:
    from ai.engine import AIEngine

    services = make_services(tmp_path)
    with make_client(services) as c:
        aid = _analyze(c, demo_tickets["DEMO-001"], role("l1"))
        r = c.post(
            "/api/v1/decisions",
            json={
                "analysis_id": aid,
                "final_decision": "resolved",
                "resolution_notes": "Re-imported the VPN profile; mail jane@example.com if needed",
                "mode": "assisted",
            },
            headers=role("l1"),
        )
        assert r.json() == {"ok": True}
    rec = services.ticket_store.get(["DEMO-001"])[0]
    assert rec.resolution_origin == "analyst_verified" and rec.verified_by == "demo-l1"
    assert rec.collection == "tickets_verified"
    assert "jane@example.com" not in (rec.resolution or "") and "Re-imported" in rec.resolution
    with open_db(services.settings.db_path) as conn:
        events = [e["event"] for e in conn.execute("SELECT event FROM ticket_events ORDER BY id")]
        status = conn.execute("SELECT status FROM tickets WHERE ticket_id='DEMO-001'").fetchone()[0]
    assert events[-2:] == ["decision.submitted", "ticket.resolved"] and status == "resolved"

    # the next similar ticket sees it as a citable, analyst-verified historical incident
    services.engine = AIEngine()
    with make_client(services) as c:
        body = c.post(
            URL,
            json={
                "title": demo_tickets["DEMO-001"]["title"],
                "description": demo_tickets["DEMO-001"]["description"],
            },
            headers=role("admin"),
        ).json()
    hit = next(s for s in body["details"]["similar"] if s["incident_id"] == "DEMO-001")
    assert hit["resolution_origin"] == "analyst_verified"


def test_overrides_are_recorded(client: Any, services: Any, demo_tickets: dict[str, Any]) -> None:
    aid = _analyze(client, demo_tickets["DEMO-003"], role("l1"))
    client.post(
        "/api/v1/decisions",
        json={
            "analysis_id": aid,
            "final_decision": "escalated",
            "final_category": "Software",
            "final_priority": "High",
            "mode": "assisted",
        },
        headers=role("l1"),
    )
    with open_db(services.settings.db_path) as conn:
        rows = {
            r["field"]: (r["old_value"], r["new_value"])
            for r in conn.execute("SELECT * FROM overrides")
        }
        dec = conn.execute("SELECT * FROM decisions").fetchone()
    assert rows == {"category": ("Hardware", "Software"), "priority": ("Medium", "High")}
    assert dec["final_decision"] == "escalated" and dec["user"] == "demo-l1"


def test_manual_decision_needs_a_manual_session(client: Any) -> None:
    r = client.post(
        "/api/v1/decisions",
        json={"analysis_id": "manual-1", "final_decision": "resolved", "mode": "manual"},
        headers=role("l1"),
    )
    assert r.status_code == 404


# --- feedback -----------------------------------------------------------------------------------
def test_feedback_is_stored_redacted_and_audited(
    client: Any, services: Any, demo_tickets: dict[str, Any]
) -> None:
    aid = _analyze(client, demo_tickets["DEMO-001"], role("l1"))
    r = _vote(client, aid, "KB-0018", comment="worked, thanks - bob@example.com")
    fid = r.json()["feedback_id"]
    with open_db(services.settings.db_path) as conn:
        row = conn.execute("SELECT * FROM feedback WHERE feedback_id=?", (fid,)).fetchone()
        actions = [r["action"] for r in conn.execute("SELECT action FROM audit_log")]
    assert row["verdict"] == "useful" and "bob@example.com" not in row["comment"]
    assert "feedback.created" in actions


@pytest.mark.parametrize(
    ("headers", "status"), [(role("kb_owner"), 403), (role("l2", "DBA"), 403), (role("admin"), 200)]
)
def test_feedback_on_someone_elses_analysis(
    client: Any, demo_tickets: dict[str, Any], headers: dict[str, str], status: int
) -> None:
    aid = _analyze(client, demo_tickets["DEMO-001"], role("l1"))
    assert _vote(client, aid, "KB-0018", headers=headers).status_code == status


def test_feedback_validation(client: Any, demo_tickets: dict[str, Any]) -> None:
    aid = _analyze(client, demo_tickets["DEMO-001"], role("l1"))
    assert _vote(client, "nope", "KB-0018").status_code == 404
    assert _vote(client, aid, "KB-0018", issue_idx=3).status_code == 400
    assert _vote(client, aid, "KB-0018", verdict="meh").status_code == 400


def test_classification_corrections_for_p2(client: Any, demo_tickets: dict[str, Any]) -> None:
    aid = _analyze(client, demo_tickets["DEMO-003"], role("l1"))
    client.post(
        "/api/v1/feedback",
        json={
            "analysis_id": aid,
            "target_type": "classification",
            "verdict": "incorrect",
            "corrected_value": "Software/Operating System",
        },
        headers=role("l1"),
    )
    assert client.get("/api/v1/internal/corrections", headers=role("l1")).status_code == 403
    items = client.get("/api/v1/internal/corrections", headers=role("l2")).json()["items"]
    assert items[0]["predicted"] == {"category": "Hardware", "subcategory": "Monitor"}
    assert items[0]["corrected_value"] == "Software/Operating System"
    assert items[0]["title"] == demo_tickets["DEMO-003"]["title"]
