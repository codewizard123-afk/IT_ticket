"""M3 readiness: P4's own ``ui.client.ApiClient`` drives every endpoint of the real API.

The transport sends the client's requests through FastAPI's TestClient instead of the network, so
any mismatch in paths, params or response shapes between P3 and P4 fails here.
"""

from __future__ import annotations

from typing import Any

import pytest

ui_client = pytest.importorskip("ui.client")


def _transport(tc: Any):  # noqa: ANN202
    def send(method: str, path: str, params: dict[str, Any], body: Any, headers: dict[str, str]):  # noqa: ANN202
        r = tc.request(method, "/api/v1" + path, params=params or None, json=body, headers=headers)
        ctype = r.headers.get("content-type", "").split(";")[0]
        if ctype == "application/json":
            return ui_client.RawResponse(r.status_code, ctype, json_body=r.json(), text=r.text)
        return ui_client.RawResponse(r.status_code, ctype, text=r.text)

    return send


def test_every_ui_client_call_works_against_the_real_api(
    client: Any, demo_tickets: dict[str, Any]
) -> None:
    api = ui_client.ApiClient(_transport(client))
    assert api.health()["status"] == "ok"

    # analyst flow (JWT login, as in the UI login page)
    api.login("dba_carol", "demo-pass-2026")
    full = api.analyze(demo_tickets["DEMO-002"])
    assert full.details.decision == "escalate" and full.escalation_required
    contract = api.analyze_contract(demo_tickets["DEMO-007"])
    assert contract.kedb_match.error_code == "KB-0003"
    aid = full.details.analysis_id
    assert api.event(ticket_id="DEMO-002", analysis_id=aid, event="ui.viewed")
    fid = api.feedback(
        {
            "analysis_id": aid,
            "target_type": "kedb_match",
            "target_ref": "KB-0001",
            "verdict": "useful",
        }
    )
    assert fid
    bundle = api.escalation_bundle(full.details.escalation.bundle_id)
    assert bundle["assignment_group"] == "DBA"
    assert api.escalation_bundle(full.details.escalation.bundle_id, fmt="md").startswith("# ")
    assert api.decide(
        {
            "analysis_id": aid,
            "final_decision": "resolved",
            "resolution_notes": "Killed the blocking session after DBA sign-off",
        }
    )
    assert api.review_queue()["items"] is not None
    assert api.search_tickets("oracle slow", size=5)["items"]
    assert api.get_ticket("TKT-0001")["ticket_id"] == "TKT-0001"
    assert api.search_kb("ORA-01013")["items"]
    assert api.get_kb("KB-0001").kb_id == "KB-0001"
    record = api.audit(aid)
    assert record["chain"]["ok"] and record["feedback"]
    assert "analysis.created" in api.audit_export(fmt="jsonl")
    assert api.metrics_mttr(group_by="mode", metric="ttd")["groups"]
    assert api.metrics_summary()["analyses"] == 2
    assert len(api.trial_tickets()) == 12

    # clarify loop
    first = api.analyze({"title": "PC broken", "description": "it does not work at all"})
    assert first.details.decision == "clarify"
    second = api.clarify(first.ticket_id, first.details.analysis_id, "Blue screen on boot")
    assert second.details.decision == "escalate"

    # KB draft → approve (kb_owner) → searchable
    draft = api.create_kb_draft(aid)
    api.login("kb_dave", "demo-pass-2026")
    assert [d["draft_id"] for d in api.kb_drafts(status="draft")] == [draft["draft_id"]]
    art = api.approve_kb_draft(draft["draft_id"])
    assert art.status == "approved"

    # admin-only verify, demo-role header path
    api.use_demo_role("admin")
    assert api.audit_verify()["ok"] is True
    api.use_demo_role("l1")
    with pytest.raises(ui_client.ApiError) as err:
        api.audit_verify()
    assert err.value.status == 403 and err.value.code == "forbidden"
