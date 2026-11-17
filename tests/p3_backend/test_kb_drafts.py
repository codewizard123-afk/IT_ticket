"""P3-T8: KB draft loop (architecture §8.9). Done when: an approved draft appears in /kb/search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend import kb_drafts
from shared.stubs import FakeLLM
from tests.p3_backend.conftest import make_client, make_services, role

NOTES = (
    "Swapped the USB-C dongle for the docking station firmware 3.2 version\n"
    "- Disable the antivirus to test the dongle\n"
    "- Re-plugged the zebrafish adapter into the rear port"
)


def _analysis_with_notes(c: Any, ticket: dict[str, Any], notes: str = NOTES) -> str:
    aid = c.post("/api/v1/analyze", json=ticket, headers=role("l2")).json()["details"][
        "analysis_id"
    ]
    r = c.post(
        "/api/v1/decisions",
        json={"analysis_id": aid, "final_decision": "resolved", "resolution_notes": notes},
        headers=role("l2"),
    )
    assert r.status_code == 200, r.text
    return aid


def _search_ids(c: Any, q: str) -> list[str]:
    return [
        i["kb_id"]
        for i in c.get("/api/v1/kb/search", params={"q": q}, headers=role("l1")).json()["items"]
    ]


def test_draft_then_approve_makes_it_searchable(
    client: Any, services: Any, demo_tickets: dict[str, Any]
) -> None:
    aid = _analysis_with_notes(client, demo_tickets["DEMO-003"])
    r = client.post("/api/v1/kb/drafts", json={"analysis_id": aid}, headers=role("l2"))
    assert r.status_code == 200, r.text
    draft = r.json()
    art = draft["article"]
    assert draft["status"] == "draft" and art["status"] == "draft"
    assert art["drafted_by"] == "template"  # offline: no Groq key in tests
    assert art["category"] == "Hardware" and art["subcategory"] == "Monitor"
    assert all("antivirus" not in s.lower() for s in art["resolution"]["steps"])
    assert any("zebrafish" in s for s in art["resolution"]["steps"])
    kb_id = art["kb_id"]
    # drafts are not searchable (resolution steps never are; the title is, once approved)
    assert kb_id not in _search_ids(client, "zebrafish adapter external monitor")
    assert client.get(f"/api/v1/kb/{kb_id}", headers=role("l1")).status_code == 404

    r = client.post(
        f"/api/v1/kb/drafts/{draft['draft_id']}/approve",
        json={"edits": {"title": "Zebrafish adapter blanks the external monitor"}},
        headers=role("kb_owner"),
    )
    assert r.status_code == 200, r.text
    approved = r.json()
    assert approved["status"] == "approved" and approved["version"] == 1
    assert approved["approved_by"] == "demo-kb_owner"
    assert approved["title"] == "Zebrafish adapter blanks the external monitor"

    assert kb_id in _search_ids(client, "zebrafish adapter external monitor")
    assert client.get(f"/api/v1/kb/{kb_id}", headers=role("l1")).json()["kb_id"] == kb_id
    governed = json.loads(services.settings.kb_governed_path.read_text(encoding="utf-8"))
    ids = [a["kb_id"] for a in governed]
    assert kb_id in ids and "KB-0001" in ids and len(ids) == len(set(ids))
    listed = client.get("/api/v1/kb/drafts", params={"status": "approved"}, headers=role("admin"))
    assert [d["draft_id"] for d in listed.json()["items"]] == [draft["draft_id"]]


def test_roles(client: Any, demo_tickets: dict[str, Any]) -> None:
    aid = _analysis_with_notes(client, demo_tickets["DEMO-003"])
    assert (
        client.post("/api/v1/kb/drafts", json={"analysis_id": aid}, headers=role("l1")).status_code
        == 403
    )
    draft = client.post("/api/v1/kb/drafts", json={"analysis_id": aid}, headers=role("l2")).json()
    url = f"/api/v1/kb/drafts/{draft['draft_id']}/approve"
    assert client.post(url, json={}, headers=role("l2")).status_code == 403
    assert client.post(url, json={}, headers=role("admin")).status_code == 200
    assert client.post(url, json={}, headers=role("admin")).status_code == 400  # already approved


def test_draft_needs_notes(client: Any, demo_tickets: dict[str, Any]) -> None:
    aid = client.post("/api/v1/analyze", json=demo_tickets["DEMO-001"], headers=role("l2")).json()[
        "details"
    ]["analysis_id"]
    r = client.post("/api/v1/kb/drafts", json={"analysis_id": aid}, headers=role("l2"))
    assert r.status_code == 400


def test_unsafe_edits_are_rejected(client: Any, demo_tickets: dict[str, Any]) -> None:
    aid = _analysis_with_notes(client, demo_tickets["DEMO-003"])
    draft = client.post("/api/v1/kb/drafts", json={"analysis_id": aid}, headers=role("l2")).json()
    bad = {"resolution": {"steps": ["Share the admin password with the user over Teams"]}}
    r = client.post(
        f"/api/v1/kb/drafts/{draft['draft_id']}/approve",
        json={"edits": bad},
        headers=role("kb_owner"),
    )
    assert r.status_code == 400
    r = client.post(
        f"/api/v1/kb/drafts/{draft['draft_id']}/approve",
        json={"edits": {"kb_id": "KB-0001"}},
        headers=role("kb_owner"),
    )
    assert r.status_code == 400  # kb_id is not editable


def test_llm_draft_is_used_and_linted(tmp_path: Path, demo_tickets: dict[str, Any]) -> None:
    llm = FakeLLM(
        responses={
            "kb_draft": {
                "title": "Monitor blanks after dock firmware update",
                "symptoms": ["monitor goes black"],
                "common_error_codes": [],
                "root_cause": "dock firmware regression",
                "diagnostic_steps": ["Check the dock firmware version"],
                "resolution": {
                    "steps": ["Roll back the dock firmware", "Turn off Windows Defender firewall"],
                    "estimated_time": "15 min",
                    "requires_admin": True,
                },
                "escalation_criteria": ["several docks fail"],
                "tags": ["dock"],
            }
        }
    )
    services = make_services(tmp_path, llm=llm)
    with make_client(services) as c:
        aid = _analysis_with_notes(c, demo_tickets["DEMO-003"])
        art = c.post("/api/v1/kb/drafts", json={"analysis_id": aid}, headers=role("l2")).json()[
            "article"
        ]
    assert art["drafted_by"] == "llm"
    assert art["resolution"]["steps"] == ["Roll back the dock firmware"]
    call = next(x for x in llm.calls if x["purpose"] == "kb_draft")
    assert "<notes>" in call["user"]


def test_next_kb_id_after_existing_articles_and_drafts(services: Any) -> None:
    from backend.db.sqlite import open_db

    conn = open_db(services.settings.db_path)
    first = kb_drafts.next_kb_id(conn, services.kb_store)
    conn.execute(
        "INSERT INTO kb_drafts (draft_id, article_json, status) VALUES ('d', ?, 'draft')",
        (json.dumps({"kb_id": first}),),
    )
    second = kb_drafts.next_kb_id(conn, services.kb_store)
    assert int(second[3:]) == int(first[3:]) + 1


@pytest.mark.parametrize(
    "text",
    [
        "Disable the antivirus temporarily",
        "Send the password to the user by email",
        "Grant domain admin to all users in the group",
        "chmod 777 /etc/app",
        "Bypass the MFA policy for this user",
    ],
)
def test_unsafe_patterns(text: str) -> None:
    assert kb_drafts.is_unsafe(text)


def test_safe_steps_pass() -> None:
    assert not kb_drafts.is_unsafe("Restart the print spooler service once")


def test_kb_search_without_query_lists_approved_articles(client: Any) -> None:
    items = client.get(
        "/api/v1/kb/search", params={"category": "Network"}, headers=role("l1")
    ).json()["items"]
    assert items and all(i["category"] == "Network" for i in items)
