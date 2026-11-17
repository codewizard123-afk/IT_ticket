"""P4-T5: review queue, KB browser (gaps, drafts, approve) and audit view."""

from __future__ import annotations

from tests.p4_ui.helpers import all_text, click, page


def seed(make_client, demo_tickets, role="l2", ids=("DEMO-001", "DEMO-002", "DEMO-003")):
    client = make_client(role)
    return client, [client.analyze(demo_tickets[i]) for i in ids]


# ---------------------------------------------------------------------------- review queue
def test_review_queue_sorted_by_priority(make_client, demo_tickets, clock):
    client, _ = seed(make_client, demo_tickets, ids=("DEMO-001", "DEMO-002", "DEMO-003"))
    clock.tick(600)
    at = page("4_Review_queue.py", client)
    assert not at.exception
    df = at.dataframe[0].value
    # DEMO-003 needs no approval and recommends -> not in the queue
    assert list(df["Ticket"]) == ["DEMO-002", "DEMO-001"]
    assert list(df["Priority"]) == ["Critical", "High"]
    assert set(df["Age"]) == {"10m"}


def test_review_queue_override_decision(make_client, demo_tickets, backend):
    client, resps = seed(make_client, demo_tickets, ids=("DEMO-002",))
    aid = resps[0].details.analysis_id
    at = page("4_Review_queue.py", client)
    at.selectbox(key="rq_open").select(aid).run()
    selects = {s.label: s for s in at.selectbox}
    selects["Priority"].select("High")
    at = click(at, "Submit decision")
    assert not at.exception
    assert backend.decisions[aid]["final_priority"] == "High"
    assert backend.decisions[aid]["final_decision"] == "escalated"
    # decided items leave the open queue
    assert not page("4_Review_queue.py", client).dataframe


def test_review_queue_filter_escalations(make_client, demo_tickets):
    client, _ = seed(make_client, demo_tickets, ids=("DEMO-001", "DEMO-006"))
    at = page("4_Review_queue.py", client)
    at.selectbox(key="rq_status").select("escalate").run()
    assert list(at.dataframe[0].value["Ticket"]) == ["DEMO-006"]


# ---------------------------------------------------------------------------- KB browser
def test_kb_browser_search_and_detail(make_client):
    at = page("5_KB_browser.py", make_client("l1"))
    assert not at.exception
    assert len(at.dataframe[0].value) == 30
    at.text_input(key="kb_q").input("gateway not responding").run()
    assert at.dataframe[0].value["KB"].iloc[0] == "KB-0018"
    at.selectbox(key="kb_open").select("KB-0018").run()
    text = all_text(at)
    assert "[KB-0018·res.1]" in text and "[KB-0018·diag.1]" in text


def test_kb_browser_coverage_gaps(make_client, demo_tickets):
    client = make_client("l2")
    client.analyze({"title": "quantum flux", "description": "quantum flux capacitor"})
    at = page("5_KB_browser.py", client)
    text = all_text(at)
    assert "DHCP" in text and "DB2" in text
    gap_tables = [d.value for d in at.dataframe if "Analyses without a KB match" in d.value.columns]
    assert gap_tables and gap_tables[0]["Analyses without a KB match"].sum() >= 1


def test_kb_draft_approval_flow(make_client, demo_tickets, backend):
    l2, resps = seed(make_client, demo_tickets, ids=("DEMO-001",))
    aid = resps[0].details.analysis_id
    l2.decide(
        {
            "analysis_id": aid,
            "final_decision": "resolved",
            "resolution_notes": "Re-imported the VPN profile\nReconnected",
            "mode": "assisted",
        }
    )
    draft = l2.create_kb_draft(aid)
    # an l1 sees the draft but cannot approve it
    at_l1 = page("5_KB_browser.py", make_client("l1"))
    assert "Only a KB owner can approve drafts." in all_text(at_l1)
    assert not [b for b in at_l1.button if b.label == "Approve and publish"]
    # the KB owner approves it, and it becomes searchable
    at = page("5_KB_browser.py", make_client("kb_owner"))
    at = click(at, "Approve and publish")
    assert not at.exception
    assert backend.drafts[draft["draft_id"]]["status"] == "approved"
    hits = make_client("l1").search_kb("Re-imported the VPN profile")["items"]
    assert draft["article"]["kb_id"] in {h["kb_id"] for h in hits} or any(
        h["kb_id"] == draft["article"]["kb_id"] for h in make_client("l1").search_kb()["items"]
    )


# ---------------------------------------------------------------------------- audit view
def test_audit_view_timeline_evidence_and_chain(make_client, demo_tickets):
    client, resps = seed(make_client, demo_tickets, ids=("DEMO-001",))
    aid = resps[0].details.analysis_id
    client.event("DEMO-001", "ui.viewed", aid)
    client.feedback({"analysis_id": aid, "target_type": "classification", "verdict": "useful"})
    at = page("6_Audit_view.py", client)
    at.selectbox(key="audit_pick").select(aid).run()
    assert not at.exception
    assert "HASH CHAIN OK" in all_text(at)
    frames = [d.value for d in at.dataframe]
    events = next(f for f in frames if "Event" in f.columns)
    assert {"ticket.submitted", "analysis.completed", "ui.viewed"} <= set(events["Event"])
    evidence = next(f for f in frames if "kind" in f.columns)
    assert "kedb" in set(evidence["kind"])
    feedback = next(f for f in frames if "Verdict" in f.columns)
    assert list(feedback["Verdict"]) == ["useful"]


def test_audit_view_shows_broken_chain(make_client, demo_tickets, backend):
    client, resps = seed(make_client, demo_tickets, role="admin", ids=("DEMO-001",))
    backend.tamper(2, {"forged": True})
    at = page("6_Audit_view.py", client)
    at.text_input(key="audit_id").input(resps[0].details.analysis_id).run()
    assert "HASH CHAIN BROKEN" in all_text(at)
    at = click(at, "Verify the whole audit chain")
    assert any("BROKEN" in str(e.value) for e in at.error)


def test_audit_view_access_denied_for_other_l1(make_client, demo_tickets):
    _, resps = seed(make_client, demo_tickets, role="admin", ids=("DEMO-001",))
    at = page("6_Audit_view.py", make_client("l1"))
    at.text_input(key="audit_id").input(resps[0].details.analysis_id).run()
    assert any("forbidden" in str(e.value) or "may not" in str(e.value) for e in at.error)


def test_audit_export_masked_for_l2(make_client, demo_tickets):
    client, _ = seed(make_client, demo_tickets, ids=("DEMO-001",))
    at = page("6_Audit_view.py", client)
    at = click(at, "Prepare export")
    assert not at.exception
    assert "masked for role l2" in all_text(at)
    assert len(at.get("download_button")) == 1
