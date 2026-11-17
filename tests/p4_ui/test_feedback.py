"""P4-T3: feedback buttons on every item + final decision form → /feedback, /decisions."""

from __future__ import annotations

from tests.p4_ui.helpers import click, page


def analyzed(make_client, demo_id: str = "DEMO-001", role: str = "l2"):
    client = make_client(role)
    at = page("1_Analyze.py", client)
    at.selectbox(key="sample_id").select(demo_id).run()
    at = click(at, "Analyze")
    assert not at.exception, at.exception
    return at, client


def thumbs(at):
    return [b for b in at.button if b.label == "👍"]


def test_every_item_has_feedback_buttons(make_client, golden):
    at, _ = analyzed(make_client, "DEMO-001")
    gold = golden["DEMO-001"]
    # classification + kedb + each step + recommendation + each similar incident
    expected = 1 + 1 + len(gold["details"]["steps"]) + 1 + len(gold["similar_incidents"])
    assert len(thumbs(at)) == expected
    assert len([b for b in at.button if b.label == "✗"]) == expected
    assert len([b for b in at.button if b.label == "?"]) == expected


def test_secondary_issue_items_have_buttons_too(make_client, golden):
    at, _ = analyzed(make_client, "DEMO-005")
    gold = golden["DEMO-005"]["details"]
    sec = gold["secondary_issues"][0]
    primary = 1 + 1 + len(gold["steps"]) + 1 + len(golden["DEMO-005"]["similar_incidents"])
    secondary = 1 + 1 + len(sec["steps"]) + 1
    assert len(thumbs(at)) == primary + secondary


def test_feedback_click_posts_and_shows_in_audit(make_client, backend):
    at, client = analyzed(make_client, "DEMO-001")
    aid = at.session_state["analysis"].details.analysis_id
    thumbs(at)[0].click().run()
    assert not at.exception
    assert len(backend.feedback) == 1
    fb = backend.feedback[0]
    assert fb["analysis_id"] == aid and fb["verdict"] == "useful"
    assert fb["target_type"] == "classification"
    # Done when: feedback rows appear in the audit view
    audit = client.audit(aid)
    assert audit["feedback"][0]["feedback_id"] == fb["feedback_id"]
    # the item now shows the recorded verdict instead of buttons
    assert len(thumbs(at)) == len(thumbs(analyzed(make_client, "DEMO-001")[0])) - 1


def test_incorrect_step_feedback_carries_citation(make_client, backend):
    at, _ = analyzed(make_client, "DEMO-003")
    crosses = [b for b in at.button if b.label == "✗"]
    crosses[2].click().run()  # cls, kedb, then the first step
    fb = backend.feedback[-1]
    assert fb["target_type"] == "step" and fb["verdict"] == "incorrect"
    assert fb["target_ref"].startswith("KB-0008#")


def test_classification_correction(make_client, backend):
    at, _ = analyzed(make_client, "DEMO-001")
    at = click(at, "Send correction")
    fb = backend.feedback[-1]
    assert fb["verdict"] == "incorrect" and fb["corrected_value"].startswith("Network/")


def test_final_decision_form_posts_decision_and_event(make_client, backend):
    at, client = analyzed(make_client, "DEMO-001")
    aid = at.session_state["analysis"].details.analysis_id
    notes = [t for t in at.text_area if t.label.startswith("Resolution notes")][0]
    notes.input("Cleared VPN credentials\nReconnected")
    at = click(at, "Submit decision")
    assert not at.exception
    dec = backend.decisions[aid]
    assert dec["final_decision"] == "resolved" and dec["mode"] == "assisted"
    assert dec["final_category"] == "Network" and dec["final_subcategory"] == "VPN"
    assert dec["kb_or_incident_used"] == "KB-0018"
    events = [e["event"] for e in backend.events if e["analysis_id"] == aid]
    assert "ui.viewed" in events and "decision.submitted" in events
    assert events.index("ui.viewed") < events.index("decision.submitted")
    assert client.audit(aid)["decision"]["final_decision"] == "resolved"
    assert any("Decision recorded" in str(s.value) for s in at.success)


def test_escalated_demo_defaults_to_escalated(make_client, backend):
    at, _ = analyzed(make_client, "DEMO-006")
    at = click(at, "Submit decision")
    aid = at.session_state["analysis"].details.analysis_id
    assert backend.decisions[aid]["final_decision"] == "escalated"


def test_kb_draft_offered_after_notes(make_client, backend):
    at, _ = analyzed(make_client, "DEMO-001", role="l2")
    notes = [t for t in at.text_area if t.label.startswith("Resolution notes")][0]
    notes.input("Reinstalled the VPN profile")
    at = click(at, "Submit decision")
    at = click(at, "📝 Propose KB draft from my notes")
    assert not at.exception
    assert len(backend.drafts) == 1
