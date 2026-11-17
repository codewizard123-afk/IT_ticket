"""P4-T2: login/role page and Analyze page (all 7 demos render; no step without a badge)."""

from __future__ import annotations

import re

import pytest

from shared.schemas import Step
from tests.p4_ui.helpers import all_markdown, all_text, click, page
from ui.components.badges import chip_spec, citation_label, source_badges
from ui.components.analysis import step_html


def analyze_demo(make_client, demo_id: str, role: str = "l2"):
    client = make_client(role)
    at = page("1_Analyze.py", client)
    at.selectbox(key="sample_id").select(demo_id).run()
    at = click(at, "Analyze")
    assert not at.exception, at.exception
    return at, client


# ---------------------------------------------------------------------------- badges
def test_citation_badges():
    assert citation_label("KB-0018#res.1") == "KB-0018·res.1"
    assert citation_label("INC-H0007") == "INC-H0007"
    assert source_badges(["KB-0018#diag.4", "INC-H0007"]) == "[KB-0018·diag.4] [INC-H0007]"
    with pytest.raises(ValueError):
        source_badges([])


def test_every_chip_has_a_text_label():
    for kind, value in [
        ("decision", "escalate"),
        ("band", "low"),
        ("priority", "Critical"),
        ("tier", "LIKELY DUPLICATE"),
        ("origin", "synthetic_kb_derived"),
        ("role", "L2"),
        ("decision", "unknown-value"),
    ]:
        label, color = chip_spec(kind, value)
        assert label and color.startswith("#")
    assert chip_spec("origin", "synthetic_kb_derived")[0] == "SYNTHETIC"
    assert chip_spec("origin", "curated_verified")[0] == "VERIFIED"


def test_step_html_escapes_and_greys_attempted():
    step = Step(
        text="<script>x</script>",
        type="resolution",
        citations=["KB-0018#res.1"],
        role="L1",
        already_attempted=True,
    )
    out = step_html(1, step)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "ALREADY TRIED" in out and "opacity:0.5" in out and "[KB-0018·res.1]" in out


# ---------------------------------------------------------------------------- login page
def test_login_page_demo_role(make_client):
    client = make_client(None)
    at = page("app", client)
    assert not at.exception
    assert "DEMO AUTH" in all_markdown(at)
    at.selectbox[0].select("kb_owner")
    at = click(at, "Use this role")
    assert client.role == "kb_owner"


def test_login_page_user_login(make_client):
    client = make_client(None)
    at = page("app", client)
    at.text_input[1].input("l2_bob")
    at.text_input[2].input("secret")
    at = click(at, "Log in")
    assert client.role == "l2" and client.assignment_group == "NET-OPS"


def test_pages_require_a_role(make_client):
    at = page("1_Analyze.py", make_client(None))
    assert any("Login" in str(w.value) for w in at.warning)


# ---------------------------------------------------------------------------- analyze page
@pytest.mark.parametrize("demo_id", [f"DEMO-00{i}" for i in range(1, 8)])
def test_all_demos_render_with_badges(demo_id, make_client, golden):
    at, _ = analyze_demo(make_client, demo_id)
    md = all_markdown(at)
    gold = golden[demo_id]
    # every step of every issue is rendered with its source badge
    issues = [gold["details"]] + gold["details"]["secondary_issues"]
    for issue in issues:
        for step in issue["steps"]:
            assert f"[{citation_label(step['citations'][0])}" in md
    # no rendered step line lacks a badge
    for line in re.findall(r"<div[^>]*>\d+\. .*?</div>", md):
        assert re.search(r"\[[A-Z]+-[A-Z0-9]+", line), line
    assert gold["details"]["decision"].upper() in md
    assert ("HUMAN APPROVAL REQUIRED" in md) == gold["human_approval_required"]


def test_demo_005_shows_secondary_issue_tab(make_client):
    at, _ = analyze_demo(make_client, "DEMO-005")
    labels = [t.label for t in at.tabs]
    assert "Issue 1 (primary)" in labels
    assert any(lbl.startswith("Issue 2") for lbl in labels)


def test_escalation_shows_group_tier_and_bundle(make_client):
    at, _ = analyze_demo(make_client, "DEMO-002")
    text = all_text(at)
    assert "DBA" in text and "L3" in text and "E4" in text
    assert len(at.get("download_button")) == 2


def test_already_tried_step_is_greyed(make_client):
    at, _ = analyze_demo(make_client, "DEMO-001")
    assert "ALREADY TRIED" in all_markdown(at)


def test_synthetic_resolution_is_badged(make_client):
    at, _ = analyze_demo(make_client, "DEMO-005")
    md = all_markdown(at)
    assert "SYNTHETIC" in md and "VERIFIED" in md


def test_l1_sees_masked_l2_steps(make_client):
    at, _ = analyze_demo(make_client, "DEMO-002", role="l1")
    md = all_markdown(at)
    assert "Handled by DBA" in md
    assert "Ask the DBA to end the blocking session" not in md


def test_ui_viewed_is_posted_once(make_client, backend):
    at, _ = analyze_demo(make_client, "DEMO-003")
    at.run()
    viewed = [e for e in backend.events if e["event"] == "ui.viewed"]
    assert len(viewed) == 1 and viewed[0]["ticket_id"] == "DEMO-003"


def test_json_toggle_shows_contract(make_client):
    at, _ = analyze_demo(make_client, "DEMO-001")
    at.toggle[0].set_value(True).run()
    assert len(at.json) == 2


def test_clarify_answer_box_reanalyses(make_client):
    client = make_client("l1")
    at = page("1_Analyze.py", client)
    at.text_input(key="f_title").input("VPN broken")
    at.text_area(key="f_description").input("VPN broken")
    at = click(at, "Analyze")
    assert "CLARIFY" in all_markdown(at)
    at.text_area[1].input("VPN client says gateway not responding when working from home")
    at = click(at, "Send answer and re-analyse")
    assert not at.exception
    md = all_markdown(at)
    assert "RECOMMEND" in md or "ESCALATE" in md
    assert "[KB-0018·" in md


def test_empty_form_is_rejected(make_client):
    at = click(page("1_Analyze.py", make_client("l1")), "Analyze")
    assert any("required" in str(e.value) for e in at.error)
