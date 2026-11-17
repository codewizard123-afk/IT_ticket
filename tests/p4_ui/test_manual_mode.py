"""P4-T6: manual mode exposes no AI output; it records manual.started / manual.decision."""

from __future__ import annotations

import re

import pytest

from tests.p4_ui.helpers import UI_DIR, all_text, click, page

PAGE = "2_Manual_mode.py"
# endpoints that carry AI output; manual mode must never reach them
AI_PATHS = re.compile(r"/(analyze|tickets/[^/]+/clarify|escalations/|audit/|review-queue)")
# words that only appear when AI output is rendered
AI_MARKERS = (
    "CONFIDENCE",
    "RECOMMEND",
    "ESCALATE",
    "CLARIFY",
    "Recommended solution",
    "KEDB match",
    "SIMILAR",
    "DUPLICATE",
    "similarity",
    "Why:",
    "Score",
    "Components",
    "HUMAN APPROVAL",
)


@pytest.fixture
def spy(backend):
    calls: list[tuple[str, str]] = []
    original = backend.handle

    def handle(method, path, **kw):
        calls.append((method, path))
        return original(method, path, **kw)

    backend.handle = handle
    return calls


def run_manual(make_client, role: str = "l1"):
    at = page(PAGE, make_client(role))
    at.selectbox(key="m_ticket").select("T01").run()
    at = click(at, "▶ Start ticket (starts the timer)")
    at.text_input(key="m_tq_T01").input("vpn password").run()
    at.text_input(key="m_kq_T01").input("vpn gateway").run()
    assert not at.exception, at.exception
    return at


def test_page_source_never_uses_ai_endpoints():
    src = (UI_DIR / "pages" / PAGE).read_text(encoding="utf-8")
    for forbidden in ("analyze(", "clarify(", "render_analysis", "escalation_bundle", "audit("):
        assert forbidden not in src


def test_no_ai_output_reachable(make_client, spy, clock):
    at = run_manual(make_client)
    clock.tick(120)
    at = click(at, "Submit decision")
    assert not at.exception
    assert not [p for _, p in spy if AI_PATHS.search(p)], spy
    text = all_text(at) + "\n".join(e.label for e in at.expander)
    for marker in AI_MARKERS:
        assert marker not in text, marker
    assert not at.dataframe  # no score tables at all


def test_ticket_hidden_until_started(make_client, backend):
    at = page(PAGE, make_client("l1"))
    at.selectbox(key="m_ticket").select("T01").run()
    trial = {t["id"]: t for t in backend.trial_tickets}
    assert trial["T01"]["description"] not in all_text(at)
    assert not [e for e in backend.events if e["event"] == "manual.started"]


def test_timer_events_and_decision(make_client, backend, clock):
    at = run_manual(make_client)
    clock.tick(150)
    at.slider[0].set_value(4)
    notes = [t for t in at.text_area if t.label.startswith("Resolution notes")][0]
    notes.input("Reset the VPN credentials")
    at = click(at, "Submit decision")
    started = [e for e in backend.events if e["event"] == "manual.started"]
    decided = [e for e in backend.events if e["event"] == "manual.decision"]
    assert len(started) == 1 and len(decided) == 1
    sid = started[0]["analysis_id"]
    assert decided[0]["analysis_id"] == sid and started[0]["mode"] == "manual"
    dec = backend.decisions[sid]
    assert dec["mode"] == "manual"
    assert dec["resolution_notes"].endswith("Self-reported confidence: 4/5")
    assert not [e for e in backend.events if e["event"] == "decision.submitted"]
    ttd = make_client("l1").metrics_mttr(group_by="mode", metric="ttd")["groups"]
    assert ttd == [{"key": "manual", "n": 1, "mean_s": 150.0, "median_s": 150.0, "p90_s": 150.0}]


def test_kb_owner_cannot_run_manual_trial(make_client):
    at = page(PAGE, make_client("kb_owner"))
    assert any("trial is for" in str(w.value) for w in at.warning)


def test_trial_tickets_available_in_assisted_mode(make_client):
    at = page("1_Analyze.py", make_client("l1"))
    at.selectbox(key="sample_id").select("T07").run()
    at = click(at, "Analyze")
    assert not at.exception
    assert any(s.label.startswith("How confident") for s in at.slider)
