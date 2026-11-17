"""P3-T3: decision gate (architecture §8.2-§8.4). Demo decisions match §16; clarify terminates."""

from __future__ import annotations

from typing import Any

import pytest

from backend import decision as gate
from backend.config import kb_clauses
from shared.schemas import (
    Alternative,
    ClassificationResult,
    IssueResult,
    KBHit,
    PriorityResult,
    Recommendation,
    SanitizedTicket,
    Signals,
    Step,
    TicketHit,
)
from shared.stubs import FakeEngine
from tests.p3_backend.conftest import DEMO_IDS, EXPECTED, gate_for, golden_engine_result


def _issue(**kw: Any) -> IssueResult:
    base: dict[str, Any] = {
        "text_span": "x",
        "classification": ClassificationResult(
            category="Network", subcategory="VPN", confidence=0.9, band="high"
        ),
        "priority": PriorityResult(code="P3", level="Medium", reason="r", impact=3, urgency=2),
        "kb_matches": [KBHit(kb_id="KB-0018", title="VPN", score=0.8, subcategory="VPN")],
        "kb_relevance": 0.8,
        "recommendation": Recommendation(
            steps=[Step(text="do it", type="resolution", citations=["KB-0018#res.1"])],
            source="KEDB",
        ),
    }
    base.update(kw)
    return IssueResult(**base)


def _gate(text: str = "a reasonably long ticket text " * 3, **kw: Any) -> gate.GateInput:
    sig = kw.pop("signals", Signals(products=["VPN client"], error_codes=["E1"], onset="today"))
    return gate.GateInput(signals=sig, text=text, **kw)


@pytest.mark.parametrize("ticket_id", DEMO_IDS)
def test_demo_decisions_match_architecture_16(
    ticket_id: str, demo_tickets: dict[str, dict[str, Any]]
) -> None:
    er = golden_engine_result(ticket_id)
    g = gate_for(demo_tickets[ticket_id], er)
    got = [gate.decide(issue, g).decision for issue in er.issues]
    assert got == EXPECTED[ticket_id]["issues"]


def test_demo_002_escalates_on_hard_rule_e4(demo_tickets: dict[str, dict[str, Any]]) -> None:
    er = golden_engine_result("DEMO-002")
    d = gate.decide(er.issues[0], gate_for(demo_tickets["DEMO-002"], er))
    assert d.hard and d.rules[0] == "E4"
    assert "E6" in d.rules  # KB-0001: month-end close blocked


def test_demo_005_secondary_fires_kb_0006_clause(demo_tickets: dict[str, dict[str, Any]]) -> None:
    er = golden_engine_result("DEMO-005")
    d = gate.decide(er.issues[1], gate_for(demo_tickets["DEMO-005"], er))
    assert any(r.text == "KB-0006: the laptop overheats" for r in d.reasons)


def test_clauses_cover_every_kb_article() -> None:
    import json

    from tests.p3_backend.conftest import ROOT

    ids = {a["kb_id"] for a in json.loads((ROOT / "data/kb/articles.json").read_text("utf-8"))}
    assert ids <= set(kb_clauses())
    allowed = {
        "scope_in", "min_users", "business_critical", "security", "environment_in",
        "change_trigger", "text_any", "text_all",
    }  # fmt: skip
    for clauses in kb_clauses().values():
        for c in clauses:
            assert c["text"] and set(c["when"]) <= allowed


# --- individual rules --------------------------------------------------------------------------
def test_recommend_when_nothing_fires() -> None:
    assert gate.decide(_issue(), _gate()).decision == "recommend"


def test_e1_low_kb_relevance_without_verified_incident() -> None:
    d = gate.decide(_issue(kb_relevance=0.2, kb_matches=[]), _gate())
    assert d.decision == "escalate" and "E1" in d.rules


def test_e1_suppressed_by_verified_incident() -> None:
    hit = TicketHit(ticket_id="INC-H1", score=0.8, resolution_origin="curated_verified")
    d = gate.decide(_issue(kb_relevance=0.2, kb_matches=[], similar=[hit]), _gate())
    assert "E1" not in d.rules


def test_e1_ignores_synthetic_incidents() -> None:
    hit = TicketHit(ticket_id="T1", score=0.95, resolution_origin="synthetic_kb_derived")
    d = gate.decide(_issue(kb_relevance=0.2, kb_matches=[], similar=[hit]), _gate())
    assert "E1" in d.rules


def test_e3_insufficient_evidence() -> None:
    d = gate.decide(_issue(recommendation=Recommendation(insufficient_evidence=True)), _gate())
    assert "E3" in d.rules and d.decision == "escalate"


def test_e4_priority_p1_is_hard() -> None:
    p1 = PriorityResult(code="P1", level="Critical", reason="r", impact=1, urgency=1)
    d = gate.decide(_issue(priority=p1), _gate(clarification_round=0))
    assert d.decision == "escalate" and d.hard


def test_e4_possible_major_incident() -> None:
    d = gate.decide(_issue(possible_major_incident=True), _gate())
    assert d.rules == ["E4"]


def test_e5_secret_detected_is_hard_and_beats_clarify() -> None:
    d = gate.decide(_issue(kb_ambiguous=True), _gate(secret_detected=True))
    assert d.decision == "escalate" and d.rules[0] == "E5"


def test_e5_security_flags() -> None:
    d = gate.decide(_issue(), _gate(signals=Signals(security_flags=["phishing"])))
    assert "E5" in d.rules


def test_e7_only_l2_steps_left_after_l1_attempted() -> None:
    steps = [
        Step(text="a", type="diagnostic", citations=["KB-1#diag.1"], already_attempted=True),
        Step(text="b", type="resolution", citations=["KB-1#res.1"], role="L2"),
    ]
    d = gate.decide(_issue(recommendation=Recommendation(steps=steps, source="KEDB")), _gate())
    assert d.rules == ["E7"]


def test_e7_not_when_no_l1_steps_exist() -> None:
    steps = [Step(text="b", type="resolution", citations=["KB-1#res.1"], role="L2")]
    d = gate.decide(_issue(recommendation=Recommendation(steps=steps, source="KEDB")), _gate())
    assert "E7" not in d.rules


def test_e6_only_considers_articles_above_tau_show() -> None:
    weak = [KBHit(kb_id="KB-0021", title="t", score=0.2)]
    g = _gate(signals=Signals(scope="multiple", products=["wifi"], error_codes=["x"]))
    assert (
        gate.rule_e6(_issue(kb_matches=weak), g, {"thresholds": {"tau_show": 0.35}}, kb_clauses())
        == []
    )


def test_unknown_predicate_fails_closed() -> None:
    assert gate.clause_holds({"no_such_key": True}, _gate()) is False


# --- clarify ------------------------------------------------------------------------------------
def test_kb_ambiguous_asks_pair_question() -> None:
    matches = [
        KBHit(kb_id="KB-0001", title="a", score=0.6, subcategory="DB performance"),
        KBHit(kb_id="KB-0003", title="b", score=0.58, subcategory="DB access"),
    ]
    d = gate.decide(_issue(kb_ambiguous=True, kb_matches=matches), _gate())
    assert d.decision == "clarify"
    assert d.questions[0] == "Is the problem slowness/timeouts, or an access/login error?"
    assert len(d.questions) <= 3


def test_sparse_ticket_asks_generic_questions() -> None:
    d = gate.decide(_issue(), _gate(text="PC broken", signals=Signals()))
    assert d.decision == "clarify" and "sparse" in d.clarify_triggers
    assert d.questions == [
        "What is the exact error message?",
        "How many users are affected?",
        "When did it start, and did anything change?",
    ]


def test_low_band_with_small_gap_clarifies() -> None:
    cls = ClassificationResult(
        category="Network",
        subcategory="VPN",
        confidence=0.4,
        band="low",
        alternatives=[Alternative(category="Network", subcategory="Wireless", score=0.35)],
    )
    d = gate.decide(_issue(classification=cls), _gate())
    assert d.decision == "clarify" and "low_gap" in d.clarify_triggers


@pytest.mark.parametrize("rnd", [1, 2])
def test_round_one_never_clarifies_so_the_loop_terminates(rnd: int) -> None:
    d = gate.decide(
        _issue(kb_ambiguous=True),
        _gate(text="PC broken", signals=Signals(), clarification_round=rnd),
    )
    assert d.decision != "clarify"


def test_generic_fake_engine_result_clarifies_then_escalates() -> None:
    t = SanitizedTicket(ticket_id="X-1", title="PC broken", description="it does not work")
    er = FakeEngine._generic(t)
    text = f"{t.title}\n{t.description}"
    first = gate.decide(er.issues[0], gate.GateInput(signals=er.signals, text=text))
    second = gate.decide(
        er.issues[0], gate.GateInput(signals=er.signals, text=text, clarification_round=1)
    )
    assert first.decision == "clarify"
    assert second.decision == "escalate"


def test_ticket_decision_precedence() -> None:
    mk = gate.IssueDecision
    assert gate.ticket_decision([mk("recommend"), mk("escalate")]) == "escalate"
    assert gate.ticket_decision([mk("clarify"), mk("recommend")]) == "clarify"
    assert gate.ticket_decision([mk("recommend")]) == "recommend"
