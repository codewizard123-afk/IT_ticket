"""P3-T3: escalation routing, tiers and the diagnostic bundle (architecture §8.2)."""

from __future__ import annotations

from typing import Any

import pytest

from backend import decision as gate
from backend import escalate
from shared.schemas import EscalationReason, IssueResult
from tests.p3_backend.conftest import EXPECTED, gate_for, golden_engine_result


@pytest.mark.parametrize("ticket_id", ["DEMO-002", "DEMO-005", "DEMO-006"])
def test_demo_routing_and_tier(ticket_id: str, demo_tickets: dict[str, dict[str, Any]]) -> None:
    er = golden_engine_result(ticket_id)
    g = gate_for(demo_tickets[ticket_id], er)
    escs = [
        escalate.route(issue, gate.decide(issue, g), i, new_id=lambda: "b-1")
        for i, issue in enumerate(er.issues)
    ]
    esc = next(e for e in escs if e)
    assert esc.assignment_group == EXPECTED[ticket_id]["group"]
    assert esc.tier == EXPECTED[ticket_id]["tier"]
    assert esc.bundle_id == "b-1"


def test_no_escalation_for_recommend(demo_tickets: dict[str, dict[str, Any]]) -> None:
    er = golden_engine_result("DEMO-001")
    d = gate.decide(er.issues[0], gate_for(demo_tickets["DEMO-001"], er))
    assert escalate.route(er.issues[0], d) is None


def _decision(*rules: str) -> gate.IssueDecision:
    return gate.IssueDecision("escalate", [EscalationReason(rule=r, text=r) for r in rules])


def _with_category(issue: IssueResult, category: str) -> IssueResult:
    cls = issue.classification.model_copy(update={"category": category})
    return issue.model_copy(update={"classification": cls})


def test_security_overrides_group_and_is_l3() -> None:
    issue = golden_engine_result("DEMO-001").issues[0]
    esc = escalate.route(issue, _decision("E5"))
    assert esc is not None and esc.assignment_group == "SEC-OPS" and esc.tier == "L3"


def test_network_major_incident_group() -> None:
    issue = golden_engine_result("DEMO-001").issues[0]
    esc = escalate.route(issue, _decision("E4"))
    assert esc is not None and esc.assignment_group == "NET-OPS-MAJOR"


def test_soft_only_is_l2_category_group() -> None:
    issue = _with_category(golden_engine_result("DEMO-003").issues[0], "Software")
    esc = escalate.route(issue, _decision("E1"))
    assert esc is not None and esc.assignment_group == "EUC-SW" and esc.tier == "L2"


def test_bundle_json_and_markdown() -> None:
    response = {
        "ticket_id": "T-1",
        "classification": {"category": "Network", "subcategory": "VPN", "confidence": 0.9},
        "priority": {"level": "High", "reason": "r"},
        "similar_incidents": [{"incident_id": "INC-1", "similarity_score": 0.9, "resolution": "x"}],
        "details": {
            "signals": {"scope": "single"},
            "classification": {"components": {"llm": 0.9}},
            "priority": {"drivers": ["d"]},
            "kedb": {"kb_id": "KB-0018", "title": "VPN", "score": 0.8, "alternatives": []},
            "steps": [
                {"text": "tried", "already_attempted": True},
                {"text": "next", "already_attempted": False},
            ],
            "escalation": {
                "reasons": [{"rule": "E6", "text": "KB-0018: many users"}],
                "assignment_group": "NET-OPS",
                "tier": "L2",
                "bundle_id": "b-9",
                "issue_idx": 0,
            },
            "secondary_issues": [],
        },
    }
    ticket = {"title": "VPN down", "description_redacted": "email <EMAIL_1>"}
    b = escalate.build_bundle(response, ticket, "b-9", "2026-01-01T00:00:00Z")
    assert b is not None
    assert b["attempted_steps"] == ["tried"] and b["remaining_steps"] == ["next"]
    assert b["priority_drivers"] == ["d"] and b["kb_matches"][0]["kb_id"] == "KB-0018"
    md = escalate.bundle_markdown(b)
    assert "# Escalation bundle b-9" in md and "E6: KB-0018: many users" in md
    assert escalate.build_bundle(response, ticket, "nope", "t") is None
