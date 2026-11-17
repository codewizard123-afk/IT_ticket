"""P3-T4: human approval rule (§8.5) and response assembly (§4). Output validates the contract."""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest

from backend import approval, assemble, escalate
from backend import decision as gate
from shared.constants import CONTRACT_KEYS
from shared.schemas import AnalysisResponse, EngineResult, SanitizedTicket
from shared.stubs import FakeEngine, FakeKBStore, load_golden
from tests.p3_backend.conftest import DEMO_IDS, EXPECTED, gate_for, golden_engine_result


def _lookup(kb: FakeKBStore):
    def get(kb_id: str):
        try:
            return kb.get_article(kb_id)
        except KeyError:
            return None

    return get


def assemble_demo(
    er: EngineResult, ticket: dict[str, Any], ticket_id: str, rnd: int = 0
) -> AnalysisResponse:
    g = gate_for(ticket, er, rnd)
    decisions = [gate.decide(i, g) for i in er.issues]
    escs = [
        escalate.route(i, d, n) for n, (i, d) in enumerate(zip(er.issues, decisions, strict=True))
    ]
    level = assemble.most_severe(er.issues).priority.level
    appr = approval.required(er.issues, gate.ticket_decision(decisions), level, er.degraded)
    inp = assemble.AssemblyInput(ticket_id, "a-1", er, decisions, escs, appr)
    return assemble.build(inp, _lookup(FakeKBStore()))


@pytest.mark.parametrize("ticket_id", DEMO_IDS)
def test_demo_response_matches_golden_and_contract(
    ticket_id: str, demo_tickets: dict[str, dict[str, Any]], contract_schema: dict
) -> None:
    er = golden_engine_result(ticket_id)
    resp = assemble_demo(er, demo_tickets[ticket_id], ticket_id)
    golden = load_golden(ticket_id)
    assert golden is not None
    body = resp.model_dump(mode="json")
    jsonschema.validate(body, contract_schema)
    contract = resp.to_contract().model_dump(mode="json")
    jsonschema.validate(contract, contract_schema)
    assert tuple(contract) == CONTRACT_KEYS

    assert resp.classification.category == golden.classification.category
    assert resp.classification.subcategory == golden.classification.subcategory
    assert resp.kedb_match.error_code == golden.kedb_match.error_code
    assert resp.details.decision == golden.details.decision
    assert resp.escalation_required is EXPECTED[ticket_id]["esc"]
    assert resp.human_approval_required is EXPECTED[ticket_id]["approval"]
    assert resp.priority.level == golden.priority.level


def test_demo_005_secondary_issue_carries_its_escalation(
    demo_tickets: dict[str, dict[str, Any]],
) -> None:
    resp = assemble_demo(golden_engine_result("DEMO-005"), demo_tickets["DEMO-005"], "DEMO-005")
    sec = resp.details.secondary_issues[0]
    assert sec.decision == "escalate" and sec.escalation is not None
    assert resp.details.escalation is not None and resp.details.escalation.issue_idx == 1
    assert resp.details.primary_decision == "recommend"
    assert sec.classification.subcategory == "CPU"


def test_demo_003_needs_no_approval(demo_tickets: dict[str, dict[str, Any]]) -> None:
    resp = assemble_demo(golden_engine_result("DEMO-003"), demo_tickets["DEMO-003"], "DEMO-003")
    assert resp.details.approval_reasons == []


def test_approval_reasons_follow_golden_format(demo_tickets: dict[str, dict[str, Any]]) -> None:
    resp = assemble_demo(golden_engine_result("DEMO-002"), demo_tickets["DEMO-002"], "DEMO-002")
    assert resp.details.approval_reasons[:3] == [
        "decision escalate",
        "priority Critical",
        "step role L2",
    ]


def test_generic_result_is_schema_valid_and_needs_approval(contract_schema: dict) -> None:
    t = SanitizedTicket(ticket_id="X-1", title="PC broken", description="it does not work")
    er = FakeEngine._generic(t)
    ticket = {"title": t.title, "description": t.description}
    resp = assemble_demo(er, ticket, "X-1")
    jsonschema.validate(resp.model_dump(mode="json"), contract_schema)
    assert resp.details.decision == "clarify"
    assert resp.kedb_match.found is False and resp.kedb_match.error_code is None
    assert resp.recommended_solution.source == "None"
    assert {"decision clarify", "confidence band low", "source None"} <= set(
        resp.details.approval_reasons
    )
    assert 1 <= len(resp.details.clarifying_questions) <= 3


def test_degraded_requires_approval() -> None:
    er = golden_engine_result("DEMO-003")
    ok, reasons = approval.required(er.issues, "recommend", "Medium", ["offline"])
    assert ok and reasons == ["degraded: offline"]


def test_similarity_scores_are_clamped_and_rounded() -> None:
    from shared.schemas import TicketHit

    out = assemble.similar_incidents([TicketHit(ticket_id="a", score=1.2345)])
    assert out[0].similarity_score == 1.0


def test_ticket_priority_is_most_severe_issue(demo_tickets: dict[str, dict[str, Any]]) -> None:
    er = golden_engine_result("DEMO-005")
    swapped = er.model_copy(update={"issues": [er.issues[1], er.issues[0]]})
    resp = assemble_demo(swapped, demo_tickets["DEMO-005"], "DEMO-005")
    assert resp.priority.level == "High"  # the VPN issue, even though it is listed second
