"""P2-T9: ``ai.engine.run`` against the golden expectations (architecture §16).

Runs offline (``GROQ_API_KEY`` unset) on the shared stubs. The ``slow`` test at the bottom runs the
same assertions on P1's real stores once they exist (``make data``).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from ai.engine import AIEngine, run
from ai.llm.groq_client import make_llm_client
from shared.constants import CITABLE_ORIGINS, KEDB_SOLUTION_MAX_CHARS, PRIORITY_LEVEL
from shared.interfaces import Engine
from shared.schemas import (
    AnalysisContract,
    EngineContext,
    EngineResult,
    SanitizedTicket,
    TicketRecord,
)
from shared.stubs import FakeKBStore, FakeLLM, FakeTicketStore

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "shared" / "contract" / "analysis_output.schema.json").read_text(encoding="utf-8")
)


def ctx(kb_store, ticket_store, llm=None, role="l2") -> EngineContext:
    return EngineContext(
        role=role,
        ticket_store=ticket_store,
        kb_store=kb_store,
        llm=llm if llm is not None else FakeLLM(available=False),
    )


def contract_from(ticket_id: str, er: EngineResult) -> dict:
    """Minimal contract view of the primary issue (P3's assemble.py does the real thing)."""
    p = er.issues[0]
    top = p.kb_matches[0] if p.kb_matches else None
    found = top is not None and p.kb_relevance >= 0.5
    sol = None
    if found:
        sol = top.excerpt
    worst = min((i.priority.code for i in er.issues), key=lambda c: int(c[1]))
    return AnalysisContract(
        ticket_id=ticket_id,
        classification={
            "category": p.classification.category,
            "subcategory": p.classification.subcategory,
            "confidence": p.classification.confidence,
        },
        priority={"level": PRIORITY_LEVEL[worst], "reason": p.priority.reason},
        similar_incidents=[
            {"incident_id": h.ticket_id, "similarity_score": h.score, "resolution": h.resolution}
            for h in p.similar
        ],
        kedb_match={
            "found": found,
            "error_code": top.kb_id if found else None,
            "solution": sol[:KEDB_SOLUTION_MAX_CHARS] if sol else None,
        },
        recommended_solution={
            "steps": [s.text for s in p.recommendation.steps],
            "source": p.recommendation.source,
        },
        escalation_required=False,
        human_approval_required=True,
    ).model_dump()


@pytest.fixture(scope="module")
def engine_results(demos) -> dict[str, EngineResult]:
    kb, ts = FakeKBStore(), FakeTicketStore()
    return {d: run(t, ctx(kb, ts)) for d, t in demos.items()}


def test_engine_satisfies_protocol() -> None:
    assert isinstance(AIEngine(), Engine)


@pytest.mark.parametrize("demo", [f"DEMO-00{i}" for i in range(1, 8)])
def test_golden_primary_issue(engine_results, golden, demo) -> None:
    er, g = engine_results[demo], golden[demo]
    primary = er.issues[0]
    assert primary.classification.category == g["classification"]["category"]
    assert primary.classification.subcategory == g["classification"]["subcategory"]
    assert primary.kb_matches[0].kb_id == g["kedb_match"]["error_code"]
    worst = min((i.priority.code for i in er.issues), key=lambda c: int(c[1]))
    assert PRIORITY_LEVEL[worst] == g["priority"]["level"]
    assert len(er.issues) == 1 + len(g["details"]["secondary_issues"])


def test_golden_demo_005_secondary(engine_results, golden) -> None:
    er, g = engine_results["DEMO-005"], golden["DEMO-005"]
    sec, gsec = er.issues[1], g["details"]["secondary_issues"][0]
    assert sec.classification.subcategory == gsec["classification"]["subcategory"]
    assert sec.kb_matches[0].kb_id == gsec["kedb_match"]["error_code"]
    assert sec.priority.level == gsec["priority"]["level"]
    cross = [s for s in er.issues[0].recommendation.steps if getattr(s, "cross_link", False)]
    assert [s.citations for s in cross] == [["KB-0019#diag.1"]]


@pytest.mark.parametrize("demo", [f"DEMO-00{i}" for i in range(1, 8)])
def test_every_step_is_cited_and_resolvable(engine_results, kb_store, ticket_store, demo):
    for issue in engine_results[demo].issues:
        rec = issue.recommendation
        assert rec.steps and rec.source != "None"
        for step in rec.steps:
            assert step.citations
            for c in step.citations:
                if "#" in c:
                    assert kb_store.chunk_text(c)
                else:
                    [record] = ticket_store.get([c])
                    assert record.resolution_origin in CITABLE_ORIGINS


@pytest.mark.parametrize("demo", [f"DEMO-00{i}" for i in range(1, 8)])
def test_contract_view_validates_against_schema(engine_results, demo) -> None:
    jsonschema.validate(contract_from(demo, engine_results[demo]), SCHEMA)


def test_degraded_flags_and_timings_offline(engine_results) -> None:
    er = engine_results["DEMO-005"]
    assert {"llm_classify_fallback", "llm_split_fallback", "llm_synth_fallback", "offline"} <= set(
        er.degraded
    )
    assert {"signals", "classify", "retrieve", "synthesize", "total"} <= set(er.timings_ms)
    assert er.versions["pipeline"] == "2.0.0" and er.versions["engine"].startswith("ai.engine")


def test_runs_with_real_groq_client_in_offline_mode(demos, kb_store, ticket_store, tmp_path):
    llm = make_llm_client(cache_path=tmp_path / "c.sqlite")  # no GROQ_API_KEY in tests
    er = run(demos["DEMO-001"], ctx(kb_store, ticket_store, llm=llm))
    assert er.issues[0].kb_matches[0].kb_id == "KB-0018"
    assert "offline" in er.degraded


def test_llm_online_path_has_no_degraded_flags(demos, kb_store, ticket_store) -> None:
    llm = FakeLLM(
        {
            "classify": {
                "issues": [
                    {
                        "text_span": "",
                        "category": "Database",
                        "subcategory": "MS SQL Server",
                        "intent": "incident",
                        "scores": {"Database/MS SQL Server": 0.95},
                    }
                ]
            },
            "synthesize": {
                "steps": [
                    {
                        "type": "diagnostic",
                        "text": "Confirm the exact login name and the database",
                        "citations": ["KB-0003#diag.1"],
                    },
                    {
                        "type": "resolution",
                        "text": "Ask the DBA to create the login and map it to the approved role",
                        "citations": ["KB-0003#res.1"],
                    },
                ]
            },
        }
    )
    er = run(demos["DEMO-007"], ctx(kb_store, ticket_store, llm=llm))
    assert er.degraded == []
    p = er.issues[0]
    assert p.classification.band == "high" and p.classification.confidence > 0.8
    assert [s.role for s in p.recommendation.steps] == ["L1", "L2"]
    assert [c["purpose"] for c in llm.calls] == ["classify", "synthesize"]


def test_clarification_is_used(kb_store, ticket_store) -> None:
    t = SanitizedTicket(ticket_id="X-1", title="Database problem", description="It is broken")
    c = ctx(kb_store, ticket_store)
    c.clarification = "Oracle reports fail with ORA-01013 for 20 users"
    er = run(t, c)
    assert er.issues[0].classification.subcategory == "Oracle"
    assert "ORA-01013" in er.signals.error_codes


def test_vague_ticket_has_low_confidence(kb_store, ticket_store) -> None:
    t = SanitizedTicket(ticket_id="X-2", title="Help", description="Something is wrong today")
    er = run(t, ctx(kb_store, ticket_store))
    assert er.issues[0].classification.band == "low"


def test_ticket_itself_is_not_its_own_similar_incident(demos, kb_store) -> None:
    ts = FakeTicketStore()
    t = demos["DEMO-001"]
    ts.upsert_live(TicketRecord(ticket_id=t.ticket_id, title=t.title, description=t.description))
    er = run(t, ctx(kb_store, ts))
    ids = {h.ticket_id for h in er.issues[0].similar + er.issues[0].keyword_only}
    assert t.ticket_id not in ids


def test_l1_role_never_gets_restricted_tickets(demos, kb_store) -> None:
    ts = FakeTicketStore()
    t = demos["DEMO-001"]
    ts.upsert_live(
        TicketRecord(
            ticket_id="SECRET-1",
            title=t.title,
            description=t.description,
            category="Network",
            subcategory="VPN",
            restricted=True,
        )
    )
    l1 = run(t, ctx(kb_store, ts, role="l1"))
    ids = {h.ticket_id for i in l1.issues for h in i.similar + i.keyword_only}
    assert "SECRET-1" not in ids


@pytest.mark.slow
def test_golden_on_real_stores(demos, golden) -> None:  # pragma: no cover - needs `make data`
    chroma = pytest.importorskip("store.chroma_store")
    ticket_store = chroma.ChromaTicketStore()
    kb_store = chroma.ChromaKBStore()
    for demo, t in demos.items():
        er = run(t, ctx(kb_store, ticket_store))
        assert (
            er.issues[0].classification.subcategory == golden[demo]["classification"]["subcategory"]
        )
        assert er.issues[0].kb_matches[0].kb_id == golden[demo]["kedb_match"]["error_code"]
