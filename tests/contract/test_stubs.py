"""The fakes in shared/stubs.py satisfy the Protocols and behave as README §8.1 describes."""

from __future__ import annotations

import pytest

from shared.constants import TAXONOMY
from shared.interfaces import Engine, KBStore, LLMClient, LLMUnavailableError, Redactor, TicketStore
from shared.schemas import EngineContext, SanitizedTicket, TicketRecord
from shared.stubs import (
    FakeEngine,
    FakeKBStore,
    FakeLLM,
    FakeTicketStore,
    fake_redact,
    golden_ids,
    load_golden,
)


@pytest.fixture(scope="module")
def kb() -> FakeKBStore:
    return FakeKBStore()


@pytest.fixture
def tickets() -> FakeTicketStore:
    return FakeTicketStore()


def _ctx(kb: FakeKBStore, tickets: FakeTicketStore) -> EngineContext:
    return EngineContext(role="l1", ticket_store=tickets, kb_store=kb, llm=FakeLLM())


def test_fakes_satisfy_protocols(kb: FakeKBStore, tickets: FakeTicketStore) -> None:
    assert isinstance(tickets, TicketStore)
    assert isinstance(kb, KBStore)
    assert isinstance(FakeLLM(), LLMClient)
    assert isinstance(FakeEngine(kb), Engine)
    assert isinstance(fake_redact, Redactor)


# --- FakeKBStore -----------------------------------------------------------------------------
def test_kb_has_all_30_articles(kb: FakeKBStore) -> None:
    assert kb.kb_ids() == [f"KB-{i:04d}" for i in range(1, 31)]


def test_kb_chunk_ids_and_resolution_not_searched(kb: FakeKBStore) -> None:
    assert kb.chunk_text("KB-0018#res.1") == "Restart the VPN client and reconnect"
    hits = kb.query_chunks("restart the VPN client and reconnect", k=50)
    assert hits and all(h.chunk_type != "resolution" for h in hits)
    assert hits[0].kb_id in {"KB-0018", "KB-0019"}


def test_kb_lookup_codes_is_normalised(kb: FakeKBStore) -> None:
    assert kb.lookup_codes(["gateway  NOT responding", "ORA-01013", "nope"]) == {
        "gateway  NOT responding": ["KB-0018"],
        "ORA-01013": ["KB-0001"],
    }
    assert set(kb.lookup_codes(["18456"])["18456"]) == {"KB-0003", "KB-0004"}
    assert "Gateway not responding" in kb.all_error_phrases()


def test_kb_where_filter(kb: FakeKBStore) -> None:
    hits = kb.query_chunks("slow", k=20, where={"kb_id": "KB-0001"})
    assert hits and {h.kb_id for h in hits} == {"KB-0001"}


# --- FakeTicketStore -------------------------------------------------------------------------
def test_ticket_store_covers_every_subcategory(tickets: FakeTicketStore) -> None:
    ids = [f"TKT-{i:04d}" for i in range(1, 37)] + [f"INC-H{i:04d}" for i in range(1, 15)]
    recs = tickets.get(ids)
    assert len(recs) == 50
    subs = {r.subcategory for r in recs}
    assert subs == {s for v in TAXONOMY.values() for s in v}


def test_ticket_query_filters_and_collections(tickets: FakeTicketStore) -> None:
    hits = tickets.query("VPN gateway not responding", k=5)
    assert hits and hits[0].subcategory == "VPN"
    verified = tickets.query("VPN", k=10, collections=("tickets_verified",))
    assert verified and all(h.resolution_origin == "curated_verified" for h in verified)
    filtered = tickets.query("slow", k=10, where={"category": {"$eq": "Database"}})
    assert all(h.category == "Database" for h in filtered)


def test_upsert_live_and_mark_verified(tickets: FakeTicketStore) -> None:
    rec = TicketRecord(ticket_id="DEMO-001", title="VPN down", description="VPN gateway down")
    tickets.upsert_live(rec)
    assert tickets.query("VPN gateway", k=50, collections=("tickets_live",))[0].ticket_id == (
        "DEMO-001"
    )
    tickets.mark_verified("DEMO-001", "Switched gateway", by="l2_bob")
    [got] = tickets.get(["DEMO-001"])
    assert (got.collection, got.resolution_origin, got.verified_by) == (
        "tickets_verified",
        "analyst_verified",
        "l2_bob",
    )


# --- FakeLLM ---------------------------------------------------------------------------------
def test_fake_llm_canned_and_offline() -> None:
    llm = FakeLLM({"classify": {"x": 1}})
    assert llm.complete_json(purpose="classify", system="s", user="u") == {"x": 1}
    assert llm.calls[0]["purpose"] == "classify"
    with pytest.raises(LLMUnavailableError):
        FakeLLM(available=False).complete_json(purpose="classify", system="s", user="u")


# --- FakeEngine ------------------------------------------------------------------------------
@pytest.mark.parametrize("demo", golden_ids())
def test_fake_engine_replays_golden(kb: FakeKBStore, tickets: FakeTicketStore, demo: str) -> None:
    golden = load_golden(demo)
    assert golden is not None
    er = FakeEngine(kb).run(
        SanitizedTicket(ticket_id=demo, title="t", description="d"), _ctx(kb, tickets)
    )
    primary = er.issues[0]
    assert primary.classification.category == golden.classification.category
    assert primary.classification.subcategory == golden.classification.subcategory
    assert primary.kb_matches[0].kb_id == golden.kedb_match.error_code
    assert [s.text for s in primary.recommendation.steps] == golden.recommended_solution.steps
    assert primary.recommendation.source == golden.recommended_solution.source
    assert len(er.issues) == 1 + len(golden.details.secondary_issues)


def test_fake_engine_demo_005_has_two_issues(kb: FakeKBStore, tickets: FakeTicketStore) -> None:
    er = FakeEngine(kb).run(
        SanitizedTicket(ticket_id="DEMO-005", title="t", description="d"), _ctx(kb, tickets)
    )
    assert [i.classification.subcategory for i in er.issues] == ["VPN", "CPU"]
    assert [m.kb_id for m in er.issues[1].kb_matches[:2]] == ["KB-0030", "KB-0006"]


def test_fake_engine_unknown_ticket_is_low_confidence(
    kb: FakeKBStore, tickets: FakeTicketStore
) -> None:
    er = FakeEngine(kb).run(
        SanitizedTicket(ticket_id="INC-X", title="hello", description="something odd"),
        _ctx(kb, tickets),
    )
    issue = er.issues[0]
    assert issue.classification.band == "low"
    assert issue.recommendation.source == "None"
    assert issue.recommendation.insufficient_evidence is True


# --- fake_redact -----------------------------------------------------------------------------
def test_fake_redact_email_and_secret_but_not_error_codes() -> None:
    text = "Mail a.b@corp.com and a.b@corp.com; password=hunter2; ORA-01013 0xc0000005"
    res = fake_redact(text)
    assert "a.b@corp.com" not in res.text and res.text.count("<EMAIL_1>") == 2
    assert "hunter2" not in res.text and res.secret_detected
    assert "ORA-01013" in res.text and "0xc0000005" in res.text
