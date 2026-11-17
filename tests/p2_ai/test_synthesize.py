"""P2-T7: resolution synthesiser. Every step has ≥ 1 citation; fallback works offline."""

from __future__ import annotations

import pytest
import yaml

from ai.kb_search import search_articles
from ai.signals import extract
from ai.synthesize import (
    EvidencePack,
    build_pack,
    recommend,
    role_for,
    source_for,
    template_recommendation,
)
from ai.ticket_search import search_similar
from ai.types import ClassificationResult, Recommendation, SimilarIncident, Step
from tests.p2_ai.fakes import (
    FIXTURES,
    FakeKBStore,
    FakeLLM,
    FakeTicketStore,
    load_fixture,
    offline_llm,
)

DEMOS = {d["id"]: d for d in load_fixture("demo_tickets.json")}
TMAP = yaml.safe_load((FIXTURES / "taxonomy_map.yaml").read_text(encoding="utf-8"))
LABELS = {
    "DEMO-001": ("Network", "VPN"),
    "DEMO-003": ("Hardware", "Monitor"),
    "DEMO-004": ("Software", "Email"),
    "DEMO-007": ("Database", "MS SQL Server"),
}


@pytest.fixture(scope="module")
def kb():
    return FakeKBStore()


@pytest.fixture(scope="module")
def ts():
    return FakeTicketStore()


def ctx(demo_id, kb, ts):
    d = DEMOS[demo_id]
    text = f"{d['title']}. {d['description']}"
    cat, sub = LABELS[demo_id]
    sig = extract(text, kb.all_error_phrases())
    kres = search_articles(
        text, kb, error_codes=sig.error_codes, category=cat, subcategory=sub, tmap=TMAP
    )
    sim = search_similar(text, ts, ClassificationResult(cat, sub, 0.9, "high")).similar
    return text, sig, kres, sim, sub


@pytest.mark.parametrize("demo_id", sorted(LABELS))
def test_offline_fallback_every_step_cited(demo_id, kb, ts):
    text, sig, kres, sim, sub = ctx(demo_id, kb, ts)
    rec, pack, degraded = recommend(
        text, sig, kres, sim, kb, offline_llm(), subcategory=sub
    )
    assert degraded == ["llm_synth_fallback"]
    assert len(rec.steps) >= 2
    citable = pack.citable()
    for s in rec.steps:
        assert s.citations and all(c in citable for c in s.citations)
    types = [s.type for s in rec.steps]
    assert types == sorted(types, key=lambda t: t != "diagnostic")  # diagnostics first
    assert [s.n for s in rec.steps] == list(range(1, len(rec.steps) + 1))
    assert rec.source == "KEDB"


def test_demo_001_restart_marked_attempted(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-001", kb, ts)
    rec, _, _ = recommend(text, sig, kres, sim, kb, offline_llm(), subcategory=sub)
    attempted = [s.text for s in rec.steps if s.already_attempted]
    assert attempted == ["Restart the VPN client"]


def test_demo_003_cable_attempted_and_no_l2(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-003", kb, ts)
    rec, _, _ = recommend(text, sig, kres, sim, kb, offline_llm(), subcategory=sub)
    assert rec.steps[0].already_attempted and "cable" in rec.steps[0].text
    assert all(s.role == "L1" for s in rec.steps)
    assert not rec.requires_admin


def test_demo_004_requires_admin_steps_are_l2(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-004", kb, ts)
    rec, _, _ = recommend(text, sig, kres, sim, kb, offline_llm(), subcategory=sub)
    roles = {s.text: s.role for s in rec.steps}
    assert roles["Repair the Office installation from Programs and Features"] == "L2"
    assert roles["Confirm the application starts normally"] == "L1"
    assert roles["Start the application in safe mode"] == "L1"
    assert rec.requires_admin


def test_demo_007_dba_step_is_l2(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-007", kb, ts)
    rec, _, _ = recommend(text, sig, kres, sim, kb, offline_llm(), subcategory=sub)
    assert any(s.role == "L2" and "DBA" in s.text for s in rec.steps)


def test_pack_excludes_synthetic_and_other_subcategories(kb, ts):
    _text, _sig, kres, sim, sub = ctx("DEMO-003", kb, ts)
    pack = build_pack(kres, sim, kb, subcategory=sub)
    assert [i.id for i in pack.incidents] == ["INC-H0003"]
    assert all(not c.startswith("TKT-") for c in pack.citable())
    fake = [
        SimilarIncident(
            "TKT-9",
            0.99,
            "Resolved via KB-0008: x",
            "SIMILAR",
            "x",
            "y",
            "synthetic_kb_derived",
            "Hardware",
            "Monitor",
        ),
        SimilarIncident(
            "INC-H9",
            0.99,
            "Fixed it",
            "SIMILAR",
            "x",
            "y",
            "curated_verified",
            "Hardware",
            "Mouse",
        ),
        SimilarIncident(
            "INC-H8",
            0.60,
            "Fixed it",
            "RELATED",
            "x",
            "y",
            "analyst_verified",
            "Hardware",
            "Monitor",
        ),
    ]
    assert build_pack(kres, fake, kb, subcategory="Monitor").incidents == []


def test_pack_limits(kb, ts):
    _text, _sig, kres, sim, sub = ctx("DEMO-007", kb, ts)
    pack = build_pack(kres, sim, kb, subcategory=sub)
    assert len(pack.articles) <= 2 and len(pack.incidents) <= 3


def test_empty_pack_gives_insufficient_evidence(kb):
    sig = extract("printer is jammed", [])
    rec, pack, degraded = recommend(
        "printer is jammed",
        sig,
        search_articles("zzz", kb, tmap={}),
        [],
        kb,
        offline_llm(),
        subcategory=None,
    )
    assert (
        pack.empty
        and rec.insufficient_evidence
        and rec.source == "None"
        and not rec.steps
    )
    assert degraded == []


def test_incident_only_pack_uses_historical_source(kb):
    from ai.types import KBSearchResult

    sim = [
        SimilarIncident(
            "INC-H0006",
            0.8,
            "NET-OPS extended the exhausted DHCP scope for the floor VLAN; clients renewed their lease.",
            "SIMILAR",
            "Laptops on one floor do not receive an IP",
            "why",
            "curated_verified",
            "Network",
            "DHCP",
        )
    ]
    rec, _, _ = recommend(
        "no ip from dhcp",
        extract("no ip from dhcp", []),
        KBSearchResult(),
        sim,
        kb,
        offline_llm(),
        subcategory="DHCP",
    )
    assert rec.source == "Historical Incident"
    assert rec.steps[0].citations == ["INC-H0006"]


GOOD_LLM = {
    "summary": "VPN gateway unreachable; use the alternate gateway.",
    "probable_cause": {
        "text": "assigned gateway unreachable",
        "citations": ["KB-0018#diag.4"],
    },
    "steps": [
        {
            "n": 1,
            "type": "resolution",
            "text": "Connect to the alternate VPN gateway listed in the client profile",
            "citations": ["KB-0018#res.1"],
            "role": "L1",
            "already_attempted": False,
        },
        {
            "n": 2,
            "type": "diagnostic",
            "text": "Restart the VPN client",
            "citations": ["KB-0018#diag.2"],
            "role": "L1",
            "already_attempted": False,
        },
        {
            "n": 3,
            "type": "diagnostic",
            "text": "Check internet connectivity without the VPN",
            "citations": ["KB-0018#diag.1"],
            "role": "L1",
            "already_attempted": False,
        },
    ],
    "alternatives": [],
    "estimated_time": "30 minutes",
    "requires_admin": False,
    "insufficient_evidence": False,
}


def test_llm_path(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-001", kb, ts)
    llm = FakeLLM({"synthesize": GOOD_LLM})
    rec, _, degraded = recommend(text, sig, kres, sim, kb, llm, subcategory=sub)
    assert degraded == []
    assert [s.type for s in rec.steps] == ["diagnostic", "diagnostic", "resolution"]
    restart = next(s for s in rec.steps if s.text == "Restart the VPN client")
    assert (
        restart.already_attempted
    )  # deterministic mark even though the LLM said False
    call = llm.calls[0]
    assert call["purpose"] == "synthesize"
    user = call["messages"][1]["content"]
    assert "KB-0018#res.1" in user and "<ticket>" in user
    assert "TKT-" not in user  # synthetic incidents never reach the pack


def test_llm_bad_output_retries_with_reason_then_falls_back(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-001", kb, ts)
    llm = FakeLLM({"synthesize": {"steps": "not a list"}})
    rec, _, degraded = recommend(text, sig, kres, sim, kb, llm, subcategory=sub)
    assert len(llm.calls) == 2
    assert "rejected" in llm.calls[1]["messages"][1]["content"]
    assert degraded == ["llm_synth_fallback"]
    assert rec.steps and all(s.citations for s in rec.steps)


def test_validator_failures_trigger_template(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-001", kb, ts)
    seen = []

    def strict(rec: Recommendation, pack: EvidencePack) -> Recommendation:
        seen.append(len(rec.steps))
        if rec.summary.startswith("Follow"):  # template output passes
            return rec
        return Recommendation(steps=[], source="None", insufficient_evidence=True)

    llm = FakeLLM({"synthesize": GOOD_LLM})
    rec, _, degraded = recommend(
        text, sig, kres, sim, kb, llm, subcategory=sub, validator=strict
    )
    assert len(llm.calls) == 2 and degraded == ["llm_synth_fallback"]
    assert rec.summary.startswith("Follow KB-0018")


def test_llm_insufficient_evidence_respected(kb, ts):
    text, sig, kres, sim, sub = ctx("DEMO-001", kb, ts)
    llm = FakeLLM({"synthesize": {"insufficient_evidence": True}})
    rec, _, degraded = recommend(text, sig, kres, sim, kb, llm, subcategory=sub)
    assert rec.insufficient_evidence and rec.source == "None" and degraded == []


def test_source_for():
    kb_step = Step("a", "diagnostic", ["KB-0001#diag.1"])
    inc_step = Step("b", "resolution", ["INC-H0001"])
    assert source_for([kb_step]) == "KEDB"
    assert source_for([inc_step]) == "Historical Incident"
    assert source_for([kb_step, inc_step]) == "KEDB + Historical Incident"
    assert source_for([]) == "None"


def test_role_for():
    assert (
        role_for("Ask the DBA to end the blocking session", "resolution", False) == "L2"
    )
    assert role_for("Update the BIOS and chipset drivers", "resolution", False) == "L2"
    assert role_for("Confirm the VPN stays connected", "resolution", True) == "L1"
    assert role_for("Roll back the display driver", "resolution", False) == "L1"
    assert role_for("Create a new Outlook profile", "resolution", True) == "L2"


def test_template_without_signals_attempts(kb, ts):
    _text, _sig, kres, sim, sub = ctx("DEMO-004", kb, ts)
    pack = build_pack(kres, sim, kb, subcategory=sub)
    rec = template_recommendation(pack, extract("Outlook crashes", []))
    assert not any(s.already_attempted for s in rec.steps)
    assert rec.probable_cause["citations"] == ["KB-0029#title"]
