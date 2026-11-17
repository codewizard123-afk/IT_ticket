"""Golden fixtures encode architecture §16 and obey the production rules of §4.2 and §8."""

from __future__ import annotations

import pytest

from shared.constants import CITABLE_ORIGINS, HARD_RULES, KEDB_SOLUTION_MAX_CHARS
from shared.stubs import FakeKBStore, FakeTicketStore

# §16: category, subcategory, kb_id, decision, escalation_required, human_approval_required, level
EXPECTED = {
    "DEMO-001": ("Network", "VPN", "KB-0018", "recommend", False, True, "High"),
    "DEMO-002": ("Database", "Oracle", "KB-0001", "escalate", True, True, "Critical"),
    "DEMO-003": ("Hardware", "Monitor", "KB-0008", "recommend", False, False, "Medium"),
    "DEMO-004": ("Software", "Email", "KB-0029", "recommend", False, True, "High"),
    "DEMO-005": ("Network", "VPN", "KB-0018", "escalate", True, True, "High"),
    "DEMO-006": ("Network", "Wireless", "KB-0021", "escalate", True, True, "High"),
    "DEMO-007": ("Database", "MS SQL Server", "KB-0003", "recommend", False, True, "High"),
}
DEMOS = sorted(EXPECTED)


@pytest.fixture(scope="module")
def kb() -> FakeKBStore:
    return FakeKBStore()


@pytest.fixture(scope="module")
def tickets() -> FakeTicketStore:
    return FakeTicketStore()


def _issues(g: dict) -> list[tuple[dict, list[dict]]]:
    """(contract-shaped issue, its detailed steps) for the primary and any secondary issue."""
    d = g["details"]
    return [(g, d["steps"])] + [(s, s["steps"]) for s in d["secondary_issues"]]


@pytest.mark.parametrize("demo", DEMOS)
def test_matches_architecture_section_16(golden: dict, demo: str) -> None:
    g = golden[demo]
    cat, sub, kb_id, decision, esc, appr, level = EXPECTED[demo]
    assert g["ticket_id"] == demo
    assert (g["classification"]["category"], g["classification"]["subcategory"]) == (cat, sub)
    assert g["kedb_match"]["error_code"] == kb_id
    assert g["details"]["kedb"]["kb_id"] == kb_id
    assert g["details"]["decision"] == decision
    assert g["escalation_required"] is esc
    assert g["human_approval_required"] is appr
    assert g["priority"]["level"] == level


def test_section_16_specifics(golden: dict) -> None:
    # DEMO-001: "restart client" is marked already attempted
    s1 = golden["DEMO-001"]["details"]["steps"]
    assert any(s["already_attempted"] and "Restart the VPN client" in s["text"] for s in s1)
    assert golden["DEMO-001"]["details"]["kedb"]["matched_error_code"] == "Gateway not responding"
    # DEMO-002: E4 hard rule, routed to DBA at L3
    e2 = golden["DEMO-002"]["details"]["escalation"]
    assert "E4" in {r["rule"] for r in e2["reasons"]}
    assert (e2["assignment_group"], e2["tier"]) == ("DBA", "L3")
    assert golden["DEMO-002"]["details"]["kedb"]["matched_error_code"] == "ORA-01013"
    assert golden["DEMO-002"]["details"]["kedb"]["alternatives"][0]["kb_id"] == "KB-0002"
    # DEMO-003: the cable step is marked already attempted
    s3 = golden["DEMO-003"]["details"]["steps"]
    assert any(s["already_attempted"] and "cable" in s["text"] for s in s3)
    # DEMO-004: matched via "Faulting application name"; KB-0030 is the alternative
    k4 = golden["DEMO-004"]["details"]["kedb"]
    assert k4["matched_error_code"] == "Faulting application name"
    assert k4["alternatives"][0]["kb_id"] == "KB-0030"
    # DEMO-005: VPN + CPU; issue 2 escalates on E6 to EUC-HW; the cross-link cites KB-0019 diag
    d5 = golden["DEMO-005"]["details"]
    assert d5["primary_decision"] == "recommend"
    sec = d5["secondary_issues"][0]
    assert (sec["classification"]["category"], sec["classification"]["subcategory"]) == (
        "Hardware",
        "CPU",
    )
    assert sec["kedb_match"]["error_code"] == "KB-0030"
    assert sec["kedb"]["alternatives"][0]["kb_id"] == "KB-0006"
    assert sec["decision"] == "escalate"
    assert sec["escalation"]["assignment_group"] == "EUC-HW"
    assert [r["rule"] for r in sec["escalation"]["reasons"]] == ["E6"]
    assert any(c.startswith("KB-0019#diag") for s in d5["steps"] for c in s["citations"])
    # DEMO-006: E6 to NET-OPS
    e6 = golden["DEMO-006"]["details"]["escalation"]
    assert [r["rule"] for r in e6["reasons"]] == ["E6"]
    assert e6["assignment_group"] == "NET-OPS"
    # DEMO-007: KB-0003 ranks above KB-0004
    k7 = golden["DEMO-007"]["details"]["kedb"]
    assert k7["alternatives"][0]["kb_id"] == "KB-0004"
    assert k7["alternatives"][0]["score"] < k7["score"]


@pytest.mark.parametrize("demo", DEMOS)
def test_steps_are_cited_and_citations_resolve(
    golden: dict, demo: str, kb: FakeKBStore, tickets: FakeTicketStore
) -> None:
    for issue, steps in _issues(golden[demo]):
        assert [s["text"] for s in steps] == issue["recommended_solution"]["steps"]
        kinds: set[str] = set()
        for step in steps:
            assert step["citations"], "every step needs a citation (CLAUDE.md rule 4)"
            kb_cites = [c for c in step["citations"] if c.startswith("KB-")]
            # the step text is the text of its first KB citation (grounding)
            if kb_cites:
                assert kb.chunk_text(kb_cites[0]) == step["text"]
            for cite in step["citations"]:
                if cite.startswith("KB-"):
                    kinds.add("kb")
                    kb.chunk_text(cite)  # KeyError if the chunk does not exist
                else:
                    kinds.add("inc")
                    [rec] = tickets.get([cite])
                    assert rec.resolution_origin in CITABLE_ORIGINS, f"{cite} is not citable"
        expected_source = {
            frozenset(): "None",
            frozenset({"kb"}): "KEDB",
            frozenset({"inc"}): "Historical Incident",
            frozenset({"kb", "inc"}): "KEDB + Historical Incident",
        }[frozenset(kinds)]
        assert issue["recommended_solution"]["source"] == expected_source


@pytest.mark.parametrize("demo", DEMOS)
def test_kedb_match_follows_section_4_2(golden: dict, demo: str, kb: FakeKBStore) -> None:
    for issue, _ in _issues(golden[demo]):
        km = issue["kedb_match"]
        assert km["found"] is True
        first_res = kb.get_article(km["error_code"]).resolution.steps[0]
        assert km["solution"] == first_res[:KEDB_SOLUTION_MAX_CHARS]


@pytest.mark.parametrize("demo", DEMOS)
def test_similar_incidents_exist_and_are_sorted(
    golden: dict, demo: str, tickets: FakeTicketStore
) -> None:
    sims = golden[demo]["similar_incidents"]
    assert len(sims) <= 5
    scores = [s["similarity_score"] for s in sims]
    assert scores == sorted(scores, reverse=True)
    details = {s["incident_id"]: s for s in golden[demo]["details"]["similar"]}
    for s in sims:
        [rec] = tickets.get([s["incident_id"]])
        assert s["resolution"] == rec.resolution
        assert details[s["incident_id"]]["resolution_origin"] == rec.resolution_origin


@pytest.mark.parametrize("demo", DEMOS)
def test_decision_and_booleans_are_consistent(golden: dict, demo: str, kb: FakeKBStore) -> None:
    g = golden[demo]
    d = g["details"]
    issue_decisions = [d["primary_decision"]] + [s["decision"] for s in d["secondary_issues"]]
    # §8.4: ticket decision and escalation_required
    assert g["escalation_required"] == ("escalate" in issue_decisions)
    for top in ("escalate", "clarify", "recommend"):
        if top in issue_decisions:
            assert d["decision"] == top
            break
    if g["escalation_required"]:
        assert d["escalation"] is not None
        if any(r["rule"] in HARD_RULES for r in d["escalation"]["reasons"]):
            assert d["escalation"]["tier"] == "L3"
    else:
        assert d["escalation"] is None
    # §8.5: approval is required iff at least one reason applies
    requires_admin = kb.get_article(g["kedb_match"]["error_code"]).resolution.requires_admin
    needs_approval = (
        d["decision"] != "recommend"
        or g["priority"]["level"] in ("Critical", "High")
        or any(s["role"] == "L2" for s in d["steps"])
        or requires_admin
        or d["classification"]["band"] != "high"
        or g["recommended_solution"]["source"] == "None"
        or bool(d["degraded"])
    )
    assert g["human_approval_required"] is needs_approval
    assert bool(d["approval_reasons"]) is needs_approval


@pytest.mark.parametrize("demo", DEMOS)
def test_band_matches_confidence(golden: dict, demo: str) -> None:
    conf = golden[demo]["classification"]["confidence"]
    band = golden[demo]["details"]["classification"]["band"]
    assert band == ("high" if conf >= 0.75 else "medium" if conf >= 0.55 else "low")


@pytest.mark.parametrize("demo", DEMOS)
def test_priority_code_matches_level(golden: dict, demo: str) -> None:
    from shared.constants import PRIORITY_LEVEL

    g = golden[demo]
    assert PRIORITY_LEVEL[g["details"]["priority"]["code"]] == g["priority"]["level"]
