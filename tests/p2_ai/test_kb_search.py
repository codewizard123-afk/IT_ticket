"""P2-T6: KEDB retrieval. Correct top-1 KB for all 7 demos (DEMO-007 → KB-0003, not KB-0004)."""

from __future__ import annotations

import pytest
import yaml

from ai.kb_search import (
    code_candidates,
    code_hits,
    kedb_contract,
    kedb_details,
    search_articles,
    search_kb,
    taxonomy_boosts,
)
from ai.signals import extract
from tests.p2_ai.fakes import FIXTURES, FakeKBStore, load_fixture

DEMOS = {d["id"]: d for d in load_fixture("demo_tickets.json")}
TMAP = yaml.safe_load((FIXTURES / "taxonomy_map.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kb():
    return FakeKBStore()


def _span(demo_id: str, part: int | None = None) -> str:
    d = DEMOS[demo_id]
    if part is None:
        return f"{d['title']}. {d['description']}"
    body = d["description"].split("(1)")[1]
    return body.split("(2)")[part - 1]


def _search(kb, text, cat, sub, **kw):
    sig = extract(text, kb.all_error_phrases())
    return search_articles(
        text,
        kb,
        error_codes=sig.error_codes,
        category=cat,
        subcategory=sub,
        tmap=TMAP,
        **kw,
    )


TOP1 = [
    ("DEMO-001", None, "Network", "VPN", "KB-0018"),
    ("DEMO-002", None, "Database", "Oracle", "KB-0001"),
    ("DEMO-003", None, "Hardware", "Monitor", "KB-0008"),
    ("DEMO-004", None, "Software", "Email", "KB-0029"),
    ("DEMO-005", 1, "Network", "VPN", "KB-0018"),
    ("DEMO-005", 2, "Hardware", "CPU", "KB-0030"),
    ("DEMO-006", None, "Network", "Wireless", "KB-0021"),
    ("DEMO-007", None, "Database", "MS SQL Server", "KB-0003"),
]


@pytest.mark.parametrize(("demo", "part", "cat", "sub", "kb_id"), TOP1)
def test_demo_top1(kb, demo, part, cat, sub, kb_id):
    res = _search(kb, _span(demo, part), cat, sub)
    assert res.matches[0].kb_id == kb_id
    assert res.found


def test_demo_007_kb0003_above_kb0004(kb):
    res = _search(kb, _span("DEMO-007"), "Database", "MS SQL Server")
    ids = [m.kb_id for m in res.matches]
    assert ids.index("KB-0003") < ids.index("KB-0004")
    assert res.matches[0].matched_error_code in (
        "SQL Server error 18456",
        "Login failed for user",
    )


def test_demo_005_cpu_has_kb0006_second(kb):
    res = _search(kb, _span("DEMO-005", 2), "Hardware", "CPU")
    assert [m.kb_id for m in res.matches[:2]] == ["KB-0030", "KB-0006"]


def test_exact_code_match_details(kb):
    res = _search(kb, _span("DEMO-001"), "Network", "VPN")
    top = res.matches[0]
    assert top.matched_field == "errors"
    assert top.matched_error_code == "Gateway not responding"
    assert "exact error code" in top.why and "VPN mapping" in top.why
    assert top.components["code"] == 1.0 and top.components["tax"] == 1.0


def test_results_capped_and_above_tau_show(kb):
    res = _search(kb, _span("DEMO-006"), "Network", "Wireless")
    assert len(res.matches) <= 3
    assert all(m.score >= 0.35 for m in res.matches)


def test_resolution_chunks_are_never_searched(kb):
    # text identical to a resolution step of KB-0018 only
    text = "Connect to the alternate VPN gateway listed in the client profile"
    res = search_articles(text, kb, tmap={})
    for m in res.matches:
        assert "#res." not in m.excerpt
    for ch in kb.query_chunks(text, 50, where={"chunk_type": {"$ne": "resolution"}}):
        assert ch.chunk_type != "resolution"


def test_feedback_prior_changes_ranking(kb):
    text = _span("DEMO-007")
    base = _search(kb, text, "Database", "MS SQL Server")
    boosted = _search(
        kb,
        text,
        "Database",
        "MS SQL Server",
        feedback_prior={"KB-0004": 0.5, "KB-0003": -0.5},
    )
    s_base = {m.kb_id: m.score for m in base.matches}
    s_new = {m.kb_id: m.score for m in boosted.matches}
    assert s_new["KB-0004"] > s_base["KB-0004"]
    assert s_new["KB-0003"] < s_base["KB-0003"]


def test_feedback_prior_is_clipped(kb):
    a = _search(
        kb, _span("DEMO-003"), "Hardware", "Monitor", feedback_prior={"KB-0008": 0.5}
    )
    b = _search(
        kb, _span("DEMO-003"), "Hardware", "Monitor", feedback_prior={"KB-0008": 9.0}
    )
    assert a.matches[0].score == b.matches[0].score


def test_ambiguity_flag(kb):
    # vague database text: slow vs. login, two different KB subcategories close together
    text = "database problem: slow and login failed"
    res = search_articles(text, kb, error_codes=[], tmap={})
    top2 = res.matches[:2]
    expected = (
        len(top2) == 2
        and top2[0].score - top2[1].score < 0.05
        and top2[0].subcategory != top2[1].subcategory
    )
    assert res.ambiguous == expected


def test_ambiguity_forced_with_config(kb):
    import copy

    from ai.config import ai_config

    cfg = copy.deepcopy(ai_config())
    cfg["retrieval"]["kb"]["ambiguity_gap"] = 1.0
    res = _search(kb, _span("DEMO-002"), "Database", "Oracle", cfg=cfg)
    assert res.ambiguous  # KB-0001 (Performance) vs KB-0003 (Access)


def test_not_found_below_tau_kb(kb):
    res = search_articles("printer prints blank pages in colour", kb, tmap=TMAP)
    assert not res.found
    assert kedb_contract(res, kb) == {
        "found": False,
        "error_code": None,
        "solution": None,
    }


def test_kedb_contract_and_details(kb):
    res = _search(kb, _span("DEMO-001"), "Network", "VPN")
    c = kedb_contract(res, kb)
    assert c == {
        "found": True,
        "error_code": "KB-0018",
        "solution": "Connect to the alternate VPN gateway listed in the client profile",
    }
    d = kedb_details(res)
    assert (
        d["kb_id"] == "KB-0018" and d["matched_error_code"] == "Gateway not responding"
    )
    assert d["alternatives"] and d["alternatives"][0]["kb_id"] == "KB-0019"


def test_solution_truncated_to_200_chars(kb):
    import copy

    arts = copy.deepcopy(load_fixture("kb_articles.json"))
    for a in arts:
        if a["kb_id"] == "KB-0018":
            a["resolution"]["steps"][0] = "Connect " + "very " * 80 + "carefully"
    store = FakeKBStore(arts)
    c = kedb_contract(_search(store, _span("DEMO-001"), "Network", "VPN"), store)
    assert len(c["solution"]) <= 200


def test_retired_articles_excluded():
    import copy

    arts = copy.deepcopy(load_fixture("kb_articles.json"))
    for a in arts:
        if a["kb_id"] == "KB-0018":
            a["status"] = "retired"
    store = FakeKBStore(arts)
    res = _search(store, _span("DEMO-001"), "Network", "VPN")
    assert "KB-0018" not in [m.kb_id for m in res.matches]


def test_taxonomy_boosts_database_symptoms():
    b = taxonomy_boosts("Database", "MS SQL Server", "Login failed for user", TMAP)
    assert b["KB-0003"] == 1.0 and b["KB-0004"] == 0.5
    b2 = taxonomy_boosts("Database", "Oracle", "query is slow", TMAP)
    assert b2["KB-0001"] == 1.0
    assert taxonomy_boosts("Network", "DHCP", "x", TMAP) == {
        "KB-0023": 0.5,
        "KB-0020": 0.5,
    }
    assert taxonomy_boosts("Network", "VPN", "x", {}) == {}


def test_taxonomy_boosts_dict_format():
    tmap = {"Network/VPN": {"primary": ["KB-0018"], "secondary": ["KB-0023"]}}
    assert taxonomy_boosts("Network", "VPN", "x", tmap) == {
        "KB-0018": 1.0,
        "KB-0023": 0.5,
    }


def test_code_candidates_and_hits(kb):
    assert code_candidates("SQL Server error 18456") == [
        "SQL Server error 18456",
        "18456",
    ]
    assert code_candidates("ORA-01013") == ["ORA-01013"]
    hits = code_hits(["SQL Server error 18456"], kb)
    assert hits == {
        "KB-0003": "SQL Server error 18456",
        "KB-0004": "SQL Server error 18456",
    }


def test_search_kb_fr07(kb):
    res = search_kb(
        "Outlook crashes with 0xc0000005",
        {"category": "Software", "subcategory": "Email"},
        2,
        kb,
    )
    assert res[0].kb_id == "KB-0029" and len(res) <= 2
