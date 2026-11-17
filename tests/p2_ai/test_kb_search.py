"""P2-T6: KEDB retrieval (architecture §7.6)."""

from __future__ import annotations

import pytest

from ai.classify import classify_text
from ai.kb_search import match_kb, rank_articles, search_kb, tax_boost, taxonomy_map
from ai.signals import extract, extract_signals, ticket_text
from shared.schemas import ClassificationResult


def cls(cat: str, sub: str) -> ClassificationResult:
    return ClassificationResult(category=cat, subcategory=sub, confidence=0.9, band="high")


DEMO_KB = {
    "DEMO-001": (("Network", "VPN"), "KB-0018"),
    "DEMO-002": (("Database", "Oracle"), "KB-0001"),
    "DEMO-003": (("Hardware", "Monitor"), "KB-0008"),
    "DEMO-004": (("Software", "Email"), "KB-0029"),
    "DEMO-006": (("Network", "Wireless"), "KB-0021"),
    "DEMO-007": (("Database", "MS SQL Server"), "KB-0003"),
}


@pytest.mark.parametrize("demo", sorted(DEMO_KB))
def test_correct_top1_kb_for_demos(demos, kb_store, demo) -> None:
    (cat, sub), kb_id = DEMO_KB[demo]
    t = demos[demo]
    res = match_kb(ticket_text(t), extract_signals(t, kb_store), cls(cat, sub), kb_store)
    assert res.hits[0].kb_id == kb_id
    assert res.relevance == res.hits[0].score


def test_demo_005_issues(kb_store, ticket_store) -> None:
    vpn = "My VPN connection drops every 10 to 15 minutes and I keep losing unsaved work."
    cpu = "My laptop is overheating and the fan runs loudly all the time, even when idle."
    phrases = kb_store.all_error_phrases()
    r1 = match_kb(vpn, extract(vpn, phrases), classify_text(vpn, ticket_store).result, kb_store)
    r2 = match_kb(cpu, extract(cpu, phrases), classify_text(cpu, ticket_store).result, kb_store)
    assert r1.ranked[0].kb_id == "KB-0018"
    assert [h.kb_id for h in r2.ranked[:2]] == ["KB-0030", "KB-0006"]


def test_demo_007_kb_0003_beats_kb_0004(demos, kb_store) -> None:
    t = demos["DEMO-007"]
    res = match_kb(
        ticket_text(t), extract_signals(t, kb_store), cls("Database", "MS SQL Server"), kb_store
    )
    top, second = res.ranked[0], res.ranked[1]
    assert (top.kb_id, second.kb_id) == ("KB-0003", "KB-0004")
    assert top.matched_error_code == "18456" and second.matched_error_code == "18456"
    assert res.ambiguous is False  # same KB subcategory (Access)


def test_hit_fields_for_code_match(demos, kb_store) -> None:
    t = demos["DEMO-001"]
    top = match_kb(
        ticket_text(t), extract_signals(t, kb_store), cls("Network", "VPN"), kb_store
    ).hits[0]
    assert top.matched_field == "errors"
    assert top.excerpt == "Gateway not responding"
    assert top.matched_error_code == "Gateway not responding"
    assert "exact error code" in top.why and "VPN mapping" in top.why
    assert top.title.startswith("VPN will not connect")
    assert top.requires_admin is False


def test_symptom_match_without_codes(demos, kb_store) -> None:
    t = demos["DEMO-006"]
    top = match_kb(
        ticket_text(t), extract_signals(t, kb_store), cls("Network", "Wireless"), kb_store
    ).ranked[0]
    assert top.matched_error_code is None
    assert top.matched_field in {"symptom", "title", "diagnostic"}
    assert top.excerpt


def test_feedback_prior_changes_ranking(demos, kb_store) -> None:
    t = demos["DEMO-007"]
    sig = extract_signals(t, kb_store)
    c = cls("Database", "MS SQL Server")
    base = match_kb(ticket_text(t), sig, c, kb_store)
    boosted = match_kb(
        ticket_text(t), sig, c, kb_store, feedback_prior={"KB-0004": 0.5, "KB-0003": -0.5}
    )
    s = {h.kb_id: h.score for h in base.ranked}
    b = {h.kb_id: h.score for h in boosted.ranked}
    assert b["KB-0004"] > s["KB-0004"] and b["KB-0003"] < s["KB-0003"]
    assert "analysts marked it useful" in next(
        h.why for h in boosted.ranked if h.kb_id == "KB-0004"
    )


def test_ambiguity_flag_for_close_different_subcategories(kb_store) -> None:
    # Oracle ticket that is both slow and failing logins: KB-0001 (Performance) vs KB-0003 (Access)
    text = "Oracle queries slow; login failed for user"
    res = rank_articles(text, [], kb_store, classification=cls("Database", "Oracle"), cfg=None)
    top2 = res.ranked[:2]
    assert {h.kb_id for h in top2} == {"KB-0001", "KB-0003"}
    assert res.ambiguous == (abs(top2[0].score - top2[1].score) < 0.05)


def test_taxonomy_map_and_database_symptoms() -> None:
    assert taxonomy_map()["Network/VPN"]["primary"] == ["KB-0018", "KB-0019"]
    boost = tax_boost(
        cls("Database", "Oracle"),
        "the report is very slow",
        {"database_symptoms": {"KB-0001": r"\bslow\b"}},
    )
    assert boost["KB-0001"] == 1.0 and boost["KB-0003"] == 0.5
    assert tax_boost(cls("Hardware", "Disk"), "disk full", {})["KB-0006"] == 0.5


def test_unrelated_text_is_not_found(kb_store) -> None:
    res = rank_articles("lunch menu for friday", [], kb_store)
    assert res.relevance < 0.5 and res.hits == []


def test_search_kb_free_text(kb_store) -> None:
    hits = search_kb("ORA-01013 report timeout", None, 3, kb_store)
    assert hits[0].kb_id == "KB-0001"
    assert len(hits) <= 3
    filtered = search_kb("slow", {"kb_id": "KB-0030"}, 5, kb_store)
    assert [h.kb_id for h in filtered] == ["KB-0030"]
