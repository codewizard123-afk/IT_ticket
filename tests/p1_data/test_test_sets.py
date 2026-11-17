"""Test/trial sets and gold (P1-T5): coverage, no gold leakage, consistency with golden fixtures."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from ingest.make_seed import taxonomy
from ingest.make_test_sets import EDGE_CATEGORIES, build, validate
from shared.schemas import TicketIn
from store.config import ROOT, load_yaml

GOLD_KEYS = (
    "expected_category",
    "expected_subcategory",
    "expected_priority",
    "expected_kb_ids",
    "expected_decision",
)


@pytest.fixture(scope="module")
def built() -> dict:
    return build()


def _read(rel: str):
    return json.loads((ROOT / rel).read_text("utf-8"))


def test_committed_files_are_current(built):
    assert _read("data/samples/sample_tickets.json") == built["samples"]
    assert _read("data/test/test_tickets.json") == built["test_tickets"]
    assert _read("data/test/trial_tickets.json") == built["trial_tickets"]
    assert _read("data/test/trial_gold.json") == built["trial_gold"]
    assert _read("eval/gold/gold_kb.json") == built["gold_kb"]


def test_input_format(built):
    for t in built["test_tickets"] + built["trial_tickets"]:
        TicketIn.model_validate(t)
        assert t["category"] is None and t["priority"] is None and t["status"] == "new"
        assert t["metadata"]["source"] in ("demo", "synthetic_team")


def test_edge_case_coverage(built):
    tests = [t for t in built["test_tickets"] if t["metadata"]["source"] == "synthetic_team"]
    got = Counter(t["metadata"]["edge_case"] for t in tests)
    for edge, n in EDGE_CATEGORIES.items():
        assert got[edge] >= n, edge
    assert len(tests) >= 25
    for t in built["test_tickets"]:
        assert all(k in t["metadata"] for k in GOLD_KEYS), t["id"]


def test_trial_has_no_gold(built):
    trial = built["trial_tickets"]
    assert [t["id"] for t in trial] == [f"T{i:02d}" for i in range(1, 13)]
    assert not any(k.startswith("expected") for t in trial for k in t["metadata"])
    assert set(built["trial_gold"]) == {t["id"] for t in trial}
    cats = Counter(g["category"] for g in built["trial_gold"].values())
    assert len(cats) == 6 and set(cats.values()) == {2}
    demos = {t["description"] for t in built["samples"]}
    assert not any(t["description"] in demos for t in trial)


def test_demos_match_golden_fixtures(built):
    """Demo gold must agree with shared/fixtures/golden (architecture §16)."""
    assert [t["id"] for t in built["samples"]] == [f"DEMO-00{i}" for i in range(1, 8)]
    for t in built["samples"]:
        g = _read(f"shared/fixtures/golden/{t['id']}.json")
        m = t["metadata"]
        assert m["expected_category"] == g["classification"]["category"]
        assert m["expected_subcategory"] == g["classification"]["subcategory"]
        assert m["expected_decision"] == g["details"]["decision"]
        assert m["expected_priority"] == g["priority"]["level"]
        assert g["kedb_match"]["error_code"] in m["expected_kb_ids"]


def test_gold_kb(built):
    items = built["gold_kb"]["items"]
    assert Counter(i["set"] for i in items) == {"demo": 7, "test": 25, "test_split": 40}
    kb = {a["kb_id"] for a in _read("data/kb/articles.json")}
    enriched = {r["ticket_id"]: r for r in _read("data/processed/tickets_enriched.json")}
    for i in items:
        assert set(i["kb_ids"]) <= kb
        assert i["accept_escalate"] or i["kb_ids"]
        if i["set"] == "test_split":
            assert enriched[i["ticket_id"]]["metadata"]["split"] == "test"
    split_subs = {i["subcategory"] for i in items if i["set"] == "test_split"}
    assert len(split_subs) == 30


def test_validate_rejects_bad_gold():
    src = load_yaml(ROOT / "data/seed/test_seed.yaml")
    kb = {a["kb_id"] for a in _read("data/kb/articles.json")}
    tax = taxonomy()
    validate(src, kb, tax)
    bad = json.loads(json.dumps(src))
    bad["test"][0]["gold"]["subcategory"] = "Toaster"
    with pytest.raises(ValueError):
        validate(bad, kb, tax)
    bad = json.loads(json.dumps(src))
    bad["trial"][0]["description"] = src["demos"][0]["description"]
    with pytest.raises(ValueError):
        validate(bad, kb, tax)
