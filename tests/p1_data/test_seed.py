"""Curated verified incidents (P1-T4): schema, coverage, KB links, safety, verified_by."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from ingest.make_seed import build, taxonomy, unsafe
from privacy.redact import redact
from shared.schemas import TicketRecord
from store.config import ROOT, load_yaml

SEED_JSON = ROOT / "data/seed/resolved_incidents.json"


@pytest.fixture(scope="module")
def seed() -> list[TicketRecord]:
    return [TicketRecord.model_validate(r) for r in json.loads(SEED_JSON.read_text("utf-8"))]


@pytest.fixture(scope="module")
def kb_ids() -> set[str]:
    return {a["kb_id"] for a in json.loads((ROOT / "data/kb/articles.json").read_text("utf-8"))}


def test_count_ids_and_origin(seed):
    assert len(seed) == 60
    assert [r.ticket_id for r in seed] == [f"INC-H{i:04d}" for i in range(1, 61)]
    for r in seed:
        assert r.resolution_origin == "curated_verified"
        assert r.verified and r.verified_by
        assert r.collection == "tickets_verified" and r.status == "resolved"
        assert r.resolution and len(r.resolution.split()) >= 12


def test_two_per_subcategory(seed):
    tax = taxonomy()
    counts = Counter((r.category, r.subcategory) for r in seed)
    expected = {(c, s) for c, subs in tax.items() for s in subs}
    assert set(counts) == expected and len(expected) == 30
    assert set(counts.values()) == {2}


def test_kb_links_exist(seed, kb_ids):
    for r in seed:
        assert r.metadata["kb_ids"] and set(r.metadata["kb_ids"]) <= kb_ids
        assert r.resolution_kb_id == r.metadata["kb_ids"][0]


def test_kb_safety_rules(seed):
    assert [r.ticket_id for r in seed if unsafe(r.resolution)] == []


@pytest.mark.parametrize(
    "text",
    [
        "Disable the antivirus and retry",
        "Ask the user to share your password with IT",
        "Format the drive and reinstall",
        "Delete the profile key from the registry",
        "Grant the user DBA rights",
    ],
)
def test_safety_lint_catches_unsafe(text):
    assert unsafe(text)


def test_no_pii_in_seed(seed):
    for r in seed:
        for text in (r.title, r.description, r.resolution):
            assert redact(text).items == [], (r.ticket_id, text)


def test_committed_json_is_current(kb_ids):
    src = load_yaml(ROOT / "data/seed/curated_incidents.yaml")
    assert build(src, kb_ids, taxonomy()) == json.loads(SEED_JSON.read_text("utf-8"))


def test_build_rejects_bad_records(kb_ids):
    base = {
        "category": "Network",
        "subcategory": "VPN",
        "kb_ids": ["KB-0018"],
        "title": "t",
        "description": "d",
        "resolution": "Restarted the client and reconnected successfully today.",
    }
    tax = taxonomy()
    ok = {"verified_by": "x", "incidents": [base]}
    assert build(ok, kb_ids, tax)[0]["verified_by"] == "x"
    for bad in (
        {**base, "subcategory": "Nope"},
        {**base, "kb_ids": ["KB-9999"]},
        {**base, "resolution": "Disable the firewall"},
    ):
        with pytest.raises(ValueError):
            build({"verified_by": "x", "incidents": [bad]}, kb_ids, tax)
