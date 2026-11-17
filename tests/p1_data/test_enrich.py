"""``ingest.enrich`` (P1-T3): every record flagged, valid origins, epochs, dups, restricted."""

from __future__ import annotations

import hashlib
import json

import pytest

from ingest.enrich import dup_groups, enrich, parse_ts
from shared.constants import RESOLUTION_ORIGINS
from shared.schemas import TicketRecord
from store.config import ROOT, data_config

TICKETS = ROOT / "data/processed/tickets.json"
ENRICHED = ROOT / "data/processed/tickets_enriched.json"


@pytest.fixture(scope="module")
def raw() -> list[dict]:
    return json.loads(TICKETS.read_text("utf-8"))


@pytest.fixture(scope="module")
def enriched(raw) -> list[dict]:
    return enrich(raw)


def test_every_record_flagged(enriched, raw):
    assert len(enriched) == len(raw) == 1500
    for r in enriched:
        rec = TicketRecord.model_validate(r)
        assert rec.derived_fields and "resolution" in rec.derived_fields
        assert rec.resolution_origin in RESOLUTION_ORIGINS
        assert rec.resolution_origin.startswith("synthetic_")  # never citable
        assert rec.metadata["resolution_origin_raw"] == "synthetic_generated"
        assert rec.collection == "tickets_hist" and not rec.verified
        assert rec.assignment_group and rec.application and rec.environment
        assert rec.text_hash and len(rec.text_hash) == 40


def test_escalated_maps_to_synthetic_escalated(enriched, raw):
    by_id = {r["ticket_id"]: r for r in enriched}
    for r in raw:
        want = "synthetic_escalated" if r["metadata"]["escalated"] else "synthetic_kb_derived"
        assert by_id[r["id"]]["resolution_origin"] == want
        assert by_id[r["id"]]["resolution_kb_id"] == r["metadata"]["related_kb_id"]


def test_dates_are_utc_epochs(enriched):
    for r in enriched:
        assert r["opened_at"].endswith("Z")
        assert isinstance(r["opened_ts"], int) and isinstance(r["resolved_ts"], int)
        assert r["resolved_ts"] >= r["opened_ts"]
    assert parse_ts("2026-05-01T06:42") == ("2026-05-01T06:42:00Z", 1777617720)
    assert parse_ts(None) == (None, None)


def test_dup_groups_follow_generator_links(enriched, raw):
    by_id = {r["ticket_id"]: r for r in enriched}
    links = [(r["id"], r["metadata"]["duplicate_of"]) for r in raw if r["metadata"]["duplicate_of"]]
    assert len(links) == 90
    for dup, orig in links:
        assert by_id[dup]["dup_group_id"] is not None
        assert by_id[dup]["dup_group_id"] == by_id[orig]["dup_group_id"]


def test_dup_groups_exact_text():
    rows = [{"id": "A2"}, {"id": "A1"}, {"id": "B"}]
    groups = dup_groups(rows, {"A1": "h", "A2": "h", "B": "x"})
    assert groups == {"A1": "DUP-A1", "A2": "DUP-A1"}


def test_restricted_seeded_and_scoped(enriched):
    subs = set(data_config()["enrich"]["restricted"]["subcategories"])
    restricted = [r for r in enriched if r["restricted"]]
    assert 2 <= len(restricted) <= 10
    assert {r["subcategory"] for r in restricted} == subs


def test_deterministic_and_source_untouched(raw):
    before = hashlib.sha1(TICKETS.read_bytes()).hexdigest()
    assert enrich(raw) == enrich(raw)
    assert hashlib.sha1(TICKETS.read_bytes()).hexdigest() == before


def test_committed_file_is_current(enriched):
    """``data/processed/tickets_enriched.json`` must equal a fresh ``enrich()`` (run
    ``python -m ingest.enrich`` after changing tickets.json or config/data.yaml)."""
    assert json.loads(ENRICHED.read_text("utf-8")) == enriched


def test_pii_not_in_hashed_text(raw, enriched):
    from privacy.redact import redact_text
    from store.text import embedding_text, text_hash

    pii = next(r for r in raw if r["metadata"]["contains_pii"])
    rec = next(r for r in enriched if r["ticket_id"] == pii["id"])
    want = text_hash(embedding_text(redact_text(pii["title"]), redact_text(pii["description"])))
    assert rec["text_hash"] == want
