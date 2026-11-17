"""``ChromaTicketStore`` / ``ChromaKBStore`` against the Protocols (P1-T1 "Done when").

Runs on a temp index built with the hashing embedder; skipped if chromadb isn't installed
(CI installs only the dev requirements).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("chromadb")

from ingest.build_all import index_stale  # noqa: E402
from shared.interfaces import KBStore, TicketStore  # noqa: E402
from shared.schemas import ChunkHit, KBArticle, TicketHit, TicketRecord  # noqa: E402
from store.chroma_store import normalize_where, record_to_metadata  # noqa: E402
from store.factory import IndexMissingError, open_stores  # noqa: E402


@pytest.fixture(scope="module")
def ts(stores):
    return stores[0]


@pytest.fixture(scope="module")
def kb(stores):
    return stores[1]


def _first_ticket(root, subcategory: str) -> dict:
    rows = json.loads((root / "data/processed/tickets.json").read_text("utf-8"))
    return next(r for r in rows if r["subcategory"] == subcategory)


# --- build + manifest -------------------------------------------------------------------------
def test_manifest_written(built_root):
    m = json.loads((built_root / "indexes/manifest.json").read_text("utf-8"))
    assert m["counts"]["tickets_hist"] == 90  # 3 per subcategory x 30
    assert m["counts"]["kb_articles"] == 60
    assert m["counts"]["kb_chunks"] > m["counts"]["kb_searchable_chunks"] > 0
    assert set(m["source_sha1"]) == {"tickets", "kb_articles"}
    assert (built_root / "data/kb/articles_governed.json").exists()
    assert (built_root / "indexes/error_codes.json").exists()
    assert not index_stale(built_root)


def test_open_stores_without_index_raises(tmp_path):
    with pytest.raises(IndexMissingError):
        open_stores(tmp_path)


def test_protocols(ts, kb):
    assert isinstance(ts, TicketStore)
    assert isinstance(kb, KBStore)


# --- TicketStore -----------------------------------------------------------------------------
def test_query_finds_itself_first(ts, built_root):
    row = _first_ticket(built_root, "VPN")
    hits = ts.query(f"{row['title']}. {row['description']}", 5)
    assert hits and isinstance(hits[0], TicketHit)
    assert hits[0].ticket_id == row["id"]
    assert hits[0].score > 0.95
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(0 <= h.score <= 1 for h in hits)
    h = hits[0]
    assert h.category == "Network" and h.subcategory == "VPN"
    assert h.collection == "tickets_hist"
    assert h.resolution and h.resolution_origin in ("synthetic_kb_derived", "synthetic_escalated")
    assert h.metadata["text_hash"] and h.text  # P2 reads these


def test_query_where_filters(ts):
    hits = ts.query("cannot connect error", 50, where={"category": "Database"})
    assert hits and {h.category for h in hits} == {"Database"}
    hits = ts.query(
        "cannot connect error", 50, where={"category": "Database", "subcategory": "Oracle"}
    )
    assert hits and {h.subcategory for h in hits} == {"Oracle"}
    hits = ts.query("error", 50, where={"subcategory": {"$in": ["VPN", "DB2"]}})
    assert {h.subcategory for h in hits} <= {"VPN", "DB2"}


def test_query_edge_cases(ts):
    assert ts.query("", 5) == []
    assert ts.query("vpn", 0) == []
    assert ts.query("vpn", 5, collections=("tickets_verified",)) == []  # not built yet in T1


def test_bm25_and_get(ts, built_root):
    row = _first_ticket(built_root, "Printer")
    ids = [tid for tid, _ in ts.bm25(row["description"], 5)]
    assert row["id"] in ids
    recs = ts.get([row["id"], "NOPE-1"])
    assert len(recs) == 1 and isinstance(recs[0], TicketRecord)
    assert recs[0].ticket_id == row["id"] and recs[0].subcategory == "Printer"


def test_upsert_live_and_mark_verified(ts):
    rec = TicketRecord(
        ticket_id="LIVE-T1-0001",
        title="Printer jams on floor 3",
        description="The shared printer on floor 3 shows Print job stuck and nothing prints.",
        category="Hardware",
        subcategory="Printer",
        status="new",
    )
    ts.upsert_live(rec)
    live = ts.query(rec.description, 3, collections=("tickets_live",))
    assert live[0].ticket_id == "LIVE-T1-0001" and live[0].collection == "tickets_live"
    assert "LIVE-T1-0001" in [t for t, _ in ts.bm25("print job stuck floor", 10)]

    ts.mark_verified("LIVE-T1-0001", "Cleared the spooler queue and restarted it.", "l2_bob")
    assert ts.query(rec.description, 3, collections=("tickets_live",)) == []
    ver = ts.query(rec.description, 3, collections=("tickets_verified",))
    assert ver[0].ticket_id == "LIVE-T1-0001"
    assert ver[0].resolution_origin == "analyst_verified"
    got = ts.get(["LIVE-T1-0001"])[0]
    assert got.verified and got.verified_by == "l2_bob" and got.collection == "tickets_verified"
    with pytest.raises(KeyError):
        ts.mark_verified("NOPE-404", "x", "y")


def test_metadata_is_scalar():
    rec = TicketRecord(
        ticket_id="X", title="t", description="d", derived_fields=["a", "b"], restricted=True
    )
    meta = record_to_metadata(rec)
    assert meta["derived_fields"] == "a,b"
    assert meta["resolution"] == "" and meta["opened_ts"] == 0
    assert all(isinstance(v, str | int | float | bool) for v in meta.values())


def test_normalize_where():
    assert normalize_where(None) is None
    assert normalize_where({"a": 1}) == {"a": 1}
    assert normalize_where({"a": 1, "b": {"$gte": 2}}) == {"$and": [{"a": 1}, {"b": {"$gte": 2}}]}
    assert normalize_where({"$and": [{"a": 1}]}) == {"a": 1}


# --- KBStore ---------------------------------------------------------------------------------
def test_query_chunks_excludes_resolution(kb):
    hits = kb.query_chunks("VPN gateway not responding cannot connect", 20)
    assert hits and isinstance(hits[0], ChunkHit)
    assert all(h.chunk_type != "resolution" for h in hits)
    assert "KB-0018" in {h.kb_id for h in hits[:5]}
    only_sym = kb.query_chunks("vpn", 10, where={"chunk_type": "symptom"})
    assert only_sym and {h.chunk_type for h in only_sym} == {"symptom"}


def test_kb_bm25_returns_chunk_ids(kb):
    ids = [cid for cid, _ in kb.bm25("Outlook Trying to connect Need Password", 5)]
    assert ids and all("#" in c for c in ids)
    assert not any("#res." in c for c in ids)
    assert any(c.startswith("KB-0048#") for c in ids)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("Gateway not responding", {"KB-0018"}),
        ("ORA-01013", {"KB-0001"}),
        ("SQL Server error 18456", {"KB-0003", "KB-0004"}),
        ("18456", {"KB-0003", "KB-0004"}),
        ("sql0911n", {"KB-0036"}),
        ("MSI error 1618", {"KB-0027"}),
    ],
)
def test_lookup_codes(kb, code, expected):
    assert set(kb.lookup_codes([code])[code]) == expected


def test_lookup_codes_unknown(kb):
    assert kb.lookup_codes(["totally unknown thing", ""]) == {}


def test_article_and_chunk_text(kb):
    art = kb.get_article("KB-0018")
    assert isinstance(art, KBArticle) and art.status == "approved" and art.source == "seed"
    assert kb.chunk_text("KB-0018#res.1") == " ".join(art.resolution.steps[0].split())
    assert kb.chunk_text("KB-0018#diag.2") == " ".join(art.diagnostic_steps[1].split())
    with pytest.raises(KeyError):
        kb.get_article("KB-9999")
    with pytest.raises(KeyError):
        kb.chunk_text("KB-0018#res.99")


def test_all_error_phrases(kb):
    phrases = kb.all_error_phrases()
    assert "Gateway not responding" in phrases and "ORA-01013" in phrases
    assert len(phrases) == len(set(phrases)) > 100


def test_index_is_redacted(ts, built_root):
    rows = json.loads((built_root / "data/processed/tickets.json").read_text("utf-8"))
    pii = [r for r in rows if r["metadata"]["contains_pii"]]
    assert pii, "fixture subset should contain at least one PII ticket"
    for rec in ts.get([r["id"] for r in pii]):
        assert "@example.com" not in rec.description and "+91" not in rec.description
        assert "@example.com" not in rec.text
