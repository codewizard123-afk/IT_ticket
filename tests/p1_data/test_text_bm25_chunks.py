"""Pure-Python P1 helpers: text, BM25, chunking, error-code index (run in CI, no numpy)."""

from __future__ import annotations

import json

import pytest

from ingest.chunks import build_chunks, build_code_index
from shared.constants import SEARCHABLE_CHUNK_TYPES
from shared.schemas import KBArticle
from store.bm25_store import BM25Index
from store.config import ROOT
from store.text import embedding_text, norm_code, text_hash, tokenize


@pytest.fixture(scope="module")
def articles() -> list[KBArticle]:
    raw = json.loads((ROOT / "data/kb/articles.json").read_text("utf-8"))
    return [KBArticle.model_validate(a) for a in raw]


# --- text ------------------------------------------------------------------------------------
def test_tokenize_keeps_error_codes_whole():
    toks = tokenize("Login fails with ORA-01013 and 0x800f0922 on SQL30081N")
    assert {"ora-01013", "0x800f0922", "sql30081n"} <= set(toks)
    assert "with" not in toks


@pytest.mark.parametrize(
    ("code", "norm"),
    [
        ("Gateway not responding", "gateway not responding"),
        ("ORA-01013", "ora 01013"),
        ("SQL Server  error 18456", "sql server error 18456"),
        ("ERROR 1205 (HY000): Lock wait", "error 1205 hy000 lock wait"),
    ],
)
def test_norm_code(code, norm):
    assert norm_code(code) == norm


def test_embedding_text_rule():
    assert embedding_text("VPN down", "VPN down") == "VPN down"
    joined = embedding_text("VPN down.", "Gateway not responding")
    assert joined == "VPN down. Gateway not responding"
    assert embedding_text("", "only body") == "only body"


def test_text_hash_normalises_case_and_space():
    assert text_hash("Printer  Offline") == text_hash("printer offline")
    assert text_hash("a") != text_hash("b")


# --- BM25 ------------------------------------------------------------------------------------
def test_bm25_ranks_matching_doc_first():
    idx = BM25Index.build(
        ["d1", "d2", "d3"],
        ["vpn gateway not responding", "printer offline in building b", "outlook keeps crashing"],
    )
    hits = idx.search("the VPN gateway is down", 2)
    assert hits[0][0] == "d1"
    assert all(s > 0 for _, s in hits)
    assert idx.search("", 3) == []


def test_bm25_add_remove_and_roundtrip(tmp_path):
    idx = BM25Index.build(["a", "b"], ["disk full warning", "mailbox full quota"])
    idx.add("c", "vpn tunnel drops")
    assert idx.search("vpn", 1)[0][0] == "c"
    idx.remove("c")
    assert "c" not in idx and idx.search("vpn", 1) == []
    idx.add("a", "replaced text about printers")  # re-add replaces
    assert len(idx) == 2
    path = tmp_path / "bm25.pkl"
    idx.save(path)
    again = BM25Index.load(path)
    assert again.search("mailbox quota", 2) == idx.search("mailbox quota", 2)


# --- chunks ----------------------------------------------------------------------------------
def test_chunk_ids_and_types(articles):
    art = next(a for a in articles if a.kb_id == "KB-0018")
    chunks = {c["id"]: c for c in build_chunks(art)}
    assert "KB-0018#title" in chunks and "KB-0018#err" in chunks and "KB-0018#esc" in chunks
    assert chunks["KB-0018#res.1"]["text"] == " ".join(art.resolution.steps[0].split())
    assert chunks["KB-0018#diag.1"]["metadata"]["n"] == 1
    assert f"KB-0018#res.{len(art.resolution.steps)}" in chunks
    for c in chunks.values():
        assert c["metadata"]["searchable"] == (c["chunk_type"] in SEARCHABLE_CHUNK_TYPES)
        assert all(not isinstance(v, list | dict) and v is not None for v in c["metadata"].values())
    assert not chunks["KB-0018#res.1"]["metadata"]["searchable"]


def test_all_articles_chunk(articles):
    assert len(articles) == 60
    total = sum(len(build_chunks(a)) for a in articles)
    assert total > 600


def test_code_index_maps_codes_to_articles(articles):
    idx = build_code_index(articles)
    assert idx["gateway not responding"] == ["KB-0018"]
    assert idx["ora 01013"] == ["KB-0001"]
    assert idx["sql server error 18456"] == ["KB-0003", "KB-0004"]


def test_code_index_skips_unapproved(articles):
    drafts = [a.model_copy(update={"status": "draft"}) for a in articles if a.kb_id == "KB-0018"]
    assert "gateway not responding" not in build_code_index(drafts)
