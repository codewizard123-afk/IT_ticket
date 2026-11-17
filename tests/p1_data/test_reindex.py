"""KB governance + ``reindex_article`` (P1-T6): approved drafts are searchable with no rebuild."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("chromadb")

from ingest.build_all import build_all  # noqa: E402
from ingest.index_kedb import govern, load_articles, load_governed, reindex_article  # noqa: E402
from shared.schemas import KBArticle  # noqa: E402
from store.config import ROOT  # noqa: E402
from store.factory import open_stores  # noqa: E402
from tests.p1_data.conftest import make_root  # noqa: E402

NEW = {
    "kb_id": "KB-0061",
    "title": "Smart badge printer prints blank visitor badges",
    "category": "Hardware",
    "subcategory": "Printer",
    "symptoms": ["Visitor badges come out of the reception badge printer completely blank."],
    "common_error_codes": ["Ribbon not detected ZX-77"],
    "root_cause": "The thermal ribbon is installed inside out after a refill.",
    "diagnostic_steps": ["Open the badge printer and check the ribbon orientation."],
    "resolution": {
        "steps": ["Reinstall the ribbon with the ink side facing the card.", "Print a test badge."],
        "estimated_time": "10 minutes",
        "requires_admin": False,
    },
    "escalation_criteria": "Escalate if the print head is damaged.",
    "status": "draft",
    "version": 1,
    "source": "analyst",
}


@pytest.fixture()
def env(tmp_path, hash_embed):
    root = make_root(tmp_path, per_subcategory=1)
    build_all(root=root, embed_fn=hash_embed)
    tickets, kb = open_stores(root, embed_fn=hash_embed)
    return root, kb, hash_embed


def _kb_ids(kb, text):
    return {h.kb_id for h in kb.query_chunks(text, 30)}


def test_governance_fields_on_seed(env):
    root, kb, _ = env
    gov = json.loads((root / "data/kb/articles_governed.json").read_text("utf-8"))
    assert len(gov) == 60
    assert {(a["status"], a["version"], a["source"]) for a in gov} == {("approved", 1, "seed")}


def test_draft_not_searchable_then_approved_is(env):
    root, kb, embed = env
    text = "visitor badges print blank ribbon not detected ZX-77"

    n = reindex_article("KB-0061", NEW, client=kb.client, embed_fn=embed, root=root, kb_store=kb)
    assert n == 7  # title, sym.1, err, diag.1, res.1, res.2, esc
    assert "KB-0061" not in _kb_ids(kb, text)  # draft: stored but not searchable
    assert kb.lookup_codes(["Ribbon not detected ZX-77"]) == {}
    assert kb.chunk_text("KB-0061#res.1").startswith("Reinstall the ribbon")

    approved = {**NEW, "status": "approved", "version": 2, "approved_by": "kb_dave"}
    reindex_article("KB-0061", approved, client=kb.client, embed_fn=embed, root=root, kb_store=kb)
    assert "KB-0061" in _kb_ids(kb, text)
    assert kb.lookup_codes(["Ribbon not detected ZX-77"]) == {
        "Ribbon not detected ZX-77": ["KB-0061"]
    }
    assert any(c.startswith("KB-0061#") for c, _ in kb.bm25(text, 10))
    assert kb.get_article("KB-0061").approved_by == "kb_dave"

    # persisted: a fresh store sees it, and the governed file has it
    _, kb2 = open_stores(root, embed_fn=embed)
    assert "KB-0061" in _kb_ids(kb2, text)
    gov = {a.kb_id: a for a in load_governed(root)}
    assert gov["KB-0061"].version == 2 and gov["KB-0061"].source == "analyst"


def test_new_version_replaces_old_chunks(env):
    root, kb, embed = env
    art = kb.get_article("KB-0018").model_dump()
    art["resolution"]["steps"] = ["Only one step now: reconnect using the regional gateway."]
    art["version"] = 2
    art["source"] = "analyst"
    reindex_article("KB-0018", art, client=kb.client, embed_fn=embed, root=root, kb_store=kb)
    assert kb.chunk_text("KB-0018#res.1").startswith("Only one step now")
    assert not kb.has_chunk("KB-0018#res.2")
    stored = kb.client.get_collection("kedb_chunks").get(where={"kb_id": "KB-0018"})
    assert "KB-0018#res.2" not in stored["ids"] and "KB-0018#res.1" in stored["ids"]


def test_retired_article_drops_out(env):
    root, kb, embed = env
    art = {**kb.get_article("KB-0048").model_dump(), "status": "retired", "version": 2}
    reindex_article("KB-0048", art, client=kb.client, embed_fn=embed, root=root, kb_store=kb)
    assert "KB-0048" not in _kb_ids(kb, "Outlook Trying to connect Need Password Disconnected")
    assert "Need Password" not in kb.all_error_phrases()


def test_rebuild_keeps_app_approved_articles(env):
    root, kb, embed = env
    approved = {**NEW, "status": "approved", "version": 2}
    reindex_article("KB-0061", approved, client=kb.client, embed_fn=embed, root=root)
    build_all(root=root, embed_fn=embed)  # full rebuild from the seed articles.json
    assert any(a.kb_id == "KB-0061" for a in load_governed(root))
    _, kb2 = open_stores(root, embed_fn=embed)
    assert "KB-0061" in _kb_ids(kb2, "visitor badges print blank ribbon not detected ZX-77")


def test_reindex_errors(env):
    root, kb, embed = env
    with pytest.raises(KeyError):
        reindex_article("KB-9999", client=kb.client, embed_fn=embed, root=root)
    with pytest.raises(ValueError):
        reindex_article("KB-0001", NEW, client=kb.client, embed_fn=embed, root=root)


def test_govern_prefers_newer_or_analyst_versions():
    seed = load_articles(ROOT / "data/kb/articles.json")[:2]
    newer = seed[0].model_copy(update={"version": 3, "source": "analyst"})
    out = {a.kb_id: a for a in govern(seed, [newer, KBArticle.model_validate(NEW)])}
    assert out[seed[0].kb_id].version == 3
    assert out[seed[1].kb_id].version == 1 and out[seed[1].kb_id].status == "approved"
    assert "KB-0061" in out
