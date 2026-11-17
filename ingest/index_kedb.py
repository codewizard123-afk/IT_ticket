"""KEDB ingest (architecture §6.4).

1. Validate ``data/kb/articles.json`` against ``KBArticle`` and add governance fields
   (``status=approved, version=1, source=seed``) → ``data/kb/articles_governed.json``.
   Entries already in the governed file that came from the app (``source != "seed"`` or a higher
   version) are kept, so a rebuild never loses an approved KB draft.
2. Field-level chunks → Chroma ``kedb_chunks`` (resolution chunks stored, not searched).
3. Error-code index → ``indexes/error_codes.json``.
4. BM25 over the searchable approved chunks → ``indexes/bm25_kedb.pkl``.
5. ``reindex_article(kb_id)`` re-chunks and upserts one article (KB-draft approval flow, P3).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ingest.chunks import build_chunks, build_code_index
from shared.schemas import KBArticle
from store.bm25_store import BM25Index
from store.chroma_store import KB_COLLECTION, EmbedFn, get_collection, reset_collection
from store.config import data_config, read_json, repo_path, write_json

log = logging.getLogger(__name__)


def index_path(name: str, root: Path | None = None) -> Path:
    return repo_path("indexes_dir", root) / data_config()["indexes"][name]


# --- 1. governance -----------------------------------------------------------------------------
def load_articles(path: Path) -> list[KBArticle]:
    raw = read_json(path)
    arts = [KBArticle.model_validate(a) for a in raw]
    ids = [a.kb_id for a in arts]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate kb_id(s) in {path}: {sorted(dupes)}")
    return arts


def govern(
    seed: Sequence[KBArticle], existing: Iterable[KBArticle] | None = None
) -> list[KBArticle]:
    """Seed articles become ``approved v1 source=seed``; app-approved versions win."""
    out: dict[str, KBArticle] = {
        a.kb_id: a.model_copy(
            update={"status": "approved", "version": max(1, a.version), "source": "seed"}
        )
        for a in seed
    }
    for a in existing or ():
        cur = out.get(a.kb_id)
        if cur is None or a.source != "seed" or a.version > cur.version:
            out[a.kb_id] = a
    return [out[k] for k in sorted(out)]


def load_governed(root: Path | None = None) -> list[KBArticle]:
    """The governed KEDB (falls back to the seed file if ``make data`` hasn't run yet)."""
    gov = repo_path("kb_governed", root)
    if gov.exists():
        return load_articles(gov)
    return govern(load_articles(repo_path("kb_articles", root)))


def write_governed(articles: Sequence[KBArticle], root: Path | None = None) -> Path:
    path = repo_path("kb_governed", root)
    write_json(path, [a.model_dump(mode="json") for a in articles])
    return path


# --- 2-4. index ---------------------------------------------------------------------------------
def _add_chunks(coll: Any, chunks: Sequence[dict[str, Any]], embed_fn: EmbedFn) -> None:
    for i in range(0, len(chunks), 500):
        batch = chunks[i : i + 500]
        vecs = embed_fn([c["text"] for c in batch])
        coll.upsert(
            ids=[c["id"] for c in batch],
            embeddings=[v.tolist() for v in vecs],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def searchable(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        c for c in chunks if c["metadata"]["searchable"] and c["metadata"]["status"] == "approved"
    ]


def build_bm25(articles: Iterable[KBArticle]) -> BM25Index:
    chunks = searchable(c for a in articles for c in build_chunks(a))
    return BM25Index.build([c["id"] for c in chunks], [c["text"] for c in chunks])


def index_kedb(
    client: Any,
    embed_fn: EmbedFn,
    articles: Sequence[KBArticle],
    *,
    root: Path | None = None,
    collection: str = KB_COLLECTION,
    write_indexes: bool = True,
) -> dict[str, int]:
    """(Re)build ``kedb_chunks`` + error-code index + BM25. Returns counts for the manifest."""
    chunks = [c for a in articles for c in build_chunks(a)]
    coll = reset_collection(client, collection)
    _add_chunks(coll, chunks, embed_fn)
    codes = build_code_index(articles)
    bm25 = build_bm25(articles)
    if write_indexes:
        write_json(index_path("error_codes", root), codes)
        bm25.save(index_path("bm25_kedb", root))
    log.info("kedb: %d articles, %d chunks, %d codes", len(articles), len(chunks), len(codes))
    return {
        "kb_articles": len(articles),
        "kb_chunks": len(chunks),
        "kb_searchable_chunks": len(searchable(chunks)),
        "error_codes": len(codes),
    }


# --- 5. single-article reindex ----------------------------------------------------------------
def reindex_article(
    kb_id: str,
    article: KBArticle | dict[str, Any] | None = None,
    *,
    client: Any = None,
    embed_fn: EmbedFn | None = None,
    root: Path | None = None,
    kb_store: Any = None,
) -> int:
    """Re-chunk and upsert one article without a full rebuild (P1-T6).

    ``article`` (if given) is written into ``articles_governed.json`` first; otherwise the
    governed copy is re-indexed as is. Draft/retired articles are stored with their status, so
    ``query_chunks`` (approved only) ignores them. Returns the number of chunks written.
    If ``kb_store`` (a ``ChromaKBStore``) is passed, its in-memory view is refreshed too.
    """
    if client is None or embed_fn is None:
        from store.factory import default_client, default_embed_fn

        client = client or default_client(root)
        embed_fn = embed_fn or default_embed_fn()

    governed = {a.kb_id: a for a in load_governed(root)}
    if article is not None:
        art = article if isinstance(article, KBArticle) else KBArticle.model_validate(article)
        if art.kb_id != kb_id:
            raise ValueError(f"article id {art.kb_id!r} does not match {kb_id!r}")
        governed[kb_id] = art
        write_governed([governed[k] for k in sorted(governed)], root)
    elif kb_id not in governed:
        raise KeyError(f"unknown KB article {kb_id!r}")
    art = governed[kb_id]

    coll = get_collection(client, KB_COLLECTION)
    coll.delete(where={"kb_id": kb_id})
    chunks = build_chunks(art)
    _add_chunks(coll, chunks, embed_fn)

    # keep the on-disk side indexes consistent with the collection
    all_arts = list(governed.values())
    write_json(index_path("error_codes", root), build_code_index(all_arts))
    bm25_file = index_path("bm25_kedb", root)
    bm25 = BM25Index.load(bm25_file) if bm25_file.exists() else build_bm25(all_arts)
    bm25.remove_where(lambda d: d.startswith(f"{kb_id}#"))
    for c in searchable(chunks):
        bm25.add(c["id"], c["text"])
    bm25.save(bm25_file)

    if kb_store is not None:
        kb_store.refresh_article(art)
    log.info("reindexed %s v%s (%s): %d chunks", kb_id, art.version, art.status, len(chunks))
    return len(chunks)
