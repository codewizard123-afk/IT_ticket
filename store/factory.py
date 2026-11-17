"""Open P1's real stores from the built indexes (``make data``). P3's ``backend/deps.py`` calls
``open_stores()``; P2 and ``eval/`` can call it directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from store.bm25_store import BM25Index
from store.chroma_store import ChromaKBStore, ChromaTicketStore, EmbedFn, open_client
from store.config import data_config, repo_path


class IndexMissingError(RuntimeError):
    """``make data`` has not been run (the API maps this to HTTP 503)."""


def default_client(root: Path | None = None) -> Any:
    return open_client(repo_path("chroma_dir", root))


def default_embed_fn() -> EmbedFn:
    from store.embedder import embed

    return embed


def _index_file(name: str, root: Path | None) -> Path:
    return repo_path("indexes_dir", root) / data_config()["indexes"][name]


def _default_redact():  # noqa: ANN202
    try:
        from privacy.redact import redact_text
    except ImportError:  # privacy lands in P1-T2
        return None
    return redact_text


def open_stores(
    root: Path | None = None,
    *,
    client: Any = None,
    embed_fn: EmbedFn | None = None,
    eval_split: bool = False,
) -> tuple[ChromaTicketStore, ChromaKBStore]:
    """``(TicketStore, KBStore)`` over the persisted Chroma index and side indexes."""
    from ingest.index_kedb import load_governed

    manifest = _index_file("manifest", root)
    if not manifest.exists():
        raise IndexMissingError(f"{manifest} not found: run `make data` first")
    client = client if client is not None else default_client(root)
    embed_fn = embed_fn or default_embed_fn()

    bm25_name = "bm25_tickets_eval" if eval_split else "bm25_tickets"
    t_bm25_path = _index_file(bm25_name, root)
    t_bm25 = BM25Index.load(t_bm25_path) if t_bm25_path.exists() else BM25Index()
    tickets = ChromaTicketStore(
        client,
        embed_fn,
        t_bm25,
        suffix="_eval" if eval_split else "",
        bm25_path=None if eval_split else t_bm25_path,
        redact=_default_redact(),
    )

    k_bm25_path = _index_file("bm25_kedb", root)
    codes_path = _index_file("error_codes", root)
    kb = ChromaKBStore(
        client,
        embed_fn,
        load_governed(root),
        bm25=BM25Index.load(k_bm25_path) if k_bm25_path.exists() else None,
        code_index=json.loads(codes_path.read_text("utf-8")) if codes_path.exists() else None,
    )
    return tickets, kb


def read_manifest(root: Path | None = None) -> dict[str, Any]:
    path = _index_file("manifest", root)
    return json.loads(path.read_text("utf-8")) if path.exists() else {}
