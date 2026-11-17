"""``python -m ingest.build_all [--eval-split]`` — build every P1 index (architecture §6, §11.1).

Steps: KEDB (governed file, ``kedb_chunks``, error codes, BM25) → enrich ``tickets.json`` →
historical tickets (``tickets_hist`` from ``tickets_enriched.json`` + BM25) → manifest
``indexes/manifest.json`` ``{source_sha1, model, built_at, counts}``.
``index_stale()`` lets P3's ``/health`` compare.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ingest.enrich import enrich
from ingest.index_kedb import govern, index_kedb, index_path, load_articles, write_governed
from ingest.index_tickets import index_tickets
from privacy.redact import redact_text
from shared.schemas import TicketRecord
from store.bm25_store import BM25Index
from store.chroma_store import EmbedFn
from store.config import read_json, repo_path, write_json

log = logging.getLogger("ingest.build_all")

SOURCE_KEYS = ("tickets", "kb_articles")


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes(root: Path | None = None) -> dict[str, str]:
    out = {}
    for key in SOURCE_KEYS:
        p = repo_path(key, root)
        if p.exists():
            out[key] = sha1_file(p)
    return out


def index_stale(root: Path | None = None) -> bool:
    """True if the indexes are missing or were built from different source files."""
    manifest = index_path("manifest", root)
    if not manifest.exists():
        return True
    return read_json(manifest).get("source_sha1") != source_hashes(root)


def build_all(
    eval_split: bool = False,
    *,
    root: Path | None = None,
    client: Any = None,
    embed_fn: EmbedFn | None = None,
) -> dict[str, Any]:
    """Build all collections and side indexes; returns the manifest."""
    from store.embedder import model_name
    from store.factory import default_client, default_embed_fn

    t0 = time.perf_counter()
    client = client if client is not None else default_client(root)
    embed_fn = embed_fn or default_embed_fn()
    counts: dict[str, int] = {}

    # 1. KEDB
    articles = govern(load_articles(repo_path("kb_articles", root)))
    write_governed(articles, root)
    counts.update(index_kedb(client, embed_fn, articles, root=root))

    # 2. enrich + historical tickets
    enriched = enrich(read_json(repo_path("tickets", root)))
    write_json(repo_path("tickets_enriched", root), enriched)
    records = [TicketRecord.model_validate(r) for r in enriched]
    bm25 = BM25Index()
    counts["tickets_hist"] = index_tickets(
        client, embed_fn, records, "tickets_hist", bm25=bm25, redact=redact_text
    )
    bm25.save(index_path("bm25_tickets", root))

    manifest = {
        "source_sha1": source_hashes(root),
        "model": getattr(embed_fn, "name", None) or model_name(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts": counts,
        "eval_split": eval_split,
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(index_path("manifest", root), manifest)
    log.info("built indexes in %.1fs: %s", manifest["build_seconds"], counts)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-split", action="store_true", help="also build *_eval collections")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
    manifest = build_all(eval_split=args.eval_split)
    print(manifest["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
