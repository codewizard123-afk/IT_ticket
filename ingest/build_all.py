"""``python -m ingest.build_all [--eval-split]`` — build every P1 index (architecture §6, §11.1).

Steps: KEDB (governed file, ``kedb_chunks``, error codes, BM25) → enrich ``tickets.json`` →
historical tickets (``tickets_hist``) → curated incidents (``tickets_verified``) → manifest
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
from ingest.make_seed import load_seed
from ingest.split import assign_splits
from privacy.redact import redact_text
from shared.schemas import TicketRecord
from store.bm25_store import BM25Index
from store.chroma_store import EmbedFn
from store.config import read_json, repo_path, write_json

log = logging.getLogger("ingest.build_all")

SOURCE_KEYS = ("tickets", "kb_articles", "seed_incidents")


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


def build_eval_collections(
    client: Any,
    embed_fn: EmbedFn,
    enriched: list[dict[str, Any]],
    seed: list[TicketRecord],
    *,
    root: Path | None = None,
) -> dict[str, int]:
    """``tickets_hist_eval`` (train split) + ``tickets_verified_eval`` + their BM25."""
    splits = assign_splits(enriched)
    train = [TicketRecord.model_validate(r) for r in enriched if splits[r["ticket_id"]] == "train"]
    bm25 = BM25Index()
    counts = {
        "tickets_hist_eval": index_tickets(
            client, embed_fn, train, "tickets_hist_eval", bm25=bm25, redact=redact_text
        ),
        "tickets_verified_eval": index_tickets(
            client, embed_fn, seed, "tickets_verified_eval", bm25=bm25, redact=redact_text
        ),
    }
    bm25.save(index_path("bm25_tickets_eval", root))
    counts["split_train"] = sum(1 for s in splits.values() if s == "train")
    counts["split_val"] = sum(1 for s in splits.values() if s == "val")
    counts["split_test"] = sum(1 for s in splits.values() if s == "test")
    return counts


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
    gov_path = repo_path("kb_governed", root)
    existing = load_articles(gov_path) if gov_path.exists() else None
    articles = govern(load_articles(repo_path("kb_articles", root)), existing)
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

    # 3. curated verified incidents (tickets_verified)
    seed = load_seed(root)
    counts["tickets_verified"] = index_tickets(
        client, embed_fn, seed, "tickets_verified", bm25=bm25, redact=redact_text
    )
    bm25.save(index_path("bm25_tickets", root))

    # 4. evaluation copies built from the 70% train split only (architecture §11.1, §14.1)
    if eval_split:
        counts.update(build_eval_collections(client, embed_fn, enriched, seed, root=root))

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
