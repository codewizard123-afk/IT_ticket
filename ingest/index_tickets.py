"""Ticket indexing (architecture §6.5).

* Embedding text: ``"{title}. {description}"`` (title ≠ description) else ``description``,
  **redacted** before embedding (CLAUDE.md rule 5). Labels are never part of the text.
* Embeddings come from ``store.embedder`` and are passed explicitly to Chroma.
* Metadata is scalar-only (lists → CSV, ``None`` → ``""``, dates → epoch ints).
* BM25 over the same text → ``indexes/bm25_tickets.pkl``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from shared.schemas import TicketRecord
from store.bm25_store import BM25Index
from store.chroma_store import BATCH, EmbedFn, record_to_metadata, reset_collection
from store.text import embedding_text, text_hash

log = logging.getLogger(__name__)

TextFn = Callable[[str], str]


def _identity(text: str) -> str:
    return text


def plain_record(raw: dict[str, Any]) -> TicketRecord:
    """A ``tickets.json`` row as a ``TicketRecord`` with no enrichment (P1-T1 fallback)."""
    meta = dict(raw.get("metadata") or {})
    origin = "synthetic_escalated" if meta.get("escalated") else "synthetic_kb_derived"
    if not raw.get("resolution"):
        origin = "none"
    return TicketRecord.model_validate(
        {
            "ticket_id": raw["id"],
            "title": raw.get("title") or "",
            "description": raw.get("description") or "",
            "category": raw.get("category"),
            "subcategory": raw.get("subcategory"),
            "priority": raw.get("priority"),
            "status": raw.get("status") or "closed",
            "source": meta.get("source"),
            "resolution": raw.get("resolution"),
            "resolution_origin": origin,
            "resolution_kb_id": meta.get("related_kb_id"),
            "collection": "tickets_hist",
            "metadata": meta,
        }
    )


def prepare(rec: TicketRecord, redact: TextFn = _identity) -> tuple[TicketRecord, str]:
    """Redact title/description, compute the embedding text and its hash."""
    title = redact(rec.title)
    desc = redact(rec.description)
    doc = embedding_text(title, desc)
    upd: dict[str, Any] = {"title": title, "description": desc, "text_hash": text_hash(doc)}
    if rec.resolution:
        upd["resolution"] = redact(rec.resolution)
    return rec.model_copy(update=upd), doc


def index_tickets(
    client: Any,
    embed_fn: EmbedFn,
    records: Sequence[TicketRecord],
    collection: str,
    *,
    redact: TextFn = _identity,
    bm25: BM25Index | None = None,
) -> int:
    """(Re)build one ticket collection. Adds every document to ``bm25`` if given."""
    coll = reset_collection(client, collection)
    prepared = [prepare(r, redact) for r in records]
    for i in range(0, len(prepared), BATCH):
        batch = prepared[i : i + BATCH]
        docs = [d for _, d in batch]
        vecs = embed_fn(docs)
        coll.add(
            ids=[r.ticket_id for r, _ in batch],
            embeddings=[v.tolist() for v in vecs],
            documents=docs,
            metadatas=[record_to_metadata(r) for r, _ in batch],
        )
    if bm25 is not None:
        for r, d in prepared:
            bm25.add(r.ticket_id, d)
    log.info("%s: indexed %d tickets", collection, len(prepared))
    return len(prepared)
