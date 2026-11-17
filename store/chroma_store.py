"""ChromaDB-backed ``TicketStore`` and ``KBStore`` (architecture §6.4–§6.6, §11.1).

Rules this module enforces:

* Embeddings are always computed by ``store.embedder`` and passed explicitly; collections are
  created with ``embedding_function=None`` so Chroma's default model is never used.
* Metadata is scalar only: lists → CSV strings, ``None`` → ``""``, dates → int epoch seconds.
* ``where`` filters accept the convenient multi-key form ``{"a": 1, "b": 2}`` and are rewritten
  to Chroma's ``{"$and": [...]}``.
* KEDB resolution chunks are stored (for ``chunk_text`` citations) but never searched, and only
  ``status=approved`` chunks are searchable.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from shared.constants import SEARCHABLE_CHUNK_TYPES
from shared.interfaces import DEFAULT_TICKET_COLLECTIONS
from shared.schemas import ChunkHit, KBArticle, TicketHit, TicketRecord
from store.bm25_store import BM25Index
from store.text import embedding_text, norm_code, text_hash

EmbedFn = Callable[[Sequence[str]], Any]

KB_COLLECTION = "kedb_chunks"
EXCERPT_CHARS = 200
# Chroma's maximum batch is ~5k; stay well below it.
BATCH = 1000

# TicketRecord fields stored as Chroma metadata (§11.1). Lists become CSV strings.
TICKET_META_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "title",
    "description",
    "category",
    "subcategory",
    "priority",
    "status",
    "source",
    "resolution",
    "resolution_origin",
    "resolution_kb_id",
    "opened_at",
    "resolved_at",
    "opened_ts",
    "resolved_ts",
    "assignment_group",
    "application",
    "environment",
    "location",
    "restricted",
    "verified",
    "verified_by",
    "collection",
    "derived_fields",
    "dup_group_id",
    "text_hash",
)
# Extra keys copied from TicketRecord.metadata into Chroma metadata when present.
TICKET_EXTRA_META: tuple[str, ...] = (
    "intent",
    "split",
    "scenario_id",
    "error_code",
    "affected_users",
    "escalated",
    "resolved_by_tier",
    "text_quality",
    "verified_ts",
    "analysis_id",
    "created_ts",
    "live_status",
    "kb_ids",
)
_INT_FIELDS = frozenset({"opened_ts", "resolved_ts", "created_ts", "verified_ts", "affected_users"})
_BOOL_FIELDS = frozenset({"restricted", "verified", "escalated"})
_LIST_FIELDS = frozenset({"derived_fields", "kb_ids"})


# =============================================================================================
# Helpers
# =============================================================================================
def open_client(path: Path | str):  # noqa: ANN201 - chromadb.ClientAPI
    """A persistent Chroma client with telemetry off."""
    import chromadb
    from chromadb.config import Settings

    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def get_collection(client: Any, name: str, *, create: bool = True):  # noqa: ANN201
    if create:
        return client.get_or_create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )
    return client.get_collection(name, embedding_function=None)


def reset_collection(client: Any, name: str):  # noqa: ANN201
    # "does not exist" errors differ across chromadb versions
    with contextlib.suppress(Exception):
        client.delete_collection(name)
    return get_collection(client, name)


def collection_exists(client: Any, name: str) -> bool:
    try:
        client.get_collection(name, embedding_function=None)
    except Exception:  # noqa: BLE001
        return False
    return True


def normalize_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Rewrite ``{"a": 1, "b": {"$in": [..]}}`` into Chroma's single-operator form."""
    if not where:
        return None
    clauses: list[dict[str, Any]] = []
    for key, cond in where.items():
        if key in ("$and", "$or"):
            subs = [w for w in (normalize_where(c) for c in cond) if w]
            if not subs:
                continue
            clauses.append(subs[0] if len(subs) == 1 else {key: subs})
        else:
            clauses.append({key: cond})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def and_where(*wheres: dict[str, Any] | None) -> dict[str, Any] | None:
    parts = [w for w in (normalize_where(w) for w in wheres) if w]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _scalar(key: str, value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if key in _LIST_FIELDS or isinstance(value, list | tuple | set):
        return ",".join(str(v) for v in value)
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def record_to_metadata(rec: TicketRecord) -> dict[str, str | int | float | bool]:
    """Flatten a ``TicketRecord`` into Chroma-safe scalar metadata (§6.5)."""
    data = rec.model_dump()
    meta = {k: _scalar(k, data.get(k)) for k in TICKET_META_FIELDS}
    for k in TICKET_EXTRA_META:
        if k in rec.metadata:
            meta[k] = _scalar(k, rec.metadata[k])
    for k in _INT_FIELDS & meta.keys():
        if meta[k] == "":
            meta[k] = 0
    return meta


def _unscalar(key: str, value: Any) -> Any:
    if key in _LIST_FIELDS:
        return [v for v in str(value).split(",") if v] if value else []
    if key in _BOOL_FIELDS:
        return bool(value)
    if key in _INT_FIELDS:
        return int(value) if value not in ("", None, 0) else None
    return None if value == "" else value


def metadata_to_record(meta: dict[str, Any], document: str | None = None) -> TicketRecord:
    fields = {k: _unscalar(k, meta.get(k)) for k in TICKET_META_FIELDS if k in meta}
    extra = {k: _unscalar(k, meta[k]) for k in TICKET_EXTRA_META if k in meta}
    fields["restricted"] = bool(meta.get("restricted", False))
    fields["verified"] = bool(meta.get("verified", False))
    fields["resolution_origin"] = meta.get("resolution_origin") or "none"
    fields.setdefault("title", "")
    fields.setdefault("description", "")
    rec = TicketRecord.model_validate({**fields, "metadata": {**meta, **extra}})
    # P2's search reads ``hit.text`` on records returned by ``get()``.
    rec.text = document or embedding_text(rec.title, rec.description)  # type: ignore[attr-defined]
    return rec


def _hit_from(meta: dict[str, Any], document: str | None, score: float, coll: str) -> TicketHit:
    rec = metadata_to_record(meta, document)
    hit = TicketHit(
        ticket_id=rec.ticket_id,
        score=round(max(0.0, min(1.0, score)), 4),
        title=rec.title,
        excerpt=(rec.description or "")[:EXCERPT_CHARS],
        category=rec.category,
        subcategory=rec.subcategory,
        priority=rec.priority,
        status=rec.status,
        resolution=rec.resolution,
        resolution_origin=rec.resolution_origin,
        collection=coll,
        metadata=dict(meta),
    )
    hit.text = rec.text  # type: ignore[attr-defined]
    return hit


def _batched(seq: Sequence[Any], n: int = BATCH) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# =============================================================================================
# Ticket store
# =============================================================================================
class ChromaTicketStore:
    """``TicketStore`` (shared/interfaces.py) over Chroma + a BM25 index.

    ``suffix`` maps the logical collection names onto physical ones, e.g. ``"_eval"`` makes
    ``tickets_hist`` read ``tickets_hist_eval`` (architecture §11.1, evaluation only).
    """

    def __init__(
        self,
        client: Any,
        embed_fn: EmbedFn,
        bm25: BM25Index | None = None,
        *,
        suffix: str = "",
        bm25_path: Path | None = None,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self.client = client
        self.embed_fn = embed_fn
        self.bm25_index = bm25 if bm25 is not None else BM25Index()
        self.suffix = suffix
        self.bm25_path = bm25_path
        self._redact = redact

    def _phys(self, logical: str) -> str:
        return f"{logical}{self.suffix}"

    def _coll(self, logical: str, create: bool = False):  # noqa: ANN202
        name = self._phys(logical)
        if not create and not collection_exists(self.client, name):
            return None
        return get_collection(self.client, name)

    # --- TicketStore protocol ------------------------------------------------------------
    def query(
        self,
        text: str,
        k: int,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = DEFAULT_TICKET_COLLECTIONS,
    ) -> list[TicketHit]:
        if k <= 0 or not text.strip():
            return []
        qvec = self.embed_fn([text])[0].tolist()
        hits: list[TicketHit] = []
        cw = normalize_where(where)
        for logical in collections:
            coll = self._coll(logical)
            if coll is None:
                continue
            n = coll.count()
            if n == 0:
                continue
            res = coll.query(
                query_embeddings=[qvec],
                n_results=min(k, n),
                where=cw,
                include=["documents", "metadatas", "distances"],
            )
            for meta, doc, dist in zip(
                res["metadatas"][0], res["documents"][0], res["distances"][0], strict=True
            ):
                hits.append(_hit_from(meta, doc, 1.0 - float(dist), logical))
        hits.sort(key=lambda h: (-h.score, h.ticket_id))
        return hits[:k]

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.bm25_index.search(text, k)

    def get(self, ids: list[str]) -> list[TicketRecord]:
        if not ids:
            return []
        found: dict[str, TicketRecord] = {}
        # verified/live win over hist when the same id exists in several collections
        for logical in ("tickets_hist", "tickets_live", "tickets_verified"):
            coll = self._coll(logical)
            if coll is None:
                continue
            res = coll.get(ids=list(ids), include=["metadatas", "documents"])
            for tid, meta, doc in zip(res["ids"], res["metadatas"], res["documents"], strict=True):
                rec = metadata_to_record(meta, doc)
                rec.collection = logical
                found[tid] = rec
        return [found[i] for i in ids if i in found]

    def upsert_live(self, rec: TicketRecord) -> None:
        """Add/replace a ticket analysed in the app (``tickets_live``). Text must be redacted."""
        now = int(time.time())
        title = self._redact(rec.title) if self._redact else rec.title
        desc = self._redact(rec.description) if self._redact else rec.description
        doc = embedding_text(title, desc)
        meta_extra = {
            "created_ts": rec.metadata.get("created_ts") or now,
            "live_status": rec.metadata.get("live_status") or rec.status or "new",
        }
        live = rec.model_copy(
            update={
                "title": title,
                "description": desc,
                "collection": "tickets_live",
                "text_hash": rec.text_hash or text_hash(doc),
                "opened_ts": rec.opened_ts or now,
                "metadata": {**rec.metadata, **meta_extra},
            }
        )
        meta = record_to_metadata(live)
        coll = self._coll("tickets_live", create=True)
        coll.upsert(
            ids=[live.ticket_id],
            embeddings=[self.embed_fn([doc])[0].tolist()],
            documents=[doc],
            metadatas=[meta],
        )
        self.bm25_index.add(live.ticket_id, doc)
        self._save_bm25()

    def mark_verified(self, ticket_id: str, resolution: str, by: str) -> None:
        """Promote a ticket to an ``analyst_verified`` incident in ``tickets_verified``.

        A live ticket is moved (removed from ``tickets_live``); a historical one is copied, so
        the historical corpus stays intact.
        """
        recs = self.get([ticket_id])
        if not recs:
            raise KeyError(f"unknown ticket {ticket_id!r}")
        src = recs[0]
        now = int(time.time())
        clean = self._redact(resolution) if self._redact else resolution
        ver = src.model_copy(
            update={
                "collection": "tickets_verified",
                "resolution": clean,
                "resolution_origin": "analyst_verified",
                "resolution_kb_id": None,
                "verified": True,
                "verified_by": by,
                "status": "resolved",
                "resolved_ts": now,
                "metadata": {**src.metadata, "verified_ts": now},
            }
        )
        doc = embedding_text(ver.title, ver.description)
        coll = self._coll("tickets_verified", create=True)
        coll.upsert(
            ids=[ticket_id],
            embeddings=[self.embed_fn([doc])[0].tolist()],
            documents=[doc],
            metadatas=[record_to_metadata(ver)],
        )
        if src.collection == "tickets_live":
            live = self._coll("tickets_live")
            if live is not None:
                live.delete(ids=[ticket_id])
        self.bm25_index.add(ticket_id, doc)
        self._save_bm25()

    # --- extras --------------------------------------------------------------------------
    def count(self, logical: str) -> int:
        coll = self._coll(logical)
        return coll.count() if coll is not None else 0

    def list(
        self,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = DEFAULT_TICKET_COLLECTIONS,
        limit: int | None = None,
    ) -> list[TicketRecord]:
        """Filter-only listing (no query text), newest ``opened_ts`` first. Used by FR-07."""
        out: list[TicketRecord] = []
        cw = normalize_where(where)
        for logical in collections:
            coll = self._coll(logical)
            if coll is None:
                continue
            res = coll.get(where=cw, include=["metadatas", "documents"])
            for meta, doc in zip(res["metadatas"], res["documents"], strict=True):
                rec = metadata_to_record(meta, doc)
                rec.collection = logical
                out.append(rec)
        out.sort(key=lambda r: (-(r.opened_ts or 0), r.ticket_id))
        return out[:limit] if limit is not None else out

    def _save_bm25(self) -> None:
        if self.bm25_path is not None:
            self.bm25_index.save(self.bm25_path)


# =============================================================================================
# KEDB store
# =============================================================================================
class ChromaKBStore:
    """``KBStore`` over the ``kedb_chunks`` collection, a chunk BM25 and the error-code index."""

    def __init__(
        self,
        client: Any,
        embed_fn: EmbedFn,
        articles: Iterable[KBArticle],
        *,
        bm25: BM25Index | None = None,
        code_index: dict[str, list[str]] | None = None,
        collection: str = KB_COLLECTION,
    ) -> None:
        from ingest.chunks import build_chunks, build_code_index  # local: avoid import cycle

        self.client = client
        self.embed_fn = embed_fn
        self.collection_name = collection
        self._articles: dict[str, KBArticle] = {a.kb_id: a for a in articles}
        self._chunks: dict[str, dict[str, Any]] = {
            c["id"]: c for a in self._articles.values() for c in build_chunks(a)
        }
        self.code_index = (
            code_index if code_index is not None else build_code_index(self._articles.values())
        )
        if bm25 is None:
            bm25 = BM25Index.build(
                [c["id"] for c in self._searchable_chunks()],
                [c["text"] for c in self._searchable_chunks()],
            )
        self.bm25_index = bm25

    def _searchable_chunks(self) -> list[dict[str, Any]]:
        return [
            c
            for c in self._chunks.values()
            if c["metadata"]["searchable"] and c["metadata"]["status"] == "approved"
        ]

    def _coll(self):  # noqa: ANN202
        return get_collection(self.client, self.collection_name)

    # --- KBStore protocol ----------------------------------------------------------------
    def query_chunks(
        self, text: str, k: int, where: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        if k <= 0 or not text.strip():
            return []
        coll = self._coll()
        n = coll.count()
        if n == 0:
            return []
        base = {"status": "approved", "chunk_type": {"$in": list(SEARCHABLE_CHUNK_TYPES)}}
        res = coll.query(
            query_embeddings=[self.embed_fn([text])[0].tolist()],
            n_results=min(k, n),
            where=and_where(base, where),
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            ChunkHit(
                chunk_id=cid,
                kb_id=meta["kb_id"],
                chunk_type=meta["chunk_type"],
                text=doc,
                score=round(max(0.0, min(1.0, 1.0 - float(dist))), 4),
                metadata=dict(meta),
            )
            for cid, meta, doc, dist in zip(
                res["ids"][0],
                res["metadatas"][0],
                res["documents"][0],
                res["distances"][0],
                strict=True,
            )
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return self.bm25_index.search(text, k)

    def lookup_codes(self, codes: list[str]) -> dict[str, list[str]]:
        """Map each code found in a ticket to KB ids: exact normalised match first, then
        containment either way (≥ 5 chars) so ``"18456"`` finds ``"SQL Server error 18456"``."""
        out: dict[str, list[str]] = {}
        for code in codes:
            n = norm_code(code)
            if not n:
                continue
            kb_ids = list(self.code_index.get(n, []))
            if not kb_ids and len(n) >= 5:
                for key, ids in self.code_index.items():
                    if len(key) >= 5 and (_token_in(n, key) or _token_in(key, n)):
                        kb_ids.extend(i for i in ids if i not in kb_ids)
            approved = [i for i in kb_ids if self._is_approved(i)]
            if approved:
                out[code] = sorted(set(approved))
        return out

    def get_article(self, kb_id: str) -> KBArticle:
        try:
            return self._articles[kb_id]
        except KeyError:
            raise KeyError(f"unknown KB article {kb_id!r}") from None

    def chunk_text(self, chunk_id: str) -> str:
        try:
            return self._chunks[chunk_id]["text"]
        except KeyError:
            raise KeyError(f"unknown chunk {chunk_id!r}") from None

    def all_error_phrases(self) -> list[str]:
        phrases = {
            c
            for a in self._articles.values()
            if a.status == "approved"
            for c in a.common_error_codes
            if c.strip()
        }
        return sorted(phrases, key=lambda p: (-len(p), p))

    # --- extras --------------------------------------------------------------------------
    def articles(self, include_unapproved: bool = False) -> list[KBArticle]:
        return [
            a
            for a in sorted(self._articles.values(), key=lambda a: a.kb_id)
            if include_unapproved or a.status == "approved"
        ]

    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def refresh_article(self, article: KBArticle) -> None:
        """Update the in-memory view after ``ingest.index_kedb.reindex_article``."""
        from ingest.chunks import build_chunks, build_code_index

        self._articles[article.kb_id] = article
        prefix = f"{article.kb_id}#"
        self._chunks = {k: v for k, v in self._chunks.items() if not k.startswith(prefix)}
        new_chunks = build_chunks(article)
        self._chunks.update({c["id"]: c for c in new_chunks})
        self.code_index = build_code_index(self._articles.values())
        self.bm25_index.remove_where(lambda d: d.startswith(prefix))
        if article.status == "approved":
            for c in new_chunks:
                if c["metadata"]["searchable"]:
                    self.bm25_index.add(c["id"], c["text"])

    def _is_approved(self, kb_id: str) -> bool:
        art = self._articles.get(kb_id)
        return art is not None and art.status == "approved"


def _token_in(needle: str, hay: str) -> bool:
    """``needle`` appears in ``hay`` as a whole-token run (e.g. ``"error 18456"``)."""
    return f" {needle} " in f" {hay} "
