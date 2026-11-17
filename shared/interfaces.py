"""Module boundaries as Protocols (architecture §6.6, §7).

Code depends on these, never on another owner's concrete classes. Real implementations
(P1 ``store/``, P2 ``ai/``) and the fakes in ``shared/stubs.py`` must both satisfy them.
Frozen after Phase 0 (README §8.3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from shared.schemas import (
    ChunkHit,
    EngineContext,
    EngineResult,
    KBArticle,
    RedactionResult,
    SanitizedTicket,
    TicketHit,
    TicketRecord,
)

if TYPE_CHECKING:
    import numpy as np

DEFAULT_TICKET_COLLECTIONS: tuple[str, ...] = ("tickets_hist", "tickets_verified", "tickets_live")


class LLMUnavailableError(RuntimeError):
    """Raised by an ``LLMClient`` when no answer can be produced (no key, breaker open, 429s).

    Callers must catch it and use their deterministic fallback (CLAUDE.md rule 6).
    """


@runtime_checkable
class TicketStore(Protocol):
    """Ticket vector store (P1, ``store/chroma_store.py``)."""

    def query(
        self,
        text: str,
        k: int,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = DEFAULT_TICKET_COLLECTIONS,
    ) -> list[TicketHit]: ...

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]: ...

    def get(self, ids: list[str]) -> list[TicketRecord]: ...

    def upsert_live(self, rec: TicketRecord) -> None: ...

    def mark_verified(self, ticket_id: str, resolution: str, by: str) -> None: ...


@runtime_checkable
class KBStore(Protocol):
    """KEDB chunk store (P1, ``store/chroma_store.py`` + ``bm25_store.py``)."""

    def query_chunks(
        self, text: str, k: int, where: dict[str, Any] | None = None
    ) -> list[ChunkHit]: ...

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:  # chunk ids
        ...

    def lookup_codes(self, codes: list[str]) -> dict[str, list[str]]:  # code -> kb_ids
        ...

    def get_article(self, kb_id: str) -> KBArticle: ...

    def chunk_text(self, chunk_id: str) -> str:  # e.g. "KB-0018#res.1"
        ...

    def all_error_phrases(self) -> list[str]: ...


@runtime_checkable
class LLMClient(Protocol):
    """Groq wrapper (P2, ``ai/llm/groq_client.py``). JSON mode, temperature 0.

    ``complete_json`` returns the parsed JSON object or raises ``LLMUnavailableError``.
    ``purpose`` is e.g. ``"classify"``, ``"synthesize"``, ``"kb_draft"`` and is logged to
    ``llm_calls`` through the ``on_call`` callback P3 injects.
    """

    @property
    def available(self) -> bool: ...

    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]: ...


@runtime_checkable
class Engine(Protocol):
    """The AI engine entry point (P2, ``ai/engine.py``)."""

    def run(self, ticket: SanitizedTicket, ctx: EngineContext) -> EngineResult: ...


@runtime_checkable
class Redactor(Protocol):
    """Privacy redaction (P1, ``privacy/redact.py``). Plain functions satisfy this too."""

    def __call__(self, text: str, keep_map: bool = True) -> RedactionResult: ...


# ``store.embedder.embed``: one shared, cached embedder (the only allowed global).
EmbedFn = Callable[[Sequence[str]], "np.ndarray"]
