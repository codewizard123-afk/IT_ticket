"""Dependency wiring (README §8.2): stores, engine, LLM client and redactor, chosen by env flags.

| Flag              | Effect                                           |
|-------------------|--------------------------------------------------|
| USE_STUB_STORES=1 | FakeTicketStore / FakeKBStore instead of Chroma  |
| USE_STUB_ENGINE=1 | FakeEngine instead of ``ai.engine``              |
| USE_STUB_REDACT=1 | fake_redact instead of ``privacy.redact``        |

Other owners' code is imported lazily here and nowhere else in ``backend/``, so a missing index
(``make data`` not run) turns into HTTP 503 instead of an import error.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend import llm_audit
from backend.errors import ApiError
from backend.settings import Settings
from shared.interfaces import Engine, KBStore, LLMClient, Redactor, TicketStore
from shared.schemas import KBArticle

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Services:
    settings: Settings
    engine: Engine
    llm: LLMClient
    redact: Redactor
    ticket_store: TicketStore | None = None
    kb_store: KBStore | None = None
    index_error: str | None = None
    clock: Callable[[], datetime] = utc_now
    # KEDB publication after a draft is approved (P3-T8); see backend/kb_drafts.py
    kb_publisher: Callable[[KBArticle], None] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def stores(self) -> tuple[TicketStore, KBStore]:
        if self.ticket_store is None or self.kb_store is None:
            raise ApiError(503, "index_missing", self.index_error or "indexes not built")
        return self.ticket_store, self.kb_store

    def kb_article(self, kb_id: str) -> KBArticle | None:
        if self.kb_store is None:
            return None
        try:
            return self.kb_store.get_article(kb_id)
        except (KeyError, LookupError, ValueError):
            return None

    @property
    def stub_engine(self) -> bool:
        return self.settings.use_stub_engine


def _redactor(settings: Settings) -> Redactor:
    if settings.use_stub_redact:
        from shared.stubs import fake_redact

        return fake_redact
    from privacy.redact import redact

    return redact


def _stores(settings: Settings) -> tuple[Any, Any, str | None]:
    if settings.use_stub_stores:
        from shared.stubs import FakeKBStore, FakeTicketStore

        return FakeTicketStore(), FakeKBStore(), None
    try:
        from store.factory import open_stores

        tickets, kb = open_stores()
        return tickets, kb, None
    except Exception as exc:  # noqa: BLE001 - IndexMissingError, chromadb missing, corrupt index
        log.warning(
            "real stores unavailable (%s); endpoints needing them return 503", type(exc).__name__
        )
        return None, None, f"stores unavailable: {type(exc).__name__}: run `make data`"


def _llm() -> LLMClient:
    try:
        from ai.llm.groq_client import make_llm_client

        return make_llm_client(on_call=llm_audit.record)
    except Exception as exc:  # noqa: BLE001
        log.warning("Groq client unavailable (%s); running offline", type(exc).__name__)
        from shared.stubs import FakeLLM

        return FakeLLM(available=False)


def _engine(settings: Settings, kb_store: Any) -> Engine:
    if settings.use_stub_engine:
        from shared.stubs import FakeEngine, FakeKBStore

        return FakeEngine(kb_store if isinstance(kb_store, FakeKBStore) else None)
    from ai.engine import AIEngine

    return AIEngine()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings.from_env()
    tickets, kb, err = _stores(settings)
    services = Services(
        settings=settings,
        engine=_engine(settings, kb),
        llm=_llm(),
        redact=_redactor(settings),
        ticket_store=tickets,
        kb_store=kb,
        index_error=err,
    )
    with contextlib.suppress(Exception):
        from backend.kb_drafts import default_publisher

        services.kb_publisher = default_publisher(services)
    return services
