"""``GET /health`` (architecture §10): status, index staleness, LLM availability, versions."""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter

from backend.config import backend_config
from backend.deps import ServicesDep
from shared.constants import PIPELINE_VERSION

router = APIRouter(tags=["health"])


def _index_state(services: Any) -> tuple[bool, dict[str, Any]]:
    if services.settings.use_stub_stores:
        return False, {}
    if services.ticket_store is None or services.kb_store is None:
        return True, {}
    stale, manifest = False, {}
    with contextlib.suppress(Exception):
        from store.factory import read_manifest

        manifest = read_manifest()
    with contextlib.suppress(Exception):
        from ingest.build_all import index_stale

        stale = bool(index_stale())
    return stale or not manifest, manifest


@router.get("/health", summary="Liveness, index staleness and LLM availability (no auth)")
def health(services: ServicesDep) -> dict[str, Any]:
    stale, manifest = _index_state(services)
    stores_ok = services.ticket_store is not None and services.kb_store is not None
    return {
        "status": "ok" if stores_ok else "degraded",
        "index_stale": stale,
        "llm_available": bool(getattr(services.llm, "available", False)),
        "versions": {
            "pipeline": PIPELINE_VERSION,
            "backend": str(backend_config().get("backend_version", "")),
            "index_built_at": str(manifest.get("built_at", "")),
            "embed_model": str(manifest.get("model", "")),
        },
        "stores": "stub" if services.settings.use_stub_stores else "chroma",
        "engine": type(services.engine).__name__,
        "demo_auth": services.settings.demo_auth,
        "index_error": services.index_error,
    }
