"""KEDB endpoints: search, article, drafts and approval (architecture §8.9, §10).

``/kb/search`` delegates ranking to P2's ``ai.kb_search.search_kb`` with the current feedback prior.
Only approved articles are searchable; drafts are visible to l2/kb_owner/admin via ``/kb/drafts``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend import access, feedback, kb_drafts
from backend.deps import DbDep, ServicesDep, UserDep
from backend.errors import not_found

router = APIRouter(prefix="/kb", tags=["kb"])


class DraftIn(BaseModel):
    analysis_id: str


class ApproveIn(BaseModel):
    edits: dict[str, Any] | None = None


def _search(
    services: Any, q: str, filters: dict[str, Any], k: int, prior: dict[str, float]
) -> list[Any]:
    from ai.kb_search import search_kb  # P2's public search API (README §7, P3-T9)

    _, kb_store = services.stores()
    return search_kb(q, filters, k, kb_store, feedback_prior=prior)


@router.get("/search", summary="Search approved KEDB articles")
def search(
    services: ServicesDep,
    conn: DbDep,
    user: UserDep,
    q: str = "",
    category: str | None = None,
    subcategory: str | None = None,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict[str, Any]:
    _, kb_store = services.stores()
    filters = {k: v for k, v in {"category": category, "subcategory": subcategory}.items() if v}
    by_id = {a.kb_id: a for a in kb_drafts.all_articles(kb_store, include_unapproved=False)}
    items: list[dict[str, Any]] = []
    if q.strip():
        for hit in _search(services, q, filters, size, feedback.kb_prior(conn)):
            art = by_id.get(hit.kb_id)
            if art is None:
                continue
            items.append(
                hit.model_dump(mode="json") | {"source": art.source, "version": art.version}
            )
    else:
        for art in by_id.values():
            if any(getattr(art, k) != v for k, v in filters.items()):
                continue
            items.append(
                {
                    "kb_id": art.kb_id,
                    "title": art.title,
                    "score": 1.0,
                    "matched_field": None,
                    "excerpt": (art.root_cause or "")[:200] or None,
                    "why": None,
                    "requires_admin": art.resolution.requires_admin,
                    "category": art.category,
                    "subcategory": art.subcategory,
                    "source": art.source,
                    "version": art.version,
                }  # fmt: skip
            )
        items = items[:size]
    return {"items": items}


@router.get("/drafts", summary="KB drafts (l2, kb_owner, admin)")
def list_drafts(conn: DbDep, user: UserDep, status: str | None = None) -> dict[str, Any]:
    access.require(conn, user, access.DRAFT_ROLES, "kb drafts")
    return {"items": kb_drafts.list_drafts(conn, status)}


@router.post("/drafts", summary="Draft a KB article from the analyst's notes")
def create_draft(
    body: DraftIn, services: ServicesDep, conn: DbDep, user: UserDep
) -> dict[str, Any]:
    return kb_drafts.create_draft(services, conn, body.analysis_id, user)


@router.post("/drafts/{draft_id}/approve", summary="Approve a draft (kb_owner, admin)")
def approve(
    draft_id: str, body: ApproveIn, services: ServicesDep, conn: DbDep, user: UserDep
) -> dict[str, Any]:
    art = kb_drafts.approve_draft(services, conn, draft_id, body.edits, user)
    return art.model_dump(mode="json")


@router.get("/{kb_id}", summary="One KEDB article")
def get_article(kb_id: str, services: ServicesDep, conn: DbDep, user: UserDep) -> dict[str, Any]:
    services.stores()
    art = services.kb_article(kb_id)
    if art is None:
        raise not_found(f"article {kb_id}")
    if art.status != "approved" and user.role not in access.APPROVE_ROLES:
        raise not_found(f"article {kb_id}")
    return art.model_dump(mode="json")
