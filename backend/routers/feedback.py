"""``POST /feedback``, ``POST /decisions`` and ``GET /internal/corrections`` (§8.8, §10)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend import access, feedback
from backend.deps import DbDep, ServicesDep, UserDep
from shared.schemas import Decision, Feedback

router = APIRouter(tags=["feedback"])
CORRECTION_ROLES = frozenset({"l2", "kb_owner", "admin"})


@router.post("/feedback", summary="Analyst feedback on one item of an analysis")
def post_feedback(
    body: Feedback, services: ServicesDep, conn: DbDep, user: UserDep
) -> dict[str, str]:
    return {"feedback_id": feedback.record_feedback(services, conn, body, user)}


@router.post("/decisions", summary="Final decision (assisted or manual mode)")
def post_decision(
    body: Decision, services: ServicesDep, conn: DbDep, user: UserDep
) -> dict[str, bool]:
    feedback.record_decision(services, conn, body, user)
    return {"ok": True}


@router.get("/internal/corrections", summary="Classification corrections (for P2 few-shot)")
def corrections(conn: DbDep, user: UserDep) -> dict[str, Any]:
    access.require(conn, user, CORRECTION_ROLES, "classification corrections")
    return {"items": feedback.corrections(conn)}


@router.get("/internal/kb-prior", summary="Current feedback prior per KB article")
def kb_prior(conn: DbDep, user: UserDep) -> dict[str, float]:
    access.require(conn, user, CORRECTION_ROLES, "kb prior")
    return feedback.kb_prior(conn)
