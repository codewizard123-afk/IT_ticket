"""Audit endpoints (architecture §10). ``/audit/{analysis_id}`` and ``/audit/export`` follow later.

TODO(P3-T5): restrict ``/audit/verify`` to the ``admin`` role once auth/access land.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend import audit
from backend.deps import DbDep

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/verify", response_model=audit.VerifyResult)
def verify(conn: DbDep) -> audit.VerifyResult:
    """Walk the hash chain; ``ok=false`` with ``broken_at`` = first tampered row id."""
    return audit.verify_chain(conn)
