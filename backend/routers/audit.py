"""Audit endpoints (architecture §10). ``/audit/{analysis_id}`` and ``/audit/export`` follow later.

TODO(P3-T5): restrict ``/audit/verify`` to the ``admin`` role once auth/access land.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from backend import audit
from backend.deps import get_db

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/verify", response_model=audit.VerifyResult)
def verify(conn: sqlite3.Connection = Depends(get_db)) -> audit.VerifyResult:  # noqa: B008
    """Walk the hash chain; ``ok=false`` with ``broken_at`` = first tampered row id."""
    return audit.verify_chain(conn)
