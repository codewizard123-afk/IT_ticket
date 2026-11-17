"""FastAPI dependencies. Tests override these via ``app.dependency_overrides``."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from backend.db.sqlite import open_db
from backend.errors import ApiError
from backend.services import Services
from shared.constants import ROLES
from shared.schemas import User


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    """One connection per request to the configured ``app.db``."""
    services: Services = request.app.state.services
    conn = open_db(services.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_current_user(request: Request) -> User:
    """Demo auth: ``X-Demo-Role`` (+ optional ``X-Demo-Group``) when ``DEMO_AUTH=1``."""
    services: Services = request.app.state.services
    role = request.headers.get("X-Demo-Role")
    if services.settings.demo_auth and role in ROLES:
        group = request.headers.get("X-Demo-Group") or None
        return User(username=f"demo-{role}", role=role, assignment_group=group)
    raise ApiError(401, "unauthenticated", "log in or send X-Demo-Role")


ServicesDep = Annotated[Services, Depends(get_services)]
DbDep = Annotated[sqlite3.Connection, Depends(get_db)]
UserDep = Annotated[User, Depends(get_current_user)]
