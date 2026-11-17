"""FastAPI dependencies. Tests override these via ``app.dependency_overrides``."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from backend import auth
from backend.db.sqlite import open_db
from backend.errors import ApiError
from backend.services import Services
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
    """``Authorization: Bearer <jwt>``; or, when ``DEMO_AUTH=1``, ``X-Demo-Role`` (+ group)."""
    services: Services = request.app.state.services
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
        return auth.decode_token(token, auth.signing_secret(services.settings.app_secret))
    role = request.headers.get("X-Demo-Role")
    if role and services.settings.demo_auth:
        return auth.demo_user(role, request.headers.get("X-Demo-Group") or None)
    raise ApiError(401, "unauthenticated", "log in or send X-Demo-Role (DEMO_AUTH=1)")


ServicesDep = Annotated[Services, Depends(get_services)]
DbDep = Annotated[sqlite3.Connection, Depends(get_db)]
UserDep = Annotated[User, Depends(get_current_user)]
