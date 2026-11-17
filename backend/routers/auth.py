"""``POST /auth/login`` (architecture §8.6, §10)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from backend import audit, auth
from backend.deps import DbDep, ServicesDep, UserDep

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/login", summary="Exchange username/password for an 8 h JWT")
def login(body: LoginIn, services: ServicesDep, conn: DbDep) -> dict[str, Any]:
    user = auth.authenticate(body.username, body.password)
    token = auth.issue_token(
        user, auth.signing_secret(services.settings.app_secret), services.clock()
    )
    audit.append(
        conn, actor=user.username, action="auth.login", entity="user", entity_id=user.username,
        payload={"role": user.role},
    )  # fmt: skip
    return {
        "token": token,
        "role": user.role,
        "assignment_group": user.assignment_group,
        "username": user.username,
        "demo_auth": services.settings.demo_auth,
    }


@router.get("/me", summary="The caller as the API sees it")
def me(user: UserDep) -> dict[str, Any]:
    return user.model_dump()
