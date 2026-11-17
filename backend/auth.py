"""Authentication (architecture §8.6): bcrypt users from ``config/users.yaml``, HS256 JWT.

``APP_SECRET`` signs tokens. If it is unset (or still ``change-me``) a random per-process secret is
used, so tokens stop working after a restart; the secret itself is never logged.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import bcrypt
import jwt

from backend.config import users_config
from backend.errors import ApiError
from shared.constants import ROLES
from shared.schemas import User

log = logging.getLogger(__name__)
_EPHEMERAL_SECRET = secrets.token_urlsafe(48)
_warned = False


def signing_secret(app_secret: str | None) -> str:
    global _warned
    if app_secret:
        return app_secret
    if not _warned:
        log.warning("APP_SECRET is not set: using an ephemeral JWT secret for this process")
        _warned = True
    return _EPHEMERAL_SECRET


def _users(cfg: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    cfg = cfg if cfg is not None else users_config()
    return {u["username"]: u for u in cfg.get("users", [])}


def lookup(username: str, cfg: Mapping[str, Any] | None = None) -> User | None:
    u = _users(cfg).get(username)
    if u is None or u.get("role") not in ROLES:
        return None
    return User(username=u["username"], role=u["role"], assignment_group=u.get("assignment_group"))


def authenticate(username: str, password: str, cfg: Mapping[str, Any] | None = None) -> User:
    u = _users(cfg).get(username)
    ok = False
    if u is not None and password:
        try:
            ok = bcrypt.checkpw(password.encode("utf-8"), str(u["password_hash"]).encode("utf-8"))
        except ValueError:
            ok = False
    user = lookup(username, cfg) if ok else None
    if user is None:
        raise ApiError(401, "unauthenticated", "unknown user or wrong password")
    return user


def issue_token(
    user: User, secret: str, now: datetime, cfg: Mapping[str, Any] | None = None
) -> str:
    jcfg = (cfg if cfg is not None else users_config()).get("jwt", {})
    claims = {
        "sub": user.username,
        "role": user.role,
        "group": user.assignment_group,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=float(jcfg.get("ttl_hours", 8)))).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm=jcfg.get("algorithm", "HS256"))


def decode_token(token: str, secret: str, cfg: Mapping[str, Any] | None = None) -> User:
    cfg = cfg if cfg is not None else users_config()
    alg = cfg.get("jwt", {}).get("algorithm", "HS256")
    try:
        claims = jwt.decode(token, secret, algorithms=[alg], options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError as exc:
        raise ApiError(401, "unauthenticated", "token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise ApiError(401, "unauthenticated", "invalid token") from exc
    user = lookup(str(claims["sub"]), cfg)
    if user is None:
        raise ApiError(401, "unauthenticated", "unknown user")
    return user


def demo_user(role: str, group: str | None, cfg: Mapping[str, Any] | None = None) -> User:
    cfg = cfg if cfg is not None else users_config()
    if role not in ROLES:
        raise ApiError(401, "unauthenticated", f"unknown demo role {role!r}")
    default = cfg.get("demo", {}).get("default_group", {}).get(role)
    return User(username=f"demo-{role}", role=role, assignment_group=group or default)
