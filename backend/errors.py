"""Error model (architecture §10): ``{"error": {"code", "message", "request_id"}}``.

Every request gets a ``request_id`` (from ``X-Request-ID`` or a new uuid), echoed in the response
header and in error bodies. FastAPI's 422 validation errors become 400 ``validation``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

CODES = {
    400: "validation",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
    503: "index_missing",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def bad_request(message: str) -> ApiError:
    return ApiError(400, "validation", message)


def not_found(message: str) -> ApiError:
    return ApiError(404, "not_found", message)


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    rid = request_id(request)
    body = {"error": {"code": code, "message": message, "request_id": rid}}
    return JSONResponse(body, status_code=status, headers={"X-Request-ID": rid})


def _validation_message(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts) or "invalid request"


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _request_id(request: Request, call_next: Any) -> Any:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid[:64]
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(request, 400, "validation", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = CODES.get(exc.status_code, "http_error")
        return error_response(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # never include exception text: it may echo ticket content
        log.error("unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
        return error_response(request, 500, "internal", "internal error")
