"""Typed client for every backend endpoint (architecture §10). The UI talks to the API only here.

``MOCK_API=1`` routes calls to the in-process :class:`ui.mock_backend.MockBackend` (golden
fixtures), so no server is needed. Otherwise calls go over HTTP to ``API_BASE_URL``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from shared.schemas import (
    AnalysisContract,
    AnalysisResponse,
    Decision,
    Event,
    Feedback,
    KBArticle,
    TicketIn,
)
from ui.mock_backend import MockBackend
from ui.settings import Settings, get_settings

View = Literal["full", "contract"]


class ApiError(Exception):
    """A non-2xx answer, carrying the §10 error model (``code``, ``message``)."""

    def __init__(self, status: int, code: str, message: str, request_id: str | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id


@dataclass
class RawResponse:
    status: int
    content_type: str
    json_body: Any = None
    text: str = ""


Transport = Callable[[str, str, dict[str, Any], Any, dict[str, str]], RawResponse]


def http_transport(base_url: str, timeout_s: float) -> Transport:
    http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def send(method: str, path: str, params: dict[str, Any], body: Any, headers: dict[str, str]):
        try:
            r = http.request(method, path, params=params or None, json=body, headers=headers)
        except httpx.HTTPError as err:
            raise ApiError(503, "unreachable", f"API not reachable at {base_url}: {err}") from err
        ctype = r.headers.get("content-type", "").split(";")[0]
        if ctype == "application/json":
            return RawResponse(r.status_code, ctype, json_body=r.json(), text=r.text)
        return RawResponse(r.status_code, ctype, text=r.text)

    return send


def mock_transport(backend: MockBackend) -> Transport:
    def send(method: str, path: str, params: dict[str, Any], body: Any, headers: dict[str, str]):
        resp = backend.handle(method, path, params=params, body=body, headers=headers)
        if resp.content_type == "application/json":
            return RawResponse(resp.status, resp.content_type, json_body=resp.body)
        return RawResponse(resp.status, resp.content_type, text=str(resp.body))

    return send


class ApiClient:
    """One method per endpoint. Auth is a JWT (``login``) or the demo-role header."""

    def __init__(
        self,
        transport: Transport,
        *,
        demo_role: str | None = None,
        demo_group: str | None = None,
        settings: Settings | None = None,
        mock: bool = False,
    ) -> None:
        self._send = transport
        self.settings = settings or get_settings()
        self.mock = mock
        self.token: str | None = None
        self.username: str | None = None
        self.role: str | None = demo_role
        self.assignment_group: str | None = demo_group

    # ------------------------------------------------------------------ plumbing
    @classmethod
    def from_env(cls, **kwargs: Any) -> ApiClient:
        s = get_settings()
        if s.mock_api:
            return cls(mock_transport(MockBackend()), settings=s, mock=True, **kwargs)
        return cls(http_transport(s.api_base_url, s.timeout_s), settings=s, **kwargs)

    def _headers(self) -> dict[str, str]:
        api = self.settings.config["api"]
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        headers = {}
        if self.role:
            headers[api["demo_role_header"]] = self.role
        if self.assignment_group:
            headers[api["demo_group_header"]] = self.assignment_group
        return headers

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> RawResponse:
        clean = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        resp = self._send(method, path, clean, body, self._headers())
        if resp.status >= 400:
            err = (resp.json_body or {}).get("error", {}) if resp.json_body else {}
            raise ApiError(
                resp.status,
                err.get("code", "http_error"),
                err.get("message", resp.text[:200]),
                err.get("request_id"),
            )
        return resp

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        return self._call(method, path, **kw).json_body

    # ------------------------------------------------------------------ auth / health
    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def login(self, username: str, password: str) -> dict[str, Any]:
        out = self._json("POST", "/auth/login", body={"username": username, "password": password})
        self.token = out["token"]
        self.username = out.get("username", username)
        self.role = out["role"]
        self.assignment_group = out.get("assignment_group")
        return out

    def use_demo_role(self, role: str, group: str | None = None) -> None:
        self.token = None
        self.username = f"demo-{role}"
        self.role = role
        self.assignment_group = group

    # ------------------------------------------------------------------ analysis
    def analyze(self, ticket: TicketIn | dict[str, Any]) -> AnalysisResponse:
        body = _ticket_body(ticket)
        data = self._json("POST", "/analyze", params={"view": "full"}, body=body)
        return AnalysisResponse.model_validate(data)

    def analyze_contract(self, ticket: TicketIn | dict[str, Any]) -> AnalysisContract:
        body = _ticket_body(ticket)
        data = self._json("POST", "/analyze", params={"view": "contract"}, body=body)
        return AnalysisContract.model_validate(data)

    def clarify(self, ticket_id: str, analysis_id: str, answer: str) -> AnalysisResponse:
        data = self._json(
            "POST",
            f"/tickets/{ticket_id}/clarify",
            body={"analysis_id": analysis_id, "answer": answer},
        )
        return AnalysisResponse.model_validate(data)

    # ------------------------------------------------------------------ search
    def search_tickets(self, q: str = "", page: int = 1, size: int = 20, **filters: Any):
        params = {"q": q, "page": page, "size": size, **filters}
        return self._json("GET", "/tickets/search", params=params)

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        return self._json("GET", f"/tickets/{ticket_id}")

    def search_kb(
        self, q: str = "", category: str | None = None, subcategory: str | None = None
    ) -> dict[str, Any]:
        params = {"q": q, "category": category, "subcategory": subcategory}
        return self._json("GET", "/kb/search", params=params)

    def get_kb(self, kb_id: str) -> KBArticle:
        return KBArticle.model_validate(self._json("GET", f"/kb/{kb_id}"))

    # ------------------------------------------------------------------ feedback / decisions
    def feedback(self, fb: Feedback | dict[str, Any]) -> str:
        body = Feedback.model_validate(fb).model_dump(exclude_none=True)
        return self._json("POST", "/feedback", body=body)["feedback_id"]

    def decide(self, decision: Decision | dict[str, Any]) -> bool:
        body = Decision.model_validate(decision).model_dump(exclude_none=True)
        return bool(self._json("POST", "/decisions", body=body).get("ok"))

    def event(
        self, ticket_id: str, event: str, analysis_id: str | None = None, mode: str = "assisted"
    ) -> bool:
        body = Event(ticket_id=ticket_id, analysis_id=analysis_id, event=event, mode=mode)
        return bool(self._json("POST", "/events", body=body.model_dump()).get("ok"))

    # ------------------------------------------------------------------ review / escalation
    def review_queue(self, status: str = "open", page: int = 1) -> dict[str, Any]:
        return self._json("GET", "/review-queue", params={"status": status, "page": page})

    def escalation_bundle(self, bundle_id: str, fmt: Literal["json", "md"] = "json") -> Any:
        resp = self._call("GET", f"/escalations/{bundle_id}", params={"fmt": fmt})
        return resp.json_body if fmt == "json" else resp.text

    # ------------------------------------------------------------------ audit
    def audit(self, analysis_id: str) -> dict[str, Any]:
        return self._json("GET", f"/audit/{analysis_id}")

    def audit_export(
        self, fmt: Literal["csv", "jsonl"] = "jsonl", date_from: str | None = None, date_to=None
    ) -> str:
        params = {"format": fmt, "from": date_from, "to": date_to}
        return self._call("GET", "/audit/export", params=params).text

    def audit_verify(self) -> dict[str, Any]:
        return self._json("GET", "/audit/verify")

    # ------------------------------------------------------------------ metrics / trial
    def metrics_mttr(
        self,
        group_by: str = "category",
        metric: str | None = None,
        mode: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        params = {
            "group_by": group_by,
            "metric": metric,
            "mode": mode,
            "from": date_from,
            "to": date_to,
        }
        return self._json("GET", "/metrics/mttr", params=params)

    def metrics_summary(self) -> dict[str, Any]:
        return self._json("GET", "/metrics/summary")

    def trial_tickets(self) -> list[dict[str, Any]]:
        data = self._json("GET", "/trial/tickets")
        return data["items"] if isinstance(data, dict) else data

    # ------------------------------------------------------------------ KB drafts
    def kb_drafts(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._json("GET", "/kb/drafts", params={"status": status})["items"]

    def create_kb_draft(self, analysis_id: str) -> dict[str, Any]:
        return self._json("POST", "/kb/drafts", body={"analysis_id": analysis_id})

    def approve_kb_draft(self, draft_id: str, edits: dict[str, Any] | None = None) -> KBArticle:
        data = self._json("POST", f"/kb/drafts/{draft_id}/approve", body={"edits": edits})
        return KBArticle.model_validate(data)


def _ticket_body(ticket: TicketIn | dict[str, Any]) -> dict[str, Any]:
    model = ticket if isinstance(ticket, TicketIn) else TicketIn.model_validate(ticket)
    return model.model_dump(mode="json", exclude_none=True)
