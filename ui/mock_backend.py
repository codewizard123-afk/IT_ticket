"""In-memory stand-in for P3's API (architecture §10), used when ``MOCK_API=1``.

It serves the golden fixtures for DEMO-001…007, a deterministic keyword-based result for any
other ticket, and keeps feedback, decisions, events, drafts and a hash-chained audit log in memory
so every UI page can be built and tested before the real backend exists.

**Nothing here is the real pipeline.** Scores are illustrative; the real API replaces all of it.
The same ``MockBackend.handle()`` is used in-process by ``ui.client`` and over HTTP by
``ui.mock_api``.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.constants import (
    ASSIGNMENT_GROUP,
    BASELINE_PRIORITY,
    LEVEL_PRIORITY,
    PIPELINE_VERSION,
    PRIORITY_LEVEL,
    ROLES,
    UI_EVENT_TYPES,
)
from shared.schemas import AnalysisResponse, Decision, Event, Feedback, KBArticle, TicketIn
from shared.stubs import GOLDEN_DIR, FakeKBStore, FakeTicketStore
from ui.settings import load_ui_config

MOCK_DATA = Path(__file__).parent / "mock_data"
PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
DECISION_RANK = {"escalate": 0, "clarify": 1, "recommend": 2}


class ApiError(Exception):
    """An API error in the architecture §10 error model."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message

    def payload(self, request_id: str = "mock") -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "request_id": request_id}}


@dataclass(frozen=True)
class Caller:
    username: str
    role: str
    group: str | None


@dataclass
class MockResponse:
    status: int
    body: Any
    content_type: str = "application/json"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _stats(key: str, values: list[float]) -> dict[str, Any]:
    return {
        "key": key,
        "n": len(values),
        "mean_s": round(statistics.fmean(values), 1),
        "median_s": round(statistics.median(values), 1),
        "p90_s": round(_percentile(values, 90), 1),
    }


def _group_of(response: dict[str, Any]) -> str:
    esc = response["details"].get("escalation") or {}
    if esc.get("assignment_group"):
        return esc["assignment_group"]
    return ASSIGNMENT_GROUP.get(response["classification"]["category"], "SERVICE-DESK")


def _best_per_key(pairs: Any) -> list[tuple[Any, float]]:
    """Keep the highest score per key, best first (alternatives must not repeat)."""
    best: dict[Any, float] = {}
    for key, score in pairs:
        best[key] = max(score, best.get(key, 0.0))
    return sorted(best.items(), key=lambda kv: -kv[1])


def filter_for_role(response: dict[str, Any], role: str) -> dict[str, Any]:
    """Display filter (§8.6): for ``l1`` users the text of L2 steps becomes "Handled by <group>"."""
    out = copy.deepcopy(response)
    if role != "l1" or "details" not in out:
        return out
    group = _group_of(out)

    def mask(steps: list[dict[str, Any]], contract_steps: list[str], grp: str) -> None:
        for i, step in enumerate(steps):
            if step.get("role") == "L2":
                masked = f"Handled by {grp}"
                if i < len(contract_steps) and contract_steps[i] == step["text"]:
                    contract_steps[i] = masked
                step["text"] = masked
                step["masked"] = True

    mask(out["details"].get("steps", []), out["recommended_solution"]["steps"], group)
    for sec in out["details"].get("secondary_issues", []):
        sec_group = (sec.get("escalation") or {}).get("assignment_group") or ASSIGNMENT_GROUP.get(
            sec["classification"]["category"], "SERVICE-DESK"
        )
        mask(sec.get("steps", []), sec["recommended_solution"]["steps"], sec_group)
    return out


class MockBackend:
    """Every endpoint of architecture §10, in memory."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.cfg = load_ui_config()["mock_api"]
        self.ticket_store = FakeTicketStore()
        self._articles: list[KBArticle] = [
            FakeKBStore().get_article(k) for k in FakeKBStore().kb_ids()
        ]
        self.kb_store = FakeKBStore(self._articles)
        self.analyses: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []
        self.decisions: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []
        self.drafts: dict[str, dict[str, Any]] = {}
        self._seq = 0
        self.trial_tickets = json.loads((MOCK_DATA / "trial_tickets.json").read_text("utf-8"))
        self.demo_tickets = json.loads((MOCK_DATA / "demo_tickets.json").read_text("utf-8"))
        self._routes: list[tuple[str, re.Pattern[str], Callable[..., Any]]] = [
            ("GET", re.compile(r"/health"), self.health),
            ("POST", re.compile(r"/auth/login"), self.login),
            ("POST", re.compile(r"/analyze"), self.analyze),
            ("POST", re.compile(r"/tickets/(?P<ticket_id>[^/]+)/clarify"), self.clarify),
            ("GET", re.compile(r"/tickets/search"), self.search_tickets),
            ("GET", re.compile(r"/tickets/(?P<ticket_id>[^/]+)"), self.get_ticket),
            ("GET", re.compile(r"/kb/search"), self.search_kb),
            ("GET", re.compile(r"/kb/drafts"), self.list_drafts),
            ("POST", re.compile(r"/kb/drafts"), self.create_draft),
            ("POST", re.compile(r"/kb/drafts/(?P<draft_id>[^/]+)/approve"), self.approve_draft),
            ("GET", re.compile(r"/kb/(?P<kb_id>[^/]+)"), self.get_kb),
            ("POST", re.compile(r"/feedback"), self.post_feedback),
            ("POST", re.compile(r"/decisions"), self.post_decision),
            ("POST", re.compile(r"/events"), self.post_event),
            ("GET", re.compile(r"/review-queue"), self.review_queue),
            ("GET", re.compile(r"/escalations/(?P<bundle_id>[^/]+)"), self.get_bundle),
            ("GET", re.compile(r"/audit/export"), self.audit_export),
            ("GET", re.compile(r"/audit/verify"), self.audit_verify),
            ("GET", re.compile(r"/audit/(?P<analysis_id>[^/]+)"), self.get_audit),
            ("GET", re.compile(r"/metrics/mttr"), self.metrics_mttr),
            ("GET", re.compile(r"/metrics/summary"), self.metrics_summary),
            ("GET", re.compile(r"/trial/tickets"), self.trial),
        ]

    # ------------------------------------------------------------------ dispatch
    def handle(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> MockResponse:
        path = "/" + path.split("?", 1)[0].strip("/")
        path = re.sub(r"^/api/v1", "", path) or "/"
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            for verb, pattern, fn in self._routes:
                m = pattern.fullmatch(path)
                if verb == method.upper() and m:
                    caller = None if fn in (self.health, self.login) else self._caller(headers)
                    result = fn(caller=caller, params=params, body=body or {}, **m.groupdict())
                    return result if isinstance(result, MockResponse) else MockResponse(200, result)
            raise ApiError(404, "not_found", f"no route for {method} {path}")
        except ApiError as err:
            return MockResponse(err.status, err.payload())
        except (ValueError, TypeError, KeyError) as err:
            return MockResponse(400, ApiError(400, "validation", str(err)).payload())

    def _caller(self, headers: dict[str, str]) -> Caller:
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer mock-token:"):
            username = auth.removeprefix("Bearer mock-token:")
            user = self.cfg["users"].get(username)
            if user:
                return Caller(username, user["role"], user.get("assignment_group"))
        role = headers.get("x-demo-role")
        if role in ROLES:
            group = headers.get("x-demo-group") or self.cfg["default_group"].get(role)
            return Caller(f"demo-{role}", role, group)
        raise ApiError(401, "unauthenticated", "log in or send X-Demo-Role")

    def _deny(self, caller: Caller, what: str) -> ApiError:
        self._audit(caller.username, "access.denied", "access", what, {"role": caller.role})
        return ApiError(403, "forbidden", f"role {caller.role} may not access {what}")

    # ------------------------------------------------------------------ audit + events
    def _audit(
        self, actor: str, action: str, entity: str, entity_id: str, payload: dict[str, Any]
    ) -> None:
        prev = self.audit[-1]["row_hash"] if self.audit else "0" * 64
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        self.audit.append(
            {
                "id": len(self.audit) + 1,
                "ts_utc": _iso(self.clock()),
                "actor": actor,
                "action": action,
                "entity": entity,
                "entity_id": entity_id,
                "payload_json": payload_json,
                "prev_hash": prev,
                "row_hash": _sha(prev + payload_json),
            }
        )

    def _event(
        self, ticket_id: str, analysis_id: str | None, event: str, actor: str, mode: str
    ) -> None:
        ts = self.clock()
        row = {
            "ticket_id": ticket_id,
            "analysis_id": analysis_id,
            "event": event,
            "ts": ts,
            "ts_utc": _iso(ts),
            "actor": actor,
            "mode": mode,
        }
        self.events.append(row)
        self._audit(actor, "event", "ticket", ticket_id, {k: v for k, v in row.items()})

    # ------------------------------------------------------------------ endpoints
    def health(self, **_: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "index_stale": False,
            "llm_available": False,
            "versions": {"pipeline": PIPELINE_VERSION, "backend": "ui.mock_backend"},
            "mock": True,
        }

    def login(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        user = self.cfg["users"].get(body.get("username", ""))
        if not user or not body.get("password"):
            raise ApiError(401, "unauthenticated", "unknown user or empty password")
        return {
            "token": f"mock-token:{body['username']}",
            "role": user["role"],
            "assignment_group": user.get("assignment_group"),
            "username": body["username"],
        }

    def _visible(self, caller: Caller, record: dict[str, Any]) -> bool:
        if caller.role == "admin":
            return True
        if caller.role == "l1":
            return record["user"] == caller.username
        if caller.role == "l2":
            return record["user"] == caller.username or record["group"] == caller.group
        return False

    def _store(self, caller: Caller, ticket: dict[str, Any], response: dict[str, Any], rnd: int):
        aid = response["details"]["analysis_id"]
        self.analyses[aid] = {
            "analysis_id": aid,
            "ticket_id": response["ticket_id"],
            "ticket": ticket,
            "response": response,
            "user": caller.username,
            "role": caller.role,
            "group": _group_of(response),
            "created_ts": self.clock(),
            "created_at": _iso(self.clock()),
            "clarification_round": rnd,
        }
        self._audit(
            caller.username,
            "analysis.created",
            "analysis",
            aid,
            {"ticket_id": response["ticket_id"], "decision": response["details"]["decision"]},
        )

    def analyze(self, caller: Caller, params: dict[str, Any], body: dict[str, Any], **_: Any):
        ticket = TicketIn.model_validate(body).model_dump()
        # intake: gold labels never reach the engine (CLAUDE.md rule 7)
        ticket["metadata"] = {
            k: v for k, v in ticket["metadata"].items() if not k.startswith("expected_")
        }
        self._seq += 1
        if not ticket.get("id"):
            ticket["id"] = f"INC-{datetime.fromtimestamp(self.clock(), UTC):%Y%m%d}-{self._seq:04d}"
        aid = f"mock-{self._seq:05d}"
        self._event(ticket["id"], aid, "ticket.submitted", caller.username, "assisted")
        response = self._run(ticket, aid, clarification_round=0)
        self._store(caller, ticket, response, 0)
        self._event(ticket["id"], aid, "analysis.completed", caller.username, "assisted")
        if response["details"]["decision"] == "clarify":
            self._event(ticket["id"], aid, "clarify.requested", caller.username, "assisted")
        return self._view(response, caller, params.get("view", "full"))

    def clarify(self, caller: Caller, ticket_id: str, body: dict[str, Any], params, **_: Any):
        prev = self.analyses.get(body.get("analysis_id", ""))
        if prev is None or prev["ticket_id"] != ticket_id:
            raise ApiError(404, "not_found", "unknown analysis for this ticket")
        if prev["clarification_round"] >= 1:
            raise ApiError(400, "validation", "only one clarification round is allowed")
        answer = str(body.get("answer", "")).strip()
        if not answer:
            raise ApiError(400, "validation", "answer is required")
        ticket = copy.deepcopy(prev["ticket"])
        ticket["description"] += "\n[Clarification] " + answer
        self._seq += 1
        aid = f"mock-{self._seq:05d}"
        self._event(ticket_id, prev["analysis_id"], "clarify.answered", caller.username, "assisted")
        response = self._run(ticket, aid, clarification_round=1)
        self._store(caller, ticket, response, 1)
        self._event(ticket_id, aid, "analysis.completed", caller.username, "assisted")
        return self._view(response, caller, params.get("view", "full"))

    def _view(self, response: dict[str, Any], caller: Caller, view: str) -> dict[str, Any]:
        out = filter_for_role(response, caller.role)
        if view == "contract":
            out.pop("details", None)
        return out

    # ------------------------------------------------------------------ mock "engine"
    def _run(self, ticket: dict[str, Any], aid: str, clarification_round: int) -> dict[str, Any]:
        golden = GOLDEN_DIR / f"{ticket['id']}.json"
        if golden.exists() and clarification_round == 0:
            resp = json.loads(golden.read_text("utf-8"))
            resp["details"]["analysis_id"] = aid
            for esc in [resp["details"].get("escalation")] + [
                s.get("escalation") for s in resp["details"].get("secondary_issues", [])
            ]:
                if esc:
                    esc["bundle_id"] = f"bundle-{aid}-{esc.get('issue_idx', 0)}"
        else:
            resp = self._generic(ticket, aid, clarification_round)
        resp["details"].setdefault("versions", {})["backend"] = "ui.mock_backend"
        return AnalysisResponse.model_validate(resp).model_dump(mode="json")

    def _generic(self, ticket: dict[str, Any], aid: str, rnd: int) -> dict[str, Any]:
        text = f"{ticket['title']}\n{ticket['description']}"
        threshold = float(self.cfg["generic_kb_threshold"])
        hits = self.ticket_store.query(text, 5)
        # score the whole text and each part (title, description, clarification answer) and keep
        # the best ranking: the stub's overlap is the share of *query* tokens, so long text dilutes
        parts = [text, ticket["title"], *ticket["description"].split("\n[Clarification] ")]
        rankings = [self.kb_store.query_chunks(p, 10) for p in parts if p.strip()]
        chunks = max(rankings, key=lambda r: r[0].score if r else 0.0)
        top = chunks[0] if chunks and chunks[0].score >= threshold else None
        if hits:
            category, subcategory = hits[0].category or "Inquiry/Help", hits[0].subcategory
            subcategory = subcategory or "Other"
        else:
            category, subcategory = "Inquiry/Help", "Other"
        top_score = hits[0].score if hits else 0.0
        confidence = round(min(0.85, 0.3 + 0.5 * top_score), 2)
        band = "high" if confidence >= 0.75 else "medium" if confidence >= 0.55 else "low"
        code = BASELINE_PRIORITY.get(category, "P4")
        level = PRIORITY_LEVEL[code]

        steps: list[dict[str, Any]] = []
        kedb = None
        kedb_match = {"found": False, "error_code": None, "solution": None}
        if top is not None:
            art = self.kb_store.get_article(top.kb_id)
            for i, s in enumerate(art.diagnostic_steps[:2], 1):
                steps.append(
                    {
                        "text": s,
                        "type": "diagnostic",
                        "citations": [f"{art.kb_id}#diag.{i}"],
                        "role": "L1",
                        "already_attempted": False,
                    }
                )
            for i, s in enumerate(art.resolution.steps[:2], 1):
                steps.append(
                    {
                        "text": s,
                        "type": "resolution",
                        "citations": [f"{art.kb_id}#res.{i}"],
                        "role": "L2" if art.resolution.requires_admin else "L1",
                        "already_attempted": False,
                    }
                )
            first_res = art.resolution.steps[0][:200] if art.resolution.steps else None
            kedb_match = {"found": True, "error_code": art.kb_id, "solution": first_res}
            kedb = {
                "kb_id": art.kb_id,
                "title": art.title,
                "score": round(min(1.0, top.score), 2),
                "matched_error_code": None,
                "matched_field": top.chunk_type,
                "excerpt": top.text[:200],
                "why": "keyword overlap (mock API)",
                "alternatives": [
                    {"kb_id": kb, "score": round(min(1.0, sc), 2)}
                    for kb, sc in _best_per_key(
                        (c.kb_id, c.score) for c in chunks if c.kb_id != art.kb_id
                    )
                ][:2],
            }

        words = len(re.findall(r"\w+", text))
        escalation = None
        questions: list[str] = []
        if rnd == 0 and (words < int(self.cfg["generic_min_words"]) or top is None):
            decision = "clarify"
            questions = [
                "What is the exact error message?",
                "How many users are affected?",
                "When did it start, and did anything change?",
            ]
        elif top is None:
            decision = "escalate"
            escalation = {
                "reasons": [
                    {"rule": "E1", "text": "no KEDB article above the relevance threshold"}
                ],
                "assignment_group": ASSIGNMENT_GROUP.get(category, "SERVICE-DESK"),
                "tier": "L2",
                "bundle_id": f"bundle-{aid}-0",
                "issue_idx": 0,
            }
        else:
            decision = "recommend"

        source = "KEDB" if steps else "None"
        approval = []
        if decision != "recommend":
            approval.append(f"decision {decision}")
        if level in ("Critical", "High"):
            approval.append(f"priority {level}")
        if any(s["role"] == "L2" for s in steps):
            approval.append("step requires admin")
        if band != "high":
            approval.append(f"confidence {band}")
        if source == "None":
            approval.append("no grounded steps")

        similar = []
        for h in hits:
            s = min(1.0, h.score)
            tier = (
                "DUPLICATE"
                if s >= 0.92
                else "LIKELY DUPLICATE"
                if s >= 0.75
                else "SIMILAR"
                if s >= 0.55
                else "RELATED"
            )
            similar.append(
                {
                    "incident_id": h.ticket_id,
                    "tier": tier,
                    "excerpt": h.excerpt,
                    "why": "keyword overlap (mock API)",
                    "resolution_origin": h.resolution_origin,
                    "category": h.category,
                    "subcategory": h.subcategory,
                    "score": round(s, 2),
                    "resolution": h.resolution,
                }
            )
        return {
            "ticket_id": ticket["id"],
            "classification": {
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
            },
            "priority": {"level": level, "reason": f"Baseline for {subcategory}"},
            "similar_incidents": [
                {
                    "incident_id": s["incident_id"],
                    "similarity_score": s["score"],
                    "resolution": s["resolution"],
                }
                for s in similar
            ],
            "kedb_match": kedb_match,
            "recommended_solution": {"steps": [s["text"] for s in steps], "source": source},
            "escalation_required": decision == "escalate",
            "human_approval_required": bool(approval),
            "details": {
                "analysis_id": aid,
                "decision": decision,
                "primary_decision": decision,
                "clarifying_questions": questions,
                "intent": "incident",
                "classification": {
                    "band": band,
                    "components": {"knn": round(top_score, 2), "llm": 0.0, "rule": 0.0},
                    "alternatives": [
                        {"category": c, "subcategory": sub, "score": sc}
                        for (c, sub), sc in _best_per_key(
                            ((h.category, h.subcategory), h.score)
                            for h in hits
                            if h.category
                            and h.subcategory
                            and (h.category, h.subcategory) != (category, subcategory)
                        )
                    ][:2],
                    "rationale": "mock API: keyword overlap with historical tickets",
                    "conflict": False,
                },
                "priority": {
                    "code": code,
                    "impact": 2 if code in ("P1", "P2") else 3,
                    "urgency": 2 if code in ("P1", "P2") else 3,
                    "drivers": [],
                },
                "similar": [
                    {k: v for k, v in s.items() if k not in ("score", "resolution")}
                    for s in similar
                ],
                "keyword_only": [],
                "kedb": kedb,
                "steps": steps,
                "escalation": escalation,
                "approval_reasons": approval,
                "secondary_issues": [],
                "redactions": {"count": 0, "types": []},
                "degraded": ["offline"],
                "timings_ms": {"total": 5},
                "versions": {"pipeline": PIPELINE_VERSION},
            },
        }

    # ------------------------------------------------------------------ search
    def _ticket_visible(self, caller: Caller, rec: dict[str, Any]) -> bool:
        if not rec.get("restricted"):
            return True
        if caller.role == "admin":
            return True
        return caller.role == "l2" and rec.get("assignment_group") == caller.group

    def search_tickets(self, caller: Caller, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        records = [r.model_dump() for r in self.ticket_store.get(list(self.ticket_store._recs))]
        q = str(params.get("q", "")).strip()
        scores: dict[str, float] = {}
        if q:
            scores = {h.ticket_id: h.score for h in self.ticket_store.query(q, len(records))}
            records = [r for r in records if r["ticket_id"] in scores]
        prio = params.get("priority")
        if prio:
            prio = LEVEL_PRIORITY.get(prio, prio)
        exact = {
            "category": params.get("category"),
            "subcategory": params.get("subcategory"),
            "priority": prio,
            "status": params.get("status"),
            "assignment_group": params.get("assignment_group"),
            "application": params.get("application"),
            "environment": params.get("environment"),
            "source": params.get("source"),
        }
        for key, want in exact.items():
            if want:
                records = [r for r in records if r.get(key) == want]
        date_from, date_to = params.get("date_from"), params.get("date_to")
        if date_from:
            records = [r for r in records if (r.get("opened_at") or "")[:10] >= str(date_from)]
        if date_to:
            records = [r for r in records if (r.get("opened_at") or "")[:10] <= str(date_to)]
        hidden = [r for r in records if not self._ticket_visible(caller, r)]
        if hidden:
            self._audit(
                caller.username,
                "access.denied",
                "tickets",
                "search",
                {"hidden": len(hidden), "role": caller.role},
            )
        records = [r for r in records if self._ticket_visible(caller, r)]
        if q:
            records.sort(key=lambda r: (-scores[r["ticket_id"]], r["ticket_id"]))
        else:
            records.sort(key=lambda r: (r.get("opened_at") or "", r["ticket_id"]), reverse=True)
        page = max(1, int(params.get("page", 1)))
        size = max(1, min(200, int(params.get("size", 20))))
        items = []
        for r in records[(page - 1) * size : page * size]:
            items.append(
                {
                    "ticket_id": r["ticket_id"],
                    "score": round(scores.get(r["ticket_id"], 1.0), 4),
                    "title": r["title"],
                    "excerpt": r["description"][:200],
                    "category": r["category"],
                    "subcategory": r["subcategory"],
                    "priority": r["priority"],
                    "status": r["status"],
                    "resolution": r["resolution"],
                    "resolution_origin": r["resolution_origin"],
                    "collection": r["collection"],
                    "metadata": {
                        k: r.get(k)
                        for k in (
                            "assignment_group",
                            "application",
                            "environment",
                            "source",
                            "opened_at",
                            "restricted",
                            "derived_fields",
                        )
                    },
                }
            )
        return {"items": items, "total": len(records), "page": page, "size": size}

    def get_ticket(self, caller: Caller, ticket_id: str, **_: Any) -> dict[str, Any]:
        recs = self.ticket_store.get([ticket_id])
        if not recs:
            for a in self.analyses.values():
                if a["ticket_id"] == ticket_id:
                    return {**a["ticket"], "ticket_id": ticket_id, "collection": "tickets_live"}
            raise ApiError(404, "not_found", f"ticket {ticket_id}")
        rec = recs[0].model_dump()
        if not self._ticket_visible(caller, rec):
            raise self._deny(caller, f"ticket {ticket_id}")
        return rec

    def search_kb(self, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        q = str(params.get("q", "")).strip()
        cat, sub = params.get("category"), params.get("subcategory")
        best: dict[str, Any] = {}
        if q:
            for c in self.kb_store.query_chunks(q, 200):
                if c.kb_id not in best or c.score > best[c.kb_id].score:
                    best[c.kb_id] = c
        items = []
        for art in self._articles:
            if art.status != "approved":
                continue
            if cat and art.category != cat or sub and art.subcategory != sub:
                continue
            if q and art.kb_id not in best:
                continue
            chunk = best.get(art.kb_id)
            items.append(
                {
                    "kb_id": art.kb_id,
                    "title": art.title,
                    "score": round(min(1.0, chunk.score), 4) if chunk else 1.0,
                    "matched_field": chunk.chunk_type if chunk else None,
                    "excerpt": chunk.text[:200] if chunk else (art.root_cause or "")[:200],
                    "why": "keyword overlap (mock API)" if chunk else None,
                    "requires_admin": art.resolution.requires_admin,
                    "category": art.category,
                    "subcategory": art.subcategory,
                    "source": art.source,
                    "version": art.version,
                }
            )
        items.sort(key=lambda i: (-i["score"], i["kb_id"]))
        return {"items": items}

    def get_kb(self, kb_id: str, **_: Any) -> dict[str, Any]:
        for art in self._articles:
            if art.kb_id == kb_id:
                return art.model_dump(mode="json")
        raise ApiError(404, "not_found", f"article {kb_id}")

    # ------------------------------------------------------------------ feedback/decisions
    def post_feedback(self, caller: Caller, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        fb = Feedback.model_validate(body).model_dump()
        if fb["analysis_id"] not in self.analyses:
            raise ApiError(404, "not_found", "unknown analysis")
        fid = f"fb-{len(self.feedback) + 1:05d}"
        row = {**fb, "feedback_id": fid, "user": caller.username, "created_at": _iso(self.clock())}
        self.feedback.append(row)
        self._audit(caller.username, "feedback", "analysis", fb["analysis_id"], row)
        return {"feedback_id": fid}

    def _ticket_for(self, analysis_id: str) -> str | None:
        if analysis_id in self.analyses:
            return self.analyses[analysis_id]["ticket_id"]
        for e in self.events:
            if e["analysis_id"] == analysis_id:
                return e["ticket_id"]
        return None

    def post_decision(self, caller: Caller, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        dec = Decision.model_validate(body).model_dump()
        ticket_id = self._ticket_for(dec["analysis_id"])
        if ticket_id is None:
            raise ApiError(404, "not_found", "unknown analysis (manual mode: post manual.started)")
        row = {**dec, "user": caller.username, "decided_at": _iso(self.clock())}
        self.decisions[dec["analysis_id"]] = row
        self._audit(caller.username, "decision", "analysis", dec["analysis_id"], row)
        if dec["mode"] == "assisted":
            self._event(
                ticket_id, dec["analysis_id"], "decision.submitted", caller.username, "assisted"
            )
        if dec["final_decision"] == "resolved":
            self._event(
                ticket_id, dec["analysis_id"], "ticket.resolved", caller.username, dec["mode"]
            )
        return {"ok": True}

    def post_event(self, caller: Caller, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        ev = Event.model_validate(body).model_dump()
        if ev["event"] not in UI_EVENT_TYPES:
            raise ApiError(400, "validation", f"UI may only post {', '.join(UI_EVENT_TYPES)}")
        self._event(ev["ticket_id"], ev["analysis_id"], ev["event"], caller.username, ev["mode"])
        return {"ok": True}

    # ------------------------------------------------------------------ review queue / bundles
    def review_queue(self, caller: Caller, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        status = params.get("status", "open")
        now = self.clock()
        items = []
        superseded = {
            e["analysis_id"]
            for e in self.events
            if e["event"] == "clarify.answered" and e["analysis_id"]
        }
        for a in self.analyses.values():
            r = a["response"]
            d = r["details"]["decision"]
            needs = d in ("clarify", "escalate") or r["human_approval_required"]
            if not needs or a["analysis_id"] in superseded:
                continue
            decided = a["analysis_id"] in self.decisions
            if status == "open" and decided:
                continue
            if status in ("clarify", "escalate") and d != status:
                continue
            if status == "approval" and not r["human_approval_required"]:
                continue
            items.append(
                {
                    "analysis_id": a["analysis_id"],
                    "ticket_id": a["ticket_id"],
                    "title": a["ticket"]["title"],
                    "decision": d,
                    "priority": r["priority"]["level"],
                    "category": r["classification"]["category"],
                    "subcategory": r["classification"]["subcategory"],
                    "confidence": r["classification"]["confidence"],
                    "human_approval_required": r["human_approval_required"],
                    "approval_reasons": r["details"].get("approval_reasons", []),
                    "assignment_group": a["group"],
                    "created_at": a["created_at"],
                    "age_s": round(now - a["created_ts"], 1),
                    "decided": decided,
                }
            )
        items.sort(key=lambda i: (PRIORITY_RANK[i["priority"]], -i["age_s"]))
        page = max(1, int(params.get("page", 1)))
        size = int(load_ui_config()["review_queue"]["page_size"])
        return {"items": items[(page - 1) * size : page * size], "total": len(items), "page": page}

    def _find_bundle(self, bundle_id: str) -> tuple[dict[str, Any], dict[str, Any], int]:
        for a in self.analyses.values():
            det = a["response"]["details"]
            escs = [det.get("escalation")] + [
                s.get("escalation") for s in det.get("secondary_issues", [])
            ]
            for e in escs:
                if e and e.get("bundle_id") == bundle_id:
                    return a, e, int(e.get("issue_idx", 0))
        raise ApiError(404, "not_found", f"bundle {bundle_id}")

    def get_bundle(self, caller: Caller, bundle_id: str, params: dict[str, Any], **_: Any):
        a, esc, idx = self._find_bundle(bundle_id)
        resp = filter_for_role(a["response"], caller.role)
        det = resp["details"]
        issue = det["secondary_issues"][idx - 1] if idx > 0 else None
        steps = (issue or det).get("steps", [])
        bundle = {
            "bundle_id": bundle_id,
            "ticket_id": a["ticket_id"],
            "title": a["ticket"]["title"],
            "description_redacted": a["ticket"]["description"],
            "classification": (issue or resp)["classification"],
            "priority": (issue or resp)["priority"],
            "assignment_group": esc["assignment_group"],
            "tier": esc["tier"],
            "reasons": esc["reasons"],
            "similar_incidents": resp["similar_incidents"][:5],
            "kedb": (issue or det).get("kedb"),
            "attempted_steps": [s["text"] for s in steps if s.get("already_attempted")],
            "remaining_steps": [s["text"] for s in steps if not s.get("already_attempted")],
            "created_at": a["created_at"],
            "masked": caller.role == "l1",
        }
        if params.get("fmt") == "md":
            lines = [
                f"# Escalation bundle {bundle_id}",
                "",
                f"**Ticket:** {bundle['ticket_id']} - {bundle['title']}",
                f"**Route to:** {bundle['assignment_group']} ({bundle['tier']})",
                "",
                "## Reasons",
                *[f"- {r['rule']}: {r['text']}" for r in bundle["reasons"]],
                "",
                "## Ticket (redacted)",
                bundle["description_redacted"],
                "",
                "## Attempted steps",
                *([f"- {s}" for s in bundle["attempted_steps"]] or ["- none"]),
                "",
                "## Remaining KB steps",
                *([f"- {s}" for s in bundle["remaining_steps"]] or ["- none"]),
            ]
            return MockResponse(200, "\n".join(lines) + "\n", "text/markdown")
        return bundle

    # ------------------------------------------------------------------ audit
    def _verify(self) -> dict[str, Any]:
        prev = "0" * 64
        for row in self.audit:
            if row["prev_hash"] != prev or row["row_hash"] != _sha(prev + row["payload_json"]):
                return {"ok": False, "broken_at": row["id"]}
            prev = row["row_hash"]
        return {"ok": True, "rows": len(self.audit)}

    def get_audit(self, caller: Caller, analysis_id: str, **_: Any) -> dict[str, Any]:
        a = self.analyses.get(analysis_id)
        if a is None:
            raise ApiError(404, "not_found", f"analysis {analysis_id}")
        if not self._visible(caller, a):
            raise self._deny(caller, f"audit {analysis_id}")
        resp = filter_for_role(a["response"], caller.role)
        det = resp["details"]
        evidence = [
            {
                "kind": "similar",
                "ref_id": s["incident_id"],
                "rank": i,
                "score": resp["similar_incidents"][i - 1]["similarity_score"]
                if i - 1 < len(resp["similar_incidents"])
                else None,
                "tier": s["tier"],
                "excerpt": s.get("excerpt"),
                "why": s.get("why"),
                "origin": s["resolution_origin"],
            }
            for i, s in enumerate(det.get("similar", []), 1)
        ]
        if det.get("kedb"):
            k = det["kedb"]
            evidence.append(
                {
                    "kind": "kedb",
                    "ref_id": k["kb_id"],
                    "rank": 1,
                    "score": k["score"],
                    "tier": None,
                    "excerpt": k.get("excerpt"),
                    "why": k.get("why"),
                    "origin": "kedb",
                }
            )
        tickets = {a["ticket_id"]}
        timeline = [
            {k: v for k, v in e.items() if k != "ts"}
            for e in self.events
            if e["analysis_id"] == analysis_id
            or (e["ticket_id"] in tickets and e["analysis_id"] is None)
        ]
        return {
            "analysis_id": analysis_id,
            "ticket_id": a["ticket_id"],
            "user": a["user"],
            "role": a["role"],
            "created_at": a["created_at"],
            "clarification_round": a["clarification_round"],
            "ticket": {"title": a["ticket"]["title"], "description": a["ticket"]["description"]},
            "response": resp,
            "evidence": evidence,
            "timeline": timeline,
            "feedback": [f for f in self.feedback if f["analysis_id"] == analysis_id],
            "decision": self.decisions.get(analysis_id),
            "chain": self._verify(),
            "filtered_for_role": caller.role,
        }

    def audit_export(self, caller: Caller, params: dict[str, Any], **_: Any) -> MockResponse:
        if caller.role == "kb_owner":
            raise self._deny(caller, "audit export")
        fmt = params.get("format", "jsonl")
        if fmt not in ("csv", "jsonl"):
            raise ApiError(400, "validation", "format must be csv or jsonl")
        lo, hi = params.get("from"), params.get("to")
        rows = [
            r
            for r in self.audit
            if (not lo or r["ts_utc"][:10] >= str(lo)) and (not hi or r["ts_utc"][:10] <= str(hi))
        ]
        masked = caller.role != "admin"
        out_rows = []
        for r in rows:
            row = dict(r)
            if masked:
                row["actor"] = "user-" + _sha(row["actor"])[:8]
                payload = json.loads(row["payload_json"])
                for key in ("actor", "user", "resolution_notes", "comment", "corrected_value"):
                    if payload.get(key):
                        payload[key] = "***"
                row["payload_json"] = json.dumps(payload, sort_keys=True)
            out_rows.append(row)
        self._audit(
            caller.username,
            "audit.exported",
            "audit",
            fmt,
            {"rows": len(out_rows), "masked": masked, "watermark": f"export-{len(self.audit)}"},
        )
        if fmt == "jsonl":
            text = "".join(json.dumps(r) + "\n" for r in out_rows)
            return MockResponse(200, text, "application/x-ndjson")
        buf = io.StringIO()
        fields = list(out_rows[0]) if out_rows else ["id"]
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
        return MockResponse(200, buf.getvalue(), "text/csv")

    def audit_verify(self, caller: Caller, **_: Any) -> dict[str, Any]:
        if caller.role != "admin":
            raise self._deny(caller, "audit verify")
        return self._verify()

    # ------------------------------------------------------------------ metrics
    def timing_rows(self) -> list[dict[str, Any]]:
        """One row per analysis/manual session with TTR, TTD and MTTR in seconds."""
        by_aid: dict[str, list[dict[str, Any]]] = {}
        for e in self.events:
            if e["analysis_id"]:
                by_aid.setdefault(e["analysis_id"], []).append(e)
        rows = []
        for aid, evs in by_aid.items():
            first = {}
            for e in evs:
                first.setdefault(e["event"], e["ts"])
            mode = "manual" if "manual.started" in first else "assisted"
            a = self.analyses.get(aid)
            dec = self.decisions.get(aid, {})
            if a:
                category = a["response"]["classification"]["category"]
                priority = a["response"]["priority"]["level"]
                decision = a["response"]["details"]["decision"]
            else:
                category = dec.get("final_category") or "unknown"
                priority = dec.get("final_priority") or "unknown"
                decision = "manual"
            ttr = (
                first["analysis.completed"] - first["ticket.submitted"]
                if {"analysis.completed", "ticket.submitted"} <= first.keys()
                else None
            )
            if mode == "manual":
                ttd = (
                    first["manual.decision"] - first["manual.started"]
                    if "manual.decision" in first
                    else None
                )
            else:
                ttd = (
                    first["decision.submitted"] - first["ui.viewed"]
                    if {"decision.submitted", "ui.viewed"} <= first.keys()
                    else None
                )
            start = first.get("ticket.submitted", first.get("manual.started"))
            mttr = first["ticket.resolved"] - start if "ticket.resolved" in first else None
            rows.append(
                {
                    "analysis_id": aid,
                    "ticket_id": evs[0]["ticket_id"],
                    "mode": mode,
                    "category": category,
                    "priority": priority,
                    "decision": decision,
                    "final_decision": dec.get("final_decision"),
                    "ttr_s": ttr,
                    "ttd_s": ttd,
                    "mttr_s": mttr,
                }
            )
        return rows

    def metrics_mttr(self, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        group_by = params.get("group_by", "category")
        if group_by not in ("category", "priority", "mode", "decision"):
            raise ApiError(400, "validation", "group_by must be category|priority|mode|decision")
        metric = params.get("metric", "mttr")
        if metric not in ("ttr", "ttd", "mttr"):
            raise ApiError(400, "validation", "metric must be ttr|ttd|mttr")
        rows = self.timing_rows()
        if params.get("mode"):
            rows = [r for r in rows if r["mode"] == params["mode"]]
        groups: dict[str, list[float]] = {}
        for r in rows:
            if r[f"{metric}_s"] is not None:
                groups.setdefault(str(r[group_by]), []).append(float(r[f"{metric}_s"]))
        return {
            "metric": metric,
            "group_by": group_by,
            "groups": [_stats(k, v) for k, v in sorted(groups.items())],
        }

    def metrics_summary(self, **_: Any) -> dict[str, Any]:
        responses = [a["response"] for a in self.analyses.values()]
        verdicts = Counter(f["verdict"] for f in self.feedback)
        total_fb = sum(verdicts.values())
        degraded = [r for r in responses if r["details"].get("degraded")]
        gaps = Counter(
            r["classification"]["subcategory"] for r in responses if not r["kedb_match"]["found"]
        )
        hist = [
            r
            for r in (x.model_dump() for x in self.ticket_store.get(list(self.ticket_store._recs)))
            if r.get("opened_ts") and r.get("resolved_ts")
        ]
        hist_s = [float(r["resolved_ts"] - r["opened_ts"]) for r in hist]
        rows = self.timing_rows()
        ttd_by_mode = {}
        for mode in ("manual", "assisted"):
            vals = [float(r["ttd_s"]) for r in rows if r["mode"] == mode and r["ttd_s"] is not None]
            if vals:
                ttd_by_mode[mode] = _stats(mode, vals)
        return {
            "analyses": len(responses),
            "decisions": dict(Counter(r["details"]["decision"] for r in responses)),
            "final_decisions": dict(Counter(d["final_decision"] for d in self.decisions.values())),
            "escalation_rate": round(
                sum(r["escalation_required"] for r in responses) / len(responses), 3
            )
            if responses
            else 0.0,
            "feedback": {
                "total": total_fb,
                **{v: verdicts.get(v, 0) for v in ("useful", "incorrect", "unresolved")},
                "rates": {
                    v: round(verdicts.get(v, 0) / total_fb, 3) if total_fb else 0.0
                    for v in ("useful", "incorrect", "unresolved")
                },
            },
            "degraded": {
                "count": len(degraded),
                "rate": round(len(degraded) / len(responses), 3) if responses else 0.0,
                "by_mode": dict(Counter(m for r in degraded for m in r["details"]["degraded"])),
            },
            "ttd_by_mode": ttd_by_mode,
            "kb_gaps": dict(gaps),
            "historical_baseline": {
                "synthetic": True,
                "label": "Historical MTTR (synthetic timestamps, context only)",
                "n": len(hist_s),
                "mean_s": round(statistics.fmean(hist_s), 1) if hist_s else None,
                "median_s": round(statistics.median(hist_s), 1) if hist_s else None,
            },
            "mock": True,
        }

    # ------------------------------------------------------------------ KB drafts
    def list_drafts(self, caller: Caller, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        status = params.get("status")
        items = [d for d in self.drafts.values() if not status or d["status"] == status]
        return {"items": items}

    def create_draft(self, caller: Caller, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        if caller.role not in ("l2", "kb_owner", "admin"):
            raise self._deny(caller, "kb draft")
        aid = body.get("analysis_id", "")
        a = self.analyses.get(aid)
        if a is None:
            raise ApiError(404, "not_found", "unknown analysis")
        notes = (self.decisions.get(aid) or {}).get("resolution_notes") or ""
        steps = [s.strip(" -•\t") for s in notes.splitlines() if s.strip(" -•\t")]
        if not steps:
            raise ApiError(400, "validation", "the decision needs resolution notes first")
        kb_num = 30 + len(self.drafts) + 1
        resp = a["response"]
        article = {
            "kb_id": f"KB-{kb_num:04d}",
            "title": a["ticket"]["title"],
            "category": resp["classification"]["category"],
            "subcategory": resp["classification"]["subcategory"],
            "symptoms": [a["ticket"]["title"]],
            "common_error_codes": [],
            "root_cause": None,
            "diagnostic_steps": [],
            "resolution": {"steps": steps, "estimated_time": None, "requires_admin": False},
            "escalation_criteria": [],
            "status": "draft",
            "version": 1,
            "source": "analyst",
        }
        draft_id = f"draft-{len(self.drafts) + 1:04d}"
        draft = {
            "draft_id": draft_id,
            "article": article,
            "status": "draft",
            "proposed_by": caller.username,
            "source_analysis_id": aid,
            "created_at": _iso(self.clock()),
            "approved_by": None,
        }
        self.drafts[draft_id] = draft
        self._audit(caller.username, "kb.draft", "kb_draft", draft_id, {"analysis_id": aid})
        return draft

    def approve_draft(self, caller: Caller, draft_id: str, body: dict[str, Any], **_: Any):
        if caller.role not in ("kb_owner", "admin"):
            raise self._deny(caller, "kb approve")
        draft = self.drafts.get(draft_id)
        if draft is None:
            raise ApiError(404, "not_found", f"draft {draft_id}")
        article = {**draft["article"], **(body.get("edits") or {})}
        article.update(status="approved", version=2, approved_by=caller.username)
        art = KBArticle.model_validate(article)
        self._articles = [a for a in self._articles if a.kb_id != art.kb_id] + [art]
        self.kb_store = FakeKBStore(self._articles)
        draft.update(status="approved", approved_by=caller.username, article=article)
        self._audit(caller.username, "kb.approved", "kb", art.kb_id, {"draft_id": draft_id})
        return art.model_dump(mode="json")

    def trial(self, **_: Any) -> dict[str, Any]:
        # gold answers never leave the backend (§9.3); the mock has none to leak
        return {"items": copy.deepcopy(self.trial_tickets)}

    # ------------------------------------------------------------------ test helpers
    def tamper(self, row_id: int, payload: dict[str, Any]) -> None:
        """Tests only: rewrite one audit row's payload without fixing the chain."""
        self.audit[row_id - 1]["payload_json"] = json.dumps(payload)
