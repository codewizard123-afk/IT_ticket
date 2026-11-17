"""Role-based access control (architecture §8.6), enforced in three places:

1. **Retrieval** - ``retrieval_filters`` become ``ctx.filters`` (Chroma ``where``), and
   ``filter_engine_result`` drops any restricted hit that slipped through.
2. **Display** - ``filter_response`` masks L2 step text for ``l1`` ("Handled by <group>").
3. **Export** - ``mask_export_row`` pseudonymises actors and blanks free text unless admin.

Every denial is written to the audit log as ``access.denied`` (``deny``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from backend import audit
from backend.config import escalation_config
from backend.errors import ApiError
from shared.schemas import EngineResult, TicketHit, User

AUDIT_ALL = frozenset({"admin"})
EXPORT_ROLES = frozenset({"l1", "l2", "admin"})
DRAFT_ROLES = frozenset({"l2", "kb_owner", "admin"})
APPROVE_ROLES = frozenset({"kb_owner", "admin"})
TRIAL_ROLES = frozenset({"l1", "l2", "admin"})
# Masked exports keep only these payload keys; everything else (ticket text, steps, notes,
# comments, rationale, evidence excerpts) becomes "***".
EXPORT_SAFE_KEYS = frozenset(
    {
        "ticket_id", "analysis_id", "decision", "escalation_required", "human_approval_required",
        "approval_reasons", "versions", "timings_ms", "degraded", "role", "clarification_round",
        "issue_idx", "target_type", "target_ref", "verdict", "final_decision", "final_category",
        "final_subcategory", "final_priority", "mode", "event", "rows", "masked", "from", "to",
        "what", "group", "draft_id", "kb_id", "status", "version",
    }
)  # fmt: skip


def deny(conn: sqlite3.Connection | None, user: User, what: str) -> ApiError:
    """Log ``access.denied`` and return the 403 to raise."""
    if conn is not None:
        audit.append(
            conn,
            actor=user.username,
            action="access.denied",
            entity="access",
            entity_id=what,
            payload={"role": user.role, "group": user.assignment_group, "what": what},
        )
    return ApiError(403, "forbidden", f"role {user.role} may not access {what}")


def require(conn: sqlite3.Connection | None, user: User, roles: frozenset[str], what: str) -> None:
    if user.role not in roles:
        raise deny(conn, user, what)


# --- retrieval ---------------------------------------------------------------------------------
def retrieval_filters(user: User) -> dict[str, Any]:
    """Chroma ``where`` fragment for ``ctx.filters``: who may see restricted tickets."""
    if user.role == "admin":
        return {}
    if user.role == "l2" and user.assignment_group:
        return {"$or": [{"restricted": False}, {"assignment_group": user.assignment_group}]}
    return {"restricted": False}


def ticket_visible(user: User, restricted: Any, group: str | None) -> bool:
    if not restricted:
        return True
    if user.role == "admin":
        return True
    return user.role == "l2" and bool(user.assignment_group) and group == user.assignment_group


def hit_visible(user: User, hit: TicketHit) -> bool:
    meta = hit.metadata or {}
    return ticket_visible(user, meta.get("restricted", False), meta.get("assignment_group"))


def filter_engine_result(er: EngineResult, user: User) -> EngineResult:
    issues = [
        i.model_copy(
            update={
                "similar": [h for h in i.similar if hit_visible(user, h)],
                "keyword_only": [h for h in i.keyword_only if hit_visible(user, h)],
            }
        )
        for i in er.issues
    ]
    return er.model_copy(update={"issues": issues})


# --- display -----------------------------------------------------------------------------------
def group_of(response: Mapping[str, Any], issue_idx: int = 0) -> str:
    """Escalation group of the issue, else its category's group."""
    details = response.get("details", {})
    routing = escalation_config().get("routing", {})
    if issue_idx == 0:
        esc = details.get("escalation") or {}
        cls = response.get("classification", {})
    else:
        sec = details.get("secondary_issues", [])[issue_idx - 1]
        esc, cls = sec.get("escalation") or {}, sec.get("classification", {})
    if esc.get("assignment_group") and int(esc.get("issue_idx", 0)) == issue_idx:
        return str(esc["assignment_group"])
    return str(
        routing.get("by_category", {}).get(
            cls.get("category"), routing.get("default_group", "SERVICE-DESK")
        )
    )


def _mask_steps(steps: list[dict[str, Any]], contract_steps: list[str], group: str) -> None:
    for i, step in enumerate(steps):
        if step.get("role") == "L2":
            masked = f"Handled by {group}"
            if i < len(contract_steps) and contract_steps[i] == step["text"]:
                contract_steps[i] = masked
            step["text"] = masked
            step["masked"] = True


def _scrub(value: Any, secrets_: list[str], masked: str) -> Any:
    if isinstance(value, str) and any(t in value for t in secrets_):
        return masked
    return value


def filter_response(response: Mapping[str, Any], user: User) -> dict[str, Any]:
    """Display filter on a JSON response (full view). ``l1`` never receives L2 step text.

    Besides the steps themselves, L2 text can surface through the KEDB solution (the article's
    first resolution step) and through similar incidents whose resolution was derived from the
    same KB step, so those fields are scrubbed too.
    """
    out = copy.deepcopy(dict(response))
    if user.role != "l1" or "details" not in out:
        return out
    details = out["details"]
    issues = [(out, details, 0)] + [
        (sec, sec, n) for n, sec in enumerate(details.get("secondary_issues", []), 1)
    ]
    for contract, det, n in issues:
        group = group_of(out, n)
        steps = det.get("steps", [])
        l2_texts = [s["text"] for s in steps if s.get("role") == "L2" and s.get("text")]
        _mask_steps(steps, contract["recommended_solution"]["steps"], group)
        if not l2_texts:
            continue
        masked = f"Handled by {group}"
        km = contract.get("kedb_match") or {}
        if km.get("solution"):
            km["solution"] = _scrub(km["solution"], l2_texts, masked)
        if det.get("kedb"):
            det["kedb"]["excerpt"] = _scrub(det["kedb"].get("excerpt"), l2_texts, masked)
        for sim in contract.get("similar_incidents", []):
            sim["resolution"] = _scrub(sim.get("resolution"), l2_texts, masked)
        for sim in det.get("similar", []) + det.get("keyword_only", []):
            sim["excerpt"] = _scrub(sim.get("excerpt"), l2_texts, masked)
    return out


# --- audit view / export -----------------------------------------------------------------------
def can_view_analysis(user: User, analysis: Mapping[str, Any], group: str) -> bool:
    if user.role == "admin":
        return True
    if user.role == "l1":
        return analysis.get("user") == user.username
    if user.role == "l2":
        return analysis.get("user") == user.username or (
            bool(user.assignment_group) and group == user.assignment_group
        )
    return False


def can_act_on_analysis(user: User, analysis: Mapping[str, Any], group: str) -> bool:
    """Clarify, feedback and decisions: the analysis owner, same-group l2, or admin."""
    if analysis.get("user") == user.username or user.role == "admin":
        return True
    return user.role == "l2" and bool(user.assignment_group) and group == user.assignment_group


def pseudonym(name: str) -> str:
    return "user-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]


def mask_export_row(row: Mapping[str, Any], user: User) -> dict[str, Any]:
    """Admins export unmasked; everyone else gets pseudonymised actors and only safe payload keys.

    The hashes stay as stored, so a masked export can't be re-verified - that is intended.
    """
    out = dict(row)
    if user.role == "admin":
        return out
    out["actor"] = pseudonym(str(out.get("actor") or ""))
    try:
        payload = json.loads(out.get("payload_json") or "null")
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        payload = {
            k: v if k in EXPORT_SAFE_KEYS or v in (None, "", [], {}) else "***"
            for k, v in payload.items()
        }
        out["payload_json"] = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return out
