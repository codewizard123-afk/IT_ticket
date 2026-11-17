"""KB draft loop (architecture §8.9).

1. ``create_draft``: the analyst's (redacted) resolution notes and unresolved/incorrect feedback
   comments are turned into an article in the exact KEDB schema - by Groq when available, else by a
   deterministic template. Saved as ``status=draft``; not searchable.
2. ``approve_draft`` (kb_owner/admin): applies edits, sets ``status=approved``, ``version+1`` and
   ``approved_by`` and publishes it: P1's ``reindex_article`` writes ``articles_governed.json``
   and re-indexes the article (stub stores: the governed file is written here instead).

Unsafe steps (disable security controls, share passwords, broad privileges) are always dropped.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend import access, audit, repo
from backend.config import ROOT, backend_config, base_config
from backend.errors import ApiError, bad_request, not_found
from backend.services import Services
from shared.interfaces import LLMUnavailableError
from shared.schemas import KBArticle, User

log = logging.getLogger(__name__)
_KB_NUM = re.compile(r"^KB-(\d+)$")
_BULLET = " -•*\t"
EDITABLE = frozenset(
    {
        "title", "category", "subcategory", "symptoms", "common_error_codes", "root_cause",
        "diagnostic_steps", "resolution", "escalation_criteria", "related_kb_ids", "tags",
        "priority_hint",
    }
)  # fmt: skip

SYSTEM_PROMPT = """You draft a known-error (KEDB) article for an IT service desk.
Use ONLY facts from the analyst notes inside <notes>; do not invent products, codes or steps.
Never include steps that disable or bypass security controls, share passwords or grant broad
privileges. Reply with one JSON object with exactly these keys:
title (string), symptoms (list of strings), common_error_codes (list of strings),
root_cause (string or null), diagnostic_steps (list of strings),
resolution (object: steps list of strings, estimated_time string or null, requires_admin bool),
escalation_criteria (list of strings), tags (list of strings)."""


def _cfg(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return dict((cfg if cfg is not None else backend_config()).get("kb_drafts", {}))


def is_unsafe(text: str, cfg: Mapping[str, Any] | None = None) -> bool:
    return any(
        re.search(p, text, flags=re.IGNORECASE) for p in _cfg(cfg).get("unsafe_patterns", [])
    )


def _safe(items: list[str], cfg: Mapping[str, Any] | None = None) -> list[str]:
    limit = int(_cfg(cfg).get("max_steps", 12))
    return [s for s in items if s and not is_unsafe(s, cfg)][:limit]


def sanitize_article(
    article: dict[str, Any], cfg: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    out = dict(article)
    out["diagnostic_steps"] = _safe([str(s) for s in out.get("diagnostic_steps") or []], cfg)
    res = dict(out.get("resolution") or {})
    res["steps"] = _safe([str(s) for s in res.get("steps") or []], cfg)
    out["resolution"] = res
    return out


# --- KB listing helpers (work with Chroma and fake stores) ---------------------------------------
def all_articles(kb_store: Any, include_unapproved: bool = True) -> list[KBArticle]:
    if kb_store is None:
        return []
    if hasattr(kb_store, "articles"):
        return list(kb_store.articles(include_unapproved=include_unapproved))
    if hasattr(kb_store, "kb_ids"):
        arts = [kb_store.get_article(k) for k in kb_store.kb_ids()]
        return [a for a in arts if include_unapproved or a.status == "approved"]
    return []


def next_kb_id(conn: sqlite3.Connection, kb_store: Any) -> str:
    nums = [int(m.group(1)) for a in all_articles(kb_store) if (m := _KB_NUM.match(a.kb_id))]
    for row in conn.execute("SELECT article_json FROM kb_drafts"):
        m = _KB_NUM.match(json.loads(row["article_json"]).get("kb_id", ""))
        if m:
            nums.append(int(m.group(1)))
    return f"KB-{max(nums, default=0) + 1:04d}"


# --- drafting ------------------------------------------------------------------------------------
def analyst_notes(conn: sqlite3.Connection, analysis_id: str) -> str:
    """Redacted notes: the decision's resolution notes + unresolved/incorrect feedback comments."""
    parts: list[str] = []
    dec = conn.execute(
        "SELECT resolution_notes FROM decisions WHERE analysis_id=?", (analysis_id,)
    ).fetchone()
    if dec and dec["resolution_notes"]:
        parts.append(dec["resolution_notes"])
    for row in conn.execute(
        "SELECT comment FROM feedback WHERE analysis_id=? "
        "AND verdict IN ('unresolved','incorrect') AND comment IS NOT NULL ORDER BY created_at",
        (analysis_id,),
    ):
        parts.append(row["comment"])
    return "\n".join(p.strip() for p in parts if p and p.strip())


def template_article(notes: str, title: str, category: str, subcategory: str) -> dict[str, Any]:
    steps = [ln.strip(_BULLET) for ln in re.split(r"[\n;]+", notes) if ln.strip(_BULLET)]
    return {
        "title": title,
        "symptoms": [title],
        "common_error_codes": [],
        "root_cause": None,
        "diagnostic_steps": [],
        "resolution": {"steps": steps, "estimated_time": None, "requires_admin": False},
        "escalation_criteria": [],
        "tags": [subcategory.lower()] if subcategory else [],
    }


def llm_article(
    services: Services, notes: str, title: str, category: str, subcategory: str
) -> dict[str, Any] | None:
    if not getattr(services.llm, "available", False):
        return None
    user = (
        f"Ticket title: {title}\nCategory: {category} / {subcategory}\n<notes>\n{notes}\n</notes>"
    )
    try:
        out = services.llm.complete_json(
            purpose="kb_draft", system=SYSTEM_PROMPT, user=user, max_tokens=900
        )
    except LLMUnavailableError:
        return None
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to the template
        log.warning("kb_draft LLM call failed: %s", type(exc).__name__)
        return None
    if not isinstance(out, dict) or out.get("insufficient_evidence") or not out.get("resolution"):
        return None
    return out


def _draft_out(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["article"] = json.loads(out.pop("article_json"))
    return out


def create_draft(
    services: Services, conn: sqlite3.Connection, analysis_id: str, user: User
) -> dict[str, Any]:
    access.require(conn, user, access.DRAFT_ROLES, "kb draft")
    a = repo.get_analysis(conn, analysis_id)
    if a is None:
        raise not_found(f"analysis {analysis_id}")
    notes = analyst_notes(conn, analysis_id)
    if not notes:
        raise bad_request("record resolution notes (decision) or an unresolved comment first")
    ticket = repo.get_ticket(conn, a["ticket_id"]) or {}
    cls = a["contract"]["classification"]
    title = str(ticket.get("title") or cls["subcategory"])
    drafted = llm_article(services, notes, title, cls["category"], cls["subcategory"])
    drafted_by = "llm" if drafted is not None else "template"
    body = drafted or template_article(notes, title, cls["category"], cls["subcategory"])
    body = sanitize_article(body)
    if not body["resolution"]["steps"]:
        raise bad_request("no safe resolution steps could be drafted from the notes")
    _, kb_store = services.stores()
    article = {
        **{k: v for k, v in body.items() if k in EDITABLE},
        "kb_id": next_kb_id(conn, kb_store),
        "category": cls["category"],
        "subcategory": cls["subcategory"],
        "status": "draft",
        "version": 0,
        "source": "analyst",
        "drafted_by": drafted_by,
    }
    try:
        art = KBArticle.model_validate(article)
    except ValidationError:
        article.update(template_article(notes, title, cls["category"], cls["subcategory"]))
        article["drafted_by"] = "template"
        art = KBArticle.model_validate(sanitize_article(article))
    row = {
        "draft_id": str(uuid.uuid4()),
        "article_json": art.model_dump_json(),
        "status": "draft",
        "proposed_by": user.username,
        "source_analysis_id": analysis_id,
        "approved_by": None,
        "created_at": repo.iso(services.clock()),
        "approved_at": None,
    }
    conn.execute(
        "INSERT INTO kb_drafts (draft_id, article_json, status, proposed_by, source_analysis_id, "
        "approved_by, created_at, approved_at) VALUES (:draft_id, :article_json, :status, "
        ":proposed_by, :source_analysis_id, :approved_by, :created_at, :approved_at)",
        row,
    )
    audit.append(
        conn, actor=user.username, action="kb.draft.created", entity="kb_draft",
        entity_id=str(row["draft_id"]),
        payload={"analysis_id": analysis_id, "kb_id": art.kb_id, "status": "draft"},
    )  # fmt: skip
    return _draft_out(row)


def list_drafts(conn: sqlite3.Connection, status: str | None = None) -> list[dict[str, Any]]:
    sql, params = "SELECT * FROM kb_drafts", []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    return [_draft_out(dict(r)) for r in conn.execute(sql + " ORDER BY created_at", params)]


# --- approval + publication -------------------------------------------------------------------
def _seed_articles() -> list[KBArticle]:
    path = ROOT / base_config()["paths"]["kb_articles"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        KBArticle.model_validate(a).model_copy(update={"status": "approved", "source": "seed"})
        for a in raw
    ]


def write_governed(path: Path, article: KBArticle) -> None:
    """Upsert ``article`` into the governed KEDB file (P1's format: the full list)."""
    if path.exists():
        arts = [KBArticle.model_validate(a) for a in json.loads(path.read_text(encoding="utf-8"))]
    else:
        arts = _seed_articles()
    by_id = {a.kb_id: a for a in arts}
    by_id[article.kb_id] = article
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [by_id[k].model_dump(mode="json") for k in sorted(by_id)]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_publisher(services: Services) -> Callable[[KBArticle], None]:
    """Make an approved article searchable without a full rebuild.

    Real stores: P1's ``ingest.index_kedb.reindex_article(kb_id, article, kb_store=...)`` writes the
    governed file, re-chunks the article into Chroma and refreshes the store's in-memory view.
    Stub stores: the fake KB store is rebuilt with the article added.
    """

    def publish(article: KBArticle) -> None:
        kb = services.kb_store
        if not services.settings.use_stub_stores:
            from ingest.index_kedb import reindex_article

            reindex_article(article.kb_id, article, kb_store=kb)
            return
        if kb is not None and hasattr(kb, "refresh_article"):
            kb.refresh_article(article)
        elif kb is not None and hasattr(kb, "kb_ids"):
            arts = {a.kb_id: a for a in all_articles(kb)}
            arts[article.kb_id] = article
            services.kb_store = type(kb)(list(arts.values()))

    return publish


def approve_draft(
    services: Services,
    conn: sqlite3.Connection,
    draft_id: str,
    edits: Mapping[str, Any] | None,
    user: User,
) -> KBArticle:
    access.require(conn, user, access.APPROVE_ROLES, "kb approve")
    row = conn.execute("SELECT * FROM kb_drafts WHERE draft_id = ?", (draft_id,)).fetchone()
    if row is None:
        raise not_found(f"draft {draft_id}")
    if row["status"] != "draft":
        raise ApiError(400, "validation", f"draft is already {row['status']}")
    unknown = set(edits or {}) - EDITABLE
    if unknown:
        raise bad_request(f"fields not editable: {', '.join(sorted(unknown))}")
    article = {**json.loads(row["article_json"]), **(edits or {})}
    before = sum(len(article.get(k) or []) for k in ("diagnostic_steps",))
    before += len((article.get("resolution") or {}).get("steps") or [])
    article = sanitize_article(article)
    after = len(article["diagnostic_steps"]) + len(article["resolution"]["steps"])
    if after < before:
        raise bad_request("the edits contain unsafe steps")
    article.update(
        status="approved",
        version=int(article.get("version") or 0) + 1,
        approved_by=user.username,
    )
    try:
        art = KBArticle.model_validate(article)
    except ValidationError as exc:
        raise bad_request(f"invalid article: {exc.errors()[0].get('msg')}") from exc
    now = repo.iso(services.clock())
    if services.settings.use_stub_stores:
        write_governed(services.settings.kb_governed_path, art)
    if services.kb_publisher is not None:
        services.kb_publisher(art)
    conn.execute(
        "UPDATE kb_drafts SET status='approved', approved_by=?, approved_at=?, article_json=? "
        "WHERE draft_id=?",
        (user.username, now, art.model_dump_json(), draft_id),
    )
    audit.append(
        conn, actor=user.username, action="kb.approved", entity="kb", entity_id=art.kb_id,
        payload={"draft_id": draft_id, "kb_id": art.kb_id, "version": art.version},
    )  # fmt: skip
    return art
