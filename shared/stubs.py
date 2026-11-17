"""In-memory fakes for every Protocol in ``shared/interfaces.py``.

Use these until the real implementation lands (README §8.2). They are deterministic, need no
network or model downloads, and are backed by ``shared/fixtures``. Scoring is deliberately
naive (token overlap); don't tune anything against it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shared.constants import (
    ASSIGNMENT_GROUP,
    BASELINE_PRIORITY,
    PIPELINE_VERSION,
    PRIORITY_LEVEL,
    SEARCHABLE_CHUNK_TYPES,
)
from shared.interfaces import DEFAULT_TICKET_COLLECTIONS, LLMUnavailableError
from shared.schemas import (
    AnalysisResponse,
    ChunkHit,
    ClassificationResult,
    EngineContext,
    EngineResult,
    IssueResult,
    KBArticle,
    KBHit,
    PriorityDetails,
    PriorityResult,
    Recommendation,
    Redaction,
    RedactionResult,
    SanitizedTicket,
    Signals,
    TicketHit,
    TicketRecord,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = FIXTURES_DIR / "golden"

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    (  # noqa: SIM905
        "a an and are as at be but by for from has have i in is it its my of on or our so that the "
        "this to was we when with not no can cannot"
    ).split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP}


def _overlap(query: set[str], doc: set[str]) -> float:
    """Share of query tokens found in the document, in [0, 1]."""
    if not query or not doc:
        return 0.0
    return len(query & doc) / len(query)


def _norm_code(code: str) -> str:
    """Error-code normalisation from architecture §6.4: lowercase, collapse spaces, drop punct."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", code.lower())).strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Tiny subset of Chroma ``where``: equality, $eq/$ne/$in/$nin/$gte/$lte/$gt/$lt, $and/$or."""
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(_matches_where(meta, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(_matches_where(meta, c) for c in cond):
                return False
            continue
        val = meta.get(key)
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, ref in cond.items():
            ok = {
                "$eq": lambda v, r: v == r,
                "$ne": lambda v, r: v != r,
                "$in": lambda v, r: v in r,
                "$nin": lambda v, r: v not in r,
                "$gte": lambda v, r: v is not None and v >= r,
                "$lte": lambda v, r: v is not None and v <= r,
                "$gt": lambda v, r: v is not None and v > r,
                "$lt": lambda v, r: v is not None and v < r,
            }[op](val, ref)
            if not ok:
                return False
    return True


# =============================================================================================
# Stores
# =============================================================================================
class FakeTicketStore:
    """``TicketStore`` over ``mini_tickets.json`` with keyword-overlap scoring."""

    def __init__(self, records: Iterable[dict[str, Any] | TicketRecord] | None = None) -> None:
        raw = records if records is not None else _load_json(FIXTURES_DIR / "mini_tickets.json")
        self._recs: dict[str, TicketRecord] = {}
        for r in raw:
            rec = r if isinstance(r, TicketRecord) else TicketRecord.model_validate(r)
            self._recs[rec.ticket_id] = rec

    def _text(self, rec: TicketRecord) -> str:
        if rec.title == rec.description:
            return rec.description
        return f"{rec.title}. {rec.description}"

    def _hit(self, rec: TicketRecord, score: float) -> TicketHit:
        return TicketHit(
            ticket_id=rec.ticket_id,
            score=round(score, 4),
            title=rec.title,
            excerpt=rec.description[:200],
            category=rec.category,
            subcategory=rec.subcategory,
            priority=rec.priority,
            status=rec.status,
            resolution=rec.resolution,
            resolution_origin=rec.resolution_origin,
            collection=rec.collection,
        )

    def query(
        self,
        text: str,
        k: int,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = DEFAULT_TICKET_COLLECTIONS,
    ) -> list[TicketHit]:
        q = _tokens(text)
        scored = [
            (_overlap(q, _tokens(self._text(r))), r)
            for r in self._recs.values()
            if r.collection in collections and _matches_where(r.model_dump(), where)
        ]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1].ticket_id))
        return [self._hit(r, s) for s, r in scored[:k]]

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return [(h.ticket_id, h.score) for h in self.query(text, k)]

    def get(self, ids: list[str]) -> list[TicketRecord]:
        return [self._recs[i] for i in ids if i in self._recs]

    def upsert_live(self, rec: TicketRecord) -> None:
        self._recs[rec.ticket_id] = rec.model_copy(update={"collection": "tickets_live"})

    def mark_verified(self, ticket_id: str, resolution: str, by: str) -> None:
        rec = self._recs[ticket_id]
        self._recs[ticket_id] = rec.model_copy(
            update={
                "collection": "tickets_verified",
                "resolution": resolution,
                "resolution_origin": "analyst_verified",
                "verified": True,
                "verified_by": by,
            }
        )


class FakeKBStore:
    """``KBStore`` over ``mini_articles.json``, chunked as in architecture §6.4."""

    def __init__(self, articles: Iterable[dict[str, Any] | KBArticle] | None = None) -> None:
        raw = articles if articles is not None else _load_json(FIXTURES_DIR / "mini_articles.json")
        self._articles: dict[str, KBArticle] = {}
        for a in raw:
            art = a if isinstance(a, KBArticle) else KBArticle.model_validate(a)
            self._articles[art.kb_id] = art
        self._chunks: dict[str, ChunkHit] = {}
        self._codes: dict[str, list[str]] = {}
        for art in self._articles.values():
            for chunk in self._chunk(art):
                self._chunks[chunk.chunk_id] = chunk
            for code in art.common_error_codes:
                self._codes.setdefault(_norm_code(code), []).append(art.kb_id)

    @staticmethod
    def _chunk(art: KBArticle) -> list[ChunkHit]:
        meta = {
            "category": art.category,
            "subcategory": art.subcategory,
            "priority_hint": art.priority_hint,
            "requires_admin": art.resolution.requires_admin,
            "status": art.status,
            "version": art.version,
        }

        def mk(cid: str, ctype: str, text: str) -> ChunkHit:
            return ChunkHit(
                chunk_id=f"{art.kb_id}#{cid}",
                kb_id=art.kb_id,
                chunk_type=ctype,  # type: ignore[arg-type]
                text=text,
                score=0.0,
                metadata={**meta, "chunk_type": ctype, "kb_id": art.kb_id},
            )

        out = [mk("title", "title_root", f"{art.title}. {art.root_cause or ''}".strip())]
        out += [mk(f"sym.{i}", "symptom", s) for i, s in enumerate(art.symptoms, 1)]
        if art.common_error_codes:
            out.append(mk("err", "errors", "; ".join(art.common_error_codes)))
        out += [mk(f"diag.{i}", "diagnostic", s) for i, s in enumerate(art.diagnostic_steps, 1)]
        out += [mk(f"res.{i}", "resolution", s) for i, s in enumerate(art.resolution.steps, 1)]
        if art.escalation_criteria:
            out.append(mk("esc", "escalation", "; ".join(art.escalation_criteria)))
        return out

    def query_chunks(
        self, text: str, k: int, where: dict[str, Any] | None = None
    ) -> list[ChunkHit]:
        q = _tokens(text)
        scored = []
        for c in self._chunks.values():
            if c.chunk_type not in SEARCHABLE_CHUNK_TYPES or c.metadata.get("status") != "approved":
                continue
            if not _matches_where(c.metadata, where):
                continue
            s = _overlap(q, _tokens(c.text))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda s: (-s[0], s[1].chunk_id))
        return [c.model_copy(update={"score": round(s, 4)}) for s, c in scored[:k]]

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]:
        return [(c.chunk_id, c.score) for c in self.query_chunks(text, k)]

    def lookup_codes(self, codes: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for code in codes:
            ids = self._codes.get(_norm_code(code))
            if ids:
                out[code] = list(ids)
        return out

    def get_article(self, kb_id: str) -> KBArticle:
        return self._articles[kb_id]

    def chunk_text(self, chunk_id: str) -> str:
        return self._chunks[chunk_id].text

    def all_error_phrases(self) -> list[str]:
        phrases = {c for a in self._articles.values() for c in a.common_error_codes}
        return sorted(phrases)

    def kb_ids(self) -> list[str]:
        """Not part of the Protocol; handy for tests."""
        return sorted(self._articles)


# =============================================================================================
# LLM
# =============================================================================================
class FakeLLM:
    """``LLMClient`` that returns canned JSON per ``purpose``.

    ``FakeLLM(available=False)`` behaves like offline mode: every call raises
    ``LLMUnavailableError`` so callers exercise their fallback path.
    """

    DEFAULT_RESPONSES: dict[str, dict[str, Any]] = {
        "classify": {
            "issues": [
                {
                    "text_span": "",
                    "category": "Network",
                    "subcategory": "VPN",
                    "intent": "incident",
                    "scores": {"Network/VPN": 0.9},
                }
            ]
        },
        "synthesize": {"insufficient_evidence": True, "steps": []},
        "kb_draft": {"insufficient_evidence": True},
    }

    def __init__(
        self, responses: dict[str, dict[str, Any]] | None = None, available: bool = True
    ) -> None:
        self._responses = {**self.DEFAULT_RESPONSES, **(responses or {})}
        self._available = available
        self.calls: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self._available

    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        self.calls.append({"purpose": purpose, "model": model, "system": system, "user": user})
        if not self._available:
            raise LLMUnavailableError("FakeLLM is offline")
        if purpose not in self._responses:
            raise LLMUnavailableError(f"FakeLLM has no canned response for {purpose!r}")
        return json.loads(json.dumps(self._responses[purpose]))  # deep copy


# =============================================================================================
# Engine
# =============================================================================================
def load_golden(ticket_id: str) -> AnalysisResponse | None:
    path = GOLDEN_DIR / f"{ticket_id}.json"
    if not path.exists():
        return None
    return AnalysisResponse.model_validate(_load_json(path))


def golden_ids() -> list[str]:
    return sorted(p.stem for p in GOLDEN_DIR.glob("*.json"))


def _issue_from_golden(
    text_span: str,
    classification: Any,
    cls_details: Any,
    priority: Any,
    pri_details: Any,
    similar: list[TicketHit],
    kedb: Any,
    kb_match_found: bool,
    steps: list[Any],
    source: str,
    kb_store: FakeKBStore | None,
    intent: str = "incident",
) -> IssueResult:
    kb_matches: list[KBHit] = []
    if kedb is not None:
        requires_admin = False
        if kb_store is not None:
            with contextlib.suppress(KeyError):
                requires_admin = kb_store.get_article(kedb.kb_id).resolution.requires_admin
        kb_matches.append(
            KBHit(
                kb_id=kedb.kb_id,
                title=kedb.title,
                score=kedb.score,
                matched_field=kedb.matched_field,
                excerpt=kedb.excerpt,
                why=kedb.why,
                matched_error_code=kedb.matched_error_code,
                requires_admin=requires_admin,
            )
        )
        for alt in kedb.alternatives:
            title = alt.kb_id
            if kb_store is not None:
                with contextlib.suppress(KeyError):
                    title = kb_store.get_article(alt.kb_id).title
            kb_matches.append(KBHit(kb_id=alt.kb_id, title=title, score=alt.score))
    cd = cls_details
    return IssueResult(
        text_span=text_span,
        classification=ClassificationResult(
            category=classification.category,
            subcategory=classification.subcategory,
            confidence=classification.confidence,
            band=cd.band if cd else "medium",
            intent=intent,  # type: ignore[arg-type]
            components=cd.components if cd else {},
            alternatives=cd.alternatives if cd else [],
            rationale=cd.rationale if cd else None,
            conflict=cd.conflict if cd else False,
        ),
        priority=PriorityResult(
            code=pri_details.code,
            level=priority.level,
            reason=priority.reason,
            impact=pri_details.impact,
            urgency=pri_details.urgency,
            drivers=pri_details.drivers,
        ),
        similar=similar,
        kb_matches=kb_matches,
        kb_relevance=kedb.score if kedb and kb_match_found else 0.0,
        recommendation=Recommendation(
            steps=steps,
            source=source,  # type: ignore[arg-type]
            insufficient_evidence=not steps,
            requires_admin=any(m.requires_admin for m in kb_matches[:1]),
        ),
    )


def engine_result_from_golden(
    resp: AnalysisResponse, kb_store: FakeKBStore | None = None
) -> EngineResult:
    """Rebuild the engine-level view of a golden response (decision fields are P3's, dropped)."""
    d = resp.details
    by_id = {s.incident_id: s for s in d.similar}
    similar = [
        TicketHit(
            ticket_id=s.incident_id,
            score=s.similarity_score,
            resolution=s.resolution,
            tier=by_id[s.incident_id].tier if s.incident_id in by_id else None,
            why=by_id[s.incident_id].why if s.incident_id in by_id else None,
            excerpt=by_id[s.incident_id].excerpt if s.incident_id in by_id else None,
            resolution_origin=(
                by_id[s.incident_id].resolution_origin if s.incident_id in by_id else "none"
            ),
            category=by_id[s.incident_id].category if s.incident_id in by_id else None,
            subcategory=by_id[s.incident_id].subcategory if s.incident_id in by_id else None,
        )
        for s in resp.similar_incidents
    ]
    issues = [
        _issue_from_golden(
            "primary",
            resp.classification,
            d.classification,
            resp.priority,
            d.priority,
            similar,
            d.kedb,
            resp.kedb_match.found,
            list(d.steps),
            resp.recommended_solution.source,
            kb_store,
            intent=d.intent,
        )
    ]
    for sec in d.secondary_issues:
        pri_details = sec.priority_details
        if pri_details is None:
            pri_details = PriorityDetails(code="P3", impact=3, urgency=2)
        issues.append(
            _issue_from_golden(
                "secondary",
                sec.classification,
                sec.classification_details,
                sec.priority,
                pri_details,
                [],
                sec.kedb,
                sec.kedb_match.found,
                list(sec.steps),
                sec.recommended_solution.source,
                kb_store,
            )
        )
    return EngineResult(
        signals=d.signals or Signals(),
        issues=issues,
        degraded=list(d.degraded),
        timings_ms={"total": 1},
        versions={**d.versions, "engine": "FakeEngine"},
    )


class FakeEngine:
    """``Engine`` that replays golden fixtures by ticket ID.

    Unknown tickets get a generic low-confidence result with no grounded steps, which the
    decision gate should turn into clarify/escalate.
    """

    def __init__(self, kb_store: FakeKBStore | None = None) -> None:
        self._kb = kb_store or FakeKBStore()

    def run(self, ticket: SanitizedTicket, ctx: EngineContext) -> EngineResult:
        golden = load_golden(ticket.ticket_id)
        if golden is not None:
            return engine_result_from_golden(golden, self._kb)
        return self._generic(ticket)

    @staticmethod
    def _generic(ticket: SanitizedTicket) -> EngineResult:
        category, subcategory = "Inquiry/Help", "Other"
        code = BASELINE_PRIORITY[category]
        return EngineResult(
            signals=Signals(),
            issues=[
                IssueResult(
                    text_span=f"{ticket.title}\n{ticket.description}",
                    classification=ClassificationResult(
                        category=category,
                        subcategory=subcategory,
                        confidence=0.3,
                        band="low",
                        components={"llm": 0.0, "knn": 0.3, "rule": 0.0, "margin": 0.0, "agree": 0},
                        rationale="FakeEngine: no golden fixture for this ticket",
                    ),
                    priority=PriorityResult(
                        code=code,  # type: ignore[arg-type]
                        level=PRIORITY_LEVEL[code],  # type: ignore[arg-type]
                        reason=f"Baseline for {subcategory}",
                        impact=3,
                        urgency=3,
                    ),
                    recommendation=Recommendation(insufficient_evidence=True),
                )
            ],
            degraded=[],
            timings_ms={"total": 1},
            versions={"pipeline": PIPELINE_VERSION, "engine": "FakeEngine"},
        )

    @staticmethod
    def assignment_group(category: str) -> str:
        return ASSIGNMENT_GROUP.get(category, "SERVICE-DESK")


# =============================================================================================
# Redaction
# =============================================================================================
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SECRET = re.compile(r"(?i)\b(password|pwd)\s*[:=]\s*\S+")


def fake_redact(text: str, keep_map: bool = True) -> RedactionResult:
    """``Redactor`` stand-in: emails → ``<EMAIL_n>``, ``password=…`` → ``<SECRET>``."""
    items: list[Redaction] = []
    seen: dict[str, str] = {}

    def _email(m: re.Match[str]) -> str:
        val = m.group(0)
        if val not in seen:
            seen[val] = f"<EMAIL_{len(seen) + 1}>"
        items.append(
            Redaction(
                type="EMAIL",
                placeholder=seen[val],
                span=(m.start(), m.end()),
                original_hash=hashlib.sha256(val.encode()).hexdigest(),
            )
        )
        return seen[val]

    def _secret(m: re.Match[str]) -> str:
        items.append(
            Redaction(
                type="SECRET",
                placeholder="<SECRET>",
                span=(m.start(), m.end()),
                original_hash=hashlib.sha256(m.group(0).encode()).hexdigest(),
            )
        )
        return f"{m.group(1)}=<SECRET>"

    out = _SECRET.sub(_secret, text)
    secret = any(i.type == "SECRET" for i in items)
    out = _EMAIL.sub(_email, out)
    return RedactionResult(text=out, items=items, secret_detected=secret)
