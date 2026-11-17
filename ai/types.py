"""P2-local data types and dependency Protocols for the AI engine.

Phase 0 (`shared/schemas.py`, `shared/interfaces.py`) has not landed yet, so the AI engine
defines the minimal types it needs here, following architecture §6.6, §7 and §17.
When `shared/` is merged these can be swapped for (or aliased to) the shared models
without changing any call sites: every engine function receives its dependencies as
parameters and only touches the attributes declared below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Scope = Literal["single", "multiple", "team", "site", "org"]
Band = Literal["high", "medium", "low"]
Intent = Literal["incident", "service_request", "how_to", "access_request"]
StepType = Literal["diagnostic", "resolution"]
Role = Literal["L1", "L2"]

VERIFIED_ORIGINS: frozenset[str] = frozenset({"curated_verified", "analyst_verified"})
SYNTHETIC_ORIGINS: frozenset[str] = frozenset(
    {"synthetic_kb_derived", "synthetic_escalated"}
)


class _AsDict:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


# --------------------------------------------------------------------------- inputs


@dataclass
class SanitizedTicket(_AsDict):
    """Redacted ticket handed to the engine (no `metadata.expected_*`)."""

    ticket_id: str
    title: str
    description: str
    application: str | None = None
    environment: str | None = None
    location: str | None = None
    restricted: bool = False
    secret_detected: bool = False

    @property
    def text(self) -> str:
        if (
            self.title
            and self.title.strip()
            and self.title.strip() not in self.description
        ):
            return f"{self.title.strip()}. {self.description.strip()}"
        return self.description.strip()


# --------------------------------------------------------------------------- store hits


@dataclass
class TicketHit(_AsDict):
    """One historical/verified/live ticket returned by `TicketStore.query`."""

    ticket_id: str
    score: float  # cosine similarity in [0, 1]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    collection: str = "tickets_hist"


@dataclass
class ChunkHit(_AsDict):
    """One KEDB chunk returned by `KBStore.query_chunks`."""

    chunk_id: str  # e.g. "KB-0018#sym.2"
    kb_id: str
    chunk_type: (
        str  # title_root | symptom | errors | diagnostic | resolution | escalation
    )
    score: float
    text: str


@dataclass
class KBResolution(_AsDict):
    steps: list[str] = field(default_factory=list)
    estimated_time: str | None = None
    requires_admin: bool = False


@dataclass
class KBArticle(_AsDict):
    kb_id: str
    title: str
    category: str = ""
    subcategory: str = ""
    symptoms: list[str] = field(default_factory=list)
    common_error_codes: list[str] = field(default_factory=list)
    root_cause: str = ""
    diagnostic_steps: list[str] = field(default_factory=list)
    resolution: KBResolution = field(default_factory=KBResolution)
    escalation_criteria: list[str] = field(default_factory=list)
    related_kb_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    priority_hint: str | None = None
    status: str = "approved"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KBArticle:
        res = d.get("resolution") or {}
        esc = d.get("escalation_criteria") or []
        if isinstance(esc, str):
            esc = [esc]
        return cls(
            kb_id=d["kb_id"],
            title=d.get("title", ""),
            category=d.get("category", "") or "",
            subcategory=d.get("subcategory", "") or "",
            symptoms=list(d.get("symptoms") or []),
            common_error_codes=list(d.get("common_error_codes") or []),
            root_cause=d.get("root_cause", "") or "",
            diagnostic_steps=list(d.get("diagnostic_steps") or []),
            resolution=KBResolution(
                steps=list(res.get("steps") or []),
                estimated_time=res.get("estimated_time"),
                requires_admin=bool(res.get("requires_admin", False)),
            ),
            escalation_criteria=list(esc),
            related_kb_ids=list(d.get("related_kb_ids") or []),
            tags=list(d.get("tags") or []),
            priority_hint=d.get("priority_hint"),
            status=d.get("status", "approved") or "approved",
        )


# --------------------------------------------------------------------------- protocols


@runtime_checkable
class TicketStore(Protocol):
    def query(
        self,
        text: str,
        k: int,
        where: dict[str, Any] | None = None,
        collections: tuple[str, ...] = (
            "tickets_hist",
            "tickets_verified",
            "tickets_live",
        ),
    ) -> list[TicketHit]: ...

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]: ...

    def get(self, ids: list[str]) -> list[TicketHit]: ...


@runtime_checkable
class KBStore(Protocol):
    def query_chunks(
        self, text: str, k: int, where: dict[str, Any] | None = None
    ) -> list[ChunkHit]: ...

    def bm25(self, text: str, k: int) -> list[tuple[str, float]]: ...

    def lookup_codes(self, codes: list[str]) -> dict[str, list[str]]: ...

    def get_article(self, kb_id: str) -> KBArticle: ...

    def chunk_text(self, chunk_id: str) -> str: ...

    def all_error_phrases(self) -> list[str]: ...


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot be used (offline, circuit open, quota, network)."""


class LLMBadOutput(ValueError):
    """Raised when the LLM answered but not with a parseable JSON object."""


@runtime_checkable
class LLMClient(Protocol):
    @property
    def available(self) -> bool: ...

    def complete_json(
        self,
        *,
        purpose: str,
        model: str,
        messages: list[dict[str, str]],
        analysis_id: str | None = None,
    ) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- engine outputs


@dataclass
class ClassificationResult(_AsDict):
    category: str
    subcategory: str
    confidence: float
    band: Band
    intent: Intent = "incident"
    conflict: bool = False
    components: dict[str, float] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    scores: dict[str, float] = field(default_factory=dict)  # "Cat/Sub" -> fused score


@dataclass
class PriorityResult(_AsDict):
    code: str  # P1..P4
    level: str  # Critical | High | Medium | Low
    reason: str
    impact: int | None = None
    urgency: int | None = None
    drivers: list[str] = field(default_factory=list)


@dataclass
class SimilarIncident(_AsDict):
    incident_id: str
    similarity_score: float
    resolution: str | None
    tier: str  # DUPLICATE | LIKELY DUPLICATE | SIMILAR | RELATED
    excerpt: str
    why: str
    resolution_origin: str
    category: str | None = None
    subcategory: str | None = None
    collection: str = "tickets_hist"
    fused_score: float = 0.0


@dataclass
class TicketSearchResult(_AsDict):
    similar: list[SimilarIncident] = field(default_factory=list)
    keyword_only: list[SimilarIncident] = field(default_factory=list)
    possible_major_incident: bool = False
    pattern: dict[str, Any] = field(default_factory=dict)


@dataclass
class KBMatch(_AsDict):
    kb_id: str
    title: str
    score: float
    matched_field: str
    excerpt: str
    why: str
    matched_error_code: str | None = None
    components: dict[str, float] = field(default_factory=dict)
    subcategory: str = ""


@dataclass
class KBSearchResult(_AsDict):
    matches: list[KBMatch] = field(default_factory=list)
    relevance: float = 0.0
    found: bool = False
    ambiguous: bool = False


@dataclass
class Step(_AsDict):
    text: str
    type: StepType
    citations: list[str]
    role: Role = "L1"
    already_attempted: bool = False
    n: int = 0


@dataclass
class Recommendation(_AsDict):
    steps: list[Step] = field(default_factory=list)
    source: str = (
        "None"  # KEDB | Historical Incident | KEDB + Historical Incident | None
    )
    summary: str = ""
    probable_cause: dict[str, Any] = field(default_factory=dict)
    estimated_time: str | None = None
    requires_admin: bool = False
    insufficient_evidence: bool = False
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueResult(_AsDict):
    text_span: str
    classification: ClassificationResult
    priority: PriorityResult
    similar: TicketSearchResult
    kb_matches: KBSearchResult
    recommendation: Recommendation


@dataclass
class EngineContext:
    ticket_store: TicketStore
    kb_store: KBStore
    llm: LLMClient
    role: str = "l1"
    feedback_prior: dict[str, float] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    clarification: str | None = None
    analysis_id: str | None = None


@dataclass
class EngineResult(_AsDict):
    signals: dict[str, Any]
    issues: list[IssueResult]
    degraded: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    priority: PriorityResult | None = (
        None  # ticket priority = most severe across issues
    )
