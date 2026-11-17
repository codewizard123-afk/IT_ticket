"""Pydantic v2 models shared by every module (architecture §17).

Frozen after Phase 0: add fields, never rename or remove (README §8.3).

Three groups of models live here:

* **Contract** (architecture §4.1): the 8-key response. These objects forbid extra keys.
* **Details** (architecture §4.3): the ``details`` extension returned by ``view=full``.
* **Internal**: tickets, KB articles, store hits, engine I/O, feedback, events and users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # avoid a runtime import cycle; interfaces only needs schemas for typing
    from shared.interfaces import KBStore, LLMClient, TicketStore

# --- Literal aliases (keep in sync with shared/constants.py) ---------------------------------
PriorityCode = Literal["P1", "P2", "P3", "P4"]
PriorityLevel = Literal["Critical", "High", "Medium", "Low"]
Intent = Literal["incident", "service_request", "how_to", "access_request"]
Band = Literal["high", "medium", "low"]
DecisionKind = Literal["recommend", "clarify", "escalate"]
Source = Literal["KEDB", "Historical Incident", "KEDB + Historical Incident", "None"]
StepType = Literal["diagnostic", "resolution"]
StepRole = Literal["L1", "L2"]
Tier = Literal["L2", "L3"]
DupTier = Literal["DUPLICATE", "LIKELY DUPLICATE", "SIMILAR", "RELATED"]
Scope = Literal["single", "multiple", "team", "site", "org"]
ResolutionOrigin = Literal[
    "synthetic_kb_derived", "synthetic_escalated", "curated_verified", "analyst_verified", "none"
]
Role = Literal["l1", "l2", "kb_owner", "admin"]
KBStatus = Literal["approved", "draft", "retired"]
ChunkType = Literal["title_root", "symptom", "errors", "diagnostic", "resolution", "escalation"]
EscalationRule = Literal["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
FeedbackTarget = Literal[
    "classification", "similar_incident", "kedb_match", "step", "recommendation"
]
FeedbackVerdict = Literal["useful", "incorrect", "unresolved"]
FinalDecision = Literal["resolved", "escalated", "needs_more_info"]
Mode = Literal["assisted", "manual"]


class _Strict(BaseModel):
    """Contract objects: no extra keys, ever (CLAUDE.md rule 3)."""

    model_config = ConfigDict(extra="forbid")


class _Open(BaseModel):
    """Extension objects: unknown keys are kept so additive changes don't break readers."""

    model_config = ConfigDict(extra="allow")


# =============================================================================================
# Input
# =============================================================================================
class TicketIn(_Open):
    """An incoming ticket, in the ``sample_tickets.json`` format (architecture §2.2)."""

    id: str | None = None
    title: str
    description: str
    category: str | None = None
    subcategory: str | None = None
    priority: str | None = None
    status: str = "new"
    resolution: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional extras for FR-07 and access control
    application: str | None = None
    environment: str | None = None
    location: str | None = None
    reporter: str | None = None
    assignment_group: str | None = None
    restricted: bool = False


class Redaction(_Strict):
    type: str  # EMAIL, PHONE, IP, PERSON, ID, SECRET, TOKEN
    placeholder: str
    span: tuple[int, int]
    original_hash: str


class RedactionResult(_Strict):
    text: str
    items: list[Redaction] = Field(default_factory=list)
    secret_detected: bool = False


class SanitizedTicket(_Strict):
    """What the engine receives: redacted text, no gold labels (``metadata.expected_*``)."""

    ticket_id: str
    title: str
    description: str
    application: str | None = None
    environment: str | None = None
    location: str | None = None
    assignment_group: str | None = None
    restricted: bool = False
    secret_detected: bool = False
    clarification: str | None = None


# =============================================================================================
# Stored records
# =============================================================================================
class TicketRecord(_Open):
    """A historical, verified or live ticket as held by the stores (architecture §6.3, §11.1)."""

    ticket_id: str = Field(validation_alias=AliasChoices("ticket_id", "id"))
    title: str
    description: str
    category: str | None = None
    subcategory: str | None = None
    priority: str | None = None
    status: str = "closed"
    source: str | None = None
    resolution: str | None = None
    resolution_origin: ResolutionOrigin = "none"
    resolution_kb_id: str | None = None
    opened_at: str | None = None
    resolved_at: str | None = None
    opened_ts: int | None = None
    resolved_ts: int | None = None
    assignment_group: str | None = None
    application: str | None = None
    environment: str | None = None
    location: str | None = None
    restricted: bool = False
    verified: bool = False
    verified_by: str | None = None
    collection: str = "tickets_hist"
    derived_fields: list[str] = Field(default_factory=list)
    dup_group_id: str | None = None
    text_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBResolution(_Open):
    steps: list[str]
    estimated_time: str | None = None
    requires_admin: bool = False


class KBArticle(_Open):
    """A KEDB runbook (``data/kb/articles.json``) plus governance fields (architecture §2.3)."""

    kb_id: str = Field(validation_alias=AliasChoices("kb_id", "id"))
    title: str
    category: str | None = None
    subcategory: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    common_error_codes: list[str] = Field(default_factory=list)
    root_cause: str | None = None
    diagnostic_steps: list[str] = Field(default_factory=list)
    resolution: KBResolution
    escalation_criteria: list[str] = Field(default_factory=list)
    related_kb_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    priority_hint: str | None = None
    last_updated: str | None = None
    # governance (added at ingest by P1)
    status: KBStatus = "approved"
    version: int = 1
    approved_by: str | None = None
    source: str = "seed"

    @field_validator("escalation_criteria", mode="before")
    @classmethod
    def _criteria_as_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v


# =============================================================================================
# Store / search hits
# =============================================================================================
class TicketHit(_Open):
    """One ticket returned by ``TicketStore.query`` or ``search_tickets`` (§6.6, §7.5)."""

    ticket_id: str
    score: float  # cosine for dense hits, fused score after RRF
    title: str | None = None
    excerpt: str | None = None
    category: str | None = None
    subcategory: str | None = None
    priority: str | None = None
    status: str | None = None
    resolution: str | None = None
    resolution_origin: ResolutionOrigin = "none"
    collection: str | None = None
    tier: DupTier | None = None
    why: str | None = None
    keyword_only: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkHit(_Open):
    """One KEDB chunk returned by ``KBStore.query_chunks`` (§6.4, §6.6)."""

    chunk_id: str  # e.g. "KB-0018#diag.2"
    kb_id: str
    chunk_type: ChunkType
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBHit(_Open):
    """One ranked KEDB article (§7.6)."""

    kb_id: str
    title: str
    score: float
    matched_field: str | None = None
    excerpt: str | None = None
    why: str | None = None
    matched_error_code: str | None = None
    requires_admin: bool = False
    category: str | None = None
    subcategory: str | None = None


# =============================================================================================
# Output contract (architecture §4.1) — exact fields, no extras
# =============================================================================================
class Classification(_Strict):
    category: str
    subcategory: str
    confidence: float = Field(ge=0, le=1)


class Priority(_Strict):
    level: PriorityLevel
    reason: str


class SimilarIncident(_Strict):
    incident_id: str
    similarity_score: float = Field(ge=0, le=1)
    resolution: str | None


class KedbMatch(_Strict):
    found: bool
    error_code: str | None  # KEDB known-error ID, e.g. "KB-0018"
    solution: str | None


class RecommendedSolution(_Strict):
    steps: list[str]
    source: Source


class AnalysisContract(_Strict):
    """``view=contract``: exactly the 8 contract keys."""

    ticket_id: str
    classification: Classification
    priority: Priority
    similar_incidents: list[SimilarIncident]
    kedb_match: KedbMatch
    recommended_solution: RecommendedSolution
    escalation_required: bool
    human_approval_required: bool


# =============================================================================================
# ``details`` extension (architecture §4.3)
# =============================================================================================
class Alternative(_Open):
    category: str
    subcategory: str
    score: float


class ClassificationDetails(_Open):
    band: Band
    components: dict[str, float] = Field(default_factory=dict)  # llm, knn, rule, margin, agree
    alternatives: list[Alternative] = Field(default_factory=list)
    rationale: str | None = None
    conflict: bool = False


class PriorityDetails(_Open):
    code: PriorityCode
    impact: int = Field(ge=1, le=3)
    urgency: int = Field(ge=1, le=3)
    drivers: list[str] = Field(default_factory=list)


class SimilarDetails(_Open):
    incident_id: str
    tier: DupTier
    excerpt: str | None = None
    why: str | None = None
    resolution_origin: ResolutionOrigin
    category: str | None = None
    subcategory: str | None = None


class KBAlternative(_Open):
    kb_id: str
    score: float


class KedbDetails(_Open):
    kb_id: str
    title: str
    score: float
    matched_error_code: str | None = None
    matched_field: str | None = None
    excerpt: str | None = None
    why: str | None = None
    alternatives: list[KBAlternative] = Field(default_factory=list)


class Step(_Open):
    """One recommended step with its evidence (§4.3 ``details.steps``, §7.7)."""

    text: str
    type: StepType
    citations: list[str] = Field(min_length=1)  # "KB-0018#res.1" or a verified incident ID
    role: StepRole = "L1"
    already_attempted: bool = False


class EscalationReason(_Open):
    rule: EscalationRule
    text: str


class Escalation(_Open):
    reasons: list[EscalationReason]
    assignment_group: str
    tier: Tier
    bundle_id: str | None = None
    issue_idx: int = 0  # which issue triggered it (0 = primary)


class SecondaryIssue(_Open):
    """A second issue split out of a multi-issue ticket (§7.2)."""

    classification: Classification
    priority: Priority
    kedb_match: KedbMatch
    recommended_solution: RecommendedSolution
    decision: DecisionKind
    escalation: Escalation | None = None
    # evidence for the secondary issue, same shape as the primary's details
    classification_details: ClassificationDetails | None = None
    priority_details: PriorityDetails | None = None
    kedb: KedbDetails | None = None
    steps: list[Step] = Field(default_factory=list)


class RedactionSummary(_Open):
    count: int = 0
    types: list[str] = Field(default_factory=list)


class Signals(_Open):
    """Deterministic signals (§7.1)."""

    error_codes: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    affected_users: int | None = None
    scope: Scope = "single"
    urgency_terms: list[str] = Field(default_factory=list)
    business_critical: bool = False
    environment: str | None = None
    location: str | None = None
    onset: str | None = None
    change_trigger: str | None = None
    multi_issue_markers: bool = False
    security_flags: list[str] = Field(default_factory=list)
    tried_steps: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class Details(_Open):
    analysis_id: str
    decision: DecisionKind  # ticket-level: escalate > clarify > recommend (§8.4)
    primary_decision: DecisionKind | None = None  # decision for issue 0 only
    clarifying_questions: list[str] = Field(default_factory=list)
    intent: Intent = "incident"
    classification: ClassificationDetails
    priority: PriorityDetails
    similar: list[SimilarDetails] = Field(default_factory=list)
    keyword_only: list[SimilarDetails] = Field(default_factory=list)
    kedb: KedbDetails | None = None
    steps: list[Step] = Field(default_factory=list)
    escalation: Escalation | None = None
    approval_reasons: list[str] = Field(default_factory=list)
    secondary_issues: list[SecondaryIssue] = Field(default_factory=list)
    signals: Signals | None = None
    redactions: RedactionSummary = Field(default_factory=RedactionSummary)
    degraded: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)


class AnalysisResponse(AnalysisContract):
    """``view=full``: the 8 contract keys plus ``details``."""

    details: Details

    def to_contract(self) -> AnalysisContract:
        return AnalysisContract.model_validate(self.model_dump(exclude={"details"}))


# =============================================================================================
# Engine I/O (architecture §7)
# =============================================================================================
class ClassificationResult(_Open):
    category: str
    subcategory: str
    confidence: float = Field(ge=0, le=1)
    band: Band
    intent: Intent = "incident"
    components: dict[str, float] = Field(default_factory=dict)
    alternatives: list[Alternative] = Field(default_factory=list)
    rationale: str | None = None
    conflict: bool = False


class PriorityResult(_Open):
    code: PriorityCode
    level: PriorityLevel
    reason: str
    impact: int = Field(ge=1, le=3)
    urgency: int = Field(ge=1, le=3)
    drivers: list[str] = Field(default_factory=list)


class Recommendation(_Open):
    """Validated output of synthesise + validate (§7.7, §7.8)."""

    steps: list[Step] = Field(default_factory=list)
    source: Source = "None"
    insufficient_evidence: bool = False
    summary: str | None = None
    probable_cause: str | None = None
    estimated_time: str | None = None
    requires_admin: bool = False
    unsupported_steps_removed: int = 0


class IssueResult(_Open):
    text_span: str
    classification: ClassificationResult
    priority: PriorityResult
    similar: list[TicketHit] = Field(default_factory=list)
    keyword_only: list[TicketHit] = Field(default_factory=list)
    kb_matches: list[KBHit] = Field(default_factory=list)
    kb_relevance: float = 0.0
    kb_ambiguous: bool = False
    possible_major_incident: bool = False
    recommendation: Recommendation = Field(default_factory=Recommendation)


class EngineResult(_Open):
    signals: Signals
    issues: list[IssueResult] = Field(min_length=1)
    degraded: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)


@dataclass
class EngineContext:
    """Everything the engine needs, injected by P3 (§7). A dataclass because it holds stores."""

    role: Role
    ticket_store: TicketStore
    kb_store: KBStore
    llm: LLMClient
    feedback_prior: dict[str, float] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    clarification: str | None = None
    clarification_round: int = 0


# =============================================================================================
# Feedback, decisions, events, users (§8.6, §8.8, §8.10, §10)
# =============================================================================================
class Feedback(_Strict):
    analysis_id: str
    issue_idx: int = 0
    target_type: FeedbackTarget
    target_ref: str | None = None
    verdict: FeedbackVerdict
    corrected_value: str | None = None
    comment: str | None = None


class Decision(_Strict):
    analysis_id: str
    final_decision: FinalDecision
    final_category: str | None = None
    final_subcategory: str | None = None
    final_priority: PriorityLevel | None = None
    kb_or_incident_used: str | None = None
    resolution_notes: str | None = None
    mode: Mode = "assisted"


class Event(_Strict):
    ticket_id: str
    analysis_id: str | None = None
    event: str
    mode: Mode = "assisted"


class User(_Strict):
    username: str
    role: Role
    assignment_group: str | None = None
