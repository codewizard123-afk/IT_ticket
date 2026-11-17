"""Shared constants: taxonomy, priority scale, roles, enums and default thresholds.

Frozen after Phase 0 (README §8.3). Runtime values come from ``config/*.yaml``; the thresholds
here are only defaults/fallbacks that mirror architecture §17.
"""

from __future__ import annotations

from typing import Final

PIPELINE_VERSION: Final = "2.0.0"
EMBED_MODEL: Final = "BAAI/bge-small-en-v1.5"

# --- Taxonomy (architecture §3.1) -------------------------------------------------------------
TAXONOMY: Final[dict[str, tuple[str, ...]]] = {
    "Network": ("VPN", "Wireless", "DNS", "DHCP", "IP Address"),
    "Database": ("Oracle", "MS SQL Server", "DB2"),
    "Hardware": ("CPU", "Disk", "Keyboard", "Memory", "Monitor", "Mouse"),
    "Software": ("Email", "Operating System"),
    "Inquiry/Help": ("Antivirus", "Internal Application"),
}
CATEGORIES: Final = tuple(TAXONOMY)
SUBCATEGORY_OTHER: Final = "Other"  # allowed when no label scores high enough; forces band "low"
SUBCATEGORY_TO_CATEGORY: Final[dict[str, str]] = {
    sub: cat for cat, subs in TAXONOMY.items() for sub in subs
}

# Baseline priority by category (§3.1). Internal codes P1..P4.
BASELINE_PRIORITY: Final[dict[str, str]] = {
    "Network": "P2",
    "Database": "P2",
    "Hardware": "P3",
    "Software": "P3",
    "Inquiry/Help": "P4",
}

# Assignment groups (§3.1, §8.2 overrides).
ASSIGNMENT_GROUP: Final[dict[str, str]] = {
    "Network": "NET-OPS",
    "Database": "DBA",
    "Hardware": "EUC-HW",
    "Software": "EUC-SW",
    "Inquiry/Help": "SERVICE-DESK",
}
GROUP_SECURITY: Final = "SEC-OPS"
GROUP_NETWORK_MAJOR: Final = "NET-OPS-MAJOR"
ASSIGNMENT_GROUPS: Final = (
    *ASSIGNMENT_GROUP.values(),
    GROUP_SECURITY,
    GROUP_NETWORK_MAJOR,
)

# --- Priority scale (§3.2) --------------------------------------------------------------------
PRIORITY_LEVEL: Final[dict[str, str]] = {
    "P1": "Critical",
    "P2": "High",
    "P3": "Medium",
    "P4": "Low",
}
LEVEL_PRIORITY: Final[dict[str, str]] = {v: k for k, v in PRIORITY_LEVEL.items()}
PRIORITY_CODES: Final = tuple(PRIORITY_LEVEL)
PRIORITY_LEVELS: Final = tuple(PRIORITY_LEVEL.values())

# --- Enums ------------------------------------------------------------------------------------
INTENTS: Final = ("incident", "service_request", "how_to", "access_request")
CONFIDENCE_BANDS: Final = ("high", "medium", "low")
DECISIONS: Final = ("recommend", "clarify", "escalate")
SOURCES: Final = ("KEDB", "Historical Incident", "KEDB + Historical Incident", "None")
STEP_TYPES: Final = ("diagnostic", "resolution")
STEP_ROLES: Final = ("L1", "L2")
ESCALATION_TIERS: Final = ("L2", "L3")
DUPLICATE_TIERS: Final = ("DUPLICATE", "LIKELY DUPLICATE", "SIMILAR", "RELATED")
SCOPES: Final = ("single", "multiple", "team", "site", "org")
RESOLUTION_ORIGINS: Final = (
    "synthetic_kb_derived",
    "synthetic_escalated",
    "curated_verified",
    "analyst_verified",
    "none",
)
# Only these origins may be cited as a "Historical Incident" source (architecture §6.3).
CITABLE_ORIGINS: Final = frozenset({"curated_verified", "analyst_verified"})
KB_STATUSES: Final = ("approved", "draft", "retired")
ESCALATION_RULES: Final = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
HARD_RULES: Final = frozenset({"E4", "E5"})
DEGRADED_MODES: Final = (
    "llm_classify_fallback",
    "llm_split_fallback",
    "llm_synth_fallback",
    "offline",
)

# --- Roles (§8.6) -----------------------------------------------------------------------------
ROLES: Final = ("l1", "l2", "kb_owner", "admin")

# --- Feedback / decisions / events (§8.8, §8.10, §10) -----------------------------------------
FEEDBACK_TARGETS: Final = (
    "classification",
    "similar_incident",
    "kedb_match",
    "step",
    "recommendation",
)
FEEDBACK_VERDICTS: Final = ("useful", "incorrect", "unresolved")
FINAL_DECISIONS: Final = ("resolved", "escalated", "needs_more_info")
MODES: Final = ("assisted", "manual")
EVENT_TYPES: Final = (
    "ticket.submitted",
    "analysis.completed",
    "ui.viewed",
    "clarify.requested",
    "clarify.answered",
    "decision.submitted",
    "ticket.resolved",
    "ticket.reopened",
    "manual.started",
    "manual.decision",
    "access.denied",
)
UI_EVENT_TYPES: Final = ("ui.viewed", "manual.started", "manual.decision")

# --- KB chunking (§6.4) -----------------------------------------------------------------------
CHUNK_TYPES: Final = ("title_root", "symptom", "errors", "diagnostic", "resolution", "escalation")
SEARCHABLE_CHUNK_TYPES: Final = ("title_root", "symptom", "errors", "diagnostic", "escalation")

# --- Chroma collections (§11.1) ---------------------------------------------------------------
TICKET_COLLECTIONS: Final = ("tickets_hist", "tickets_verified", "tickets_live")
KB_COLLECTION: Final = "kedb_chunks"

# --- Output contract (§4.1). These 8 keys never change. ----------------------------------------
CONTRACT_KEYS: Final = (
    "ticket_id",
    "classification",
    "priority",
    "similar_incidents",
    "kedb_match",
    "recommended_solution",
    "escalation_required",
    "human_approval_required",
)
KEDB_SOLUTION_MAX_CHARS: Final = 200
MAX_SIMILAR_INCIDENTS: Final = 5
MAX_ISSUES: Final = 2
MAX_CLARIFYING_QUESTIONS: Final = 3

# --- Default thresholds (§17; real values live in config/ai.yaml and config/backend.yaml) ------
DEFAULT_THRESHOLDS: Final[dict[str, float | int]] = {
    "classifier_w_llm": 0.45,
    "classifier_w_knn": 0.35,
    "classifier_w_rule": 0.20,
    "band_high": 0.75,
    "band_medium": 0.55,
    "knn_k": 15,
    "tau_kb": 0.50,
    "tau_show": 0.35,
    "kb_ambiguity_gap": 0.05,
    "dup_near": 0.92,
    "dup_similar": 0.75,
    "related_min": 0.55,
    "grounding_min": 0.30,
    "min_steps": 2,
    "sparse_words": 12,
    "low_gap": 0.10,
    "max_clarify_rounds": 1,
    "verified_min_cos": 0.75,
}
