"""KEDB retrieval (architecture §7.6, FR-03, FR-04). Owner: P2.

```
kb_score = 0.45·sem + 0.15·bm25 + 0.20·code_hit + 0.12·tax + 0.08·fb
sem      = max chunk similarity per article (best chunk = matched excerpt)
code_hit = 1 if any ticket error code maps to the article
tax      = 1.0 primary / 0.5 secondary mapping (config/taxonomy_map.yaml, soft boost)
fb       = feedback_prior[kb_id] ∈ [-0.5, 0.5] (passed in by P3)
τ_show = 0.35 · τ_kb = 0.50 · ambiguity: top1 − top2 < 0.05 and different subcategory
```
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai.config import ai_config, load_yaml
from ai.signals import extract_error_codes, signals_config
from shared.interfaces import KBStore
from shared.schemas import ChunkHit, ClassificationResult, KBArticle, KBHit, Signals

log = logging.getLogger(__name__)

_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")
_FIELD_NAME = {
    "title_root": "title",
    "symptom": "symptom",
    "errors": "errors",
    "diagnostic": "diagnostic",
    "escalation": "escalation",
    "resolution": "resolution",
}


# --- taxonomy map ------------------------------------------------------------------------------
def _norm_map(raw: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Accept ``{"Cat/Sub": {...}}`` or ``{"Cat": {"Sub": {...}}}`` (P1's file may be nested)."""
    out: dict[str, dict[str, list[str]]] = {}
    for key, val in (raw or {}).items():
        if not isinstance(val, dict):
            continue
        if "primary" in val or "secondary" in val:
            out[key] = {
                "primary": list(val.get("primary") or []),
                "secondary": list(val.get("secondary") or []),
            }
        else:
            for sub, inner in val.items():
                if isinstance(inner, dict):
                    out[f"{key}/{sub}"] = {
                        "primary": list(inner.get("primary") or []),
                        "secondary": list(inner.get("secondary") or []),
                    }
    return out


@lru_cache(maxsize=4)
def taxonomy_map(config_dir: Path | None = None) -> dict[str, dict[str, list[str]]]:
    """P1's ``config/taxonomy_map.yaml`` if present, else the fallback copy in ``ai.yaml``."""
    p1 = load_yaml("taxonomy_map.yaml", config_dir)
    mapping = _norm_map(p1.get("mapping", p1)) if p1 else {}
    return mapping or _norm_map(ai_config(config_dir).get("taxonomy_map_fallback") or {})


def tax_boost(
    classification: ClassificationResult | None, text: str, cfg: dict[str, Any]
) -> dict[str, float]:
    """``{kb_id: 1.0 | 0.5}`` for the predicted subcategory (Database: chosen by symptom)."""
    if classification is None:
        return {}
    entry = taxonomy_map().get(f"{classification.category}/{classification.subcategory}")
    if not entry:
        return {}
    boost = {kb: 0.5 for kb in entry["secondary"]}
    primary = entry["primary"]
    if classification.category == "Database":
        symptoms = cfg.get("database_symptoms") or {}
        hit = [kb for kb in primary if symptoms.get(kb) and re.search(symptoms[kb], text, re.I)]
        for kb in primary:
            boost[kb] = 1.0 if (kb in hit or not hit) else 0.5
    else:
        boost.update({kb: 1.0 for kb in primary})
    return boost


# --- retrieval ---------------------------------------------------------------------------------
def _queries(text: str, cfg: dict[str, Any]) -> list[str]:
    mq = cfg.get("kb_multi_query") or {}
    sentences = [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= int(mq.get("min_words", 3))]
    out = [text]
    for s in sentences[: int(mq.get("max_sentences", 6))]:
        if s not in out:
            out.append(s)
    return out


def _best_chunks(
    text: str, kb_store: KBStore, where: dict[str, Any] | None, cfg: dict[str, Any]
) -> dict[str, ChunkHit]:
    """Best chunk per article over the full text and each sentence (multi-query)."""
    k = int(cfg["retrieval"]["kb"]["k_chunks"])
    best: dict[str, ChunkHit] = {}
    for q in _queries(text, cfg):
        try:
            hits = kb_store.query_chunks(q, k, where)
        except Exception:  # noqa: BLE001 - store failure → no semantic evidence
            log.warning("KB chunk query failed")
            continue
        for h in hits:
            if h.metadata.get("status", "approved") != "approved":
                continue
            cur = best.get(h.kb_id)
            if cur is None or h.score > cur.score:
                best[h.kb_id] = h
    return best


def _bm25(text: str, kb_store: KBStore, cfg: dict[str, Any]) -> dict[str, float]:
    k = int(cfg["retrieval"]["kb"]["k_chunks"])
    try:
        raw = kb_store.bm25(text, k)
    except Exception:  # noqa: BLE001
        log.warning("KB BM25 failed")
        return {}
    top = max((s for _, s in raw), default=0.0)
    out: dict[str, float] = {}
    if top <= 0:
        return out
    for chunk_id, score in raw:
        kb_id = chunk_id.split("#", 1)[0]
        out[kb_id] = max(out.get(kb_id, 0.0), score / top)
    return out


def _code_hits(codes: list[str], kb_store: KBStore) -> dict[str, str]:
    """``{kb_id: first ticket code that maps to it}``."""
    if not codes:
        return {}
    try:
        lookup = kb_store.lookup_codes(codes)
    except Exception:  # noqa: BLE001
        log.warning("KB code lookup failed")
        return {}
    out: dict[str, str] = {}
    for code in codes:
        for kb_id in lookup.get(code, []):
            out.setdefault(kb_id, code)
    return out


@dataclass
class KBSearchResult:
    hits: list[KBHit] = field(default_factory=list)  # top_n with score ≥ τ_show
    relevance: float = 0.0  # best kb_score (contract kedb_match.found = relevance ≥ τ_kb)
    ambiguous: bool = False
    ranked: list[KBHit] = field(default_factory=list)  # every scored article (for evaluation)


def _why(
    code: str | None,
    field_name: str | None,
    tax: float,
    fb: float,
    classification: ClassificationResult | None,
) -> str:
    parts = []
    if code:
        parts.append(f"exact error code '{code}'")
    if field_name and field_name != "errors":
        parts.append(f"{field_name} match")
    if tax >= 1.0 and classification is not None:
        parts.append(f"{classification.subcategory} mapping")
    elif tax > 0 and classification is not None:
        parts.append(f"secondary {classification.subcategory} mapping")
    if fb > 0:
        parts.append("analysts marked it useful")
    elif fb < 0:
        parts.append("analysts marked it incorrect")
    return " + ".join(parts) or "semantic match"


def rank_articles(
    text: str,
    error_codes: list[str],
    kb_store: KBStore,
    *,
    classification: ClassificationResult | None = None,
    feedback_prior: dict[str, float] | None = None,
    where: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> KBSearchResult:
    cfg = cfg or ai_config()
    kcfg = cfg["retrieval"]["kb"]
    w = cfg["retrieval"]["weights_kb"]
    fbp = feedback_prior or {}

    chunks = _best_chunks(text, kb_store, where, cfg)
    bm = _bm25(text, kb_store, cfg)
    codes = _code_hits(error_codes, kb_store)
    tax = tax_boost(classification, text, cfg)
    kb_ids = set(chunks) | set(bm) | set(codes) | {k for k, v in tax.items() if v > 0}
    if where:
        kb_ids &= set(chunks) | set(codes)

    scored: list[KBHit] = []
    for kb_id in kb_ids:
        try:
            art: KBArticle = kb_store.get_article(kb_id)
        except Exception:  # noqa: BLE001 - unknown/retired article
            continue
        if art.status != "approved":
            continue
        chunk = chunks.get(kb_id)
        sem = max(0.0, min(1.0, chunk.score)) if chunk else 0.0
        fb = max(-0.5, min(0.5, float(fbp.get(kb_id, 0.0))))
        t = tax.get(kb_id, 0.0)
        code = codes.get(kb_id)
        score = (
            float(w["sem"]) * sem
            + float(w["bm25"]) * bm.get(kb_id, 0.0)
            + float(w["code"]) * (1.0 if code else 0.0)
            + float(w["tax"]) * t
            + float(w["fb"]) * fb
        )
        if code:
            field_name, excerpt = "errors", code
        elif chunk:
            field_name, excerpt = _FIELD_NAME.get(chunk.chunk_type), chunk.text
        else:
            field_name, excerpt = None, None
        scored.append(
            KBHit(
                kb_id=kb_id,
                title=art.title,
                score=round(max(0.0, min(1.0, score)), 4),
                matched_field=field_name,
                excerpt=excerpt[:300] if excerpt else None,
                why=_why(code, field_name, t, fb, classification),
                matched_error_code=code,
                requires_admin=art.resolution.requires_admin,
                category=art.category,
                subcategory=art.subcategory,
                chunk_id=chunk.chunk_id if chunk else None,
            )
        )
    scored.sort(key=lambda h: (-h.score, h.kb_id))
    relevance = scored[0].score if scored else 0.0
    ambiguous = bool(
        len(scored) >= 2
        and scored[0].score - scored[1].score < float(kcfg["ambiguity_gap"])
        and scored[0].subcategory != scored[1].subcategory
        and scored[1].score >= float(kcfg["tau_show"])
    )
    shown = [h for h in scored if h.score >= float(kcfg["tau_show"])][: int(kcfg["top_n"])]
    return KBSearchResult(hits=shown, relevance=relevance, ambiguous=ambiguous, ranked=scored)


def match_kb(
    text: str,
    signals: Signals,
    classification: ClassificationResult | None,
    kb_store: KBStore,
    feedback_prior: dict[str, float] | None = None,
    cfg: dict[str, Any] | None = None,
) -> KBSearchResult:
    """KEDB matches for one issue (contract ``kedb_match``, ``details.kedb``)."""
    return rank_articles(
        text,
        list(signals.error_codes),
        kb_store,
        classification=classification,
        feedback_prior=feedback_prior,
        cfg=cfg,
    )


def search_kb(
    q: str,
    filters: dict[str, Any] | None,
    k: int,
    kb_store: KBStore,
    feedback_prior: dict[str, float] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[KBHit]:
    """Free-text KEDB search for ``/kb/search`` (no taxonomy boost; codes taken from ``q``)."""
    try:
        phrases = list(kb_store.all_error_phrases())
    except Exception:  # noqa: BLE001
        phrases = []
    codes = extract_error_codes(q, phrases, signals_config())
    where = {k2: v for k2, v in (filters or {}).items() if v not in (None, "", [])} or None
    res = rank_articles(q, codes, kb_store, feedback_prior=feedback_prior, where=where, cfg=cfg)
    return res.ranked[:k]
