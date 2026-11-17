"""KEDB retrieval (architecture §7.6, FR-03/FR-04).

    kb_score = 0.45·sem + 0.15·bm25 + 0.20·code_hit + 0.12·tax + 0.08·fb
    sem      = max chunk cosine per article (best chunk = matched excerpt)
    bm25     = best chunk BM25 per article, normalised by the best BM25 of the query
    code_hit = 1 if any ticket error code maps to the article (error-code index)
    tax      = 1.0 primary / 0.5 secondary mapping (`config/taxonomy_map.yaml`, P1)
    fb       = feedback_prior[kb_id] clipped to [-0.5, 0.5] (passed in by P3)

τ_show = 0.35 (display) · τ_kb = 0.50 (`kedb_match.found`) ·
ambiguity: top1 − top2 < 0.05 and different subcategory → `ambiguous`.
Resolution chunks are stored for citation only and are never searched.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any

from ai.config import ai_config, signals_config, taxonomy_map
from ai.signals import extract_error_codes
from ai.text import truncate
from ai.types import ChunkHit, KBMatch, KBSearchResult, KBStore

SEARCHABLE_WHERE: dict[str, Any] = {
    "$and": [{"chunk_type": {"$ne": "resolution"}}, {"status": "approved"}]
}

_FIELD_NAMES = {
    "title_root": "title",
    "symptom": "symptoms",
    "errors": "errors",
    "diagnostic": "diagnostics",
    "escalation": "escalation",
}


@cache
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# --------------------------------------------------------------------------- taxonomy map


def taxonomy_boosts(
    category: str | None,
    subcategory: str | None,
    text: str,
    tmap: dict[str, Any] | None = None,
) -> dict[str, float]:
    """kb_id → 1.0 (primary) / 0.5 (secondary) for the predicted label.

    Accepts P1's map either as `mappings: [{category, subcategory, primary, secondary}]` or as
    `{"<Category>/<Subcategory>": {primary, secondary}}`. Within Database the primary article
    is chosen by symptom (`database_symptoms`); the other primaries drop to 0.5.
    """
    tmap = tmap if tmap is not None else taxonomy_map()
    if not category or not subcategory or not tmap:
        return {}
    entry: dict[str, Any] | None = None
    for m in tmap.get("mappings", []) or []:
        if m.get("category") == category and m.get("subcategory") == subcategory:
            entry = m
            break
    if entry is None:
        entry = tmap.get(f"{category}/{subcategory}")
    if not entry:
        return {}
    boosts: dict[str, float] = {}
    for kid in entry.get("secondary", []) or []:
        boosts[kid] = 0.5
    primaries = list(entry.get("primary", []) or [])
    if category == "Database" and tmap.get("database_symptoms"):
        chosen = [
            s["kb_id"]
            for s in tmap["database_symptoms"]
            if _rx(s["pattern"]).search(text)
        ]
        chosen = [k for k in chosen if k in primaries]
        if chosen:
            for kid in primaries:
                boosts[kid] = 1.0 if kid in chosen else 0.5
            return boosts
    for kid in primaries:
        boosts[kid] = 1.0
    return boosts


def code_candidates(code: str) -> list[str]:
    """Lookup keys for one ticket code: the literal code, plus the bare number for
    'SQL Server error 18456' / 'MSI error 1603' style codes."""
    out = [code]
    m = re.search(r"\b(?:error|code)\s+(\d{3,6})\b", code, re.IGNORECASE)
    if m:
        out.append(m.group(1))
    return out


def code_hits(error_codes: list[str], store: KBStore) -> dict[str, str]:
    """kb_id → the literal ticket code that mapped to it (first one wins)."""
    if not error_codes:
        return {}
    keys: dict[str, str] = {}
    for code in error_codes:
        for cand in code_candidates(code):
            keys.setdefault(cand, code)
    found = store.lookup_codes(list(keys))
    out: dict[str, str] = {}
    for cand, kb_ids in found.items():
        for kid in kb_ids:
            out.setdefault(kid, keys.get(cand, cand))
    return out


# --------------------------------------------------------------------------- search


def _kb_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def search_articles(
    text: str,
    store: KBStore,
    *,
    error_codes: list[str] | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    feedback_prior: dict[str, float] | None = None,
    tmap: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> KBSearchResult:
    cfg = cfg if cfg is not None else ai_config()
    kcfg = cfg.get("retrieval", {}).get("kb", {})
    w = cfg.get("retrieval", {}).get(
        "weights_kb", {"sem": 0.45, "bm25": 0.15, "code": 0.20, "tax": 0.12, "fb": 0.08}
    )
    k_chunks = int(kcfg.get("k_chunks", 20))
    k_bm25 = int(kcfg.get("k_bm25", k_chunks))
    top_n = int(kcfg.get("top_n", 3))
    tau_show = float(kcfg.get("tau_show", 0.35))
    tau_kb = float(kcfg.get("tau_kb", 0.50))
    gap = float(kcfg.get("ambiguity_gap", 0.05))

    # semantic: best chunk per article
    best_chunk: dict[str, ChunkHit] = {}
    for ch in store.query_chunks(text, k_chunks, where=SEARCHABLE_WHERE):
        if ch.chunk_type == "resolution":
            continue
        if ch.kb_id not in best_chunk or ch.score > best_chunk[ch.kb_id].score:
            best_chunk[ch.kb_id] = ch

    # BM25: best chunk per article, normalised
    bm25_raw: dict[str, float] = {}
    for cid, s in store.bm25(text, k_bm25):
        if "#res." in cid:
            continue
        kid = _kb_of(cid)
        bm25_raw[kid] = max(bm25_raw.get(kid, 0.0), s)
    bmax = max(bm25_raw.values(), default=0.0)
    bm25 = {k: v / bmax for k, v in bm25_raw.items()} if bmax > 0 else {}

    codes = code_hits(error_codes or [], store)
    tax = taxonomy_boosts(category, subcategory, text, tmap)
    fb_prior = feedback_prior or {}

    candidates = set(best_chunk) | set(bm25) | set(codes) | set(tax)
    scored: list[KBMatch] = []
    for kid in candidates:
        try:
            art = store.get_article(kid)
        except KeyError:
            continue  # mapped in the taxonomy but not in the KEDB (or retired)
        if art.status != "approved":
            continue
        sem = best_chunk[kid].score if kid in best_chunk else 0.0
        b = bm25.get(kid, 0.0)
        c = 1.0 if kid in codes else 0.0
        t = tax.get(kid, 0.0)
        f = max(-0.5, min(0.5, float(fb_prior.get(kid, 0.0))))
        score = (
            w["sem"] * sem + w["bm25"] * b + w["code"] * c + w["tax"] * t + w["fb"] * f
        )
        score = max(0.0, min(1.0, score))

        if c:
            matched_field, excerpt = "errors", codes[kid]
        elif kid in best_chunk:
            ch = best_chunk[kid]
            matched_field, excerpt = (
                _FIELD_NAMES.get(ch.chunk_type, ch.chunk_type),
                ch.text,
            )
        else:
            matched_field, excerpt = "title", art.title

        why_parts = []
        if c:
            why_parts.append("exact error code")
        if sem and kid in best_chunk:
            why_parts.append(
                f"{_FIELD_NAMES.get(best_chunk[kid].chunk_type, 'text')} match {sem:.2f}"
            )
        if t:
            why_parts.append(
                f"{subcategory} mapping"
                + (" (primary)" if t >= 1.0 else " (secondary)")
            )
        if f > 0:
            why_parts.append("analyst feedback +")
        elif f < 0:
            why_parts.append("analyst feedback -")
        scored.append(
            KBMatch(
                kb_id=kid,
                title=art.title,
                score=round(score, 4),
                matched_field=matched_field,
                excerpt=truncate(excerpt, 200),
                why=" + ".join(why_parts) or "keyword match",
                matched_error_code=codes.get(kid),
                components={
                    "sem": round(sem, 3),
                    "bm25": round(b, 3),
                    "code": c,
                    "tax": t,
                    "fb": round(f, 3),
                },
                subcategory=art.subcategory,
            )
        )

    scored.sort(key=lambda m: (-m.score, m.kb_id))
    relevance = scored[0].score if scored else 0.0
    ambiguous = (
        len(scored) >= 2
        and scored[0].score - scored[1].score < gap
        and scored[0].subcategory != scored[1].subcategory
        and scored[1].score >= tau_show
    )
    shown = [m for m in scored if m.score >= tau_show][:top_n]
    return KBSearchResult(
        matches=shown,
        relevance=round(relevance, 4),
        found=relevance >= tau_kb,
        ambiguous=ambiguous,
    )


def kedb_contract(
    result: KBSearchResult, store: KBStore, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Contract `kedb_match` object: `{found, error_code (KB id), solution}`."""
    cfg = cfg if cfg is not None else ai_config()
    n = int(cfg.get("retrieval", {}).get("kb", {}).get("solution_chars", 200))
    if not result.found or not result.matches:
        return {"found": False, "error_code": None, "solution": None}
    top = result.matches[0]
    art = store.get_article(top.kb_id)
    first = art.resolution.steps[0] if art.resolution.steps else None
    return {
        "found": True,
        "error_code": top.kb_id,
        "solution": truncate(first, n) if first else None,
    }


def kedb_details(result: KBSearchResult) -> dict[str, Any] | None:
    """`details.kedb` block (architecture §4.3)."""
    if not result.matches:
        return None
    top = result.matches[0]
    return {
        "kb_id": top.kb_id,
        "title": top.title,
        "score": round(top.score, 2),
        "matched_error_code": top.matched_error_code,
        "matched_field": top.matched_field,
        "excerpt": top.excerpt,
        "why": top.why,
        "alternatives": [
            {"kb_id": m.kb_id, "score": round(m.score, 2)} for m in result.matches[1:]
        ],
        "ambiguous": result.ambiguous,
        "relevance": round(result.relevance, 2),
    }


def search_kb(
    q: str,
    filters: dict[str, Any] | None,
    k: int,
    store: KBStore,
    *,
    cfg: dict[str, Any] | None = None,
) -> list[KBMatch]:
    """FR-07 KB search: free text plus optional `category`/`subcategory` (ticket taxonomy)
    filters, which act as the taxonomy boost. Error codes in `q` are matched exactly."""
    cfg = cfg if cfg is not None else ai_config()
    codes, _ = extract_error_codes(q, store.all_error_phrases(), signals_config())
    filters = filters or {}
    res = search_articles(
        q,
        store,
        error_codes=codes,
        category=filters.get("category"),
        subcategory=filters.get("subcategory"),
        cfg=cfg,
    )
    return res.matches[:k]
