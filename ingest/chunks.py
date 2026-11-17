"""KEDB field-level chunking and the error-code index (architecture §6.4).

| chunk_type   | text                                   | id               |
|--------------|----------------------------------------|------------------|
| title_root   | title + root_cause                     | ``KB-0018#title``  |
| symptom      | each symptom                           | ``KB-0018#sym.n``  |
| errors       | joined error codes                     | ``KB-0018#err``    |
| diagnostic   | each diagnostic step                   | ``KB-0018#diag.n`` |
| resolution   | each resolution step (stored, not searched) | ``KB-0018#res.n`` |
| escalation   | escalation criteria                    | ``KB-0018#esc``    |

Numbering is 1-based so ``KB-0018#res.1`` is the first resolution step.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from shared.constants import SEARCHABLE_CHUNK_TYPES
from shared.schemas import KBArticle
from store.text import norm_code


def _meta(art: KBArticle, chunk_type: str, n: int) -> dict[str, Any]:
    return {
        "kb_id": art.kb_id,
        "chunk_type": chunk_type,
        "n": n,
        "title": art.title,
        "category": art.category or "",
        "subcategory": art.subcategory or "",
        "priority_hint": art.priority_hint or "",
        "requires_admin": bool(art.resolution.requires_admin),
        "status": art.status,
        "version": int(art.version),
        "source": art.source,
        "tags": ",".join(art.tags),
        "searchable": chunk_type in SEARCHABLE_CHUNK_TYPES,
    }


def build_chunks(art: KBArticle) -> list[dict[str, Any]]:
    """All chunks of one article as ``{"id", "kb_id", "chunk_type", "text", "metadata"}``."""
    out: list[dict[str, Any]] = []

    def add(cid: str, ctype: str, text: str, n: int = 0) -> None:
        text = " ".join(text.split())
        if text:
            out.append(
                {
                    "id": f"{art.kb_id}#{cid}",
                    "kb_id": art.kb_id,
                    "chunk_type": ctype,
                    "text": text,
                    "metadata": _meta(art, ctype, n),
                }
            )

    add("title", "title_root", f"{art.title}. {art.root_cause or ''}")
    for i, s in enumerate(art.symptoms, 1):
        add(f"sym.{i}", "symptom", s, i)
    if art.common_error_codes:
        add("err", "errors", "; ".join(art.common_error_codes))
    for i, s in enumerate(art.diagnostic_steps, 1):
        add(f"diag.{i}", "diagnostic", s, i)
    for i, s in enumerate(art.resolution.steps, 1):
        add(f"res.{i}", "resolution", s, i)
    if art.escalation_criteria:
        add("esc", "escalation", " ".join(art.escalation_criteria))
    return out


def build_code_index(articles: Iterable[KBArticle]) -> dict[str, list[str]]:
    """Normalised error code → sorted KB ids (approved articles only)."""
    idx: dict[str, set[str]] = {}
    for art in articles:
        if art.status != "approved":
            continue
        for code in art.common_error_codes:
            n = norm_code(code)
            if n:
                idx.setdefault(n, set()).add(art.kb_id)
    return {k: sorted(v) for k, v in sorted(idx.items())}
