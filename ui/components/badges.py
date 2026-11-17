"""Badges and chips: pure functions (testable without Streamlit).

Every chip carries colour **and** a text label (architecture §9.2 accessibility rule), and every
recommendation step gets a source badge built from its citations.
"""

from __future__ import annotations

import html

from ui.settings import load_ui_config

_FALLBACK = {"color": "#616161", "label": None}


def chip_spec(kind: str, value: str | None) -> tuple[str, str]:
    """(label, colour) for a chip of ``kind`` (decision, band, priority, tier, origin, role)."""
    spec = load_ui_config()["badges"].get(kind, {}).get(str(value), _FALLBACK)
    label = spec.get("label") or str(value or "UNKNOWN").upper()
    return label, spec["color"]


def chip(kind: str, value: str | None, prefix: str = "") -> str:
    label, color = chip_spec(kind, value)
    text = html.escape(f"{prefix}{label}")
    return (
        f'<span style="background:{color};color:#fff;border-radius:4px;'
        f'padding:1px 6px;font-size:0.8em;font-weight:600;white-space:nowrap">{text}</span>'
    )


def plain_chip(text: str, color: str = "#455a64") -> str:
    return (
        f'<span style="background:{color};color:#fff;border-radius:4px;'
        f'padding:1px 6px;font-size:0.8em;white-space:nowrap">{html.escape(text)}</span>'
    )


def citation_label(citation: str) -> str:
    """``KB-0018#res.1`` -> ``KB-0018·res.1``; incident IDs stay as they are."""
    kb, _, part = citation.partition("#")
    return f"{kb}·{part}" if part else kb


def source_badges(citations: list[str]) -> str:
    """``[KB-0018·res.1] [INC-H0007]``. Raises if there is nothing to cite (§9.2 rule)."""
    if not citations:
        raise ValueError("a step without a citation must never be rendered")
    return " ".join(f"[{citation_label(c)}]" for c in citations)


def origin_is_synthetic(origin: str | None) -> bool:
    return bool(origin) and str(origin).startswith("synthetic")


def priority_code_to_level(code: str | None) -> str | None:
    return {"P1": "Critical", "P2": "High", "P3": "Medium", "P4": "Low"}.get(code or "", code)
