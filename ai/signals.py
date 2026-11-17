"""Deterministic signal extractor (architecture §7.1). Owner: P2.

Pure function of the (already redacted) ticket text, the KEDB phrase codes and
``config/signals.yaml``. No LLM, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai.config import load_yaml
from shared.interfaces import KBStore
from shared.schemas import SanitizedTicket, Signals

_STOP_WORDS = """
    a about after again all already also am an and any are as at be been but by can cannot
    could did do does doing for from get gets got had has have having he her his how i if
    in into is it its just keep keeps me more most my need needs no not now of on once
    only or other our out over please same she since so some still than that the their
    them then there these they this those to too twice under until up us very was we were
    what when where which while who why will with would you your yes every few even each
    like one two being able time times
"""
_STOP = frozenset(_STOP_WORDS.split())
_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")
_NEGATION = re.compile(r"\b(?:not|never|n't|haven't|hasn't|didn't|no)\s*$", re.IGNORECASE)
_TRIED_PREFIX = re.compile(r"^(?:(?:have|had|already)\s+)+", re.IGNORECASE)


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    value: str


@lru_cache(maxsize=4)
def signals_config(config_dir: Path | None = None) -> dict[str, Any]:
    return load_yaml("signals.yaml", config_dir)


def _flags(spec: dict[str, Any] | str) -> int:
    if isinstance(spec, dict) and spec.get("case_sensitive"):
        return 0
    return re.IGNORECASE


def _pattern(spec: dict[str, Any] | str) -> str:
    return spec["pattern"] if isinstance(spec, dict) else spec


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    """Match a KEDB phrase code case-insensitively, tolerant to whitespace, on word edges."""
    parts = [re.escape(p) for p in phrase.split()]
    body = r"\s+".join(parts)
    left = r"\b" if phrase[:1].isalnum() else ""
    right = r"\b" if phrase[-1:].isalnum() else ""
    return re.compile(left + body + right, re.IGNORECASE)


def _drop_contained(matches: list[_Match], preferred: list[_Match] | None = None) -> list[_Match]:
    """Remove matches whose span sits inside another kept match (``preferred`` wins ties)."""
    pref = preferred or []
    out: list[_Match] = []
    for m in sorted(matches, key=lambda m: (m.start, -(m.end - m.start), m not in pref)):
        if any(o.start <= m.start and m.end <= o.end for o in out):
            continue
        out.append(m)
    seen: set[str] = set()
    uniq = []
    for m in out:
        key = m.value.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


def extract_error_codes(text: str, phrases: list[str], cfg: dict[str, Any]) -> list[str]:
    """Regex families first (in text order), then KEDB phrase codes (in text order)."""
    family_hits: list[_Match] = []
    for spec in (cfg.get("error_code_families") or {}).values():
        for m in re.finditer(_pattern(spec), text, _flags(spec)):
            value = m.groupdict().get("code") or m.group(0)
            family_hits.append(_Match(m.start(), m.end(), _norm(value)))
    phrase_hits: list[_Match] = []
    for phrase in phrases:
        if not phrase.strip():
            continue
        for m in _phrase_regex(phrase).finditer(text):
            phrase_hits.append(_Match(m.start(), m.end(), phrase))
            break
    # A phrase covering a family match wins (e.g. "Error 809" vs "809", same span → phrase);
    # a family match strictly covering a phrase wins (e.g. "SQL Server error 18456" vs "18456").
    kept = _drop_contained(family_hits + phrase_hits, preferred=phrase_hits)
    fam = [m for m in kept if m in family_hits]
    phr = [m for m in kept if m not in family_hits]
    values = [m.value for m in sorted(fam, key=lambda m: m.start)]
    values += [m.value for m in sorted(phr, key=lambda m: m.start)]
    seen: set[str] = set()
    out = []
    for v in values:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _users_and_scope(text: str, cfg: dict[str, Any]) -> tuple[int | None, str]:
    ucfg = cfg.get("users") or {}
    users: int | None = None
    for pat in ucfg.get("count_patterns", []):
        for m in re.finditer(pat, text, re.IGNORECASE):
            n = int(m.group("n"))
            if n > 0 and (users is None or n > users):
                users = n
        if users is not None:
            break
    scope = "single"
    for name in ("org", "site", "team", "multiple"):
        pat = (ucfg.get("scope_patterns") or {}).get(name)
        if pat and re.search(pat, text, re.IGNORECASE):
            scope = name
            break
    if users is not None:
        by_count = "team" if users >= 10 else "multiple" if users >= 2 else "single"
        order = ["single", "multiple", "team", "site", "org"]
        scope = max(scope, by_count, key=order.index)
    return users, scope


def _urgency_terms(text: str, cfg: dict[str, Any]) -> list[str]:
    out = []
    for item in cfg.get("urgency") or []:
        if re.search(item["pattern"], text, re.IGNORECASE):
            out.append(item["label"])
    return out


def urgency_levels(cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """``{urgency label: level}`` from ``config/signals.yaml`` (used by ``ai/priority.py``)."""
    cfg = cfg if cfg is not None else signals_config()
    return {u["label"]: int(u.get("level", 2)) for u in cfg.get("urgency") or []}


def _first(pattern: str | None, text: str, group: str | None = None) -> str | None:
    if not pattern:
        return None
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    val = m.groupdict().get(group) if group else None
    return _norm(val or m.group(0))


def _location(text: str, cfg: dict[str, Any]) -> str | None:
    lcfg = cfg.get("location") or {}
    parts = []
    building = _first(lcfg.get("building"), text)
    if building:
        parts.append(building[0].upper() + building[1:])
    floor = _first(lcfg.get("floor"), text)
    if floor:
        parts.append(floor)
    if not parts:
        site = _first(lcfg.get("site"), text, "site")
        if site:
            parts.append(site)
    return ", ".join(parts) or None


def _environment(text: str, cfg: dict[str, Any]) -> str | None:
    for name, pat in (cfg.get("environment") or {}).items():
        if re.search(pat, text, re.IGNORECASE):
            return name
    return None


def _tried_steps(text: str, cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for pat in cfg.get("tried_steps") or []:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if _NEGATION.search(text[max(0, m.start() - 12) : m.start()]):
                continue
            step = _TRIED_PREFIX.sub("", _norm(m.group(0)))
            if step and step.lower() not in {s.lower() for s in out}:
                out.append(step)
    return out


def _security(text: str, cfg: dict[str, Any]) -> list[str]:
    pat = cfg.get("security")
    if not pat:
        return []
    out: list[str] = []
    for m in re.finditer(pat, text, re.IGNORECASE):
        term = m.group(0).lower()
        if term not in out:
            out.append(term)
    return out


def _keywords(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        w = w.strip("-")
        if w in _STOP or w.isdigit() or len(w) < 3:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def extract(
    text: str,
    phrases: list[str] | None = None,
    *,
    environment: str | None = None,
    location: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> Signals:
    """Extract every §7.1 signal from ``text``. Explicit ticket fields override text guesses."""
    cfg = cfg if cfg is not None else signals_config()
    users, scope = _users_and_scope(text, cfg)
    products = [
        name
        for name, pat in (cfg.get("products") or {}).items()
        if re.search(pat, text, re.IGNORECASE)
    ]
    bc = cfg.get("business_critical")
    change = _first(cfg.get("change_trigger"), text, "what")
    return Signals(
        error_codes=extract_error_codes(text, phrases or [], cfg),
        products=products,
        affected_users=users,
        scope=scope,  # type: ignore[arg-type]
        urgency_terms=_urgency_terms(text, cfg),
        business_critical=bool(bc and re.search(bc, text, re.IGNORECASE)),
        environment=environment or _environment(text, cfg),
        location=location or _location(text, cfg),
        onset=_first(cfg.get("onset"), text),
        change_trigger=change,
        multi_issue_markers=bool(
            cfg.get("multi_issue") and re.search(cfg["multi_issue"], text, re.I | re.M)
        ),
        security_flags=_security(text, cfg),
        tried_steps=_tried_steps(text, cfg),
        keywords=_keywords(text, int(cfg.get("keywords_max", 8))),
    )


def ticket_text(ticket: SanitizedTicket) -> str:
    """Title + description (+ clarification), the text every P2 stage works on."""
    parts = [ticket.title.strip(), ticket.description.strip()]
    if ticket.clarification:
        parts.append(f"[Clarification] {ticket.clarification.strip()}")
    return "\n".join(p for p in parts if p)


def extract_signals(
    ticket: SanitizedTicket, kb_store: KBStore | None = None, cfg: dict[str, Any] | None = None
) -> Signals:
    """Signals for a sanitised ticket, using the KEDB phrase codes from ``kb_store``."""
    phrases: list[str] = []
    if kb_store is not None:
        try:
            phrases = list(kb_store.all_error_phrases())
        except Exception:  # noqa: BLE001 - a store hiccup must not stop analysis
            phrases = []
    return extract(
        ticket_text(ticket),
        phrases,
        environment=ticket.environment,
        location=ticket.location,
        cfg=cfg,
    )
