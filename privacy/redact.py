"""PII and secret redaction (architecture §6.2). Used offline (before indexing) and online
(P3 calls it before any LLM or embedding call).

    redact(text, keep_map=True) -> RedactionResult(text, items, secret_detected)

* Patterns and false-positive guards live in ``config/privacy.yaml``.
* **Stable placeholders**: the same value gets the same placeholder within one call
  (``<EMAIL_1>`` twice for the same address). ``SECRET``/``TOKEN`` are not numbered.
* ``Redaction.span`` is the ``(start, end)`` of the placeholder **in the redacted text**, and
  ``original_hash`` is a sha256 of the original value; the original itself is never stored.
* IP originals are needed for the L2 bundle (§8.2): use ``redact_with_map()``, which also
  returns ``{placeholder: original}`` for the caller to keep in memory only.
* Error codes such as ``ORA-01013``, ``0xc0000005`` and ``SQL Server error 18456`` are protected
  and never redacted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.schemas import Redaction, RedactionResult
from store.config import CONFIG_DIR, load_yaml

# Types whose placeholders carry no number (§6.2 table).
UNNUMBERED = frozenset({"SECRET", "TOKEN"})
SECRET_TYPES = frozenset({"SECRET", "TOKEN"})


@dataclass(frozen=True)
class _Rule:
    type: str
    regex: re.Pattern[str]
    value_group: str | None


@dataclass(frozen=True)
class _Config:
    rules: tuple[_Rule, ...]
    protected: tuple[re.Pattern[str], ...]
    person_stop: frozenset[str]
    names: re.Pattern[str] | None
    order: tuple[str, ...]


@lru_cache(maxsize=4)
def _config(path: Path | None = None) -> _Config:
    cfg = load_yaml(path or CONFIG_DIR / "privacy.yaml")
    rules: list[_Rule] = []
    order: list[str] = []
    for typ, entries in (cfg.get("patterns") or {}).items():
        order.append(typ)
        for e in entries:
            flags = 0 if e.get("case_sensitive") else re.IGNORECASE
            rules.append(_Rule(typ, re.compile(e["regex"], flags), e.get("value_group")))
    names = cfg.get("names") or {}
    first = [re.escape(n) for n in names.get("first", [])]
    last = [re.escape(n) for n in names.get("last", [])]
    names_re = None
    if first:
        alt_first = "|".join(first)
        alt_last = "|".join(last) or r"(?!)"
        names_re = re.compile(rf"\b(?:{alt_first})(?:\s+(?:{alt_last}))?\b")
    if "PERSON" not in order:
        order.append("PERSON")
    return _Config(
        rules=tuple(rules),
        protected=tuple(re.compile(p, re.IGNORECASE) for p in cfg.get("protected") or []),
        person_stop=frozenset(cfg.get("person_stopwords") or []),
        names=names_re,
        order=tuple(order),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidates(text: str, cfg: _Config) -> list[tuple[int, int, str, int]]:
    """``(start, end, type, priority)`` for every raw match, before overlap resolution."""
    prio = {t: i for i, t in enumerate(cfg.order)}
    out: list[tuple[int, int, str, int]] = []
    for rule in cfg.rules:
        for m in rule.regex.finditer(text):
            if rule.value_group:
                s, e = m.span(rule.value_group)
            else:
                s, e = m.span()
            if e <= s:
                continue
            value = text[s:e]
            if rule.type == "PERSON":
                words = value.split()
                if words and words[0] in cfg.person_stop:
                    continue
                if len(words) == 2 and words[1] in cfg.person_stop:
                    e = s + len(words[0])
            out.append((s, e, rule.type, prio[rule.type]))
    if cfg.names is not None:
        for m in cfg.names.finditer(text):
            out.append((m.start(), m.end(), "PERSON", prio["PERSON"]))
    return out


def _protected_spans(text: str, cfg: _Config) -> list[tuple[int, int]]:
    return [m.span() for p in cfg.protected for m in p.finditer(text)]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _select(text: str, cfg: _Config) -> list[tuple[int, int, str]]:
    """Drop guarded matches, then keep non-overlapping ones (type priority, then longest)."""
    guards = _protected_spans(text, cfg)
    cands = [
        c
        for c in _candidates(text, cfg)
        # secrets and tokens are always redacted, even inside something that looks like a code
        if c[2] in SECRET_TYPES or not any(_overlaps((c[0], c[1]), g) for g in guards)
    ]
    cands.sort(key=lambda c: (c[3], -(c[1] - c[0]), c[0]))
    chosen: list[tuple[int, int, str]] = []
    for s, e, typ, _ in cands:
        if not any(_overlaps((s, e), (cs, ce)) for cs, ce, _ in chosen):
            chosen.append((s, e, typ))
    return sorted(chosen)


def redact_with_map(
    text: str, *, config_path: Path | None = None
) -> tuple[RedactionResult, dict[str, str]]:
    """Redact ``text`` and also return ``{placeholder: original}`` (keep it in memory only)."""
    cfg = _config(config_path)
    if not text:
        return RedactionResult(text=text or ""), {}
    chosen = _select(text, cfg)
    counters: dict[str, int] = {}
    seen: dict[tuple[str, str], str] = {}
    mapping: dict[str, str] = {}
    items: list[Redaction] = []
    parts: list[str] = []
    pos = 0
    out_len = 0
    for s, e, typ in chosen:
        value = text[s:e]
        key = (typ, value.lower() if typ in ("EMAIL", "PERSON") else value)
        if typ in UNNUMBERED:
            ph = f"<{typ}>"
        elif key in seen:
            ph = seen[key]
        else:
            counters[typ] = counters.get(typ, 0) + 1
            ph = f"<{typ}_{counters[typ]}>"
            seen[key] = ph
        if typ not in UNNUMBERED:
            mapping.setdefault(ph, value)
        chunk = text[pos:s]
        parts.append(chunk)
        out_len += len(chunk)
        items.append(
            Redaction(
                type=typ,
                placeholder=ph,
                span=(out_len, out_len + len(ph)),
                original_hash=_sha(value),
            )
        )
        parts.append(ph)
        out_len += len(ph)
        pos = e
    parts.append(text[pos:])
    result = RedactionResult(
        text="".join(parts),
        items=items,
        secret_detected=any(i.type in SECRET_TYPES for i in items),
    )
    return result, mapping


def redact(text: str, keep_map: bool = True) -> RedactionResult:
    """``Redactor`` protocol entry point (shared/interfaces.py).

    ``keep_map`` is accepted for interface compatibility; the original values are never part
    of the result. Callers that need originals (the L2 bundle) use ``redact_with_map``.
    """
    return redact_with_map(text)[0]


def redact_text(text: str) -> str:
    """Convenience for indexers: just the redacted string."""
    return redact(text).text


def summary(result: RedactionResult) -> dict[str, Any]:
    """``details.redactions`` shape: ``{"count": n, "types": [...]}``."""
    return {"count": len(result.items), "types": sorted({i.type for i in result.items})}
