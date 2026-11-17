"""Deterministic signal extractor (architecture §7.1). No LLM, no network.

`extract(text, kb_phrases)` returns a `Signals` record: error codes (regex families plus
KEDB phrase codes from `KBStore.all_error_phrases()`), affected users and scope, urgency,
business-critical and security flags, tried steps, location, onset and change trigger.
The input text must already be redacted.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import cache
from typing import Any, Literal

from ai.config import signals_config
from ai.text import STOPWORDS, tokenize

Scope = Literal["single", "multiple", "team", "site", "org"]
_SCOPE_RANK = {"single": 0, "multiple": 1, "team": 2, "site": 3, "org": 4}


@dataclass
class Signals:
    error_codes: list[str] = field(default_factory=list)
    error_code_families: dict[str, str] = field(default_factory=dict)  # code -> family
    products: list[str] = field(default_factory=list)
    affected_users: int | None = None
    scope: Scope = "single"
    scope_explicit: bool = False  # False when scope is only the default guess
    scope_evidence: str | None = None  # e.g. "15 users", "multiple colleagues"
    urgency_terms: list[str] = field(default_factory=list)
    urgency_level: int | None = None  # 1 (U1) or 2 (U2); None = no urgency signal
    business_critical: bool = False
    environment: str | None = None
    location: str | None = None
    onset: str | None = None
    change_trigger: str | None = None
    multi_issue_markers: bool = False
    security_flags: list[str] = field(default_factory=list)
    tried_steps: list[str] = field(default_factory=list)
    tried_actions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flags(case_sensitive: bool = False) -> int:
    return 0 if case_sensitive else re.IGNORECASE


@cache
def _compile(pattern: str, flags: int) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    left = r"\b" if phrase[:1].isalnum() else ""
    right = r"\b" if phrase[-1:].isalnum() else ""
    return _compile(left + escaped + right, re.IGNORECASE)


def _contains(text: str, phrase: str) -> bool:
    return bool(_phrase_regex(phrase).search(text))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def extract_error_codes(
    text: str, kb_phrases: list[str] | None, cfg: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    codes: list[str] = []
    families: dict[str, str] = {}
    spans: list[tuple[int, int]] = []

    def add(code: str, family: str, span: tuple[int, int]) -> None:
        # skip a code fully inside one already captured (e.g. "error 18456" in
        # "SQL Server error 18456")
        if any(s <= span[0] and span[1] <= e for s, e in spans):
            return
        spans.append(span)
        codes.append(code)
        families.setdefault(code, family)

    for item in cfg.get("error_codes", []):
        rx = _compile(item["pattern"], _flags(bool(item.get("case_sensitive"))))
        for m in rx.finditer(text):
            add(m.group(0).strip(), item["family"], m.span())

    min_chars = int(cfg.get("phrase_min_chars", 5))
    for phrase in sorted(set(kb_phrases or []), key=len, reverse=True):
        p = phrase.strip()
        if not p:
            continue
        if len(p) < min_chars and not any(ch.isdigit() for ch in p):
            continue
        m = _phrase_regex(p).search(text)
        if m:
            add(p, "kedb_phrase", m.span())
    return _dedupe(codes), families


def extract_users(
    text: str, cfg: dict[str, Any]
) -> tuple[int | None, Scope, bool, str | None]:
    """Return `(count, scope, explicit, evidence)`; evidence is the phrase behind the scope."""
    ucfg = cfg.get("users", {})
    count: int | None = None
    count_noun = "users"
    for pat in ucfg.get("count_patterns", []):
        m = _compile(pat, re.IGNORECASE).search(text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 100000:
                count = n
                noun = re.search(
                    r"(users|people|staff|colleagues|employees|members|agents|persons)\b",
                    m.group(0),
                    re.IGNORECASE,
                )
                count_noun = noun.group(1).lower() if noun else "users"
                break

    scope: Scope = "single"
    explicit = False
    evidence: str | None = None
    for sc, phrases in (ucfg.get("scope_phrases") or {}).items():
        hit = next((p for p in phrases if _contains(text, p)), None)
        if hit and _SCOPE_RANK[sc] > _SCOPE_RANK[scope]:
            scope = sc
            explicit = True
            evidence = hit
    if count is not None:
        explicit = True
        for rule in ucfg.get("count_scope", []):
            if count >= int(rule["min"]):
                if _SCOPE_RANK[rule["scope"]] >= _SCOPE_RANK[scope]:
                    scope = rule["scope"]
                    evidence = f"{count} {count_noun}"
                break
        if evidence is None:
            evidence = f"{count} {count_noun}"
    return count, scope, explicit, evidence


def extract_urgency(text: str, cfg: dict[str, Any]) -> tuple[list[str], int | None]:
    terms: list[str] = []
    level: int | None = None
    for lvl_name, phrases in (cfg.get("urgency") or {}).items():
        lvl = int(str(lvl_name).lstrip("Uu"))
        for p in phrases:
            if _contains(text, p):
                terms.append(p)
                level = lvl if level is None else min(level, lvl)
    return _dedupe(terms), level


def extract_tried_steps(text: str, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    steps: list[str] = []
    actions: list[str] = []
    n_obj = int(cfg.get("tried_object_words", 4))
    for item in cfg.get("tried_steps", []):
        rx = _compile(item["pattern"], re.IGNORECASE)
        for m in rx.finditer(text):
            tail = text[m.end() :]
            tail = re.split(r"[.,;:!?\n]| but | and | then ", tail, maxsplit=1)[0]
            obj = " ".join(tail.split()[:n_obj])
            phrase = (m.group(0) + (" " + obj if obj else "")).strip()
            steps.append(phrase)
            actions.append(item["action"])
    return _dedupe(steps), _dedupe(actions)


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = _compile(pat, re.IGNORECASE).search(text)
        if m:
            return m.group(0).strip()
    return None


def extract_keywords(text: str, n: int) -> list[str]:
    toks = [
        t
        for t in tokenize(text)
        if len(t) > 2 and not t.isdigit() and t not in STOPWORDS
    ]
    counts = Counter(toks)
    order = {t: i for i, t in reversed(list(enumerate(toks)))}
    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    return ranked[:n]


def extract(
    text: str, kb_phrases: list[str] | None = None, cfg: dict[str, Any] | None = None
) -> Signals:
    """Extract all signals from redacted ticket text."""
    cfg = cfg if cfg is not None else signals_config()
    codes, families = extract_error_codes(text, kb_phrases, cfg)
    users, scope, explicit, scope_evidence = extract_users(text, cfg)
    urgency_terms, urgency_level = extract_urgency(text, cfg)
    tried, actions = extract_tried_steps(text, cfg)

    products = [
        name
        for name, kws in (cfg.get("products") or {}).items()
        if any(
            _contains(text, k) if k[-1:].isalnum() else k.lower() in text.lower()
            for k in kws
        )
    ]
    environment = None
    for env, phrases in (cfg.get("environment") or {}).items():
        if any(_contains(text, p) for p in phrases):
            environment = env
            break

    multi = any(
        _compile(p, re.IGNORECASE | re.DOTALL).search(text)
        for p in cfg.get("multi_issue", [])
    )

    return Signals(
        error_codes=codes,
        error_code_families=families,
        products=products,
        affected_users=users,
        scope=scope,
        scope_explicit=explicit,
        scope_evidence=scope_evidence,
        urgency_terms=urgency_terms,
        urgency_level=urgency_level,
        business_critical=any(
            _contains(text, p) for p in cfg.get("business_critical", [])
        ),
        environment=environment,
        location=_first_match(text, cfg.get("location", [])),
        onset=_first_match(text, cfg.get("onset", [])),
        change_trigger=_first_match(text, cfg.get("change_trigger", [])),
        multi_issue_markers=multi,
        security_flags=_dedupe(
            [p for p in cfg.get("security", []) if _contains(text, p)]
        ),
        tried_steps=tried,
        tried_actions=actions,
        keywords=extract_keywords(text, int(cfg.get("max_keywords", 12))),
    )


def step_already_attempted(
    step_text: str, signals: Signals, cfg: dict[str, Any] | None = None
) -> bool:
    """True if a recommended step repeats something the user says they already tried.

    A step matches a tried action when it contains one of the action's `step_keywords` and,
    if the user named an object (e.g. "restarted the VPN client"), shares a content word
    with that object.
    """
    if not signals.tried_steps:
        return False
    cfg = cfg if cfg is not None else signals_config()
    by_action = {it["action"]: it for it in cfg.get("tried_steps", [])}
    step_l = step_text.lower()
    step_toks = set(tokenize(step_text))
    for phrase in signals.tried_steps:
        for action in signals.tried_actions:
            item = by_action.get(action)
            if not item or not _compile(item["pattern"], re.IGNORECASE).search(phrase):
                continue
            if not any(kw in step_l for kw in item.get("step_keywords", [])):
                continue
            if not item.get("needs_object"):
                return True
            verb = _compile(item["pattern"], re.IGNORECASE).search(phrase)
            obj_toks = set(tokenize(phrase[verb.end() :] if verb else ""))
            if not obj_toks or obj_toks & step_toks:
                return True
    return False
