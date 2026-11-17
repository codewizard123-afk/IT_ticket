"""P2-T4: priority engine. Demo priorities must match architecture §16."""

from __future__ import annotations

import pytest

from ai.priority import MATRIX, compute, most_severe, ticket_priority
from ai.signals import extract
from ai.types import ClassificationResult, PriorityResult
from tests.p2_ai.fakes import FakeKBStore, load_fixture

DEMOS = {d["id"]: d for d in load_fixture("demo_tickets.json")}
PHRASES = FakeKBStore().all_error_phrases()


def cls(cat: str, sub: str, intent: str = "incident") -> ClassificationResult:
    return ClassificationResult(cat, sub, 0.9, "high", intent=intent)  # type: ignore[arg-type]


def demo_sig(demo_id: str, span: str | None = None):
    d = DEMOS[demo_id]
    return extract(span or f"{d['title']}. {d['description']}", PHRASES)


# (demo, category, subcategory, kb_id, kb_hint, kb_relevance, expected level, reason parts)
CASES = [
    ("DEMO-001", "Network", "VPN", "KB-0018", "P2", 0.8, "High",
     ["deadline today", "floor: VPN baseline"]),
    ("DEMO-002", "Database", "Oracle", "KB-0001", "P2", 0.8, "Critical",
     ["15 users affected", "month-end"]),
    ("DEMO-003", "Hardware", "Monitor", "KB-0008", "P3", 0.8, "Medium",
     ["Baseline for Monitor"]),
    ("DEMO-004", "Software", "Email", "KB-0029", "P2", 0.8, "High",
     ["end of day", "floor: KB-0029 hint P2"]),
    ("DEMO-006", "Network", "Wireless", "KB-0021", "P3", 0.8, "High",
     ["multiple colleagues affected", "floor: Wireless baseline"]),
    ("DEMO-007", "Database", "MS SQL Server", "KB-0003", "P3", 0.8, "High",
     ["Baseline for MS SQL Server"]),
]  # fmt: skip


@pytest.mark.parametrize(
    ("demo", "cat", "sub", "kb", "hint", "rel", "level", "parts"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_demo_priorities(demo, cat, sub, kb, hint, rel, level, parts):
    p = compute(demo_sig(demo), cls(cat, sub), kb_id=kb, kb_hint=hint, kb_relevance=rel)
    assert p.level == level
    for part in parts:
        assert part in p.reason
    assert len(p.reason.split("; ")) <= 2


def test_demo_005_ticket_priority_is_high():
    d = DEMOS["DEMO-005"]
    vpn_span = d["description"].split("(2)")[0]
    cpu_span = d["description"].split("(2)")[1]
    p1 = compute(
        demo_sig("DEMO-005", vpn_span),
        cls("Network", "VPN"),
        kb_id="KB-0018",
        kb_hint="P2",
        kb_relevance=0.8,
    )
    p2 = compute(
        demo_sig("DEMO-005", cpu_span),
        cls("Hardware", "CPU"),
        kb_id="KB-0030",
        kb_hint="P3",
        kb_relevance=0.6,
    )
    assert p1.level == "High"
    assert "losing unsaved work" in p1.reason and "floor: VPN baseline" in p1.reason
    assert ticket_priority([p1, p2]).level == "High"


def test_kb_hint_ignored_below_tau_kb():
    s = extract("Outlook crashes, I need it by end of day", [])
    p = compute(
        s, cls("Software", "Email"), kb_id="KB-0029", kb_hint="P2", kb_relevance=0.3
    )
    assert p.code == "P3"


def test_security_is_critical_for_many_users():
    s = extract("Ransomware: files are encrypted for 20 users", [])
    p = compute(s, cls("Inquiry/Help", "Antivirus"))
    assert p.code == "P1" and p.impact == 1 and p.urgency == 1
    assert "security" in p.reason


def test_how_to_uses_matrix_only():
    s = extract("How do I run an antivirus scan?", [])
    p = compute(s, cls("Inquiry/Help", "Antivirus", intent="how_to"))
    assert p.code == "P4" and p.urgency == 3


def test_service_request_not_floored_by_baseline():
    s = extract("Please install a new license for Visio", [])
    p = compute(
        s,
        cls("Network", "VPN", intent="service_request"),
        kb_hint="P2",
        kb_relevance=0.9,
    )
    assert p.code == "P4"


def test_matrix_complete():
    assert MATRIX[(1, 1)] == "P1" and MATRIX[(3, 3)] == "P4" and len(MATRIX) == 9


def test_drivers_include_mapping():
    p = compute(demo_sig("DEMO-002"), cls("Database", "Oracle"))
    assert any(d.endswith("→ I1") for d in p.drivers)
    assert any(d.endswith("→ U1") for d in p.drivers)


def test_most_severe_and_ticket_priority_ties():
    assert most_severe("P3", None, "P2") == "P2"
    a = PriorityResult("P2", "High", "a")
    b = PriorityResult("P2", "High", "b")
    assert ticket_priority([a, b]).reason == "a"
    with pytest.raises(ValueError, match="no issues"):
        ticket_priority([])
