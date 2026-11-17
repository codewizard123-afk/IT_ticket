"""P2-T4: impact × urgency priority engine (architecture §7.4, §16)."""

from __future__ import annotations

import pytest

from ai.priority import compute_priority, most_severe
from ai.signals import extract, extract_signals, ticket_text
from shared.schemas import ClassificationResult, Signals


def cls(category: str, sub: str, intent: str = "incident") -> ClassificationResult:
    return ClassificationResult(
        category=category, subcategory=sub, confidence=0.9, band="high", intent=intent
    )


# (demo, category, subcategory, kb_id, kb_hint, expected level, reason must contain)
DEMO_CASES = [
    ("DEMO-001", "Network", "VPN", "KB-0018", "P2", "High", "deadline today"),
    ("DEMO-002", "Database", "Oracle", "KB-0001", "P2", "Critical", "15 users affected"),
    ("DEMO-003", "Hardware", "Monitor", "KB-0008", "P3", "Medium", "Baseline for Monitor"),
    ("DEMO-004", "Software", "Email", "KB-0029", "P2", "High", "floor: KB-0029 hint P2"),
    ("DEMO-006", "Network", "Wireless", "KB-0021", "P3", "High", "multiple users affected"),
    ("DEMO-007", "Database", "MS SQL Server", "KB-0003", "P2", "High", "Baseline for MS SQL"),
]


@pytest.mark.parametrize(("demo", "cat", "sub", "kb", "hint", "level", "reason"), DEMO_CASES)
def test_demo_priorities_match_architecture_16(
    demos, kb_store, demo, cat, sub, kb, hint, level, reason
) -> None:
    t = demos[demo]
    p = compute_priority(
        cls(cat, sub),
        extract_signals(t, kb_store),
        text=ticket_text(t),
        kb_id=kb,
        kb_hint=hint,
        kb_relevance=0.8,
    )
    assert p.level == level
    assert reason in p.reason


def test_demo_001_drivers(demos, kb_store) -> None:
    p = compute_priority(
        cls("Network", "VPN"),
        extract_signals(demos["DEMO-001"], kb_store),
        kb_id="KB-0018",
        kb_hint="P2",
        kb_relevance=0.8,
    )
    assert (p.code, p.impact, p.urgency) == ("P2", 3, 1)
    assert p.drivers == ["deadline today → U1", "floor: VPN baseline P2"]
    assert p.reason == "deadline today; floor: VPN baseline"


def test_demo_002_is_p1_with_both_drivers(demos, kb_store) -> None:
    p = compute_priority(cls("Database", "Oracle"), extract_signals(demos["DEMO-002"], kb_store))
    assert (p.code, p.impact, p.urgency) == ("P1", 1, 1)
    assert p.reason == "15 users affected; month-end"


def test_demo_005_issues() -> None:
    vpn = compute_priority(
        cls("Network", "VPN"),
        extract("My VPN connection drops every 10 to 15 minutes and I keep losing unsaved work."),
    )
    cpu = compute_priority(
        cls("Hardware", "CPU"), extract("My laptop is overheating and the fan runs loudly.")
    )
    assert (vpn.level, vpn.urgency) == ("High", 2)
    assert vpn.reason == "losing unsaved work; floor: VPN baseline"
    assert (cpu.level, cpu.reason) == ("Medium", "Baseline for CPU")
    assert most_severe([vpn.code, cpu.code]) == "P2"


def test_kb_hint_ignored_below_tau_kb() -> None:
    sig = Signals(urgency_terms=["end of day"])
    low = compute_priority(
        cls("Software", "Email"), sig, kb_id="KB-0029", kb_hint="P2", kb_relevance=0.3
    )
    high = compute_priority(
        cls("Software", "Email"), sig, kb_id="KB-0029", kb_hint="P2", kb_relevance=0.6
    )
    assert (low.code, high.code) == ("P3", "P2")


def test_how_to_uses_matrix_only() -> None:
    p = compute_priority(cls("Software", "Email", intent="how_to"), Signals())
    assert (p.code, p.urgency) == ("P4", 3)


def test_security_is_p1() -> None:
    p = compute_priority(cls("Inquiry/Help", "Antivirus"), Signals(security_flags=["ransomware"]))
    assert p.code == "P1"


def test_site_scope_and_shared_resource() -> None:
    site = compute_priority(cls("Network", "DNS"), Signals(scope="site"))
    assert site.impact == 1
    shared = compute_priority(
        cls("Hardware", "Disk"), Signals(), text="The shared drive is full and read-only"
    )
    assert shared.impact == 2


def test_most_severe() -> None:
    assert most_severe(["P3", None, "P2", "P4"]) == "P2"
    assert most_severe([]) == "P4"
