"""P2-T2: deterministic signal extraction. Golden signals for all 7 demos."""

from __future__ import annotations

import pytest

from ai.signals import extract, step_already_attempted
from tests.p2_ai.fakes import FakeKBStore, load_fixture

DEMOS = {d["id"]: d for d in load_fixture("demo_tickets.json")}


@pytest.fixture(scope="module")
def phrases() -> list[str]:
    return FakeKBStore().all_error_phrases()


def sig(demo_id: str, phrases: list[str]):
    d = DEMOS[demo_id]
    return extract(f"{d['title']}. {d['description']}", phrases)


def test_demo_001_vpn(phrases):
    s = sig("DEMO-001", phrases)
    assert "Gateway not responding" in s.error_codes
    assert s.error_code_families["Gateway not responding"] == "kedb_phrase"
    assert "vpn" in s.products
    assert "deadline today" in s.urgency_terms and s.urgency_level == 1
    assert not s.business_critical
    assert s.scope == "single" and s.affected_users is None
    assert any("restarted the VPN client" in t for t in s.tried_steps)
    assert s.onset and "since this morning" in s.onset.lower()
    assert s.environment == "remote"
    assert not s.security_flags and not s.multi_issue_markers


def test_demo_002_oracle(phrases):
    s = sig("DEMO-002", phrases)
    assert "ORA-01013" in s.error_codes
    assert s.affected_users == 15
    assert s.scope in ("team", "site", "org") and s.scope_explicit
    assert "month-end" in s.urgency_terms and s.urgency_level == 1
    assert s.business_critical  # month-end close / finance team
    assert "oracle" in s.products


def test_demo_003_monitor(phrases):
    s = sig("DEMO-003", phrases)
    assert s.error_codes == []
    assert "monitor" in s.products
    assert s.urgency_level is None and s.urgency_terms == []
    assert s.scope == "single" and not s.scope_explicit
    assert any("different cable" in t for t in s.tried_steps)
    assert s.change_trigger and "update" in s.change_trigger.lower()
    assert s.onset and "update" in s.onset.lower()


def test_demo_004_outlook(phrases):
    s = sig("DEMO-004", phrases)
    assert "0xc0000005" in s.error_codes
    assert "Faulting application name" in s.error_codes
    assert "outlook" in s.products
    assert "end of day" in s.urgency_terms and s.urgency_level == 1


def test_demo_005_multi_issue(phrases):
    s = sig("DEMO-005", phrases)
    assert s.multi_issue_markers
    assert {"vpn", "laptop"} <= set(s.products)
    assert s.urgency_level == 1  # losing unsaved work
    assert not s.security_flags


def test_demo_006_wifi(phrases):
    s = sig("DEMO-006", phrases)
    assert "wifi" in s.products
    assert s.scope == "multiple" and s.scope_explicit
    assert s.location and "building b" in s.location.lower()
    assert s.environment == "office"


def test_demo_007_sqlserver(phrases):
    s = sig("DEMO-007", phrases)
    assert "SQL Server error 18456" in s.error_codes
    assert "error 18456" not in s.error_codes  # nested match collapsed
    assert "Login failed for user" in s.error_codes
    assert "sql server" in s.products
    assert s.urgency_level is None
    assert s.change_trigger and "new joiner" in s.change_trigger.lower()


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Install fails with MSI error 1603 on my laptop", "MSI error 1603"),
        (
            "Blue screen with IRQL_NOT_LESS_OR_EQUAL after boot",
            "IRQL_NOT_LESS_OR_EQUAL",
        ),
        ("DB2 connect fails with SQL1060N for the batch user", "SQL1060N"),
        ("ODBC returns SQLSTATE 08001 when connecting", "SQLSTATE 08001"),
    ],
)
def test_error_code_families(text, code):
    assert code in extract(text, []).error_codes


def test_security_flags():
    s = extract(
        "I clicked a suspicious link in a phishing email and now files are encrypted",
        [],
    )
    assert {"suspicious", "phishing", "files are encrypted"} <= set(s.security_flags)


def test_scope_phrases():
    assert extract("The whole team cannot open the share", []).scope == "team"
    assert extract("Everyone in the company lost email", []).scope == "org"
    assert extract("approximately 4 people on my desk row", []).affected_users == 4


def test_deadline_alone_is_not_business_critical():
    s = extract("I have a deadline today for my presentation", [])
    assert s.urgency_level == 1 and not s.business_critical


def test_production_is_business_critical():
    assert extract("The production server is unreachable", []).business_critical


def test_step_already_attempted(phrases):
    s1 = sig("DEMO-001", phrases)
    assert step_already_attempted("Restart the VPN client", s1)
    assert not step_already_attempted("Restart the application server", s1)
    assert not step_already_attempted("Check internet connectivity without the VPN", s1)
    s3 = sig("DEMO-003", phrases)
    assert step_already_attempted(
        "Check and reseat the video cable or try a different cable", s3
    )
    assert not step_already_attempted("Roll back or update the display driver", s3)


def test_keywords_are_content_words(phrases):
    s = sig("DEMO-006", phrases)
    assert "wi-fi" in s.keywords or "wireless" in s.keywords
    assert "the" not in s.keywords


def test_redacted_placeholders_are_not_codes():
    s = extract("Contact <EMAIL_1> or call <PHONE_1>, user <PERSON_1> on <IP_1>", [])
    assert s.error_codes == []
