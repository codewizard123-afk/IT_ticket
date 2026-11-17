"""``privacy.redact`` (P1-T2): ≥ 30 positive/negative cases plus dataset-wide checks."""

from __future__ import annotations

import json
import re

import pytest

from privacy.redact import redact, redact_text, redact_with_map, summary
from shared.interfaces import Redactor
from shared.schemas import RedactionResult
from store.config import ROOT


def _types(text: str) -> list[str]:
    return [i.type for i in redact(text).items]


# --- positives -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "typ", "gone"),
    [
        ("Reach me at ananya.joshi@example.com today", "EMAIL", "ananya.joshi@example.com"),
        ("mail: A.B-c+tag@sub.corp.co.in", "EMAIL", "A.B-c+tag@sub.corp.co.in"),
        ("Please call me on +91 90000 20059.", "PHONE", "90000 20059"),
        ("Call +1 (555) 123-4567 after 5", "PHONE", "123-4567"),
        ("UK desk +44 20 7946 0958", "PHONE", "7946 0958"),
        ("my number is 555-123-4567", "PHONE", "555-123-4567"),
        ("mobile 9876543210 please", "PHONE", "9876543210"),
        ("mobile 98765-43210 please", "PHONE", "98765-43210"),
        ("Server 10.20.30.40 is down", "IP", "10.20.30.40"),
        ("gateway 192.168.1.1, dns 8.8.8.8", "IP", "192.168.1.1"),
        ("host 2001:0db8:85a3:0000:0000:8a2e:0370:7334 unreachable", "IP", "2001:0db8"),
        ("link-local fe80::1ff:fe23:4567:890a fails", "IP", "fe80::1ff"),
        ("My employee ID is EMP856932 thanks", "ID", "EMP856932"),
        ("asset tag LT-00123456 is broken", "ID", "LT-00123456"),
        ("Contact Neha Iyer on the phone", "PERSON", "Neha Iyer"),
        ("Hi, my name is Rohan Mehta and VPN fails", "PERSON", "Rohan Mehta"),
        ("user: jdoe42 cannot log in", "PERSON", "jdoe42"),
        ("Thanks, Priya", "PERSON", "Priya"),
        ("Server=db1;User Id=app;Password=Sup3r$ecret;", "SECRET", "Sup3r$ecret"),
        ("I set pwd=hunter2 yesterday", "SECRET", "hunter2"),
        ('password: "my pass phrase"', "SECRET", "my pass phrase"),
        ("key gsk_abcdefghijklmnopqrstuvwx123", "TOKEN", "gsk_abcdefghijklmnopqrstuvwx123"),
        ("OPENAI sk-proj-AbCdEfGhIjKlMnOpQrSt", "TOKEN", "sk-proj-AbCdEfGhIjKlMnOpQrSt"),
        ("Authorization: Bearer abcdefghijklmnop.qrstuvwx", "TOKEN", "abcdefghijklmnop.qrstuvwx"),
        (
            "hash d41d8cd98f00b204e9800998ecf8427e leaked",
            "TOKEN",
            "d41d8cd98f00b204e9800998ecf8427e",
        ),
    ],
)
def test_redacts(text, typ, gone):
    res = redact(text)
    assert typ in [i.type for i in res.items], res
    assert gone not in res.text
    assert res.secret_detected == (typ in ("SECRET", "TOKEN"))


# --- negatives: false-positive guards ----------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Queries time out with ORA-01013: user requested cancel of current operation",
        "Event Viewer shows exception code 0xc0000005 in OUTLOOK.EXE",
        "SQL Server error 18456: Login failed for user",
        "Windows Update fails with 0x800f0922 again",
        "SQLSTATE 08001 when the pool is exhausted",
        "DB2 returns SQL30081N and SQL0911N reason code 68",
        "MSI error 1618: another installation is in progress",
        "Login fails with 'ERROR 1045 (28000): Access denied for user'",
        "Event ID 4740 shows the lockout source",
        "HTTP 500 Internal Server Error on the portal",
        "Device has a 169.254.x.x address",
        "Bounce: 550 5.2.2 mailbox full",
        "See KB-0018 and INC0892, opened 2026-05-01 09:42",
        "Approximately 15 users in the finance team are affected",
        "I am unable to connect since Monday",
        "This is Outlook crashing again",
        "Build 10.0.19045.3693 after the update",
        "Version 1.2.3.4567 installed",
    ],
)
def test_does_not_redact(text):
    res = redact(text)
    assert res.text == text, res.items
    assert not res.secret_detected


# --- behaviour --------------------------------------------------------------------------------
def test_stable_placeholders_and_spans():
    text = "Mail a@x.com or b@x.com; again A@x.com. Call +91 90000 11111 or +91 90000 11111."
    res, mapping = redact_with_map(text)
    assert res.text.count("<EMAIL_1>") == 2 and res.text.count("<EMAIL_2>") == 1
    assert res.text.count("<PHONE_1>") == 2 and "<PHONE_2>" not in res.text
    for item in res.items:
        assert res.text[item.span[0] : item.span[1]] == item.placeholder
        assert re.fullmatch(r"[0-9a-f]{64}", item.original_hash)
    assert mapping["<EMAIL_1>"] == "a@x.com"
    assert "a@x.com" not in json.dumps(res.model_dump())


def test_ip_original_kept_only_in_map():
    res, mapping = redact_with_map("VPN gateway 10.1.2.3 not responding")
    assert res.text == "VPN gateway <IP_1> not responding"
    assert mapping == {"<IP_1>": "10.1.2.3"}


def test_secret_placeholder_unnumbered_and_flag():
    res = redact("pwd=abc123 and password=xyz789")
    assert res.text == "pwd=<SECRET> and password=<SECRET>"
    assert res.secret_detected


def test_empty_and_plain():
    assert redact("").text == ""
    plain = "Printer offline in Building B"
    assert redact(plain) == RedactionResult(text=plain)


def test_protocol_and_helpers():
    assert isinstance(redact, Redactor)
    assert redact_text("x a@b.co") == "x <EMAIL_1>"
    assert summary(redact("a@b.co +91 90000 11111")) == {"count": 2, "types": ["EMAIL", "PHONE"]}


# --- dataset-wide checks ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tickets() -> list[dict]:
    return json.loads((ROOT / "data/processed/tickets.json").read_text("utf-8"))


def test_all_pii_tickets_redacted(tickets):
    pii = [t for t in tickets if t["metadata"]["contains_pii"]]
    assert len(pii) == 30
    leaks = []
    for t in pii:
        out = redact(t["description"]).text
        if re.search(r"@example\.com|\+91|EMP\d|Contact [A-Z][a-z]+ [A-Z]", out):
            leaks.append((t["id"], out))
    assert leaks == []


def test_no_false_positives_on_clean_tickets(tickets):
    clean = [t for t in tickets if not t["metadata"]["contains_pii"]]
    changed = [
        (t["id"], redact(t["description"]).items) for t in clean if redact(t["description"]).items
    ]
    assert changed == []


def test_kedb_error_codes_never_redacted():
    arts = json.loads((ROOT / "data/kb/articles.json").read_text("utf-8"))
    codes = {c for a in arts for c in a["common_error_codes"]}
    assert len(codes) > 150
    changed = {c: redact(c).text for c in codes if redact(c).text != c}
    assert changed == {}
