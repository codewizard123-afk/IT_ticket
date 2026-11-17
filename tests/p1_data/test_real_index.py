"""Sanity checks on the real index built by ``make data`` (bge-small). Marked ``slow``:
excluded from CI, run locally with ``pytest -m slow tests/p1_data``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("chromadb")

from store.factory import IndexMissingError, open_stores  # noqa: E402


@pytest.fixture(scope="module")
def real():
    try:
        return open_stores()
    except IndexMissingError:
        pytest.skip("run `make data` first")


@pytest.mark.parametrize(
    ("text", "kb_id", "subcategory"),
    [
        (
            "VPN not connecting - Gateway not responding. Since this morning I cannot connect "
            "to the VPN from home.",
            "KB-0018",
            "VPN",
        ),
        (
            "Monitor blacks out after update. My external monitor goes black every few minutes.",
            "KB-0008",
            "Monitor",
        ),
        (
            "Outlook keeps asking for my password and shows Disconnected / Trying to connect.",
            "KB-0048",
            "Email",
        ),
    ],
)
def test_real_retrieval(real, text, kb_id, subcategory):
    tickets, kb = real
    kb_ids = [h.kb_id for h in kb.query_chunks(text, 10)]
    assert kb_id in kb_ids[:5]
    subs = [h.subcategory for h in tickets.query(text, 5)]
    assert subs.count(subcategory) >= 3
