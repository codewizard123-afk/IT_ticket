"""P4-T9: worked examples rendered from audit records."""

from __future__ import annotations

import re

from tests.p4_ui.helpers import page
from ui.examples import EXAMPLES, generate
from ui.settings import ROOT

SECTIONS = ("## 1. Input", "Classification", "Similar cases", "Recommended solution", "Evidence")


def test_generate_all_examples(tmp_path, make_client, golden):
    paths = generate(make_client("l2"), tmp_path, "test")
    assert len(paths) == len(EXAMPLES) >= 3
    texts = {p.name: p.read_text(encoding="utf-8") for p in paths}
    for name, md in texts.items():
        for section in SECTIONS:
            assert section in md, (name, section)
        assert "Audit hash chain: verified." in md
        # every numbered step line carries a source badge
        for line in re.findall(r"^\d+\. .*$", md, flags=re.M):
            assert re.search(r"`\[(KB-\d{4}·(res|diag)\.\d+|INC-[\w-]+)\]`", line), line
    one = texts["01-recommend-vpn-DEMO-001.md"]
    assert "**RECOMMEND**" in one and "KB-0018" in one and "~~already tried~~" in one
    two = texts["02-clarify-then-recommend.md"]
    assert "Decision **CLARIFY**" in two and "**RECOMMEND**" in two
    three = texts["03-escalate-oracle-DEMO-002.md"]
    assert "**ESCALATE**" in three and "**DBA**" in three and "E4" in three
    four = texts["04-multi-issue-DEMO-005.md"]
    assert "Issue 2 (secondary, decision escalate)" in four and "EUC-HW" in four


def test_committed_examples_cover_each_decision_path():
    folder = ROOT / "docs" / "examples"
    files = sorted(p.name for p in folder.glob("0*.md"))
    assert files == sorted(e["file"] for e in EXAMPLES)
    joined = "\n".join((folder / f).read_text(encoding="utf-8") for f in files)
    for path in ("**RECOMMEND**", "Decision **CLARIFY**", "**ESCALATE**", "Issue 2 (secondary"):
        assert path in joined


def test_audit_view_offers_worked_example(make_client, demo_tickets):
    client = make_client("l2")
    aid = client.analyze(demo_tickets["DEMO-001"]).details.analysis_id
    at = page("6_Audit_view.py", client)
    at.text_input(key="audit_id").input(aid).run()
    assert not at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert "⬇ Worked example (Markdown)" in labels
