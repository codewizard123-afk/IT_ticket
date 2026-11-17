"""P4-T10: `make demo` runs all 7 demos and checks the contract."""

from __future__ import annotations

from ui.demo import run


def test_demo_passes_on_mock(make_client, capsys):
    assert run(make_client("l2"), quiet=True) == 0
    out = capsys.readouterr().out
    for i in range(1, 8):
        assert f"DEMO-00{i}" in out
    assert out.count("✓") == 7 and "FAIL" not in out


def test_demo_prints_contract_json(make_client, capsys):
    run(make_client("l2"), quiet=False)
    out = capsys.readouterr().out
    assert '"human_approval_required"' in out and '"details"' not in out
