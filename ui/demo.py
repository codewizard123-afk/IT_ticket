"""``make demo``: analyse DEMO-001…007 through the API and print the contract JSON.

Every response is validated against ``shared/contract/analysis_output.schema.json``; the exit
code is 1 if any response breaks the contract or the API cannot be reached. Gold labels
(``metadata.expected_*``) are stripped before sending, and are only used to print ✓/✗ against
architecture §16 at the end.

    python -m ui.demo                    # real API at API_BASE_URL
    MOCK_API=1 python -m ui.demo         # golden fixtures, no backend
    python -m ui.demo --quiet            # summary table only
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import jsonschema

from ui.client import ApiClient, ApiError
from ui.settings import ROOT

SCHEMA = ROOT / "shared" / "contract" / "analysis_output.schema.json"
GOLDEN = ROOT / "shared" / "fixtures" / "golden"
MOCK_DATA = ROOT / "ui" / "mock_data"


def load_demos() -> list[dict[str, Any]]:
    real = ROOT / "data" / "samples" / "sample_tickets.json"
    path = real if real.exists() else MOCK_DATA / "demo_tickets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def expected(demo_id: str) -> dict[str, Any] | None:
    path = GOLDEN / f"{demo_id}.json"
    if not path.exists():
        return None
    g = json.loads(path.read_text(encoding="utf-8"))
    return {
        "category": g["classification"]["category"],
        "subcategory": g["classification"]["subcategory"],
        "kb": g["kedb_match"]["error_code"],
        "decision": g["details"]["decision"],
        "esc": g["escalation_required"],
        "approval": g["human_approval_required"],
    }


def run(client: ApiClient, quiet: bool = False) -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    rows, failures = [], 0
    for ticket in load_demos():
        meta = {k: v for k, v in (ticket.get("metadata") or {}).items() if not k.startswith("exp")}
        ticket = {**ticket, "metadata": meta}
        try:
            full = client.analyze(ticket).model_dump(mode="json")
        except ApiError as err:
            print(f"{ticket.get('id')}: API error {err.status} {err.code}: {err.message}")
            return 1
        contract = {k: v for k, v in full.items() if k != "details"}
        try:
            jsonschema.validate(contract, schema)
            valid = True
        except jsonschema.ValidationError as err:
            valid, failures = False, failures + 1
            print(f"{contract['ticket_id']}: CONTRACT VIOLATION: {err.message}")
        if not quiet:
            print(f"\n=== {contract['ticket_id']} · {ticket['title']}")
            print(json.dumps(contract, indent=2))
        got = {
            "category": contract["classification"]["category"],
            "subcategory": contract["classification"]["subcategory"],
            "kb": contract["kedb_match"]["error_code"],
            "decision": full["details"]["decision"],
            "esc": contract["escalation_required"],
            "approval": contract["human_approval_required"],
        }
        exp = expected(contract["ticket_id"])
        rows.append((contract["ticket_id"], got, exp, valid))

    print(
        "\nTicket    Category/Subcategory          KB        Decision   Esc    Appr   §16  Schema"
    )
    for tid, got, exp, valid in rows:
        match = "-" if exp is None else ("✓" if got == exp else "✗")
        print(
            f"{tid:<9} {got['category'] + '/' + got['subcategory']:<29} {got['kb'] or '-':<9} "
            f"{got['decision']:<10} {str(got['esc']).lower():<6} {str(got['approval']).lower():<6} "
            f"{match:<4} {'ok' if valid else 'FAIL'}"
        )
    if client.mock:
        print("\n(MOCK_API=1: these are the golden fixtures, not live analysis.)")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the 7 demo tickets through the API")
    parser.add_argument("--role", default="l2", help="demo role (X-Demo-Role)")
    parser.add_argument("--quiet", action="store_true", help="summary table only")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = ApiClient.from_env(demo_role=args.role)
    sys.exit(run(client, quiet=args.quiet))


if __name__ == "__main__":
    main()
