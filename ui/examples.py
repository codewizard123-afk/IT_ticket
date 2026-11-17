"""Worked examples (P4-T9): audit record → Markdown (input → classification → similar cases →
recommendation → evidence).

Used by the Audit view's "worked example" download and by ``make examples``, which regenerates
``docs/examples/`` from whichever API ``ui.client`` points at (``MOCK_API=1`` = golden fixtures)::

    python -m ui.examples --out docs/examples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ui.client import ApiClient
from ui.components.badges import chip_spec, citation_label
from ui.settings import ROOT

MOCK_DATA = Path(__file__).parent / "mock_data"


def _steps_md(steps: list[dict[str, Any]]) -> list[str]:
    out = []
    for n, s in enumerate(steps, 1):
        badges = " ".join(f"`[{citation_label(c)}]`" for c in s["citations"])
        tried = " ~~already tried~~" if s.get("already_attempted") else ""
        out.append(f"{n}. {s['text']} {badges} · {s.get('role', 'L1')} · {s['type']}{tried}")
    return out or ["_No grounded steps._"]


def _issue_md(title: str, contract: dict[str, Any], det: dict[str, Any]) -> list[str]:
    cls, pr = contract["classification"], contract["priority"]
    cd, pd = det.get("classification") or {}, det.get("priority") or {}
    lines = [
        f"### {title}",
        "",
        "**Classification**",
        "",
        f"- {cls['category']} / **{cls['subcategory']}**, confidence {cls['confidence']:.2f}"
        + (f" (band {cd['band']})" if cd.get("band") else ""),
    ]
    if cd.get("components"):
        lines.append(
            "- Components: " + ", ".join(f"{k} {v:g}" for k, v in cd["components"].items())
        )
    if cd.get("rationale"):
        lines.append(f"- Rationale: {cd['rationale']}")
    if cd.get("alternatives"):
        alts = ", ".join(
            f"{a['category']}/{a['subcategory']} ({a['score']:.2f})" for a in cd["alternatives"]
        )
        lines.append(f"- Alternatives: {alts}")
    lines += [
        "",
        "**Priority**",
        "",
        f"- **{pr['level']}**"
        + (f" ({pd['code']})" if pd.get("code") else "")
        + f": {pr['reason']}",
    ]
    for d in pd.get("drivers", []):
        lines.append(f"- Driver: {d}")
    km, kd = contract["kedb_match"], det.get("kedb")
    lines += ["", "**KEDB match**", ""]
    if km["found"] and kd:
        lines.append(f"- **{km['error_code']}**: {kd['title']} (score {kd['score']:.2f})")
        if kd.get("matched_error_code"):
            lines.append(f"- Matched code in the ticket: `{kd['matched_error_code']}`")
        lines.append(f"- Matched field: {kd.get('matched_field')}; why: {kd.get('why')}")
        if kd.get("excerpt"):
            lines.append(f"- Excerpt: “{kd['excerpt']}”")
        if kd.get("alternatives"):
            alts = ", ".join(f"{a['kb_id']} ({a['score']:.2f})" for a in kd["alternatives"])
            lines.append(f"- Alternatives: {alts}")
    else:
        lines.append("- No KEDB article above the relevance threshold.")
    lines += [
        "",
        f"**Recommended solution** (source: {contract['recommended_solution']['source']})",
        "",
        *_steps_md(det.get("steps", [])),
    ]
    return lines


def worked_example_md(
    record: dict[str, Any],
    *,
    title: str,
    summary: str,
    origin: str,
    clarified_from: dict[str, Any] | None = None,
) -> str:
    """One worked example from an audit record (``GET /audit/{analysis_id}``)."""
    resp, det = record["response"], record["response"]["details"]
    decision = det["decision"]
    lines = [
        f"# {title}",
        "",
        f"> {summary}",
        ">",
        f"> Source: {origin}. Analysis `{record['analysis_id']}`, rendered for role "
        f"`{record['filtered_for_role']}`. Audit hash chain: "
        + ("verified." if record.get("chain", {}).get("ok") else "**NOT verified**."),
        "",
        "## 1. Input",
        "",
        f"**{record['ticket_id']} · {record['ticket']['title']}**",
        "",
        "```text",
        record["ticket"]["description"],
        "```",
        "",
    ]
    if clarified_from:
        prev = clarified_from["response"]["details"]
        lines += [
            "## 2. First pass: the assistant asks",
            "",
            f"Decision **{prev['decision'].upper()}** (analysis `{clarified_from['analysis_id']}`, "
            f"band {prev['classification']['band']}). Questions:",
            "",
            *[f"- {q}" for q in prev.get("clarifying_questions", [])],
            "",
            "The analyst's answer is appended to the description as `[Clarification] …` and the "
            "ticket is re-analysed once (a second low-confidence result would escalate).",
            "",
        ]
    n = 3 if clarified_from else 2
    lines += [f"## {n}. Analysis", ""]
    lines += _issue_md("Issue 1 (primary)", resp, det)
    for i, sec in enumerate(det.get("secondary_issues", []), 2):
        sec_det = {
            "classification": sec.get("classification_details"),
            "priority": sec.get("priority_details"),
            "kedb": sec.get("kedb"),
            "steps": sec.get("steps", []),
        }
        lines += [""] + _issue_md(
            f"Issue {i} (secondary, decision {sec['decision']})", sec, sec_det
        )
    lines += ["", f"## {n + 1}. Similar cases", ""]
    sim_det = {s["incident_id"]: s for s in det.get("similar", [])}
    if resp["similar_incidents"]:
        lines += [
            "| Incident | Score | Tier | Origin | Why | Resolution |",
            "|---|---|---|---|---|---|",
        ]
        for s in resp["similar_incidents"]:
            d = sim_det.get(s["incident_id"], {})
            origin_label = chip_spec("origin", d.get("resolution_origin"))[0]
            res = (s["resolution"] or "-").replace("|", "/")
            lines.append(
                f"| {s['incident_id']} | {s['similarity_score']:.2f} | {d.get('tier', '-')} | "
                f"{origin_label} | {d.get('why', '-')} | {res} |"
            )
        lines += [
            "",
            "SYNTHETIC resolutions are shown for context only and are never cited as evidence.",
        ]
    else:
        lines.append("_None above the display threshold._")
    lines += ["", f"## {n + 2}. Decision", "", f"**{decision.upper()}**"]
    escs = [det.get("escalation")] + [s.get("escalation") for s in det.get("secondary_issues", [])]
    seen = set()
    for esc in [e for e in escs if e]:
        if esc.get("bundle_id") in seen:
            continue
        seen.add(esc.get("bundle_id"))
        reasons = "; ".join(f"{r['rule']}: {r['text']}" for r in esc["reasons"])
        lines.append(
            f"- Escalate issue {esc.get('issue_idx', 0) + 1} to **{esc['assignment_group']}** "
            f"(tier {esc['tier']}). Reasons: {reasons}"
        )
    lines += [
        f"- `escalation_required`: **{str(resp['escalation_required']).lower()}**",
        f"- `human_approval_required`: **{str(resp['human_approval_required']).lower()}**"
        + (f" ({'; '.join(det['approval_reasons'])})" if det.get("approval_reasons") else ""),
        "",
        f"## {n + 3}. Evidence trail (audit)",
        "",
        "| Kind | Ref | Rank | Score | Tier | Why |",
        "|---|---|---|---|---|---|",
    ]
    for e in record.get("evidence", []):
        score = f"{e['score']:.2f}" if isinstance(e.get("score"), int | float) else "-"
        lines.append(
            f"| {e['kind']} | {e['ref_id']} | {e['rank']} | {score} | {e.get('tier') or '-'} | "
            f"{e.get('why') or '-'} |"
        )
    lines += ["", "**Events**", ""]
    lines += [f"- `{e['ts_utc']}` {e['event']} ({e['mode']})" for e in record.get("timeline", [])]
    if record.get("decision"):
        d = record["decision"]
        lines += ["", f"**Analyst decision:** {d['final_decision']} ({d.get('mode')})"]
    lines += [
        "",
        "## Contract output",
        "",
        "```json",
        json.dumps({k: v for k, v in resp.items() if k != "details"}, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- generator
EXAMPLES = [
    {
        "file": "01-recommend-vpn-DEMO-001.md",
        "title": "Worked example 1: recommend (DEMO-001, VPN gateway not responding)",
        "summary": "An exact KEDB phrase code gives a confident match; the step the user already "
        "tried is greyed out; High priority means an analyst approves before applying.",
        "ticket": "DEMO-001",
    },
    {
        "file": "02-clarify-then-recommend.md",
        "title": "Worked example 2: clarify → recommend (vague VPN ticket)",
        "summary": "A two-word ticket is too sparse to act on, so the assistant asks one round of "
        "questions; the answer brings a KEDB match and a cited recommendation.",
        "ticket": {
            "title": "VPN broken",
            "description": "VPN broken",
            "metadata": {"source": "ui"},
        },
        "answer": "VPN client says gateway not responding when working from home",
    },
    {
        "file": "03-escalate-oracle-DEMO-002.md",
        "title": "Worked example 3: escalate (DEMO-002, Oracle month-end slowdown)",
        "summary": "15 users and a blocked month-end close make it Critical, a hard rule (E4), so "
        "the ticket goes to the DBA team at L3 with a diagnostic bundle.",
        "ticket": "DEMO-002",
    },
    {
        "file": "04-multi-issue-DEMO-005.md",
        "title": "Worked example 4: multi-issue (DEMO-005, VPN drops + overheating)",
        "summary": "The splitter separates two issues: the VPN issue gets a recommendation, while "
        "the overheating issue hits a KB escalation clause (E6) and escalates to EUC-HW.",
        "ticket": "DEMO-005",
    },
]


def generate(client: ApiClient, out_dir: Path, origin: str) -> list[Path]:
    demos = {
        t["id"]: t
        for t in json.loads((MOCK_DATA / "demo_tickets.json").read_text(encoding="utf-8"))
    }
    real = ROOT / "data" / "samples" / "sample_tickets.json"
    if real.exists():
        demos.update({t["id"]: t for t in json.loads(real.read_text(encoding="utf-8"))})
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ex in EXAMPLES:
        ticket = demos[ex["ticket"]] if isinstance(ex["ticket"], str) else ex["ticket"]
        ticket = {
            **ticket,
            "metadata": {
                k: v for k, v in (ticket.get("metadata") or {}).items() if not k.startswith("exp")
            },
        }
        resp = client.analyze(ticket)
        clarified_from = None
        if ex.get("answer"):
            clarified_from = client.audit(resp.details.analysis_id)
            resp = client.clarify(resp.ticket_id, resp.details.analysis_id, ex["answer"])
        record = client.audit(resp.details.analysis_id)
        md = worked_example_md(
            record,
            title=ex["title"],
            summary=ex["summary"],
            origin=origin,
            clarified_from=clarified_from,
        )
        path = out_dir / ex["file"]
        path.write_text(md, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "examples")
    parser.add_argument("--role", default="l2", help="demo role the examples are rendered for")
    args = parser.parse_args(argv)
    client = ApiClient.from_env(demo_role=args.role)
    origin = (
        "**mock API** (`MOCK_API=1`, Phase 0 golden fixtures; scores are illustrative). "
        "Regenerate with `make examples` against the real API"
        if client.mock
        else f"real API at `{client.settings.api_base_url}`"
    )
    for p in generate(client, args.out, origin):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
