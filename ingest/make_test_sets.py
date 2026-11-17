"""``python -m ingest.make_test_sets`` (``make test-sets``) — build the test, trial and gold files
from ``data/seed/test_seed.yaml`` (architecture §6.3.3, §9.3, §14.1; P1-T5).

Outputs:

* ``data/samples/sample_tickets.json`` — the 7 demos in the input contract format
* ``data/test/test_tickets.json``      — demos + edge cases, gold in ``metadata.expected_*``
* ``data/test/trial_tickets.json``     — T01…T12 **without** gold (safe for ``/trial/tickets``)
* ``data/test/trial_gold.json``        — trial gold, never sent to the UI
* ``eval/gold/gold_kb.json``           — KB gold: demos + test set + sampled test-split tickets

Every team-created ticket carries ``metadata.source = "synthetic_team"`` (demos: ``"demo"``).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ingest.make_seed import taxonomy
from shared.constants import DECISIONS, PRIORITY_LEVELS
from store.config import ROOT, data_config, load_yaml, read_json, repo_path, write_json

EDGE_CATEGORIES = {
    "gap": 4,
    "out_of_taxonomy": 3,
    "vague": 3,
    "contradictory": 2,
    "security": 3,
    "keyword_trap": 3,
    "pii": 2,
    "pii_secret": 1,
    "prompt_injection": 2,
    "multi_issue": 2,
}
SAMPLED_SPLIT_TICKETS = 40


def _text(s: str) -> str:
    return " ".join(str(s).split())


def validate(src: dict[str, Any], kb_ids: set[str], tax: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for section in ("demos", "test", "trial"):
        for t in src[section]:
            tid = t["id"]
            if tid in seen:
                raise ValueError(f"duplicate id {tid}")
            seen.add(tid)
            g = t["gold"]
            cat, sub = g.get("category"), g.get("subcategory")
            if cat is not None and cat not in tax:
                raise ValueError(f"{tid}: unknown category {cat!r}")
            if sub is not None and sub not in tax.get(cat, []):
                raise ValueError(f"{tid}: {sub!r} is not a subcategory of {cat!r}")
            if g["priority"] not in PRIORITY_LEVELS:
                raise ValueError(f"{tid}: bad priority {g['priority']!r}")
            if g["decision"] not in DECISIONS:
                raise ValueError(f"{tid}: bad decision {g['decision']!r}")
            sec = g.get("secondary") or {}
            unknown = [k for k in [*g["kb_ids"], *sec.get("kb_ids", [])] if k not in kb_ids]
            if unknown:
                raise ValueError(f"{tid}: unknown KB ids {unknown}")
    demo_text = {_text(t["description"]) for t in src["demos"]}
    if any(_text(t["description"]) in demo_text for t in src["trial"]):
        raise ValueError("trial tickets must not overlap with the demos")


def as_input(t: dict[str, Any], source: str, with_gold: bool) -> dict[str, Any]:
    """A ticket in the ``sample_tickets.json`` input format (architecture §2.2)."""
    meta: dict[str, Any] = {"source": source}
    if t.get("edge"):
        meta["edge_case"] = t["edge"]
    if with_gold:
        g = t["gold"]
        meta.update(
            {
                "expected_category": g.get("category"),
                "expected_subcategory": g.get("subcategory"),
                "expected_priority": g["priority"],
                "expected_kb_ids": list(g["kb_ids"]),
                "expected_decision": g["decision"],
            }
        )
        if g.get("secondary"):
            meta["expected_secondary"] = g["secondary"]
    return {
        "id": t["id"],
        "title": _text(t["title"]),
        "description": _text(t["description"]),
        "category": None,
        "subcategory": None,
        "priority": None,
        "status": "new",
        "resolution": None,
        "metadata": meta,
    }


def sample_split_tickets(enriched: list[dict[str, Any]], n: int, seed: int) -> list[dict]:
    """~``n`` test-split tickets stratified by subcategory (1 each, then seeded extras),
    skipping duplicates so a group is never sampled twice."""
    rng = random.Random(seed)
    pool = [
        r
        for r in enriched
        if r["metadata"].get("split") == "test" and not r["metadata"].get("duplicate_of")
    ]
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["ticket_id"]):
        by_sub[r["subcategory"]].append(r)
    picked = [rng.choice(by_sub[s]) for s in sorted(by_sub)]
    rest = [r for r in pool if r not in picked]
    picked += rng.sample(sorted(rest, key=lambda r: r["ticket_id"]), max(0, n - len(picked)))
    return sorted(picked, key=lambda r: r["ticket_id"])


def gold_entry(t: dict[str, Any], set_name: str, label_source: str) -> dict[str, Any]:
    g = t["gold"]
    return {
        "ticket_id": t["id"],
        "set": set_name,
        "category": g.get("category"),
        "subcategory": g.get("subcategory"),
        "kb_ids": list(g["kb_ids"]),
        "accept_escalate": not g["kb_ids"] or g["decision"] == "escalate",
        "decision": g["decision"],
        "label_source": label_source,
    }


def build(root: Path | None = None) -> dict[str, Any]:
    """All output objects keyed by their path key (pure; ``run`` writes them)."""
    from ingest.index_kedb import load_articles

    src = load_yaml(repo_path("test_seed", root))
    kb_ids = {a.kb_id for a in load_articles(repo_path("kb_articles", root))}
    validate(src, kb_ids, taxonomy(root))
    seed = int(data_config().get("split", {}).get("seed", 42))

    demos = [as_input(t, "demo", True) for t in src["demos"]]
    tests = [as_input(t, "synthetic_team", True) for t in src["test"]]
    trial = [as_input(t, "synthetic_team", False) for t in src["trial"]]
    trial_gold = {
        t["id"]: {
            "category": t["gold"]["category"],
            "subcategory": t["gold"]["subcategory"],
            "priority": t["gold"]["priority"],
            "kb_ids": list(t["gold"]["kb_ids"]),
            "decision": t["gold"]["decision"],
        }
        for t in src["trial"]
    }

    gold = [gold_entry(t, "demo", "team") for t in src["demos"]]
    gold += [gold_entry(t, "test", "team") for t in src["test"]]
    enriched_path = repo_path("tickets_enriched", root)
    if enriched_path.exists():
        for r in sample_split_tickets(read_json(enriched_path), SAMPLED_SPLIT_TICKETS, seed):
            kb = r["metadata"].get("related_kb_id")
            gold.append(
                {
                    "ticket_id": r["ticket_id"],
                    "set": "test_split",
                    "category": r["category"],
                    "subcategory": r["subcategory"],
                    "kb_ids": [kb] if kb else [],
                    "accept_escalate": not kb,
                    "decision": None,
                    "label_source": "generator_scenario_link",
                }
            )
    return {
        "samples": demos,
        "test_tickets": demos + tests,
        "trial_tickets": trial,
        "trial_gold": trial_gold,
        "gold_kb": {"version": 1, "items": gold},
    }


def run(root: Path | None = None) -> list[Path]:
    out = []
    for key, obj in build(root).items():
        path = repo_path(key, root)
        write_json(path, obj)
        out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for p in run():
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
