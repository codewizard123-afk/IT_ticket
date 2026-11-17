"""``python -m ingest.make_seed`` — ``data/seed/curated_incidents.yaml`` →
``data/seed/resolved_incidents.json`` (architecture §6.3.2, P1-T4).

The YAML is the human-edited source (title, description, resolution, KB ids, review). This
script only adds the mechanical fields (IDs ``INC-H0001…``, seeded timestamps, assignment group,
``resolution_origin=curated_verified``) and validates every record:

* ``TicketRecord`` schema, taxonomy from ``config/taxonomy_map.yaml``
* every ``kb_ids`` entry exists in the KEDB
* resolutions pass the KB safety rules (same blocklist as ``ai/validate.py``, §7.8)
* every record has ``verified_by``
"""

from __future__ import annotations

import argparse
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shared.schemas import TicketRecord
from store.config import CONFIG_DIR, ROOT, data_config, load_yaml, read_json, repo_path, write_json

SEED_SOURCE = "data/seed/curated_incidents.yaml"
UNSAFE = re.compile(
    r"disable (the )?(antivirus|firewall|uac|defender|endpoint protection)"
    r"|share (your |their |the )?password"
    r"|format (the )?(drive|disk)"
    r"|delete .*registry"
    r"|grant .*(dba|owner|full control|admin)",
    re.IGNORECASE,
)
DURATION_H = {"P1": 2, "P2": 4, "P3": 8, "P4": 24}


def taxonomy_map() -> dict[str, dict[str, dict[str, list[str]]]]:
    """``config/taxonomy_map.yaml`` ``mapping``: category → subcategory → primary/secondary."""
    raw = load_yaml(CONFIG_DIR / "taxonomy_map.yaml")
    return raw.get("mapping") or raw.get("map") or {}


def taxonomy(root: Path | None = None) -> dict[str, list[str]]:
    """``{category: [subcategory, …]}`` from P1's taxonomy map (the 6x5 ticket taxonomy).
    Config always comes from the repo's ``config/``; ``root`` is accepted for symmetry."""
    raw = load_yaml(CONFIG_DIR / "taxonomy_map.yaml")
    mapping = raw.get("mapping") or raw.get("map") or {}
    return {cat: list(subs) for cat, subs in mapping.items()}


def unsafe(text: str) -> bool:
    return bool(UNSAFE.search(text))


def build(src: dict[str, Any], kb_ids: set[str], tax: dict[str, list[str]]) -> list[dict]:
    rng = random.Random(int(src.get("seed", 42)))
    groups = data_config().get("assignment_groups", {})
    start = datetime(2026, 6, 1, tzinfo=UTC)
    out: list[dict] = []
    for i, inc in enumerate(src["incidents"], 1):
        cat, sub = inc["category"], inc["subcategory"]
        if sub not in tax.get(cat, []):
            raise ValueError(f"#{i}: {cat}/{sub} is not in the taxonomy map")
        missing = [k for k in inc["kb_ids"] if k not in kb_ids]
        if missing:
            raise ValueError(f"#{i}: unknown KB ids {missing}")
        if unsafe(inc["resolution"]):
            raise ValueError(f"#{i}: resolution violates the KB safety rules")
        prio = inc.get("priority", "P3")
        opened = start + timedelta(
            days=rng.randint(0, 150), hours=rng.randint(8, 17), minutes=rng.randint(0, 59)
        )
        resolved = opened + timedelta(hours=DURATION_H[prio] * rng.uniform(0.5, 1.5))
        rec = TicketRecord.model_validate(
            {
                "ticket_id": f"INC-H{i:04d}",
                "title": inc["title"].strip(),
                "description": " ".join(inc["description"].split()),
                "category": cat,
                "subcategory": sub,
                "priority": prio,
                "status": "resolved",
                "source": "curated_team",
                "resolution": " ".join(inc["resolution"].split()),
                "resolution_origin": "curated_verified",
                "resolution_kb_id": inc["kb_ids"][0] if inc["kb_ids"] else None,
                "opened_at": opened.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "resolved_at": resolved.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "opened_ts": int(opened.timestamp()),
                "resolved_ts": int(resolved.timestamp()),
                "assignment_group": inc.get("assignment_group") or groups.get(cat),
                "application": inc.get("application") or f"{cat} general",
                "environment": inc.get("environment", "office"),
                "restricted": False,
                "verified": True,
                "verified_by": inc.get("verified_by") or src["verified_by"],
                "collection": "tickets_verified",
                "derived_fields": ["opened_at", "resolved_at"],
                "metadata": {
                    "source": "curated_team",
                    "kb_ids": list(inc["kb_ids"]),
                    "fills_gap": bool(inc.get("fills_gap", False)),
                    "drafted_by": src.get("drafted_by"),
                    "review_status": inc.get("review_status") or src.get("review_status"),
                },
            }
        )
        if not rec.verified_by:
            raise ValueError(f"#{i}: verified_by is required")
        out.append(rec.model_dump(mode="json"))
    return out


def run(root: Path | None = None) -> Path:
    from ingest.index_kedb import load_articles

    src = load_yaml((root or ROOT) / SEED_SOURCE)
    kb = {a.kb_id for a in load_articles(repo_path("kb_articles", root))}
    out = repo_path("seed_incidents", root)
    write_json(out, build(src, kb, taxonomy(root)))
    return out


def load_seed(root: Path | None = None) -> list[TicketRecord]:
    path = repo_path("seed_incidents", root)
    if not path.exists():
        return []
    return [TicketRecord.model_validate(r) for r in read_json(path)]


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(f"wrote {run()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
