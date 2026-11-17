"""``python -m ingest.enrich`` — ``tickets.json`` → ``data/processed/tickets_enriched.json``
(architecture §6.3.1). ``tickets.json`` itself is never modified.

The supplied corpus (``it_tickets_synth_v1``) is already synthetic and carries timestamps,
assignment group, application, environment and a resolution in ``metadata``. Enrichment
therefore *normalises* rather than invents:

* lifts those fields to the top level in ``TicketRecord`` form and lists them in
  ``derived_fields`` (they are generator output, not observed ITSM data);
* dates → ISO-8601 UTC strings + int epoch ``opened_ts``/``resolved_ts`` (Chroma ``$gte``);
* ``resolution_origin``: the generator's ``synthetic_generated`` is mapped to
  ``synthetic_escalated`` (escalated tickets) or ``synthetic_kb_derived`` — **never cited**;
  the raw value is kept as ``metadata.resolution_origin_raw``;
* ``restricted``: a seeded share of the configured subcategories (access-control demo, §8.6);
* ``dup_group_id``: generator ``duplicate_of`` links ∪ exact normalised-text duplicates;
* ``text_hash`` over the **redacted** embedding text (matches what the engine sees).

Deterministic: seeded RNG (``config/data.yaml: enrich.seed``).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from privacy.redact import redact_text
from shared.schemas import TicketRecord
from store.config import data_config, read_json, repo_path, write_json
from store.text import embedding_text, text_hash


def parse_ts(value: str | None) -> tuple[str | None, int | None]:
    """``"2026-05-01T06:42"`` (naive = UTC) → ``("2026-05-01T06:42:00Z", epoch)``."""
    if not value:
        return None, None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ"), int(dt.timestamp())


def map_origin(raw: dict[str, Any], cfg: dict[str, Any]) -> str:
    if not raw.get("resolution"):
        return "none"
    omap = cfg.get("origin_map") or {}
    if (raw.get("metadata") or {}).get("escalated"):
        return omap.get("escalated", "synthetic_escalated")
    return omap.get("default", "synthetic_kb_derived")


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def dup_groups(rows: Sequence[dict[str, Any]], hashes: dict[str, str]) -> dict[str, str]:
    """``ticket_id → "DUP-<smallest id>"`` for every ticket in a group of ≥ 2."""
    uf = _UnionFind()
    ids = {r["id"] for r in rows}
    for r in rows:
        uf.find(r["id"])
        dup_of = (r.get("metadata") or {}).get("duplicate_of")
        if dup_of and dup_of in ids:
            uf.union(r["id"], dup_of)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for tid, h in hashes.items():
        by_hash[h].append(tid)
    for members in by_hash.values():
        for other in members[1:]:
            uf.union(members[0], other)
    groups: dict[str, list[str]] = defaultdict(list)
    for tid in ids:
        groups[uf.find(tid)].append(tid)
    return {tid: f"DUP-{root}" for root, ms in groups.items() if len(ms) > 1 for tid in ms}


def pick_restricted(rows: Sequence[dict[str, Any]], cfg: dict[str, Any], seed: int) -> set[str]:
    rcfg = cfg.get("restricted") or {}
    rate = float(rcfg.get("rate", 0.02))
    rng = random.Random(seed)
    chosen: set[str] = set()
    for sub in rcfg.get("subcategories") or []:
        pool = sorted(r["id"] for r in rows if r.get("subcategory") == sub)
        if pool:
            chosen.update(rng.sample(pool, max(1, round(rate * len(pool)))))
    return chosen


def enrich(rows: Sequence[dict[str, Any]], cfg: dict[str, Any] | None = None) -> list[dict]:
    """Enrich raw ``tickets.json`` rows. Returns JSON-ready dicts (``TicketRecord`` schema)."""
    cfg = cfg if cfg is not None else data_config().get("enrich", {})
    seed = int(cfg.get("seed", 42))
    groups_cfg = data_config().get("assignment_groups", {})
    derived = list(cfg.get("derived_fields") or [])

    redacted = {
        r["id"]: embedding_text(redact_text(r.get("title") or ""), redact_text(r["description"]))
        for r in rows
    }
    hashes = {tid: text_hash(t) for tid, t in redacted.items()}
    dups = dup_groups(rows, hashes)
    restricted = pick_restricted(rows, cfg, seed)

    out: list[dict] = []
    for r in rows:
        meta = dict(r.get("metadata") or {})
        opened_at, opened_ts = parse_ts(meta.get("opened_at"))
        resolved_at, resolved_ts = parse_ts(meta.get("resolved_at"))
        origin = map_origin(r, cfg)
        meta["resolution_origin_raw"] = meta.get("resolution_origin")
        rec = TicketRecord.model_validate(
            {
                "ticket_id": r["id"],
                "title": r.get("title") or "",
                "description": r.get("description") or "",
                "category": r.get("category"),
                "subcategory": r.get("subcategory"),
                "priority": r.get("priority"),
                "status": r.get("status") or "closed",
                "source": meta.get("source"),
                "resolution": r.get("resolution"),
                "resolution_origin": origin,
                "resolution_kb_id": meta.get("related_kb_id"),
                "opened_at": opened_at,
                "resolved_at": resolved_at,
                "opened_ts": opened_ts,
                "resolved_ts": resolved_ts,
                "assignment_group": meta.get("assignment_group")
                or groups_cfg.get(r.get("category"), "SERVICE-DESK"),
                "application": meta.get("application") or f"{r.get('category')} general",
                "environment": meta.get("environment") or "unspecified",
                "location": meta.get("location"),
                "restricted": r["id"] in restricted,
                "verified": False,
                "collection": "tickets_hist",
                "derived_fields": derived,
                "dup_group_id": dups.get(r["id"]),
                "text_hash": hashes[r["id"]],
                "metadata": meta,
            }
        )
        out.append(rec.model_dump(mode="json"))
    return out


def load_enriched(root: Path | None = None) -> list[TicketRecord]:
    return [TicketRecord.model_validate(r) for r in read_json(repo_path("tickets_enriched", root))]


def run(root: Path | None = None) -> Path:
    rows = read_json(repo_path("tickets", root))
    out = repo_path("tickets_enriched", root)
    write_json(out, enrich(rows))
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    path = run()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
