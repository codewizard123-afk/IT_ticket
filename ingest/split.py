"""Evaluation split (architecture §14.1): 70/15/15, stratified by subcategory, with every
exact-duplicate group kept inside one split.

The supplied corpus already carries a generator split in ``metadata.split``; with
``split.use_metadata_split: true`` (config/data.yaml) that split is used as long as it keeps
duplicate groups together, otherwise a seeded stratified split is computed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from store.config import data_config

SPLITS = ("train", "val", "test")


def _group_key(r: dict[str, Any]) -> str:
    return r.get("dup_group_id") or r["ticket_id"]


def metadata_split_ok(records: Sequence[dict[str, Any]]) -> bool:
    """Every record has a valid ``metadata.split`` and no duplicate group straddles splits."""
    seen: dict[str, str] = {}
    for r in records:
        s = (r.get("metadata") or {}).get("split")
        if s not in SPLITS:
            return False
        if seen.setdefault(_group_key(r), s) != s:
            return False
    return True


def stratified_split(
    records: Sequence[dict[str, Any]], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """``ticket_id → split``: per subcategory, shuffle duplicate groups and fill train/val/test."""
    rng = random.Random(seed)
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[r.get("subcategory") or ""][_group_key(r)].append(r["ticket_id"])
    out: dict[str, str] = {}
    for sub in sorted(groups):
        keys = sorted(groups[sub])
        rng.shuffle(keys)
        n = sum(len(groups[sub][k]) for k in keys)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        filled = 0
        for k in keys:
            split = "train" if filled < n_train else "val" if filled < n_train + n_val else "test"
            for tid in groups[sub][k]:
                out[tid] = split
            filled += len(groups[sub][k])
    return out


def assign_splits(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    cfg = data_config().get("split", {})
    if cfg.get("use_metadata_split", True) and metadata_split_ok(records):
        return {r["ticket_id"]: r["metadata"]["split"] for r in records}
    ratios = cfg.get("ratios") or {"train": 0.70, "val": 0.15, "test": 0.15}
    return stratified_split(records, ratios, int(cfg.get("seed", 42)))
