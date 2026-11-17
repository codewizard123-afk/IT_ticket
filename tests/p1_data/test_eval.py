"""Evaluation (P1-T7): metric correctness (CI), split rules, and an end-to-end smoke run."""

from __future__ import annotations

import json
import math

import pytest

from eval import metrics as M
from ingest.split import assign_splits, metadata_split_ok, stratified_split
from store.config import ROOT


# --- metrics ---------------------------------------------------------------------------------
def test_accuracy_and_f1():
    y = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    assert M.accuracy(y, p) == 0.75
    prf = M.per_label_prf(y, p)
    assert prf["a"]["precision"] == 1.0 and prf["a"]["recall"] == 0.5
    assert math.isclose(prf["b"]["f1"], 0.8)
    assert math.isclose(M.macro_f1(y, p), (2 / 3 + 0.8) / 2)
    assert M.accuracy([], []) == 0.0


def test_confusion_and_pairs():
    m = M.confusion(["a", "b", "b"], ["a", "a", "z"], ["a", "b"])
    assert m == [[1, 0, 0], [1, 0, 1]]
    assert M.pair_confusions(["Oracle", "DB2"], ["DB2", "DB2"], [("Oracle", "DB2")]) == {
        "Oracle <-> DB2": 1
    }


def test_ranking_metrics():
    rels = [[False, True, False], [True, False, False], [False, False, False]]
    assert M.success_at_k(rels, 2) == pytest.approx(2 / 3)
    assert M.precision_at_k(rels, 2) == pytest.approx((0.5 + 0.5 + 0) / 3)
    assert M.mrr_at_k(rels, 3) == pytest.approx((0.5 + 1 + 0) / 3)
    assert M.ndcg_at_k([[True, False]], 2) == 1.0
    assert M.ndcg_at_k([[False, True]], 2, [1]) == pytest.approx(1 / math.log2(3))
    assert M.hit_at_k([["x", "y"], ["z"]], [{"y"}, {"q"}], 1) == 0.0
    assert M.hit_at_k([["x", "y"], ["z"]], [{"y"}, {"q"}], 2) == 0.5
    assert M.rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_calibration_and_thresholds():
    ece, bins = M.reliability([0.9, 0.9, 0.1, 0.1], [True, True, False, False], n_bins=10)
    assert ece == pytest.approx(0.1)
    assert [b["n"] for b in bins] == [2, 2]
    conf = [i / 100 for i in range(40, 100)]
    ok = [c >= 0.7 for c in conf]
    assert M.threshold_for_precision(conf, ok, 0.9) == 0.67  # 30/33 correct at >= 0.67
    bands = M.band_accuracy(["high", "low", "high"], [True, False, False])
    assert bands["high"] == {"n": 2, "accuracy": 0.5}


def test_priority_and_percentile():
    assert M.within_one(["High", "Low"], ["Critical", "High"]) == 0.5
    assert M.level_distance("Critical", "Low") == 3
    assert M.percentile([1, 2, 3, 4], 50) == 2.5
    assert M.percentile([], 95) == 0.0


# --- split -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def enriched():
    return json.loads((ROOT / "data/processed/tickets_enriched.json").read_text("utf-8"))


def test_metadata_split_used_and_keeps_dups_together(enriched):
    assert metadata_split_ok(enriched)
    splits = assign_splits(enriched)
    counts = {s: sum(v == s for v in splits.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 1055, "val": 225, "test": 220}
    groups: dict[str, set[str]] = {}
    for r in enriched:
        if r["dup_group_id"]:
            groups.setdefault(r["dup_group_id"], set()).add(splits[r["ticket_id"]])
    assert groups and all(len(v) == 1 for v in groups.values())


def test_stratified_fallback(enriched):
    splits = stratified_split(enriched, {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    by_sub: dict[str, dict[str, int]] = {}
    for r in enriched:
        d = by_sub.setdefault(r["subcategory"], {"train": 0, "val": 0, "test": 0})
        d[splits[r["ticket_id"]]] += 1
    for d in by_sub.values():
        assert 30 <= d["train"] <= 40 and d["val"] >= 4 and d["test"] >= 4
    for r in enriched:
        if r["dup_group_id"]:
            mates = [x for x in enriched if x["dup_group_id"] == r["dup_group_id"]]
            assert {splits[x["ticket_id"]] for x in mates} == {splits[r["ticket_id"]]}
            break
    assert stratified_split(enriched, {"train": 0.7, "val": 0.15, "test": 0.15}, 42) == splits


# --- end-to-end smoke ------------------------------------------------------------------------
def test_eval_smoke(tmp_path, hash_embed):
    pytest.importorskip("chromadb")
    from eval.run import evaluate
    from ingest.build_all import build_all
    from store.factory import open_stores
    from tests.p1_data.conftest import make_root

    root = make_root(tmp_path, per_subcategory=6)
    for rel in ("data/test", "eval/gold", "config"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    for rel in ("data/test/test_tickets.json", "eval/gold/gold_kb.json"):
        (root / rel).write_text((ROOT / rel).read_text("utf-8"), encoding="utf-8")
    build_all(root=root, embed_fn=hash_embed, eval_split=True)
    stores = open_stores(root, embed_fn=hash_embed, eval_split=True)
    m = evaluate("test", root=root, out_dir=tmp_path / "out", stores=stores, plots=False)
    assert (tmp_path / "out/metrics.json").exists() and (tmp_path / "out/errors.jsonl").exists()
    assert m["n_demo"] == 7 and m["n_test_set"] == 25
    assert 0 <= m["knn_baseline"]["split"]["subcategory_accuracy"] <= 1
    assert m["privacy"]["redaction_recall"] == 1.0
    assert m["privacy"]["error_code_false_positives"] == 0
    assert set(m["kb_ablations_reference_ranker"]) >= {"full", "no_bm25", "no_error_codes"}
    if "engine" in m:  # P2's engine is importable in this environment
        assert m["engine"]["traceability"]["citation_validity"] == 1.0
        assert m["engine"]["traceability"]["non_citable_citations"] == 0
