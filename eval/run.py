"""``python -m eval.run --split test`` (``make eval``) — the P1 evaluation (architecture §14).

Runs against the ``*_eval`` collections (built from the 70% train split, so test tickets are
never their own neighbours) and writes:

* ``eval/out/metrics.json``   every metric in §14.2 that can be measured offline, + ablations
* ``eval/out/errors.jsonl``   misclassifications and KB misses, for error analysis
* ``eval/out/confusion_category.png``, ``confusion_subcategory.png``, ``reliability.png``
* ``eval/out/tuning.json``    with ``--split val``: recommended bands / kNN k for P2

Systems evaluated:

* **engine** — P2's ``ai.engine`` (fused classifier, hybrid search, KB ranking, synthesis,
  validation), run offline exactly as the API would with ``GROQ_API_KEY`` unset.
* **knn baseline** — P1's taxonomy-agnostic kNN over the train split (the "kNN-only" ablation).

Decision accuracy needs P3's decision gate; it is reported as pending until ``backend``
exposes it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval import metrics as M
from ingest.make_seed import taxonomy, taxonomy_map
from privacy.redact import redact, redact_text
from shared.constants import CITABLE_ORIGINS, PRIORITY_LEVEL
from store.config import ROOT, read_json, repo_path

log = logging.getLogger("eval.run")

FOCUS_PAIRS = (
    ("Oracle", "DB2", "MS SQL Server", "PostgreSQL", "MySQL"),
    ("VPN", "Wireless"),
    ("CPU", "Laptop"),
    ("Email", "Internal Application"),
    ("Password Reset", "Account Lockout"),
)
TARGETS = {
    "category_accuracy": 0.90,
    "subcategory_macro_f1": 0.80,
    "duplicate_recall_at_5": 0.85,
    "kb_hit_at_3": 0.85,
    "high_band_accuracy": 0.90,
    "citation_validity": 1.0,
    "redaction_recall": 0.95,
    "latency_p95_ms": 8000,
}
KB_WEIGHTS = {"sem": 0.45, "bm25": 0.15, "code": 0.20, "tax": 0.12}
CODE_RE = re.compile(
    r"ORA-\d{5}|0x[0-9a-fA-F]{8}|SQL\d{4,5}N|SQLSTATE \w{5}|(?:SQL Server )?error \d{3,5}",
    re.IGNORECASE,
)


# =============================================================================================
# Data
# =============================================================================================
@dataclass
class Item:
    ticket_id: str
    title: str
    description: str
    set: str  # split | demo | test
    category: str | None = None
    subcategory: str | None = None
    priority_level: str | None = None
    kb_ids: list[str] = field(default_factory=list)
    accept_escalate: bool = False
    decision: str | None = None
    edge: str | None = None
    dup_group_id: str | None = None

    @property
    def text(self) -> str:
        from store.text import embedding_text

        return embedding_text(redact_text(self.title), redact_text(self.description))


def load_items(split: str, root: Path | None = None, limit: int | None = None) -> list[Item]:
    enriched = read_json(repo_path("tickets_enriched", root))
    gold_kb = {g["ticket_id"]: g for g in read_json(repo_path("gold_kb", root))["items"]}
    items = []
    for r in enriched:
        if r["metadata"].get("split") != split:
            continue
        kb = r["metadata"].get("related_kb_id")
        items.append(
            Item(
                ticket_id=r["ticket_id"],
                title=r["title"],
                description=r["description"],
                set="split",
                category=r["category"],
                subcategory=r["subcategory"],
                priority_level=PRIORITY_LEVEL.get(r["priority"]),
                kb_ids=[kb] if kb else [],
                accept_escalate=not kb,
                dup_group_id=r.get("dup_group_id"),
            )
        )
    if limit:
        items = items[:limit]
    for t in read_json(repo_path("test_tickets", root)):
        m = t["metadata"]
        g = gold_kb.get(t["id"], {})
        items.append(
            Item(
                ticket_id=t["id"],
                title=t["title"],
                description=t["description"],
                set="demo" if m["source"] == "demo" else "test",
                category=m.get("expected_category"),
                subcategory=m.get("expected_subcategory"),
                priority_level=m.get("expected_priority"),
                kb_ids=list(m.get("expected_kb_ids") or []),
                accept_escalate=bool(g.get("accept_escalate")),
                decision=m.get("expected_decision"),
                edge=m.get("edge_case"),
            )
        )
    return items


# =============================================================================================
# Systems
# =============================================================================================
def knn_predict(ts: Any, text: str, k: int, power: float = 1.0) -> tuple[str, str, float]:
    """Similarity-weighted kNN vote over ``tickets_hist`` (architecture §7.3 voter A)."""
    hits = ts.query(text, k, collections=("tickets_hist",))
    votes: dict[tuple[str, str], float] = defaultdict(float)
    for h in hits:
        votes[(h.category, h.subcategory)] += max(h.score, 0.0) ** power
    if not votes:
        return "", "", 0.0
    total = sum(votes.values())
    (cat, sub), best = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
    return cat, sub, best / total


def detect_codes(text: str, phrases: Sequence[str]) -> list[str]:
    low = text.lower()
    codes = [p for p in phrases if len(p) >= 4 and p.lower() in low]
    return codes + [m.group(0) for m in CODE_RE.finditer(text)]


def kb_rank(
    text: str,
    kb: Any,
    tax_boost: dict[str, float],
    phrases: Sequence[str],
    use: frozenset[str] = frozenset(KB_WEIGHTS),
) -> list[tuple[str, float]]:
    """P1's reference KB ranker for the retrieval ablations (same formula as §7.6, minus fb)."""
    sem: dict[str, float] = {}
    for h in kb.query_chunks(text, 30):
        sem[h.kb_id] = max(sem.get(h.kb_id, 0.0), h.score)
    bm: dict[str, float] = {}
    if "bm25" in use:
        raw = kb.bm25(text, 30)
        top = raw[0][1] if raw else 1.0
        for cid, s in raw:
            kid = cid.split("#")[0]
            bm[kid] = max(bm.get(kid, 0.0), s / top)
    code: set[str] = set()
    if "code" in use:
        for ids in kb.lookup_codes(detect_codes(text, phrases)).values():
            code.update(ids)
    cands = set(sem) | set(bm) | code | (set(tax_boost) if "tax" in use else set())
    scored = {
        k: KB_WEIGHTS["sem"] * sem.get(k, 0.0)
        + KB_WEIGHTS["bm25"] * bm.get(k, 0.0)
        + KB_WEIGHTS["code"] * (k in code)
        + (KB_WEIGHTS["tax"] * tax_boost.get(k, 0.0) if "tax" in use else 0.0)
        for k in cands
    }
    return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))


def tax_boost_for(category: str, subcategory: str, tmap: dict[str, Any]) -> dict[str, float]:
    entry = (tmap.get(category) or {}).get(subcategory) or {}
    boost = {k: 0.5 for k in entry.get("secondary", [])}
    boost.update({k: 1.0 for k in entry.get("primary", [])})
    return boost


def engine_runner(ts: Any, kb: Any):  # noqa: ANN201
    """Return ``run(item) -> EngineResult`` for P2's engine, or ``None`` if it isn't there."""
    try:
        from ai.engine import AIEngine
        from ai.llm.groq_client import make_llm_client
    except ImportError as exc:  # P2 not merged / deps missing
        log.warning("ai.engine unavailable (%s): engine metrics skipped", exc)
        return None
    from shared.schemas import EngineContext, SanitizedTicket

    engine = AIEngine()
    llm = make_llm_client()

    def run(item: Item):  # noqa: ANN202
        red_t, red_d = redact(item.title), redact(item.description)
        ticket = SanitizedTicket(
            ticket_id=item.ticket_id,
            title=red_t.text,
            description=red_d.text,
            secret_detected=red_t.secret_detected or red_d.secret_detected,
        )
        ctx = EngineContext(role="admin", ticket_store=ts, kb_store=kb, llm=llm)
        return engine.run(ticket, ctx)

    return run


# =============================================================================================
# Evaluation sections
# =============================================================================================
def classification_metrics(
    gold: Sequence[Item], preds: Sequence[tuple[str, str]], subs: Sequence[str]
) -> dict[str, Any]:
    yc = [i.category for i in gold]
    ys = [i.subcategory for i in gold]
    pc = [p[0] for p in preds]
    ps = [p[1] for p in preds]
    return {
        "n": len(gold),
        "category_accuracy": M.accuracy(yc, pc),
        "category_macro_f1": M.macro_f1(yc, pc, sorted(set(yc) | set(pc), key=str)),
        "subcategory_accuracy": M.accuracy(ys, ps),
        "subcategory_macro_f1": M.macro_f1(ys, ps, sorted(set(ys) | set(ps), key=str)),
        "focus_pair_confusions": M.pair_confusions(ys, ps, FOCUS_PAIRS),
        "label_space": len(subs),
    }


def retrieval_metrics(items: Sequence[Item], ts: Any, k_dense: int = 30) -> dict[str, Any]:
    """Similar-ticket retrieval on the split (relevance = same subcategory)."""
    out: dict[str, dict[str, list]] = {
        m: {"rels": [], "dup": []} for m in ("dense", "bm25", "hybrid")
    }
    cat_of: dict[str, tuple[str, str]] = {}
    trap = []
    dup_targets = 0
    for it in items:
        text = it.text
        dense = ts.query(text, k_dense, collections=("tickets_hist",))
        for h in dense:
            cat_of[h.ticket_id] = (h.category, h.subcategory)
        bm = [tid for tid, _ in ts.bm25(text, k_dense) if not tid.startswith("INC-H")]
        missing = [t for t in bm if t not in cat_of]
        for rec in ts.get(missing):
            cat_of[rec.ticket_id] = (rec.category, rec.subcategory)
        rankings = {
            "dense": [h.ticket_id for h in dense],
            "bm25": bm,
            "hybrid": M.rrf([[h.ticket_id for h in dense], bm]),
        }
        same_group = {
            h.ticket_id
            for h in dense
            if it.dup_group_id and h.metadata.get("dup_group_id") == it.dup_group_id
        }
        if it.dup_group_id:
            recs = ts.list(where={"dup_group_id": it.dup_group_id}, collections=("tickets_hist",))
            same_group |= {r.ticket_id for r in recs}
        for name, ranked in rankings.items():
            rel = [cat_of.get(t, ("", ""))[1] == it.subcategory for t in ranked[:10]]
            out[name]["rels"].append(rel)
            if same_group:
                out[name]["dup"].append(bool(set(ranked[:5]) & same_group))
        if same_group:
            dup_targets += 1
        trap.extend(cat_of.get(t, ("", ""))[0] != it.category for t in bm[:5])
    res: dict[str, Any] = {"n": len(items), "duplicate_queries": dup_targets}
    for name, d in out.items():
        res[name] = {
            "precision_at_5": M.precision_at_k(d["rels"], 5),
            "success_at_5": M.success_at_k(d["rels"], 5),
            "mrr_at_10": M.mrr_at_k(d["rels"], 10),
            "ndcg_at_10": M.ndcg_at_k(d["rels"], 10, [10] * len(d["rels"])),
            "duplicate_recall_at_5": (sum(d["dup"]) / len(d["dup"])) if d["dup"] else None,
        }
    res["keyword_trap_rate_bm25_top5"] = sum(trap) / len(trap) if trap else 0.0
    return res


def duplicate_metrics(full_ts: Any, root: Path | None = None, k: int = 5) -> dict[str, Any]:
    """Duplicate retrieval on the full ``tickets_hist`` index: each generator duplicate
    (``metadata.duplicate_of``) must retrieve its original in the top ``k`` (itself excluded).
    Measured on the full index because duplicate groups never straddle the eval split."""
    from store.text import embedding_text

    rows = [
        r
        for r in read_json(repo_path("tickets_enriched", root))
        if r["metadata"].get("duplicate_of")
    ]
    rels, scen = [], []
    for r in rows:
        text = embedding_text(redact_text(r["title"]), redact_text(r["description"]))
        hits = full_ts.query(text, k + 1, collections=("tickets_hist",))
        hits = [h for h in hits if h.ticket_id != r["ticket_id"]][:k]
        rels.append([h.ticket_id == r["metadata"]["duplicate_of"] for h in hits])
        scen.append(
            [h.metadata.get("scenario_id") == r["metadata"].get("scenario_id") for h in hits]
        )
    return {
        "n": len(rows),
        "recall_at_5": M.success_at_k(rels, 5),
        "mrr_at_5": M.mrr_at_k(rels, 5),
        "same_scenario_recall_at_5": M.success_at_k(scen, 5),
        "same_scenario_precision_at_5": M.precision_at_k(scen, 5),
    }


def privacy_metrics(root: Path | None = None) -> dict[str, Any]:
    pii_re = re.compile(
        r"[\w.]+@example\.com|\+91 \d{5} \d{5}|EMP\d{6}|(?<=Contact )[A-Z][a-z]+ [A-Z][a-z]+"
    )
    texts = [
        t["description"]
        for t in read_json(repo_path("tickets", root))
        if t["metadata"].get("contains_pii")
    ]
    texts += [
        t["description"]
        for t in read_json(repo_path("test_tickets", root))
        if (t["metadata"].get("edge_case") or "").startswith("pii")
    ]
    total = found = 0
    for t in texts:
        spans = pii_re.findall(t)
        total += len(spans)
        out = redact(t).text
        found += sum(1 for s in spans if s not in out)
    arts = read_json(repo_path("kb_articles", root))
    codes = sorted({c for a in arts for c in a["common_error_codes"]})
    fp = [c for c in codes if redact(c).text != c]
    return {
        "pii_texts": len(texts),
        "pii_spans": total,
        "redaction_recall": found / total if total else 1.0,
        "error_codes_checked": len(codes),
        "error_code_false_positives": len(fp),
        "error_code_fp_rate": len(fp) / len(codes) if codes else 0.0,
    }


def _plot_confusion(path: Path, labels: list[str], matrix: list[list[int]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [*labels, "other"]
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.42), max(5, len(labels) * 0.38)))
    ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_reliability(path: Path, curves: dict[str, list[dict[str, float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    for name, bins in curves.items():
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o", label=name)
    ax.set_xlabel("confidence")
    ax.set_ylabel("subcategory accuracy")
    ax.set_title("Reliability (subcategory)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================================
# Main
# =============================================================================================
def evaluate(
    split: str = "test",
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    limit: int | None = None,
    use_engine: bool = True,
    plots: bool = True,
    stores: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    from store.config import data_config
    from store.factory import IndexMissingError, open_stores

    t_start = time.perf_counter()
    out_dir = out_dir or (root or ROOT) / data_config()["paths"]["eval_out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if stores is None:
        try:
            ts, kb = open_stores(root, eval_split=True)
        except IndexMissingError:
            raise SystemExit("index missing: run `make data-eval` first") from None
        if ts.count("tickets_hist") == 0:
            raise SystemExit("*_eval collections missing: run `make data-eval` first")
    else:
        ts, kb = stores

    tmap = taxonomy(root)
    tax_map_raw = taxonomy_map()
    subs = [s for c in tmap.values() for s in c]
    items = load_items(split, root, limit)
    split_items = [i for i in items if i.set == "split"]
    labelled = [i for i in items if i.category and i.subcategory]
    phrases = kb.all_error_phrases()
    errors: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "split": split,
        "n_items": len(items),
        "n_split_tickets": len(split_items),
        "n_demo": sum(i.set == "demo" for i in items),
        "n_test_set": sum(i.set == "test" for i in items),
        "targets": TARGETS,
    }

    # --- 1. kNN baseline (P1) ---------------------------------------------------------------
    knn_k = 15
    knn = {i.ticket_id: knn_predict(ts, i.text, knn_k) for i in labelled}
    metrics["knn_baseline"] = {
        "k": knn_k,
        "split": classification_metrics(
            split_items, [knn[i.ticket_id][:2] for i in split_items], subs
        ),
    }

    # --- 2. engine (P2) ---------------------------------------------------------------------
    run_engine = engine_runner(ts, kb) if use_engine else None
    eng: dict[str, Any] = {}
    if run_engine is not None:
        for it in items:
            eng[it.ticket_id] = run_engine(it)
    if eng:
        metrics["engine"] = engine_metrics(items, eng, subs, kb, ts, errors, tax_map=tmap)

    # --- 3. calibration (engine, else kNN) --------------------------------------------------
    curves: dict[str, list[dict[str, float]]] = {}
    knn_conf = [knn[i.ticket_id][2] for i in split_items]
    knn_ok = [knn[i.ticket_id][1] == i.subcategory for i in split_items]
    ece, bins = M.reliability(knn_conf, knn_ok)
    metrics["knn_baseline"]["calibration"] = {"ece": ece, "bins": bins}
    curves["kNN baseline"] = bins
    if eng:
        conf = [eng[i.ticket_id].issues[0].classification.confidence for i in split_items]
        ok = [
            eng[i.ticket_id].issues[0].classification.subcategory == i.subcategory
            for i in split_items
        ]
        ece_e, bins_e = M.reliability(conf, ok)
        metrics["engine"]["calibration"] = {"ece": ece_e, "bins": bins_e}
        curves["engine"] = bins_e

    # --- 4. retrieval + ablations -----------------------------------------------------------
    metrics["retrieval"] = retrieval_metrics(split_items, ts)

    try:
        full_ts = stores[0] if stores is not None else open_stores(root)[0]
        metrics["duplicates"] = duplicate_metrics(full_ts, root)
    except IndexMissingError:
        metrics["duplicates"] = {"status": "full index missing: run `make data`"}

    # --- 5. KB recommendation ablations (P1 reference ranker) -------------------------------
    kb_items = [i for i in items if i.kb_ids]
    variants = {
        "full": frozenset(KB_WEIGHTS),
        "no_bm25": frozenset({"sem", "code", "tax"}),
        "no_error_codes": frozenset({"sem", "bm25", "tax"}),
        "no_taxonomy_boost": frozenset({"sem", "bm25", "code"}),
        "dense_only": frozenset({"sem"}),
    }
    kb_abl: dict[str, Any] = {"n": len(kb_items)}
    for name, use in variants.items():
        ranked = []
        for it in kb_items:
            cat, sub, _ = knn.get(it.ticket_id) or knn_predict(ts, it.text, knn_k)
            boost = tax_boost_for(cat, sub, tax_map_raw)
            ranked.append([k for k, _ in kb_rank(it.text, kb, boost, phrases, use)])
        gold = [set(i.kb_ids) for i in kb_items]
        kb_abl[name] = {
            "hit_at_1": M.hit_at_k(ranked, gold, 1),
            "hit_at_3": M.hit_at_k(ranked, gold, 3),
        }
    metrics["kb_ablations_reference_ranker"] = kb_abl

    # --- 6. privacy -------------------------------------------------------------------------
    metrics["privacy"] = privacy_metrics(root)

    # --- 7. tuning (validation split) -------------------------------------------------------
    if split == "val":
        metrics["tuning"] = tune(split_items, ts, eng)
        (out_dir / "tuning.json").write_text(json.dumps(metrics["tuning"], indent=2) + "\n")

    metrics["decision"] = {
        "status": "pending",
        "note": "decision accuracy and escalation recall need P3's backend decision gate "
        "(backend/decision.py); gold decisions are in data/test/test_tickets.json",
    }
    metrics["contract"] = {
        "status": "covered by tests/contract on golden fixtures; live /analyze check pending P3",
    }
    metrics["runtime_s"] = round(time.perf_counter() - t_start, 1)

    # --- outputs ----------------------------------------------------------------------------
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    if plots:
        try:
            _write_plots(out_dir, split_items, eng, knn, tmap, curves)
        except ImportError:
            log.warning("matplotlib not installed: plots skipped")
    return metrics


def engine_metrics(
    items: Sequence[Item],
    eng: dict[str, Any],
    subs: Sequence[str],
    kb: Any,
    ts: Any,
    errors: list[dict[str, Any]],
    *,
    tax_map: dict[str, list[str]],
) -> dict[str, Any]:
    from shared.constants import TAXONOMY

    def cls(i: Item):  # noqa: ANN202
        return eng[i.ticket_id].issues[0].classification

    split_items = [i for i in items if i.set == "split"]
    demo_test = [i for i in items if i.set in ("demo", "test") and i.category and i.subcategory]
    out: dict[str, Any] = {
        "split": classification_metrics(
            split_items, [(cls(i).category, cls(i).subcategory) for i in split_items], subs
        ),
        "demo_and_test": classification_metrics(
            demo_test, [(cls(i).category, cls(i).subcategory) for i in demo_test], subs
        ),
    }
    shared_subs = {s for v in TAXONOMY.values() for s in v}
    data_subs = set(subs)
    in_shared = [i for i in split_items if i.subcategory in shared_subs]
    out["taxonomy_alignment"] = {
        "dataset_subcategories": len(data_subs),
        "engine_label_space_subcategories": len(shared_subs),
        "dataset_subcategories_engine_cannot_emit": sorted(data_subs - shared_subs),
        "split_tickets_with_emittable_gold": len(in_shared),
        "subcategory_accuracy_on_emittable": M.accuracy(
            [i.subcategory for i in in_shared], [cls(i).subcategory for i in in_shared]
        ),
    }
    demos = [i for i in items if i.set == "demo"]
    out["demos"] = {
        "category_subcategory_correct": sum(
            (cls(i).category, cls(i).subcategory) == (i.category, i.subcategory) for i in demos
        ),
        "kb_top1_correct": sum(
            bool(eng[i.ticket_id].issues[0].kb_matches)
            and eng[i.ticket_id].issues[0].kb_matches[0].kb_id in i.kb_ids
            for i in demos
        ),
        "n": len(demos),
    }

    # bands
    ok = [cls(i).subcategory == i.subcategory for i in split_items]
    out["bands"] = M.band_accuracy([cls(i).band for i in split_items], ok)

    # priority
    pr = [i for i in items if i.set in ("demo", "test") and i.priority_level]
    out["priority"] = {
        "demo_test_within_one": M.within_one(
            [i.priority_level for i in pr], [eng[i.ticket_id].issues[0].priority.level for i in pr]
        ),
        "demo_test_exact": M.accuracy(
            [i.priority_level for i in pr], [eng[i.ticket_id].issues[0].priority.level for i in pr]
        ),
        "demo_within_one": M.within_one(
            [i.priority_level for i in pr if i.set == "demo"],
            [eng[i.ticket_id].issues[0].priority.level for i in pr if i.set == "demo"],
        ),
        "split_vs_synthetic_exact": M.accuracy(
            [i.priority_level for i in split_items],
            [eng[i.ticket_id].issues[0].priority.level for i in split_items],
        ),
        "split_vs_synthetic_within_one": M.within_one(
            [i.priority_level for i in split_items],
            [eng[i.ticket_id].issues[0].priority.level for i in split_items],
        ),
    }

    # KB recommendation (engine)
    kb_items = [i for i in items if i.kb_ids]
    ranked = [[h.kb_id for h in eng[i.ticket_id].issues[0].kb_matches] for i in kb_items]
    gold = [set(i.kb_ids) for i in kb_items]
    gap_items = [i for i in items if not i.kb_ids and i.accept_escalate and i.set != "split"]
    out["kb"] = {
        "n": len(kb_items),
        "hit_at_1": M.hit_at_k(ranked, gold, 1),
        "hit_at_3": M.hit_at_k(ranked, gold, 3),
        "by_set": {
            s: {
                "n": sum(i.set == s for i in kb_items),
                "hit_at_3": M.hit_at_k(
                    [r for r, i in zip(ranked, kb_items, strict=True) if i.set == s],
                    [g for g, i in zip(gold, kb_items, strict=True) if i.set == s],
                    3,
                ),
            }
            for s in ("split", "demo", "test")
        },
        "gap_tickets": len(gap_items),
        "gap_tickets_not_found": sum(
            eng[i.ticket_id].issues[0].kb_relevance < 0.50 for i in gap_items
        ),
    }

    # traceability
    steps = valid = synthetic = 0
    removed = 0
    sources: Counter = Counter()
    verified_ids = {r.ticket_id for r in ts.list(collections=("tickets_verified",))}
    for i in items:
        for issue in eng[i.ticket_id].issues:
            rec = issue.recommendation
            sources[rec.source] += 1
            removed += rec.unsupported_steps_removed
            for s in rec.steps:
                steps += 1
                cites = s.citations
                good = bool(cites) and all(
                    kb.has_chunk(c) if c.startswith("KB-") else c in verified_ids for c in cites
                )
                valid += good
                for c in cites:
                    if not c.startswith("KB-"):
                        origin = (ts.get([c]) or [None])[0]
                        if origin is None or origin.resolution_origin not in CITABLE_ORIGINS:
                            synthetic += 1
    out["traceability"] = {
        "steps": steps,
        "citation_validity": valid / steps if steps else 1.0,
        "non_citable_citations": synthetic,
        "unsupported_steps_removed": removed,
        "source_distribution": dict(sources),
        "historical_incident_share": (
            (sources["Historical Incident"] + sources["KEDB + Historical Incident"])
            / sum(sources.values())
            if sources
            else 0.0
        ),
    }

    # edge cases (test set) — behaviour that does not need P3's gate
    edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in items:
        if i.set != "test":
            continue
        er = eng[i.ticket_id]
        iss = er.issues[0]
        edge[i.edge or "other"].append(
            {
                "ticket_id": i.ticket_id,
                "pred": f"{iss.classification.category}/{iss.classification.subcategory}",
                "band": iss.classification.band,
                "kb_top": iss.kb_matches[0].kb_id if iss.kb_matches else None,
                "kb_relevance": round(iss.kb_relevance, 2),
                "issues": len(er.issues),
                "security_flags": er.signals.security_flags,
                "gold_decision": i.decision,
            }
        )
    out["edge_cases"] = edge

    # ops
    totals = [eng[i.ticket_id].timings_ms.get("total", 0) for i in items]
    stages: dict[str, list[int]] = defaultdict(list)
    degraded: Counter = Counter()
    for i in items:
        for k, v in eng[i.ticket_id].timings_ms.items():
            stages[k].append(v)
        degraded.update(eng[i.ticket_id].degraded)
    out["ops"] = {
        "latency_ms": {
            k: {"p50": M.percentile(v, 50), "p95": M.percentile(v, 95)} for k, v in stages.items()
        },
        "latency_p95_ms": M.percentile(totals, 95),
        "degraded_counts": dict(degraded),
        "llm_calls": 0,
        "mode": "offline (GROQ_API_KEY unset)",
    }

    # errors.jsonl
    for i in items:
        c = cls(i)
        iss = eng[i.ticket_id].issues[0]
        kb_top = [h.kb_id for h in iss.kb_matches[:3]]
        wrong_cls = i.subcategory and c.subcategory != i.subcategory
        kb_miss = bool(i.kb_ids) and not (set(kb_top) & set(i.kb_ids))
        if wrong_cls or kb_miss:
            errors.append(
                {
                    "ticket_id": i.ticket_id,
                    "set": i.set,
                    "text": i.text[:240],
                    "gold": f"{i.category}/{i.subcategory}",
                    "pred": f"{c.category}/{c.subcategory}",
                    "confidence": c.confidence,
                    "band": c.band,
                    "gold_kb": i.kb_ids,
                    "pred_kb_top3": kb_top,
                    "error": [n for n, f in (("classification", wrong_cls), ("kb", kb_miss)) if f],
                }
            )
    return out


def tune(split_items: Sequence[Item], ts: Any, eng: dict[str, Any]) -> dict[str, Any]:
    """Validation-split recommendations for P2 (architecture §7.3: high band ≥ 90% accurate)."""
    res: dict[str, Any] = {"n": len(split_items)}
    sweep = {}
    for k in (5, 10, 15, 20, 30):
        for power in (1.0, 2.0, 4.0):
            preds = [knn_predict(ts, i.text, k, power) for i in split_items]
            sweep[f"k={k},power={power}"] = M.macro_f1(
                [i.subcategory for i in split_items], [p[1] for p in preds]
            )
    best = max(sweep, key=sweep.get)
    res["knn_sweep_subcategory_macro_f1"] = sweep
    res["knn_best"] = best
    if eng:
        conf = [eng[i.ticket_id].issues[0].classification.confidence for i in split_items]
        ok = [
            eng[i.ticket_id].issues[0].classification.subcategory == i.subcategory
            for i in split_items
        ]
        res["engine_band_thresholds"] = {
            "high_for_90pct_accuracy": M.threshold_for_precision(conf, ok, 0.90),
            "medium_for_75pct_accuracy": M.threshold_for_precision(conf, ok, 0.75),
            "current": {"high": 0.75, "medium": 0.55},
        }
    return res


def _write_plots(
    out_dir: Path,
    split_items: Sequence[Item],
    eng: dict[str, Any],
    knn: dict[str, tuple[str, str, float]],
    tmap: dict[str, list[str]],
    curves: dict[str, list[dict[str, float]]],
) -> None:
    cats = list(tmap)
    subs = [s for c in tmap.values() for s in c]

    def pred(i: Item) -> tuple[str, str]:
        if eng:
            c = eng[i.ticket_id].issues[0].classification
            return c.category, c.subcategory
        return knn[i.ticket_id][:2]

    who = "engine" if eng else "kNN baseline"
    _plot_confusion(
        out_dir / "confusion_category.png",
        cats,
        M.confusion([i.category for i in split_items], [pred(i)[0] for i in split_items], cats),
        f"Category confusion ({who})",
    )
    _plot_confusion(
        out_dir / "confusion_subcategory.png",
        subs,
        M.confusion([i.subcategory for i in split_items], [pred(i)[1] for i in split_items], subs),
        f"Subcategory confusion ({who})",
    )
    _plot_confusion(
        out_dir / "confusion_subcategory_knn.png",
        subs,
        M.confusion(
            [i.subcategory for i in split_items], [knn[i.ticket_id][1] for i in split_items], subs
        ),
        "Subcategory confusion (kNN baseline)",
    )
    _plot_reliability(out_dir / "reliability.png", curves)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="cap split tickets (smoke runs)")
    parser.add_argument("--no-engine", action="store_true", help="skip P2's engine")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"), format="%(message)s")
    m = evaluate(
        args.split, limit=args.limit, use_engine=not args.no_engine, plots=not args.no_plots
    )
    summary = {
        "knn_subcat_macro_f1": round(m["knn_baseline"]["split"]["subcategory_macro_f1"], 3),
        "retrieval_hybrid_success_at_5": round(m["retrieval"]["hybrid"]["success_at_5"], 3),
        "duplicate_recall_at_5": m["duplicates"].get("recall_at_5"),
        "redaction_recall": round(m["privacy"]["redaction_recall"], 3),
    }
    if "engine" in m:
        e = m["engine"]
        summary.update(
            {
                "engine_cat_acc": round(e["split"]["category_accuracy"], 3),
                "engine_subcat_macro_f1": round(e["split"]["subcategory_macro_f1"], 3),
                "engine_kb_hit_at_3": round(e["kb"]["hit_at_3"], 3),
                "citation_validity": round(e["traceability"]["citation_validity"], 3),
            }
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
