"""Evaluation metrics (architecture §14.2). Pure Python, no numpy, so they run in CI."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

from shared.constants import PRIORITY_LEVELS


# --- classification -------------------------------------------------------------------------
def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def per_label_prf(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> dict[Hashable, dict[str, float]]:
    labels = list(labels) if labels is not None else sorted(set(y_true), key=str)
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for t, p in zip(y_true, y_pred, strict=True):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out = {}
    for lab in labels:
        prec = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        rec = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp[lab] + fn[lab],
        }
    return out


def macro_f1(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> float:
    """Macro-F1 over ``labels`` (default: the labels present in ``y_true``)."""
    prf = per_label_prf(y_true, y_pred, labels)
    return sum(v["f1"] for v in prf.values()) / len(prf) if prf else 0.0


def confusion(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable]
) -> list[list[int]]:
    """Rows = true label, columns = predicted label (+ a final "other" column)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * (len(labels) + 1) for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        if t in idx:
            m[idx[t]][idx.get(p, len(labels))] += 1
    return m


def pair_confusions(
    y_true: Sequence[str], y_pred: Sequence[str], pairs: Sequence[Sequence[str]]
) -> dict[str, int]:
    """How often members of each focus group are confused with each other (§14.2)."""
    out = {}
    for group in pairs:
        g = set(group)
        n = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t in g and p in g and t != p)
        out[" <-> ".join(group)] = n
    return out


# --- ranking ---------------------------------------------------------------------------------
def hit_at_k(ranked: Sequence[Sequence[str]], gold: Sequence[set[str]], k: int) -> float:
    pairs = [(r, g) for r, g in zip(ranked, gold, strict=True) if g]
    if not pairs:
        return 0.0
    return sum(bool(set(r[:k]) & g) for r, g in pairs) / len(pairs)


def precision_at_k(rels: Sequence[Sequence[bool]], k: int) -> float:
    if not rels:
        return 0.0
    return sum(sum(r[:k]) / k for r in rels) / len(rels)


def success_at_k(rels: Sequence[Sequence[bool]], k: int) -> float:
    if not rels:
        return 0.0
    return sum(any(r[:k]) for r in rels) / len(rels)


def mrr_at_k(rels: Sequence[Sequence[bool]], k: int) -> float:
    if not rels:
        return 0.0
    total = 0.0
    for r in rels:
        for i, rel in enumerate(r[:k], 1):
            if rel:
                total += 1 / i
                break
    return total / len(rels)


def ndcg_at_k(
    rels: Sequence[Sequence[bool]], k: int, n_relevant: Sequence[int] | None = None
) -> float:
    """Binary nDCG@k. ``n_relevant[i]`` caps the ideal DCG (defaults to hits in the list)."""
    if not rels:
        return 0.0
    total = 0.0
    for i, r in enumerate(rels):
        dcg = sum(1 / math.log2(j + 2) for j, rel in enumerate(r[:k]) if rel)
        n_rel = n_relevant[i] if n_relevant is not None else sum(r)
        ideal = sum(1 / math.log2(j + 2) for j in range(min(k, n_rel)))
        total += dcg / ideal if ideal else 0.0
    return total / len(rels)


# --- calibration -----------------------------------------------------------------------------
def reliability(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error and the reliability-curve bins."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct, strict=True):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, ok))
    n = len(confidences) or 1
    ece = 0.0
    out = []
    for i, b in enumerate(bins):
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
        out.append(
            {"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": len(b), "conf": conf, "acc": acc}
        )
    return ece, out


def band_accuracy(bands: Sequence[str], correct: Sequence[bool]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for band in ("high", "medium", "low"):
        sel = [ok for b, ok in zip(bands, correct, strict=True) if b == band]
        out[band] = {"n": len(sel), "accuracy": sum(sel) / len(sel) if sel else 0.0}
    return out


def threshold_for_precision(
    confidences: Sequence[float], correct: Sequence[bool], target: float, min_n: int = 10
) -> float | None:
    """Smallest threshold t (0.01 grid) with accuracy(conf ≥ t) ≥ target over ≥ min_n items."""
    for step in range(30, 100):
        t = step / 100
        sel = [ok for c, ok in zip(confidences, correct, strict=True) if c >= t]
        if len(sel) >= min_n and sum(sel) / len(sel) >= target:
            return t
    return None


# --- priority / misc -------------------------------------------------------------------------
def level_distance(a: str, b: str) -> int:
    return abs(PRIORITY_LEVELS.index(a) - PRIORITY_LEVELS.index(b))


def within_one(true_levels: Sequence[str], pred_levels: Sequence[str]) -> float:
    if not true_levels:
        return 0.0
    ok = sum(level_distance(t, p) <= 1 for t, p in zip(true_levels, pred_levels, strict=True))
    return ok / len(true_levels)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100)."""
    if not values:
        return 0.0
    v = sorted(values)
    pos = (len(v) - 1) * p / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion of several ranked id lists."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, 1):
            score[doc] = score.get(doc, 0.0) + 1 / (k + r)
    return sorted(score, key=lambda d: (-score[d], d))
