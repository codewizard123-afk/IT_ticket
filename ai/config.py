"""Config loading for the AI engine (P2).

Values come from ``config/ai.yaml``, ``config/rules.yaml`` and ``config/signals.yaml``; every key
has a code default here so a missing file or key never breaks the engine (CLAUDE.md: don't
hard-code thresholds, but stay robust).
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

AI_DEFAULTS: dict[str, Any] = {
    "embedding": {"model": "BAAI/bge-small-en-v1.5", "normalize": True},
    "llm": {
        "classify_model": "llama-3.1-8b-instant",
        "synth_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "temperature": 0,
        "timeout_s": 20,
        "max_retries": 2,
        "backoff_base_s": 1.0,
        "backoff_max_s": 8.0,
        "rpm": 25,
        "tpm": 5500,
        "cache": "data/llm_cache.sqlite",
        "breaker_failures": 3,
        "breaker_cooldown_s": 60,
    },
    "classifier": {
        "weights": {"llm": 0.45, "knn": 0.35, "rule": 0.20},
        "weights_no_llm": {"knn": 0.65, "rule": 0.35},
        "no_llm_conf_cap": 0.80,
        "single_voter_cap": 0.54,
        "knn_k": 15,
        "knn_power": 2.0,
        "knn_min_rel": 0.5,
        "bands": {"high": 0.75, "medium": 0.55},
        "conflict_min": 0.40,
        "other_min": 0.30,
        "split_min": 0.35,
    },
    "retrieval": {
        "tickets": {
            "k_dense": 30,
            "k_bm25": 30,
            "rrf_k": 60,
            "top_n": 5,
            "rerank": False,
            "related_min": 0.55,
        },
        "ticket_bonus": {
            "same_category": 0.10,
            "same_subcategory": 0.05,
            "verified_resolution": 0.05,
        },
        "kb": {"k_chunks": 20, "top_n": 3, "tau_show": 0.35, "tau_kb": 0.50, "ambiguity_gap": 0.05},
        "weights_kb": {"sem": 0.45, "bm25": 0.15, "code": 0.20, "tax": 0.12, "fb": 0.08},
    },
    "duplicates": {"near": 0.92, "similar": 0.75, "live_window_h": 72, "cluster_min": 3},
    "synthesis": {
        "max_kb_articles": 2,
        "max_verified_incidents": 3,
        "verified_min_cos": 0.75,
        "max_pack_chars": 10000,
    },
    "validator": {"grounding_min": 0.30, "min_steps": 2},
    "taxonomy_map_fallback": {},
}


def deep_merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``over`` (neither input is mutated)."""
    out = copy.deepcopy(base)
    for key, val in (over or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_yaml(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load ``config/<name>`` or return ``{}`` if it doesn't exist."""
    path = (config_dir or CONFIG_DIR) / name
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=4)
def ai_config(config_dir: Path | None = None) -> dict[str, Any]:
    """``config/ai.yaml`` merged over the code defaults. Cached; treat the result as read-only."""
    return deep_merge(AI_DEFAULTS, load_yaml("ai.yaml", config_dir))
