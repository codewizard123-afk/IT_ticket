"""P1 test fixtures: a small, self-contained repo root built with the hashing embedder.

Store tests never download a model and never touch the real ``data/chroma`` index.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from store.config import ROOT


def _subset_tickets(per_subcategory: int = 3) -> list[dict]:
    rows = json.loads((ROOT / "data/processed/tickets.json").read_text("utf-8"))
    seen: dict[str, int] = defaultdict(int)
    out = []
    for r in rows:
        if seen[r["subcategory"]] < per_subcategory:
            seen[r["subcategory"]] += 1
            out.append(r)
    return out


def make_root(dst: Path, per_subcategory: int = 3) -> Path:
    """Copy the KEDB, curated incidents and a ticket subset into ``dst`` (repo layout)."""
    (dst / "data/processed").mkdir(parents=True)
    (dst / "data/kb").mkdir(parents=True)
    (dst / "data/seed").mkdir(parents=True)
    shutil.copy(ROOT / "data/kb/articles.json", dst / "data/kb/articles.json")
    shutil.copy(
        ROOT / "data/seed/resolved_incidents.json", dst / "data/seed/resolved_incidents.json"
    )
    (dst / "data/processed/tickets.json").write_text(
        json.dumps(_subset_tickets(per_subcategory)), encoding="utf-8"
    )
    return dst


@pytest.fixture(scope="session")
def hash_embed():
    pytest.importorskip("numpy")
    from store.embedder import HashingEmbedder

    return HashingEmbedder()


@pytest.fixture(scope="session")
def built_root(tmp_path_factory: pytest.TempPathFactory, hash_embed):
    """A temp root with ``build_all`` run once (session-scoped for speed)."""
    pytest.importorskip("chromadb")
    from ingest.build_all import build_all

    root = make_root(tmp_path_factory.mktemp("p1root"))
    build_all(root=root, embed_fn=hash_embed)
    return root


@pytest.fixture(scope="session")
def stores(built_root: Path, hash_embed):
    from store.factory import open_stores

    return open_stores(built_root, embed_fn=hash_embed)
