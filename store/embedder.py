"""The one shared embedder (architecture §6.5, §6.6).

``embed(texts)`` returns L2-normalised float32 vectors from ``BAAI/bge-small-en-v1.5``. The
model is loaded lazily, once per process (the only global the codebase allows, CLAUDE.md).

``HashingEmbedder`` is a deterministic, dependency-light stand-in (token-hash bag of words) for
unit tests and for ``P1_EMBEDDER=hash`` when the model cannot be downloaded. Never mix the two
in one index: the manifest records which one built it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Sequence
from typing import Any

from store.config import base_config
from store.text import tokenize

_MODEL: Any = None
_LOCK = threading.Lock()
HASH_MODEL_NAME = "hashing-bow-384"


def model_name() -> str:
    if os.environ.get("P1_EMBEDDER", "").lower() == "hash":
        return HASH_MODEL_NAME
    return str(base_config().get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"))


def _get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                _MODEL = SentenceTransformer(model_name(), device="cpu")
    return _MODEL


def embed(texts: Sequence[str]):  # noqa: ANN201 - returns np.ndarray (numpy is optional in CI)
    """Embed ``texts`` (already redacted) into normalised vectors, shape ``(n, dim)``."""
    if os.environ.get("P1_EMBEDDER", "").lower() == "hash":
        return HashingEmbedder()(texts)
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _get_model().encode(
        list(texts),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words embedder: token and bigram hashes into ``dim`` buckets."""

    name = HASH_MODEL_NAME

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not security relevant
        return int.from_bytes(h[:4], "little") % self.dim, (1.0 if h[4] & 1 else -1.0)

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        for tok in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]:
            i, sign = self._bucket(tok)
            vec[i] += sign
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def __call__(self, texts: Sequence[str]):  # noqa: ANN204
        import numpy as np

        return np.asarray([self.vector(t) for t in texts], dtype=np.float32).reshape(
            len(texts), self.dim
        )
