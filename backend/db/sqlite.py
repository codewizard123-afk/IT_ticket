"""SQLite access for ``data/app.db`` (architecture §11.2).

Connections run in autocommit mode (``isolation_level=None``); multi-statement writes go through
:func:`transaction`, which takes the write lock up front (``BEGIN IMMEDIATE``) so read-then-write
sequences such as the audit hash chain cannot interleave.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend.config import ROOT, backend_config, base_config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
MEMORY = ":memory:"


def default_db_path() -> Path:
    """``APP_DB_PATH`` if set, else ``paths.<db.path_key>`` from base.yaml under the repo root."""
    env = os.environ.get("APP_DB_PATH")
    if env:
        return Path(env)
    key = backend_config().get("db", {}).get("path_key", "app_db")
    return ROOT / base_config()["paths"][key]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with dict-like rows. Creates the parent directory for file databases."""
    cfg = backend_config().get("db", {})
    is_memory = str(path) == MEMORY
    if not is_memory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        isolation_level=None,
        check_same_thread=False,  # FastAPI runs sync endpoints in a thread pool
        timeout=cfg.get("busy_timeout_ms", 5000) / 1000,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not is_memory:
        conn.execute(f"PRAGMA journal_mode = {cfg.get('journal_mode', 'WAL')}")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Apply ``schema.sql``. Safe to call repeatedly."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def open_db(path: str | Path | None = None) -> sqlite3.Connection:
    """Connect to ``path`` (default: :func:`default_db_path`) and ensure the schema exists."""
    conn = connect(path if path is not None else default_db_path())
    init_db(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, rolling back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
