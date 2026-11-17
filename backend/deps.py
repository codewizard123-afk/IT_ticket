"""FastAPI dependency factories (README §8.2). Tests override these via ``dependency_overrides``."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from backend.db.sqlite import open_db


def get_db() -> Iterator[sqlite3.Connection]:
    """One connection per request to ``APP_DB_PATH`` / ``paths.app_db``."""
    conn = open_db()
    try:
        yield conn
    finally:
        conn.close()
