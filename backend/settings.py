"""Runtime settings from the environment (README §8.2, §13). Never log ``app_secret``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.config import ROOT, backend_config, base_config

_TRUE = {"1", "true", "yes", "on"}


def _flag(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.split("#", 1)[0].strip().lower() in _TRUE


def _path(env: Mapping[str, str], name: str, key: str) -> Path:
    raw = env.get(name)
    if raw:
        return Path(raw)
    return ROOT / base_config()["paths"][key]


@dataclass(frozen=True)
class Settings:
    demo_auth: bool = False
    app_secret: str | None = None
    use_stub_stores: bool = False
    use_stub_engine: bool = False
    use_stub_redact: bool = False
    db_path: Path = ROOT / "data" / "app.db"
    kb_governed_path: Path = ROOT / "data" / "kb" / "articles_governed.json"
    trial_tickets_path: Path = ROOT / "data" / "test" / "trial_tickets.json"
    tickets_enriched_path: Path = ROOT / "data" / "processed" / "tickets_enriched.json"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        db_key = backend_config().get("db", {}).get("path_key", "app_db")
        secret = (env.get("APP_SECRET") or "").strip()
        return cls(
            demo_auth=_flag(env, "DEMO_AUTH"),
            app_secret=secret if secret and secret != "change-me" else None,
            use_stub_stores=_flag(env, "USE_STUB_STORES"),
            use_stub_engine=_flag(env, "USE_STUB_ENGINE"),
            use_stub_redact=_flag(env, "USE_STUB_REDACT"),
            db_path=_path(env, "APP_DB_PATH", db_key),
            kb_governed_path=_path(env, "KB_GOVERNED_PATH", "kb_governed"),
            trial_tickets_path=ROOT / "data" / "test" / "trial_tickets.json",
            tickets_enriched_path=_path(env, "TICKETS_ENRICHED_PATH", "tickets_enriched"),
        )
