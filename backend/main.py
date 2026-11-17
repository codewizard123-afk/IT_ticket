"""FastAPI app (architecture §10).

Run ``uvicorn backend.main:app --port 8000``; OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend import errors
from backend.config import backend_config, base_config
from backend.db.sqlite import open_db
from backend.routers import analyze, audit, auth
from backend.services import Services, build_services

ROUTERS = (auth.router, analyze.router, audit.router)


def create_app(services: Services | None = None) -> FastAPI:
    """App factory. Tests pass ready-made ``Services``; production builds them from the env."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "services", None) is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(override=False)
            except ImportError:  # pragma: no cover - python-dotenv is in requirements/base.txt
                pass
            app.state.services = build_services()
        open_db(app.state.services.settings.db_path).close()  # create the schema up front
        yield

    app = FastAPI(
        title="IT Ticket Analysis & Resolution Assistant",
        version=str(backend_config().get("backend_version", "0")),
        lifespan=lifespan,
    )
    app.state.services = services
    errors.install(app)
    base = base_config().get("api", {}).get("base_path", "/api/v1")
    for router in ROUTERS:
        app.include_router(router, prefix=base)
    return app


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = create_app()
