"""Jinja2 prompt templates. Each template defines `system()` and `user()` macros."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent)),
    undefined=StrictUndefined,
    autoescape=False,  # prompts are plain text, never HTML
    keep_trailing_newline=False,
)


def render_messages(template: str, **ctx: Any) -> list[dict[str, str]]:
    """Render `<template>` into chat messages `[system, user]`."""
    module = _ENV.get_template(template).make_module(ctx)
    return [
        {"role": "system", "content": str(module.system()).strip()},  # type: ignore[attr-defined]
        {"role": "user", "content": str(module.user()).strip()},  # type: ignore[attr-defined]
    ]
