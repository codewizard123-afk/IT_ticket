"""Prompt templates (``*.j2``). Only ``{{ name }}`` substitution is used, so no Jinja runtime is
needed; the files stay Jinja-compatible."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@lru_cache(maxsize=16)
def load(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def render(name: str, **values: object) -> str:
    """Render ``ai/prompts/<name>``. Unknown variables raise ``KeyError`` (fail loudly)."""

    def sub(m: re.Match[str]) -> str:
        return str(values[m.group(1)])

    return _VAR.sub(sub, load(name)).strip()
