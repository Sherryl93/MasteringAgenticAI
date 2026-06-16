"""Assemble a token-bounded code context string for an LLM agent.

We cap the number of files and the per-file character budget (from settings)
so a single agent call can't blow through the context window or cost budget.
"""
from __future__ import annotations

from typing import Optional

from ..config import Settings, get_settings
from ..repo_loader import read_file_snippet


def build_code_context(
    repo_dir: str,
    files: list[str],
    settings: Optional[Settings] = None,
    max_total_chars: int = 60000,
) -> str:
    """Concatenate file snippets with headers, bounded by a global char cap."""
    settings = settings or get_settings()
    parts: list[str] = []
    used = 0
    for rel in files:
        snippet = read_file_snippet(repo_dir, rel, settings)
        block = f"\n### FILE: {rel}\n```python\n{snippet}\n```\n"
        if used + len(block) > max_total_chars:
            parts.append(f"\n<<context budget reached; {rel} and later files omitted>>\n")
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)
