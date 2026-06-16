"""Ruff linter wrapper. Produces a structured ToolResult of lint findings."""
from __future__ import annotations

import json

from ..state import ToolResult
from ._subprocess import module_cmd, run_cli


def run_ruff(repo_dir: str, target_path: str, timeout: int = 120) -> ToolResult:
    """Run ruff over the target path and return parsed JSON diagnostics.

    Degrades gracefully: if ruff is missing or errors, returns ok=False with
    an empty data list so the bug agent can still run on file contents alone.
    """
    target = target_path or "."
    out = run_cli(
        module_cmd("ruff", "check", target, "--output-format", "json"),
        cwd=repo_dir,
        timeout=timeout,
    )

    if not out.ok:
        return ToolResult(
            name="ruff", ok=False, summary=f"ruff did not run: {out.stderr}",
            raw=out.stderr, data=[],
        )

    diagnostics = []
    try:
        diagnostics = json.loads(out.stdout) if out.stdout.strip() else []
    except json.JSONDecodeError:
        return ToolResult(
            name="ruff", ok=False,
            summary="ruff ran but JSON output could not be parsed.",
            raw=out.stdout or out.stderr, data=[],
        )

    summary = f"ruff: {len(diagnostics)} lint finding(s)."
    return ToolResult(
        name="ruff", ok=True, summary=summary,
        raw=out.stdout, data=diagnostics,
    )
