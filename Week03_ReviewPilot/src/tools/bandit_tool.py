"""Bandit security scanner wrapper. Static AST-based; runs no target code."""
from __future__ import annotations

import json

from ..state import ToolResult
from ._subprocess import module_cmd, run_cli


def run_bandit(repo_dir: str, target_path: str, timeout: int = 180) -> ToolResult:
    """Run bandit recursively over the target path, returning JSON results.

    Bandit analyzes the AST and never imports/executes the scanned code, which
    is exactly why it is safe to point at an untrusted repo.
    """
    target = target_path or "."
    out = run_cli(
        module_cmd("bandit", "-r", target, "-f", "json", "-q"),
        cwd=repo_dir,
        timeout=timeout,
    )

    # Bandit exits 1 when it finds issues; stdout still holds valid JSON.
    if not out.stdout.strip():
        return ToolResult(
            name="bandit", ok=out.ok,
            summary=f"bandit produced no output: {out.stderr}".strip(),
            raw=out.stderr, data=[],
        )

    try:
        payload = json.loads(out.stdout)
    except json.JSONDecodeError:
        return ToolResult(
            name="bandit", ok=False,
            summary="bandit ran but JSON output could not be parsed.",
            raw=out.stdout or out.stderr, data=[],
        )

    results = payload.get("results", [])
    by_sev: dict[str, int] = {}
    for r in results:
        sev = r.get("issue_severity", "UNDEFINED")
        by_sev[sev] = by_sev.get(sev, 0) + 1
    sev_str = ", ".join(f"{k}:{v}" for k, v in sorted(by_sev.items())) or "none"
    summary = f"bandit: {len(results)} issue(s) ({sev_str})."

    return ToolResult(
        name="bandit", ok=True, summary=summary,
        raw=out.stdout, data=results,
    )
