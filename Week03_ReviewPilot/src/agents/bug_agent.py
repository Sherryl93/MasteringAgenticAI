"""Bug agent: bugs, anti-patterns, maintainability risks.

Owns ruff-derived findings (deterministic) and augments them with LLM
reasoning over the source. Owning ruff conversion here keeps each tool's
findings single-sourced and avoids duplication across agents.
"""
from __future__ import annotations

from ..state import ReviewState
from ..utils.findings import findings_from_ruff
from . import prompts
from ._base import as_dicts, code_context, run_llm_findings


def bug_agent(state: ReviewState) -> dict:
    ruff = state.get("ruff", {}) or {}
    tool_findings = findings_from_ruff(ruff.get("data", []) or [], state.get("repo_dir", ""))

    user = (
        f"RUFF SUMMARY: {ruff.get('summary', 'n/a')}\n"
        f"RUFF FINDINGS (JSON): {ruff.get('data', [])}\n\n"
        f"SOURCE FILES:\n{code_context(state)}"
    )
    llm_findings, degraded, model_used = run_llm_findings(
        "bug", prompts.BUG_AGENT, user, state.get("repo_dir", "")
    )

    all_findings = as_dicts(tool_findings) + as_dicts(llm_findings)
    log = (
        f"bug_agent: {len(tool_findings)} ruff + {len(llm_findings)} LLM finding(s)"
        + (" (LLM degraded)" if degraded else f" [model: {model_used}]")
    )
    return {
        "findings": all_findings,
        "logs": [log],
        "llm_degraded": degraded,
        "models_used": [model_used] if model_used else [],
    }
