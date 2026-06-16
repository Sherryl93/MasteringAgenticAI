"""Security agent: bandit findings + LLM security review."""
from __future__ import annotations

from ..state import ReviewState
from ..utils.findings import findings_from_bandit
from . import prompts
from ._base import as_dicts, code_context, run_llm_findings


def security_agent(state: ReviewState) -> dict:
    bandit = state.get("bandit", {}) or {}
    tool_findings = findings_from_bandit(bandit.get("data", []) or [], state.get("repo_dir", ""))

    user = (
        f"BANDIT SUMMARY: {bandit.get('summary', 'n/a')}\n"
        f"BANDIT FINDINGS (JSON): {bandit.get('data', [])}\n\n"
        f"SOURCE FILES:\n{code_context(state)}"
    )
    llm_findings, degraded, model_used = run_llm_findings(
        "security", prompts.SECURITY_AGENT, user, state.get("repo_dir", "")
    )

    all_findings = as_dicts(tool_findings) + as_dicts(llm_findings)
    log = (
        f"security_agent: {len(tool_findings)} bandit + {len(llm_findings)} LLM finding(s)"
        + (" (LLM degraded)" if degraded else f" [model: {model_used}]")
    )
    return {
        "findings": all_findings,
        "logs": [log],
        "llm_degraded": degraded,
        "models_used": [model_used] if model_used else [],
    }
