"""Shared plumbing for analysis agents."""
from __future__ import annotations

from typing import Optional

from ..config import get_settings
from ..llm_client import LLMClient
from ..state import Finding, ReviewState
from ..utils.batching import build_code_context
from ..utils.findings import normalize_path

# One shared client instance; it is stateless aside from config.
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(get_settings())
    return _client


def code_context(state: ReviewState, max_total_chars: int = 60000) -> str:
    return build_code_context(
        state["repo_dir"],
        state.get("python_files", []),
        get_settings(),
        max_total_chars=max_total_chars,
    )


def run_llm_findings(
    agent: str, system: str, user: str, repo_dir: str = ""
) -> tuple[list[Finding], bool, Optional[str]]:
    """Call the LLM for findings. Returns (findings, degraded, model_used).

    `degraded` is True when the LLM produced nothing usable, signalling the
    caller to rely on tool-derived findings alone. `model_used` is the model
    that actually answered (primary or a fallback), or None if degraded.
    Paths are normalized as a safeguard so a model can never leak a local
    machine path into the report.
    """
    client = get_client()
    findings = client.structured_findings(system, user, agent=agent)
    if findings is None:
        return [], True, None
    for f in findings:
        f.file = normalize_path(f.file, repo_dir)
    return findings, False, client.last_model


def as_dicts(findings: list[Finding]) -> list[dict]:
    return [f.model_dump() for f in findings]
