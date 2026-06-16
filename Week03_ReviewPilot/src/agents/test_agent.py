"""Test agent: coverage gaps and missing tests from static analysis.

Works purely from the static test analysis and source code — no tests are
executed. Findings are *summarized*, not one-per-file: a single "No test
suite detected" (or "partial coverage") finding lists the highest-priority
modules to test, instead of flooding the report with one row per untested
file.
"""
from __future__ import annotations

from ..state import Finding, ReviewState
from . import prompts
from ._base import as_dicts, code_context, run_llm_findings

# Modules whose names suggest core logic worth testing first. Lower-value
# files (package markers) are deprioritized.
_PRIORITY_HINTS = (
    "main",
    "app",
    "graph",
    "pipeline",
    "agent",
    "service",
    "client",
    "config",
    "embed",
    "ingest",
    "store",
    "eval",
)
_LOW_VALUE = ("__init__.py",)


def _priority_score(path: str) -> int:
    """Lower score = higher priority. __init__.py sinks to the bottom."""
    name = path.rsplit("/", 1)[-1].lower()
    if name in _LOW_VALUE:
        return 100
    for i, hint in enumerate(_PRIORITY_HINTS):
        if hint in name:
            return i
    return 50


def prioritize_modules(modules: list[str], limit: int = 8) -> list[str]:
    """Rank untested modules by likely importance and cap the list."""
    ranked = sorted(modules, key=lambda m: (_priority_score(m), m))
    return ranked[:limit]


def _summary_finding(analysis: dict) -> list[Finding]:
    """One grouped finding for missing/weak test coverage."""
    untested = analysis.get("untested_modules", []) or []
    n_test_files = analysis.get("n_test_files", 0)
    n_source = analysis.get("n_source_files", 0)
    if not untested and n_test_files:
        return []  # tests exist and every module is covered — nothing to flag

    top = prioritize_modules(untested)
    bullet_list = "\n".join(f"  - {m}" for m in top) or "  - (none)"
    extra = len(untested) - len(top)
    more = f"\n  - …and {extra} more module(s)." if extra > 0 else ""

    # Coverage findings are deliberately capped below Critical/High so they
    # never outrank real security vulnerabilities or bugs in the ranking.
    if n_test_files == 0:
        title = "No test suite detected"
        severity = "Medium"
        rationale = (
            f"Static analysis found 0 test files across {n_source} source "
            f"module(s). The project has no automated regression safety net."
        )
    else:
        title = f"Partial test coverage ({n_test_files} test file(s), "
        title += f"{len(untested)} untested module(s))"
        severity = "Low"
        rationale = (
            f"{len(untested)} of {n_source} source module(s) have no obvious "
            f"matching test file."
        )

    suggestion = (
        "Add unit tests, prioritizing these higher-signal modules first:\n"
        f"{bullet_list}{more}"
    )
    return [
        Finding(
            agent="test",
            severity=severity,
            file=top[0] if top else "",
            title=title,
            rationale=rationale,
            suggestion=suggestion,
            evidence={
                "source": "test_analysis",
                "untested_count": len(untested),
                "priority_modules": top,
            },
        )
    ]


def test_agent(state: ReviewState) -> dict:
    analysis = (state.get("test_analysis", {}) or {}).get("data", {}) or {}

    user = (
        f"STATIC TEST ANALYSIS (JSON): {analysis}\n\n"
        f"NOTE: tests were not executed.\n\n"
        f"SOURCE FILES:\n{code_context(state)}"
    )
    llm_findings, degraded, model_used = run_llm_findings(
        "test", prompts.TEST_AGENT, user, state.get("repo_dir", "")
    )

    if degraded:
        findings = _summary_finding(analysis)
        log = f"test_agent: LLM degraded, {len(findings)} grouped coverage finding(s)"
    else:
        findings = llm_findings
        log = f"test_agent: {len(findings)} LLM finding(s) [model: {model_used}]"

    return {
        "findings": as_dicts(findings),
        "logs": [log],
        "llm_degraded": degraded,
        "models_used": [model_used] if model_used else [],
    }
