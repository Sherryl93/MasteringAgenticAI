"""Graph state and the structured data contracts agents speak.

The `Finding` schema is the central contract: every analysis agent emits
findings in this shape, and the ranking/report agents consume them. Keeping
it strict is what lets the parallel agents merge cleanly and the report stay
deterministic.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Severity = Literal["Critical", "High", "Medium", "Low"]
AgentName = Literal["bug", "security", "test"]

SEVERITY_ORDER: dict[str, int] = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


class Finding(BaseModel):
    """A single review finding. This is the unit every agent produces.

    `agent` has a placeholder default because LLM responses don't include it —
    `LLMClient.structured_findings` stamps the correct producing agent after
    parsing. Without a default, every LLM finding would fail validation and the
    whole LLM path would silently degrade to tool-only.
    """

    agent: AgentName = "bug"
    # Functional category, assigned by the classifier at ranking time:
    # "Correctness Risk" | "Security Risk" | "Test Coverage Gap" |
    # "Maintainability" | "Style". Empty until classified.
    category: str = ""
    severity: Severity = "Medium"
    file: str = ""
    line: Optional[int] = None
    title: str
    rationale: str = ""
    suggestion: str = ""
    # Free-form provenance, e.g. {"rule": "B105", "source": "bandit"}.
    evidence: dict[str, Any] = Field(default_factory=dict)

    def sort_key(self) -> tuple[int, str]:
        return (SEVERITY_ORDER.get(self.severity, 99), self.file)


class FindingList(BaseModel):
    """Wrapper used as the LLM structured-output target (JSON object root)."""

    findings: list[Finding] = Field(default_factory=list)


class ToolResult(TypedDict, total=False):
    """Raw output of a static tool run (ruff/bandit/test analysis)."""

    name: str
    ok: bool
    summary: str
    raw: str
    data: Any


class ReviewState(TypedDict, total=False):
    """LangGraph state.

    List-typed fields that parallel agents write to use `operator.add`
    reducers so concurrent writes merge instead of clobbering each other.
    """

    # Inputs
    repo_url: str
    target_path: str

    # Pipeline bookkeeping
    repo_dir: str
    python_files: list[str]
    errors: Annotated[list[str], operator.add]
    logs: Annotated[list[str], operator.add]

    # Tool outputs (single-writer, no reducer needed)
    ruff: ToolResult
    bandit: ToolResult
    test_analysis: ToolResult

    # Agent findings (parallel writers -> reducer merges them)
    findings: Annotated[list[dict], operator.add]

    # Downstream artifacts
    ranked: list[dict]
    report_md: str
    # True if ANY parallel agent fell back to tool-only; reducer ORs the
    # concurrent writes from the three analysis agents.
    llm_degraded: Annotated[bool, operator.or_]
    # Models that actually answered (primary or fallback), merged across agents.
    models_used: Annotated[list[str], operator.add]
