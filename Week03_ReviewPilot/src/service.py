"""Backend service layer: the single entry point the UI calls.

This is the seam between backend and UI. The Streamlit app imports exactly
one function from here — `run_review` — and renders the returned
`ReviewResult`. The UI can be redesigned freely without touching the graph,
agents, or tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings
from .graph import build_graph


@dataclass
class ReviewResult:
    """Everything the UI needs to render a completed (or failed) review."""

    ok: bool
    repo_url: str
    target_path: str
    findings: list[dict] = field(default_factory=list)
    report_md: str = ""
    ruff_summary: str = ""
    bandit_summary: str = ""
    test_summary: str = ""
    logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    llm_degraded: bool = False
    models_used: list[str] = field(default_factory=list)
    report_path: Optional[str] = None

    @property
    def model_status(self) -> str:
        """Human-readable summary of which model(s) produced the findings."""
        if self.llm_degraded and not self.models_used:
            return "Tool-only (no LLM) — ruff/bandit/static analysis"
        if self.models_used:
            uniq = sorted(set(self.models_used))
            return ", ".join(uniq)
        return "—"

    # Convenience accessors used by the UI metric cards.
    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.get("severity") == severity)

    @property
    def total(self) -> int:
        return len(self.findings)

    def by_agent(self, agent: str) -> list[dict]:
        return [f for f in self.findings if f.get("agent") == agent]

    # Category-based grouping (matches the report sections).
    _AGENT_FALLBACK = {
        "security": "Security Risk",
        "test": "Test Coverage Gap",
        "bug": "Correctness Risk",
    }

    def by_category(self, *categories: str) -> list[dict]:
        wanted = set(categories)
        out = []
        for f in self.findings:
            cat = f.get("category") or self._AGENT_FALLBACK.get(f.get("agent", ""), "Correctness Risk")
            if cat in wanted:
                out.append(f)
        return out


def _save_report(report_md: str, repo_url: str, settings: Settings) -> Optional[str]:
    """Persist the report to reports/ with a deterministic, repo-derived name."""
    try:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        slug = repo_url.rstrip("/").split("/")[-1].replace(".git", "") or "report"
        path = reports_dir / f"{slug}_review.md"
        path.write_text(report_md, encoding="utf-8")
        return str(path)
    except Exception:
        return None


def run_review(
    repo_url: str,
    target_path: str,
    settings: Optional[Settings] = None,
    save: bool = True,
) -> ReviewResult:
    """Run the full review pipeline and return a structured result.

    Never raises: any pipeline failure is captured into `ReviewResult.errors`
    with `ok=False`, so the UI can always render something.
    """
    settings = settings or get_settings()
    graph = build_graph(settings)

    initial: dict = {
        "repo_url": repo_url.strip(),
        "target_path": target_path.strip(),
        "errors": [],
        "logs": [],
        "findings": [],
        "models_used": [],
    }

    try:
        final = graph.invoke(initial)
    except Exception as e:
        return ReviewResult(
            ok=False,
            repo_url=repo_url,
            target_path=target_path,
            errors=[f"Pipeline stopped: {e}"],
        )

    report_md = final.get("report_md", "")
    report_path = _save_report(report_md, repo_url, settings) if save and report_md else None

    return ReviewResult(
        ok=True,
        repo_url=repo_url,
        target_path=target_path,
        findings=final.get("ranked", []),
        report_md=report_md,
        ruff_summary=(final.get("ruff", {}) or {}).get("summary", ""),
        bandit_summary=(final.get("bandit", {}) or {}).get("summary", ""),
        test_summary=(final.get("test_analysis", {}) or {}).get("summary", ""),
        logs=final.get("logs", []),
        errors=final.get("errors", []),
        llm_degraded=final.get("llm_degraded", False),
        models_used=final.get("models_used", []),
        report_path=report_path,
    )
