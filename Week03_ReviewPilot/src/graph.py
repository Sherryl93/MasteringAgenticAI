"""LangGraph pipeline wiring.

Flow:

    validate_input -> clone_repo -> scan_python_files
    -> run_ruff -> run_bandit -> static_test_analysis
    -> (bug_agent || security_agent || test_agent)   # parallel fan-out
    -> ranking_agent -> report_agent -> human_review_gate -> END

The three analysis agents fan out from `static_test_analysis` and fan in at
`ranking_agent`. Concurrent writes to `findings`/`logs` merge via the
`operator.add` reducers declared in state.py.

`human_review_gate` is a no-op marker node: the actual human approval happens
in the Streamlit UI (gating report download), which keeps a single, simple
HITL mechanism. The node is kept so the graph matches the documented design.
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from .agents import bug_agent, ranking_agent, report_agent, security_agent, test_agent
from .config import Settings, get_settings
from .repo_loader import clone_repo, scan_python_files, validate_repo_url
from .state import ReviewState
from .tools import analyze_tests, run_bandit, run_ruff


# --------------------------------------------------------------------- nodes
def validate_input(state: ReviewState) -> dict:
    url = state.get("repo_url", "")
    ok, msg = validate_repo_url(url)
    if not ok:
        # Raise to stop the graph early; the runner converts this to a clean error.
        raise ValueError(f"Invalid repository URL: {msg}")
    return {"logs": [f"validate_input: {url} ok"]}


def clone_repo_node(state: ReviewState, settings: Settings) -> dict:
    path, msg = clone_repo(state["repo_url"], settings)
    if path is None:
        raise RuntimeError(f"clone_repo failed: {msg}")
    return {"repo_dir": path, "logs": [f"clone_repo: {msg}"]}


def scan_node(state: ReviewState, settings: Settings) -> dict:
    files, msg = scan_python_files(state["repo_dir"], state.get("target_path", ""), settings)
    if not files:
        raise RuntimeError(f"scan_python_files: {msg}")
    return {"python_files": files, "logs": [f"scan_python_files: {msg}"]}


def ruff_node(state: ReviewState) -> dict:
    result = run_ruff(state["repo_dir"], state.get("target_path", ""))
    errors = [] if result["ok"] else [f"ruff: {result['summary']}"]
    return {"ruff": result, "logs": [f"run_ruff: {result['summary']}"], "errors": errors}


def bandit_node(state: ReviewState) -> dict:
    result = run_bandit(state["repo_dir"], state.get("target_path", ""))
    errors = [] if result["ok"] else [f"bandit: {result['summary']}"]
    return {"bandit": result, "logs": [f"run_bandit: {result['summary']}"], "errors": errors}


def test_analysis_node(state: ReviewState) -> dict:
    result = analyze_tests(state["repo_dir"], state.get("python_files", []))
    return {"test_analysis": result, "logs": [f"static_test_analysis: {result['summary']}"]}


def human_review_gate(state: ReviewState) -> dict:
    # No-op marker: human approval is enforced in the Streamlit UI.
    return {"logs": ["human_review_gate: report ready, awaiting UI approval"]}


# --------------------------------------------------------------------- build
def build_graph(settings: Optional[Settings] = None):
    """Compile and return the ReviewPilot LangGraph."""
    settings = settings or get_settings()
    g = StateGraph(ReviewState)

    g.add_node("validate_input", validate_input)
    g.add_node("clone_repo", lambda s: clone_repo_node(s, settings))
    g.add_node("scan_python_files", lambda s: scan_node(s, settings))
    g.add_node("run_ruff", ruff_node)
    g.add_node("run_bandit", bandit_node)
    g.add_node("static_test_analysis", test_analysis_node)
    g.add_node("bug_agent", bug_agent)
    g.add_node("security_agent", security_agent)
    g.add_node("test_agent", test_agent)
    g.add_node("ranking_agent", ranking_agent)
    g.add_node("report_agent", report_agent)
    g.add_node("human_review_gate", human_review_gate)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "clone_repo")
    g.add_edge("clone_repo", "scan_python_files")
    g.add_edge("scan_python_files", "run_ruff")
    g.add_edge("run_ruff", "run_bandit")
    g.add_edge("run_bandit", "static_test_analysis")

    # Fan-out: the three analysis agents run in parallel.
    g.add_edge("static_test_analysis", "bug_agent")
    g.add_edge("static_test_analysis", "security_agent")
    g.add_edge("static_test_analysis", "test_agent")

    # Fan-in: ranking waits for all three before running.
    g.add_edge("bug_agent", "ranking_agent")
    g.add_edge("security_agent", "ranking_agent")
    g.add_edge("test_agent", "ranking_agent")

    g.add_edge("ranking_agent", "report_agent")
    g.add_edge("report_agent", "human_review_gate")
    g.add_edge("human_review_gate", END)

    return g.compile()
