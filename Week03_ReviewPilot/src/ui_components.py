"""Pure Streamlit rendering functions.

These functions ONLY render a `ReviewResult` (or simpler values) into the UI.
They contain no graph/agent/tool logic, so the interface can be redesigned or
restyled later without touching the backend. `app.py` owns layout, session
state, and the single call into `service.run_review`.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .service import ReviewResult

try:  # lightweight; degrade gracefully if the package is missing
    from streamlit_mermaid import st_mermaid

    _HAS_MERMAID = True
except Exception:  # pragma: no cover
    _HAS_MERMAID = False

_SEV_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

# ------------------------------------------------------------------ pipeline diagram
# Emoji-labelled nodes (ids mirror src/graph.py order).
_NODE_LABELS = {
    "vin": "✅ validate_input",
    "clone": "📥 clone_repo",
    "scan": "🔎 scan_python_files",
    "ruff": "🧹 run_ruff",
    "bandit": "🛡️ run_bandit",
    "stest": "🧪 static_test_analysis",
    "bug": "🐛 bug_agent",
    "sec": "🔐 security_agent",
    "test": "🧩 test_agent",
    "rank": "📊 ranking_agent",
    "report": "📝 report_agent",
    "gate": "🧑‍⚖️ human_review_gate",
}
_AGENT_IDS = {"bug", "sec", "test"}
# Vibrant base colour class per node group.
_GROUP = {
    "vin": "inpC", "clone": "loadC", "scan": "loadC",
    "ruff": "toolC", "bandit": "toolC", "stest": "toolC",
    "bug": "agentC", "sec": "agentC", "test": "agentC",
    "rank": "synthC", "report": "synthC", "gate": "gateC",
}


def _node_decl(nid: str) -> str:
    """Node declaration with a distinct shape per role."""
    label = _NODE_LABELS[nid]
    if nid == "vin":
        return f'{nid}(["{label}"])'        # stadium = entry point
    if nid == "gate":
        return f'{nid}{{{{"{label}"}}}}'    # hexagon = decision / HITL
    return f'{nid}("{label}")'              # rounded = process


def build_pipeline_mermaid(done: bool, degraded: bool, approved: bool) -> str:
    """Vibrant Mermaid flowchart.

    Before a run, nodes are coloured by role (a lively palette). After a run,
    nodes turn green (completed); the agents turn orange if the LLM degraded to
    tool-only, and the human-review gate is blue until the report is approved.
    """
    lines = ["flowchart TD"]
    for nid in ["vin", "clone", "scan", "ruff", "bandit", "stest"]:
        lines.append("  " + _node_decl(nid))
    lines += [
        '  subgraph PAR["⚡ parallel fan-out"]',
        "    direction LR",
        "    " + _node_decl("bug"),
        "    " + _node_decl("sec"),
        "    " + _node_decl("test"),
        "  end",
    ]
    for nid in ["rank", "report", "gate"]:
        lines.append("  " + _node_decl(nid))

    lines += [
        "  vin --> clone --> scan --> ruff --> bandit --> stest",
        "  stest --> bug",
        "  stest --> sec",
        "  stest --> test",
        "  bug --> rank",
        "  sec --> rank",
        "  test --> rank",
        "  rank --> report --> gate",
    ]

    # Vibrant class palette.
    lines += [
        "classDef inpC fill:#7e57c2,stroke:#5e35b1,color:#fff,stroke-width:2px;",
        "classDef loadC fill:#26a69a,stroke:#00897b,color:#fff,stroke-width:2px;",
        "classDef toolC fill:#29b6f6,stroke:#0288d1,color:#fff,stroke-width:2px;",
        "classDef agentC fill:#ec407a,stroke:#c2185b,color:#fff,stroke-width:2px;",
        "classDef synthC fill:#ab47bc,stroke:#8e24aa,color:#fff,stroke-width:2px;",
        "classDef gateC fill:#ffa726,stroke:#fb8c00,color:#222,stroke-width:2px;",
        "classDef doneC fill:#43a047,stroke:#1b5e20,color:#fff,stroke-width:3px;",
        "classDef degC fill:#fb8c00,stroke:#e65100,color:#fff,stroke-width:3px;",
        "classDef waitC fill:#1e88e5,stroke:#0d47a1,color:#fff,stroke-width:3px;",
    ]

    buckets: dict[str, list[str]] = {}
    for nid in _NODE_LABELS:
        if done:
            if nid in _AGENT_IDS:
                cls = "degC" if degraded else "doneC"
            elif nid == "gate":
                cls = "doneC" if approved else "waitC"
            else:
                cls = "doneC"
        else:
            cls = _GROUP[nid]
        buckets.setdefault(cls, []).append(nid)
    for cls, ids in buckets.items():
        lines.append(f"class {','.join(ids)} {cls}")

    # Styled links + tinted parallel-fan-out box.
    lines.append("linkStyle default stroke:#b0bec5,stroke-width:2px;")
    lines.append(
        "style PAR fill:#fff8e1,stroke:#fb8c00,stroke-width:1px,stroke-dasharray:5 5;"
    )
    return "\n".join(lines)


def render_pipeline_diagram(result: ReviewResult | None, approved: bool = False) -> None:
    """Vibrant pipeline diagram with state-driven highlighting (no controls).

    Colour-coded by role before a run; reflects completion / degraded / pending
    approval after a run. Backend-agnostic — driven purely from `result`.
    """
    has_result = bool(result and result.ok)
    degraded = bool(result.llm_degraded) if result else False
    code = build_pipeline_mermaid(done=has_result, degraded=degraded, approved=approved)

    if _HAS_MERMAID:
        st_mermaid(code, height="600px")
    else:  # graceful fallback — never break the app
        st.info("Install `streamlit-mermaid` to render the diagram.")
        st.code(code, language="mermaid")


def render_status(message: str, kind: str = "info") -> None:
    {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}.get(
        kind, st.info
    )(message)


def render_metric_cards(result: ReviewResult) -> None:
    """Summary metric cards: total + per-severity counts."""
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("Total findings", result.total)
    c1.metric("🔴 Critical", result.count("Critical"))
    c2.metric("🟠 High", result.count("High"))
    c3.metric("🟡 Medium", result.count("Medium"))
    c4.metric("🟢 Low", result.count("Low"))

    # Show which model actually produced the findings (primary or fallback).
    if result.llm_degraded and not result.models_used:
        st.warning(
            "🤖 **Model:** tool-only (no LLM) — findings from ruff / bandit / "
            "static test analysis. The LLM was unavailable."
        )
    else:
        st.success(f"🤖 **Model used:** {result.model_status} — LLM active ✅")


def _findings_df(findings: list[dict]) -> pd.DataFrame:
    if not findings:
        return pd.DataFrame(
            columns=["id", "severity", "category", "file", "line", "title"]
        )
    rows = [
        {
            "id": f.get("id", ""),
            "severity": f"{_SEV_EMOJI.get(f.get('severity',''),'')} {f.get('severity','')}",
            "category": f.get("category", "") or f.get("agent", ""),
            "file": f.get("file", ""),
            "line": f.get("line", ""),
            "title": f.get("title", ""),
        }
        for f in findings
    ]
    return pd.DataFrame(rows)


def render_findings_table(findings: list[dict], empty_msg: str = "No findings.") -> None:
    if not findings:
        st.info(empty_msg)
        return
    st.dataframe(_findings_df(findings), use_container_width=True, hide_index=True)


def render_finding_details(findings: list[dict]) -> None:
    """Expandable per-finding detail (rationale + suggestion)."""
    for f in findings:
        emoji = _SEV_EMOJI.get(f.get("severity", ""), "")
        loc = f.get("file", "")
        if f.get("line"):
            loc += f":{f['line']}"
        with st.expander(f"{emoji} {f.get('id','')} · {f.get('severity')} · {f.get('title')}"):
            st.markdown(f"**Location:** `{loc or 'n/a'}`")
            st.markdown(f"**Why:** {f.get('rationale','').strip() or 'n/a'}")
            st.markdown(f"**Suggestion:** {f.get('suggestion','').strip() or 'n/a'}")


def render_tool_summaries(result: ReviewResult) -> None:
    st.markdown(f"- **Ruff:** {result.ruff_summary or 'not run'}")
    st.markdown(f"- **Bandit:** {result.bandit_summary or 'not run'}")
    st.markdown(f"- **Static test analysis:** {result.test_summary or 'not run'}")
    st.caption(
        "Test coverage is assessed statically — ReviewPilot never executes the "
        "target repository's code. See README for the rationale."
    )


def render_logs(logs: list[str], errors: list[str]) -> None:
    if errors:
        st.markdown("**Pipeline notes / errors**")
        st.code("\n".join(errors), language="text")
    st.markdown("**Execution log**")
    st.code("\n".join(logs) if logs else "(no logs)", language="text")
