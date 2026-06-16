"""Report agent: render the final Markdown review report.

Deterministic Markdown generation from ranked findings + tool summaries.
Produces a complete report even in the tool-only fallback path.
"""
from __future__ import annotations

from ..state import SEVERITY_ORDER, ReviewState

_SEV_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}


def _counts(ranked: list[dict]) -> dict[str, int]:
    counts = {sev: 0 for sev in SEVERITY_ORDER}
    for f in ranked:
        counts[f.get("severity", "Low")] = counts.get(f.get("severity", "Low"), 0) + 1
    return counts


# Fallback category derivation for findings built without the classifier
# (e.g. hand-constructed dicts in unit tests). Real runs always set category.
_AGENT_CATEGORY = {
    "security": "Security Risk",
    "test": "Test Coverage Gap",
    "bug": "Correctness Risk",
}

# Section order: real defects first, advisory style last.
_CATEGORY_SECTIONS = [
    ("⚠️ Correctness Risks", "Correctness Risk"),
    ("🔐 Security Risks", "Security Risk"),
    ("🧪 Test Coverage Gaps", "Test Coverage Gap"),
    ("🔧 Maintainability", "Maintainability"),
    ("🎨 Code Quality (Style)", "Style"),
]


def _finding_category(f: dict) -> str:
    return f.get("category") or _AGENT_CATEGORY.get(f.get("agent", ""), "Correctness Risk")


def _evidence_str(f: dict) -> str:
    """Render merged evidence (tool rules + lines) for a finding."""
    ev = f.get("evidence") or {}
    parts: list[str] = []
    rules = ev.get("rules") or ([ev.get("rule")] if ev.get("rule") else [])
    rules = [r for r in rules if r]
    if rules:
        parts.append("rules: " + ", ".join(rules))
    lines = ev.get("lines") or ([f.get("line")] if f.get("line") else [])
    lines = [str(x) for x in lines if x]
    if len(lines) > 1:
        parts.append("lines: " + ", ".join(lines))
    if ev.get("merged_count", 0) > 1:
        parts.append(f"merged {ev['merged_count']} findings")
    return " · ".join(parts)


def _category_section(title: str, category: str, ranked: list[dict]) -> str:
    rows = [f for f in ranked if _finding_category(f) == category]
    if not rows:
        return f"## {title}\n\n_No findings._\n"
    lines = [f"## {title}\n"]
    for f in rows:
        emoji = _SEV_EMOJI.get(f.get("severity", "Low"), "")
        loc = f.get("file", "")
        if f.get("line"):
            loc += f":{f['line']}"
        evidence = _evidence_str(f)
        block = (
            f"### {emoji} {f.get('id', '')} · {f.get('severity')} · {f.get('title')}\n"
            f"- **Location:** `{loc or 'n/a'}`\n"
            f"- **Why:** {f.get('rationale', '').strip() or 'n/a'}\n"
            f"- **Suggestion:** {f.get('suggestion', '').strip() or 'n/a'}\n"
        )
        if evidence:
            block += f"- **Evidence:** {evidence}\n"
        lines.append(block)
    return "\n".join(lines) + "\n"


def _clean_title(title: str) -> str:
    """Trim trailing punctuation so inline sentences don't show '..'."""
    return (title or "").strip().rstrip(".").strip()


def _executive_summary(state: ReviewState, ranked: list[dict], counts: dict[str, int]) -> list[str]:
    """Deterministic executive-summary bullets derived from ranked findings."""
    n = len(ranked)
    sec = [f for f in ranked if _finding_category(f) == "Security Risk"]
    correctness = [f for f in ranked if _finding_category(f) == "Correctness Risk"]
    tests = [f for f in ranked if _finding_category(f) == "Test Coverage Gap"]
    style = [f for f in ranked if _finding_category(f) == "Style"]
    serious = [f for f in ranked if f.get("severity") in ("Critical", "High")]

    bullets = ["## Executive Summary", ""]

    if n == 0:
        bullets += ["- ✅ No findings surfaced by the agents and static tools.", ""]
        return bullets

    # Headline risk posture.
    if counts["Critical"]:
        bullets.append(
            f"- 🔴 **{counts['Critical']} critical** issue(s) require immediate attention."
        )
    if counts["High"]:
        bullets.append(f"- 🟠 **{counts['High']} high-severity** issue(s) should be prioritized.")
    if not serious:
        bullets.append("- 🟢 No critical or high-severity issues; findings are medium/low.")

    # Security headline (highest-priority security finding).
    if sec:
        top = sec[0]
        bullets.append(
            f"- 🔒 **Security:** {len(sec)} finding(s); top — {_clean_title(top.get('title'))} "
            f"(`{top.get('file','n/a')}`)."
        )
    else:
        bullets.append("- 🔒 **Security:** no security findings.")

    # Correctness headline.
    if correctness:
        bullets.append(f"- 🐛 **Correctness:** {len(correctness)} risk(s) identified.")

    # Test posture.
    if tests:
        bullets.append(f"- 🧪 **Tests:** {_clean_title(tests[0].get('title'))}.")

    # Style is advisory only.
    if style:
        bullets.append(
            f"- 🎨 **Style:** {len(style)} low-priority code-quality suggestion(s)."
        )

    # Single top recommendation (never a style finding).
    focus = serious or sec or correctness or ranked
    bullets.append(
        f"- 👉 **Recommended focus:** start with {focus[0].get('id','')} — "
        f"{_clean_title(focus[0].get('title'))}."
    )
    bullets.append("")
    return bullets


def build_report_md(state: ReviewState) -> str:
    ranked = state.get("ranked", []) or []
    counts = _counts(ranked)
    degraded = state.get("llm_degraded", False)

    ruff = state.get("ruff", {}) or {}
    bandit = state.get("bandit", {}) or {}
    tests = state.get("test_analysis", {}) or {}

    models_used = sorted(set(state.get("models_used", []) or []))
    if models_used:
        model_line = ", ".join(models_used)
    elif degraded:
        model_line = "tool-only (no LLM)"
    else:
        model_line = "n/a"

    header = [
        "# ReviewPilot Report",
        "",
        f"- **Repository:** {state.get('repo_url', 'n/a')}",
        f"- **Target path:** `{state.get('target_path', '.') or '.'}`",
        f"- **Python files reviewed:** {len(state.get('python_files', []))}",
        f"- **LLM model(s) used:** {model_line}",
        f"- **Total findings:** {len(ranked)}",
        "",
        "| Severity | Count |",
        "|---|---|",
        f"| 🔴 Critical | {counts['Critical']} |",
        f"| 🟠 High | {counts['High']} |",
        f"| 🟡 Medium | {counts['Medium']} |",
        f"| 🟢 Low | {counts['Low']} |",
        "",
    ]
    if degraded:
        header.append(
            "> ⚠️ One or more LLM agents were unavailable; this report includes "
            "tool-derived findings (ruff/bandit/static test analysis).\n"
        )

    tool_summary = [
        "## Tool Summary",
        "",
        f"- **Ruff:** {ruff.get('summary', 'not run')}",
        f"- **Bandit:** {bandit.get('summary', 'not run')}",
        f"- **Static test analysis:** {tests.get('summary', 'not run')}",
        "",
        "> Test coverage is assessed statically (no target code is executed). "
        "See README for the rationale.",
        "",
    ]

    priority = ["## Priority Findings", ""]
    if ranked:
        priority.append("| ID | Severity | Category | File | Title |")
        priority.append("|---|---|---|---|---|")
        for f in ranked:
            loc = f.get("file", "")
            if f.get("line"):
                loc += f":{f['line']}"
            priority.append(
                f"| {f.get('id','')} | {f.get('severity')} | {_finding_category(f)} "
                f"| `{loc}` | {f.get('title')} |"
            )
    else:
        priority.append("_No findings._")
    priority += [
        "",
        "> ℹ️ Some findings are style or maintainability suggestions and do not "
        "represent functional defects.",
        "",
    ]

    exec_summary = _executive_summary(state, ranked, counts)
    body = "\n".join(header + exec_summary + tool_summary + priority)
    for title, category in _CATEGORY_SECTIONS:
        body += "\n" + _category_section(title, category, ranked)

    errors = state.get("errors", []) or []
    if errors:
        body += "\n## Pipeline Notes\n\n" + "\n".join(f"- {e}" for e in errors) + "\n"

    return body


def report_agent(state: ReviewState) -> dict:
    md = build_report_md(state)
    return {"report_md": md, "logs": [f"report_agent: built report ({len(md)} chars)"]}
