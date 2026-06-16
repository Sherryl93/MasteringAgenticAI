"""Convert raw tool output into Finding objects.

These deterministic conversions guarantee the report has real, tool-backed
findings even when every LLM is unavailable (the tool-only fallback path).

All file paths are normalized to **repo-relative, forward-slash** form so the
report never leaks local machine paths (e.g. C:\\Users\\...\\.tmp_repos\\...).
"""
from __future__ import annotations

from ..state import Finding

# Map bandit severity strings to our scale.
_BANDIT_SEV = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low", "UNDEFINED": "Low"}


def normalize_path(path: str, repo_dir: str) -> str:
    """Return a repo-relative, forward-slash path.

    Handles both absolute paths (ruff) and already-relative backslash paths
    (bandit) by stripping the clone directory prefix when present and
    converting separators. Never raises.
    """
    if not path:
        return ""
    p = str(path).replace("\\", "/")
    root = str(repo_dir or "").replace("\\", "/").rstrip("/")
    if root and p.startswith(root):
        p = p[len(root):]
    return p.lstrip("/")


def findings_from_ruff(diagnostics: list[dict], repo_dir: str = "") -> list[Finding]:
    out: list[Finding] = []
    for d in diagnostics:
        code = d.get("code") or "RUFF"
        loc = d.get("location") or {}
        out.append(
            Finding(
                agent="bug",
                severity="Low",
                file=normalize_path(d.get("filename", ""), repo_dir),
                line=loc.get("row"),
                title=f"[{code}] {d.get('message', 'lint issue')}",
                rationale="Reported by ruff static linting.",
                suggestion=(d.get("fix") or {}).get("message", "") if d.get("fix") else "",
                evidence={"rule": code, "source": "ruff"},
            )
        )
    return out


def findings_from_bandit(results: list[dict], repo_dir: str = "") -> list[Finding]:
    out: list[Finding] = []
    for r in results:
        sev = _BANDIT_SEV.get(r.get("issue_severity", "UNDEFINED"), "Low")
        # Promote high-confidence high-severity issues to Critical.
        if r.get("issue_severity") == "HIGH" and r.get("issue_confidence") == "HIGH":
            sev = "Critical"
        out.append(
            Finding(
                agent="security",
                severity=sev,
                file=normalize_path(r.get("filename", ""), repo_dir),
                line=r.get("line_number"),
                title=f"[{r.get('test_id', 'B000')}] {r.get('issue_text', 'security issue')}",
                rationale="Reported by bandit static security analysis.",
                suggestion=r.get("more_info", ""),
                evidence={
                    "test_id": r.get("test_id"),
                    "confidence": r.get("issue_confidence"),
                    "source": "bandit",
                },
            )
        )
    return out
