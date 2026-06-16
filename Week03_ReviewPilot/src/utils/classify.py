"""Finding classification and deduplication (the report-quality layer).

Two responsibilities, applied at ranking time:

1. `classify` — assign a functional category (Correctness/Security/Test/
   Maintainability/Style) and enforce severity rules so that *style* findings
   (e.g. Ruff F541) can never be Medium/High or outrank real issues.

2. `dedupe` — merge findings that describe the same root cause at the same
   file, preferring the LLM-enriched title and the higher severity while
   retaining the tool ID (e.g. Bandit B301) as evidence. Repetitive style
   findings (e.g. multiple F541s in one file) collapse into a single entry.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

from ..state import SEVERITY_ORDER, Finding

# Lower number = higher priority within the same severity. Security and real
# correctness bugs rank above coverage, maintainability, and style.
CATEGORY_PRIORITY = {
    "Security Risk": 0,
    "Correctness Risk": 0,
    "Test Coverage Gap": 1,
    "Maintainability": 2,
    "Style": 3,
}

# Ruff rules that are genuine correctness problems (real runtime risk).
_CORRECTNESS_RULES = {"F821", "F823", "E999", "F706", "F707", "F811"}
# Ruff rules that are maintainability (dead/unused code), not bugs.
_MAINT_RULES = {"F401", "F841"}
_MAINT_KEYWORDS = (
    "maintainab", "readab", "duplicat", "refactor", "complex",
    "naming", "docstring", "dead code", "magic number",
)
# Content signals used to re-route a finding whose producing agent mislabels it
# (e.g. the test agent surfacing a security issue instead of a coverage gap).
_SECURITY_KW = (
    "api key", "api-key", "apikey", "secret", "password", "credential",
    "injection", "deserial", "pickle", "ssrf", "rce", "path traversal",
    "hardcoded", "eval(", "exec(", "exposure", "expos", "insecure",
)
_CORRECTNESS_KW = (
    "race condition", "deadlock", "off-by-one", "infinite loop", "unbounded",
    "crash", "null", "none type", "incorrect", "data loss",
)
_TEST_KW = (
    "test", "coverage", "untested", "assertion", "assert", "edge case",
    "mock", "fixture", "regression",
)

_CODE_PREFIX = re.compile(r"^\[([A-Za-z]\d+)\]\s*(.*)$")

# A known root cause authoritatively sets the category — this overrides an LLM
# that mislabels a style nit (e.g. an f-string without placeholders) as a bug.
_ROOT_CAUSE_CATEGORY = {
    "fstring-no-placeholder": "Style",
    "pickle-deserialization": "Security Risk",
    "pickle-import": "Security Risk",
    "url-open-scheme": "Security Risk",
    "try-except-pass": "Security Risk",
}


def _clean_title(f: Finding) -> None:
    """Strip a leading `[CODE] ` tool prefix into evidence, keeping the title
    readable (e.g. '[B301] Pickle ...' -> 'Pickle ...', rule kept in evidence)."""
    m = _CODE_PREFIX.match(f.title or "")
    if m:
        code, rest = m.group(1).upper(), m.group(2).strip()
        f.evidence.setdefault("rule", code)
        if rest:
            f.title = rest


def classify(f: Finding) -> Finding:
    """Assign a category and enforce severity rules in place. Returns `f`."""
    _clean_title(f)
    ev = f.evidence or {}
    src = ev.get("source")
    rule = str(ev.get("rule") or ev.get("test_id") or "").upper()
    title = (f.title or "").lower()

    text = f"{title} {(f.rationale or '').lower()}"
    if f.agent == "security" or src == "bandit":
        cat = "Security Risk"
    elif f.agent == "test":
        # The test agent should report coverage gaps; if it instead surfaces a
        # security/correctness issue (no test wording), classify by content.
        if any(k in text for k in _TEST_KW):
            cat = "Test Coverage Gap"
        elif any(k in text for k in _SECURITY_KW):
            cat = "Security Risk"
        elif any(k in text for k in _CORRECTNESS_KW):
            cat = "Correctness Risk"
        else:
            cat = "Test Coverage Gap"
    elif src == "ruff" or rule[:1] in ("E", "W"):
        if rule in _CORRECTNESS_RULES:
            cat = "Correctness Risk"
        elif rule in _MAINT_RULES:
            cat = "Maintainability"
        else:
            cat = "Style"
    else:  # LLM bug-agent finding
        cat = "Maintainability" if any(k in title for k in _MAINT_KEYWORDS) else "Correctness Risk"

    # A recognized root cause wins — prevents an LLM from promoting a style nit
    # (e.g. f-string without placeholders) into a Correctness/Medium "bug".
    rc = root_cause(f)
    if rc in _ROOT_CAUSE_CATEGORY:
        cat = _ROOT_CAUSE_CATEGORY[rc]

    f.category = cat
    rank = SEVERITY_ORDER.get(f.severity, 2)
    if cat == "Style":
        # Style never represents a functional defect -> capped at Low so it can
        # never outrank security / correctness / coverage findings.
        f.severity = "Low"
    elif cat == "Maintainability" and rank < SEVERITY_ORDER["Medium"]:
        f.severity = "Medium"
    elif cat == "Test Coverage Gap" and rank < SEVERITY_ORDER["Medium"]:
        f.severity = "Medium"
    return f


def root_cause(f: Finding) -> Optional[str]:
    """A coarse root-cause tag used to merge findings about the same issue."""
    ev = f.evidence or {}
    rule = str(ev.get("rule") or ev.get("test_id") or "").upper()
    t = (f.title or "").lower()
    if rule == "B301" or ("pickle" in t and ("load" in t or "deserial" in t)):
        return "pickle-deserialization"
    if rule == "B403" or ("pickle" in t and "import" in t):
        return "pickle-import"
    if rule == "B310" or "urlopen" in t or "url open" in t or "audit url" in t:
        return "url-open-scheme"
    if rule == "F541" or ("f-string" in t and "placeholder" in t):
        return "fstring-no-placeholder"
    if (
        rule == "B110"
        or "try, except, pass" in t
        or "try except pass" in t
        or "broad except" in t
        or "overly broad exception" in t
    ):
        return "try-except-pass"
    return None


def _is_tool(f: Finding) -> bool:
    return bool((f.evidence or {}).get("source")) or (f.title or "").strip().startswith("[")


def _rule_of(f: Finding) -> Optional[str]:
    ev = f.evidence or {}
    return ev.get("rule") or ev.get("test_id")


def _merge(group: list[Finding]) -> Finding:
    """Merge a group of same-root-cause findings into one enriched finding."""
    if len(group) == 1:
        return group[0]

    # Highest severity in the group wins.
    best = min(group, key=lambda f: SEVERITY_ORDER.get(f.severity, 99))
    # Prefer an LLM-enriched (non-tool) title; else the richest tool title.
    llm = [f for f in group if not _is_tool(f)]
    primary = max(llm or group, key=lambda f: len(f.title or "") + len(f.rationale or ""))

    rules = sorted({str(_rule_of(f)) for f in group if _rule_of(f)})
    lines = sorted({f.line for f in group if f.line is not None})
    sources = sorted({(f.evidence or {}).get("source") for f in group if (f.evidence or {}).get("source")})

    return Finding(
        agent=primary.agent,
        category=primary.category,
        severity=best.severity,
        file=primary.file,
        line=primary.line if primary.line is not None else (lines[0] if lines else None),
        title=primary.title,
        rationale=primary.rationale or next((f.rationale for f in group if f.rationale), ""),
        suggestion=primary.suggestion or next((f.suggestion for f in group if f.suggestion), ""),
        evidence={
            "rules": rules,
            "lines": lines,
            "sources": sources,
            "merged_count": len(group),
        },
    )


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Merge findings by (file, root-cause); else drop exact duplicates."""
    groups: "OrderedDict[tuple, list[Finding]]" = OrderedDict()
    for f in findings:
        rc = root_cause(f)
        if rc:
            key = ("rc", f.file, rc)
        else:
            key = ("ex", f.file, f.line, (f.title or "").strip().lower())
        groups.setdefault(key, []).append(f)
    return [_merge(g) for g in groups.values()]
