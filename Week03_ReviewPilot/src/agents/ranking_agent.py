"""Ranking agent: fan-in point that classifies, deduplicates, and orders.

Pipeline at this node (all deterministic — no LLM):
  classify  -> assign category + enforce severity rules (style capped to Low)
  dedupe    -> merge same-root-cause findings (LLM title + higher severity,
               tool IDs retained as evidence)
  sort      -> by severity, then category priority (security/correctness above
               coverage above maintainability above style), then file
  number    -> assign stable F-001 display ids
"""
from __future__ import annotations

from ..state import SEVERITY_ORDER, Finding, ReviewState
from ..utils.classify import CATEGORY_PRIORITY, classify, dedupe


def _sort_key(f: dict) -> tuple:
    return (
        SEVERITY_ORDER.get(f.get("severity", "Low"), 99),       # severity first
        CATEGORY_PRIORITY.get(f.get("category", ""), 9),         # then category
        f.get("file", ""),
    )


def ranking_agent(state: ReviewState) -> dict:
    raw = state.get("findings", []) or []

    # Reconstruct -> classify (category + severity rules) -> deduplicate/merge.
    findings = [classify(Finding(**d)) for d in raw]
    merged = dedupe(findings)

    ranked = [f.model_dump() for f in merged]
    ranked.sort(key=_sort_key)
    for i, f in enumerate(ranked, start=1):
        f["id"] = f"F-{i:03d}"

    counts = {sev: 0 for sev in SEVERITY_ORDER}
    for f in ranked:
        counts[f.get("severity", "Low")] = counts.get(f.get("severity", "Low"), 0) + 1

    log = (
        f"ranking_agent: {len(raw)} raw -> {len(ranked)} after dedup "
        f"(C:{counts['Critical']} H:{counts['High']} "
        f"M:{counts['Medium']} L:{counts['Low']})"
    )
    return {"ranked": ranked, "logs": [log]}
