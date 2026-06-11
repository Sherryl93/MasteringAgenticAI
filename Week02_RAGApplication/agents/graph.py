"""LangGraph wiring for the Macro-Micro Bridge Agent.

Graph shape:

        parse_query                 (resolve ticker + expand query once)
        /          \\
   macro_node   company_node        (run in parallel)
        \\          /
   check_retrieval_quality          (join: enough context to proceed?)
        /          \\
  refusal_node   synthesis_node     (conditional routing)
        |              |
       END       generate_answer
                       |
                      END

macro_node and company_node are independent branches off parse_query; the
quality gate fires once both have completed (LangGraph super-step join) and then
routes to either the refusal path or the synthesis path.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import company_agent, macro_agent, synthesis_agent
from pipeline import config, embed_store


class BridgeState(TypedDict):
    """Shared state threaded through every node."""

    question: str
    ticker: str
    expanded_queries: list[str]
    macro_context: list[str]
    company_context: list[str]
    macro_sources: list[str]
    company_sources: list[str]
    retrieval_ok: bool
    synthesis: str
    verdict: str
    answer: str
    ticker_note: str


# ──────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────
def parse_query(state: BridgeState) -> dict:
    """Resolve the ticker and clean the question.

    Resolution (see _resolve_ticker) never lets a different ticker mentioned in
    the question silently override an explicit selection: a single mentioned
    ticker wins (and is noted as an override), but when several are mentioned the
    selected one is kept if it is among them, otherwise the first by position is
    chosen and the decision is surfaced via `ticker_note`.
    """
    question = state["question"].strip()
    ticker, ticker_note = _resolve_ticker(question, state.get("ticker"))
    ticker = config.validate_ticker(ticker)
    if ticker_note:
        print(f"[graph] parse_query — NOTE  {ticker_note}")

    # Warm shared local models once, single-threaded, BEFORE macro_node and
    # company_node fan out in parallel. Otherwise both branches race to download
    # the same FlashRank/BGE model on first use (WinError 32 on Windows).
    try:
        embed_store.get_embedder()
        embed_store.get_reranker()
    except Exception as exc:  # noqa: BLE001 — non-fatal; nodes will retry under lock
        print(f"[graph] WARN  model pre-warm failed ({exc})")

    # Expand once here so both retrieval nodes reuse the result (one LLM call).
    try:
        expanded = embed_store.expand_query(question, synthesis_agent.get_llm())
    except Exception as exc:  # noqa: BLE001 — fall back to the bare question
        print(f"[graph] WARN  query expansion failed ({exc}); using original only")
        expanded = [question]

    print(
        f"[graph] parse_query — ticker={ticker} "
        f"expanded={len(expanded)} queries question={question!r}"
    )
    return {
        "question": question,
        "ticker": ticker,
        "expanded_queries": expanded,
        "ticker_note": ticker_note or "",
    }


def _tickers_from_text(text: str) -> list[str]:
    """Every supported ticker mentioned in the text, in order of appearance.

    Matches both the ticker symbols (NVDA, AAPL, ...) and the company-name
    aliases in config.COMPANY_NAME_ALIASES ("nvidia" -> NVDA, "apple" -> AAPL),
    so a name-only question resolves to the right company instead of the default.
    De-duplicated, preserving first-seen position.
    """
    upper = text.upper()
    hits: list[tuple[int, str]] = []

    for symbol in config.SUPPORTED_TICKERS:
        match = re.search(rf"\$?\b{re.escape(symbol)}\b", upper)
        if match:
            hits.append((match.start(), symbol))

    for name, symbol in config.COMPANY_NAME_ALIASES.items():
        match = re.search(rf"\b{re.escape(name.upper())}\b", upper)
        if match:
            hits.append((match.start(), symbol))

    hits.sort(key=lambda h: h[0])
    ordered: list[str] = []
    for _, symbol in hits:
        if symbol not in ordered:
            ordered.append(symbol)
    return ordered


def _resolve_ticker(question: str, state_ticker: str | None) -> tuple[str, str | None]:
    """Resolve the active ticker plus an optional human-readable note.

    Designed so an explicit selection is never silently overridden:
      * nothing mentioned   -> the selected/default ticker (no note);
      * exactly one         -> that ticker (the question's explicit intent),
                               noting an override only if it differs from the
                               selection;
      * several mentioned   -> keep the selected ticker if it is one of them
                               (respect the dropdown/flag), else take the first
                               by position. Only one company stream is retrieved,
                               so the choice is surfaced, never made silently.
    """
    selected = (state_ticker or "").strip().upper() or None
    if selected and selected not in config.SUPPORTED_TICKERS:
        selected = None

    mentioned = _tickers_from_text(question)

    if not mentioned:
        return (selected or config.TICKER, None)

    if len(mentioned) == 1:
        chosen = mentioned[0]
        if selected and chosen != selected:
            return (
                chosen,
                f"ticker resolved from the question text: {chosen} "
                f"(overrides selected {selected})",
            )
        return (chosen, None)

    # Several tickers mentioned — disambiguate without dropping the selection.
    if selected and selected in mentioned:
        others = ", ".join(t for t in mentioned if t != selected)
        return (
            selected,
            f"question also mentions {others}; analyzing the selected "
            f"ticker {selected}",
        )
    chosen = mentioned[0]
    return (
        chosen,
        f"question mentions multiple tickers ({', '.join(mentioned)}); "
        f"analyzing the first: {chosen}",
    )


def macro_node(state: BridgeState) -> dict:
    """Macro retrieval branch — Pinecone 'macro' namespace + FlashRank."""
    result = macro_agent.retrieve_macro(
        state["question"], state.get("expanded_queries")
    )
    return {
        "macro_context": result["context"],
        "macro_sources": result["sources"],
    }


def company_node(state: BridgeState) -> dict:
    """Company retrieval branch — per-ticker namespace + FlashRank."""
    ticker = state.get("ticker", config.TICKER)
    result = company_agent.retrieve_company(
        state["question"], ticker, state.get("expanded_queries")
    )
    return {
        "company_context": result["context"],
        "company_sources": result["sources"],
    }


def check_retrieval_quality(state: BridgeState) -> BridgeState:
    """Check whether both retrieval branches returned enough context.

    Routes to refusal if either stream is empty so the LLM is never asked to
    synthesise an answer from missing evidence.
    """
    macro_ok = len(state.get("macro_context", [])) >= config.MIN_MACRO_CHUNKS
    company_ok = len(state.get("company_context", [])) >= config.MIN_COMPANY_CHUNKS
    state["retrieval_ok"] = macro_ok and company_ok

    if not macro_ok or not company_ok:
        missing = []
        if not macro_ok:
            missing.append("macro")
        if not company_ok:
            missing.append("company")
        state["verdict"] = config.VERDICT_INSUFFICIENT
        state["answer"] = (
            f"I could not find sufficient context to answer "
            f"this question. The following retrieval streams "
            f"returned no results: {', '.join(missing)}. "
            f"This may mean the question is outside the scope "
            f"of the ingested corpus, or the corpus needs "
            f"refreshing. Try running: python main.py --refresh"
        )
        state["synthesis"] = ""
    return state


def route_after_retrieval(state: BridgeState) -> str:
    """Route to 'synthesis_node' if both streams have context, else 'refusal_node'."""
    macro_ok = len(state.get("macro_context", [])) >= config.MIN_MACRO_CHUNKS
    company_ok = len(state.get("company_context", [])) >= config.MIN_COMPANY_CHUNKS
    if macro_ok and company_ok:
        return "synthesis_node"
    return "refusal_node"


def refusal_node(state: BridgeState) -> BridgeState:
    """Terminal node — answer was already set by check_retrieval_quality."""
    return state


def synthesis_node(state: BridgeState) -> dict:
    """Reconcile both context streams into a synthesis + verdict."""
    result = synthesis_agent.run_synthesis(
        state["question"],
        state.get("macro_context", []),
        state.get("company_context", []),
    )
    return {"synthesis": result["synthesis"], "verdict": result["verdict"]}


def generate_answer(state: BridgeState) -> dict:
    """Produce the final cited investor answer from the synthesis."""
    answer = synthesis_agent.generate_answer(
        state["question"], state.get("synthesis", "")
    )
    return {"answer": answer}


# ──────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────
def build_graph():
    """Build and compile the LangGraph application."""
    builder = StateGraph(BridgeState)

    builder.add_node("parse_query", parse_query)
    builder.add_node("macro_node", macro_node)
    builder.add_node("company_node", company_node)
    builder.add_node("check_retrieval_quality", check_retrieval_quality)
    builder.add_node("refusal_node", refusal_node)
    builder.add_node("synthesis_node", synthesis_node)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "parse_query")
    # Fan out into the two parallel retrieval branches.
    builder.add_edge("parse_query", "macro_node")
    builder.add_edge("parse_query", "company_node")
    # Join: the quality gate waits for both branches to finish.
    builder.add_edge("macro_node", "check_retrieval_quality")
    builder.add_edge("company_node", "check_retrieval_quality")
    # Conditional routing: synthesise if context is sufficient, else refuse.
    builder.add_conditional_edges(
        "check_retrieval_quality",
        route_after_retrieval,
        {
            "synthesis_node": "synthesis_node",
            "refusal_node": "refusal_node",
        },
    )
    builder.add_edge("refusal_node", END)
    builder.add_edge("synthesis_node", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


def run(question: str, ticker: str = config.TICKER) -> BridgeState:
    """Convenience runner: execute the full graph for one question.

    `ticker` seeds the state; an explicit ticker mentioned in `question`
    still takes precedence inside parse_query.
    """
    app = build_graph()
    return app.invoke({"question": question, "ticker": ticker})
