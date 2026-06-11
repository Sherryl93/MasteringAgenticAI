"""Command-line entrypoint for the Macro-Micro Bridge Agent.

Examples:
    # Rebuild the vector store (ingest -> chunk -> embed -> upsert)
    python main.py --refresh

    # Ask a question (runs the full macro + company + synthesis graph)
    python main.py --question "Should I be long NVDA given the macro environment?"

    # No question given -> uses the default demo question
    python main.py
"""

from __future__ import annotations

import argparse

from agents import graph, synthesis_agent
from pipeline import config

DEFAULT_QUESTION = "Should I be long NVDA given the current macro environment?"

# Map canonical verdicts to a short console label.
_VERDICT_TAG = {
    synthesis_agent.VERDICT_BULLISH: "🟢 ALIGNED BULLISH",
    synthesis_agent.VERDICT_BEARISH: "🔴 ALIGNED BEARISH",
    synthesis_agent.VERDICT_CONFLICT: "🟠 CONFLICT DETECTED",
    synthesis_agent.VERDICT_UNKNOWN: "⚪ UNDETERMINED",
}


def _print_stream(title: str, contexts: list[str], sources: list[str]) -> None:
    """Pretty-print one retrieval stream (macro or company)."""
    print(f"\n──────── {title} ────────")
    if not contexts:
        print("(no chunks retrieved)")
        return
    for i, (badge, text) in enumerate(zip(sources, contexts), start=1):
        snippet = text.strip().replace("\n", " ")
        print(f"\n[{i}] ({badge}) {snippet[:280]}...")


def run_question(question: str, ticker: str) -> None:
    """Execute the graph for one question and print the full result."""
    config.validate_env()
    ticker = config.validate_ticker(ticker)
    print(f"\n=== Macro-Micro Bridge Agent — {ticker} ===")
    print(f"Question: {question}")

    state = graph.run(question, ticker)
    resolved = state.get("ticker", ticker)
    note = state.get("ticker_note")
    if note:
        print(f"(note: {note})")
    elif resolved != ticker:
        print(f"(ticker resolved from query text: {resolved})")

    _print_stream(
        "MACRO CONTEXT",
        state.get("macro_context", []),
        state.get("macro_sources", []),
    )
    _print_stream(
        "COMPANY CONTEXT",
        state.get("company_context", []),
        state.get("company_sources", []),
    )

    verdict = state.get("verdict", synthesis_agent.VERDICT_UNKNOWN)
    print(f"\n──────── SYNTHESIS ────────")
    print(f"\nVerdict: {_VERDICT_TAG.get(verdict, verdict)}\n")
    print(state.get("synthesis", "(no synthesis produced)"))

    print(f"\n──────── FINAL ANSWER ────────\n")
    print(state.get("answer", "(no answer produced)"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro-Micro Bridge Agent CLI")
    parser.add_argument(
        "--question", "-q", default=None, help="investment question to analyze"
    )
    parser.add_argument(
        "--ticker",
        "-t",
        default=config.TICKER,
        help=f"ticker to analyze (default {config.TICKER}; "
        f"one of {', '.join(config.SUPPORTED_TICKERS)})",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild the main vector store first"
    )
    args = parser.parse_args()

    # Validate the ticker up front so every entrypoint fails fast.
    ticker = config.validate_ticker(args.ticker)
    print(f"[main] using ticker: {ticker}")

    if args.refresh:
        # Imported lazily so a plain query run doesn't import ingestion deps.
        from pipeline.refresh import refresh_main

        refresh_main(ticker)

    # If the user only asked to refresh, don't force a query.
    if args.question is None and args.refresh:
        return

    run_question(args.question or DEFAULT_QUESTION, ticker)


if __name__ == "__main__":
    main()
