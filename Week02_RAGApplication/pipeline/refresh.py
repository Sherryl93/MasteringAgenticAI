"""Rebuild the vector store from scratch.

Orchestrates the full pipeline:
    load -> semantic chunk -> embed -> delete namespace -> upsert

Main index (PINECONE_INDEX, 768-dim):
    namespace "macro"        <- FOMC + CPI + NFP  (universal, ticker-independent)
    namespace "company-<t>"  <- one per ticker, e.g. company-nvda, company-aapl

Nebius experiment index (PINECONE_INDEX_NEBIUS, 1536-dim):
    namespace "nebius-exp" <- macro + default-ticker company, BAAI/bge-en-icl
    (built only when explicitly requested; never used by the live agent)

Usage:
    python -m pipeline.refresh --macro                 # macro only
    python -m pipeline.refresh --company NVDA          # one ticker (semantic)
    python -m pipeline.refresh --fixed NVDA            # one ticker (fixed-size eval)
    python -m pipeline.refresh --all                   # macro + every ticker
    python -m pipeline.refresh --nebius                # experiment index
"""

from __future__ import annotations

import argparse

from langchain_core.documents import Document

from pipeline import config, embed_store
from pipeline.ingest_company import ingest_company
from pipeline.ingest_macro import load_macro_documents


def refresh_macro() -> int:
    """Rebuild the universal macro namespace (ticker-independent)."""
    config.validate_env()
    index = embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)

    print("\n=== Refreshing MACRO namespace (universal) ===")
    macro_docs = load_macro_documents()
    macro_chunks = embed_store.chunk_documents(macro_docs)
    n = embed_store.upsert_chunks(index, config.NAMESPACE_MACRO, macro_chunks)
    print(f"[refresh] MACRO done — {n} vectors in '{config.NAMESPACE_MACRO}'")
    return n


def refresh_company(ticker: str) -> int:
    """Rebuild one ticker's company namespace (f"company-{ticker}")."""
    config.validate_env()
    ticker = config.validate_ticker(ticker)
    namespace = config.company_namespace(ticker)
    index = embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)

    print(f"\n=== Refreshing COMPANY namespace for {ticker} ({namespace}) ===")
    company_docs = ingest_company(ticker)
    company_chunks = embed_store.chunk_documents(company_docs)
    n = embed_store.upsert_chunks(index, namespace, company_chunks)
    print(f"[refresh] COMPANY done — {n} vectors in '{namespace}'")
    return n


def refresh_fixed(ticker: str = config.TICKER) -> tuple[int, int]:
    """Rebuild the fixed-size-chunking namespaces (evaluation only).

    Same source corpus, but chunked with chunk_documents_fixed() into the
    'macro-fixed' and 'company-fixed-<ticker>' namespaces. upsert_chunks clears
    each namespace first and embeds with the shared BGE embedder. Returns
    (macro_chunks, company_chunks).
    """
    config.validate_env()
    ticker = config.validate_ticker(ticker)
    index = embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)
    company_ns = config.company_fixed_namespace(ticker)

    print(f"\n=== Refreshing FIXED-SIZE namespaces for {ticker} ===")
    macro_docs = load_macro_documents()
    company_docs = ingest_company(ticker)
    macro_chunks = embed_store.chunk_documents_fixed(macro_docs)
    company_chunks = embed_store.chunk_documents_fixed(company_docs)

    n_macro = embed_store.upsert_chunks(
        index, config.NAMESPACE_MACRO_FIXED, macro_chunks
    )
    n_company = embed_store.upsert_chunks(index, company_ns, company_chunks)
    print(f"[refresh] Fixed macro:   {n_macro} chunks → Pinecone")
    print(f"[refresh] Fixed company: {n_company} chunks → Pinecone")
    return n_macro, n_company


def refresh_main(ticker: str = config.TICKER) -> None:
    """Refresh semantic (live) + fixed (eval) namespaces in one run."""
    n_semantic_macro = refresh_macro()
    n_semantic_company = refresh_company(ticker)
    n_fixed_macro, n_fixed_company = refresh_fixed(ticker)

    print("\n[refresh] ===== COMPLETION SUMMARY =====")
    print(f"[refresh] Semantic macro:   {n_semantic_macro} chunks")
    print(f"[refresh] Semantic company: {n_semantic_company} chunks")
    print(f"[refresh] Fixed macro:      {n_fixed_macro} chunks")
    print(f"[refresh] Fixed company:    {n_fixed_company} chunks")


def refresh_all() -> None:
    """Refresh the macro corpus once, then every supported ticker's company."""
    refresh_macro()
    for ticker in config.SUPPORTED_TICKERS:
        refresh_company(ticker)
    print(
        f"\n[refresh] ALL done — macro + company for "
        f"{', '.join(config.SUPPORTED_TICKERS)}"
    )


def refresh_nebius(ticker: str = config.TICKER) -> None:
    """Rebuild the Nebius experiment namespace (1536-dim) on its own index."""
    config.validate_env()
    ticker = config.validate_ticker(ticker)
    index = embed_store.ensure_index(
        config.PINECONE_INDEX_NEBIUS, config.EMBED_DIM_NEBIUS
    )

    print("\n=== Refreshing NEBIUS experiment namespace ===")
    docs: list[Document] = []
    docs.extend(load_macro_documents())
    docs.extend(ingest_company(ticker))

    chunks = embed_store.chunk_documents(docs)
    nebius_embedder = embed_store.NebiusEmbeddings()
    n = embed_store.upsert_chunks(
        index, config.NAMESPACE_NEBIUS, chunks, embedder=nebius_embedder
    )
    print(
        f"\n[refresh] NEBIUS done — {n} vectors in namespace "
        f"'{config.NAMESPACE_NEBIUS}' on index '{config.PINECONE_INDEX_NEBIUS}'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the vector store.")
    parser.add_argument("--macro", action="store_true", help="rebuild macro namespace")
    parser.add_argument(
        "--company",
        metavar="TICKER",
        help="rebuild one ticker's company namespace",
    )
    parser.add_argument(
        "--all", action="store_true", help="rebuild macro + every supported ticker"
    )
    parser.add_argument(
        "--fixed",
        metavar="TICKER",
        help="rebuild only the fixed-size eval namespaces for one ticker",
    )
    parser.add_argument(
        "--nebius", action="store_true", help="rebuild Nebius experiment index"
    )
    args = parser.parse_args()

    # Default to semantic + fixed for the default ticker if nothing specified.
    if not (args.macro or args.company or args.all or args.fixed or args.nebius):
        refresh_main(config.TICKER)
        return

    if args.all:
        refresh_all()
    if args.macro:
        refresh_macro()
    if args.company:
        refresh_company(args.company)
    if args.fixed:
        refresh_fixed(args.fixed)
    if args.nebius:
        refresh_nebius()


if __name__ == "__main__":
    main()
