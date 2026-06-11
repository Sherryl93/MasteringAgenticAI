"""Company retrieval agent.

Retrieves company evidence (NVDA 10-K + earnings 8-Ks) for a question:
  1. embed the query with BAAI/bge-base-en-v1.5 (same model as ingestion);
  2. pull top_k=10 candidates from the Pinecone "company" namespace;
  3. rerank locally with FlashRank (ms-marco-MiniLM-L-12-v2);
  4. return the top 3 chunks plus their source badges (10-K / Earnings).

This module is consumed by the LangGraph `company_node`.
"""

from __future__ import annotations

from pipeline import config, embed_store


def _badge(source_type: str) -> str:
    """Map a stored source_type to its display badge."""
    return config.SOURCE_BADGES.get(source_type, source_type.upper())


def retrieve_company(
    question: str,
    ticker: str = config.TICKER,
    expanded_queries: list[str] | None = None,
) -> dict:
    """Run multi-query company retrieval + rerank for a ticker's namespace.

    The namespace is resolved at runtime as f"company-{ticker.lower()}". Retrieves
    candidates for each expanded sub-query, deduplicates the pool by chunk text,
    then reranks once against the original question and returns the top 3.

    If `expanded_queries` is not supplied, it expands the question via the LLM.
    In the graph, parse_query supplies them so this node makes no extra LLM call.

    Returns {context, sources, chunks, namespace}.
    """
    if not expanded_queries:
        from agents.synthesis_agent import get_llm

        expanded_queries = embed_store.expand_query(question, get_llm())

    namespace = config.company_namespace(ticker)
    index = embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)

    # Hybrid candidate pool: dense (Pinecone) + sparse (BM25) fused across all
    # sub-queries, then reranked once by FlashRank.
    pooled = embed_store.hybrid_pool(index, namespace, expanded_queries)
    top_chunks = embed_store.rerank(question, pooled)

    # Ensure the ticker rides along so citations can read "NVDA 10-K".
    for c in top_chunks:
        if not c.get("ticker"):
            c["ticker"] = ticker

    context = [c["text"] for c in top_chunks]
    sources = [config.citation_label(c) for c in top_chunks]
    for c in top_chunks:
        c["badge"] = _badge(c["source_type"])
        c["citation"] = config.citation_label(c)

    print(
        f"[company_agent] {len(expanded_queries)} queries -> {len(pooled)} pooled "
        f"-> {len(top_chunks)} reranked from '{namespace}'"
    )
    return {
        "context": context,
        "sources": sources,
        "chunks": top_chunks,
        "namespace": namespace,
    }


if __name__ == "__main__":
    result = retrieve_company("How did data center revenue trend last quarter?", "NVDA")
    for badge, text in zip(result["sources"], result["context"]):
        print(f"\n[{badge}] {text[:240]}...")
