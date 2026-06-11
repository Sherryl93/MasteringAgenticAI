"""Macro retrieval agent.

Retrieves macro-economic evidence (FOMC minutes, CPI, NFP) for a question:
  1. embed the query with BAAI/bge-base-en-v1.5 (same model as ingestion);
  2. pull top_k=10 candidates from the Pinecone "macro" namespace;
  3. rerank locally with FlashRank (ms-marco-MiniLM-L-12-v2);
  4. return the top 3 chunks plus their source badges (FOMC / CPI / NFP).

This module is consumed by the LangGraph `macro_node`.
"""

from __future__ import annotations

from pipeline import config, embed_store


def _badge(source_type: str) -> str:
    """Map a stored source_type to its display badge."""
    return config.SOURCE_BADGES.get(source_type, source_type.upper())


def retrieve_macro(question: str, expanded_queries: list[str] | None = None) -> dict:
    """Run multi-query macro retrieval + rerank.

    Retrieves candidates for each expanded sub-query from the "macro" namespace,
    deduplicates the pooled candidates by chunk text, then reranks the combined
    pool once against the original question and returns the top 3.

    If `expanded_queries` is not supplied (e.g. standalone use), it expands the
    question itself via the LLM. In the graph, parse_query supplies them so this
    node makes no extra LLM call.

    Returns {context, sources, chunks}.
    """
    if not expanded_queries:
        from agents.synthesis_agent import get_llm

        expanded_queries = embed_store.expand_query(question, get_llm())

    index = embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)

    # Hybrid candidate pool: dense (Pinecone) + sparse (BM25) fused across all
    # sub-queries, then reranked once by FlashRank.
    pooled = embed_store.hybrid_pool(index, config.NAMESPACE_MACRO, expanded_queries)
    top_chunks = embed_store.rerank(question, pooled)

    context = [c["text"] for c in top_chunks]
    sources = [config.citation_label(c) for c in top_chunks]
    for c in top_chunks:
        c["badge"] = _badge(c["source_type"])
        c["citation"] = config.citation_label(c)

    print(
        f"[macro_agent] {len(expanded_queries)} queries -> {len(pooled)} pooled "
        f"-> {len(top_chunks)} reranked"
    )
    return {"context": context, "sources": sources, "chunks": top_chunks}


if __name__ == "__main__":
    result = retrieve_macro("What is the Fed's current stance on rate cuts?")
    for badge, text in zip(result["sources"], result["context"]):
        print(f"\n[{badge}] {text[:240]}...")
