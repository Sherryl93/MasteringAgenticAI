"""Embedding, chunking, and Pinecone vector-store operations.

Single source of truth for:
  - the local BGE embedder (BAAI/bge-base-en-v1.5, 768-dim) used for BOTH
    ingestion and retrieval — never mix models across the two;
  - the SemanticChunker strategy (percentile @ 65) with an 8000-char hard cap;
  - Pinecone serverless index creation and namespaced batch upserts;
  - the NebiusEmbeddings adapter (BAAI/bge-en-icl, 1536-dim) used ONLY by the
    separate comparison experiment — never by the main pipeline.
"""

from __future__ import annotations

import pickle
import re
import threading
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from pipeline import config

# Module-level singletons so heavy models load once per process.
_embedder: HuggingFaceEmbeddings | None = None
_pinecone: Pinecone | None = None
_reranker = None  # FlashRank Ranker, lazily loaded

# Lock guarding lazy model init — the macro/company branches run in parallel and
# must not download/load the same model twice (causes WinError 32 on Windows).
_model_lock = threading.Lock()

# FlashRank downloads its model here. A stable local dir avoids the locked/corrupt
# zip problems seen with FlashRank's default C:\tmp path on Windows.
RERANK_CACHE_DIR = Path(".flashrank_cache")

# Local BM25 (sparse) indexes are pickled here, one file per namespace.
BM25_DIR = Path("bm25_store")
_bm25_cache: dict[str, dict | None] = {}


# ──────────────────────────────────────────────────────────────
# Local embedder — BAAI/bge-base-en-v1.5 (ingestion AND retrieval)
# ──────────────────────────────────────────────────────────────
def get_embedder() -> HuggingFaceEmbeddings:
    """Return the shared local BGE embedder (768-dim, normalized)."""
    global _embedder
    if _embedder is None:
        with _model_lock:  # double-checked: parallel branches load it once
            if _embedder is None:
                _embedder = HuggingFaceEmbeddings(
                    model_name=config.EMBED_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embedder


# ──────────────────────────────────────────────────────────────
# Nebius embedder — BAAI/bge-en-icl (1536-dim) — EXPERIMENT ONLY
# ──────────────────────────────────────────────────────────────
class NebiusEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by the Nebius OpenAI API.

    Used solely for the embedding-comparison experiment. It must NOT be used
    for main-pipeline ingestion or retrieval.
    """

    def __init__(
        self,
        model: str = config.NEBIUS_EMBED_MODEL,
        base_url: str = config.NEBIUS_BASE_URL,
        api_key: str | None = None,
    ) -> None:
        # Imported lazily so the main pipeline never requires the openai client.
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or config.nebius_api_key(),
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


# ──────────────────────────────────────────────────────────────
# Chunking — SemanticChunker (percentile @ 65) + 8000-char cap
# ──────────────────────────────────────────────────────────────
def chunk_documents(docs: list[Document]) -> list[Document]:
    """Semantically chunk documents, then hard-split oversized chunks.

    Identical strategy for macro and company corpora.
    """
    chunker = SemanticChunker(
        embeddings=get_embedder(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=config.CHUNK_THRESHOLD,
    )
    semantic_chunks = chunker.split_documents(docs)

    fallback = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_CHARS,
        chunk_overlap=200,
    )
    final_chunks: list[Document] = []
    for chunk in semantic_chunks:
        if len(chunk.page_content) > config.MAX_CHUNK_CHARS:
            final_chunks.extend(fallback.split_documents([chunk]))
        else:
            final_chunks.append(chunk)

    # Drop low-signal fragments (lone names, two-word stubs, number-only rows) so
    # they never get embedded/stored — the same quality bar rerank() applies.
    kept = [
        c
        for c in final_chunks
        if not _is_low_signal(c.page_content)
        and not _is_machine_metadata(c.page_content)
    ]

    print(
        f"[embed] chunked {len(docs)} docs -> {len(semantic_chunks)} semantic "
        f"-> {len(final_chunks)} sized -> {len(kept)} kept "
        f"(dropped {len(final_chunks) - len(kept)} low-signal)"
    )
    return kept


def chunk_documents_fixed(documents: list[Document]) -> list[Document]:
    """Fixed-size chunking (evaluation only) — RecursiveCharacterTextSplitter.

    Splits at FIXED_CHUNK_SIZE / FIXED_CHUNK_OVERLAP, then drops low-signal
    fragments: chunks shorter than FIXED_MIN_CHARS, and chunks whose alphabetic
    characters fall below FIXED_MIN_ALPHA_PCT of total (number-only table
    fragments, page headers). Source metadata is preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.FIXED_CHUNK_SIZE,
        chunk_overlap=config.FIXED_CHUNK_OVERLAP,
    )
    raw_chunks = splitter.split_documents(documents)

    filtered: list[Document] = []
    for chunk in raw_chunks:
        text = chunk.page_content
        if len(text) < config.FIXED_MIN_CHARS:
            continue
        alpha = sum(ch.isalpha() for ch in text)
        if len(text) == 0 or (alpha / len(text)) < config.FIXED_MIN_ALPHA_PCT:
            continue
        # Drop chunks that are mostly XBRL noise (>30% of lines look like
        # type declarations / long camelCase identifiers).
        if _xbrl_line_fraction(text) > 0.30:
            continue
        filtered.append(chunk)

    print(
        f"[embed] fixed-chunked {len(documents)} docs -> {len(raw_chunks)} raw "
        f"-> {len(filtered)} kept (size={config.FIXED_CHUNK_SIZE}, "
        f"overlap={config.FIXED_CHUNK_OVERLAP})"
    )
    return filtered


# XBRL line signatures: a colon-separated type declaration, a long camelCase
# identifier (>20 chars), or a prefix_CamelCase identifier.
_XBRL_COLON = re.compile(r"[a-z]+:[A-Za-z]")
_XBRL_PREFIX = re.compile(r"[a-z]{2,10}_[A-Z][A-Za-z]{10,}")
_XBRL_CAMEL = re.compile(r"[A-Za-z]{20,}")


def _is_xbrl_line(line: str) -> bool:
    """True if a line looks like XBRL metadata rather than prose."""
    if _XBRL_COLON.search(line) or _XBRL_PREFIX.search(line):
        return True
    for token in _XBRL_CAMEL.findall(line):
        if re.search(r"[a-z][A-Z]", token):  # camelCase boundary
            return True
    return False


def _xbrl_line_fraction(text: str) -> float:
    """Fraction of non-blank lines in `text` that look like XBRL metadata."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    bad = sum(1 for ln in lines if _is_xbrl_line(ln))
    return bad / len(lines)


def check_chunk_quality(chunks: list) -> dict:
    """Scan chunks for known garbage patterns; return a quality report.

    Returns {total, clean, flagged, flagged_examples} where flagged_examples is
    the first 3 flagged chunk snippets.
    """
    xbrl_patterns = [
        r"\b[a-z]+:[A-Z][a-z]",  # namespace:TypeName
        r"[a-z]{2,10}_[A-Z][A-Za-z]{10,}",  # prefix_CamelCaseIdentifier
        r"Balance Type\s*:",
        r"Period Type\s*:",
        r"Data Type\s*:",
    ]
    flagged = []
    for chunk in chunks:
        text = chunk.page_content
        if any(re.search(p, text) for p in xbrl_patterns):
            flagged.append(text[:150])
    return {
        "total": len(chunks),
        "clean": len(chunks) - len(flagged),
        "flagged": len(flagged),
        "flagged_examples": flagged[:3],
    }


# ──────────────────────────────────────────────────────────────
# Pinecone — client, index lifecycle, namespaced batch upserts
# ──────────────────────────────────────────────────────────────
def get_pinecone() -> Pinecone:
    """Return the shared Pinecone client."""
    global _pinecone
    if _pinecone is None:
        _pinecone = Pinecone(api_key=config.pinecone_api_key())
    return _pinecone


def ensure_index(index_name: str, dimension: int):
    """Create the serverless index if absent, then return a handle to it.

    Tolerant of a concurrent create: the parallel macro/company branches can both
    reach here before the index exists, so an "already exists" conflict from a
    racing branch is treated as success rather than an error.
    """
    pc = get_pinecone()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        print(f"[embed] creating Pinecone index '{index_name}' (dim={dimension})")
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=config.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — another branch may have won the race
            msg = str(exc).lower()
            if "already exists" in msg or "409" in msg or "conflict" in msg:
                print(f"[embed] index '{index_name}' already created concurrently")
            else:
                raise
    return pc.Index(index_name)


def _chunk_metadata(doc: Document) -> dict:
    """Flatten a chunk's payload into Pinecone-storable metadata."""
    meta = {
        "text": doc.page_content,
        "source_type": doc.metadata.get("source_type", "unknown"),
        "url": doc.metadata.get("url", ""),
    }
    if "ticker" in doc.metadata:
        meta["ticker"] = doc.metadata["ticker"]
    return meta


def upsert_chunks(
    index,
    namespace: str,
    chunks: list[Document],
    embedder: Embeddings | None = None,
) -> int:
    """Embed chunks and upsert into a namespace, replacing prior vectors.

    Always clears the namespace first, then upserts in batches of 100.
    Returns the number of vectors written.
    """
    if not chunks:
        # Clear dense + sparse so a failed/empty re-ingest never leaves stale
        # data to be served silently — the retrieval gate then refuses honestly
        # instead of answering from last refresh's corpus.
        print(
            f"[embed] no chunks for namespace '{namespace}' — clearing dense + "
            f"sparse indexes"
        )
        _delete_namespace(index, namespace)
        _delete_bm25_index(namespace)
        return 0

    embedder = embedder or get_embedder()
    texts = [c.page_content for c in chunks]
    vectors_values = embedder.embed_documents(texts)

    records = []
    for chunk, values in zip(chunks, vectors_values):
        records.append(
            {
                "id": str(uuid.uuid4()),
                "values": values,
                "metadata": _chunk_metadata(chunk),
            }
        )

    # Always delete existing vectors before re-uploading.
    _delete_namespace(index, namespace)

    batch = config.UPSERT_BATCH_SIZE
    for start in range(0, len(records), batch):
        index.upsert(vectors=records[start : start + batch], namespace=namespace)
        print(
            f"[embed] upserted {min(start + batch, len(records))}/{len(records)} "
            f"into '{namespace}'"
        )

    # Build the matching local BM25 (sparse) index over the same chunks so the
    # live agents can retrieve hybrid (dense + sparse) from this namespace.
    build_bm25_index(namespace, chunks)
    return len(records)


def _delete_namespace(index, namespace: str) -> None:
    """Delete all vectors in a namespace, tolerating a not-yet-existing one."""
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"[embed] cleared existing vectors in namespace '{namespace}'")
    except Exception as exc:  # noqa: BLE001 — namespace may not exist yet
        print(f"[embed] namespace '{namespace}' nothing to clear ({exc})")


def namespace_count(index, namespace: str) -> int:
    """Return the vector count for a namespace (0 if absent)."""
    try:
        stats = index.describe_index_stats()
        ns = stats.get("namespaces", {}) or {}
        return int(ns.get(namespace, {}).get("vector_count", 0))
    except Exception:  # noqa: BLE001
        return 0


# ──────────────────────────────────────────────────────────────
# Retrieval — Pinecone similarity search + FlashRank reranking
# ──────────────────────────────────────────────────────────────
def get_reranker():
    """Return the shared FlashRank reranker (local, no API key)."""
    global _reranker
    if _reranker is None:
        with _model_lock:  # double-checked: avoid concurrent model download
            if _reranker is None:
                from flashrank import Ranker

                RERANK_CACHE_DIR.mkdir(exist_ok=True)
                _reranker = Ranker(
                    model_name=config.RERANK_MODEL,
                    cache_dir=str(RERANK_CACHE_DIR),
                )
    return _reranker


def query_namespace(
    index,
    namespace: str,
    question: str,
    top_k: int = config.TOP_K_RETRIEVE,
) -> list[dict]:
    """Embed the query and pull top_k candidate chunks from a namespace.

    Uses the same local BGE embedder as ingestion — never a different model.
    Returns dicts: {text, source_type, url, score}.
    """
    query_vec = get_embedder().embed_query(question)
    resp = index.query(
        namespace=namespace,
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
    )
    candidates = []
    for match in resp.get("matches", []):
        meta = match.get("metadata", {}) or {}
        candidates.append(
            {
                "text": meta.get("text", ""),
                "source_type": meta.get("source_type", "unknown"),
                "url": meta.get("url", ""),
                "ticker": meta.get("ticker", ""),
                "score": match.get("score", 0.0),
            }
        )
    return candidates


def _is_low_signal(text: str) -> bool:
    """True for tiny / non-prose fragments (lone names, stray numbers, stubs)."""
    t = (text or "").strip()
    if len(t) < config.MIN_SIGNAL_CHARS:
        return True
    alpha = sum(ch.isalpha() for ch in t)
    return alpha / len(t) < config.MIN_SIGNAL_ALPHA_PCT


# Markers that appear only in XBRL/iXBRL taxonomy + JSON metadata, never in the
# narrative prose of a 10-K or earnings release.
_XBRL_MARKERS = (
    "xbrltype", "auth_ref", "namespace prefix", "xbrli:", "us-gaap:",
    "dei_document", "durationitemtype", "monetaryitemtype", "stringitemtype",
    "sharesitemtype", "data type:", "period type:", "balance type:",
)
_TAXONOMY_TOKEN = re.compile(r"\b[a-z]{2,6}_[A-Z][A-Za-z]{5,}")
# EDGAR R-file index / table-of-contents line, e.g.
# "false false R84.htm 9955564 - Disclosure - Segment Information ...".
_EDGAR_INDEX = re.compile(r"(?:true|false)\s+(?:true|false)\s+R\d+\.htm", re.IGNORECASE)


def _is_machine_metadata(text: str) -> bool:
    """True for XBRL taxonomy / JSON schema / EDGAR-index chunks (machine
    plumbing, not prose).

    Ticker-agnostic: catches the structured-data dictionary embedded in any
    SEC filing — element definitions (``<ticker>_CamelCaseConcept``), data-type
    tags (``xbrli:durationItemType``), ``"auth_ref": []`` JSON blobs, and the
    EDGAR R-file index/table-of-contents — all of which read as gibberish to a
    human yet survive HTML/tag stripping.
    """
    lower = text.lower()
    if any(m in lower for m in _XBRL_MARKERS):
        return True
    if _EDGAR_INDEX.search(text):           # "false false R84.htm ..." index lines
        return True
    if lower.count(".htm") >= 3:            # dense R-file table-of-contents block
        return True
    # A chunk dominated by boolean index tokens (false/true line starts).
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        bool_lines = sum(
            1 for ln in lines if ln.lstrip().lower().startswith(("false", "true"))
        )
        if bool_lines / len(lines) > 0.4:
            return True
    if text:  # dense JSON / markup punctuation => schema blob, not prose
        punct = sum(text.count(ch) for ch in '{}[]"')
        if punct / len(text) > 0.06:
            return True
    return len(_TAXONOMY_TOKEN.findall(text)) >= 2  # prefix_CamelCase elements


def _is_near_duplicate(text: str, kept: list[dict]) -> bool:
    """True if `text` is near-identical to any already-kept chunk.

    Token Jaccard plus a containment check, so boilerplate that repeats across
    documents (e.g. the same earnings webcast notice each quarter) is collapsed
    even when a few words differ.
    """
    tokens = set(re.findall(r"\w+", text.lower()))
    if not tokens:
        return True
    for item in kept:
        other = set(re.findall(r"\w+", (item.get("text") or "").lower()))
        if not other:
            continue
        inter = len(tokens & other)
        union = len(tokens | other)
        if union and inter / union >= config.NEAR_DUP_JACCARD:
            return True
        smaller = min(len(tokens), len(other))
        if smaller and inter / smaller >= 0.90:  # one chunk contained in another
            return True
    return False


def rerank(
    question: str,
    candidates: list[dict],
    top_n: int = config.TOP_K_RERANK,
) -> list[dict]:
    """Rerank with FlashRank, dropping low-signal and near-duplicate chunks.

    FlashRank scores every candidate; we then walk them best-first and keep the
    top_n that are neither low-signal fragments nor near-duplicates of a chunk
    already kept. If the filter would leave nothing, fall back to the raw top_n
    so a query never refuses purely because of the quality filter.
    """
    if not candidates:
        return []
    from flashrank import RerankRequest

    passages = [
        {"id": i, "text": c["text"], "meta": c} for i, c in enumerate(candidates)
    ]
    ranked = get_reranker().rerank(RerankRequest(query=question, passages=passages))

    scored: list[dict] = []
    for r in ranked:
        item = dict(r["meta"])
        item["rerank_score"] = float(r.get("score", 0.0))
        scored.append(item)

    out: list[dict] = []
    for item in scored:
        if len(out) >= top_n:
            break
        text = item.get("text", "")
        if (
            _is_low_signal(text)
            or _is_machine_metadata(text)
            or _is_near_duplicate(text, out)
        ):
            continue
        out.append(item)

    if not out:  # everything filtered — better some context than none
        out = scored[:top_n]
    return out


def retrieve_and_rerank(
    index,
    namespace: str,
    question: str,
) -> list[dict]:
    """Full retrieval: top_k similarity search -> FlashRank top_n."""
    candidates = query_namespace(index, namespace, question)
    return rerank(question, candidates)


def retrieve_fixed(query: str, namespace: str, k: int = 3) -> list[dict]:
    """Plain similarity retrieval with NO reranking (Config A / B in eval).

    Embeds the query with the same local BGE embedder and returns the top-k
    nearest chunks from the given namespace. Used only by the evaluation harness.
    """
    index = ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)
    candidates = query_namespace(index, namespace, query, top_k=k)
    return candidates[:k]


# ──────────────────────────────────────────────────────────────
# Sparse retrieval — local BM25 (no API, no cost) + hybrid fusion
# ──────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Lowercase word-token split used for BM25 indexing and querying."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(namespace: str, chunks: list[Document]) -> None:
    """Build + pickle a BM25 index over the same chunks uploaded to a namespace."""
    from rank_bm25 import BM25Okapi

    records = [_chunk_metadata(c) for c in chunks]  # {text, source_type, url, ...}
    tokenized = [_tokenize(r["text"]) for r in records]
    if not tokenized:
        return
    bm25 = BM25Okapi(tokenized)

    BM25_DIR.mkdir(exist_ok=True)
    path = BM25_DIR / f"{namespace}.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"bm25": bm25, "records": records}, fh)
    # Invalidate any cached copy; load_bm25 re-reads it (mtime-keyed) on next use.
    _bm25_cache.pop(namespace, None)
    print(f"[embed] built BM25 index for '{namespace}' ({len(records)} chunks)")


def load_bm25(namespace: str) -> dict | None:
    """Load (and cache) the BM25 index for a namespace, or None if absent.

    Cache entries are keyed on the pickle's mtime, and a missing index is NOT
    cached — so an index built or rebuilt by a later refresh (even from another
    process, e.g. while a Streamlit session stays up) is picked up instead of a
    stale in-memory result.
    """
    path = BM25_DIR / f"{namespace}.pkl"
    if not path.exists():
        _bm25_cache.pop(namespace, None)
        return None
    mtime = path.stat().st_mtime
    cached = _bm25_cache.get(namespace)
    if cached is not None and cached.get("_mtime") == mtime:
        return cached
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    data["_mtime"] = mtime
    _bm25_cache[namespace] = data
    return data


def _delete_bm25_index(namespace: str) -> None:
    """Remove a namespace's BM25 index file and cache entry (tolerates absence)."""
    _bm25_cache.pop(namespace, None)
    path = BM25_DIR / f"{namespace}.pkl"
    try:
        path.unlink()
        print(f"[embed] removed BM25 index for '{namespace}'")
    except FileNotFoundError:
        pass


def bm25_search(
    namespace: str, query: str, top_k: int = config.TOP_K_RETRIEVE
) -> list[dict]:
    """Return the top_k BM25 (sparse) matches for a query, or [] if no index."""
    data = load_bm25(namespace)
    if not data:
        return []
    bm25 = data["bm25"]
    records = data["records"]
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:top_k]:
        if scores[i] <= 0:
            break
        item = dict(records[i])
        item["score"] = float(scores[i])
        out.append(item)
    return out


def fuse_rrf(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over any number of ranked candidate lists.

    Each chunk's fused score is sum(1 / (k + rank)) across the lists it appears
    in, deduplicated by chunk text. Returns candidates ordered by fused score.
    """
    scores: dict[str, float] = {}
    store: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            key = item["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            store.setdefault(key, item)
    ordered = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [store[kk] for kk in ordered]


def hybrid_pool(index, namespace: str, queries: list[str]) -> list[dict]:
    """Fuse dense (Pinecone) + sparse (BM25) candidates across sub-queries.

    Runs every sub-query through both the dense and sparse retrievers, then
    Reciprocal-Rank-Fuses all the resulting ranked lists into one candidate pool
    (deduplicated by chunk text). The pool is meant to be handed to FlashRank.
    Gracefully degrades to dense-only multi-query fusion if no BM25 index exists.
    """
    ranked_lists: list[list[dict]] = []
    for q in queries:
        ranked_lists.append(query_namespace(index, namespace, q))
    for q in queries:
        ranked_lists.append(bm25_search(namespace, q))
    return fuse_rrf(*ranked_lists)


# ──────────────────────────────────────────────────────────────
# Query expansion — LLM rewrites a vague question into sub-queries
# ──────────────────────────────────────────────────────────────
def expand_query(question: str, llm) -> list[str]:
    """Rewrite the question into retrieval-optimized sub-queries.

    Uses the LLM to produce QUERY_EXPANSION_COUNT sub-queries phrased to match
    the vocabulary of SEC filings, Fed minutes, and BLS reports. Returns the
    original question plus the expansions, deduplicated and order-preserved.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a financial research assistant. "
                "Rewrite the following investment question into "
                "{n} specific search queries optimized for "
                "retrieving relevant passages from SEC filings, "
                "Fed minutes, and BLS economic reports. "
                "Output only the queries, one per line, "
                "no numbering, no explanation.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "n": config.QUERY_EXPANSION_COUNT})

    expanded = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    all_queries = [question] + expanded
    # Deduplicate preserving order.
    seen = set()
    unique = []
    for q in all_queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[: config.QUERY_EXPANSION_COUNT + 1]
