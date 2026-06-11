"""Central configuration for the Macro-Micro Bridge Agent.

All constants, corpus URLs, and environment loading live here.
No magic strings should appear elsewhere in the codebase.
"""

import os
import re

from dotenv import load_dotenv

# Load .env once, at import time, for every entrypoint that imports config.
load_dotenv()

# ──────────────────────────────────────────────────────────────
# Core constants
# ──────────────────────────────────────────────────────────────
TICKER = "NVDA"  # default ticker when none is specified at runtime

# Whitelist of tickers the system will ingest / query. Validate against this
# everywhere a ticker enters the system.
SUPPORTED_TICKERS = ["NVDA", "AAPL", "MSFT", "AMD"]

# Common company names / aliases -> ticker symbol. Lets a question phrased with a
# name ("Is Apple a buy?") resolve to the right company instead of silently
# falling back to the default ticker. Keys are matched case-insensitively on
# word boundaries; values must be members of SUPPORTED_TICKERS.
COMPANY_NAME_ALIASES = {
    "NVIDIA": "NVDA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "AMD": "AMD",
    "ADVANCED MICRO DEVICES": "AMD",
}

PINECONE_INDEX = "macro-micro"
PINECONE_INDEX_NEBIUS = "macro-micro-nebius"

NAMESPACE_MACRO = "macro"  # universal macro corpus — never ticker-specific
NAMESPACE_COMPANY_PREFIX = "company"  # company namespaces are f"company-{ticker}"
NAMESPACE_NEBIUS = "nebius-exp"

# Fixed-size-chunking namespaces — evaluation only, never the live pipeline.
NAMESPACE_MACRO_FIXED = "macro-fixed"
NAMESPACE_COMPANY_FIXED_PREFIX = "company-fixed"

# Main pipeline embedder (local, HuggingFace sentence-transformers)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768

# Nebius comparison-experiment embedder (remote, OpenAI-compatible API)
EMBED_DIM_NEBIUS = 1536
NEBIUS_EMBED_MODEL = "BAAI/bge-en-icl"

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Retrieval / rerank
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 3

# Chunking
CHUNK_THRESHOLD = 65          # SemanticChunker percentile breakpoint amount
MAX_CHUNK_CHARS = 8000        # oversized-chunk hard cap

# Retrieval / ingestion quality filter — used by rerank() and the live semantic
# chunker to drop tiny low-signal fragments (lone names, two-word stubs) and
# near-duplicate boilerplate (e.g. the same earnings-call webcast notice
# repeated across quarters) before they reach the answer or the dashboard.
MIN_SIGNAL_CHARS = 100        # drop chunks shorter than this
MIN_SIGNAL_ALPHA_PCT = 0.40   # drop chunks below this alphabetic-char fraction
NEAR_DUP_JACCARD = 0.80       # drop a chunk this token-similar to a kept one

# Fixed-size chunking — evaluation comparison only (Config A)
FIXED_CHUNK_SIZE = 512
FIXED_CHUNK_OVERLAP = 64
FIXED_MIN_CHARS = 100         # drop chunks shorter than this
FIXED_MIN_ALPHA_PCT = 0.40    # drop chunks below this alphabetic-char fraction

# Reranker model (FlashRank, local, no API key)
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# Query expansion — number of LLM-rewritten sub-queries per question
QUERY_EXPANSION_COUNT = 3

# Pinecone serverless placement
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_METRIC = "cosine"

# Upsert batching
UPSERT_BATCH_SIZE = 100

# Retrieval-quality gate / refusal path
MIN_MACRO_CHUNKS = 1
MIN_COMPANY_CHUNKS = 1
VERDICT_INSUFFICIENT = "INSUFFICIENT CONTEXT"

# ──────────────────────────────────────────────────────────────
# Macro corpus — FOMC minutes (PDF via PyMuPDFLoader)
# ──────────────────────────────────────────────────────────────
FOMC_URLS = [
    # 2024 — rate plateau context
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20241107.pdf",
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20241218.pdf",
    # 2025 — rate cut cycle
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20250129.pdf",
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20250319.pdf",
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20251029.pdf",
    # 2026 — most current regime
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20260128.pdf",
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20260318.pdf",
    "https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20260429.pdf",
]

# ──────────────────────────────────────────────────────────────
# Macro corpus — BLS CPI releases (HTML via WebBaseLoader)
# ──────────────────────────────────────────────────────────────
BLS_CPI_URLS = [
    "https://www.bls.gov/news.release/cpi.htm",
    "https://www.bls.gov/news.release/archives/cpi_01132026.htm",
    "https://www.bls.gov/news.release/archives/cpi_10242025.htm",
]

# ──────────────────────────────────────────────────────────────
# Macro corpus — BLS NFP / Employment Situation releases (HTML)
# ──────────────────────────────────────────────────────────────
BLS_NFP_URLS = [
    "https://www.bls.gov/news.release/empsit.htm",
    "https://www.bls.gov/news.release/archives/empsit_01092026.htm",
    "https://www.bls.gov/news.release/archives/empsit_11202025.htm",
]

# HTML tags BeautifulSoup should keep when parsing BLS pages.
BLS_PARSE_TAGS = ["p", "table", "h2", "h3"]

# ──────────────────────────────────────────────────────────────
# Source-type metadata labels
# ──────────────────────────────────────────────────────────────
SRC_FOMC = "fomc"
SRC_CPI = "cpi"
SRC_NFP = "nfp"
SRC_10K = "10k"
SRC_EARNINGS = "earnings_transcript"

# Human-readable badges used by the UI / synthesis citations.
SOURCE_BADGES = {
    SRC_FOMC: "FOMC",
    SRC_CPI: "CPI",
    SRC_NFP: "NFP",
    SRC_10K: "10-K",
    SRC_EARNINGS: "Earnings",
}


def citation_label(meta: dict) -> str:
    """Build a specific citation from a chunk's metadata, not just a badge.

    Derives the filing/release date from the source URL where possible, e.g.
        FOMC minutes 2026-04-29   ·   CPI 2026-01-13   ·   NVDA 10-K
    Falls back to the plain badge when no date is encodable in the URL.
    """
    source_type = meta.get("source_type", "")
    url = meta.get("url", "") or ""
    badge = SOURCE_BADGES.get(source_type, source_type.upper() or "SOURCE")

    # FOMC minutes: .../fomcminutes20260429.pdf  -> YYYYMMDD
    m = re.search(r"fomcminutes(\d{4})(\d{2})(\d{2})", url)
    if m:
        return f"{badge} {m.group(1)}-{m.group(2)}-{m.group(3)}"

    # BLS archives: cpi_01132026.htm / empsit_01092026.htm  -> MMDDYYYY
    m = re.search(r"(?:cpi|empsit)_(\d{2})(\d{2})(\d{4})", url)
    if m:
        return f"{badge} {m.group(3)}-{m.group(1)}-{m.group(2)}"

    # Company filings: prefix with the ticker.
    if source_type in (SRC_10K, SRC_EARNINGS):
        ticker = meta.get("ticker", "")
        return f"{ticker} {badge}".strip()

    # Live BLS landing pages carry no date in the URL.
    if source_type in (SRC_CPI, SRC_NFP):
        return f"{badge} (latest)"

    return badge

# ──────────────────────────────────────────────────────────────
# Ticker validation / namespace resolution
# ──────────────────────────────────────────────────────────────
def validate_ticker(ticker: str) -> str:
    """Return the normalized (upper-case) ticker or raise if unsupported."""
    if not ticker:
        raise ValueError("No ticker provided.")
    normalized = ticker.strip().upper()
    if normalized not in SUPPORTED_TICKERS:
        raise ValueError(
            f"Unsupported ticker '{ticker}'. "
            f"Supported tickers: {', '.join(SUPPORTED_TICKERS)}."
        )
    return normalized


def company_namespace(ticker: str) -> str:
    """Resolve the per-ticker company namespace, e.g. NVDA -> 'company-nvda'."""
    normalized = validate_ticker(ticker)
    return f"{NAMESPACE_COMPANY_PREFIX}-{normalized.lower()}"


def company_fixed_namespace(ticker: str) -> str:
    """Per-ticker fixed-size company namespace, e.g. NVDA -> 'company-fixed-nvda'."""
    return f"{NAMESPACE_COMPANY_FIXED_PREFIX}-{validate_ticker(ticker).lower()}"


# ──────────────────────────────────────────────────────────────
# Environment access — fail loudly if a required key is missing
# ──────────────────────────────────────────────────────────────
REQUIRED_ENV_VARS = ["PINECONE_API_KEY", "NEBIUS_API_KEY"]


def require_env(name: str) -> str:
    """Return an environment variable's value or raise a clear error."""
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and set it."
        )
    return value


def validate_env() -> None:
    """Validate all required environment variables are present at startup."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your keys."
        )


def pinecone_api_key() -> str:
    return require_env("PINECONE_API_KEY")


def nebius_api_key() -> str:
    return require_env("NEBIUS_API_KEY")
