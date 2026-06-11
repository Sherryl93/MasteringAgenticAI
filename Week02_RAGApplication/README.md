# Macro-Micro Bridge Agent

> Grounded investment-thesis generation from macro **and** company signals — a LangGraph multi-agent RAG pipeline.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg">
  <img alt="LangGraph" src="https://img.shields.io/badge/orchestration-LangGraph-7c3aed.svg">
  <img alt="Vector store" src="https://img.shields.io/badge/vector%20store-Pinecone-0d9488.svg">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Llama--3.3--70B%20via%20Nebius-f97362.svg">
  <img alt="No OpenAI" src="https://img.shields.io/badge/OpenAI-not%20used-lightgrey.svg">
</p>

The agent answers questions like *"Should I be long NVDA given the current macro environment?"* by running **two independent retrieval chains in parallel** — one over macro-economic sources (Fed minutes, CPI, jobs reports), one over a company's SEC filings — then reconciling them into a single cited verdict: **ALIGNED BULLISH**, **ALIGNED BEARISH**, or **CONFLICT DETECTED**. If retrieval comes back empty, it **refuses** rather than hallucinating.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [Usage](#usage)
- [Dynamic ticker selection](#dynamic-ticker-selection)
- [Verdicts & the refusal path](#verdicts--the-refusal-path)
- [Evaluation (RAGAS, three configs)](#evaluation-ragas-three-configs)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Two parallel retrieval chains** — macro (universal) and company (per-ticker), reconciled by a synthesis node.
- **Dynamic ticker selection** — choose NVDA / AAPL / MSFT / AMD at runtime (CLI flag, Streamlit selectbox, or detected from the question text). Each ticker gets its own Pinecone namespace; the macro corpus is shared.
- **Grounded, no-hallucination design** — a retrieval-quality gate routes to an explicit **refusal** when either stream is empty.
- **Query expansion** — the LLM rewrites each question into multiple retrieval-optimized sub-queries to improve recall on vague questions.
- **Hybrid retrieval** — dense (Pinecone) + sparse (local BM25) candidates are Reciprocal-Rank-Fused across all sub-queries before reranking.
- **Semantic chunking + local reranking** — `SemanticChunker` then FlashRank cross-encoder rerank, all local (no API cost).
- **Retrieval hygiene** — low-signal fragments, near-duplicate boilerplate, and SEC XBRL / EDGAR-index noise are filtered at both ingestion and retrieval, so cited context stays human-readable. The 10-K is ingested from its clean primary document, not the bulky full submission.
- **Specific citations** — sources are labeled with the filing/release date derived from the URL (`FOMC 2026-04-29`, `CPI 2026-01-13`, `NVDA 10-K`), not just a generic badge.
- **LCEL chains** — synthesis and answer generation are built with the LangChain Expression Language pipe (`prompt | llm | parser`).
- **RAGAS evaluation** — a three-config comparison (fixed-size vs semantic vs semantic+rerank) with a Nebius-hosted judge.
- **No OpenAI** — generation and judging run on `meta-llama/Llama-3.3-70B-Instruct` via Nebius. The `openai` package is used only as the OpenAI-compatible transport to the Nebius endpoint.
- **Streamlit dashboard** — side-by-side macro / company / synthesis columns with source badges and a colored verdict.

---

## Architecture

```
                         parse_query
              (resolve ticker · expand query once)
                        /             \
              macro_node           company_node          ← parallel retrieval branches
         (namespace "macro")   (namespace "company-<ticker>")
                        \             /
                         (join)  ▼
                   check_retrieval_quality               ← gate: enough context?
                        /             \
              route == refusal   route == synthesis
                     ▼                     ▼
                refusal_node         synthesis_node       ← verdict: BULLISH / BEARISH / CONFLICT
                     │                     ▼
                    END              generate_answer       ← final cited answer
                                           │
                                          END
```

Each retrieval branch: **expand query → for each sub-query retrieve dense (BGE+Pinecone) and sparse (BM25) → RRF-fuse the pool → FlashRank → top-3**.

---

## Tech stack

| Layer            | Choice |
|------------------|--------|
| Orchestration    | **LangGraph** `StateGraph` with a typed state + conditional routing |
| Embedding        | `BAAI/bge-base-en-v1.5` (768-dim), local — **same model for ingest + retrieval** |
| Chunking (live)  | `SemanticChunker` (percentile @ 65), 8000-char hard cap via `RecursiveCharacterTextSplitter` |
| Chunking (eval)  | Fixed-size `RecursiveCharacterTextSplitter` (512 / 64) with low-signal filtering |
| Vector store     | **Pinecone** serverless, index `macro-micro`, namespaces `macro` + `company-<ticker>` |
| Retrieval (live) | hybrid: dense top_k=10 + **BM25** sparse, RRF-fused → **FlashRank** (`ms-marco-MiniLM-L-12-v2`, local) → top_n = 3 |
| Generation LLM   | `meta-llama/Llama-3.3-70B-Instruct` via **Nebius** (`temperature=0`), wired as LCEL chains |
| Query expansion  | Same Nebius LLM, run once per question in `parse_query` |
| Evaluation       | **RAGAS** (faithfulness, context_precision, context_recall) with a Nebius judge |
| Dashboard        | **Streamlit** |

> **No OpenAI models or keys** are used anywhere. The `openai` package appears only as the OpenAI-compatible transport client pointed at the Nebius `base_url` (for `NebiusEmbeddings` and the RAGAS judge).

### Nebius embedding experiment (optional)
A separate index `macro-micro-nebius` (1536-dim, namespace `nebius-exp`) stores the same corpus embedded with `BAAI/bge-en-icl` via the Nebius API — used **only** for an embedding-comparison experiment, never by the live agent.

---

## Project structure

```
macro-micro-bridge/
├── .env.example            # copy to .env and fill keys
├── requirements.txt
├── main.py                 # CLI entrypoint (--ticker, --refresh, -q)
├── app.py                  # Streamlit dashboard
├── pipeline/
│   ├── config.py           # all constants, ticker whitelist, namespace + env helpers
│   ├── ingest_macro.py     # FOMC PDFs + BLS CPI/NFP HTML
│   ├── ingest_company.py   # SEC 10-K + earnings 8-Ks (EDGAR), auto CIK lookup
│   ├── embed_store.py      # embedder, chunkers, Pinecone, dense+BM25 hybrid, expand_query
│   └── refresh.py          # rebuild semantic + fixed-size + nebius namespaces
├── agents/
│   ├── macro_agent.py      # macro multi-query hybrid retrieval + rerank
│   ├── company_agent.py    # per-ticker company hybrid retrieval + rerank
│   ├── synthesis_agent.py  # LCEL synthesis + answer chains (Nebius LLM)
│   └── graph.py            # LangGraph wiring, parallel branches, refusal routing
├── evaluate/
│   ├── ragas_eval.py       # three-config RAGAS comparison (A/B/C)
│   └── results/            # generated: RESULTS.md, RESULTS.json, per-config CSVs
├── bm25_store/             # generated: local BM25 indexes (git-ignored)
└── README.md
```

---

## Prerequisites

- **Python 3.11**
- A **Pinecone** account (serverless free tier) → `PINECONE_API_KEY`
- A **Nebius Token Factory** account → `NEBIUS_API_KEY`
- ~1.5 GB disk for dependencies (PyTorch, transformers) + ~440 MB for the BGE model (downloaded on first use)

---

## Setup

### 1. Create an isolated virtual environment (recommended)

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure secrets

```powershell
copy .env.example .env      # Windows
# cp .env.example .env       # macOS / Linux
```

Then edit `.env` (see below). `.env` is git-ignored — never commit it.

---

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `PINECONE_API_KEY` | ✅ | Pinecone vector store |
| `NEBIUS_API_KEY`   | ✅ | LLM generation, query expansion, RAGAS judge, Nebius embeddings |
| `LANGCHAIN_TRACING_V2` | optional | Set `true` to send traces to LangSmith |
| `LANGCHAIN_API_KEY`    | optional | LangSmith API key (from smith.langchain.com) |
| `LANGCHAIN_PROJECT`    | optional | LangSmith project name, e.g. `macro-micro-bridge` |

The app **fails loudly** at startup if either required key is missing. LangSmith tracing is entirely optional and off unless you set the three `LANGCHAIN_*` variables.

---

## Usage

> Activate the venv first (`.\venv\Scripts\Activate.ps1`).

### Build the vector store (first run)

Ingests → chunks → embeds → upserts. One `--refresh` populates **both** the live (semantic) and evaluation (fixed-size) namespaces for the chosen ticker, plus the shared macro corpus. Existing vectors are cleared before each upload.

```powershell
python main.py --refresh                 # default ticker (NVDA)
python main.py --refresh --ticker AAPL   # ingest Apple's filings
```

Finer-grained control via the refresh module:

```powershell
python -m pipeline.refresh --macro            # macro corpus only (universal)
python -m pipeline.refresh --company MSFT     # one ticker's semantic namespace
python -m pipeline.refresh --fixed NVDA       # one ticker's fixed-size eval namespace
python -m pipeline.refresh --all              # macro + every supported ticker
python -m pipeline.refresh --nebius           # the 1536-dim experiment index
```

### Ask a question (CLI)

```powershell
python main.py -q "Should I be long NVDA given the current macro environment?"
python main.py -q "Is AAPL a buy right now?" --ticker AAPL
```

### Launch the dashboard

```powershell
streamlit run app.py
```

The sidebar has a **ticker selectbox**, a hand-drawn execution-graph diagram, and live Pinecone vector counts. The main view shows three columns — macro context, company context, and the synthesis verdict — with per-chunk source badges (FOMC / CPI / NFP / 10-K / Earnings).

---

## Dynamic ticker selection

The macro corpus (Fed / CPI / jobs) is **universal** and shared. Only the company corpus is ticker-specific, and each ticker lives in its own Pinecone namespace:

```
macro            ← shared by every ticker
company-nvda     company-aapl     company-msft     company-amd
```

Supported tickers are whitelisted in `config.SUPPORTED_TICKERS = ["NVDA", "AAPL", "MSFT", "AMD"]` and validated everywhere a ticker enters the system. The active ticker is resolved with this precedence:

1. a supported ticker explicitly mentioned in the question (e.g. *"long **AAPL**?"*),
2. the `--ticker` flag / sidebar selection,
3. the default (`config.TICKER` = NVDA).

CIKs for the SEC accession URLs are resolved from a built-in dict for the four supported tickers (fast, no network) and fall back to a live SEC `company_tickers.json` lookup for anything else.

---

## Verdicts & the refusal path

The synthesis node emits exactly one verdict:

| Verdict | Meaning |
|---------|---------|
| `ALIGNED BULLISH` | Macro and company signals both support upside |
| `ALIGNED BEARISH` | Both signals point to downside / caution |
| `CONFLICT DETECTED` | Signals disagree — the node explains *what* clashes |
| `INSUFFICIENT CONTEXT` | Retrieval returned too little context — the agent refuses and suggests `--refresh` |

The `check_retrieval_quality` node runs **after** both retrieval branches and **before** the LLM. If either stream returns fewer than the minimum chunks (`MIN_MACRO_CHUNKS` / `MIN_COMPANY_CHUNKS`), the graph routes to a terminal `refusal_node` and the LLM is never asked to synthesize from missing evidence.

---

## Evaluation (RAGAS, three configs)

```powershell
python -m evaluate.ragas_eval
```

Compares three retrieval configurations over **13 hand-written reference questions** (spanning factual, ambiguous, cross-document, and unanswerable types) and writes a comparison with A→B and B→C deltas, a computed winner, and a data-driven analysis:

| Config | Chunking | Rerank |
|--------|----------|--------|
| **A** | Fixed-size (512/64) | No |
| **B** | Semantic | No |
| **C** | Semantic | FlashRank |

- **Metrics:** `faithfulness`, `context_precision`, `context_recall` (`answer_relevancy` is intentionally excluded — it can break on non-OpenAI providers).
- **Judge:** `meta-llama/Llama-3.3-70B-Instruct` via Nebius (`ragas.llms.llm_factory`).
- **Judge embeddings:** `BAAI/bge-en-icl` via Nebius.
- **Outputs:** results are persisted to `evaluate/results/RESULTS.md`, `RESULTS.json`, and per-config CSVs (not just printed).
- **Scope:** the eval deliberately isolates chunking/rerank — it does **not** use the live agent's query-expansion or BM25-hybrid retrieval, so each delta is attributable to the variable under test.

Run `python main.py --refresh` (or `python -m pipeline.refresh --fixed NVDA`) first so the fixed-size namespaces exist.

---

## Troubleshooting

- **SEC EDGAR 403s:** `pipeline/ingest_company.py` declares a requester email for SEC fair-access (`research@example.com` by default). Set it to a real contact email if you get blocked.
- **First retrieval is slow:** the local BGE embedder (~440 MB) and FlashRank model download on first use, then stay cached.
- **Windows console `UnicodeEncodeError`:** some CLI prints use `→`/`—`. If a strict cp1252 console errors, set `PYTHONUTF8=1`.
- **Hugging Face symlink warning on Windows:** harmless; enable Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence it.
- **Pinecone index already exists:** `ensure_index` tolerates concurrent creation by the two parallel branches, so this is handled automatically.
- **RAGAS / langchain compatibility:** `requirements.txt` pins `ragas>=0.4,<0.5` (the API this harness targets). RAGAS 0.4.x eagerly imports a `langchain_community.chat_models.vertexai` path that langchain 1.x dropped, so `evaluate/ragas_eval.py` registers a small no-op shim (`_ensure_ragas_importable`) — Vertex AI is never used (judging runs on Nebius). This keeps the modern langchain 1.x stack intact; no downgrade needed.
- **Never commit `.env`** — it is git-ignored along with `venv/`, `sec-edgar-filings/`, and `bm25_store/`.

---

## License

No license file is included yet. Add a `LICENSE` (e.g. MIT) before publishing if you intend others to reuse the code.
