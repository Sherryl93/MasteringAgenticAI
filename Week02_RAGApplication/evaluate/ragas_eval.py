"""RAGAS evaluation — no OpenAI models anywhere.

Judge LLM         : meta-llama/Llama-3.3-70B-Instruct via Nebius (llm_factory)
Judge embeddings  : BAAI/bge-en-icl via Nebius (OpenAIEmbeddings -> Nebius URL)
Metrics           : faithfulness, context_precision, context_recall
                    (answer_relevancy is intentionally excluded — it can break
                     on non-OpenAI providers)

Three pipeline configurations are compared over the same question set:
    Config A : fixed-size chunking, NO rerank      (top-3 by similarity)
    Config B : semantic chunking,   NO rerank      (top-3 by similarity)
    Config C : semantic chunking,   WITH FlashRank (top-3 after rerank)

Scope note (eval/live isolation): these configs deliberately isolate the
*chunking* and *reranking* variables. They retrieve from a single question and
do NOT apply the query-expansion or BM25-hybrid retrieval that the live agent
uses, so each metric delta is attributable to chunking/rerank alone rather than
to confounding retrieval tricks. See build_sample() for the rationale.

Results are persisted to RESULTS.md, RESULTS.json, and per-config CSVs.

Run:
    python -m evaluate.ragas_eval
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from agents import synthesis_agent
from pipeline import config, embed_store

# NOTE: ragas / langchain_openai / openai are imported lazily inside the judge
# and evaluation functions. ragas pulls a large dependency tree and some ragas
# releases eagerly import paths that newer langchain_community has dropped, so
# importing this module must not require ragas — only RUNNING the eval does.


def _ensure_ragas_importable() -> None:
    """Compatibility shim: ragas 0.4.x eagerly imports
    ``langchain_community.chat_models.vertexai`` (ChatVertexAI), which the
    modern langchain 1.x ``langchain_community`` no longer ships. We never use
    Vertex AI (judging runs on Nebius), so register a harmless stub module so
    ``import ragas`` succeeds without downgrading the whole langchain stack.
    """
    import importlib
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(name)
        return  # real module present — nothing to do
    except Exception:  # noqa: BLE001
        pass

    stub = types.ModuleType(name)

    class ChatVertexAI:  # placeholder; never instantiated in this pipeline
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ChatVertexAI is not available; this pipeline judges via Nebius."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[name] = stub


# Register the shim at import time, before any (lazy) ragas import fires.
_ensure_ragas_importable()

# ──────────────────────────────────────────────────────────────
# Evaluation set — 13 questions across question types
#   factual        : single-document lookups (the easy baseline)
#   ambiguous      : underspecified, retriever must disambiguate
#   cross_document : require BOTH macro and company evidence together
#   unanswerable   : outside the corpus — should retrieve little / refuse
# ──────────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    # --- factual ---
    {
        "category": "factual",
        "question": "What does NVDA's most recent 10-K say about the company's "
        "primary revenue growth drivers?",
        "reference": "NVIDIA's 10-K attributes its revenue growth primarily to its "
        "Data Center segment, driven by demand for GPUs and accelerated computing "
        "platforms used in AI and large language model training and inference.",
    },
    {
        "category": "factual",
        "question": "How did NVDA's data center revenue trend in the latest "
        "earnings release?",
        "reference": "NVIDIA's latest earnings release reported continued strong "
        "year-over-year growth in Data Center revenue, reaching a new record on "
        "sustained AI infrastructure demand.",
    },
    {
        "category": "factual",
        "question": "What is the Federal Reserve's current stance on interest "
        "rates according to the latest FOMC minutes?",
        "reference": "The most recent FOMC minutes indicate the Committee is on a "
        "measured, data-dependent easing path, weighing progress on inflation "
        "against labor-market conditions before adjusting the policy rate.",
    },
    {
        "category": "factual",
        "question": "What do the latest CPI releases show about the direction "
        "of inflation?",
        "reference": "The latest CPI releases show inflation continuing to moderate "
        "toward the Fed's target, with core CPI easing on a year-over-year basis "
        "though still above 2 percent.",
    },
    {
        "category": "factual",
        "question": "What do recent nonfarm payroll (NFP) reports indicate about "
        "the strength of the labor market?",
        "reference": "Recent NFP / Employment Situation reports indicate a cooling "
        "but still-resilient labor market, with slower monthly job gains and an "
        "unemployment rate that remains historically low.",
    },
    {
        "category": "factual",
        "question": "What risk factors does NVDA's 10-K highlight for its business?",
        "reference": "NVIDIA's 10-K highlights risks including dependence on demand "
        "for its products, intense competition, supply-chain and manufacturing "
        "concentration, export controls and geopolitical tensions, and rapid "
        "technological change.",
    },
    # --- ambiguous (underspecified; retriever must disambiguate) ---
    {
        "category": "ambiguous",
        "question": "How are things looking for NVDA right now?",
        "reference": "NVIDIA's recent filings and earnings show strong Data Center "
        "and AI-driven revenue growth, while the macro backdrop (Fed easing, "
        "moderating inflation, a cooling labor market) is broadly supportive, "
        "subject to the risk factors noted in its 10-K.",
    },
    {
        "category": "ambiguous",
        "question": "Is now a good time given the economy?",
        "reference": "The macro context shows the Fed on a measured easing path with "
        "inflation moderating and the labor market cooling but resilient; any "
        "company-specific judgment must rest on NVIDIA's filings and earnings rather "
        "than on the macro backdrop alone.",
    },
    # --- cross_document (need macro AND company evidence) ---
    {
        "category": "cross_document",
        "question": "Do the Fed's rate path and NVDA's earnings trajectory point in "
        "the same direction for the stock?",
        "reference": "Macro sources show the Fed easing as inflation moderates and "
        "the labor market cools, a generally supportive backdrop, while NVIDIA's "
        "earnings show record Data Center / AI-driven growth; together these signals "
        "are broadly aligned, though NVIDIA's 10-K risk factors temper the view.",
    },
    {
        "category": "cross_document",
        "question": "How might the current inflation and labor data affect demand for "
        "NVDA's data center products?",
        "reference": "CPI shows inflation moderating and NFP shows a cooling but "
        "resilient labor market, an easing macro backdrop; NVIDIA's filings tie Data "
        "Center demand to AI infrastructure investment, so the supportive macro "
        "conditions are broadly consistent with continued demand, per the context.",
    },
    {
        "category": "cross_document",
        "question": "What is the combined macro and company case for being long NVDA?",
        "reference": "The combined case rests on a supportive macro backdrop (Fed "
        "easing, moderating CPI, resilient employment) and NVIDIA's record Data "
        "Center / AI-driven revenue growth, balanced against the competition, "
        "supply-chain, and export-control risks flagged in its 10-K.",
    },
    # --- unanswerable (outside corpus; should retrieve little / refuse) ---
    {
        "category": "unanswerable",
        "question": "What dividend per share will NVDA pay in 2030?",
        "reference": "The provided context does not contain information about NVIDIA's "
        "future 2030 dividend per share, so this cannot be answered from the corpus.",
    },
    {
        "category": "unanswerable",
        "question": "What did the European Central Bank decide at its last meeting?",
        "reference": "The corpus covers U.S. Federal Reserve minutes and U.S. BLS "
        "data, not European Central Bank decisions, so this cannot be answered from "
        "the provided context.",
    },
]

# For a fast, low-cost run on a rate-limited / free tier, evaluate a
# representative 6-question subset spanning every category (factual company +
# macro, ambiguous, cross-document, unanswerable). To run the full set instead,
# set:  EVAL_QUESTIONS = _ALL_EVAL_QUESTIONS
_ALL_EVAL_QUESTIONS = EVAL_QUESTIONS
EVAL_QUESTIONS = [_ALL_EVAL_QUESTIONS[i] for i in (0, 1, 2, 6, 8, 11)]


# ──────────────────────────────────────────────────────────────
# Nebius-backed judges
# ──────────────────────────────────────────────────────────────
def get_judge_llm():
    """RAGAS judge LLM: Llama-3.3-70B via Nebius using llm_factory."""
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        base_url=config.NEBIUS_BASE_URL,
        api_key=os.environ["NEBIUS_API_KEY"],
    )
    return llm_factory(config.LLM_MODEL, provider="openai", client=client)


def get_judge_embeddings():
    """RAGAS judge embeddings: BAAI/bge-en-icl via Nebius."""
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = OpenAIEmbeddings(
        base_url=config.NEBIUS_BASE_URL,
        api_key=os.environ["NEBIUS_API_KEY"],
        model=config.NEBIUS_EMBED_MODEL,
        check_embedding_ctx_length=False,  # Nebius is not OpenAI's tokenizer
    )
    return LangchainEmbeddingsWrapper(emb)


# ──────────────────────────────────────────────────────────────
# The three evaluation configurations
# ──────────────────────────────────────────────────────────────
CONFIG_A = {
    "name": "Config A — fixed-size, no rerank",
    "macro_ns": config.NAMESPACE_MACRO_FIXED,
    "company_ns": config.company_fixed_namespace(config.TICKER),
    "rerank": False,
    "retriever": "fixed",
}
CONFIG_B = {
    "name": "Config B — semantic, no rerank",
    "macro_ns": config.NAMESPACE_MACRO,
    "company_ns": config.company_namespace(config.TICKER),
    "rerank": False,
    "retriever": "semantic",
}
CONFIG_C = {
    "name": "Config C — semantic + FlashRank rerank",
    "macro_ns": config.NAMESPACE_MACRO,
    "company_ns": config.company_namespace(config.TICKER),
    "rerank": True,
    "retriever": "semantic",
}
CONFIGS = [CONFIG_A, CONFIG_B, CONFIG_C]


# ──────────────────────────────────────────────────────────────
# Retrieval dispatch per config
# ──────────────────────────────────────────────────────────────
def _retrieve(query: str, namespace: str, cfg: dict) -> list[dict]:
    """Retrieve for one stream according to the config's strategy.

    fixed     -> retrieve_fixed (plain top-k similarity, no rerank)
    semantic  -> retrieve_and_rerank if cfg['rerank'] else plain top-k similarity
    """
    if cfg["retriever"] == "fixed":
        return embed_store.retrieve_fixed(query, namespace, k=config.TOP_K_RERANK)
    # semantic
    if cfg["rerank"]:
        index = embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)
        return embed_store.retrieve_and_rerank(index, namespace, query)
    return embed_store.retrieve_fixed(query, namespace, k=config.TOP_K_RERANK)


def build_sample(item: dict, cfg: dict) -> dict:
    """Retrieve macro + company context for a config, generate an answer.

    EVAL/LIVE ISOLATION (intentional): the live agent expands each question into
    several sub-queries and retrieves hybrid (dense + BM25). This harness does
    NEITHER — it retrieves from the single, original question with dense-only
    similarity (plus FlashRank for Config C). That keeps each A/B/C metric delta
    attributable to the chunking/rerank change under test, not to query expansion
    or sparse fusion. The live retrieval quality is exercised by the app/CLI, not
    measured here.
    """
    question = item["question"]
    macro = _retrieve(question, cfg["macro_ns"], cfg)
    company = _retrieve(question, cfg["company_ns"], cfg)

    macro_ctx = [c["text"] for c in macro]
    company_ctx = [c["text"] for c in company]
    contexts = macro_ctx + company_ctx

    synth = synthesis_agent.run_synthesis(question, macro_ctx, company_ctx)
    answer = synthesis_agent.generate_answer(question, synth["synthesis"])

    return {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": item["reference"],
    }


# ──────────────────────────────────────────────────────────────
# Evaluation driver
# ──────────────────────────────────────────────────────────────
def evaluate_config(cfg: dict, judge_llm, judge_emb):
    """Build the dataset for one config dict and run RAGAS over it."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    print(f"\n=== Building samples for {cfg['name']} ===")
    rows = [build_sample(item, cfg) for item in EVAL_QUESTIONS]
    dataset = EvaluationDataset.from_list(rows)

    print(f"=== Running RAGAS for {cfg['name']} ===")
    # Low concurrency keeps us under the Nebius free-tier rate limit so judge
    # calls succeed on the first try instead of entering long retry/backoff
    # (which otherwise stalls the progress bar for tens of minutes per item).
    run_config = RunConfig(max_workers=2, max_retries=3, max_wait=20, timeout=90)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
    )
    return result


def _mean_scores(result) -> dict:
    """Average each metric column across the eval questions."""
    df = result.to_pandas()
    means = {}
    for metric in ("faithfulness", "context_precision", "context_recall"):
        if metric in df.columns:
            means[metric] = float(df[metric].mean())
    return means


# Pretty metric labels + the metric keys we report on.
_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
_METRIC_KEYS = list(_METRIC_LABELS)

# Where evaluation artifacts are written.
RESULTS_DIR = Path("evaluate/results")


def _config_slug(cfg: dict) -> str:
    """Filename-safe key from a config name, e.g. 'config_a'."""
    head = cfg["name"].split("—")[0].strip().lower()
    return head.replace(" ", "_") or "config"


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # excludes NaN


def _analysis(scored: list[tuple[dict, dict]]) -> list[str]:
    """Generate analysis lines computed from the ACTUAL numbers (no prose)."""
    lines: list[str] = []

    # Best config per metric, from the data.
    for key in _METRIC_KEYS:
        ranked = [(cfg["name"], s.get(key)) for cfg, s in scored if _is_num(s.get(key))]
        if ranked:
            name, val = max(ranked, key=lambda t: t[1])
            lines.append(f"- Best {_METRIC_LABELS[key]}: {name} ({val:.3f}).")

    # Consecutive deltas (A→B is the chunking change, B→C is the rerank change).
    effect = {1: "chunking (A->B)", 2: "reranking (B->C)"}
    for i in range(1, len(scored)):
        prev_s = scored[i - 1][1]
        cur_s = scored[i][1]
        label = effect.get(i, f"step {i} ({scored[i-1][0]['name']}→{scored[i][0]['name']})")
        for key in _METRIC_KEYS:
            a, b = prev_s.get(key), cur_s.get(key)
            if not (_is_num(a) and _is_num(b)):
                continue
            d = b - a
            direction = "improved" if d > 0 else "reduced" if d < 0 else "left unchanged"
            pretty = label[:1].upper() + label[1:]  # keep A->B casing intact
            lines.append(
                f"- {pretty} {direction} {_METRIC_LABELS[key]} by {d:+.3f}."
            )
    return lines


def build_report(scored: list[tuple[dict, dict]], n_questions: int) -> tuple[str, dict]:
    """Build the markdown report + a JSON-serializable data dict from the scores."""
    names = [cfg["name"] for cfg, _ in scored]

    # --- markdown table ---
    head = "| Metric | " + " | ".join(names) + " | " + " | ".join(
        f"{names[i].split('—')[0].strip()}->{names[i+1].split('—')[0].strip()}"
        for i in range(len(names) - 1)
    ) + " |"
    sep = "|" + "---|" * (1 + len(names) + max(0, len(names) - 1))
    rows = [head, sep]
    for key in _METRIC_KEYS:
        vals = [s.get(key, float("nan")) for _, s in scored]
        cells = [f"{v:.3f}" if _is_num(v) else "n/a" for v in vals]
        deltas = [
            f"{(vals[i+1]-vals[i]):+.3f}" if _is_num(vals[i]) and _is_num(vals[i+1]) else "n/a"
            for i in range(len(vals) - 1)
        ]
        rows.append(f"| {_METRIC_LABELS[key]} | " + " | ".join(cells + deltas) + " |")
    table = "\n".join(rows)

    # --- winner from summed means ---
    winner_cfg, _ = max(
        scored, key=lambda p: sum(v for v in p[1].values() if _is_num(v))
    )

    analysis = _analysis(scored)
    md = (
        f"# RAGAS Evaluation Results\n\n"
        f"_Generated {datetime.now().isoformat(timespec='seconds')} - "
        f"{n_questions} questions - judge: {config.LLM_MODEL} via Nebius_\n\n"
        f"## Score comparison\n\n{table}\n\n"
        f"**Winner (highest summed mean): {winner_cfg['name']}**\n\n"
        f"## Analysis (computed from the numbers above)\n\n"
        + "\n".join(analysis)
        + "\n"
    )

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_questions": n_questions,
        "judge_model": config.LLM_MODEL,
        "configs": [
            {"name": cfg["name"], "scores": {k: s.get(k) for k in _METRIC_KEYS}}
            for cfg, s in scored
        ],
        "winner": winner_cfg["name"],
        "analysis": analysis,
    }
    return md, data


def persist_results(
    scored: list[tuple[dict, dict]],
    results: list,
    n_questions: int,
) -> None:
    """Write per-config CSVs, RESULTS.md, and RESULTS.json; echo to stdout."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-config per-question CSVs.
    for (cfg, _), result in zip(scored, results):
        csv_path = RESULTS_DIR / f"ragas_{_config_slug(cfg)}.csv"
        result.to_pandas().to_csv(csv_path, index=False)
        print(f"[eval] wrote {csv_path}")

    md, data = build_report(scored, n_questions)

    md_path = RESULTS_DIR / "RESULTS.md"
    json_path = RESULTS_DIR / "RESULTS.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[eval] wrote {md_path}")
    print(f"[eval] wrote {json_path}")

    print("\n" + md)


def main() -> None:
    config.validate_env()
    judge_llm = get_judge_llm()
    judge_emb = get_judge_embeddings()

    results = [evaluate_config(cfg, judge_llm, judge_emb) for cfg in CONFIGS]
    scored = [(cfg, _mean_scores(r)) for cfg, r in zip(CONFIGS, results)]
    persist_results(scored, results, len(EVAL_QUESTIONS))


if __name__ == "__main__":
    main()
