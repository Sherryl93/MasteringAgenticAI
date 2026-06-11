"""Synthesis agent.

Reconciles the two independent research streams (macro + company) into a
single grounded view, emitting exactly one of three verdicts:

    VERDICT: ALIGNED BULLISH
    VERDICT: ALIGNED BEARISH
    VERDICT: CONFLICT DETECTED

Generation uses Llama-3.3-70B-Instruct via Nebius (OpenAI-compatible API).
No OpenAI models or keys are involved.
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pipeline import config

# Module-level LLM singleton.
_llm: ChatOpenAI | None = None

# Canonical verdict strings — the only three allowed outcomes.
VERDICT_BULLISH = "VERDICT: ALIGNED BULLISH"
VERDICT_BEARISH = "VERDICT: ALIGNED BEARISH"
VERDICT_CONFLICT = "VERDICT: CONFLICT DETECTED"
VERDICT_UNKNOWN = "VERDICT: UNDETERMINED"

# Synthesis system prompt — sell-side briefing note, precision + specific citations.
SYNTHESIS_SYSTEM_PROMPT = """You are a sell-side equity analyst writing a pre-trade briefing note. You have two evidence streams. Write with precision and cite specific numbers, dates, and document sources.

MACRO CONTEXT (FOMC minutes / CPI / NFP):
{macro_context}

COMPANY CONTEXT (10-K / earnings transcripts):
{company_context}

Rules:
- Never write "based on the provided context" or "here-are-the-summaries" — start directly with findings
- Always cite the specific document type and date, e.g. "FOMC Jan 2026 minutes note..." or "NVDA Q1 FY2027 earnings show..."
- Quote specific figures where available: rates, revenue, guidance numbers, percentages
- Keep each stream summary to 2-3 sentences maximum
- Output exactly one of these verdict lines, on its own line, with no extra text around it:
  VERDICT: ALIGNED BULLISH
  VERDICT: ALIGNED BEARISH
  VERDICT: CONFLICT DETECTED
- If CONFLICT DETECTED: name the specific tension, e.g. "Fed minutes signal rate caution while NVDA guidance projects 15% sequential revenue growth"
- Never hallucinate numbers not present in context
- If context lacks specific figures say "figures not retrieved" not a made-up number"""

# Final-answer prompt — portfolio-manager briefing, two paragraphs, no filler.
ANSWER_SYSTEM_PROMPT = """You are a portfolio manager's assistant. Given the synthesis below, write a direct 2-paragraph answer to the original question.

Synthesis:
{synthesis}

Original question: {question}

Rules:
- First paragraph: state the verdict and the key tension or alignment in one sentence, then explain with evidence
- Second paragraph: name the single biggest risk to the bull or bear case
- Use plain language — no jargon, no filler phrases
- Maximum 120 words total
- Never start with "In response to your question" or "Based on the analysis"
- Start with the verdict signal directly, e.g. "The evidence points to..." or "Macro and company signals conflict here..." """


def get_llm() -> ChatOpenAI:
    """Return the shared Nebius-hosted Llama-3.3-70B chat model."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=config.NEBIUS_BASE_URL,
            api_key=config.nebius_api_key(),
            model=config.LLM_MODEL,
            temperature=0,
        )
    return _llm


def _join(chunks: list[str]) -> str:
    """Format a context list into a numbered block for the prompt."""
    if not chunks:
        return "(no context retrieved)"
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))


def extract_verdict(text: str) -> str:
    """Detect which canonical verdict the synthesis text contains."""
    upper = text.upper()
    if VERDICT_CONFLICT in upper:
        return VERDICT_CONFLICT
    if VERDICT_BULLISH in upper:
        return VERDICT_BULLISH
    if VERDICT_BEARISH in upper:
        return VERDICT_BEARISH
    return VERDICT_UNKNOWN


def run_synthesis(question: str, macro_context: list[str], company_context: list[str]) -> dict:
    """Produce the reconciled synthesis text and its detected verdict."""
    synthesis_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SYNTHESIS_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        | get_llm()
        | StrOutputParser()
    )
    raw = synthesis_chain.invoke(
        {
            "question": question,
            "macro_context": _join(macro_context),
            "company_context": _join(company_context),
        }
    )
    verdict = extract_verdict(raw)
    # Drop the literal "VERDICT: ..." line from the stored text: the UI renders
    # the verdict as a badge from `verdict`, so leaving it in the prose would
    # show it twice.
    synthesis = "\n".join(
        ln for ln in raw.splitlines() if not ln.strip().upper().startswith("VERDICT:")
    ).strip()
    synthesis = re.sub(r"\n{3,}", "\n\n", synthesis)
    print(f"[synthesis_agent] verdict detected: {verdict}")
    return {"synthesis": synthesis, "verdict": verdict}


def generate_answer(question: str, synthesis: str) -> str:
    """Produce the final cited investor answer from the synthesis output."""
    answer_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                ("human", "Write the briefing answer now."),
            ]
        )
        | get_llm()
        | StrOutputParser()
    )
    return answer_chain.invoke({"question": question, "synthesis": synthesis})


if __name__ == "__main__":
    demo = run_synthesis(
        "Should I be long NVDA given the current macro environment?",
        ["FOMC minutes: the Committee signaled a measured easing path."],
        ["NVDA 10-K: data center revenue grew strongly year over year."],
    )
    print(demo["synthesis"])
    print("\n--- FINAL ANSWER ---\n")
    print(generate_answer("Should I be long NVDA?", demo["synthesis"]))
