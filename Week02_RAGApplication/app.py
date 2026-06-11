"""Streamlit dashboard for the Macro-Micro Bridge Agent.

Financial-terminal styling. All pipeline logic (cached resources, graph
invocation, state reads, sidebar status, error/refusal handling) is unchanged
from the original — only visual presentation and copy differ.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components

from pipeline import config

# ──────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Macro-Micro Bridge", layout="wide")

# ── Design system ────────────────────────────────────────────
BG = "#0f1117"        # near-black background
SURFACE = "#1a1d27"   # card surface
BORDER = "#2d3148"    # subtle separator
TEXT = "#e8eaf0"      # primary text
MUTED = "#8b90a8"     # secondary text
TEAL = "#2dd4bf"      # macro stream
CORAL = "#fb7185"     # company stream
PURPLE = "#a78bfa"    # synthesis
GREEN = "#4ade80"     # bullish
RED = "#f87171"       # bearish
ORANGE = "#fb923c"    # conflict
GRAY = "#6b7280"      # undetermined / insufficient
MONO = "'JetBrains Mono','Fira Code',monospace"
SANS = "'Inter',system-ui,sans-serif"

# Verdict -> (label, accent color). Text uses the accent; bg a translucent tint.
VERDICT_STYLE = {
    "VERDICT: ALIGNED BULLISH": ("ALIGNED BULLISH", GREEN),
    "VERDICT: ALIGNED BEARISH": ("ALIGNED BEARISH", RED),
    "VERDICT: CONFLICT DETECTED": ("CONFLICT DETECTED", ORANGE),
    "VERDICT: UNDETERMINED": ("UNDETERMINED", GRAY),
    config.VERDICT_INSUFFICIENT: ("INSUFFICIENT", GRAY),
}

# ── Global CSS (literal hex — no f-string so CSS braces stay intact) ──
st.markdown(
    """
    <style>
      .stApp { background-color:#0f1117; color:#e8eaf0;
               font-family:'Inter',system-ui,sans-serif; }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { padding-top:2.2rem; }
      [data-testid="stSidebar"] { background-color:#141721;
               border-right:1px solid #2d3148; }
      [data-testid="stSidebar"] * { color:#e8eaf0; }
      .stTextInput input, .stTextArea textarea {
               background-color:#1a1d27 !important; color:#e8eaf0 !important;
               border:1px solid #2d3148 !important; border-radius:6px; }
      .stTextInput input::placeholder { color:#8b90a8; }
      div[data-baseweb="input"] { background-color:#1a1d27 !important; }
      div[data-baseweb="select"] > div { background-color:#1a1d27 !important;
               border-color:#2d3148 !important; color:#e8eaf0 !important; }
      .stButton>button { background-color:#a78bfa; color:#0f1117; font-weight:700;
               border:none; border-radius:6px; }
      .stButton>button:hover { background-color:#8b6df0; color:#0f1117; }
      [data-testid="stSpinner"] * { color:#a78bfa; }
      hr { border-color:#2d3148; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Cached resources — initialize once, not on every query
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedder():
    from pipeline import embed_store

    return embed_store.get_embedder()


@st.cache_resource(show_spinner=False)
def get_index():
    from pipeline import embed_store

    return embed_store.ensure_index(config.PINECONE_INDEX, config.EMBED_DIM)


@st.cache_resource(show_spinner=False)
def get_app():
    from agents import graph

    return graph.build_graph()


# ──────────────────────────────────────────────────────────────
# Rendering helpers (presentation only)
# ──────────────────────────────────────────────────────────────
def mono_label(text: str, color: str = MUTED, margin: str = "14px 0 6px") -> None:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:10px;text-transform:uppercase;"
        f"letter-spacing:0.12em;color:{color};margin:{margin};'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def stream_header(label: str, accent: str) -> None:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.12em;color:{accent};font-weight:700;margin-bottom:8px;'>"
        f"{html.escape(label)}</div>",
        unsafe_allow_html=True,
    )


def mono_caption(text: str) -> None:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;color:{MUTED};"
        f"margin-bottom:10px;'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def source_pill(text: str, accent: str) -> str:
    return (
        f"<span style='font-family:{MONO};font-size:10px;background:{accent}22;"
        f"color:{accent};padding:2px 7px;border-radius:4px;letter-spacing:0.03em;'>"
        f"{html.escape(text)}</span>"
    )


def chunk_card(badge: str, text: str, accent: str) -> None:
    snippet = html.escape(text.strip()[:400])
    st.markdown(
        f"<div style='border-left:2px solid {accent};background:{SURFACE};"
        f"padding:9px 11px;margin-bottom:8px;border-radius:0 5px 5px 0;'>"
        f"{source_pill(badge, accent)}"
        f"<div style='font-size:13px;color:{TEXT};margin-top:7px;line-height:1.55;'>"
        f"{snippet}</div></div>",
        unsafe_allow_html=True,
    )


def empty_state(msg: str) -> None:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:12px;color:{MUTED};"
        f"padding:10px 0;'>{html.escape(msg)}</div>",
        unsafe_allow_html=True,
    )


def render_verdict(verdict: str) -> None:
    label, color = VERDICT_STYLE.get(verdict, ("UNDETERMINED", GRAY))
    st.markdown(
        f"<div style='font-family:{MONO};font-size:15px;font-weight:800;"
        f"text-align:center;letter-spacing:0.05em;color:{color};"
        f"background:linear-gradient(180deg,{color}26,{color}12);"
        f"border:1px solid {color}66;border-radius:8px;padding:10px 16px;"
        f"width:100%;box-sizing:border-box;box-shadow:0 2px 10px {color}22;"
        f"margin-bottom:12px;'>{label}</div>",
        unsafe_allow_html=True,
    )


def body_text(text: str, size: int = 13) -> None:
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div style='font-size:{size}px;color:{TEXT};line-height:1.6;'>{safe}</div>",
        unsafe_allow_html=True,
    )


def refusal_card(ticker: str) -> None:
    st.markdown(
        f"<div style='background:{ORANGE}1a;border:1px solid {ORANGE}66;"
        f"border-radius:6px;padding:12px;color:{TEXT};font-size:13px;line-height:1.6;'>"
        f"Retrieval returned insufficient context.<br>"
        f"<span style='font-family:{MONO};font-size:12px;color:{ORANGE};'>"
        f"Run: python main.py --refresh --ticker {html.escape(ticker)}</span></div>",
        unsafe_allow_html=True,
    )


def status_row(dot: str, label: str, value: str) -> None:
    st.markdown(
        f"<div style='font-family:{MONO};font-size:11px;margin:5px 0;'>"
        f"<span style='color:{dot};'>●</span> "
        f"<span style='color:{MUTED};display:inline-block;width:84px;"
        f"white-space:nowrap;vertical-align:top;'>{html.escape(label)}</span> "
        f"<span style='color:{TEXT};'>{html.escape(str(value))}</span></div>",
        unsafe_allow_html=True,
    )


def render_flow_diagram() -> None:
    """Hand-drawn SVG of the LangGraph pipeline (replaces the exported PNG)."""
    mono = "ui-monospace,'JetBrains Mono','Fira Code',Menlo,monospace"

    # (label, cx, cy, w, h, accent, is_terminal)
    nodes = [
        ("__start__", 160, 52, 92, 30, PURPLE, True),
        ("parse_query", 160, 104, 132, 34, "#7c8497", False),
        ("company_node", 84, 162, 126, 34, CORAL, False),
        ("macro_node", 236, 162, 126, 34, TEAL, False),
        ("check_retrieval_quality", 160, 224, 206, 34, PURPLE, False),
        ("synthesis_node", 84, 292, 126, 34, GREEN, False),
        ("refusal_node", 236, 292, 118, 34, ORANGE, False),
        ("generate_answer", 84, 360, 136, 34, PURPLE, False),
        ("__end__", 160, 440, 92, 30, PURPLE, True),
    ]
    # (path_d, is_dashed_conditional)
    edges = [
        ("M160,67 L160,87", False),
        ("M160,121 C160,134 84,132 84,145", False),
        ("M160,121 C160,134 236,132 236,145", False),
        ("M84,179 C84,194 160,192 160,207", False),
        ("M236,179 C236,194 160,192 160,207", False),
        ("M160,241 C160,259 84,258 84,275", True),
        ("M160,241 C160,259 236,258 236,275", True),
        ("M84,309 L84,343", False),
        ("M84,377 C84,406 160,404 160,425", False),
        ("M236,309 C236,402 160,404 160,425", False),
    ]

    edge_svg = ""
    for d, dashed in edges:
        stroke = "#3a4060" if dashed else "#4a5276"
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        marker = "url(#arrowd)" if dashed else "url(#arrow)"
        edge_svg += (
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.6"'
            f'{dash} marker-end="{marker}"/>'
        )

    node_svg = ""
    for label, cx, cy, w, h, accent, terminal in nodes:
        x, y = cx - w / 2, cy - h / 2
        rx = h / 2 if terminal else 9
        fs = 9 if len(label) > 15 else 10
        if terminal:
            node_svg += (
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                f'fill="url(#term)" stroke="#c3b5ff" stroke-width="1.1" '
                f'filter="url(#soft)"/>'
                f'<text x="{cx}" y="{cy + 3.5}" text-anchor="middle" '
                f'font-family="{mono}" font-size="{fs}" font-weight="700" '
                f'fill="#0f1117">{label}</text>'
            )
        else:
            node_svg += (
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                f'fill="url(#node)" stroke="{accent}" stroke-width="1.2" '
                f'filter="url(#soft)"/>'
                f'<rect x="{x + 5}" y="{y + 8}" width="3.2" height="{h - 16}" '
                f'rx="1.6" fill="{accent}"/>'
                f'<text x="{cx + 3}" y="{cy + 3.5}" text-anchor="middle" '
                f'font-family="{mono}" font-size="{fs}" font-weight="600" '
                f'fill="{TEXT}">{label}</text>'
            )

    branch_labels = (
        f'<text x="115" y="262" text-anchor="middle" font-family="{mono}" '
        f'font-size="8" fill="{GREEN}">yes</text>'
        f'<text x="207" y="262" text-anchor="middle" font-family="{mono}" '
        f'font-size="8" fill="{ORANGE}">no</text>'
    )

    svg = (
        '<svg viewBox="0 0 320 470" xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        '<linearGradient id="node" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#262a3b"/>'
        '<stop offset="1" stop-color="#171a24"/></linearGradient>'
        f'<linearGradient id="term" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{TEAL}"/>'
        f'<stop offset="0.55" stop-color="{PURPLE}"/>'
        f'<stop offset="1" stop-color="{CORAL}"/></linearGradient>'
        '<filter id="soft" x="-30%" y="-30%" width="160%" height="160%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="2.4" flood-color="#000000" '
        'flood-opacity="0.5"/></filter>'
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="6.5" refY="3" '
        'orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#6b7290"/></marker>'
        '<marker id="arrowd" markerWidth="9" markerHeight="9" refX="6.5" refY="3" '
        'orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#565d7a"/></marker>'
        '</defs>'
        f'{edge_svg}{branch_labels}{node_svg}'
        '</svg>'
    )

    doc = (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        'html,body{margin:0;padding:0;background:transparent;}'
        '.wrap{max-width:330px;margin:0 auto;background:#141721;'
        'border:1px solid #2d3148;border-radius:12px;'
        'padding:12px 8px 10px;box-shadow:0 6px 18px rgba(0,0,0,0.35);}'
        '.ttl{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:0.2em;'
        'text-transform:uppercase;color:#8b90a8;text-align:center;margin:0 0 6px;}'
        'svg{display:block;width:100%;height:auto;}'
        '</style></head><body><div class="wrap">'
        f'<div class="ttl">Execution Graph</div>{svg}</div></body></html>'
    )
    # Width is capped at ~330px, so the SVG (viewBox 320x470) renders at a
    # predictable height; size the iframe to fit fully and never clip.
    components.html(doc, height=520)


# ──────────────────────────────────────────────────────────────
# Guard: required keys must be present
# ──────────────────────────────────────────────────────────────
try:
    config.validate_env()
except EnvironmentError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# Sidebar — pipeline status + graph image
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    # Branded header — gradient logo chip + wordmark (replaces plain "PIPELINE").
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:0 0 16px;'>"
        f"<div style='width:34px;height:34px;border-radius:9px;flex:0 0 auto;"
        f"background:linear-gradient(135deg,{TEAL},{PURPLE} 55%,{CORAL});"
        f"display:flex;align-items:center;justify-content:center;"
        f"box-shadow:0 4px 12px rgba(124,92,255,0.35);font-family:{SANS};"
        f"font-weight:800;font-size:15px;color:#0f1117;letter-spacing:-0.04em;'>"
        f"M&times;</div>"
        f"<div style='line-height:1.12;'>"
        f"<div style='font-family:{SANS};font-weight:800;font-size:15px;"
        f"background:linear-gradient(90deg,{TEAL},{PURPLE} 60%,{CORAL});"
        f"-webkit-background-clip:text;background-clip:text;color:transparent;'>"
        f"Macro&middot;Micro</div>"
        f"<div style='font-family:{MONO};font-size:9px;letter-spacing:0.22em;"
        f"text-transform:uppercase;color:{MUTED};margin-top:2px;'>Control Panel"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # Dynamic ticker selection — macro is universal; company is ticker-specific.
    selected_ticker = st.selectbox(
        "Ticker",
        options=config.SUPPORTED_TICKERS,
        index=config.SUPPORTED_TICKERS.index(config.TICKER),
    )
    company_ns = config.company_namespace(selected_ticker)

    # Pipeline status (same logic: connect + namespace counts).
    connected = False
    macro_n = company_n = 0
    status_err = None
    try:
        from pipeline import embed_store

        index = get_index()
        macro_n = embed_store.namespace_count(index, config.NAMESPACE_MACRO)
        company_n = embed_store.namespace_count(index, company_ns)
        connected = True
    except Exception as exc:  # noqa: BLE001
        status_err = exc

    st.markdown(
        f"<hr style='border:none;border-top:1px solid {BORDER};margin:12px 0;'>",
        unsafe_allow_html=True,
    )
    if connected:
        status_row(GREEN, "PINECONE", f"{config.PINECONE_INDEX}  connected")
    else:
        status_row(RED, "PINECONE", "not connected")
    status_row(GREEN if connected else GRAY, "MACRO", f"{macro_n} vectors")
    status_row(GREEN if connected else GRAY, "COMPANY", f"{company_n} vectors")
    status_row(GREEN, "LLM", f"{config.LLM_MODEL.split('/')[-1]} · Nebius")
    status_row(GREEN, "EMBEDDER", f"{config.EMBED_MODEL.split('/')[-1]} · local")
    if status_err is not None:
        st.caption(str(status_err))

    st.markdown(
        f"<hr style='border:none;border-top:1px solid {BORDER};margin:14px 0;'>",
        unsafe_allow_html=True,
    )
    render_flow_diagram()


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='display:flex;align-items:center;gap:13px;'>"
    # gradient accent bar tying together the macro/synthesis/company hues
    f"<div style='width:4px;height:40px;border-radius:3px;"
    f"background:linear-gradient(180deg,{TEAL},{PURPLE},{CORAL});'></div>"
    f"<div>"
    f"<div style='font-family:{MONO};font-size:10px;letter-spacing:0.24em;"
    f"text-transform:uppercase;color:{MUTED};margin-bottom:4px;'>"
    f"Macro &times; Micro &middot; Investment Intelligence</div>"
    # gradient-filled wordmark
    f"<div style='font-family:{SANS};font-weight:800;font-size:34px;line-height:1;"
    f"letter-spacing:-0.02em;background:linear-gradient(90deg,{TEAL} 0%,"
    f"{PURPLE} 52%,{CORAL} 100%);-webkit-background-clip:text;"
    f"background-clip:text;color:transparent;'>Macro-Micro Bridge</div>"
    f"</div></div>"
    f"<div style='font-family:{MONO};font-size:13px;color:{MUTED};margin-top:9px;'>"
    f"Investment thesis &middot; macro &times; company signals</div>"
    f"<hr style='border:none;border-top:1px solid {BORDER};margin:14px 0 20px;'>",
    unsafe_allow_html=True,
)

# ── Input row ────────────────────────────────────────────────
col_in, col_btn = st.columns([5, 1])
with col_in:
    question = st.text_input(
        "Investment question",
        placeholder="e.g. Should I be long NVDA given current macro?",
        label_visibility="collapsed",
    )
with col_btn:
    analyze = st.button("Analyze ↗", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Query execution + output
# ──────────────────────────────────────────────────────────────
if analyze and question.strip():
    with st.spinner("Running macro and company agents..."):
        try:
            # Warm cached resources before the graph executes.
            get_embedder()
            get_index()
            app = get_app()
            state = app.invoke(
                {"question": question.strip(), "ticker": selected_ticker}
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            state, error = None, exc

    if error is not None:
        st.error(f"Query failed: {error}")
    else:
        col_macro, col_company, col_synth = st.columns(3)

        with col_macro:
            stream_header("MACRO", TEAL)
            macro_ctx = state.get("macro_context", [])
            macro_src = state.get("macro_sources", [])
            if not macro_ctx:
                empty_state("— no context retrieved —")
            for badge, text in zip(macro_src, macro_ctx):
                chunk_card(badge, text, TEAL)

        with col_company:
            resolved_ticker = state.get("ticker", selected_ticker)
            stream_header(f"COMPANY · {resolved_ticker}", CORAL)
            ticker_note = state.get("ticker_note")
            if ticker_note:
                mono_caption(f"⚠ {ticker_note}")
            mono_caption(f"ns: {config.company_namespace(resolved_ticker)}")
            company_ctx = state.get("company_context", [])
            company_src = state.get("company_sources", [])
            if not company_ctx:
                empty_state("— no context retrieved —")
            for badge, text in zip(company_src, company_ctx):
                chunk_card(badge, text, CORAL)

        with col_synth:
            stream_header("SYNTHESIS", PURPLE)
            verdict = state.get("verdict", "VERDICT: UNDETERMINED")
            render_verdict(verdict)
            st.markdown(
                f"<hr style='border:none;border-top:1px solid {BORDER};"
                f"margin:12px 0;'>",
                unsafe_allow_html=True,
            )
            if verdict == config.VERDICT_INSUFFICIENT:
                # Refusal path: no synthesis was produced; surface guidance.
                refusal_card(resolved_ticker)
            else:
                body_text(state.get("synthesis", "(no synthesis)"), 13)
                mono_label("ANSWER")
                body_text(state.get("answer", "(no answer)"), 14)

elif analyze:
    st.warning("Please enter a question first.")
else:
    st.markdown(
        f"<div style='text-align:center;color:{MUTED};font-family:{MONO};"
        f"font-size:13px;padding:52px 0;'>Enter a question above to analyze</div>",
        unsafe_allow_html=True,
    )
