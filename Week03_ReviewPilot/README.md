# 🛫 ReviewPilot — Multi-Agent Code Review Assistant

> Point it at a Python GitHub repo and get a prioritized, human-approved code review in under 5 minutes — bugs, security, and missing tests — **without ever running the repo's code.**

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg">
  <img alt="LangGraph" src="https://img.shields.io/badge/orchestration-LangGraph-7c3aed.svg">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Qwen3--235B%20via%20Nebius-f97362.svg">
  <img alt="UI" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg">
  <img alt="Static analysis" src="https://img.shields.io/badge/static-Ruff%20%2B%20Bandit-0d9488.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-25%20passing-brightgreen.svg">
  <img alt="Read-only" src="https://img.shields.io/badge/mode-read--only%20%C2%B7%20no%20auto--fix-lightgrey.svg">
</p>

ReviewPilot is a **LangGraph multi-agent** system that reviews **public Python GitHub repositories** for bugs, anti-patterns, security risks, and test-coverage gaps. It clones a repo read-only, runs static analysis tools, dispatches three LLM review agents **in parallel**, then **deduplicates, classifies, ranks**, and renders the findings into a Markdown report — all behind a **human approval gate**. It never pushes code, opens PRs, modifies the repo, or executes the target's code.

---

## Table of contents

- [Features](#features)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [How findings are ranked](#how-findings-are-ranked)
- [Why static analysis (no `pytest`)](#why-static-analysis-no-pytest)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Guarantees](#guarantees)

---

## Features

- 🧠 **Multi-agent LangGraph pipeline** — `bug`, `security`, and `test` agents run in **parallel** (fan-out → fan-in), with deterministic `ranking` and `report` agents.
- 🔁 **Never breaks** — per-model **retry → fallback → fast** model, and a **tool-only** degradation path. With no API key or a full LLM outage you *still* get a report (Ruff + Bandit + static tests), clearly flagged.
- 🔒 **Static-only & safe** — Ruff + Bandit + AST test analysis. The target repo's code is **never executed**. ([why?](#why-static-analysis-no-pytest))
- 🎯 **Prioritized, deduplicated report** — an **executive summary**, findings grouped by **category** (Correctness / Security / Test Coverage / Maintainability / Style), severity rules so **style can never outrank security**, and duplicate tool+LLM findings **merged into one**.
- 🧑‍⚖️ **Human-in-the-loop** — the report download is gated behind an explicit **approval checkbox**.
- 🗺️ **Interactive Streamlit UI** — sidebar inputs, a live **model-status** panel, a **vibrant pipeline diagram**, metric cards, and per-area tabs. UI rendering is fully separated from the backend.
- 🧹 **No leaks** — paths are normalized to repo-relative form; local machine paths never appear in the report.

---

## Quick start

```powershell
# Windows PowerShell
cd reviewpilot
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env        # then add your NEBIUS_API_KEY
python -m streamlit run app.py
```

```bash
# macOS / Linux
cd reviewpilot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then add your NEBIUS_API_KEY
streamlit run app.py
```

In the UI: enter a **GitHub URL** + **target folder**, click **▶ Run Review**, inspect findings across the tabs, then tick the **approval checkbox** to enable the report download.

> 💡 **No API key?** ReviewPilot still runs — it produces a **tool-only report** from Ruff, Bandit, and static test analysis, clearly flagged as degraded.

---

## Architecture

```
                              START
                                │
                          validate_input
                                │
                           clone_repo            (GitPython · read-only · depth 1)
                                │
                        scan_python_files        (AST · capped by MAX_FILES)
                                │
            run_ruff  →  run_bandit  →  static_test_analysis
                       ┌────────┼────────┐
                  bug_agent  security_agent  test_agent   ← PARALLEL fan-out
                   (+Ruff)     (+Bandit)   (static tests)
                       └────────┼────────┘
                          ranking_agent          ← fan-in: classify → dedupe → cap → sort
                                │
                           report_agent          (deterministic Markdown)
                                │
                         human_review_gate       (approval enforced in the UI)
                                │
                               END
```

The three analysis agents fan out and back in at `ranking_agent`; concurrent writes to shared state merge via `operator.add` / `operator.or_` reducers.

**Backend / UI separation:** the UI calls exactly **one** backend function — `src/service.py::run_review` — which returns a plain `ReviewResult`. All rendering lives in `src/ui_components.py`. You can redesign the interface without touching agents, tools, or the graph.

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | **LangGraph** `StateGraph`, `TypedDict` state, `operator.add` / `operator.or_` reducers, parallel fan-out |
| LLM | **Nebius Token Factory** (OpenAI-compatible). Primary `Qwen/Qwen3-235B-A22B-Instruct-2507`, fallback `deepseek-ai/DeepSeek-V3.2`, fast `openai/gpt-oss-120b-fast` |
| Structured output | JSON mode + **Pydantic** `Finding` schema + parse-repair fallback |
| Lint / anti-patterns | **Ruff** (`python -m ruff`, JSON) |
| Security scan | **Bandit** (AST-based — runs no target code) |
| Test analysis | Custom **AST analyzer** (test discovery, ratio, untested modules) — no `pytest` on target |
| Repo access | **GitPython** — shallow, read-only clone |
| Ranking | Deterministic classify → dedupe/merge → severity caps → category-priority sort |
| Dashboard | **Streamlit** + `streamlit-mermaid` pipeline diagram |
| Tests | **pytest** smoke suite (offline) + opt-in live E2E |

---

## How findings are ranked

Every finding is assigned a **category** and ordered so real defects come first:

| Category | Example | Severity rule |
|---|---|---|
| 🔐 Security Risk | Unsafe `pickle.load` (B301) | as detected (HIGH+HIGH → Critical) |
| ⚠️ Correctness Risk | Race condition, unguarded env var | as detected |
| 🧪 Test Coverage Gap | "No test suite detected" | capped at **Medium** |
| 🔧 Maintainability | Unused import / dead code | capped at Medium |
| 🎨 Style | f-string without placeholders (F541) | capped at **Low** |

Three deterministic rules make the report trustworthy:

1. **Style never outranks real issues** — style is capped at Low; a recognized *root cause* overrides an LLM that mislabels a nit as a "Medium bug."
2. **Deduplication** — findings about the same `(file, root-cause)` are **merged into one**, keeping the LLM-enriched title + the higher severity + the **tool ID as evidence** (so you see one *"Unsafe Deserialization via pickle.load"*, not that plus a raw *"[B301] …"* line).
3. **Content-aware categories** — a finding's category reflects its *nature*, not just which agent emitted it.

---

## Why static analysis (no `pytest`)

ReviewPilot **does not execute the target repository's code or test suite.** The Test Agent reasons over a static, AST-based analysis instead. This is deliberate:

1. **Safety — no untrusted code execution.** Running an arbitrary repo's `pytest` executes that repo's code on your host (import-time side effects, `conftest.py`, fixtures). For a tool meant to point at *any* GitHub URL, that's an unacceptable risk.
2. **Reproducibility.** Static analysis gives the same result on any machine; live execution depends on the repo's full dependency tree, env vars, network, and secrets.
3. **Reliability.** Most target repos need heavy uninstalled dependencies to even import, so their tests would fail to collect — making any "coverage" number cosmetic.

**Trade-off (stated honestly):** no runtime line-coverage percentage. For an advisory tool that flags *where* tests are missing, the structural signal is sufficient and far safer. Ruff and Bandit are used precisely because they too are purely static (AST-based) and never import the code.

---

## Configuration

Copy `.env.example` → `.env` and add your key. `.env` is git-ignored — **never commit it.**

| Variable | Purpose |
|---|---|
| `NEBIUS_API_KEY` | Nebius Token Factory key (omit → tool-only mode) |
| `NEBIUS_BASE_URL` | `https://api.studio.nebius.com/v1` |
| `NEBIUS_MODEL` | Primary — default `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| `NEBIUS_FALLBACK_MODEL` | Fallback — default `deepseek-ai/DeepSeek-V3.2` |
| `NEBIUS_FAST_MODEL` | Fast fallback — default `openai/gpt-oss-120b-fast` |
| `DEFAULT_REPO_URL` / `DEFAULT_TARGET_PATH` | UI defaults |
| `MAX_FILES_TO_REVIEW` / `MAX_FILE_CHARS` | Token / cost bounds |
| `CLONE_DIR` | Where repos are shallow-cloned (default `.tmp_repos`) |

<details>
<summary><b>Note on model IDs &amp; cost</b></summary>

All configured IDs are validated against the Nebius `/models` endpoint at startup; the sidebar reflects whether the **primary** model is available. The primary is an **Instruct** model chosen for reliable JSON output — reasoning models can return empty content under JSON mode. A typical review makes **3 LLM calls (~52k tokens)** and costs roughly **1–2 cents**.
</details>

---

## Usage

```powershell
# Launch the app
python -m streamlit run app.py

# Run the smoke tests (offline, deterministic)
python -m pytest tests/ -q

# Also run the live demo-repo end-to-end test (needs a key + network)
$env:REVIEWPILOT_E2E=1 ; python -m pytest -q
```

The sidebar shows an **Environment** panel (git ✅, API key ✅, primary model validated ✅) and a **live pipeline diagram** that turns green as the run completes. After a review, the **🤖 Model used** banner confirms which model actually answered.

---

## Project structure

```
reviewpilot/
├── app.py                 # Streamlit entry (layout + 1 backend call)
├── requirements.txt
├── .env.example
├── reports/
│   └── sample_report.md   # a committed example report
├── src/
│   ├── config.py          # settings + startup validation (git, models)
│   ├── state.py           # ReviewState + Finding contract + reducers
│   ├── graph.py           # LangGraph wiring (parallel fan-out)
│   ├── service.py         # run_review() — the backend/UI seam
│   ├── repo_loader.py     # clone + scan (read-only)
│   ├── llm_client.py      # Nebius client: timeout/retry/fallback/JSON
│   ├── ui_components.py   # pure Streamlit render fns + pipeline diagram
│   ├── tools/             # ruff, bandit, static test analyzer
│   ├── agents/            # bug, security, test, ranking, report (+ prompts)
│   └── utils/             # batching, tool→Finding conversion, classify/dedupe
└── tests/
    └── test_smoke.py
```

---

## Troubleshooting

- **`streamlit` is not recognized** — the installer warns it isn't on PATH. Use the module form: `python -m streamlit run app.py`.
- **`git is not on PATH`** (Environment panel ❌) — install Git and restart the terminal; GitPython needs a real `git` to clone.
- **`models ❌` in the sidebar** — the **primary** model ID isn't in the Nebius catalog. Check `NEBIUS_MODEL`; the run will still work via a fallback, but fix the ID for the intended model.
- **Report shows a "tool-only" banner** — no `NEBIUS_API_KEY`, or all models failed. Add a valid key; this is the graceful-degradation path, not a crash.
- **Windows console `UnicodeEncodeError`** — only affects printing emoji to a strict cp1252 console; the app/report (UTF-8) are unaffected. Set `PYTHONUTF8=1` if running CLI snippets.
- **Never commit `.env`** — it's git-ignored along with `.tmp_repos/`, `__pycache__/`, `*.log`, and generated reports.

---

## Guarantees

ReviewPilot **never** pushes code, creates PRs, modifies repos, deletes files, executes target code, or auto-fixes. It only clones (read-only), analyzes **statically**, and generates findings and a Markdown report. The tool advises — **a human decides.**
