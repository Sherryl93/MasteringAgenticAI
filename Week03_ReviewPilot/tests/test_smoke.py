"""Smoke tests for ReviewPilot.

These run WITHOUT a Nebius API key and WITHOUT network access: they exercise
the deterministic backend (state contracts, tool parsing, ranking, report
rendering, graph compilation) so the pipeline is verifiably correct even in
tool-only mode. A network-dependent end-to-end test is provided separately
and skipped by default.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.agents._base as agent_base  # noqa: E402
from src.agents.ranking_agent import ranking_agent  # noqa: E402
from src.agents.report_agent import build_report_md  # noqa: E402
from src.agents.security_agent import security_agent  # noqa: E402
from src.agents.test_agent import prioritize_modules  # noqa: E402
from src.agents.test_agent import test_agent as run_test_agent  # noqa: E402
from src.config import Settings  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.repo_loader import scan_python_files, validate_repo_url  # noqa: E402
from src.service import run_review  # noqa: E402
from src.state import Finding  # noqa: E402
from src.tools.test_analyzer import analyze_tests  # noqa: E402
from src.utils.classify import classify, dedupe  # noqa: E402
from src.utils.findings import (  # noqa: E402
    findings_from_bandit,
    findings_from_ruff,
    normalize_path,
)


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_finding_schema_defaults():
    f = Finding(agent="bug", title="x")
    assert f.severity == "Medium"
    assert f.model_dump()["agent"] == "bug"


def test_ruff_conversion():
    diagnostics = [
        {"code": "F401", "message": "unused import", "filename": "a.py",
         "location": {"row": 3}}
    ]
    out = findings_from_ruff(diagnostics)
    assert len(out) == 1 and out[0].agent == "bug" and out[0].line == 3


def test_normalize_path_strips_local_machine_paths():
    repo = r"C:\Users\me\.tmp_repos\repo-abc"
    # Absolute (ruff-style) path with the clone dir prefix.
    abs_p = r"C:\Users\me\.tmp_repos\repo-abc\pkg\main.py"
    assert normalize_path(abs_p, repo) == "pkg/main.py"
    # Already-relative (bandit-style) backslash path.
    assert normalize_path(r"pkg\util.py", repo) == "pkg/util.py"
    # No local prefix should leak through.
    assert "C:" not in normalize_path(abs_p, repo)


def test_ruff_paths_are_repo_relative():
    repo = "/tmp/.tmp_repos/r"
    diagnostics = [{
        "code": "F541", "message": "f-string", "filename": "/tmp/.tmp_repos/r/pkg/main.py",
        "location": {"row": 1},
    }]
    out = findings_from_ruff(diagnostics, repo)
    assert out[0].file == "pkg/main.py"


def test_bandit_high_confidence_promotes_to_critical():
    results = [
        {"test_id": "B602", "issue_text": "subprocess shell=True",
         "filename": "a.py", "line_number": 10,
         "issue_severity": "HIGH", "issue_confidence": "HIGH"}
    ]
    out = findings_from_bandit(results)
    assert out[0].severity == "Critical"


def test_classify_style_capped_to_low_and_title_cleaned():
    f = Finding(agent="bug", severity="High", file="a.py", line=5,
                title="[F541] f-string without placeholders",
                evidence={"rule": "F541", "source": "ruff"})
    classify(f)
    assert f.category == "Style"
    assert f.severity == "Low"                 # style capped to Low
    assert f.title == "f-string without placeholders"  # [F541] prefix stripped


def test_dedupe_merges_pickle_finding():
    group = [
        Finding(agent="security", severity="Medium", file="x.py", line=600,
                title="[B301] Pickle and modules that wrap it can be unsafe",
                evidence={"test_id": "B301", "source": "bandit"}),
        Finding(agent="security", severity="High", file="x.py", line=600,
                title="Unsafe Deserialization via pickle.load() on Untrusted Data",
                rationale="pickle.load on attacker-controlled bytes enables RCE"),
    ]
    merged = dedupe([classify(f) for f in group])
    assert len(merged) == 1
    m = merged[0]
    assert "Unsafe Deserialization" in m.title      # LLM-enriched title preferred
    assert m.severity == "High"                     # higher severity retained
    assert "B301" in (m.evidence.get("rules") or [])  # tool ID kept as evidence


def test_dedupe_collapses_repeated_f541_and_ranks_below_security():
    findings = [
        Finding(agent="bug", severity="Low", file="m.py", line=70,
                title="[F541] f-string without any placeholders",
                evidence={"rule": "F541", "source": "ruff"}).model_dump(),
        Finding(agent="bug", severity="Low", file="m.py", line=74,
                title="[F541] f-string without any placeholders",
                evidence={"rule": "F541", "source": "ruff"}).model_dump(),
        Finding(agent="security", severity="Medium", file="m.py", line=600,
                title="[B301] pickle.load unsafe deserialization",
                evidence={"test_id": "B301", "source": "bandit"}).model_dump(),
    ]
    ranked = ranking_agent({"findings": findings})["ranked"]
    style = [f for f in ranked if f["category"] == "Style"]
    assert len(style) == 1                       # two F541s collapsed into one
    assert style[0]["severity"] == "Low"
    assert ranked[0]["category"] == "Security Risk"  # security outranks style


def test_ranking_dedups_and_sorts():
    findings = [
        Finding(agent="bug", severity="Low", file="a.py", title="dup").model_dump(),
        Finding(agent="bug", severity="Low", file="a.py", title="dup").model_dump(),
        Finding(agent="security", severity="Critical", file="b.py", title="sqli").model_dump(),
    ]
    out = ranking_agent({"findings": findings})
    ranked = out["ranked"]
    assert len(ranked) == 2  # duplicate removed
    assert ranked[0]["severity"] == "Critical"  # sorted first
    assert ranked[0]["id"] == "F-001"


def test_static_test_analysis(tmp_path):
    # One source module, one test file -> ratio and untested detection work.
    (tmp_path / "calc.py").write_text("def add(a,b):\n    return a+b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(1,2)==3\n"
    )
    files = ["calc.py", "test_calc.py"]
    result = analyze_tests(str(tmp_path), files)
    assert result["ok"] is True
    assert result["data"]["n_test_files"] == 1
    assert result["data"]["total_test_functions"] == 1
    assert "calc.py" not in result["data"]["untested_modules"]


def test_test_agent_groups_untested_files(monkeypatch):
    # Force the degraded (tool-only) path so the test is hermetic regardless
    # of whether a real API key is present in the environment.
    class _NoLLM:
        def structured_findings(self, system, user, agent):
            return None

    monkeypatch.setattr(agent_base, "get_client", lambda: _NoLLM())

    # 15 untested modules, no test files -> ONE grouped finding, not 15.
    untested = [f"pkg/mod_{i}.py" for i in range(15)] + ["pkg/__init__.py"]
    state = {
        "repo_dir": "",
        "python_files": [],
        "test_analysis": {"data": {
            "n_source_files": 16, "n_test_files": 0,
            "untested_modules": untested,
        }},
    }
    out = run_test_agent(state)  # forced degraded path
    findings = out["findings"]
    assert len(findings) == 1
    assert findings[0]["title"] == "No test suite detected"
    # Capped at Medium so it cannot outrank real security/bug findings.
    assert findings[0]["severity"] == "Medium"
    # __init__.py is deprioritized out of the top list.
    assert "pkg/__init__.py" not in findings[0]["evidence"]["priority_modules"]


def test_prioritize_modules_deprioritizes_init():
    mods = ["pkg/__init__.py", "pkg/main.py", "pkg/helpers.py"]
    ranked = prioritize_modules(mods)
    assert ranked[0] != "pkg/__init__.py"
    assert ranked[-1] == "pkg/__init__.py"


def test_report_renders_with_no_llm():
    state = {
        "repo_url": "https://github.com/x/y",
        "target_path": "pkg",
        "python_files": ["a.py"],
        "ranked": [
            Finding(agent="security", severity="High", file="a.py", line=2,
                    title="hardcoded secret").model_dump()
        ],
        "ruff": {"summary": "ruff: 0 findings."},
        "bandit": {"summary": "bandit: 1 issue."},
        "test_analysis": {"summary": "static tests: 0 test files."},
        "llm_degraded": True,
    }
    md = build_report_md(state)
    assert "# ReviewPilot Report" in md
    assert "hardcoded secret" in md
    assert "tool-derived findings" in md  # degraded banner present


def test_ranking_security_outranks_test_at_same_severity():
    findings = [
        Finding(agent="test", severity="Medium", file="z.py", title="No test suite").model_dump(),
        Finding(agent="security", severity="Medium", file="a.py", title="pickle").model_dump(),
        Finding(agent="bug", severity="Medium", file="b.py", title="bug").model_dump(),
    ]
    ranked = ranking_agent({"findings": findings})["ranked"]
    order = [f["agent"] for f in ranked]
    assert order == ["security", "bug", "test"]  # security/bug above test


def test_ranking_caps_test_severity_below_security():
    # Even if an LLM rates a test finding Critical, it must not outrank a
    # real Medium security finding.
    findings = [
        Finding(agent="test", severity="Critical", file="z.py", title="No test suite").model_dump(),
        Finding(agent="security", severity="Medium", file="a.py", title="pickle").model_dump(),
    ]
    ranked = ranking_agent({"findings": findings})["ranked"]
    test_row = next(f for f in ranked if f["agent"] == "test")
    assert test_row["severity"] == "Medium"        # capped down from Critical
    assert ranked[0]["agent"] == "security"        # security now ranks first


def test_ranking_real_bug_outranks_low_test():
    findings = [
        Finding(agent="test", severity="Medium", file="z.py", title="No test suite").model_dump(),
        Finding(agent="security", severity="Critical", file="a.py", title="rce").model_dump(),
    ]
    ranked = ranking_agent({"findings": findings})["ranked"]
    assert ranked[0]["agent"] == "security"  # Critical security first


def test_pipeline_mermaid_builds_with_states():
    from src.ui_components import build_pipeline_mermaid

    # Pre-run: vibrant role-based colours.
    base = build_pipeline_mermaid(done=False, degraded=False, approved=False)
    assert base.startswith("flowchart TD")
    assert "subgraph PAR" in base  # parallel fan-out shown
    assert "classDef agentC" in base  # vibrant palette present
    assert "class bug,sec,test agentC" in base  # agents coloured by role

    # Completed + degraded + unapproved: agents orange, gate awaiting (blue).
    done = build_pipeline_mermaid(done=True, degraded=True, approved=False)
    assert "class bug,sec,test degC" in done  # degraded agents highlighted
    assert "class gate waitC" in done  # gate awaiting approval


def test_executive_summary_present():
    state = {
        "repo_url": "https://github.com/x/y", "target_path": "pkg",
        "python_files": ["a.py"],
        "ranked": [
            Finding(agent="security", severity="Critical", file="a.py", title="rce").model_dump(),
            Finding(agent="test", severity="Medium", file="b.py", title="No test suite detected").model_dump(),
        ],
        "ruff": {"summary": "ruff: ok"}, "bandit": {"summary": "bandit: ok"},
        "test_analysis": {"summary": "tests: ok"},
    }
    md = build_report_md(state)
    assert "## Executive Summary" in md
    assert "Recommended focus" in md


# ---- Edge cases: bad URL, wrong folder, no Python files (no network) -------
def test_bad_repo_url_rejected():
    ok, _ = validate_repo_url("not-a-real-url")
    assert ok is False


def test_run_review_bad_url_degrades_cleanly():
    # validate_input raises before any network call; service returns ok=False.
    r = run_review("not-a-real-url", "")
    assert r.ok is False
    assert r.errors


def test_scan_wrong_folder(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    files, msg = scan_python_files(str(tmp_path), "does_not_exist")
    assert files == [] and "not found" in msg.lower()


def test_scan_no_python_files(tmp_path):
    (tmp_path / "README.md").write_text("# docs\n")
    files, msg = scan_python_files(str(tmp_path), "")
    assert files == [] and "no python files" in msg.lower()


# ---- Nebius LLM path (mocked; no live key required) ------------------------
def test_llm_client_parses_structured_findings():
    client = LLMClient(Settings(nebius_api_key="test-key"))
    canned = (
        '{"findings":[{"severity":"High","file":"pkg/main.py","line":10,'
        '"title":"SQL injection","rationale":"unparameterized query",'
        '"suggestion":"use bound parameters"}]}'
    )
    client._raw_complete = lambda model, system, user: canned  # type: ignore
    out = client.structured_findings("sys", "usr", agent="security")
    assert out is not None and len(out) == 1
    assert out[0].agent == "security" and out[0].severity == "High"
    assert out[0].file == "pkg/main.py"


def test_security_agent_llm_path_non_degraded(monkeypatch):
    class FakeClient:
        last_model = "fake-model-v1"

        def structured_findings(self, system, user, agent):
            # Model returns an absolute path; agent must normalize it.
            return [Finding(agent=agent, severity="Critical",
                            file="/abs/.tmp/r/pkg/x.py", title="hardcoded secret")]

    monkeypatch.setattr(agent_base, "get_client", lambda: FakeClient())
    state = {"repo_dir": "/abs/.tmp/r", "python_files": [], "bandit": {"data": []}}
    out = security_agent(state)
    assert out["llm_degraded"] is False
    files = [f["file"] for f in out["findings"]]
    assert files == ["pkg/x.py"]  # normalized, no local path leak
    assert out["models_used"] == ["fake-model-v1"]  # model surfaced


@pytest.mark.skipif(
    not os.getenv("REVIEWPILOT_E2E"),
    reason="network/LLM end-to-end test; set REVIEWPILOT_E2E=1 to enable",
)
def test_end_to_end_demo_repo():
    r = run_review(
        "https://github.com/Sherryl93/MasteringAgenticAI", "Week02_RAGApplication"
    )
    assert r.ok
    assert r.total > 0
    assert "# ReviewPilot Report" in r.report_md
