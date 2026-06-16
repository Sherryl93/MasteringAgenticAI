"""ReviewPilot — Streamlit entry point.

This file owns ONLY layout, session state, and a single call into the backend
(`service.run_review`). All rendering is delegated to `src.ui_components`, and
all review logic lives behind the service layer. The UI can be polished or
replaced without touching the agent backend.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import check_environment, get_settings
from src.service import run_review
from src import ui_components as ui

st.set_page_config(page_title="ReviewPilot", page_icon="🛫", layout="wide")

settings = get_settings()


# --------------------------------------------------------------------- sidebar
def render_sidebar() -> tuple[str, str]:
    st.sidebar.title("🛫 ReviewPilot")
    st.sidebar.caption("Multi-agent code review · LangGraph + Nebius")

    repo_url = st.sidebar.text_input("GitHub repo URL", value=settings.default_repo_url)
    target_path = st.sidebar.text_input("Target folder", value=settings.default_target_path)

    env = check_environment(settings)
    with st.sidebar.expander("Environment", expanded=not env.ok):
        st.write("git:", "✅" if env.git_available else "❌")
        st.write("API key:", "✅" if env.api_key_present else "❌ (tool-only mode)")
        model_icon = {True: "✅", False: "❌", None: "—"}[env.models_validated]
        st.write("primary model:", model_icon)
        st.caption(f"`{settings.model}`")
        st.caption(
            f"fallback: `{settings.fallback_model}` · fast: `{settings.fast_model}`"
        )
        for m in env.messages:
            st.caption(m)

    run = st.sidebar.button("▶ Run Review", type="primary", use_container_width=True)

    # Interactive pipeline diagram, directly below the Run Review button.
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🗺️ Pipeline")
        ui.render_pipeline_diagram(
            result=st.session_state.get("result"),
            approved=bool(st.session_state.get("hitl_approved", False)),
        )

    return repo_url, target_path, run


# ------------------------------------------------------------------- main tabs
def render_results(result) -> None:
    if not result.ok:
        ui.render_status(
            "Review could not complete. See errors below.", "error"
        )
        ui.render_logs(result.logs, result.errors)
        return

    ui.render_metric_cards(result)

    tabs = st.tabs(
        ["Summary", "Code Quality", "Security", "Tests", "Raw Logs", "Markdown Report"]
    )

    with tabs[0]:
        st.subheader("Tool summary")
        ui.render_tool_summaries(result)
        st.subheader("Priority findings")
        ui.render_findings_table(result.findings)

    with tabs[1]:
        st.subheader("🎨 Code Quality")
        st.caption("Correctness, maintainability, and style findings.")
        cq = result.by_category("Correctness Risk", "Maintainability", "Style")
        ui.render_findings_table(cq, "No code-quality findings.")
        ui.render_finding_details(cq)

    with tabs[2]:
        st.subheader("🔒 Security")
        sec = result.by_category("Security Risk")
        ui.render_findings_table(sec, "No security findings.")
        ui.render_finding_details(sec)

    with tabs[3]:
        st.subheader("🧪 Tests & coverage")
        st.caption(result.test_summary)
        tests = result.by_category("Test Coverage Gap")
        ui.render_findings_table(tests, "No test-gap findings.")
        ui.render_finding_details(tests)

    with tabs[4]:
        ui.render_logs(result.logs, result.errors)

    with tabs[5]:
        st.subheader("Markdown report")
        # Human-in-the-loop gate: approval required before download.
        approved = st.checkbox(
            "✅ I have reviewed these findings and approve this report",
            key="hitl_approved",
        )
        st.download_button(
            "⬇ Download report (.md)",
            data=result.report_md,
            file_name="reviewpilot_report.md",
            mime="text/markdown",
            disabled=not approved,
            help="Approve above to enable download.",
        )
        if not approved:
            st.info("Download is disabled until you approve the findings above.")
        st.divider()
        st.markdown(result.report_md)


def main() -> None:
    repo_url, target_path, run = render_sidebar()
    st.title("Code Review Results")

    if run:
        # Reset any prior approval when a new review starts.
        st.session_state.pop("hitl_approved", None)
        with st.status("Running multi-agent review…", expanded=True) as status:
            st.write("Cloning repo, scanning files, running ruff/bandit/static tests…")
            st.write("Fanning out bug / security / test agents in parallel…")
            result = run_review(repo_url, target_path, settings)
            if result.ok:
                status.update(label="Review complete ✅", state="complete")
            else:
                status.update(label="Review failed ❌", state="error")
        st.session_state["result"] = result

    if "result" in st.session_state:
        render_results(st.session_state["result"])
    else:
        st.info("Configure the repo in the sidebar and click **Run Review**.")


if __name__ == "__main__":
    main()
