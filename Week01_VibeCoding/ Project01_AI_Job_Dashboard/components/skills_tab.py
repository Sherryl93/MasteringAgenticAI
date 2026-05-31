"""Tab 3 — Skills & Future-Proofing: skill demand, protection, upskilling, role explorer."""
import streamlit as st
import plotly.graph_objects as go

from utils.config import PLOTLY_TEMPLATE


def render(df_filtered):
    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    st.subheader("🧠 Skills & Future-Proofing")
    st.caption(
        "What skills protect you from AI disruption? "
        "Who needs to upskill the most? Find your path to 2030."
    )

    # -----------------------------------------------------------------------
    # Parse Required_Skills into one row per (record, skill)
    # -----------------------------------------------------------------------
    skills_exploded = df_filtered.assign(
        _skill=df_filtered["Required_Skills"].fillna("").str.split(",")
    ).explode("_skill")
    skills_exploded["_skill"] = skills_exploded["_skill"].str.strip()
    skills_exploded = skills_exploded[skills_exploded["_skill"] != ""]

    skill_counts = skills_exploded["_skill"].value_counts()
    risk_by_skill = skills_exploded.groupby("_skill")["AI_Replacement_Risk"].mean()
    top15_skills = skill_counts.head(15).index

    # -----------------------------------------------------------------------
    # 1. KPI cards
    # -----------------------------------------------------------------------
    upskill_pct = (df_filtered["Upskilling_Needed"] == "Yes").mean() * 100
    most_protective_skill = risk_by_skill.loc[top15_skills].idxmin()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🛠️ Total Unique Skills", skills_exploded["_skill"].nunique())
    kpi2.metric("📚 Roles Needing Upskilling", f"{upskill_pct:.1f}%")
    kpi3.metric("🏆 Most In-Demand Skill", skill_counts.index[0])
    kpi4.metric("🛡️ Most Protective Skill", most_protective_skill)

    # -----------------------------------------------------------------------
    # 2. Skill demand + skill protection
    # -----------------------------------------------------------------------
    col_demand, col_protect = st.columns(2)

    with col_demand:
        top15 = skill_counts.head(15).sort_values(ascending=False)
        fig_skills = go.Figure(
            go.Bar(
                x=top15.values,
                y=top15.index,
                orientation="h",
                marker=dict(color=top15.values, colorscale="Blues"),
                hovertemplate="<b>%{y}</b><br>Demand count: %{x}<extra></extra>",
            )
        )
        fig_skills.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Top 15 Most In-Demand Skills by 2030",
            xaxis_title="Number of Roles Requiring Skill",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_skills, use_container_width=True)

    with col_protect:
        risk_top15 = risk_by_skill.loc[top15_skills].sort_values(ascending=True)
        protect_colors = risk_top15.apply(
            lambda v: "#2ecc71" if v < 0.45 else ("#f39c12" if v <= 0.55 else "#e74c3c")
        )
        fig_protect = go.Figure(
            go.Bar(
                x=risk_top15.values,
                y=risk_top15.index,
                orientation="h",
                marker_color=protect_colors,
                hovertemplate="<b>%{y}</b><br>Avg AI Risk: %{x:.1%}<extra></extra>",
            )
        )
        fig_protect.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Which Skills Lower Your AI Risk?",
            xaxis_title="Avg AI Replacement Risk",
            xaxis_tickformat=".0%",
            xaxis_range=[0, 0.6],
            yaxis=dict(autorange="reversed"),
        )
        fig_protect.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text="Risk Threshold",
            annotation_position="top",
        )
        fig_protect.add_annotation(
            xref="paper", yref="paper", x=0.0, y=1.08,
            text="Lower = Safer ←", showarrow=False,
            font=dict(size=13, color="#2ecc71"),
        )
        st.plotly_chart(fig_protect, use_container_width=True)

    # -----------------------------------------------------------------------
    # 3. Upskilling gap by role (full width)
    # -----------------------------------------------------------------------
    upskill_role = (
        df_filtered.assign(_needed=(df_filtered["Upskilling_Needed"] == "Yes"))
        .groupby("Job_Title", as_index=False)["_needed"]
        .mean()
        .rename(columns={"_needed": "pct"})
        .sort_values("pct", ascending=False)
    )
    upskill_role["pct"] = upskill_role["pct"] * 100
    upskill_role["_color"] = upskill_role["pct"].apply(
        lambda v: "#e74c3c" if v > 55 else ("#f39c12" if v >= 45 else "#2ecc71")
    )
    fig_upskill_role = go.Figure(
        go.Bar(
            x=upskill_role["pct"],
            y=upskill_role["Job_Title"],
            orientation="h",
            marker_color=upskill_role["_color"],
            hovertemplate="<b>%{y}</b><br>Upskilling Needed: %{x:.1f}%<extra></extra>",
        )
    )
    fig_upskill_role.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Upskilling Gap by Role — Who Needs to Catch Up Before 2030?",
        xaxis_title="% Workers Needing Upskilling",
        yaxis=dict(autorange="reversed"),
        height=600,
    )
    st.plotly_chart(fig_upskill_role, use_container_width=True)
    st.caption(
        "💡 Roles above 55% urgently need workforce "
        "reskilling programs to stay relevant by 2030."
    )

    # -----------------------------------------------------------------------
    # 4. Explore a specific role
    # -----------------------------------------------------------------------
    with st.expander("🔍 Explore a Specific Role", expanded=True):
        role_options = sorted(df_filtered["Job_Title"].dropna().unique())
        selected_role = st.selectbox("Choose a Job Role", options=role_options)

        role_df = df_filtered[df_filtered["Job_Title"] == selected_role]

        role_salary = role_df["Average_Salary_USD"].mean()
        role_risk = role_df["AI_Replacement_Risk"].mean()
        role_demand = role_df["Future_Demand_Score"].mean()
        role_growth = role_df["Job_Growth_2030"].mean()
        role_upskill_pct = (role_df["Upskilling_Needed"] == "Yes").mean() * 100

        risk_badge = "🟢" if role_risk < 0.4 else ("🟡" if role_risk <= 0.7 else "🔴")

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Salary", f"${role_salary:,.0f}")
        m2.metric("AI Risk", f"{risk_badge} {role_risk:.1%}")
        m3.metric("Future Demand", f"{role_demand:.1%}")

        col_role_skills, col_role_trend = st.columns(2)

        with col_role_skills:
            role_skills = role_df.assign(
                _skill=role_df["Required_Skills"].fillna("").str.split(",")
            ).explode("_skill")
            role_skills["_skill"] = role_skills["_skill"].str.strip()
            role_skills = role_skills[role_skills["_skill"] != ""]
            top10_role = role_skills["_skill"].value_counts().head(10).sort_values(
                ascending=False
            )
            fig_role_skills = go.Figure(
                go.Bar(
                    x=top10_role.values,
                    y=top10_role.index,
                    orientation="h",
                    marker=dict(color=top10_role.values, colorscale="Blues"),
                    hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
                )
            )
            fig_role_skills.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=f"Top 10 Skills for {selected_role}",
                xaxis_title="Count",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_role_skills, use_container_width=True)

        with col_role_trend:
            trend_counts = role_df["Hiring_Trend_2026"].value_counts()
            trend_color_map = {
                "Growing": "#2ecc71",
                "Stable": "#f39c12",
                "Declining": "#e74c3c",
            }
            trend_colors = [
                trend_color_map.get(t, "#888888") for t in trend_counts.index
            ]
            fig_trend = go.Figure(
                go.Pie(
                    labels=trend_counts.index,
                    values=trend_counts.values,
                    marker=dict(colors=trend_colors),
                )
            )
            fig_trend.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=f"Hiring Trend 2026 — {selected_role}",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.metric(
            "📚 Upskilling Required",
            f"{role_upskill_pct:.0f}% of {selected_role} workers need "
            "upskilling to stay relevant by 2030",
        )

        risk_level = "LOW" if role_risk < 0.4 else ("MEDIUM" if role_risk <= 0.7 else "HIGH")
        growth_word = "grow" if role_growth >= 0 else "decline"
        st.info(
            f"Based on current data: {selected_role} has a {risk_level} AI "
            f"replacement risk of {role_risk:.0%}. With a future demand score "
            f"of {role_demand:.0%}, this role is projected to {growth_word} by "
            f"{abs(role_growth):.0f}% before 2030. {role_upskill_pct:.0f}% of "
            "current workers need upskilling."
        )
