"""Tab 1 — Survival Matrix: KPIs, AI survival scatter, job growth, top-5 tables."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.config import PLOTLY_TEMPLATE


def render(df_filtered):
    if df_filtered.empty:
        st.warning("No data matches the current filters. Adjust the filters in the sidebar.")
        return

    # -----------------------------------------------------------------------
    # 1. KPI metric cards
    # -----------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    avg_risk = df_filtered["AI_Replacement_Risk"].mean()
    avg_demand = df_filtered["Future_Demand_Score"].mean()
    avg_growth = df_filtered["Job_Growth_2030"].mean()
    pct_upskill = (df_filtered["Upskilling_Needed"] == "Yes").mean() * 100

    col1.metric("⚠️ Avg AI Risk", f"{avg_risk * 100:.1f}%")
    col2.metric("📈 Avg Future Demand", f"{avg_demand * 100:.1f}%")
    col3.metric("🌱 Avg Job Growth", f"{'+' if avg_growth > 0 else ''}{avg_growth:.1f}")
    col4.metric("📚 Need Upskilling", f"{pct_upskill:.1f}%")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 2. Hero scatter — AI Survival Matrix
    # -----------------------------------------------------------------------
    st.subheader("AI Survival Matrix")

    role_agg = (
        df_filtered.groupby("Job_Title", as_index=False)
        .agg(
            AI_Replacement_Risk=("AI_Replacement_Risk", "mean"),
            Future_Demand_Score=("Future_Demand_Score", "mean"),
            Average_Salary_USD=("Average_Salary_USD", "mean"),
            Job_Growth_2030=("Job_Growth_2030", "mean"),
        )
    )

    # Shift growth so all values are positive before sizing
    growth_shifted = role_agg["Job_Growth_2030"] + abs(role_agg["Job_Growth_2030"].min()) + 1
    role_agg["_size"] = growth_shifted

    fig_scatter = px.scatter(
        role_agg,
        x="AI_Replacement_Risk",
        y="Future_Demand_Score",
        size="_size",
        color="Average_Salary_USD",
        color_continuous_scale="viridis",
        size_max=40,
        custom_data=["Job_Title", "Average_Salary_USD", "Job_Growth_2030"],
    )
    fig_scatter.update_traces(
        marker=dict(sizemin=10),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "AI Risk: %{x:.0%}<br>"
            "Future Demand: %{y:.0%}<br>"
            "Avg Salary: $%{customdata[1]:,.0f}<br>"
            "Job Growth: %{customdata[2]:.1f}%<extra></extra>"
        ),
    )

    # Quadrant divider lines
    fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray")

    # Quadrant annotations
    quadrants = {
        (0.25, 0.80): "🟢 Safe Zone",
        (0.75, 0.80): "🟡 Adapt or Thrive",
        (0.25, 0.25): "🔵 Stable but Stagnant",
        (0.75, 0.25): "🔴 High Danger",
    }
    for (qx, qy), label in quadrants.items():
        fig_scatter.add_annotation(
            x=qx, y=qy, text=label, showarrow=False,
            font=dict(size=14), opacity=0.85,
        )

    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="AI Replacement Risk",
        yaxis_title="Future Demand Score",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------------
    # 3. Horizontal bar — Projected Job Growth by 2030
    # -----------------------------------------------------------------------
    st.subheader("Projected Job Growth by 2030")

    growth_by_role = (
        df_filtered.groupby("Job_Title", as_index=False)["Job_Growth_2030"]
        .mean()
        .sort_values("Job_Growth_2030", ascending=True)
    )
    growth_by_role["_color"] = growth_by_role["Job_Growth_2030"].apply(
        lambda v: "#2ecc71" if v >= 0 else "#e74c3c"
    )

    fig_growth = go.Figure(
        go.Bar(
            x=growth_by_role["Job_Growth_2030"],
            y=growth_by_role["Job_Title"],
            orientation="h",
            marker_color=growth_by_role["_color"],
            hovertemplate="<b>%{y}</b><br>Job Growth: %{x:.1f}%<extra></extra>",
        )
    )
    fig_growth.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Avg Projected Job Growth 2030 (%)",
        yaxis_title="",
        height=600,
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    st.info(
        "⚠️ Note: Hiring Trend 2026 (sidebar filter) reflects current "
        "employer hiring sentiment this year. Job Growth 2030 reflects "
        "long-term projections. A role can be declining to hire today "
        "but still projected to grow by 2030 — try filtering by "
        "'Declining' in the sidebar to see this in action."
    )

    # -----------------------------------------------------------------------
    # 4. Top 5 Roles by Hiring Trend 2026
    # -----------------------------------------------------------------------
    st.subheader("📋 Top 5 Roles by Hiring Trend 2026")

    def _top5_by_trend(trend, sort_col, ascending):
        grouped = (
            df_filtered[df_filtered["Hiring_Trend_2026"] == trend]
            .groupby("Job_Title", as_index=False)
            .agg(
                **{
                    "AI Risk": ("AI_Replacement_Risk", "mean"),
                    "Demand": ("Future_Demand_Score", "mean"),
                    "Avg Salary": ("Average_Salary_USD", "mean"),
                }
            )
            .sort_values(sort_col, ascending=ascending)
            .head(5)
        )
        return grouped.style.format(
            {
                "AI Risk": "{:.2%}",
                "Demand": "{:.2%}",
                "Avg Salary": "${:,.0f}",
            }
        )

    col_growing, col_stable, col_declining = st.columns(3)

    with col_growing:
        st.markdown("🟢 Growing")
        st.dataframe(
            _top5_by_trend("Growing", "Demand", False),
            use_container_width=True,
            hide_index=True,
        )

    with col_stable:
        st.markdown("🟡 Stable")
        st.dataframe(
            _top5_by_trend("Stable", "AI Risk", True),
            use_container_width=True,
            hide_index=True,
        )

    with col_declining:
        st.markdown("🔴 Declining")
        st.dataframe(
            _top5_by_trend("Declining", "Demand", False),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "💡 Declining ≠ Dead. Some declining-to-hire roles still "
        "show strong future demand — they may be pausing hiring now "
        "while remaining valuable long-term."
    )
