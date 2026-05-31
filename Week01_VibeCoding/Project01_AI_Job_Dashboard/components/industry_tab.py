"""Tab 2 — Industry Deep-Dive: KPIs, risk/demand, salary-vs-growth, heatmaps."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.config import PLOTLY_TEMPLATE


def render(df_filtered):
    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    st.subheader("🏭 Industry Deep-Dive")
    st.caption(
        "How does AI impact vary across industries? "
        "Use the sidebar to filter by Automation Level "
        "and Country to compare."
    )

    # -----------------------------------------------------------------------
    # 1. KPI cards
    # -----------------------------------------------------------------------
    industry_means = df_filtered.groupby("Industry").agg(
        salary=("Average_Salary_USD", "mean"),
        risk=("AI_Replacement_Risk", "mean"),
        growth=("Job_Growth_2030", "mean"),
    )

    top_salary_ind = industry_means["salary"].idxmax()
    top_risk_ind = industry_means["risk"].idxmax()
    top_growth_ind = industry_means["growth"].idxmax()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🏭 Industries Analyzed", df_filtered["Industry"].nunique())
    kpi2.metric(
        "💰 Highest Avg Salary",
        f"{top_salary_ind} · ${industry_means.loc[top_salary_ind, 'salary']:,.0f}",
    )
    kpi3.metric(
        "⚠️ Most At-Risk Industry",
        f"{top_risk_ind} · {industry_means.loc[top_risk_ind, 'risk']:.1%}",
    )
    kpi4.metric(
        "🌱 Fastest Growing",
        f"{top_growth_ind} · {industry_means.loc[top_growth_ind, 'growth']:+.1f}%",
    )

    # -----------------------------------------------------------------------
    # 2. Industry risk + future demand
    # -----------------------------------------------------------------------
    col_risk, col_demand = st.columns(2)

    with col_risk:
        risk_by_ind = (
            df_filtered.groupby("Industry", as_index=False)["AI_Replacement_Risk"]
            .mean()
            .sort_values("AI_Replacement_Risk", ascending=False)
        )
        risk_by_ind["_color"] = risk_by_ind["AI_Replacement_Risk"].apply(
            lambda v: "#e74c3c" if v > 0.53 else ("#f39c12" if v >= 0.49 else "#2ecc71")
        )
        fig_ind_risk = go.Figure(
            go.Bar(
                x=risk_by_ind["AI_Replacement_Risk"],
                y=risk_by_ind["Industry"],
                orientation="h",
                marker_color=risk_by_ind["_color"],
                hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1%}<extra></extra>",
            )
        )
        fig_ind_risk.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="AI Replacement Risk by Industry",
            xaxis_title="Avg AI Replacement Risk",
            xaxis_tickformat=".0%",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_ind_risk, use_container_width=True)

    with col_demand:
        demand_by_ind = (
            df_filtered.groupby("Industry", as_index=False)["Future_Demand_Score"]
            .mean()
            .sort_values("Future_Demand_Score", ascending=False)
        )
        fig_ind_demand = go.Figure(
            go.Bar(
                x=demand_by_ind["Future_Demand_Score"],
                y=demand_by_ind["Industry"],
                orientation="h",
                marker=dict(
                    color=demand_by_ind["Future_Demand_Score"],
                    colorscale="Blues",
                ),
                hovertemplate="<b>%{y}</b><br>Demand: %{x:.1%}<extra></extra>",
            )
        )
        fig_ind_demand.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Future Demand Score by Industry",
            xaxis_title="Avg Future Demand Score",
            xaxis_tickformat=".0%",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_ind_demand, use_container_width=True)

    # -----------------------------------------------------------------------
    # 3. Salary vs Job Growth (grouped, dual y-axis)
    # -----------------------------------------------------------------------
    sal_growth = (
        df_filtered.groupby("Industry", as_index=False)
        .agg(
            salary=("Average_Salary_USD", "mean"),
            growth=("Job_Growth_2030", "mean"),
        )
        .sort_values("salary", ascending=False)
    )

    fig_sg = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sg.add_trace(
        go.Bar(
            x=sal_growth["Industry"],
            y=sal_growth["salary"],
            name="Avg Salary (USD)",
            marker_color="#3498db",
            hovertemplate="<b>%{x}</b><br>Avg Salary: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_sg.add_trace(
        go.Bar(
            x=sal_growth["Industry"],
            y=sal_growth["growth"],
            name="Avg Job Growth 2030 (%)",
            marker_color="#2ecc71",
            hovertemplate="<b>%{x}</b><br>Avg Growth: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_sg.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Salary vs Job Growth by Industry",
        barmode="group",
    )
    fig_sg.update_yaxes(title_text="Salary (USD)", secondary_y=False)
    fig_sg.update_yaxes(title_text="Growth (%)", secondary_y=True)
    st.plotly_chart(fig_sg, use_container_width=True)

    # -----------------------------------------------------------------------
    # 4. Risk heatmap (Industry × Automation Level) + Upskilling gap
    # -----------------------------------------------------------------------
    col_heat2, col_upskill = st.columns(2)

    with col_heat2:
        automation_order = ["Low", "Medium", "High"]
        heat_auto = df_filtered.pivot_table(
            index="Industry",
            columns="Automation_Level",
            values="AI_Replacement_Risk",
            aggfunc="mean",
        )
        heat_auto = heat_auto.reindex(
            columns=[c for c in automation_order if c in heat_auto.columns]
        )
        fig_heat_auto = go.Figure(
            go.Heatmap(
                z=heat_auto.values,
                x=heat_auto.columns,
                y=heat_auto.index,
                colorscale="Reds",
                text=heat_auto.round(2).values,
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b> — %{x}<br>Avg Risk: %{z:.2f}<extra></extra>",
            )
        )
        fig_heat_auto.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Risk by Industry × Automation Level",
        )
        st.plotly_chart(fig_heat_auto, use_container_width=True)

    with col_upskill:
        upskill = (
            df_filtered.assign(
                _needed=(df_filtered["Upskilling_Needed"] == "Yes")
            )
            .groupby("Industry", as_index=False)["_needed"]
            .mean()
            .rename(columns={"_needed": "pct"})
            .sort_values("pct", ascending=False)
        )
        upskill["pct"] = upskill["pct"] * 100
        upskill["_color"] = upskill["pct"].apply(
            lambda v: "#e74c3c" if v > 55 else ("#f39c12" if v >= 45 else "#2ecc71")
        )
        fig_upskill = go.Figure(
            go.Bar(
                x=upskill["pct"],
                y=upskill["Industry"],
                orientation="h",
                marker_color=upskill["_color"],
                hovertemplate="<b>%{y}</b><br>Upskilling Needed: %{x:.1f}%<extra></extra>",
            )
        )
        fig_upskill.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Upskilling Gap by Industry",
            xaxis_title="% Roles Needing Upskilling",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_upskill, use_container_width=True)

    # -----------------------------------------------------------------------
    # 5. Country context (within filtered data)
    # -----------------------------------------------------------------------
    with st.expander("🌍 Country Context (within filtered data)"):
        col_crisk, col_cheat = st.columns(2)

        with col_crisk:
            crisk = (
                df_filtered.groupby("Country", as_index=False)["AI_Replacement_Risk"]
                .mean()
                .sort_values("AI_Replacement_Risk", ascending=False)
            )
            crisk["_color"] = crisk["AI_Replacement_Risk"].apply(
                lambda v: "#e74c3c" if v > 0.53 else ("#f39c12" if v >= 0.49 else "#2ecc71")
            )
            fig_crisk = go.Figure(
                go.Bar(
                    x=crisk["AI_Replacement_Risk"],
                    y=crisk["Country"],
                    orientation="h",
                    marker_color=crisk["_color"],
                    hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1%}<extra></extra>",
                )
            )
            fig_crisk.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="AI Risk by Country",
                xaxis_title="Avg AI Replacement Risk",
                xaxis_tickformat=".0%",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_crisk, use_container_width=True)

        with col_cheat:
            cheat = df_filtered.pivot_table(
                index="Country",
                columns="Industry",
                values="AI_Replacement_Risk",
                aggfunc="mean",
            )
            fig_cheat = go.Figure(
                go.Heatmap(
                    z=cheat.values,
                    x=cheat.columns,
                    y=cheat.index,
                    colorscale="Reds",
                    text=cheat.round(2).values,
                    texttemplate="%{text}",
                    hovertemplate="<b>%{y}</b> — %{x}<br>Avg Risk: %{z:.2f}<extra></extra>",
                )
            )
            fig_cheat.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Risk Heatmap: Country × Industry",
            )
            st.plotly_chart(fig_cheat, use_container_width=True)

        st.caption(
            "🌍 Note: Only 10 countries are represented in this dataset "
            "with similar risk scores (0.49–0.52). Differences are more "
            "meaningful at the Industry level above."
        )
