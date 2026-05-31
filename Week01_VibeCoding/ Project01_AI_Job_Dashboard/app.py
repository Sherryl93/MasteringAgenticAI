"""AI Job Survival Dashboard 2030 — main entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

from utils.config import configure_page
from utils.data_loader import render_sidebar
from components import survival_matrix_tab, industry_tab, skills_tab

# Page config + theme (must run before any other Streamlit command).
configure_page()

st.title("AI Job Survival Dashboard 2030")

# Sidebar (data source + filters) returns the filtered dataframe.
df_filtered = render_sidebar()

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["🎯 Survival Matrix", "🏭 Industry Deep-Dive", "🧠 Skills & Future-Proofing"]
)

with tab1:
    survival_matrix_tab.render(df_filtered)

with tab2:
    industry_tab.render(df_filtered)

with tab3:
    skills_tab.render(df_filtered)
