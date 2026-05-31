"""Data loading and the sidebar (data source + global filters)."""
import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path: str = "AI_Impact_on_Jobs_2030.csv") -> pd.DataFrame:
    """Load the default dataset from disk (cached)."""
    df = pd.read_csv(path)
    return df


def render_sidebar() -> pd.DataFrame:
    """Render the data uploader and global filters in the sidebar.

    Returns the filtered dataframe used by every tab.
    """
    # --- Data source -------------------------------------------------------
    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload your CSV data",
        type=["csv"],
        help="Upload a CSV file with the same column structure "
             "as the default dataset"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Custom data loaded!")
    else:
        df = load_data()

    st.sidebar.markdown(
        "Default dataset: [AI Impact on Jobs 2030]"
        "(https://www.kaggle.com/datasets/muhammadwaqas023/ai-impact-in-future-on-jobs-market-in-2030) "
        "(3,000 records). Upload your own CSV with matching columns "
        "to analyze custom data."
    )

    st.sidebar.divider()

    # --- Filters -----------------------------------------------------------
    st.sidebar.header("🔎 Filters")

    industries = sorted(df["Industry"].dropna().unique())
    countries = sorted(df["Country"].dropna().unique())

    selected_industries = st.sidebar.multiselect(
        "Industry", options=industries, default=industries
    )
    selected_countries = st.sidebar.multiselect(
        "Country", options=countries, default=countries
    )
    selected_automation = st.sidebar.multiselect(
        "Automation Level",
        options=sorted(df["Automation_Level"].dropna().unique()),
        default=sorted(df["Automation_Level"].dropna().unique())
    )

    # --- Apply filters -----------------------------------------------------
    df_filtered = df[
        df["Industry"].isin(selected_industries)
        & df["Country"].isin(selected_countries)
        & df["Automation_Level"].isin(selected_automation)
    ]

    st.sidebar.markdown(f"**{len(df_filtered):,}** of **{len(df):,}** records shown")

    st.sidebar.markdown(
        "---\n"
        "💡 **How to use:**\n"
        "- Filter by Industry or Country to compare subsets\n"
        "- Filter by Automation Level to see high-risk segments\n"
        "- Upload your own CSV to analyze custom workforce data"
    )

    return df_filtered
