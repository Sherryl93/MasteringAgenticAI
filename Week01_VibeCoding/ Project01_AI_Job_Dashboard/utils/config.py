"""Page configuration and shared visual constants."""
import streamlit as st

# Plotly template used by every chart in the app.
PLOTLY_TEMPLATE = "plotly_dark"


def configure_page():
    """Set the Streamlit page config and apply the dark theme.

    Must be called once, before any other Streamlit command runs.
    """
    st.set_page_config(
        page_title="AI Job Survival Dashboard 2030",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark theme styling
    st.markdown(
        """
        <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            section[data-testid="stSidebar"] { background-color: #161a23; }
        </style>
        """,
        unsafe_allow_html=True,
    )
