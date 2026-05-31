# AI Job Survival Dashboard 2030

An interactive Streamlit dashboard that analyzes how AI is projected to impact
the global job market by 2030 — AI replacement risk, future demand, salaries,
job growth, and the skills that keep roles future-proof.

## Features

The app is organized into three tabs, all driven by the sidebar filters
(Industry, Country, Automation Level) and an optional CSV upload.

- **🎯 Survival Matrix** — headline KPIs, an AI Survival Matrix scatter
  (risk vs. demand, sized by growth, colored by salary), projected job growth
  by role, and Top-5 role tables split by 2026 hiring trend.
- **🏭 Industry Deep-Dive** — industry KPIs, risk and future-demand rankings,
  a dual-axis Salary-vs-Growth chart, a Risk × Automation-Level heatmap, an
  upskilling-gap chart, and an expandable country-context section.
- **🧠 Skills & Future-Proofing** — most in-demand skills, which skills lower
  AI risk, upskilling gaps by role, and a role explorer with per-role skills,
  hiring-trend breakdown, and a generated summary.

## Project structure

```
My_Assignment/
├── app.py                      # Entry point: page config, sidebar, tab routing
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
├── AI_Impact_on_Jobs_2030.csv  # Default dataset
├── utils/
│   ├── __init__.py
│   ├── config.py               # Page config, dark theme, PLOTLY_TEMPLATE
│   └── data_loader.py          # load_data() + sidebar (uploader & filters)
└── components/
    ├── __init__.py
    ├── survival_matrix_tab.py  # Tab 1
    ├── industry_tab.py         # Tab 2
    └── skills_tab.py           # Tab 3
```

## Data

The default dataset is **AI Impact on Jobs 2030** (3,000 records), sourced from
[Kaggle](https://www.kaggle.com/datasets/muhammadwaqas023/ai-impact-in-future-on-jobs-market-in-2030).

You can also upload your own CSV from the sidebar. It must use the same column
structure as the default dataset, including:
`Job_Title`, `Industry`, `Country`, `Automation_Level`, `AI_Replacement_Risk`,
`Future_Demand_Score`, `Average_Salary_USD`, `Job_Growth_2030`,
`Required_Skills`, `Upskilling_Needed`, `Hiring_Trend_2026`.

## Setup & run

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

## Notes

- All charts use a shared dark Plotly template (`PLOTLY_TEMPLATE` in
  `utils/config.py`) and transparent backgrounds.
- `load_data()` is cached with `@st.cache_data` so the default CSV is read once.
- Color thresholds (green / amber / red) are used consistently across charts to
  flag low / moderate / high AI replacement risk and upskilling need.
