"""Data quality — monthly coverage heatmap + detail table + markdown download."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import streamlit as st

from cce_hack.agent_tools import build_context_package, context_json_for_prompt
from cce_hack.data_quality_viz import build_data_quality_markdown_report, coverage_heatmap_figure, monthly_coverage_matrix
from cce_hack.streamlit_shell import friendly_column_label_plain, inject_theme_css, numeric_series_cols, page_config, render_global_sidebar

page_config(title="Data Quality — CCE")
inject_theme_css()
df = render_global_sidebar()

st.title("Data quality")

pkg = build_context_package(df)
if "error" in pkg:
    st.error(pkg["error"])
    st.stop()

num_cols = numeric_series_cols(df)[:24]
mat, _months = monthly_coverage_matrix(df, num_cols)
if mat is not None:
    st.subheader("Coverage heatmap (% filled per month)")
    st.plotly_chart(coverage_heatmap_figure(mat), use_container_width=True)
else:
    st.info("Not enough numeric columns / time for a monthly heatmap.")

st.subheader("Detail table")
cov_rows = []
for col, info in pkg["coverage_by_column"].items():
    cov_rows.append(
        {
            "What it measures": friendly_column_label_plain(col),
            "Sensor ID (CSV)": col,
            "How much is filled": float(info["filled_percent"]),
            "First date with values": info["first_date"],
            "Last date with values": info["last_date"],
        }
    )
st.dataframe(
    pd.DataFrame(cov_rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "How much is filled": st.column_config.ProgressColumn("Filled %", min_value=0, max_value=100, format="%.0f%%"),
        "What it measures": st.column_config.TextColumn("What you're looking at", width="large"),
        "Sensor ID (CSV)": st.column_config.TextColumn("Sensor ID (CSV)", width="medium"),
    },
)

md_report = build_data_quality_markdown_report(df)
st.download_button(
    "Download QA summary (.md)",
    data=md_report.encode("utf-8"),
    file_name="data_quality_summary.md",
    mime="text/markdown",
)

with st.expander("Machine JSON (optional LLM context)", expanded=False):
    st.code(context_json_for_prompt(pkg, indent=2), language="json")
