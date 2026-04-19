"""
MooringMind — Mission Control (main entry).

  pip install -e .
  streamlit run Home.py

Other screens live under ``pages/``. Sidebar: chart theme, data health, time window, optional CSV upload, Gemini/Groq keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from cce_hack.dynamic_insights import (
    insight_headline_metrics,
    insight_mooring_window,
    insight_multisensor_normalized,
)
from cce_hack.key_findings import key_findings_mission
from cce_hack.mission_alerts import aragonite_habitat_sentence
from cce_hack.mission_ui import (
    mission_alert_badges_html,
    normalized_six_series_figure,
    render_mooring_map_pydeck,
    render_six_core_metrics,
)
from cce_hack.streamlit_shell import inject_theme_css, page_config, render_global_sidebar

page_config(title="MooringMind")
inject_theme_css()

df = render_global_sidebar()

st.title("MooringMind — Mission Control")

render_mooring_map_pydeck()
st.caption(insight_mooring_window(df))

st.subheader("Headline sensors — latest vs 30-day mean")
render_six_core_metrics(df)
st.caption(insight_headline_metrics(df))

st.markdown(mission_alert_badges_html(df), unsafe_allow_html=True)

st.subheader("Carbonate / habitat readout")
st.markdown(aragonite_habitat_sentence(df))

st.subheader("Multi-sensor timing (each series scaled 0–1 for the last 90 days)")
st.caption(
    "Each line is a different measurement, min–max normalized **within this window only** so you can see whether "
    "stressors **move together** (shared forcing) or **split** (different processes or sensor dropouts). "
    "Use **Analytics** for real units."
)
fig = normalized_six_series_figure(df, max_days=90)
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
    st.caption(insight_multisensor_normalized(df, max_days=90))
else:
    st.info("Need overlapping time series for several of: pH, SST, salinity, O₂, nitrate, chlorophyll.")

with st.expander("Auto-detected facts (from this file)", expanded=False):
    for line in key_findings_mission(df):
        st.markdown(f"- {line}")
