"""
CCE Mooring Lab — **Mission Control** is the Home entry (judge landing).

  pip install -e .
  streamlit run streamlit_app.py

Other screens live under ``pages/``. Sidebar: data health, time window, optional Claude.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from cce_hack.key_findings import key_findings_mission
from cce_hack.mission_alerts import aragonite_habitat_sentence
from cce_hack.mission_ui import (
    mission_alert_badges_html,
    normalized_six_series_figure,
    render_mooring_map_pydeck,
    render_six_core_metrics,
)
from cce_hack.streamlit_shell import inject_theme_css, page_config, render_global_sidebar

page_config(title="Mission Control — CCE Mooring Lab")
inject_theme_css()

df = render_global_sidebar()

st.title("Mooring health — Mission Control")
st.markdown(
    """
<div class="hero">
  <h1>At-a-glance mooring QA (California Current)</h1>
  <p>
    <strong>For judges:</strong> fixed CCE map pins, six headline sensors (auto-picked from your CSV), deltas vs <strong>30-day trailing means</strong>,
    deterministic alert pills, and one normalized multi-sensor timeline. Sidebar <strong>time window</strong> applies everywhere — start with <strong>All data</strong> if a merged file has sensors in different years.
  </p>
</div>
    """,
    unsafe_allow_html=True,
)

render_mooring_map_pydeck()
st.subheader("Headline sensors — latest vs 30-day mean")
render_six_core_metrics(df)

st.markdown(mission_alert_badges_html(df), unsafe_allow_html=True)

st.subheader("Carbonate / habitat readout")
st.markdown(aragonite_habitat_sentence(df))

st.subheader("Six variables on one axis (min–max normalized)")
fig = normalized_six_series_figure(df, max_days=90)
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need overlapping time series for several of: pH, SST, salinity, O₂, nitrate, chlorophyll.")

with st.expander("Key findings from this view", expanded=False):
    for line in key_findings_mission(df):
        st.markdown(f"- {line}")
