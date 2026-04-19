"""Analytics — narrative tabs + global date range."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cce_hack.acidification_co2sys import run_co2sys_on_dataframe
from cce_hack.column_pick import friendly_axis_label, pick_best_column
from cce_hack.dynamic_insights import insight_analytics_physical, insight_ph_aragonite, insight_ts_pair
from cce_hack.key_findings import key_findings_analytics
from cce_hack.plot_theme import apply_plotly
from cce_hack.streamlit_shell import CHART_H_FULL, CHART_H_HALF, filter_date_range, inject_theme_css, page_config, render_global_sidebar
from cce_hack.viz_extras import hovmoller_sst_depth_time, normalize_rows_01, seasonal_radar_frame

page_config(title="Analytics — CCE")
inject_theme_css()
df0 = render_global_sidebar()

st.title("Analytics")

if "time" not in df0.columns:
    st.error("Need a time column.")
    st.stop()

tser = pd.to_datetime(df0["time"], utc=True, errors="coerce")
tmin, tmax = tser.min(), tser.max()
c1, c2 = st.columns(2)
with c1:
    d_start = st.date_input("Plot from (UTC date)", value=pd.Timestamp(tmin).date(), key="an_start")
with c2:
    d_end = st.date_input("Plot through (UTC date)", value=pd.Timestamp(tmax).date(), key="an_end")
df = filter_date_range(df0, d_start, d_end)
if df.empty:
    st.warning("No rows in that date range — widen the window.")
    st.stop()

st.caption(
    f"**{len(df):,}** rows in this date filter (same mooring table as **Home** / **Analysis Lab** — Lab adds optional stats & toy ML)."
)

tab_phys, tab_bio, tab_acid, tab_ext = st.tabs(["Physical", "Biological", "Acidification", "Extremes"])

with tab_phys:
    sst_c = pick_best_column(df, "sst")
    sal_c = pick_best_column(df, "salinity")
    if sst_c and sal_c:
        cols_ts = ["time", sst_c, sal_c]
        chlc = pick_best_column(df, "chl")
        no3c = pick_best_column(df, "no3")
        if chlc:
            cols_ts.append(chlc)
        elif no3c:
            cols_ts.append(no3c)
        ts = df[cols_ts].dropna(subset=[sst_c, sal_c]).copy()
        ts["time"] = pd.to_datetime(ts["time"], utc=True, errors="coerce")
        color = chlc if chlc and chlc in ts.columns else (no3c if no3c and no3c in ts.columns else None)
        fig = px.scatter(
            ts,
            x=sal_c,
            y=sst_c,
            color=color,
            hover_data=["time"] if "time" in ts.columns else None,
            title="T–S diagram",
            labels={sal_c: friendly_axis_label(sal_c), sst_c: friendly_axis_label(sst_c)},
        )
        st.plotly_chart(apply_plotly(fig).update_layout(height=CHART_H_FULL), use_container_width=True)
        st.caption(insight_analytics_physical(df, sst_c, sal_c))
    else:
        st.warning("Could not find a usable temperature + salinity column pair in this file.")
    c1, c2 = st.columns(2)
    with c1:
        if sst_c:
            d = df[["time", sst_c]].copy()
            d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
            d[sst_c] = pd.to_numeric(d[sst_c], errors="coerce")
            d = d.dropna(subset=["time", sst_c])
            fig2 = px.line(d, x="time", y=sst_c, title=friendly_axis_label(sst_c), labels={sst_c: friendly_axis_label(sst_c)})
            st.plotly_chart(apply_plotly(fig2).update_layout(height=CHART_H_HALF, xaxis_title="Time (UTC)"), use_container_width=True)
    with c2:
        if sal_c:
            d = df[["time", sal_c]].copy()
            d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
            d[sal_c] = pd.to_numeric(d[sal_c], errors="coerce")
            d = d.dropna(subset=["time", sal_c])
            fig3 = px.line(d, x="time", y=sal_c, title=friendly_axis_label(sal_c), labels={sal_c: friendly_axis_label(sal_c)})
            st.plotly_chart(apply_plotly(fig3).update_layout(height=CHART_H_HALF, xaxis_title="Time (UTC)"), use_container_width=True)

with tab_bio:
    chlb = pick_best_column(df, "chl")
    no3b = pick_best_column(df, "no3")
    cols_bio = [c for c in (chlb, no3b) if c]
    if cols_bio:
        d = df[["time"] + cols_bio].copy()
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        for c in cols_bio:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["time"], how="all")
        dm = d.melt(id_vars="time", var_name="variable", value_name="value").dropna(subset=["value"])
        dm["variable"] = dm["variable"].map(lambda c: friendly_axis_label(str(c)))
        fig = px.line(dm, x="time", y="value", color="variable", title="Chlorophyll & nitrate", labels={"value": "Concentration"})
        st.plotly_chart(apply_plotly(fig).update_layout(height=CHART_H_FULL, xaxis_title="Time (UTC)"), use_container_width=True)
        st.caption(insight_ts_pair(df, chlb, no3b))
    else:
        st.warning("No chlorophyll or nitrate column detected in this file.")

with tab_acid:
    ph_a = pick_best_column(df, "ph")
    t_a = pick_best_column(df, "sst")
    s_a = pick_best_column(df, "salinity")
    if ph_a:
        d = df[["time", ph_a]].copy()
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        d[ph_a] = pd.to_numeric(d[ph_a], errors="coerce")
        d = d.dropna(subset=["time", ph_a])
        fig = px.line(
            d,
            x="time",
            y=ph_a,
            title=friendly_axis_label(ph_a),
            labels={ph_a: friendly_axis_label(ph_a)},
        )
        st.plotly_chart(apply_plotly(fig).update_layout(height=CHART_H_HALF, xaxis_title="Time (UTC)"), use_container_width=True)
    if ph_a and t_a and s_a:
        co2 = run_co2sys_on_dataframe(df, salinity_col=s_a, temp_col=t_a, ph_col=ph_a)
    else:
        co2 = run_co2sys_on_dataframe(df)
    if co2 is not None:
        fig2 = px.line(co2, x="time", y="saturation_aragonite", title="Ω_aragonite (pH + T + S, assumed TA)")
        st.plotly_chart(apply_plotly(fig2).update_layout(height=CHART_H_HALF, xaxis_title="Time (UTC)"), use_container_width=True)
        st.caption(insight_ph_aragonite(co2))
    else:
        st.caption("PyCO2SYS needs overlapping pH, temperature, and salinity on the same timestamps (any depth column we can resolve).")

with tab_ext:
    hov = hovmoller_sst_depth_time(df)
    if hov is not None:
        fig = px.imshow(
            hov.T,
            aspect="auto",
            labels=dict(x="time", y="depth / instrument", color="SST °C"),
            title="Hovmöller — SST vs time × depth",
        )
        st.plotly_chart(apply_plotly(fig).update_layout(height=CHART_H_FULL), use_container_width=True)
        st.caption(
            "Each row is a depth/instrument slice; color is SST — **vertical bands** are coherent warming/cooling across depths; "
            "**slanted** features often mean advection or phase offsets between instruments."
        )
    else:
        st.caption("Hovmöller needs multi-depth SST columns (`sst_c_d32m`, …).")
    num = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("mooring_id", "latitude", "longitude")
    ][:8]
    radf = seasonal_radar_frame(df, [c for c in num if c in df.columns][:6])
    if radf is not None:
        radn = normalize_rows_01(radf, [c for c in radf.columns if c != "season"])
        feats = [c for c in radn.columns if c != "season"]
        theta_labels = [friendly_axis_label(c) for c in feats]
        fig = go.Figure()
        for _, row in radn.iterrows():
            vals = [float(row[c]) for c in feats]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=theta_labels + [theta_labels[0]],
                    fill="toself",
                    name=str(row["season"]),
                )
            )
        st.plotly_chart(
            apply_plotly(fig).update_layout(title="Seasonal radar (min–max normalized)", height=CHART_H_FULL),
            use_container_width=True,
        )
        st.caption(
            "Each season is one closed loop — **distance from center** is how “high” that variable sits vs its own seasonal range "
            f"({', '.join(theta_labels[:4])}{'…' if len(theta_labels) > 4 else ''}). Compare loop shapes, not absolute chemistry."
        )

with st.expander("Key findings from this view", expanded=False):
    for line in key_findings_analytics(df):
        st.markdown(f"- {line}")
