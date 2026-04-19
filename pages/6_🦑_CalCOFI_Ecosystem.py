"""CalCOFI larvae & zooplankton vs CCE mooring era — context + optional AI narrative."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cce_hack.claude_narrative import calcofi_story_llm
from cce_hack.config import DATA_PROC
from cce_hack.plot_theme import apply_plotly
from cce_hack.streamlit_shell import (
    CHART_H_HALF,
    effective_llm_api_key,
    effective_llm_model,
    effective_llm_provider,
    inject_theme_css,
    page_config,
    render_global_sidebar,
)

LARVAE_PATH = DATA_PROC / "calcofi_larvae_daily.csv"
ZOO_PATH = DATA_PROC / "calcofi_zooplankton_daily.csv"

page_config(title="CalCOFI — CCE")
inject_theme_css()
df_moor = render_global_sidebar()

st.title("CalCOFI + mooring context")
st.markdown(
    "**CalCOFI** = decades of ship-based larvae & zooplankton off the U.S. West Coast. "
    "**CCE moorings** = high-resolution physics & biogeochemistry mostly from ~2009 onward. "
    "They answer different questions; overlap in **time** is limited but **space** is the same ecosystem."
)

if not LARVAE_PATH.exists() and not ZOO_PATH.exists():
    st.warning(f"No CalCOFI CSVs found. Expected `{LARVAE_PATH.name}` and/or `{ZOO_PATH.name}` under `data/processed/`.")
    st.stop()

larv = pd.DataFrame()
zoo = pd.DataFrame()
species: list[str] = []  # filled when larvae CSV is present

# --- Larvae ---
if LARVAE_PATH.exists():
    st.subheader("Fish larvae (CalCOFI)")
    larv = pd.read_csv(LARVAE_PATH)
    larv["date"] = pd.to_datetime(larv["date"], errors="coerce")
    larv = larv.dropna(subset=["date"])
    all_taxa = sorted(larv["scientific_name"].dropna().unique().tolist())
    prefer = [
        "Engraulis mordax",
        "Sardinops sagax",
        "Scomber japonicus",
        "Merluccius productus",
        "Paralichthys californicus",
    ]
    default_taxa = [t for t in prefer if t in all_taxa]
    cap = min(8, max(2, len(all_taxa)))
    for t in all_taxa:
        if len(default_taxa) >= cap:
            break
        if t not in default_taxa:
            default_taxa.append(t)
    st.caption(
        f"**{len(all_taxa)} larval taxa** in `calcofi_larvae_daily.csv` — default selection shows several common fish larvae "
        "(not “only two species”; add/remove in the box below)."
    )
    species = st.multiselect(
        "Larval taxa to plot (scientific name)",
        all_taxa,
        default=default_taxa[: min(8, len(default_taxa))],
        key="calcofi_species_ms",
    )
    if species:
        sub = larv[larv["scientific_name"].isin(species)].copy()
        g = sub.groupby(["date", "scientific_name"], observed=False)["larvae_count"].sum().reset_index()
        fig = go.Figure()
        for sp in species:
            ss = g[g["scientific_name"] == sp]
            if not ss.empty:
                fig.add_trace(go.Scatter(x=ss["date"], y=ss["larvae_count"], mode="lines", name=sp, connectgaps=False))
        fig.update_layout(
            height=CHART_H_HALF,
            title="Daily larvae counts (CalCOFI)",
            xaxis_title="Date",
            yaxis_title="Larvae per day (summed across stations in file)",
        )
        st.plotly_chart(apply_plotly(fig), use_container_width=True)
        st.caption(
            f"Showing **{len(species)}** of **{len(all_taxa)}** taxa — spike trains are survey cruises; flat gaps are no sampling."
        )
    st.caption("Long historical tail predates moorings — use the overlap chart below for joint era.")

# --- Zooplankton ---
if ZOO_PATH.exists():
    st.subheader("Zooplankton (CalCOFI)")
    zoo = pd.read_csv(ZOO_PATH)
    zoo["date"] = pd.to_datetime(zoo["date"], errors="coerce")
    zoo = zoo.dropna(subset=["date"])
    fig2 = go.Figure()
    if "total_plankton" in zoo.columns:
        fig2.add_trace(go.Scatter(x=zoo["date"], y=zoo["total_plankton"], mode="lines", name="total_plankton"))
    if "small_plankton" in zoo.columns:
        fig2.add_trace(go.Scatter(x=zoo["date"], y=zoo["small_plankton"], mode="lines", name="small_plankton"))
    fig2.update_layout(
        height=CHART_H_HALF,
        title="Daily zooplankton indices (aggregated in processed file)",
        yaxis_title="Index (file units)",
    )
    st.plotly_chart(apply_plotly(fig2), use_container_width=True)
    st.caption("These are **summary columns** from the CalCOFI pipeline (not individual species counts).")

# --- Mooring overlap (monthly) ---
st.subheader("Mooring era overlay (monthly means)")
t_m = pd.to_datetime(df_moor["time"], utc=True, errors="coerce")
moor = df_moor.copy()
moor["time"] = t_m
moor = moor.dropna(subset=["time"])
moor["ym"] = moor["time"].dt.to_period("M").dt.to_timestamp()
ph_col = "ph_total" if "ph_total" in moor.columns else ("ph" if "ph" in moor.columns else None)
if ph_col:
    m_ph = moor.groupby("ym", observed=False)[ph_col].mean().reset_index()
    m_ph.columns = ["ym", "moor_ph"]
else:
    m_ph = pd.DataFrame(columns=["ym", "moor_ph"])

if LARVAE_PATH.exists() and species:
    sub = larv[larv["scientific_name"].isin(species)].copy()
    sub["ym"] = sub["date"].dt.to_period("M").dt.to_timestamp()
    m_l = sub.groupby("ym", observed=False)["larvae_count"].mean().reset_index()
    m_l.columns = ["ym", "larvae_mean"]
    combo = m_ph.merge(m_l, on="ym", how="inner")
    if not combo.empty:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(x=combo["ym"], y=combo["larvae_mean"], name="CalCOFI larvae (mean)", marker_color="#2ca02c", yaxis="y")
        )
        fig3.add_trace(
            go.Scatter(
                x=combo["ym"],
                y=combo["moor_ph"],
                name="Mooring pH (mean)",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#1f77b4"),
            )
        )
        fig3.update_layout(
            height=CHART_H_HALF + 40,
            title="Months where **both** mooring pH and CalCOFI larvae exist (inner join)",
            yaxis=dict(title="larvae (mean/day)", side="left"),
            yaxis2=dict(title="pH", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(apply_plotly(fig3), use_container_width=True)
        st.caption(
            f"**{len(combo)}** overlapping months in the current mooring time window — only months present in **both** datasets appear."
        )
    else:
        st.info("No overlapping months between selected larvae and mooring pH in the current time window.")
else:
    st.caption("Select larvae species above to build a mooring overlap chart.")

st.markdown(
    "- **Species Validation** page: iNaturalist **citizen observations** vs mooring anomalies (recent years).\n"
    "- **This page**: **CalCOFI** research surveys vs mooring (long history vs high-res sensors)."
)

# --- AI bridge ---
if effective_llm_api_key():
    if st.button("Ask AI to connect CalCOFI + moorings"):
        moor_t0 = str(moor["time"].min())[:10] if len(moor) else "n/a"
        moor_t1 = str(moor["time"].max())[:10] if len(moor) else "n/a"
        mid = moor["mooring_id"] if "mooring_id" in moor.columns else moor.get("station", pd.Series(dtype=str))
        moor_s = f"Mooring rows={len(moor)}, span {moor_t0}..{moor_t1}, moorings={mid.dropna().astype(str).unique().tolist()[:6]}"
        if LARVAE_PATH.exists() and len(larv):
            larv_s = f"Larvae rows={len(larv)}, span {larv['date'].min()}..{larv['date'].max()}"
        else:
            larv_s = "No larvae file or empty."
        if ZOO_PATH.exists() and len(zoo):
            zoo_s = f"Zoo rows={len(zoo)}, span {zoo['date'].min()}..{zoo['date'].max()}"
        else:
            zoo_s = "No zooplankton file or empty."
        with st.spinner(f"Calling {effective_llm_provider()}…"):
            out = calcofi_story_llm(
                effective_llm_api_key(),
                effective_llm_provider(),
                effective_llm_model(),
                mooring_summary=moor_s,
                larvae_summary=larv_s,
                zoo_summary=zoo_s,
            )
        with st.chat_message("assistant"):
            st.markdown(out)
else:
    st.caption("Add an API key in the sidebar (Gemini recommended) for the AI bridge.")
