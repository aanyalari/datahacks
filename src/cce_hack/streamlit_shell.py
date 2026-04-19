"""Shared Streamlit setup: theme, sidebar, time window, data health (multipage + Home)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from cce_hack.column_pick import pick_best_column
from cce_hack.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MOORING,
    MOORING_SITES,
)
from cce_hack.data import load_mooring_from_upload, load_mooring_table, pick_default_csv
from cce_hack.sample_data import ensure_sample_csv

# --- Layout constants (import on pages for consistent chart heights) ---
CHART_H_FULL = 380
CHART_H_HALF = 320

TIME_WINDOW_LABELS = ["Last 30 days", "Last 90 days", "Last year", "All data"]
TIME_WINDOW_KEYS = ["30d", "90d", "365d", "all"]


def inject_theme_css() -> None:
    st.markdown(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700&display=swap');
  html, body, [class*="css"]  {{ font-family: "DM Sans", "Segoe UI", system-ui, sans-serif; }}
  .block-container {{ padding-top: 0.85rem; max-width: 1320px; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.35rem; font-weight: 600; }}
  div[data-testid="stMetricDelta"] {{ font-size: 0.75rem; }}
  .hero {{
    padding: 1rem 1.15rem;
    border-radius: 12px;
    background: linear-gradient(120deg, rgba(0, 100, 180, 0.12) 0%, rgba(0, 100, 180, 0.04) 100%);
    border: 1px solid rgba(0, 100, 180, 0.2);
    border-left: 4px solid #0064B4;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }}
  .hero h1 {{
    margin: 0 0 0.4rem 0;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #023047;
    line-height: 1.2;
  }}
  .hero p {{
    margin: 0;
    line-height: 1.5;
    color: #1b2838;
    font-size: 0.98rem;
  }}
  .status-badge {{
    display: inline-block;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.35rem 0.2rem 0;
    line-height: 1.25;
  }}
  .status-red {{ background: rgba(220, 70, 70, 0.14); color: #7f1d1d; border: 1px solid rgba(180,60,60,0.35); }}
  .status-amber {{ background: rgba(230, 170, 40, 0.16); color: #713f12; border: 1px solid rgba(200,150,40,0.35); }}
  .status-green {{ background: rgba(40, 180, 120, 0.14); color: #14532d; border: 1px solid rgba(60,160,100,0.35); }}
  .alert-strip {{ margin: 0.5rem 0 0.75rem 0; }}
</style>
        """,
        unsafe_allow_html=True,
    )


def page_config(*, title: str = "MooringMind") -> None:
    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _load_disk_csv(path: str) -> pd.DataFrame:
    return load_mooring_table(path)


def mooring_site_map_df() -> pd.DataFrame:
    rows = []
    for name, geo in MOORING_SITES.items():
        rows.append({"mooring": name, "lat": geo["latitude"], "lon": geo["longitude"]})
    return pd.DataFrame(rows)


def six_core_columns(df: pd.DataFrame) -> list[str]:
    """Best-matching columns for the six dashboard roles (may be depth vs surface)."""
    out: list[str] = []
    for role in ("ph", "sst", "salinity", "o2", "no3", "chl"):
        c = pick_best_column(df, role)
        if c:
            out.append(c)
    return out


def pct_rows_all_six_core(df: pd.DataFrame) -> float:
    cols = six_core_columns(df)
    want = 6
    if len(cols) < want or "time" not in df.columns:
        return float("nan")
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    ok = sub.notna().all(axis=1)
    return float(100.0 * ok.mean())


def apply_time_window(df: pd.DataFrame) -> pd.DataFrame:
    """Filter by sidebar ``cce_time_window`` session key."""
    if df is None or df.empty or "time" not in df.columns:
        return df
    key = st.session_state.get("cce_time_window", "90d")
    if key == "all":
        return df.copy()
    days = {"30d": 30, "90d": 90, "365d": 365}.get(key, 90)
    tt = pd.to_datetime(df["time"], utc=True, errors="coerce")
    tmax = tt.max()
    if pd.isna(tmax):
        return df.copy()
    cut = tmax - pd.Timedelta(days=days)
    return df.loc[tt >= cut].copy()


def render_global_sidebar() -> pd.DataFrame:
    """
    Sidebar: data health, time window, CSV upload, mooring, optional Gemini/Groq keys, help expander.
    Returns dataframe **after** time-window filter.
    """
    default_path = pick_default_csv()
    if not default_path.exists():
        ensure_sample_csv(default_path)

    if "cce_time_window" not in st.session_state:
        # Default "all": merged mooring panels often have sensors active in different years;
        # a short trailing window can leave every column NaN (empty KPIs / plots).
        st.session_state.cce_time_window = "all"
    if "cce_chart_theme" not in st.session_state:
        st.session_state.cce_chart_theme = "light"

    st.sidebar.markdown("## MooringMind")
    st.sidebar.caption("Mission Control • Analytics • Quality • Predictions")
    st.sidebar.divider()

    st.sidebar.header("Data")
    up = st.sidebar.file_uploader(
        "Replace mooring CSV (optional)",
        type=["csv"],
        help="Loads your file for this browser session only. Leave empty to use the default path on disk (all pages read the same table).",
    )
    _theme_label = st.sidebar.radio(
        "Chart colors",
        ("Light (recommended for demos)", "Dark"),
        index=0 if st.session_state.get("cce_chart_theme", "light") == "light" else 1,
        horizontal=True,
        help="Plotly template for every chart. Light mode improves contrast on printed slides and Streamlit light UI.",
    )
    st.session_state.cce_chart_theme = "light" if _theme_label.startswith("Light") else "dark"

    if up is not None:
        try:
            df = load_mooring_from_upload(up)
        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")
            st.stop()
    else:
        df = _load_disk_csv(str(default_path))

    if "mooring_id" in df.columns:
        moorings = sorted(df["mooring_id"].dropna().unique().tolist())
        idx = moorings.index(DEFAULT_MOORING) if DEFAULT_MOORING in moorings else 0
        pick = st.sidebar.selectbox("Mooring slice", moorings, index=idx)
        df = df[df["mooring_id"] == pick].copy()

    # --- Time window (global) ---
    st.sidebar.divider()
    st.sidebar.subheader("Time window")
    cur = st.session_state.cce_time_window
    try:
        tw_index = TIME_WINDOW_KEYS.index(cur)
    except ValueError:
        tw_index = 3  # All data
    tw_label = st.sidebar.selectbox(
        "Apply to all pages",
        TIME_WINDOW_LABELS,
        index=tw_index,
        label_visibility="visible",
    )
    st.session_state.cce_time_window = TIME_WINDOW_KEYS[TIME_WINDOW_LABELS.index(tw_label)]

    st.sidebar.caption(
        "If KPIs or charts look empty on a merged file, choose **All data** — depth vs surface sensors may not overlap in the last 90 days."
    )

    df_win = apply_time_window(df)

    return df_win


def numeric_series_cols(df: pd.DataFrame) -> list[str]:
    skip = {"mooring_id", "latitude", "longitude"}
    return [
        c
        for c in df.columns
        if c not in skip and c != "time" and pd.api.types.is_numeric_dtype(df[c])
    ]


def effective_llm_provider() -> str:
    v = (st.session_state.get("llm_provider") or os.environ.get("CCE_LLM_PROVIDER") or "gemini").strip().lower()
    return v if v in ("gemini", "groq") else "gemini"


def effective_llm_api_key() -> str:
    p = effective_llm_provider()
    if p == "gemini":
        return (
            st.session_state.get("gemini_api_key")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip()
    return (st.session_state.get("groq_api_key") or os.environ.get("GROQ_API_KEY") or "").strip()


def effective_llm_model() -> str:
    p = effective_llm_provider()
    if p == "gemini":
        return (st.session_state.get("gemini_model") or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    return (st.session_state.get("groq_model") or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()


_SENSOR_LABELS: dict[str, str] = {
    "sst_c": "Water temperature (°C)",
    "salinity_psu": "Salinity",
    "conductivity_s_m": "Conductivity",
    "ph_total": "pH",
    "no3": "Nitrate",
    "chl_mg_m3": "Chlorophyll",
    "chl_mg_m3_d40m": "Chlorophyll (~40 m)",
    "pco2_uatm": "pCO₂",
    "wind_speed_ms": "Wind speed",
    "air_temp_c": "Air temperature",
}


def friendly_column_label_plain(col: str) -> str:
    m = re.match(r"^(?P<base>.+)_d(?P<dep>\d+)m$", col)
    base = m.group("base") if m else col
    dep = int(m.group("dep")) if m else None
    core = _SENSOR_LABELS.get(base) or _SENSOR_LABELS.get(col) or col.replace("_", " ")
    if dep is not None:
        return f"{core} — near {dep} m depth"
    return core


def filter_date_range(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Further restrict ``df`` to ``[start, end]`` calendar days on ``time`` (UTC)."""
    if df is None or df.empty or "time" not in df.columns:
        return df
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    m = pd.Series(True, index=df.index)
    if start is not None:
        st0 = pd.Timestamp(start)
        if st0.tzinfo is None:
            st0 = st0.tz_localize("UTC")
        m &= t >= st0
    if end is not None:
        et = pd.Timestamp(end)
        if et.tzinfo is None:
            et = et.tz_localize("UTC")
        m &= t <= et + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return df.loc[m].copy()
