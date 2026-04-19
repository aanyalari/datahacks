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
from cce_hack.config import DEFAULT_CLAUDE_MODEL, DEFAULT_MOORING, MOORING_SITES
from cce_hack.data import load_mooring_from_upload, load_mooring_table, pick_default_csv
from cce_hack.ingest_raw import PANEL_FILENAME
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
    padding: 0.95rem 1.1rem;
    border-radius: 10px;
    background: rgba(0, 100, 180, 0.08);
    border-left: 4px solid #0064B4;
    margin-bottom: 0.65rem;
  }}
  .hero h1 {{ margin: 0 0 0.35rem 0; font-size: 1.5rem; letter-spacing: -0.02em; color: #e8f2ff; }}
  .hero p {{ margin: 0; line-height: 1.45; opacity: 0.95; }}
  .status-badge {{
    display: inline-block;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.35rem 0.2rem 0;
    line-height: 1.25;
  }}
  .status-red {{ background: rgba(220, 70, 70, 0.22); color: #ffb4b4; border: 1px solid rgba(255,120,120,0.45); }}
  .status-amber {{ background: rgba(230, 170, 40, 0.18); color: #ffe6a8; border: 1px solid rgba(255,200,100,0.4); }}
  .status-green {{ background: rgba(40, 180, 120, 0.18); color: #b8ffd9; border: 1px solid rgba(100,220,160,0.35); }}
  .alert-strip {{ margin: 0.5rem 0 0.75rem 0; }}
</style>
        """,
        unsafe_allow_html=True,
    )


def page_config(*, title: str = "CCE Mooring Lab") -> None:
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
    Sidebar: data health, time window, CSV upload, mooring, Anthropic, help expander.
    Returns dataframe **after** time-window filter.
    """
    default_path = pick_default_csv()
    if not default_path.exists():
        ensure_sample_csv(default_path)

    if "cce_time_window" not in st.session_state:
        # Default "all": merged mooring panels often have sensors active in different years;
        # a short trailing window can leave every column NaN (empty KPIs / plots).
        st.session_state.cce_time_window = "all"

    st.sidebar.header("Data")
    up = st.sidebar.file_uploader("Upload mooring CSV (optional)", type=["csv"])
    st.sidebar.caption(f"Default file: `{default_path.name}`")
    st.sidebar.caption(f"Raw → processed: `{PANEL_FILENAME}` when present.")

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

    # --- Data health (filtered) ---
    st.sidebar.divider()
    st.sidebar.subheader("Data health")
    n = len(df_win)
    t0 = pd.to_datetime(df_win["time"], utc=True, errors="coerce").min() if n and "time" in df_win.columns else pd.NaT
    t1 = pd.to_datetime(df_win["time"], utc=True, errors="coerce").max() if n and "time" in df_win.columns else pd.NaT
    pct6 = pct_rows_all_six_core(df_win)
    st.sidebar.metric("Rows (window)", f"{n:,}")
    st.sidebar.metric(
        "UTC span",
        f"{str(t0)[:10]} → {str(t1)[:10]}" if pd.notna(t0) and pd.notna(t1) else "—",
    )
    st.sidebar.metric(
        "% rows with all 6 core fields",
        "—" if pct6 != pct6 else f"{pct6:.1f}%",
        help="pH, SST, salinity, O₂ (if present), nitrate, chlorophyll (if present) all non-null",
    )

    with st.sidebar.expander("How to use this app", expanded=False):
        st.markdown(
            """
**Offline (no API key)**

1. Run `streamlit run streamlit_app.py`.
2. Leave CSV upload empty to use the built-in / synthetic file.
3. **Mission Control** (Home) + **Analytics**, **Data Quality**, **Lab**, and sklearn on **AI Predictions** need **no network**.

**Optional Claude**

1. Paste `ANTHROPIC_API_KEY` below (or set env var before launch).
2. Open **AI Predictions** — Claude runs **only** when you click a Claude button.
            """
        )

    st.sidebar.divider()
    st.sidebar.header("Claude (optional)")
    st.sidebar.text_input(
        "ANTHROPIC_API_KEY",
        type="password",
        key="anthropic_api_key",
        help="Optional — only for AI Predictions buttons.",
    )
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key and not st.session_state.get("anthropic_api_key"):
        st.session_state["anthropic_api_key"] = env_key
    st.sidebar.text_input(
        "Model id",
        value=os.environ.get("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL),
        key="anthropic_model",
    )

    return df_win


def numeric_series_cols(df: pd.DataFrame) -> list[str]:
    skip = {"mooring_id", "latitude", "longitude"}
    return [
        c
        for c in df.columns
        if c not in skip and c != "time" and np.issubdtype(df[c].dtype, np.number)
    ]


def effective_anthropic_key() -> str:
    return (st.session_state.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()


def effective_anthropic_model() -> str:
    return (st.session_state.get("anthropic_model") or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL).strip()


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
