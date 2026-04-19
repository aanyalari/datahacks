"""Mission Control visuals: pydeck map, core metrics row, alert pills, normalized multi-series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cce_hack.column_pick import friendly_axis_label, pick_best_column
from cce_hack.mission_alerts import pick_chl_column, pick_o2_column
from cce_hack.plot_theme import apply_plotly
from cce_hack.streamlit_shell import mooring_site_map_df

CHART_H_FULL = 380


def render_mooring_map_pydeck() -> None:
    """CCE1 / CCE2 reference pins (hardcoded via ``mooring_site_map_df``)."""
    try:
        import pydeck as pdk
    except ImportError:
        mdf = mooring_site_map_df()
        st.map(mdf, latitude="lat", longitude="lon", use_container_width=True)
        st.caption("Install `pydeck` for enhanced pins: `pip install pydeck`")
        return

    mdf = mooring_site_map_df()
    mdf["name"] = mdf["mooring"]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=mdf,
        get_position=["lon", "lat"],
        get_radius=60000,
        get_fill_color=[0, 120, 200, 230],
        pickable=True,
    )
    view = pdk.ViewState(latitude=float(mdf["lat"].mean()), longitude=float(mdf["lon"].mean()), zoom=5.8, pitch=0)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "{name}"},
    )
    try:
        st.pydeck_chart(deck, use_container_width=True)
    except Exception:
        st.map(mdf, latitude="lat", longitude="lon", use_container_width=True)
        st.caption("PyDeck map style unavailable in this environment — fell back to `st.map`.")


def _latest_and_30d_mean(df: pd.DataFrame, col: str | None) -> tuple[float | None, float | None]:
    if not col or col not in df.columns or "time" not in df.columns:
        return None, None
    d = df[["time", col]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["time", col]).sort_values("time")
    if d.empty:
        return None, None
    latest = float(d[col].iloc[-1])
    t1 = d["time"].iloc[-1]
    w = d[d["time"] >= t1 - pd.Timedelta(days=30)]
    if len(w) < 3:
        return latest, float(d[col].mean())
    mu = float(w[col].mean())
    return latest, mu


def _metric_severity(kind: str, latest: float | None, mu: float | None) -> str:
    """Return 'green' | 'yellow' | 'red' | 'na' for dashboard coloring."""
    if latest is None or np.isnan(latest):
        return "na"
    if kind == "ph":
        if latest < 7.75:
            return "red"
        if latest < 7.95:
            return "yellow"
        return "green"
    if kind == "o2":
        if latest < 2.0:
            return "red"
        if latest < 4.0:
            return "yellow"
        return "green"
    if kind == "chl":
        if latest > 10.0:
            return "red"
        if latest > 5.0:
            return "yellow"
        return "green"
    if kind == "no3":
        if latest > 30.0:
            return "red"
        if latest > 20.0:
            return "yellow"
        return "green"
    if kind == "temp":
        if latest > 22.0:
            return "red"
        if latest > 18.0:
            return "yellow"
        return "green"
    if kind == "sal":
        if latest < 32.0 or latest > 35.5:
            return "yellow"
        return "green"
    return "green"


def _delta_color(sev: str) -> str:
    if sev == "red":
        return "inverse"
    if sev == "yellow":
        return "off"
    return "normal"


def render_six_core_metrics(df: pd.DataFrame) -> None:
    """Six metrics in one row: pH, SST, salinity, O₂, nitrate, chlorophyll — delta vs 30-day mean."""
    ph_c = pick_best_column(df, "ph")
    sst_c = pick_best_column(df, "sst")
    sal_c = pick_best_column(df, "salinity")
    o2c = pick_o2_column(df)
    no3_c = pick_best_column(df, "no3")
    chlc = pick_chl_column(df)
    specs: list[tuple[str, str | None, str]] = [
        ("pH", ph_c, "ph"),
        ("Temperature", sst_c, "temp"),
        ("Salinity", sal_c, "sal"),
        ("Dissolved O₂", o2c, "o2"),
        ("Nitrate", no3_c, "no3"),
        ("Chlorophyll", chlc, "chl"),
    ]
    cols = st.columns(6)
    for i, (label, col, kind) in enumerate(specs):
        with cols[i]:
            lat, mu = _latest_and_30d_mean(df, col)
            if col is None or col not in df.columns or lat is None:
                st.metric(label, "—", help="No numeric values in the current time window for this sensor family.")
                continue
            nice = friendly_axis_label(col)
            sev = _metric_severity(kind, lat, mu)
            delta = None if mu is None or np.isnan(mu) else round(lat - mu, 4)
            st.metric(
                label,
                f"{lat:.3g}" if kind != "ph" else f"{lat:.3f}",
                delta=delta,
                delta_color=_delta_color(sev),
                help=f"{nice} · column `{col}` · 30-day mean ≈ {mu:.3g}" if mu is not None else f"{nice} · `{col}`",
            )


def mission_alert_badges_html(df: pd.DataFrame) -> str:
    """Deterministic pill-style alerts (HTML)."""
    pills: list[str] = []
    ph_c = pick_best_column(df, "ph")
    ph, _ = _latest_and_30d_mean(df, ph_c)
    if ph is not None and ph < 7.75:
        pills.append(
            '<span class="status-badge status-red">⚠️ Ocean acidification stress — aragonite saturation likely below 1.0</span>'
        )
    o2c = pick_o2_column(df)
    o2, _ = _latest_and_30d_mean(df, o2c) if o2c else (None, None)
    if o2 is not None and o2 < 2.0:
        pills.append('<span class="status-badge status-red">🚨 Hypoxic conditions detected</span>')
    chlc = pick_chl_column(df)
    chl, _ = _latest_and_30d_mean(df, chlc) if chlc else (None, None)
    if chl is not None and chl > 5.0:
        pills.append('<span class="status-badge status-amber">🌿 Phytoplankton bloom in progress</span>')
    if not pills:
        pills.append('<span class="status-badge status-green">✓ No threshold alerts on latest window</span>')
    return "<div class='alert-strip'>" + " ".join(pills) + "</div>"


def normalized_six_series_figure(df: pd.DataFrame, *, max_days: int = 90) -> go.Figure | None:
    """Last ``max_days`` of display window: six variables scaled 0–1 on one axis."""
    o2c = pick_o2_column(df)
    chlc = pick_chl_column(df)
    use: list[tuple[str, str]] = []
    for lab, role in [
        ("pH", "ph"),
        ("Temperature", "sst"),
        ("Salinity", "salinity"),
        ("O₂", "o2"),
        ("Nitrate", "no3"),
        ("Chlorophyll", "chl"),
    ]:
        if role == "o2":
            c = o2c
        elif role == "chl":
            c = chlc
        else:
            c = pick_best_column(df, role)
        if c and c in df.columns:
            use.append((friendly_axis_label(c), c))
    if len(use) < 2 or "time" not in df.columns:
        return None
    d = df[["time"] + [c for _, c in use]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    t1 = d["time"].max()
    d = d[d["time"] >= t1 - pd.Timedelta(days=max_days)]
    if len(d) < 10:
        return None
    step = max(1, len(d) // 4000)
    d = d.iloc[::step]
    fig = go.Figure()
    for lab, c in use:
        s = pd.to_numeric(d[c], errors="coerce")
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if hi <= lo or np.isnan(lo):
            continue
        z = (s - lo) / (hi - lo)
        fig.add_trace(go.Scatter(x=d["time"], y=z, mode="lines", name=lab, connectgaps=False))
    fig.update_layout(
        height=CHART_H_FULL,
        title="Core variables (min–max normalized to 0–1) — recent window",
        yaxis_title="normalized (0 = window min, 1 = window max)",
        xaxis_title="time (UTC)",
        legend=dict(orientation="h", y=1.08),
    )
    return apply_plotly(fig)
