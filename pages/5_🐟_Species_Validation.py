"""Species sightings (iNaturalist) vs mooring anomaly stress — validation view."""

from __future__ import annotations

import sys
from datetime import date, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cce_hack.anomaly import detect_anomalies, fit_mooring_isolation_scores
from cce_hack.column_pick import friendly_axis_label
from cce_hack.config import MOORING_SITES
from cce_hack.inaturalist import INAT_PLACE_CALIFORNIA, INAT_SPECIES, fetch_inaturalist_observations, synthetic_species_observations
from cce_hack.claude_narrative import interpret_species_correlation_llm
from cce_hack.plot_theme import apply_plotly
from cce_hack.streamlit_shell import (
    effective_llm_api_key,
    effective_llm_model,
    effective_llm_provider,
    inject_theme_css,
    page_config,
    render_global_sidebar,
)

_SPECIES_COLORS = {
    "Northern Anchovy": "#1565C0",
    "Pacific Sardine": "#2E7D32",
    "Humboldt Squid": "#E65100",
    "Blue Whale": "#6A1B9A",
}


@st.cache_data(ttl=3600)
def _cached_inaturalist(
    taxon_sci: str,
    place_id: int,
    days_back: int,
    quality_grade: str,
) -> pd.DataFrame:
    return fetch_inaturalist_observations(
        taxon_sci,
        place_id=place_id,
        days_back=days_back,
        quality_grade=quality_grade,
    )


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _weekly_species_counts(sight: pd.DataFrame) -> pd.DataFrame:
    if sight.empty or "date" not in sight.columns:
        return pd.DataFrame()
    s = sight.copy()
    s["_d"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["_d"])
    s["week"] = s["_d"].dt.to_period("W-MON").dt.start_time
    g = s.groupby(["week", "common_name"], observed=False).size().unstack(fill_value=0)
    return g.sort_index()


def _ts_to_naive_utc(x: object) -> object:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    xs = pd.Timestamp(x)
    if xs.tzinfo is not None:
        return xs.tz_convert(timezone.utc).to_pydatetime().replace(tzinfo=None)
    return xs.to_pydatetime()


def _weekly_mooring_stress(out: pd.DataFrame) -> pd.DataFrame:
    if out is None or out.empty or "time" not in out.columns:
        return pd.DataFrame(columns=["week", "stress"])
    t = pd.to_datetime(out["time"], utc=True, errors="coerce")
    tmp = pd.DataFrame({"t": t, "stress": -pd.to_numeric(out["iso_score"], errors="coerce")})
    tmp = tmp.dropna(subset=["t", "stress"]).set_index("t")
    w = tmp["stress"].resample("1W").mean().reset_index()
    w.columns = ["week", "stress"]
    w["week"] = w["week"].map(_ts_to_naive_utc)
    return w


def _naive_utc(ts: object) -> object | None:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.tz_convert(timezone.utc).to_pydatetime().replace(tzinfo=None)


def _build_correlation_table(
    events: pd.DataFrame,
    sight: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "Anomaly date (mooring)",
        "Stress driver (mooring)",
        "Sightings 14d before",
        "Sightings 14d after",
        "Change %",
    ]
    if events.empty or sight.empty:
        return pd.DataFrame(columns=cols)
    s = sight.copy()
    s["_d"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s = s.dropna(subset=["_d"])
    if s.empty:
        return pd.DataFrame(columns=cols)
    d_lo, d_hi = s["_d"].min(), s["_d"].max()
    rows: list[dict[str, object]] = []
    for _, er in events.iterrows():
        t = pd.to_datetime(er["time"], utc=True, errors="coerce")
        if pd.isna(t):
            continue
        t0 = (t - pd.Timedelta(days=14)).tz_localize(None).normalize()
        t1 = t.tz_localize(None).normalize()
        t2 = (t + pd.Timedelta(days=14)).tz_localize(None).normalize()
        # Skip mooring events that cannot overlap iNaturalist dates (avoids long tables of 0/0)
        if d_hi < t0 or d_lo > t2:
            continue
        before = s[(s["_d"] >= t0) & (s["_d"] < t1)]
        after = s[(s["_d"] >= t1) & (s["_d"] <= t2)]
        b_ct = int(len(before))
        a_ct = int(len(after))
        if b_ct > 0:
            chg = 100.0 * (a_ct - b_ct) / b_ct
        else:
            chg = float("nan")
        raw_var = str(er.get("variable", "") or "")
        nice_var = friendly_axis_label(raw_var) if raw_var else "—"
        rows.append(
            {
                "Anomaly date (mooring)": str(t1.date()),
                "Stress driver (mooring)": nice_var,
                "Sightings 14d before": b_ct,
                "Sightings 14d after": a_ct,
                "Change %": chg,
            }
        )
    return pd.DataFrame(rows)


page_config(title="Species Validation — CCE")
inject_theme_css()
df_moor = render_global_sidebar()

st.title("🐟 Species Validation")
st.markdown(
    "Do our detected ocean stress events correlate with changes in species sightings near the California Current? "
    "This page cross-references mooring anomaly flags with iNaturalist research-grade observations."
)

events_df = detect_anomalies(df_moor, contamination=0.05, top_n=80)
out_if, _feats = fit_mooring_isolation_scores(df_moor, contamination=0.05)

left, right = st.columns([1, 1])
with left:
    species_pick = st.multiselect(
        "Species (display filter)",
        list(INAT_SPECIES.keys()),
        default=list(INAT_SPECIES.keys()),
    )
with right:
    today = date.today()
    dr = st.date_input(
        "Observation window",
        value=(today - timedelta(days=1825), today),
        help="Filters charts after fetch and sets how far back to request from iNaturalist. Pick a span that **overlaps** your mooring anomaly years or the before/after table will be empty.",
    )
    if isinstance(dr, tuple) and len(dr) == 2:
        d_start, d_end = dr[0], dr[1]
    else:
        d_start, d_end = today - timedelta(days=365), today

days_back = max(7, (today - d_start).days + 1)

if st.button("Fetch iNaturalist data", type="primary"):
    with st.spinner("Fetching species observations from iNaturalist..."):
        merged_parts: list[pd.DataFrame] = []
        for common, sci in INAT_SPECIES.items():
            sub = _cached_inaturalist(sci, INAT_PLACE_CALIFORNIA, days_back, "research")
            if sub is not None and not sub.empty:
                sub = sub.copy()
                sub["common_name"] = common
                sub["species"] = sci
                merged_parts.append(sub[["date", "lat", "lon", "species", "common_name"]])
        if merged_parts:
            st.session_state["inat_data"] = pd.concat(merged_parts, ignore_index=True)
            st.session_state["inat_synthetic"] = False
        else:
            st.session_state["inat_synthetic"] = True
            st.session_state["inat_data"] = synthetic_species_observations(
                events_df["time"] if not events_df.empty else pd.Series(dtype="datetime64[ns, UTC]"),
                n=200,
                days_back=days_back,
                species_keys=list(INAT_SPECIES.keys()),
            )

if "inat_data" not in st.session_state or st.session_state["inat_data"] is None:
    st.info("Click the button above to load species sightings from iNaturalist.")
    st.stop()

if st.session_state.get("inat_synthetic"):
    st.warning("Using synthetic species data — iNaturalist API unavailable or returned no results.")

inat_raw: pd.DataFrame = st.session_state["inat_data"].copy()
inat_raw["date"] = pd.to_datetime(inat_raw["date"], errors="coerce").dt.date.astype(str)
mask_date = (pd.to_datetime(inat_raw["date"], errors="coerce").dt.date >= d_start) & (
    pd.to_datetime(inat_raw["date"], errors="coerce").dt.date <= d_end
)
sight = inat_raw.loc[mask_date].copy()
if not species_pick:
    st.warning("Select at least one species to plot sightings.")
    sight = sight.iloc[0:0]
else:
    sight = sight[sight["common_name"].isin(species_pick)].copy()

# --- Chart 1: weekly species vs stress (x-axis aligned to where sightings exist) ---
week_counts = _weekly_species_counts(sight)
week_stress_full = _weekly_mooring_stress(out_if) if out_if is not None else pd.DataFrame()

x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
week_stress_plot = week_stress_full
if not week_counts.empty:
    wc_idx = pd.to_datetime(week_counts.index, errors="coerce")
    t_obs_lo, t_obs_hi = wc_idx.min(), wc_idx.max()
    pad = pd.Timedelta(days=21)
    x_range = (pd.Timestamp(t_obs_lo) - pad, pd.Timestamp(t_obs_hi) + pad)
    if not week_stress_full.empty:
        ws = week_stress_full.copy()
        ws["_wt"] = pd.to_datetime(ws["week"], errors="coerce")
        m = (ws["_wt"] >= x_range[0]) & (ws["_wt"] <= x_range[1])
        week_stress_plot = ws.loc[m].drop(columns=["_wt"], errors="ignore")
elif not week_stress_full.empty:
    ws = week_stress_full.copy()
    ws["_wt"] = pd.to_datetime(ws["week"], errors="coerce")
    t_lo, t_hi = ws["_wt"].min(), ws["_wt"].max()
    pad = pd.Timedelta(days=30)
    x_range = (pd.Timestamp(t_lo) - pad, pd.Timestamp(t_hi) + pad)
    week_stress_plot = week_stress_full

fig = make_subplots(specs=[[{"secondary_y": True}]])

if not week_counts.empty:
    show_cols = list(week_counts.columns) if not species_pick else [c for c in week_counts.columns if c in species_pick]
    wx = pd.to_datetime(week_counts.index, errors="coerce")
    for col in show_cols:
        color = _SPECIES_COLORS.get(str(col), "#546E7A")
        fig.add_trace(
            go.Bar(
                x=wx,
                y=week_counts[col],
                name=str(col),
                marker_color=color,
                legendgroup=str(col),
            ),
            secondary_y=False,
        )
else:
    fig.add_trace(go.Bar(x=[], y=[], name="No sightings in window"), secondary_y=False)

if not week_stress_plot.empty:
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(week_stress_plot["week"], errors="coerce"),
            y=week_stress_plot["stress"],
            name="Mooring stress (−IF score, weekly mean)",
            mode="lines+markers",
            line=dict(color="#C62828", width=2),
            marker=dict(size=5, color="#C62828"),
        ),
        secondary_y=True,
    )

_vline_events = events_df.sort_values("severity", ascending=False).head(40) if not events_df.empty else events_df
for _, er in _vline_events.iterrows():
    ts_naive = _naive_utc(er["time"])
    if ts_naive is None:
        continue
    ts = pd.Timestamp(ts_naive)
    if x_range is not None and (ts < x_range[0] or ts > x_range[1]):
        continue
    fig.add_vline(x=ts_naive, line_width=1, line_dash="dash", line_color="rgba(198,40,40,0.55)")

fig.update_layout(
    barmode="stack",
    title="Weekly sightings vs mooring stress (aligned calendar span)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=520,
    margin=dict(l=48, r=48, t=72, b=48),
    yaxis_title="Weekly observation count",
    yaxis2_title="Mooring anomaly severity (weekly mean −IF)",
)
if x_range is not None:
    fig.update_xaxes(range=[x_range[0], x_range[1]])
fig.update_yaxes(secondary_y=False)
fig.update_yaxes(secondary_y=True, showgrid=False)
st.plotly_chart(apply_plotly(fig), use_container_width=True)
if week_counts.empty:
    st.caption(
        "No weekly sightings in this filter — widen **Observation window**, fetch again, or check species filters. "
        "iNaturalist only returns recent years unless you request a longer **days back** span."
    )
else:
    nwk = int(len(week_counts))
    cap = (
        f"**{nwk}** weeks with sightings (stacked bars). The red stress line is **cropped to the same calendar span** "
        "so it is comparable to sightings (previously it showed the full mooring record while sightings were only recent). "
        "Dashed verticals are top anomaly dates **inside this span**."
    )
    if week_stress_plot.empty and not week_stress_full.empty:
        cap += (
            " *Stress line omitted:* no weekly IF means inside this cropped span — nudge the observation window "
            "or confirm mooring times overlap sightings."
        )
    st.caption(cap)

# --- Map ---
st.subheader("Sightings map (California Current)")
sight_map = sight.dropna(subset=["lat", "lon"]).copy()
sight_map["r"] = sight_map["common_name"].map(lambda n: _hex_to_rgb(_SPECIES_COLORS.get(n, "#888888"))[0])
sight_map["g"] = sight_map["common_name"].map(lambda n: _hex_to_rgb(_SPECIES_COLORS.get(n, "#888888"))[1])
sight_map["b"] = sight_map["common_name"].map(lambda n: _hex_to_rgb(_SPECIES_COLORS.get(n, "#888888"))[2])
sight_map["tip"] = sight_map["common_name"].astype(str) + " · " + sight_map["species"].astype(str)

try:
    import pydeck as pdk

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=sight_map,
            get_position=["lon", "lat"],
            get_radius=2500,
            get_fill_color="[r, g, b, 200]",
            pickable=True,
        )
    ]
    moor_rows = [
        {"name": k, "lat": v["latitude"], "lon": v["longitude"]}
        for k, v in MOORING_SITES.items()
    ]
    moor_df = pd.DataFrame(moor_rows)
    moor_df["tip"] = moor_df["name"] + " mooring"
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=moor_df,
            get_position=["lon", "lat"],
            get_radius=18000,
            get_fill_color=[255, 255, 255, 240],
            stroked=True,
            get_line_color=[40, 40, 40, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
    )
    view = pdk.ViewState(latitude=33.5, longitude=-121.5, zoom=6.2, pitch=0)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"text": "{tip}"},
    )
    st.pydeck_chart(deck, use_container_width=True)
except Exception:
    if not sight_map.empty:
        st.map(sight_map.rename(columns={"lat": "lat", "lon": "lon"}), use_container_width=True)
    else:
        st.info("No geolocated sightings in the selected window.")

st.markdown(
    "**Legend** — sightings: "
    "<span style='color:#1565C0'>⬤</span> Northern Anchovy · "
    "<span style='color:#2E7D32'>⬤</span> Pacific Sardine · "
    "<span style='color:#E65100'>⬤</span> Humboldt Squid · "
    "<span style='color:#6A1B9A'>⬤</span> Blue Whale · "
    "<span style='color:#37474F'>⬤</span> CCE moorings",
    unsafe_allow_html=True,
)

# --- Correlation table ---
st.subheader("Sightings before vs after each anomaly")
corr_tbl = _build_correlation_table(events_df, sight)
if corr_tbl.empty:
    st.caption(
        "No rows: either no sightings after filters, or **no mooring anomaly dates overlap** your sighting dates. "
        "Widen the observation window toward the years where your mooring shows stress, then **Fetch** again."
    )
else:
    display_df = corr_tbl.copy()

    def _style_change_pct(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        try:
            x = float(v)
        except (TypeError, ValueError):
            return ""
        if x < 0:
            return "color: #c0392b; font-weight: 600;"
        if x > 0:
            return "color: #1e8449; font-weight: 600;"
        return ""

    styler = display_df.style.format({"Change %": "{:.1f}%"}, na_rep="—").map(_style_change_pct, subset=["Change %"])
    st.dataframe(styler, use_container_width=True, hide_index=True)

# --- AI interpretation (optional; uses sidebar provider + key) ---
if effective_llm_api_key():
    if st.button("Ask AI to interpret the correlation"):
        with st.spinner(f"Calling {effective_llm_provider()}…"):
            text = interpret_species_correlation_llm(
                effective_llm_api_key(),
                effective_llm_provider(),
                effective_llm_model(),
                events_df=events_df,
                corr_df=corr_tbl,
            )
        with st.chat_message("assistant"):
            st.markdown(text)
else:
    st.caption("Add a **Gemini** or **Groq** API key in the sidebar (free tiers) to enable AI interpretation.")
