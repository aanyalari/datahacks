"""AI Predictions — friendly UI for forecast, seasonal outlook, anomalies, soft sensor."""

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

from cce_hack.anomaly_iso import (
    build_anomaly_rank_table,
    feature_z_scores_at_row,
    isolation_forest_anomalies,
)
from cce_hack.config import DEFAULT_HORIZON_HOURS, LAG_HOURS
from cce_hack.features import add_calendar_features, add_lags
from cce_hack.pipeline import default_lag_columns, run_forecast_experiment
from cce_hack.plot_theme import PLOTLY_BASE, apply_plotly
from cce_hack.soft_sensor import candidate_soft_sensor_targets, train_soft_sensor
from cce_hack.streamlit_shell import (
    CHART_H_FULL,
    friendly_column_label_plain,
    inject_theme_css,
    numeric_series_cols,
    page_config,
    render_global_sidebar,
)

page_config(title="AI Predictions — CCE")
inject_theme_css()
df = render_global_sidebar()

st.title("AI predictions")
st.caption(
    "Four simple tools, one tab each. Pick a sensor, click a button, get a chart."
)

num = numeric_series_cols(df)


def _label(col: str) -> str:
    return friendly_column_label_plain(col)


# ---- plain-English helpers for anomaly explanations ----
_UNIT_FOR = {
    "wind_speed_ms": "m/s",
    "sst_c": "°C",
    "air_temp_c": "°C",
    "ph_total": "",
    "chl_mg_m3": "mg/m³",
    "salinity_psu": "PSU",
    "dissolved_oxygen_mg_l": "mg/L",
    "pco2_uatm": "µatm",
    "no3": "µmol/L",
    "conductivity_s_m": "S/m",
}


def _unit_for(col: str) -> str:
    base = col.split("_d")[0] if "_d" in col else col
    return _UNIT_FOR.get(col) or _UNIT_FOR.get(base, "")


def _format_value(col: str, val: float) -> str:
    if val != val:
        return "—"
    unit = _unit_for(col)
    return f"{val:.2f} {unit}".strip() if unit else f"{val:.3f}"


def _magnitude_word(abs_z: float) -> str:
    if abs_z >= 3:
        return "extremely"
    if abs_z >= 2:
        return "much"
    if abs_z >= 1.5:
        return "noticeably"
    if abs_z >= 1:
        return "somewhat"
    return "slightly"


def _direction_word(z: float) -> str:
    return "higher" if z >= 0 else "lower"


# (column_substring, "high"/"low") -> plausible real-world cause
_ANOM_REASONS: dict[tuple[str, str], str] = {
    ("wind", "high"): "This usually happens during **storms or strong weather fronts** passing over the mooring.",
    ("wind", "low"): "Very calm winds often go with a **stable high-pressure system** sitting over the area.",
    ("chl", "high"): "Chlorophyll spikes typically mean a **phytoplankton bloom**, often fed by **upwelling** or nutrient-rich runoff.",
    ("chl", "low"): "Could indicate **low biological productivity** — e.g. nutrient-poor water or the period after a bloom collapses.",
    ("ph_total", "high"): "When phytoplankton **photosynthesise heavily** (e.g. during a bloom) they pull CO₂ out of the water, making it more alkaline.",
    ("ph_total", "low"): "Acidic water often comes from **respiration, decomposition, or upwelled CO₂-rich deep water**.",
    ("sst", "high"): "Could be a **marine heatwave**, El Niño influence, or a stretch of calm sunny weather warming the surface.",
    ("sst", "low"): "Often a sign of **upwelling** — cold, deep water rising to the surface — or a cold-water intrusion.",
    ("sal", "high"): "Suggests strong **evaporation** or unusually little freshwater input.",
    ("sal", "low"): "Usually **freshwater input** — heavy rain, river runoff, or melting upstream — diluting the seawater.",
    ("dissolved_oxygen", "high"): "Often follows **heavy photosynthesis** (blooms) or cold, oxygen-rich water reaching the sensor.",
    ("dissolved_oxygen", "low"): "Can signal **hypoxia** — warm water, decomposition, or stratification cutting off the oxygen supply.",
    ("pco2", "high"): "Driven by **respiration or decomposition** in the water, or **upwelling** of CO₂-rich deep water.",
    ("pco2", "low"): "Phytoplankton are likely **consuming CO₂ via photosynthesis**, often during a bloom.",
    ("no3", "high"): "Suggests **upwelling** of nutrient-rich deep water or a runoff event delivering nitrate.",
    ("no3", "low"): "Probably means **nutrients are being used up by phytoplankton** faster than they're replaced.",
    ("air_temp", "high"): "Likely a **heatwave** or a sunny calm spell warming the air above the buoy.",
    ("air_temp", "low"): "Suggests a **cold front** or unusually cool weather over the mooring.",
}


def _reason_for(col: str, z: float) -> str:
    direction = "high" if z >= 0 else "low"
    key = col.lower()
    for (frag, d), text in _ANOM_REASONS.items():
        if frag in key and d == direction:
            return text
    return ""


tab_short, tab_anom, tab_soft = st.tabs(
    [
        "Short forecast",
        "Find unusual moments",
        "Reconstruct a sensor",
    ]
)

# =============================================================================
# 1) Short forecast
# =============================================================================
with tab_short:
    st.subheader("Predict a sensor a few hours into the future")
    st.markdown(
        "Pick a sensor and how far ahead you want to predict. "
        "We train a model on past patterns of **all** sensors and then test it on the **most recent slice of data** "
        "the model has never seen. The chart compares the model's guess to what actually happened, plus two simple guesses for context: "
        "**“same as now”** and **“same as 24 hours ago.”**"
    )

    preferred = [
        c
        for c in ("sst_c", "ph_total", "chl_mg_m3", "dissolved_oxygen_mg_l", "salinity_psu", "pco2_uatm")
        if c in num
    ]
    options = preferred if preferred else num
    c1, c2 = st.columns([2, 1])
    with c1:
        fc_target = st.selectbox(
            "Sensor to predict",
            options,
            format_func=_label,
            key="hgb_target",
        )
    with c2:
        horizon_opts = [6, 12, 24, 48, 72]
        h_ix = horizon_opts.index(DEFAULT_HORIZON_HOURS) if DEFAULT_HORIZON_HOURS in horizon_opts else 2
        fc_horizon = st.select_slider(
            "Hours ahead", options=horizon_opts, value=horizon_opts[h_ix], key="hgb_hz"
        )

    if st.button("Train model", type="primary", key="btn_hgb"):
        with st.spinner("Training…"):
            exp = run_forecast_experiment(df, fc_target, horizon_h=fc_horizon)
        st.session_state["fc_exp"] = exp

    exp = st.session_state.get("fc_exp")
    if exp is None:
        st.info("Pick a sensor and horizon, then click **Train model**.")
    else:
        res = exp.result

        # ---- Forward forecast at the latest available timestamp ----
        forecast_value: float | None = None
        forecast_time: pd.Timestamp | None = None
        latest_value: float | None = None
        latest_time: pd.Timestamp | None = None
        try:
            base_cols = [c for c in default_lag_columns() if c in df.columns]
            if res.target in df.columns and res.target not in base_cols:
                base_cols = [res.target] + base_cols
            df_ts = df.copy()
            df_ts["time"] = pd.to_datetime(df_ts["time"], utc=True, errors="coerce")
            df_ts = df_ts.dropna(subset=["time"]).sort_values("time")
            d_full = add_lags(df_ts, base_cols, LAG_HOURS)
            d_full = add_calendar_features(d_full)
            tgt_present = d_full.dropna(subset=[res.target])
            if not tgt_present.empty:
                last_row = tgt_present.iloc[[-1]]
                latest_time = pd.to_datetime(last_row["time"].iloc[0], utc=True)
                latest_value = float(last_row[res.target].iloc[0])
                forecast_time = latest_time + pd.Timedelta(hours=res.horizon_h)
                feat_cols_present = [c for c in res.feature_columns if c in last_row.columns]
                if feat_cols_present:
                    X_last = last_row[feat_cols_present]
                    forecast_value = float(res.pipeline.predict(X_last)[0])
        except Exception:
            forecast_value = None

        target_lbl = _label(res.target)
        if forecast_value is None:
            st.warning("Could not compute a forward forecast for this sensor with current data.")
        else:
            ts_str = forecast_time.strftime("%Y-%m-%d %H:%M UTC") if forecast_time is not None else f"+{res.horizon_h}h"
            now_str = latest_time.strftime("%Y-%m-%d %H:%M UTC") if latest_time is not None else ""
            delta_str = ""
            if latest_value is not None:
                delta = forecast_value - latest_value
                delta_str = f" ({delta:+.3f} vs now)"
            st.markdown(
                f"""
<div class="hero" style="margin: 0.5rem 0 0.75rem 0;">
  <h1 style="font-size: 1.2rem; margin-bottom: 0.25rem;">
    Predicted {target_lbl} in {res.horizon_h} hours
  </h1>
  <p style="margin: 0;">
    <span style="font-size: 2rem; font-weight: 700;">{forecast_value:.3f}</span>
    <span style="opacity: 0.75; margin-left: 0.5rem;">{delta_str}</span>
    <br/>
    <small>at <strong>{ts_str}</strong>{f" · current value {latest_value:.3f} at {now_str}" if latest_value is not None else ""}</small>
  </p>
</div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# 2) Find unusual moments (Isolation Forest)
# =============================================================================
with tab_anom:
    st.subheader("Find unusual moments in the data")
    st.markdown(
        "Looks at several sensors **together** and flags any moment where the combination is unusual compared to the rest of the record. "
        "Useful for spotting sensor faults, storms, blooms, or other rare events you didn't know to look for."
    )

    default_feats = [c for c in ("sst_c", "salinity_psu", "ph_total", "chl_mg_m3", "wind_speed_ms") if c in num]
    if len(default_feats) < 2:
        default_feats = num[: min(5, len(num))]
    feats = st.multiselect(
        "Sensors to consider together",
        num,
        default=default_feats[: min(6, len(default_feats))],
        format_func=_label,
        key="if_feats",
    )

    with st.expander("Advanced", expanded=False):
        cont = st.slider(
            "How sensitive should the detector be? (higher = flags more moments)",
            0.005,
            0.15,
            0.02,
            0.005,
            key="if_cont",
        )

    if not feats or len(feats) < 2:
        st.warning("Pick at least two sensors.")
    else:
        cont_val = st.session_state.get("if_cont", 0.02)
        out, _y_pred, _model, _scaler = isolation_forest_anomalies(df, feats, contamination=cont_val)
        if out is None:
            st.warning("Need at least 80 overlapping rows of those sensors.")
        else:
            n_total = len(out)
            n_flag = int(out["anomaly"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Moments scored", f"{n_total:,}")
            m2.metric("Moments flagged", f"{n_flag:,}")
            m3.metric("Flag rate", f"{(n_flag / max(n_total,1) * 100):.1f}%")

            plot_col = st.selectbox("Show on chart", feats, format_func=_label, key="if_plotcol")

            ref_series = pd.to_numeric(df[plot_col], errors="coerce")
            mu_p = float(ref_series.mean())
            sig_p = float(ref_series.std())
            lo_band = mu_p - 2 * sig_p if sig_p == sig_p and sig_p > 0 else None
            hi_band = mu_p + 2 * sig_p if sig_p == sig_p and sig_p > 0 else None

            dplot = out.copy()
            dplot["time"] = pd.to_datetime(dplot["time"], utc=True, errors="coerce")
            dplot = dplot.dropna(subset=["time"]).sort_values("time")
            step = max(1, len(dplot) // 8000)
            dplot = dplot.iloc[::step]

            fig = go.Figure()
            if lo_band is not None and hi_band is not None:
                fig.add_trace(
                    go.Scatter(
                        x=list(dplot["time"]) + list(dplot["time"][::-1]),
                        y=[hi_band] * len(dplot) + [lo_band] * len(dplot),
                        fill="toself",
                        fillcolor="rgba(122,168,255,0.12)",
                        line=dict(width=0),
                        name="Typical range (±2σ)",
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=dplot["time"],
                    y=dplot[plot_col],
                    mode="lines",
                    name=_label(plot_col),
                    line=dict(width=1.2, color="#7aa8ff"),
                )
            )
            bad = dplot[dplot["anomaly"] == 1].copy()
            if not bad.empty:
                if sig_p and sig_p > 0:
                    bad["_z"] = (pd.to_numeric(bad[plot_col], errors="coerce") - mu_p) / sig_p
                else:
                    bad["_z"] = np.nan
                bad["_dir"] = np.where(bad["_z"] >= 0, "above normal", "below normal")
                hover = [
                    f"<b>{ts:%Y-%m-%d %H:%M UTC}</b><br>{_label(plot_col)}: {v:.3f}"
                    + (f"<br>{abs(z):.1f}σ {d}" if z == z else "")
                    for ts, v, z, d in zip(bad["time"], bad[plot_col], bad["_z"], bad["_dir"])
                ]
                fig.add_trace(
                    go.Scatter(
                        x=bad["time"],
                        y=bad[plot_col],
                        mode="markers",
                        name="Unusual moment",
                        marker=dict(color="#ff4466", size=11, line=dict(width=1.2, color="#fff"), symbol="circle"),
                        hovertext=hover,
                        hoverinfo="text",
                    )
                )
            fig.update_layout(
                height=CHART_H_FULL,
                title=f"{_label(plot_col)} — unusual moments highlighted",
                xaxis_title="Date (UTC)",
                yaxis_title=_label(plot_col),
                hovermode="closest",
                legend=dict(orientation="h", y=1.06, x=0),
                **{k: v for k, v in PLOTLY_BASE.items() if k not in ("margin",)},
            )
            st.plotly_chart(apply_plotly(fig), use_container_width=True)
            st.caption(
                "Blue band = typical range for this sensor (±2 standard deviations from its long-term average). "
                "Red dots are moments where the **combination** of all selected sensors looked unusual — even if this one sensor stayed inside the band."
            )

            # ---------------- Plain-English summary of anomalies ----------------
            ref_all = df[feats].apply(pd.to_numeric, errors="coerce")
            mu_all, sig_all = ref_all.mean(), ref_all.std().replace(0, np.nan)
            anom = out[out["anomaly"] == 1].copy()
            anom["time"] = pd.to_datetime(anom["time"], utc=True, errors="coerce")
            anom = anom.dropna(subset=["time"])

            if anom.empty:
                st.info("No unusual moments were flagged with the current sensitivity.")
            else:
                drivers = []
                for _, r in anom.iterrows():
                    z_best, var_best = 0.0, None
                    for c in feats:
                        if c not in r.index or pd.isna(r[c]):
                            continue
                        m, s = float(mu_all.get(c, np.nan)), float(sig_all.get(c, np.nan))
                        if not (m == m) or not (s == s) or s == 0:
                            continue
                        z = (float(r[c]) - m) / s
                        if abs(z) > abs(z_best):
                            z_best, var_best = z, c
                    if var_best is not None:
                        drivers.append((var_best, z_best))

                first_t = anom["time"].min()
                last_t = anom["time"].max()

                month_counts = anom["time"].dt.to_period("M").value_counts().sort_values(ascending=False)
                peak_month_sentence = ""
                if len(month_counts) and int(month_counts.iloc[0]) >= max(3, int(0.15 * len(anom))):
                    pm = month_counts.index[0].to_timestamp()
                    peak_month_sentence = (
                        f" The biggest cluster was in **{pm:%B %Y}** "
                        f"({int(month_counts.iloc[0])} of them happened that month)."
                    )

                st.markdown("#### Why these moments stand out")
                st.markdown(
                    f"Out of all the time points checked, the detector picked out **{len(anom):,}** "
                    f"that didn't look like the rest of the record — they happened "
                    f"between **{first_t:%B %Y}** and **{last_t:%B %Y}**.{peak_month_sentence}"
                )

                if drivers:
                    drv_df = pd.DataFrame(drivers, columns=["var", "z"])
                    top_vars = drv_df["var"].value_counts().head(3)
                    st.markdown("**What was driving most of them**")
                    for v, cnt in top_vars.items():
                        sub = drv_df[drv_df["var"] == v]["z"]
                        avg_z = float(sub.mean())
                        mag = _magnitude_word(abs(avg_z))
                        dir_word = _direction_word(avg_z)
                        reason = _reason_for(v, avg_z)
                        line = (
                            f"- **{_label(v)}** was **{mag} {dir_word} than usual** "
                            f"in **{int(cnt)}** of the {len(anom)} flagged moments."
                        )
                        if reason:
                            line += f" {reason}"
                        st.markdown(line)

                top3 = anom.nsmallest(3, "iso_score")
                if not top3.empty:
                    st.markdown("**The three most extreme moments**")
                    for _, r in top3.iterrows():
                        z_best, var_best, val_best = 0.0, None, np.nan
                        for c in feats:
                            if c not in r.index or pd.isna(r[c]):
                                continue
                            m, s = float(mu_all.get(c, np.nan)), float(sig_all.get(c, np.nan))
                            if not (m == m) or not (s == s) or s == 0:
                                continue
                            z = (float(r[c]) - m) / s
                            if abs(z) > abs(z_best):
                                z_best, var_best, val_best = z, c, float(r[c])
                        if var_best is None:
                            continue
                        mag = _magnitude_word(abs(z_best))
                        dir_word = _direction_word(z_best)
                        reason = _reason_for(var_best, z_best)
                        when = f"{r['time']:%-d %B %Y, %-I %p UTC}"
                        line = (
                            f"- **{when}** — {_label(var_best)} reached "
                            f"**{_format_value(var_best, val_best)}**, which is **{mag} {dir_word}** than the typical range."
                        )
                        if reason:
                            line += f" {reason}"
                        st.markdown(line)

                st.caption(
                    "The detector looks at all the selected sensors **together** — a moment is flagged when their combined "
                    "behaviour is rare compared with the rest of the record. The reasons above are common real-world "
                    "explanations, not a definitive cause."
                )

            rank10 = build_anomaly_rank_table(df, out, feats, n=10)
            if not rank10.empty:
                rank_display = rank10.drop(columns=[c for c in ("_col",) if c in rank10.columns])
                st.markdown("**Top 10 most unusual moments**")
                st.dataframe(rank_display, use_container_width=True, hide_index=True, height=320)
                st.caption(
                    "**Sensor (what moved most)** is the sensor that was farthest from normal at that moment; "
                    "the **Z** value is how many standard deviations away from its long-term average it was."
                )

                with st.expander("Why was a specific moment flagged?", expanded=False):
                    pick_i = st.selectbox(
                        "Pick a row",
                        list(range(len(rank10))),
                        format_func=lambda i: (
                            f"{rank10.iloc[i]['When (UTC)']} — {rank10.iloc[i]['Sensor (what moved most)']} "
                            f"(z={rank10.iloc[i]['Z vs whole mooring file']})"
                        ),
                        key="driver_pick",
                    )
                    t_str = rank10.iloc[pick_i]["When (UTC)"]
                    t_match = pd.to_datetime(t_str, utc=True, errors="coerce")
                    row_series = None
                    if pd.notna(t_match):
                        near = out.copy()
                        near["time"] = pd.to_datetime(near["time"], utc=True, errors="coerce")
                        delta = (near["time"] - t_match).abs()
                        if delta.notna().any():
                            row_series = near.loc[delta.idxmin()]
                    if row_series is None:
                        row_series = out.nsmallest(1, "iso_score").iloc[0]
                    ref = df[feats].apply(pd.to_numeric, errors="coerce")
                    mu, sig = ref.mean(), ref.std().replace(0, np.nan)
                    ztab = feature_z_scores_at_row(row_series, feats, mu, sig)
                    if not ztab.empty:
                        ztab = ztab.copy()
                        ztab["feature_label"] = ztab["feature"].apply(_label)
                        ztab_top = ztab.head(8).iloc[::-1]
                        ztab_top["dir"] = np.where(ztab_top["z"] >= 0, "above normal", "below normal")
                        ztab_top["abs_z"] = ztab_top["z"].abs()
                        zplot = go.Figure(
                            go.Bar(
                                x=ztab_top["z"],
                                y=ztab_top["feature_label"],
                                orientation="h",
                                marker=dict(
                                    color=ztab_top["z"],
                                    colorscale="RdBu_r",
                                    cmin=-max(3, ztab_top["abs_z"].max()),
                                    cmax=max(3, ztab_top["abs_z"].max()),
                                    line=dict(width=0),
                                ),
                                text=[f"{z:+.1f}σ" for z in ztab_top["z"]],
                                textposition="outside",
                                hovertemplate="%{y}: %{x:+.2f}σ<extra></extra>",
                            )
                        )
                        zplot.add_vline(x=0, line=dict(color="rgba(255,255,255,0.4)", width=1))
                        for x in (-2, 2):
                            zplot.add_vline(x=x, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"))
                        zplot.update_layout(
                            height=min(380, 100 + 34 * len(ztab_top)),
                            title="How unusual was each sensor at this moment?",
                            xaxis_title="Standard deviations from normal (0 = average)",
                            yaxis_title="",
                            showlegend=False,
                            **{k: v for k, v in PLOTLY_BASE.items() if k not in ("margin",)},
                        )
                        st.plotly_chart(apply_plotly(zplot), use_container_width=True)

                        top_drv = ztab.iloc[0]
                        direction = "above" if top_drv["z"] >= 0 else "below"
                        ts_lbl = (
                            f"{t_match:%Y-%m-%d %H:%M UTC}"
                            if pd.notna(t_match)
                            else str(row_series.get("time", ""))[:19]
                        )
                        flagged_count = int(((ztab["z"].abs() >= 2)).sum())
                        st.markdown(
                            f"**Why this moment was flagged:** at **{ts_lbl}**, "
                            f"**{_label(top_drv['feature'])}** was **{abs(top_drv['z']):.1f}σ {direction} normal** "
                            f"(value ≈ {top_drv['value']:.3f}). "
                            f"In total **{flagged_count}** of the {len(ztab)} selected sensors were beyond ±2σ at the same time, "
                            "which is what made the combination stand out."
                        )
                        st.caption("Dotted lines mark the ±2σ 'still normal' zone. Bars beyond it are the ones pushing this moment into unusual territory.")

# =============================================================================
# 3) Soft sensor — reconstruct one channel from the others
# =============================================================================
with tab_soft:
    st.subheader("Reconstruct a sensor from the others")
    st.markdown(
        "Pretend one sensor has failed. Can we estimate what it *would* have read using the **other sensors**? "
        "We train on the older part of the record and test on the newest part the model has never seen. "
        "If the model's reconstruction matches what the real sensor reported, that sensor's value is **predictable** from its neighbours — "
        "useful for **gap-filling** or as a **sanity check** when a sensor starts behaving oddly."
    )

    soft_targets = candidate_soft_sensor_targets(df) or num
    soft_target = st.selectbox(
        "Sensor to reconstruct",
        soft_targets,
        format_func=_label,
        key="soft_target",
    )

    with st.expander("Advanced", expanded=False):
        other_num = [c for c in num if c != soft_target]
        soft_feats = st.multiselect(
            "Which other sensors to use as inputs (empty = all of them)",
            other_num,
            default=[],
            format_func=_label,
            key="soft_feats",
        )
        valid_frac = st.slider(
            "Test on the last X% of the data",
            0.1,
            0.4,
            0.2,
            0.05,
            key="soft_vf",
        )

    if st.button("Train reconstruction", type="primary", key="btn_soft"):
        with st.spinner("Training…"):
            res = train_soft_sensor(
                df,
                target=soft_target,
                feature_cols=st.session_state.get("soft_feats") or None,
                valid_frac=float(st.session_state.get("soft_vf", 0.2)),
            )
        st.session_state["soft_res"] = res

    res = st.session_state.get("soft_res")
    if res is None:
        st.info("Pick a sensor and click **Train reconstruction**.")
    else:
        skill = (
            100.0 * (1.0 - res.valid_mae / res.mean_baseline_mae)
            if res.mean_baseline_mae and res.mean_baseline_mae > 1e-9
            else float("nan")
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Reconstruction error (avg)", f"{res.valid_mae:.3f}")
        m2.metric(
            "Variance explained",
            f"{(res.valid_r2 * 100):.0f}%" if not np.isnan(res.valid_r2) else "—",
            help="How much of the sensor's wiggle the model captures (100% = perfect).",
        )
        m3.metric(
            "Better than guessing the average",
            f"{skill:+.0f}%" if not np.isnan(skill) else "—",
        )

        vf = res.valid_frame
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=vf["time"], y=vf["actual"], name="Real sensor", line=dict(width=1.2)))
        figr.add_trace(go.Scatter(x=vf["time"], y=vf["predicted"], name="Reconstructed", line=dict(width=1.6)))
        figr.update_layout(
            height=CHART_H_FULL,
            title=f"{_label(res.target)} — reconstruction on the most recent slice",
            xaxis_title="Date (UTC)",
            yaxis_title=_label(res.target),
            **{k: v for k, v in PLOTLY_BASE.items() if k != "margin"},
        )
        st.plotly_chart(apply_plotly(figr), use_container_width=True)
        st.caption(
            "If the two lines overlap, the other sensors carry enough information to stand in for this one. "
            "If they diverge, this sensor measures something the others can't capture — keep the real reading."
        )

        if not res.importance.empty:
            with st.expander("Which other sensors helped most?", expanded=False):
                top = res.importance.head(10).copy()
                top["feature"] = top["feature"].apply(_label)
                fig_imp = px.bar(top, x="importance", y="feature", orientation="h")
                fig_imp.update_layout(
                    height=min(360, 100 + 24 * len(top)),
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Importance (higher = the model leaned on it more)",
                    yaxis_title="",
                )
                st.plotly_chart(apply_plotly(fig_imp), use_container_width=True)
