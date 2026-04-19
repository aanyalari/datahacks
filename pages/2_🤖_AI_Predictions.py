"""AI / ML — Isolation Forest, ranked anomalies, hypoxia gauge, optional LLM chat (sidebar provider)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cce_hack.anomaly_iso import build_anomaly_rank_table, isolation_forest_anomalies, top_anomaly_events
from cce_hack.claude_narrative import explain_single_anomaly, interpret_top_anomalies
from cce_hack.column_pick import friendly_axis_label
from cce_hack.plot_theme import apply_plotly, current_plotly_theme
from cce_hack.risk_scores import hypoxia_risk_breakdown, hypoxia_risk_score_0_100
from cce_hack.streamlit_shell import (
    CHART_H_FULL,
    CHART_H_HALF,
    effective_llm_api_key,
    effective_llm_model,
    effective_llm_provider,
    inject_theme_css,
    numeric_series_cols,
    page_config,
    render_global_sidebar,
)

page_config(title="AI Predictions — CCE")
inject_theme_css()
df = render_global_sidebar()

st.title("AI predictions & anomalies")
st.caption(
    "**Isolation Forest** flags rows that are unusual in **combined** feature space (not per-variable spikes). "
    "**Stress index** (gauge) is a fixed formula from recent O₂ / pH / T / nitrate vs earlier data. "
    "**Gemini / Groq** blurbs only run when you click a button (sidebar key)."
)

num = numeric_series_cols(df)
default_feats = [c for c in ("sst_c", "salinity_psu", "ph_total", "chl_mg_m3", "wind_speed_ms", "no3") if c in num]
if len(default_feats) < 2:
    default_feats = num[: min(5, len(num))]

feats = st.multiselect("Features for anomaly detector", num, default=default_feats[: min(6, len(default_feats))])
cont = st.slider("Expected anomaly fraction (contamination)", 0.005, 0.15, 0.02, 0.005)

if not feats or len(feats) < 2:
    st.warning("Pick at least two numeric features.")
    st.stop()

out, y_pred, model, scaler = isolation_forest_anomalies(df, feats, contamination=cont)
if out is None:
    st.warning("Need ≥80 overlapping rows with non-null features.")
    st.stop()

st.success(f"Scored **{len(out):,}** rows · **{int(out['anomaly'].sum())}** anomaly flags.")

rank10 = build_anomaly_rank_table(df, out, feats, n=10)
risk = hypoxia_risk_score_0_100(df)
brk = hypoxia_risk_breakdown(df)
gauge_bar = "#2563eb" if current_plotly_theme() == "light" else "#7aa8ff"
_tw = str(st.session_state.get("cce_time_window", "all"))
_tend = ""
if len(df) and "time" in df.columns:
    _mx = pd.to_datetime(df["time"], utc=True, errors="coerce").max()
    if pd.notna(_mx):
        _tend = str(_mx.date())
_gauge_key = f"stress-{_tw}-{len(df)}-{_tend}"

# --- Hypoxia gauge ---
g = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Stress index (0–100, hypoxia-style)"},
        number={"suffix": ""},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_bar},
            "steps": [
                {"range": [0, 35], "color": "rgba(40,180,120,0.25)"},
                {"range": [35, 65], "color": "rgba(230,170,40,0.25)"},
                {"range": [65, 100], "color": "rgba(220,70,70,0.28)"},
            ],
        },
    )
)
st.plotly_chart(
    apply_plotly(g).update_layout(height=220, margin=dict(l=24, r=24, t=48, b=24)),
    use_container_width=True,
    key=_gauge_key,
)
parts = []
for label, zkey in (
    ("O₂", "z_o2_recent"),
    ("pH", "z_ph_recent"),
    ("SST", "z_temp_recent"),
    ("NO₃", "z_nitrate_recent"),
):
    zv = brk.get(zkey)
    if zv == zv and isinstance(zv, (int, float)):
        parts.append(f"{label} tail-vs-window **z≈{float(zv):+.2f}**")
st.caption(
    "Each z compares the **last ~5 days** to the **mean and standard deviation of everything in the current sidebar window** "
    "(so changing **Time window** or **Mooring slice** changes the baseline and the score). "
    + (" · ".join(parts) if parts else "Not enough history for z-scores.")
    + " — **not** a regulatory forecast. "
    + f"Current window: **{_tw}** · **{len(df):,}** rows · index **{risk:.2f}**."
)
miss = [k for k, v in (("O₂", brk.get("o2_col")), ("pH", brk.get("ph_col")), ("SST", brk.get("temp_col")), ("NO₃", brk.get("no3_col"))) if not v]
if miss:
    st.caption("Missing columns (score stays muted): **" + "**, **".join(miss) + "**.")

plot_col = feats[0]
plot_nice = friendly_axis_label(plot_col)
left, right = st.columns([1.15, 1.0], gap="medium")

with left:
    st.subheader("Timeline + anomalies")
    dplot = out.copy()
    dplot["time"] = pd.to_datetime(dplot["time"], utc=True, errors="coerce")
    dplot = dplot.dropna(subset=["time"]).sort_values("time")
    step = max(1, len(dplot) // 8000)
    dplot = dplot.iloc[::step]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dplot["time"], y=dplot[plot_col], mode="lines", name=plot_nice, line=dict(width=1.2))
    )
    bad = dplot[dplot["anomaly"] == 1]
    if not bad.empty:
        fig.add_trace(
            go.Scatter(
                x=bad["time"],
                y=bad[plot_col],
                mode="markers",
                name="anomaly",
                marker=dict(color="#ff4466", size=11, line=dict(width=1, color="#fff")),
            )
        )
    fig.update_layout(
        height=CHART_H_FULL,
        title=f"{plot_nice} (line) + anomaly hits (red points)",
        xaxis_title="time (UTC)",
    )
    st.plotly_chart(apply_plotly(fig), use_container_width=True)
    n_bad = int(dplot["anomaly"].sum()) if "anomaly" in dplot.columns else 0
    st.caption(
        f"**{plot_nice}** — red dots are rows the Isolation Forest called unusual given **{len(feats)}** features together "
        f"({n_bad} points shown after decimation). Compare dates with **Analytics** for a process story."
    )

with right:
    st.subheader("Top 10 anomaly events")
    st.caption(
        "Each row is one timestamp with a very low isolation score. **Sensor (what moved most)** is the variable whose value "
        "was farthest from the file-wide mean at that instant (largest |Z|)."
    )
    show_cols = [c for c in rank10.columns if c != "_col"]
    st.dataframe(rank10[show_cols], use_container_width=True, hide_index=True, height=CHART_H_FULL)

st.divider()
st.subheader("AI analyst (sidebar provider)")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Summarize top anomalies", type="primary"):
        top3 = top_anomaly_events(out, n=3)
        with st.spinner(f"Calling {effective_llm_provider()}…"):
            txt = interpret_top_anomalies(
                effective_llm_api_key(),
                effective_llm_provider(),
                effective_llm_model(),
                top3,
                context_lines=f"Features used: {', '.join(feats)}.",
            )
        st.session_state["claude_top"] = txt
    if st.session_state.get("claude_top"):
        with st.chat_message("assistant"):
            st.markdown(st.session_state["claude_top"])

with col_b:
    if rank10.empty:
        st.caption("No ranked rows — run detector on more data.")
    else:
        idx = st.selectbox(
            "Pick a row to explain",
            list(range(len(rank10))),
            format_func=lambda i: (
                f"{rank10.iloc[i]['When (UTC)']} | {rank10.iloc[i]['Sensor (what moved most)']} | "
                f"z={rank10.iloc[i]['Z vs whole mooring file']}"
            ),
        )
        if st.button("Explain this anomaly"):
            row = rank10.iloc[idx].drop(labels=["_col"], errors="ignore")
            ev = pd.DataFrame([row]).to_markdown(index=False)
            with st.spinner(f"Calling {effective_llm_provider()}…"):
                txt2 = explain_single_anomaly(
                    effective_llm_api_key(),
                    effective_llm_provider(),
                    effective_llm_model(),
                    event_markdown=ev,
                    feature_context=", ".join(feats),
                )
            st.session_state["claude_one"] = txt2
        if st.session_state.get("claude_one"):
            with st.chat_message("assistant"):
                st.markdown(st.session_state["claude_one"])

st.divider()
st.subheader("Naive 24h straight-line extrapolation")
st.caption(
    "We called the old section “toy forecast” internally — it is **only** a straight line fit to recent points, extended 24h. "
    "That shows the **shape** of the simplest possible continuation; it is **not** ocean forecasting. "
    "Rename in UI reflects what it actually does."
)
fc_col = st.selectbox("Series", feats, index=0, format_func=lambda c: friendly_axis_label(c))
sub = df[["time", fc_col]].dropna().copy()
sub["time"] = pd.to_datetime(sub["time"], utc=True, errors="coerce")
sub = sub.dropna(subset=["time"]).sort_values("time")
if len(sub) >= 48:
    tail = sub.tail(168).copy()
    tnum = (tail["time"] - tail["time"].iloc[0]).dt.total_seconds().to_numpy()
    yv = tail[fc_col].to_numpy(dtype=float)
    m, b = np.polyfit(tnum, yv, 1)
    t_last = tnum[-1]
    tf = np.linspace(t_last, t_last + 24 * 3600, 25)
    yf = m * tf + b
    tf_dt = tail["time"].iloc[0] + pd.to_timedelta(tf, unit="s")
    fh = go.Figure()
    fh.add_trace(go.Scatter(x=tail["time"], y=yv, name="history", mode="lines"))
    fh.add_trace(go.Scatter(x=tf_dt, y=yf, name="24h straight-line", mode="lines", line=dict(dash="dash")))
    fh.update_layout(height=CHART_H_HALF, title=f"Naive extrapolation — {friendly_axis_label(fc_col)}")
    st.plotly_chart(apply_plotly(fh), use_container_width=True)
else:
    st.caption("Need at least **48** samples with valid time + series values to draw the ribbon.")
