import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="California Ocean Health Monitor",
    page_icon="🌊",
    layout="wide"
)

# ─── DB ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "ocean_health"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )

@st.cache_data(ttl=300)
def query(sql):
    with get_conn().cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall())

# ─── CONSTANTS ──────────────────────────────────────────────────────
ANCHOVY_THRESHOLD  = 7.9
SHELLFISH_THRESHOLD = 7.75
COLORS = {"CCE1": "#1f77b4", "CCE2": "#ff7f0e"}

# ─── DATA ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_ph():
    df = query("SELECT date, station, ph FROM mooring_master WHERE ph IS NOT NULL ORDER BY date")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_larvae():
    df = query("SELECT date, scientific_name, larvae_10m2 FROM calcofi_larvae ORDER BY date")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_zoo():
    df = query("SELECT date, total_plankton FROM calcofi_zooplankton ORDER BY date")
    df["date"] = pd.to_datetime(df["date"])
    return df

# ─── HEADER ─────────────────────────────────────────────────────────
st.title("🌊 California Ocean Health Monitor")
st.caption(
    "Ocean acidification tracking for California coastal resource managers · "
    "CCE Mooring Network + CalCOFI · Powered by AI forecasting"
)

# ─── SIDEBAR ────────────────────────────────────────────────────────
ph_df = load_ph()

with st.sidebar:
    st.header("Filters")
    selected_station = st.selectbox("Station", ["Both", "CCE1", "CCE2"])

    min_date = ph_df["date"].min().date()
    max_date = ph_df["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    st.divider()
    st.markdown("**Biological Thresholds**")
    st.markdown(f"🟡 **pH {ANCHOVY_THRESHOLD}** — Anchovy larvae survival collapses")
    st.markdown(f"🔴 **pH {SHELLFISH_THRESHOLD}** — Shellfish cannot form shells")
    st.divider()
    st.markdown("**Stations**")
    st.markdown("📍 **CCE1** — 33.4°N, 122.5°W (offshore Point Conception)")
    st.markdown("📍 **CCE2** — Southern California Bight")

# ─── FILTER ─────────────────────────────────────────────────────────
if len(date_range) == 2:
    ph_f = ph_df[
        (ph_df["date"] >= pd.Timestamp(date_range[0])) &
        (ph_df["date"] <= pd.Timestamp(date_range[1]))
    ]
else:
    ph_f = ph_df

if selected_station != "Both":
    ph_f = ph_f[ph_f["station"] == selected_station]

# ─── TABS ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 pH Trends & Thresholds",
    "🔮 AI Threshold Forecast",
    "🐟 Ecosystem Impact",
    "📊 Station Comparison"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: pH TRENDS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ocean pH Over Time")
    st.caption("Dashed lines mark biological danger thresholds for California fisheries")

    fig = go.Figure()
    for station, color in COLORS.items():
        if selected_station != "Both" and station != selected_station:
            continue
        s = ph_f[ph_f["station"] == station].sort_values("date").copy()
        if s.empty:
            continue
        s["smooth"] = s["ph"].rolling(30, min_periods=5).mean()
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["ph"], mode="lines",
            name=f"{station} daily", line=dict(color=color, width=1), opacity=0.25
        ))
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["smooth"], mode="lines",
            name=f"{station} 30-day avg", line=dict(color=color, width=2.5)
        ))

    fig.add_hline(y=ANCHOVY_THRESHOLD, line_dash="dash", line_color="orange",
                  annotation_text="Anchovy threshold (7.9)", annotation_position="bottom right")
    fig.add_hline(y=SHELLFISH_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="Shellfish threshold (7.75)", annotation_position="bottom right")
    fig.update_layout(height=450, yaxis_title="pH", xaxis_title="Date",
                      yaxis=dict(range=[7.55, 8.35]), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    cols = st.columns(4)
    col_idx = 0
    for station in ["CCE1", "CCE2"]:
        if selected_station != "Both" and station != selected_station:
            continue
        s = ph_f[ph_f["station"] == station].sort_values("date")
        if s.empty:
            continue
        current = s["ph"].iloc[-1]
        start   = s["ph"].iloc[0]
        below_anchovy   = (s["ph"] < ANCHOVY_THRESHOLD).sum()
        below_shellfish = (s["ph"] < SHELLFISH_THRESHOLD).sum()
        cols[col_idx].metric(f"{station} Latest pH", f"{current:.3f}",
                             delta=f"{current - start:+.3f} since start", delta_color="inverse")
        cols[col_idx + 1].metric(f"{station} Days Below Anchovy Threshold", int(below_anchovy))
        col_idx += 2

# ══════════════════════════════════════════════════════════════════════
# TAB 2: FORECAST
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("AI Threshold Crossing Forecast")
    st.caption("Prophet time-series model forecasting when each station crosses biological danger thresholds")

    horizon_years = st.slider("Forecast horizon (years)", 1, 10, 5)

    def run_forecast(station_df, years):
        df_p = (station_df[["date", "ph"]]
                .rename(columns={"date": "ds", "ph": "y"})
                .dropna())
        df_p["ds"] = pd.to_datetime(df_p["ds"])
        monthly = df_p.set_index("ds").resample("ME").mean().reset_index().dropna()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(monthly)
        future = m.make_future_dataframe(periods=years * 365, freq="D")
        return m.predict(future), monthly

    def crossing_date(forecast, threshold):
        future = forecast[forecast["ds"] > pd.Timestamp.now()]
        hits = future[future["yhat"] <= threshold]
        return hits["ds"].iloc[0] if not hits.empty else None

    for station, color in COLORS.items():
        if selected_station != "Both" and station != selected_station:
            continue
        s = ph_df[ph_df["station"] == station].sort_values("date")
        if len(s) < 24:
            st.warning(f"{station}: insufficient data to forecast")
            continue

        with st.spinner(f"Forecasting {station}..."):
            forecast, hist = run_forecast(s, horizon_years)

        anchovy_cross   = crossing_date(forecast, ANCHOVY_THRESHOLD)
        shellfish_cross = crossing_date(forecast, SHELLFISH_THRESHOLD)

        st.markdown(f"### Station {station}")
        c1, c2 = st.columns(2)
        c1.metric(
            "🟡 Anchovy Threshold (pH 7.9)",
            anchovy_cross.strftime("%b %Y") if anchovy_cross else "Beyond forecast window",
            delta=(f"in {(anchovy_cross - pd.Timestamp.now()).days // 30} months"
                   if anchovy_cross else None),
            delta_color="inverse"
        )
        c2.metric(
            "🔴 Shellfish Threshold (pH 7.75)",
            shellfish_cross.strftime("%b %Y") if shellfish_cross else "Beyond forecast window",
            delta=(f"in {(shellfish_cross - pd.Timestamp.now()).days // 30} months"
                   if shellfish_cross else None),
            delta_color="inverse"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="markers",
                                 name="Monthly observed",
                                 marker=dict(color=color, size=5)))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                 mode="lines", name="Forecast",
                                 line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself", fillcolor=color, opacity=0.12,
            line=dict(color="rgba(0,0,0,0)"), name="95% confidence"
        ))
        fig.add_hline(y=ANCHOVY_THRESHOLD, line_dash="dash", line_color="orange",
                      annotation_text="Anchovy (7.9)")
        fig.add_hline(y=SHELLFISH_THRESHOLD, line_dash="dash", line_color="red",
                      annotation_text="Shellfish (7.75)")
        if anchovy_cross:
            fig.add_vline(x=anchovy_cross, line_dash="dot", line_color="orange")
        if shellfish_cross:
            fig.add_vline(x=shellfish_cross, line_dash="dot", line_color="red")

        fig.update_layout(height=420, yaxis_title="pH", xaxis_title="Date",
                          yaxis=dict(range=[7.4, 8.5]), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ECOSYSTEM IMPACT
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ocean Chemistry vs. Biological Health")
    st.caption("Connecting mooring pH data with CalCOFI fish larvae surveys (1951–2023)")

    larvae_df = load_larvae()
    zoo_df    = load_zoo()

    anchovy = larvae_df[larvae_df["scientific_name"] == "Engraulis mordax"].copy()
    sardine = larvae_df[larvae_df["scientific_name"] == "Sardinops sagax"].copy()

    # Annual aggregates
    ph_ann = ph_df.copy()
    ph_ann["year"] = ph_ann["date"].dt.year
    ph_ann = ph_ann.groupby(["year", "station"])["ph"].mean().reset_index()
    ph_cce1 = ph_ann[ph_ann["station"] == "CCE1"]

    for df in [anchovy, sardine, zoo_df]:
        df["year"] = df["date"].dt.year

    anch_ann = anchovy.groupby("year")["larvae_10m2"].mean().reset_index()
    sard_ann = sardine.groupby("year")["larvae_10m2"].mean().reset_index()
    zoo_ann  = zoo_df.groupby("year")["total_plankton"].mean().reset_index()

    merged = anch_ann.merge(ph_cce1, on="year", how="inner")

    if not merged.empty:
        st.markdown("#### Annual pH (CCE1) vs. Anchovy Larvae Density")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["year"], y=merged["ph"],
                                 mode="lines+markers", name="pH (CCE1)",
                                 line=dict(color="steelblue", width=2), yaxis="y1"))
        fig.add_trace(go.Scatter(x=merged["year"], y=merged["larvae_10m2"],
                                 mode="lines+markers", name="Anchovy larvae / 10m²",
                                 line=dict(color="darkorange", width=2), yaxis="y2"))
        fig.add_hline(y=ANCHOVY_THRESHOLD, line_dash="dash", line_color="orange",
                      annotation_text="Anchovy threshold", yref="y1")
        fig.update_layout(
            height=430, hovermode="x unified",
            yaxis=dict(title="pH (CCE1)", color="steelblue",
                       side="left", range=[7.8, 8.3]),
            yaxis2=dict(title="Larvae / 10m²", color="darkorange",
                        side="right", overlaying="y"),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### pH vs. Anchovy Larvae Density (Scatter + Trend)")
        fig2 = px.scatter(merged, x="ph", y="larvae_10m2", trendline="ols",
                          hover_data=["year"], color_discrete_sequence=["darkorange"],
                          labels={"ph": "Annual Mean pH (CCE1)",
                                  "larvae_10m2": "Anchovy Larvae / 10m²"})
        fig2.add_vline(x=ANCHOVY_THRESHOLD, line_dash="dash", line_color="orange",
                       annotation_text="Anchovy threshold (7.9)")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Zooplankton Abundance Over Time")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=zoo_ann["year"], y=zoo_ann["total_plankton"],
                              mode="lines+markers", fill="tozeroy",
                              name="Total Plankton", line=dict(color="teal", width=2)))
    fig3.update_layout(height=320, yaxis_title="Mean Plankton Abundance",
                       xaxis_title="Year", hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4: STATION COMPARISON
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Station-by-Station Acidification Rate")
    st.caption("Linear regression over full pH record — which zone is degrading fastest?")

    results = []
    for station in ["CCE1", "CCE2"]:
        s = ph_df[ph_df["station"] == station].sort_values("date").dropna(subset=["ph"])
        if len(s) < 10:
            continue
        s = s.copy()
        s["t"] = (s["date"] - s["date"].min()).dt.days
        reg = LinearRegression().fit(s[["t"]], s["ph"])
        results.append({
            "Station": station,
            "Mean pH": round(s["ph"].mean(), 4),
            "Min pH Recorded": round(s["ph"].min(), 4),
            "pH Change / Decade": round(reg.coef_[0] * 3650, 4),
            "Days of Data": len(s),
        })

    if results:
        res_df = pd.DataFrame(results)

        c1, c2 = st.columns(2)
        fig_rate = px.bar(res_df, x="Station", y="pH Change / Decade",
                          color="pH Change / Decade",
                          color_continuous_scale="RdYlGn_r",
                          title="Acidification Rate (pH units per decade)")
        fig_rate.add_hline(y=0, line_color="black", line_width=1)
        fig_rate.update_layout(height=380)
        c1.plotly_chart(fig_rate, use_container_width=True)

        fig_mean = px.bar(res_df, x="Station", y="Mean pH",
                          color="Mean pH", color_continuous_scale="RdYlGn",
                          title="Mean pH by Station",
                          range_y=[7.6, 8.3])
        fig_mean.add_hline(y=ANCHOVY_THRESHOLD, line_dash="dash",
                           line_color="orange", annotation_text="Anchovy threshold")
        fig_mean.add_hline(y=SHELLFISH_THRESHOLD, line_dash="dash",
                           line_color="red", annotation_text="Shellfish threshold")
        fig_mean.update_layout(height=380)
        c2.plotly_chart(fig_mean, use_container_width=True)

        st.dataframe(res_df.set_index("Station"), use_container_width=True)
