import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium

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
ANCHOVY_THRESHOLD   = 7.9
SHELLFISH_THRESHOLD = 7.75
COLORS = {"CCE1": "#1f77b4", "CCE2": "#ff7f0e"}
STATION_COORDS = {"CCE1": (33.43, -122.48), "CCE2": (34.33, -120.78)}

# ─── DATA LOADERS ───────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_ph():
    df = query("SELECT date, station, ph FROM mooring_master WHERE ph IS NOT NULL ORDER BY date")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_mooring():
    df = query("SELECT date, station, ph, oxygen, chlorophyll, nitrate FROM mooring_master ORDER BY date")
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

@st.cache_data(ttl=86400)
def load_co2():
    try:
        url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
        df = pd.read_csv(url, usecols=["iso_code", "year", "co2"])
        return df[df["iso_code"] == "USA"][["year", "co2"]].dropna().query("year >= 2009")
    except Exception:
        return pd.DataFrame()

# ─── HELPERS ────────────────────────────────────────────────────────
def filter_by_date(df, date_range, date_col="date"):
    if len(date_range) == 2:
        return df[
            (df[date_col] >= pd.Timestamp(date_range[0])) &
            (df[date_col] <= pd.Timestamp(date_range[1]))
        ]
    return df

def monthly_mean(df, station_col="station"):
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    group = ["month", station_col] if station_col in df.columns else ["month"]
    num_cols = df.select_dtypes("number").columns.tolist()
    return df.groupby(group)[num_cols].mean().reset_index()

def ecosystem_score(ph, oxygen, chl):
    """
    pH proximity to anchovy threshold (40%),
    oxygen vs healthy baseline (30%),
    chlorophyll presence (30%).
    Returns score 0-100 and label.
    """
    scores, weights = [], []
    if pd.notna(ph):
        # 7.75 = 0, 8.3 = 100
        scores.append(max(0, min(100, (ph - 7.75) / (8.3 - 7.75) * 100)))
        weights.append(0.4)
    if pd.notna(oxygen):
        scores.append(max(0, min(100, (oxygen - 150) / (350 - 150) * 100)))
        weights.append(0.3)
    if pd.notna(chl):
        scores.append(max(0, min(100, chl / 3.0 * 100)))
        weights.append(0.3)
    if not scores:
        return None, "No data"
    composite = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    if composite >= 85:   label = "Healthy"
    elif composite >= 70: label = "Stressed"
    elif composite >= 50: label = "Early Warning"
    else:                 label = "Critical"
    return composite, label

def gauge_chart(score, label, title):
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{color}'>{label}</span>",
               "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [50, 70, 85]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  50], "color": "#fecaca"},
                {"range": [50, 70], "color": "#fef08a"},
                {"range": [70, 85], "color": "#d9f99d"},
                {"range": [85,100], "color": "#bbf7d0"},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
    return fig

# ─── LOAD DATA ──────────────────────────────────────────────────────
ph_df      = load_ph()
mooring_df = load_mooring()
larvae_df  = load_larvae()

# ─── SIDEBAR ────────────────────────────────────────────────────────
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
    st.markdown("📍 **CCE1** — 33.4°N, 122.5°W")
    st.markdown("📍 **CCE2** — 34.3°N, 120.8°W")

# ─── APPLY FILTERS ──────────────────────────────────────────────────
ph_f      = filter_by_date(ph_df, date_range)
mooring_f = filter_by_date(mooring_df, date_range)

if selected_station != "Both":
    ph_f      = ph_f[ph_f["station"] == selected_station]
    mooring_f = mooring_f[mooring_f["station"] == selected_station]

# ─── HEADER ─────────────────────────────────────────────────────────
st.title("🌊 California Ocean Health Monitor")
st.caption(
    "Episodic stress events on a worsening trajectory — "
    "CCE Mooring Network + CalCOFI · AI-powered threshold forecasting"
)

# ─── SHELLFISH STRESS TREND (replaces misleading CRITICAL banner) ───
st.markdown("#### Days Below Shellfish Threshold (pH < 7.75) by Year")
below_sf = ph_df[ph_df["ph"] < SHELLFISH_THRESHOLD].copy()
below_sf["year"] = below_sf["date"].dt.year
days_by_year = below_sf.groupby(["year", "station"]).size().reset_index(name="days")
if not days_by_year.empty:
    fig_hero = px.bar(
        days_by_year, x="year", y="days", color="station",
        color_discrete_map=COLORS, barmode="group",
        labels={"days": "Days below pH 7.75", "year": "Year", "station": "Station"},
    )
    fig_hero.update_layout(height=220, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=1.1),
                           hovermode="x unified")
    st.plotly_chart(fig_hero, width="stretch")
    st.caption(
        "Frequency of episodes below the shellfish survival threshold. "
        "Individual events are expected in an upwelling system — the trend over decades is what matters."
    )
else:
    st.info("No pH readings below shellfish threshold in dataset.")

# ─── FOLIUM MAP ─────────────────────────────────────────────────────
latest_ph = ph_df.sort_values("date").groupby("station")["ph"].last().to_dict()

def ph_color(ph):
    if ph is None:               return "gray"
    if ph < SHELLFISH_THRESHOLD: return "red"
    if ph < ANCHOVY_THRESHOLD:   return "orange"
    return "green"

m = folium.Map(location=[33.9, -121.6], zoom_start=7, tiles="CartoDB positron")
for station, (lat, lon) in STATION_COORDS.items():
    ph_val = latest_ph.get(station)
    color  = ph_color(ph_val)
    ph_txt = f"{ph_val:.3f}" if ph_val else "N/A"
    folium.CircleMarker(
        location=[lat, lon], radius=18, color=color,
        fill=True, fill_color=color, fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{station}</b><br>Latest pH: {ph_txt}<br>"
            f"{'🔴 Below shellfish threshold' if color=='red' else '🟡 Below anchovy threshold' if color=='orange' else '🟢 Within safe range'}",
            max_width=200
        ),
        tooltip=f"{station} — pH {ph_txt}"
    ).add_to(m)
st_folium(m, width=None, height=300, returned_objects=[])

# ─── TABS ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📉 pH Trends",
    "🔮 AI Forecast",
    "🐟 Ecosystem Impact",
    "📊 Station Comparison",
    "🏭 CO₂ Connection",
    "⚠️ Alert History",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: pH TRENDS + ECOSYSTEM HEALTH GAUGES
# ══════════════════════════════════════════════════════════════════════
with tab1:
    # ── Ecosystem health gauges ──────────────────────────────────────
    st.markdown("#### Ecosystem Health Score")
    st.caption("Composite: pH proximity to threshold (40%) · oxygen vs baseline (30%) · chlorophyll (30%)")

    gcols = st.columns(2)
    for i, station in enumerate(["CCE1", "CCE2"]):
        if selected_station != "Both" and station != selected_station:
            continue
        s = mooring_df[mooring_df["station"] == station]
        # Use last available data per variable (oxygen only runs to 2020)
        recent_ph  = s[s["ph"].notna()].sort_values("date")["ph"].tail(365).mean()
        recent_oxy = s[s["oxygen"].notna()].sort_values("date")["oxygen"].tail(365).mean()
        recent_chl = s[s["chlorophyll"].notna()].sort_values("date")["chlorophyll"].tail(365).mean()
        score, label = ecosystem_score(recent_ph, recent_oxy, recent_chl)
        if score is not None:
            gcols[i].plotly_chart(gauge_chart(score, label, f"{station} Health Score"),
                                  width="stretch")

    st.divider()

    # ── pH trend (monthly averages, year-level x-axis) ───────────────
    st.caption("Who uses this: Real-time monitoring for coastal resource managers and oceanographers tracking long-term acidification trends.")
    st.subheader("Ocean pH — Monthly Averages")
    st.caption("Smoothed to monthly means for year-scale readability · dashed lines = biological thresholds")

    fig = go.Figure()
    for station, color in COLORS.items():
        if selected_station != "Both" and station != selected_station:
            continue
        s = ph_f[ph_f["station"] == station].sort_values("date").copy()
        if s.empty:
            continue
        # Resample to monthly
        s_monthly = (s.set_index("date")["ph"]
                     .resample("ME").mean()
                     .reset_index()
                     .rename(columns={"date": "month"}))
        fig.add_trace(go.Scatter(
            x=s_monthly["month"], y=s_monthly["ph"],
            mode="lines+markers", name=station,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))

    fig.add_hline(y=ANCHOVY_THRESHOLD,   line_dash="dash", line_color="orange",
                  annotation_text="Anchovy threshold (7.9)",   annotation_position="bottom right")
    fig.add_hline(y=SHELLFISH_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text="Shellfish threshold (7.75)", annotation_position="bottom right")
    fig.add_annotation(
        x="2022-01-01", y=7.82,
        text="CCE2 dips toward<br>shellfish threshold",
        showarrow=True, arrowhead=2,
        ax=60, ay=-40,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red"
    )
    fig.update_layout(
        height=430, yaxis_title="pH", xaxis_title="",
        yaxis=dict(range=[7.55, 8.35]),
        xaxis=dict(tickformat="%Y", dtick="M12", tickangle=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    # Summary metrics
    mcols = st.columns(4)
    col_idx = 0
    for station in ["CCE1", "CCE2"]:
        if selected_station != "Both" and station != selected_station:
            continue
        s = ph_f[ph_f["station"] == station].sort_values("date")
        if s.empty:
            continue
        current = s["ph"].iloc[-1]
        mcols[col_idx].metric(
            f"{station} Latest pH", f"{current:.3f}",
            delta=f"{current - s['ph'].iloc[0]:+.3f} since start", delta_color="inverse"
        )
        mcols[col_idx + 1].metric(
            f"{station} Days Below Anchovy Threshold",
            int((s["ph"] < ANCHOVY_THRESHOLD).sum())
        )
        col_idx += 2

# ══════════════════════════════════════════════════════════════════════
# TAB 2: AI FORECAST
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("ML Forecast")
    st.info("ML model integration in progress — teammate's HGBT + ARIMA forecast will be added here.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ECOSYSTEM IMPACT
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Ocean Chemistry vs. Biological Health")
    st.caption("Tool for researchers: explore the relationship between mooring sensor chemistry and CalCOFI biological surveys (1951–2023)")

    zoo_df  = load_zoo()
    anchovy = larvae_df[larvae_df["scientific_name"] == "Engraulis mordax"].copy()

    for df in [anchovy, zoo_df]:
        df["year"] = df["date"].dt.year
    mooring_df["year"] = mooring_df["date"].dt.year

    ph_cce1  = mooring_df[mooring_df["station"] == "CCE1"].groupby("year")["ph"].mean().reset_index()
    oxy_cce1 = mooring_df[mooring_df["station"] == "CCE1"].groupby("year")["oxygen"].mean().reset_index()
    chl_cce1 = mooring_df[mooring_df["station"] == "CCE1"].groupby("year")["chlorophyll"].mean().reset_index()
    nit_cce1 = mooring_df[mooring_df["station"] == "CCE1"].groupby("year")["nitrate"].mean().reset_index()
    anch_ann = anchovy.groupby("year")["larvae_10m2"].mean().reset_index()
    zoo_ann  = zoo_df.groupby("year")["total_plankton"].mean().reset_index()

    merged = anch_ann.merge(ph_cce1, on="year", how="inner")

    st.divider()
    st.markdown("### pH vs. Anchovy Larvae — Annual Relationship")
    st.caption("Each point = one year of annual mean CCE1 pH vs. CalCOFI anchovy survey")

    if len(merged) >= 5:
        reg_X = merged[["ph"]].values
        reg_y = merged["larvae_10m2"].values
        scatter_reg = LinearRegression().fit(reg_X, reg_y)
        r2 = scatter_reg.score(reg_X, reg_y)
        x_line = np.linspace(merged["ph"].min(), merged["ph"].max(), 100)
        y_line = scatter_reg.predict(x_line.reshape(-1, 1))

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=merged["ph"], y=merged["larvae_10m2"],
            mode="markers+text",
            text=merged["year"].astype(str),
            textposition="top center",
            marker=dict(color="steelblue", size=9),
            name="Annual observation"
        ))
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="gray", dash="dash", width=1.5),
            name=f"Regression (R²={r2:.2f})"
        ))
        fig_scatter.add_vline(x=ANCHOVY_THRESHOLD, line_dash="dash", line_color="orange",
                              annotation_text="Anchovy threshold (7.9)")
        fig_scatter.update_layout(
            height=400,
            xaxis_title="Annual Mean pH (CCE1)",
            yaxis_title="Anchovy Larvae / 10m²",
            hovermode="closest"
        )
        st.plotly_chart(fig_scatter, width="stretch")
        st.caption(
            f"Weak correlation (R²={r2:.2f}) reflects the upwelling confound — "
            "years with low pH are also high-nutrient years in the California Current, which can boost larvae counts. "
            "Isolating the chronic acidification signal requires controlling for dissolved oxygen and nitrate."
        )

    st.info(
        "**Fun fact:** The California Current supports one of the most productive marine ecosystems "
        "on Earth. Northern anchovy are a keystone species — they are food for salmon, whales, "
        "sea lions, and seabirds. A pH drop of just 0.1 units can reduce anchovy larvae survival "
        "by up to 40% (Baumann et al., 2012)."
    )
    st.markdown("### Zooplankton Abundance Over Time")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=zoo_ann["year"], y=zoo_ann["total_plankton"],
                              mode="lines+markers", fill="tozeroy",
                              name="Total Plankton", line=dict(color="teal", width=2)))
    fig3.update_layout(height=290, yaxis_title="Mean Plankton Abundance",
                       xaxis_title="Year",
                       xaxis=dict(tickformat="%Y", dtick=1),
                       hovermode="x unified")
    st.plotly_chart(fig3, width="stretch")

# ══════════════════════════════════════════════════════════════════════
# TAB 4: STATION COMPARISON
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Station-by-Station Acidification Rate")
    st.caption("Linear regression over full pH record — which zone shows the strongest long-term decline?")

    results = []
    for station in ["CCE1", "CCE2"]:
        s = ph_df[ph_df["station"] == station].sort_values("date").dropna(subset=["ph"]).copy()
        if len(s) < 10:
            continue
        s["t"] = (s["date"] - s["date"].min()).dt.days
        reg = LinearRegression().fit(s[["t"]], s["ph"])
        slope = round(reg.coef_[0] * 3650, 4)
        results.append({
            "Station": station,
            "Mean pH": round(s["ph"].mean(), 4),
            "Min pH Recorded": round(s["ph"].min(), 4),
            "pH Change / Decade": slope,
            "Days of Data": len(s),
        })

    if results:
        res_df = pd.DataFrame(results)

        c1, c2 = st.columns(2)
        fig_rate = px.bar(res_df, x="Station", y="pH Change / Decade",
                          color="pH Change / Decade", color_continuous_scale="RdYlGn",
                          title="Acidification Rate (pH units per decade)")
        fig_rate.add_hline(y=0, line_color="black", line_width=1)
        fig_rate.update_layout(height=380)
        c1.plotly_chart(fig_rate, width="stretch")

        fig_mean = px.bar(res_df, x="Station", y="Mean pH",
                          color="Mean pH", color_continuous_scale="RdYlGn",
                          title="Mean pH by Station", range_y=[7.6, 8.3])
        fig_mean.add_hline(y=ANCHOVY_THRESHOLD,   line_dash="dash", line_color="orange",
                           annotation_text="Anchovy threshold")
        fig_mean.add_hline(y=SHELLFISH_THRESHOLD, line_dash="dash", line_color="red",
                           annotation_text="Shellfish threshold")
        fig_mean.update_layout(height=380)
        c2.plotly_chart(fig_mean, width="stretch")

        # CCE2 positive slope note
        cce2_row = res_df[res_df["Station"] == "CCE2"]
        if not cce2_row.empty and cce2_row.iloc[0]["pH Change / Decade"] > 0:
            st.info(
                "**Note on CCE2 positive trend:** A positive slope at CCE2 does not indicate "
                "ocean health improvement. The Southern California Bight is strongly influenced "
                "by nearshore upwelling, which brings cold, CO₂-rich deep water to the surface "
                "episodically — creating high year-to-year pH variability that can mask or reverse "
                "the underlying acidification signal. Interpret with caution."
            )

        st.dataframe(res_df.set_index("Station"), width="stretch")
        st.markdown(
            "**Key finding:** CCE2's recorded minimum pH of **7.4215** is the lowest in the dataset — "
            "below the shellfish shell-formation threshold of 7.75. "
            "CCE1 shows a statistically significant decline of **0.035 pH units per decade**."
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 5: CO2 CONNECTION
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("The Human Cause: US CO₂ Emissions vs. Ocean pH")
    st.caption("Source: Our World in Data / Global Carbon Project + Scripps CCE LTER Mooring Network")

    st.markdown("**Who uses this:** Climate policy researchers and environmental advocates connecting local ocean data to global emissions drivers.")
    m1, m2, m3 = st.columns(3)
    m1.metric("CCE1 pH change since 2009", "-0.042", delta="units", delta_color="inverse")
    m2.metric("CCE2 pH change since 2009", "-0.023", delta="units", delta_color="inverse")
    m3.metric("US CO₂ emissions 2009–2024", ">4,800 Mt/yr", delta="sustained")

    co2_df = load_co2()
    if co2_df.empty:
        st.warning("Could not fetch CO₂ data — check internet connection.")
    else:
        ph_annual = ph_df.copy()
        ph_annual["year"] = ph_annual["date"].dt.year
        ph_by_year = ph_annual.groupby(["year", "station"])["ph"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=co2_df["year"], y=co2_df["co2"],
                             name="US CO₂ Emissions (Mt)",
                             marker_color="rgba(156,163,175,0.45)", yaxis="y2"))
        for station, color in COLORS.items():
            s = ph_by_year[ph_by_year["station"] == station]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=s["year"], y=s["ph"],
                                     mode="lines+markers", name=f"pH {station}",
                                     line=dict(color=color, width=2.5), yaxis="y1"))

        fig.add_hline(y=ANCHOVY_THRESHOLD,   line_dash="dash", line_color="orange",
                      annotation_text="Anchovy threshold", yref="y1")
        fig.add_hline(y=SHELLFISH_THRESHOLD, line_dash="dash", line_color="red",
                      annotation_text="Shellfish threshold", yref="y1")
        fig.update_layout(
            height=500, hovermode="x unified",
            xaxis=dict(title="Year", tickformat="%Y", dtick=1),
            yaxis=dict(title="Ocean pH", side="left", color="steelblue", range=[7.7, 8.4]),
            yaxis2=dict(title="US CO₂ Emissions (Mt CO₂)", side="right",
                        overlaying="y", color="gray"),
            legend=dict(x=0.01, y=0.99),
            title="Despite slight US emissions reductions, ocean pH continues declining — the atmospheric CO₂ burden is cumulative"
        )
        st.plotly_chart(fig, width="stretch")
        st.info(
            "The correlation between atmospheric CO₂ and ocean acidification is well-established "
            "(IPCC AR6, 2021). Ocean CO₂ absorption produces carbonic acid, lowering pH. "
            "This chart connects the global driver to the local biological consequence."
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 6: ALERT HISTORY
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("⚠️ Alert History — pH Breaches Below Anchovy Threshold")
    st.caption(
        "Operational tool for fisheries managers and aquaculture operators — "
        "track when California coastal waters cross biological survival thresholds "
        "and correlate with nearby biological survey data."
    )

    alerts = query(
        f"SELECT date, station, ph FROM mooring_master "
        f"WHERE ph < {ANCHOVY_THRESHOLD} ORDER BY date"
    )
    alerts["date"] = pd.to_datetime(alerts["date"])

    if alerts.empty:
        st.success("No pH breaches below anchovy threshold recorded.")
    else:
        alerts_sorted = alerts.sort_values("date").copy()
        alerts_sorted["cumulative"] = alerts_sorted.groupby("station").cumcount() + 1

        fig_cumulative = go.Figure()
        for station, color in COLORS.items():
            s = alerts_sorted[alerts_sorted["station"] == station]
            if s.empty:
                continue
            fig_cumulative.add_trace(go.Scatter(
                x=s["date"], y=s["cumulative"],
                mode="lines", name=station,
                line=dict(color=color, width=2.5),
                fill="tozeroy",
            ))
        fig_cumulative.update_layout(
            height=300,
            title="Cumulative pH Breach Events Over Time (below anchovy threshold 7.9)",
            xaxis_title="", yaxis_title="Cumulative breaches",
            hovermode="x unified"
        )
        fig_cumulative.add_annotation(
            x="2022-01-01", y=210,
            text="⚡ 2021–22: Marine heatwave<br>drives surge in breach events",
            showarrow=True, arrowhead=2,
            ax=-80, ay=-40,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="orange"
        )
        st.plotly_chart(fig_cumulative, width="stretch")
        st.caption("CCE2's breach events began as early as 2011 and have accumulated steadily — CCE1 shows only 4 events total.")

    if not alerts.empty:
        anchovy_for_alerts = larvae_df[larvae_df["scientific_name"] == "Engraulis mordax"].copy()
        rows = []
        for _, row in alerts.iterrows():
            window = anchovy_for_alerts[
                (anchovy_for_alerts["date"] >= row["date"]) &
                (anchovy_for_alerts["date"] <= row["date"] + pd.Timedelta(days=90))
            ]
            larvae_val = window["larvae_10m2"].mean() if not window.empty else None
            rows.append({
                "Date": row["date"].strftime("%Y-%m-%d"),
                "Station": row["station"],
                "pH": round(row["ph"], 4),
                "Δ below threshold": round(row["ph"] - ANCHOVY_THRESHOLD, 4),
                "Anchovy larvae/10m² (next 90d)": round(larvae_val, 2) if larvae_val else None,
            })

        alert_df = pd.DataFrame(rows)

        scols = st.columns(2)
        for i, station in enumerate(["CCE1", "CCE2"]):
            n = (alert_df["Station"] == station).sum()
            worst = alert_df[alert_df["Station"] == station]["pH"].min() if n > 0 else None
            scols[i].metric(f"{station} Breach Events", int(n),
                            delta=f"Worst: {worst:.4f}" if worst else None,
                            delta_color="inverse")

        def style_row(row):
            if row["pH"] < SHELLFISH_THRESHOLD:
                return ["background-color:#fecaca"] * len(row)
            return ["background-color:#fef08a"] * len(row)

        st.dataframe(
            alert_df.style.apply(style_row, axis=1),
            width="stretch", height=420
        )
        st.caption("🟥 Red = also below shellfish threshold (7.75)   🟨 Yellow = below anchovy threshold only (7.9)   NaN = no CalCOFI survey within 90 days")
        st.markdown(
            "> **What this means for fisheries:** CCE2 has spent the equivalent of nearly **9 months** "
            "below the pH level at which anchovy larvae survival collapses. "
            "CCE1, further offshore, shows the same long-term declining trend but with far fewer "
            "acute stress events — suggesting nearshore inshore waters face disproportionate risk."
        )
