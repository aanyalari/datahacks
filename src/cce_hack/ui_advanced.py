"""Advanced analysis sections for Streamlit (temporal, OA, coupling, ML, viz)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from cce_hack.acidification_co2sys import omega_profile_isochemical, ph_variability_index, run_co2sys_on_dataframe
from cce_hack.column_pick import pick_best_column
from cce_hack.cross_column import (
    granger_matrix,
    lagged_cross_correlation,
    redfield_proxy_frame,
    rolling_correlation_vs_time,
    ts_diagram_frame,
)
from cce_hack.dimred_cluster import run_hdbscan, run_kmeans, run_pca_biplot, run_umap_2d
from cce_hack.ml_extras import arima_daily_forecast, lstm_sequence_forecast, random_forest_with_shap, regime_classifier
from cce_hack.plot_theme import apply_plotly, plotly_theme_kwargs
from cce_hack.temporal_ops import anomaly_flags, rolling_stats, stl_decompose_daily
from cce_hack.viz_extras import hovmoller_sst_depth_time, normalize_rows_01, pairplot_frame, seasonal_radar_frame
from cce_hack.wavelet_ops import morlet_coherence


def _numeric(df: pd.DataFrame) -> list[str]:
    skip = {"mooring_id", "latitude", "longitude"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def render_temporal_tab(df: pd.DataFrame) -> None:
    st.caption("STL on daily means; rolling stats on hourly interpolation; anomaly flags on daily residuals.")
    num = _numeric(df)
    c1 = st.selectbox("STL / rolling / anomalies — column", num, key="stl_col")
    c1b, c2b = st.columns(2)
    with c1b:
        stl = stl_decompose_daily(df, c1, period_days=365)
        if stl is not None:
            sm = stl.melt(id_vars=["time"], var_name="component", value_name="value")
            st.plotly_chart(
                apply_plotly(px.line(sm, x="time", y="value", color="component")).update_layout(height=420, title=f"STL — {c1}"),
                use_container_width=True,
            )
        else:
            st.info("Not enough daily data for STL (need ~2 years). Try another column or shorter period.")
    with c2b:
        roll = rolling_stats(df, c1)
        if roll:
            wsel = st.selectbox("Rolling window", list(roll.keys()), key="rollwin")
            rw = roll[wsel]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=rw["time"], y=rw["mean"], name="mean"), secondary_y=False)
            fig.add_trace(go.Scatter(x=rw["time"], y=rw["std"], name="std", line=dict(dash="dot")), secondary_y=True)
            fig.update_layout(title=f"Rolling {wsel}", height=380, **{k: v for k, v in plotly_theme_kwargs().items() if k != "margin"})
            st.plotly_chart(apply_plotly(fig), use_container_width=True)
        else:
            st.info("Rolling statistics unavailable for this column.")

    an = anomaly_flags(df, c1)
    if an is not None:
        st.subheader("Anomaly flags (daily residual vs 30d rolling median)")
        st.plotly_chart(
            apply_plotly(
                px.scatter(an, x="time", y="residual_vs_rolling_median", color="flag_z", hover_data=["flag_iqr"])
            ).update_layout(height=400),
            use_container_width=True,
        )


def render_acidification_tab(df: pd.DataFrame) -> None:
    st.caption("PyCO2SYS with **assumed total alkalinity** (~2300 µmol/kg) — standard hack when TA is not measured. Adjust in sidebar if you add a control.")
    ta = st.number_input("Assumed TA (µmol/kg)", value=2300.0, min_value=1800.0, max_value=2800.0, step=10.0, key="ta_assumed")
    p_dbar = st.number_input("Sensor pressure (dbar)", value=40.0, min_value=0.0, max_value=500.0, step=5.0, key="p_dbar")

    co2 = run_co2sys_on_dataframe(df, ta_umolkg=ta, pressure_dbar=p_dbar)
    if co2 is None:
        st.warning("PyCO2SYS run failed (need pH + T + S overlap). Install: pip install PyCO2SYS")
        return
    st.success(f"Computed carbonate system on **{len(co2):,}** timestamps.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            apply_plotly(
                px.line(co2, x="time", y="saturation_aragonite", title="Ω aragonite")
            ).update_layout(height=380),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            apply_plotly(px.line(co2, x="time", y="revelle_factor", title="Revelle factor")).update_layout(height=380),
            use_container_width=True,
        )
    st.plotly_chart(
        apply_plotly(px.line(co2, x="time", y="pCO2_uatm", title="pCO₂ (wet, µatm)")).update_layout(height=340),
        use_container_width=True,
    )

    phv = ph_variability_index(df)
    if phv is not None:
        st.plotly_chart(
            apply_plotly(px.line(phv, x="time", y="daily_range", title="pH daily range (max−min)")).update_layout(height=320),
            use_container_width=True,
        )

    st.subheader("Isochemical Ω profile vs pressure (illustrative horizon)")
    ph_c = pick_best_column(df, "ph")
    t_c = pick_best_column(df, "sst")
    s_c = pick_best_column(df, "salinity")
    if ph_c and t_c and s_c and "time" in df.columns:
        parcel = df[["time", ph_c, t_c, s_c]].dropna().sort_values("time")
        if len(parcel) == 0:
            st.caption("No co-located pH + temperature + salinity rows for an illustrative depth profile.")
        else:
            mid = len(parcel) // 2
            row = parcel.iloc[mid]
            prof = omega_profile_isochemical(
                float(row[s_c]),
                float(row[t_c]),
                ta,
                float(row[ph_c]),
                surface_pressure_dbar=p_dbar,
            )
            if prof is not None:
                hz = prof.attrs.get("horizon_depth_m", float("nan"))
                st.caption(
                    f"One **mid-record** water parcel (row {mid + 1:,} of {len(parcel):,} co-located samples): "
                    f"shallowest depth where Ω_aragonite drops **below 1** ≈ **{hz:.1f} m** (hydrostatic P/1.02 dbar·m⁻¹)."
                )
                st.plotly_chart(
                    apply_plotly(
                        px.line(prof, x="depth_m_approx", y="saturation_aragonite", title="Ω aragonite vs depth")
                    ).update_layout(height=380),
                    use_container_width=True,
                )
    else:
        st.caption("Need overlapping **pH**, **temperature**, and **salinity** columns for the illustrative Ω vs depth curve.")


def render_cross_tab(df: pd.DataFrame) -> None:
    num = _numeric(df)
    a = st.selectbox("Series A (lagged XCF)", num, index=min(3, len(num) - 1), key="lcc_a")
    b = st.selectbox("Series B", num, index=min(5, len(num) - 1), key="lcc_b")
    lcc = lagged_cross_correlation(df, a, b)
    if lcc is not None:
        st.plotly_chart(
            apply_plotly(px.line(lcc, x="lag_days", y="pearson_r", title=f"Lagged correlation {a} vs {b}")).update_layout(height=380),
            use_container_width=True,
        )

    st.subheader("Granger causality matrix (daily; min p-value over lags)")
    gcols = st.multiselect("Variables for Granger", num, default=num[: min(5, len(num))], key="gr_cols")
    if len(gcols) >= 2:
        gm = granger_matrix(df, gcols, maxlag=7)
        if gm is not None:
            st.plotly_chart(
                apply_plotly(
                    px.imshow(gm, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis_r", zmin=0, zmax=1)
                ).update_layout(title="p-values (row = target, col = predictor)", height=480),
                use_container_width=True,
            )

    ref = st.selectbox("Rolling corr reference", num, key="rcref")
    rroll = rolling_correlation_vs_time(df, ref, num, window="30D")
    if rroll is not None:
        rm = rroll.set_index("time").drop(columns=[], errors="ignore")
        st.plotly_chart(
            apply_plotly(px.imshow(rm.T, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)).update_layout(
                title=f"Rolling r vs {ref} (30D)", height=420, yaxis_title="variable"
            ),
            use_container_width=True,
        )

    ts = ts_diagram_frame(df)
    if ts is not None:
        color = "chl_mg_m3" if "chl_mg_m3" in ts.columns else ("no3" if "no3" in ts.columns else None)
        fig = px.scatter(ts, x="salinity_psu", y="sst_c", color=color, hover_data=["time"] if "time" in ts.columns else None)
        st.plotly_chart(apply_plotly(fig).update_layout(title="T–S diagram", height=440), use_container_width=True)

    rf = redfield_proxy_frame(df)
    if rf is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                apply_plotly(px.line(rf, x="time", y="no3_over_o2sat", title="NO₃ / O₂ saturation (proxy)")).update_layout(height=340),
                use_container_width=True,
            )
        if "no3_chl_ratio" in rf.columns:
            with c2:
                st.plotly_chart(
                    apply_plotly(px.line(rf, x="time", y="no3_chl_ratio", title="NO₃ : Chl ratio")).update_layout(height=340),
                    use_container_width=True,
                )

    st.subheader("Wavelet coherence (Morlet, simplified)")
    w1 = st.selectbox("Wavelet series 1", num, key="w1")
    w2 = st.selectbox("Wavelet series 2", num, index=min(1, len(num) - 1), key="w2")
    d = df[["time", w1, w2]].dropna().sort_values("time")
    if len(d) > 2000:
        d = d.iloc[:: max(1, len(d) // 2000)]
    per, coh, _ = morlet_coherence(d[w1].to_numpy(), d[w2].to_numpy(), sampling_period=1.0)
    if per is not None:
        st.plotly_chart(
            apply_plotly(go.Figure(go.Scatter(x=per, y=coh, mode="lines"))).update_layout(
                title="Scale-averaged wavelet coherence", xaxis_title="Period (h)", height=360, **plotly_theme_kwargs()
            ),
            use_container_width=True,
        )


def render_dimred_tab(df: pd.DataFrame) -> None:
    num = _numeric(df)
    feats = st.multiselect("Features for PCA / clustering / UMAP", num, default=num[: min(8, len(num))], key="dr_feats")
    if len(feats) < 3:
        st.warning("Pick at least 3 features.")
        return
    pca = run_pca_biplot(df, feats, n_components=2)
    if pca:
        sc = pca["scores"]
        ld = pca["loadings"]
        st.caption("Explained variance: " + ", ".join(f"{v*100:.1f}%" for v in pca["explained_variance_ratio"]))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc["PC1"], y=sc["PC2"], mode="markers", marker=dict(size=5, opacity=0.35), name="scores"))
        for i, row in ld.iterrows():
            fig.add_trace(
                go.Scatter(x=[0, row["PC1"] * 3], y=[0, row["PC2"] * 3], mode="lines+markers+text", text=["", i], name=str(i))
            )
        st.plotly_chart(apply_plotly(fig).update_layout(title="PCA biplot (loadings scaled ×3)", height=500), use_container_width=True)

    k = st.slider("K-means clusters", 2, 8, 4)
    km = run_kmeans(df, feats, k=k)
    if km is not None:
        st.plotly_chart(
            apply_plotly(px.scatter(km, x="time", y="regime_kmeans", title="K-means regime over time")).update_layout(height=280),
            use_container_width=True,
        )

    hdb = run_hdbscan(df, feats, min_cluster_size=40)
    if hdb is not None:
        st.plotly_chart(
            apply_plotly(px.scatter(hdb, x="time", y="regime_hdbscan", title="HDBSCAN clusters (−1 = noise)")).update_layout(height=260),
            use_container_width=True,
        )

    um = run_umap_2d(df, feats)
    if um is not None:
        st.plotly_chart(
            apply_plotly(px.scatter(um, x="UMAP1", y="UMAP2", title="UMAP-2D state space")).update_layout(height=440),
            use_container_width=True,
        )

    if km is not None:
        st.subheader("Regime classifier (RF predicts K-means label)")
        dfc = df.merge(km, on="time", how="inner")
        rep = regime_classifier(dfc, feats, regime_col="regime_kmeans")
        if rep:
            st.text(rep["report"])
            st.metric("Holdout accuracy", f"{rep['accuracy']:.3f}")


def render_fancy_tab(df: pd.DataFrame) -> None:
    num = _numeric(df)
    hov = hovmoller_sst_depth_time(df)
    if hov is not None:
        st.plotly_chart(
            apply_plotly(px.imshow(hov.T, aspect="auto", labels=dict(x="time", y="depth", color="SST °C"))).update_layout(
                title="Hovmöller — multi-depth SST", height=420
            ),
            use_container_width=True,
        )

    radf = seasonal_radar_frame(df, num[: min(6, len(num))])
    if radf is not None:
        radn = normalize_rows_01(radf, [c for c in radf.columns if c != "season"])
        feats = [c for c in radn.columns if c != "season"]
        fig = go.Figure()
        for _, row in radn.iterrows():
            vals = [float(row[c]) for c in feats]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=feats + [feats[0]],
                    fill="toself",
                    name=str(row["season"]),
                )
            )
        st.plotly_chart(apply_plotly(fig).update_layout(title="Seasonal radar (min–max normalized)", height=480), use_container_width=True)

    ref = st.selectbox("Rolling heatmap ref", num, key="rhref")
    rroll = rolling_correlation_vs_time(df, ref, num, window="45D")
    if rroll is not None:
        mat = rroll.set_index("time").dropna(axis=0, how="all")
        st.plotly_chart(
            apply_plotly(px.imshow(mat.T, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)).update_layout(
                title=f"Rolling correlation with {ref}", height=440
            ),
            use_container_width=True,
        )

    km = run_kmeans(df, num[: min(8, len(num))], k=4)
    pp = pairplot_frame(df, num[: min(6, len(num))], regime_col=None)
    if pp is not None and km is not None:
        pp = pp.merge(km, on="time", how="inner")
    if pp is not None:
        cols = [c for c in pp.columns if c not in ("time", "regime_kmeans")]
        color = "regime_kmeans" if "regime_kmeans" in pp.columns else None
        st.plotly_chart(
            apply_plotly(px.scatter_matrix(pp, dimensions=cols, color=color, opacity=0.3)).update_layout(
                title="Pairplot (subset)", height=720
            ),
            use_container_width=True,
        )


def render_ml_advanced_tab(df: pd.DataFrame) -> None:
    num = _numeric(df)
    t_arima = st.selectbox("ARIMA target (daily)", [c for c in ("ph_total", "no3", "chl_mg_m3", "sst_c") if c in df.columns], key="arcol")
    ar = arima_daily_forecast(df, t_arima, order=(2, 1, 2), horizon_days=28)
    if ar:
        fh = go.Figure()
        fh.add_trace(go.Scatter(x=ar["history"].index, y=ar["history"].values, name="history"))
        fh.add_trace(go.Scatter(x=ar["forecast_index"], y=ar["forecast_mean"], name="forecast"))
        fh.add_trace(
            go.Scatter(
                x=np.concatenate([ar["forecast_index"], ar["forecast_index"][::-1]]),
                y=np.concatenate([ar["ci_high"], ar["ci_low"][::-1]]),
                fill="toself",
                name="95% CI",
                line=dict(color="rgba(122,168,255,0.2)"),
            )
        )
        st.plotly_chart(apply_plotly(fh).update_layout(title=f"SARIMAX/ARIMA — {t_arima} (AIC={ar['aic']:.0f})", height=420), use_container_width=True)
    else:
        st.info("ARIMA needs longer daily series.")

    st.subheader("Random Forest + SHAP (chlorophyll drivers)")
    if "chl_mg_m3" in df.columns:
        feats = [c for c in num if c != "chl_mg_m3"]
        with st.spinner("Fitting RF + SHAP…"):
            sh = random_forest_with_shap(df, "chl_mg_m3", feats[:15])
        if sh:
            st.metric("Train R² (subsample)", f"{sh['r2_train']:.3f}")
            imp = np.abs(sh["shap_values"]).mean(axis=0)
            imp_df = pd.DataFrame({"feature": sh["feature_names"], "mean(|SHAP|)": imp}).sort_values("mean(|SHAP|)", ascending=False).head(15)
            st.plotly_chart(apply_plotly(px.bar(imp_df, x="mean(|SHAP|)", y="feature", orientation="h")).update_layout(height=440), use_container_width=True)
    else:
        st.caption("No chlorophyll column for RF target.")

    st.subheader("LSTM next step (TensorFlow optional)")
    lstm_col = st.selectbox("LSTM series", [c for c in ("ph_total", "sst_c", "no3") if c in df.columns], key="lstm_c")
    if st.button("Train LSTM (~30–60s)", key="btn_lstm"):
        with st.spinner("Training LSTM…"):
            out = lstm_sequence_forecast(df, lstm_col)
        if out:
            st.success(f"Validation MAE (z-space): **{out['valid_mae_z']:.4f}**")
        else:
            st.info("Install TensorFlow: `pip install tensorflow` (or use CPU build).")
