"""Lagged cross-correlation, Granger tests, rolling correlations, T-S diagram, O2 proxy / ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def o2_saturation_umolkg(temp_c: np.ndarray, sal_psu: np.ndarray) -> np.ndarray:
    """
    Garcia & Gordon (1992) / combined fit for O2 solubility in seawater (µmol/kg), 1 atm air.
    Valid approx. 0–40 °C, 0–40 PSU. Not measured O2 — saturation reference for ratios.
    """
    T = np.asarray(temp_c, dtype=float)
    S = np.asarray(sal_psu, dtype=float)
    Ts = np.log((298.15 - T) / (273.15 + T))
    A0 = 2.00907
    A1 = 3.33914
    A2 = 4.91487
    A3 = 4.84255
    A4 = -9.9248e-2
    A5 = 3.7867e-3
    A6 = -4.8833e-5
    B0 = -6.24523e-3
    B1 = -7.08614e-3
    B2 = -1.0371e-2
    B3 = -1.16866e-2
    lnC = (
        A0
        + A1 * Ts
        + A2 * Ts**2
        + A3 * Ts**3
        + A4 * Ts**4
        + A5 * Ts**5
        + A6 * Ts**6
        + S * (B0 + B1 * Ts + B2 * Ts**2 + B3 * Ts**3)
    )
    return np.exp(lnC) * 1000.0 / 22.391  # µmol/kg approx scaling path from mL/L literature fits


def lagged_cross_correlation(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_lag_hours: int = 30 * 24,
    step_hours: int = 24,
) -> pd.DataFrame | None:
    """Pearson r at integer day lags on daily means (B leads A by lag when lag>0)."""
    if col_a not in df.columns or col_b not in df.columns:
        return None
    d = df[["time", col_a, col_b]].dropna().sort_values("time")
    if len(d) < 60:
        return None
    a = d.set_index("time")[col_a].resample("D").mean()
    b = d.set_index("time")[col_b].resample("D").mean()
    j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    if len(j) < 60:
        return None
    aa = j["a"].to_numpy()
    bb = j["b"].to_numpy()
    lags = list(range(-max_lag_hours // 24, max_lag_hours // 24 + 1, max(1, step_hours // 24)))
    rs = []
    for L in lags:
        if L == 0:
            xa, xb = aa, bb
        elif L > 0:
            xa, xb = aa[L:], bb[:-L]
        else:
            k = -L
            xa, xb = aa[:-k], bb[k:]
        if len(xa) < 30:
            rs.append(np.nan)
            continue
        rs.append(float(np.corrcoef(xa, xb)[0, 1]))
    return pd.DataFrame({"lag_days": lags, "pearson_r": rs})


def granger_matrix(
    df: pd.DataFrame,
    cols: list[str],
    maxlag: int = 7,
    daily: bool = True,
) -> pd.DataFrame | None:
    """Pairwise Granger p-value: entry (row=target, col=predictor) = min p-value over lags."""
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return None
    d = df[["time"] + use].dropna(subset=["time"]).sort_values("time").set_index("time")
    if daily:
        parts = [d[c].resample("D").mean().rename(c) for c in use]
        d = pd.concat(parts, axis=1).interpolate(limit_direction="both", limit=4000).ffill().bfill()
    d = d[use].dropna()
    if len(d) < 3 * maxlag + 20:
        return None
    n = len(use)
    pmat = np.full((n, n), np.nan)
    np.fill_diagonal(pmat, 0.0)
    for i, ci in enumerate(use):
        for j, cj in enumerate(use):
            if i == j:
                continue
            try:
                # [Y, X]: test whether X (ci) Granger-causes Y (cj); store at row=cj, col=ci
                g = grangercausalitytests(d[[cj, ci]], maxlag=maxlag, verbose=False)
                pvs = [g[L][0]["ssr_ftest"][1] for L in range(1, maxlag + 1)]
                pmat[j, i] = float(np.nanmin(pvs))
            except Exception:
                pmat[j, i] = np.nan
    return pd.DataFrame(pmat, index=use, columns=use)


def rolling_correlation_vs_time(
    df: pd.DataFrame,
    ref_col: str,
    other_cols: list[str],
    window: str = "30D",
) -> pd.DataFrame | None:
    """Rolling Pearson r(ref, x) on hourly-interpolated series."""
    if ref_col not in df.columns:
        return None
    others = [c for c in other_cols if c in df.columns and c != ref_col]
    if not others:
        return None
    d = df[["time", ref_col] + others].dropna(subset=["time"]).sort_values("time")
    h = d.set_index("time").resample("h").mean().interpolate(limit_direction="both", limit=72)
    h = h.dropna(how="all", axis=1)
    if ref_col not in h.columns:
        return None
    minp = max(48, int(pd.Timedelta(window).total_seconds() // 3600 // 2))
    rows = []
    for c in others:
        if c not in h.columns:
            continue
        pair = h[[ref_col, c]].dropna()
        if len(pair) < minp * 2:
            continue
        rr = pair[ref_col].rolling(window, min_periods=minp).corr(pair[c])
        rows.append(rr.rename(c))
    if not rows:
        return None
    out = pd.concat(rows, axis=1)
    out.index.name = "time"
    return out.reset_index()


def ts_diagram_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Scatter-ready T–S with optional chlorophyll / nitrate for color."""
    if "sst_c" not in df.columns or "salinity_psu" not in df.columns:
        return None
    cols = ["time", "sst_c", "salinity_psu"]
    for c in ("chl_mg_m3", "no3"):
        if c in df.columns:
            cols.append(c)
    return df[cols].dropna(subset=["sst_c", "salinity_psu"])


def redfield_proxy_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """NO3 / O2_saturation (µmol/kg) as a crude anomaly proxy; NO3:Chl uptake ratio."""
    if "no3" not in df.columns:
        return None
    d = df[["time", "no3", "sst_c", "salinity_psu"]].dropna(subset=["time", "no3"]).copy()
    if "sst_c" not in d.columns or "salinity_psu" not in d.columns:
        d["o2_sat_umolkg"] = np.nan
    else:
        d["o2_sat_umolkg"] = o2_saturation_umolkg(d["sst_c"].to_numpy(), d["salinity_psu"].to_numpy())
    d["no3_over_o2sat"] = d["no3"] / (d["o2_sat_umolkg"] + 1e-6)
    if "chl_mg_m3" in df.columns:
        d["chl_mg_m3"] = df["chl_mg_m3"].reindex(d.index).values
        d["no3_chl_ratio"] = d["no3"] / (d["chl_mg_m3"] + 1e-6)
    return d
