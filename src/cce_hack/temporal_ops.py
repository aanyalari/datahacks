"""STL decomposition, rolling statistics, and anomaly flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def _daily_series(df: pd.DataFrame, col: str) -> pd.Series:
    d = df[["time", col]].dropna().sort_values("time")
    if d.empty:
        return pd.Series(dtype=float)
    s = d.set_index("time")[col].resample("D").mean()
    return s.interpolate(limit_direction="both")


def stl_decompose_daily(df: pd.DataFrame, col: str, period_days: int = 365) -> pd.DataFrame | None:
    """STL on daily means. Requires >= 2 full seasonal cycles."""
    s = _daily_series(df, col)
    if len(s) < 2 * period_days + 10:
        return None
    seasonal = min(91, max(7, (len(s) // 12) | 1))
    if seasonal % 2 == 0:
        seasonal += 1
    try:
        res = STL(s.astype(float), period=period_days, seasonal=seasonal, robust=True).fit()
    except Exception:
        return None
    out = pd.DataFrame(
        {"trend": res.trend, "seasonal": res.seasonal, "resid": res.resid},
        index=s.index,
    )
    out.index.name = "time"
    return out.reset_index()


def rolling_stats(
    df: pd.DataFrame,
    col: str,
    windows: tuple[str, ...] = ("7D", "30D", "90D"),
) -> dict[str, pd.DataFrame] | None:
    """Rolling mean, std, min, max on hourly-interpolated series; one DataFrame per window label."""
    if col not in df.columns:
        return None
    d = df[["time", col]].dropna().sort_values("time").set_index("time")
    h = d.resample("h").mean().interpolate(limit_direction="both", limit=48)
    if h.empty:
        return None
    out: dict[str, pd.DataFrame] = {}
    for w in windows:
        minp = max(24, int(pd.Timedelta(w).total_seconds() // 3600 // 6))
        g = h[col].rolling(w, min_periods=minp)
        out[w] = pd.DataFrame(
            {
                "time": h.index,
                "mean": g.mean().to_numpy(),
                "std": g.std().to_numpy(),
                "min": g.min().to_numpy(),
                "max": g.max().to_numpy(),
            }
        )
    return out


def anomaly_flags(
    df: pd.DataFrame,
    col: str,
    z_thresh: float = 3.0,
    iqr_mult: float = 1.5,
) -> pd.DataFrame | None:
    """Z-score and Tukey IQR flags on daily residuals from a 30-day rolling median."""
    s = _daily_series(df, col)
    if len(s) < 60:
        return None
    roll_med = s.rolling("30D", min_periods=15).median()
    resid = s - roll_med
    z = (resid - resid.mean()) / (resid.std(ddof=1) + 1e-9)
    q1, q3 = resid.quantile(0.25), resid.quantile(0.75)
    iqr = q3 - q1 + 1e-12
    low, high = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    zv = z.to_numpy(dtype=float)
    rv = resid.to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "time": s.index.to_numpy(),
            "value": s.to_numpy(dtype=float),
            "residual_vs_rolling_median": rv,
            "z_score": zv,
            "flag_z": (np.abs(zv) > z_thresh),
            "flag_iqr": (rv < float(low)) | (rv > float(high)),
        }
    )
    return out
