"""Hovmöller-style panels, radar charts, rolling correlation heatmaps, pair grids."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hovmoller_sst_depth_time(df: pd.DataFrame, max_time_points: int = 400) -> pd.DataFrame | None:
    """Time × depth matrix using multi-depth SST columns (no vertical interpolation)."""
    pairs = [(32, "sst_c_d32m"), (38, "sst_c_d38m"), (39, "sst_c_d39m")]
    cols = [c for _, c in pairs if c in df.columns]
    if len(cols) < 2:
        return None
    m = df[["time"] + cols].dropna(how="all", subset=cols).sort_values("time")
    if m.empty:
        return None
    step = max(1, len(m) // max_time_points)
    m = m.iloc[::step].set_index("time")
    rename = {c: f"{d} m" for d, c in pairs if c in m.columns}
    return m.rename(columns=rename)


def seasonal_radar_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame | None:
    """Mean per meteorological season for radar / spider chart."""
    use = [c for c in feature_cols if c in df.columns]
    if len(use) < 3:
        return None
    d = df[["time"] + use].dropna(subset=["time"]).copy()
    m = d["time"].dt.month
    d["season"] = np.select(
        [np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Fall",
    )
    g = d.groupby("season", as_index=False)[use].mean()
    return g


def normalize_rows_01(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = g.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lo, hi = out[c].min(), out[c].max()
        out[c] = (out[c] - lo) / (hi - lo + 1e-12) if hi > lo else 0.5
    return out


def pairplot_frame(df: pd.DataFrame, cols: list[str], regime_col: str | None = None) -> pd.DataFrame | None:
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return None
    extra = ["time"] + ([regime_col] if regime_col and regime_col in df.columns else [])
    return df[use + extra].dropna(subset=use[:2])
