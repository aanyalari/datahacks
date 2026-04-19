"""Simple deterministic risk composites (no ML)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cce_hack.column_pick import pick_best_column
from cce_hack.mission_alerts import pick_o2_column


def _time_sorted_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", col]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["time", col]).sort_values("time")
    if len(d) < 20:
        return None
    return d.set_index("time")[col]


def _z_tail_vs_window(df: pd.DataFrame, col: str | None, *, tail_days: float = 5.0) -> float:
    """
    How different is the **tail** (last ``tail_days``) from the **whole current dataframe**?

    Uses mean/std of **all** points in the sidebar-filtered window. Changing **Time window**
    (30d / 90d / all) changes that baseline, so the z-score and the 0–100 index move with the filter.
    """
    if not col:
        return 0.0
    s = _time_sorted_series(df, col)
    if s is None or len(s) < max(20, int(tail_days) + 5):
        return 0.0
    t_end = s.index.max()
    t_cut = t_end - pd.Timedelta(days=tail_days)
    tail = s[s.index >= t_cut]
    if len(tail) < 2:
        return 0.0
    mu = float(s.mean())
    sig = float(s.std())
    if sig == 0 or np.isnan(sig):
        return 0.0
    mr = float(tail.mean())
    return (mr - mu) / sig


def hypoxia_risk_score_0_100(df: pd.DataFrame) -> float:
    """
    0–100 gauge: weighted z-style terms (clipped), no ML.

    Tail (last ~5d) vs **mean/std of the entire current time window** so the score **responds**
    when you change sidebar **Time window** or mooring slice (baseline is recomputed from visible rows).
    """
    o2c = pick_o2_column(df)
    ph_c = pick_best_column(df, "ph")
    t_c = pick_best_column(df, "sst")
    n_c = pick_best_column(df, "no3")

    z_o2 = _z_tail_vs_window(df, o2c) if o2c else 0.0
    low_o2 = max(0.0, -z_o2)
    z_ph = _z_tail_vs_window(df, ph_c) if ph_c else 0.0
    low_ph = max(0.0, -z_ph)
    z_t = _z_tail_vs_window(df, t_c) if t_c else 0.0
    high_t = max(0.0, z_t)
    z_n = _z_tail_vs_window(df, n_c) if n_c else 0.0
    high_n = max(0.0, z_n)

    raw = 0.4 * low_o2 + 0.3 * low_ph + 0.2 * high_t + 0.1 * high_n
    score = 100.0 * (1.0 - np.exp(-raw / 1.8))
    return float(np.clip(score, 0.0, 100.0))


def hypoxia_risk_breakdown(df: pd.DataFrame) -> dict[str, float | str | None]:
    """For UI captions — same logic as score, exposes z-shapes."""
    o2c = pick_o2_column(df)
    ph_c = pick_best_column(df, "ph")
    t_c = pick_best_column(df, "sst")
    n_c = pick_best_column(df, "no3")
    z_o2 = _z_tail_vs_window(df, o2c) if o2c else float("nan")
    z_ph = _z_tail_vs_window(df, ph_c) if ph_c else float("nan")
    z_t = _z_tail_vs_window(df, t_c) if t_c else float("nan")
    z_n = _z_tail_vs_window(df, n_c) if n_c else float("nan")
    return {
        "score": hypoxia_risk_score_0_100(df),
        "z_o2_recent": z_o2,
        "z_ph_recent": z_ph,
        "z_temp_recent": z_t,
        "z_nitrate_recent": z_n,
        "o2_col": o2c,
        "ph_col": ph_c,
        "temp_col": t_c,
        "no3_col": n_c,
    }
