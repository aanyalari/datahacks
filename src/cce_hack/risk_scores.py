"""Simple deterministic risk composites (no ML)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cce_hack.column_pick import pick_best_column
from cce_hack.mission_alerts import pick_o2_column


def _z_latest_vs_series(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 20:
        return 0.0
    latest = float(s.iloc[-1])
    mu, sig = float(s.mean()), float(s.std())
    if sig == 0 or np.isnan(sig):
        return 0.0
    return (latest - mu) / sig


def hypoxia_risk_score_0_100(df: pd.DataFrame) -> float:
    """
    0–100 gauge: weighted z-style terms (clipped), no ML.

    low_O2 → high when O₂ is **below** mean (uses max(0, -z)).
    low_pH → high when pH **below** mean.
    high_temp / high_nitrate → high when **above** mean.
    """
    o2c = pick_o2_column(df)
    ph_c = pick_best_column(df, "ph")
    t_c = pick_best_column(df, "sst")
    n_c = pick_best_column(df, "no3")
    z_o2 = _z_latest_vs_series(df[o2c]) if o2c else 0.0
    low_o2 = max(0.0, -z_o2)
    z_ph = _z_latest_vs_series(df[ph_c]) if ph_c else 0.0
    low_ph = max(0.0, -z_ph)
    z_t = _z_latest_vs_series(df[t_c]) if t_c else 0.0
    high_t = max(0.0, z_t)
    z_n = _z_latest_vs_series(df[n_c]) if n_c else 0.0
    high_n = max(0.0, z_n)
    raw = 0.4 * low_o2 + 0.3 * low_ph + 0.2 * high_t + 0.1 * high_n
    # squash into 0–100 (typical raw < 3)
    score = 100.0 * (1.0 - np.exp(-raw / 1.8))
    return float(np.clip(score, 0.0, 100.0))
