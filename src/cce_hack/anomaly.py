"""Isolation Forest anomaly events for cross-page reuse (e.g. species validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cce_hack.anomaly_iso import isolation_forest_anomalies


def _default_feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"mooring_id", "latitude", "longitude", "time"}
    cols: list[str] = []
    n = len(df)
    need = max(20, min(80, n // 5)) if n else 20
    for c in df.columns:
        if c in skip:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= need:
            cols.append(c)
    return cols


def fit_mooring_isolation_scores(
    df: pd.DataFrame,
    *,
    contamination: float = 0.05,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Fit IsolationForest once; return scored frame (``time``, ``iso_score``, ``anomaly``, …) and feature list.
    """
    feats = feature_cols if feature_cols is not None else _default_feature_cols(df)
    feats = [c for c in feats if c in df.columns]
    if len(feats) < 2 or "time" not in df.columns:
        return None, feats
    out, _, _, _ = isolation_forest_anomalies(df, feats, contamination=contamination)
    if out is None or out.empty:
        return None, feats
    return out, feats


def detect_anomalies(df: pd.DataFrame, *, contamination: float = 0.05, top_n: int = 100) -> pd.DataFrame:
    """
    sklearn IsolationForest (``contamination``) on available numeric mooring columns.

    Returns columns: ``time``, ``variable``, ``zscore``, ``severity`` (``abs(zscore)``),
    one row per top strangeness timestamp (most negative ``iso_score`` first).
    """
    out, feats = fit_mooring_isolation_scores(df, contamination=contamination)
    if out is None or out.empty or len(feats) < 2:
        return pd.DataFrame(columns=["time", "variable", "zscore", "severity"])

    ref = df[feats].apply(pd.to_numeric, errors="coerce")
    mu = ref.mean()
    sig = ref.std().replace(0, np.nan)

    rows_out: list[dict[str, object]] = []
    ranked = out.nsmallest(int(top_n), "iso_score")
    seen_time: set[tuple[int, int, int]] = set()
    for _, row in ranked.iterrows():
        t = row["time"]
        ts = pd.to_datetime(t, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        key = (int(ts.year), int(ts.month), int(ts.day))
        if key in seen_time:
            continue
        seen_time.add(key)

        zbest, vbest = 0.0, feats[0]
        for c in feats:
            if c not in row.index or pd.isna(row[c]):
                continue
            m, s = float(mu.get(c, np.nan)), float(sig.get(c, np.nan))
            if np.isnan(m) or np.isnan(s) or s == 0:
                continue
            z = (float(row[c]) - m) / s
            if abs(z) > abs(zbest):
                zbest, vbest = z, c

        sev = float(abs(zbest))
        rows_out.append(
            {
                "time": pd.to_datetime(t, utc=True, errors="coerce"),
                "variable": vbest,
                "zscore": float(zbest),
                "severity": sev,
            }
        )

    ev = pd.DataFrame(rows_out)
    if ev.empty:
        return pd.DataFrame(columns=["time", "variable", "zscore", "severity"])
    return ev
