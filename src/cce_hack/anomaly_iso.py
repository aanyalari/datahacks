"""Isolation Forest anomalies on mooring numeric features (CPU-friendly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from cce_hack.column_pick import friendly_axis_label


def isolation_forest_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    contamination: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, IsolationForest, StandardScaler] | tuple[None, None, None, None]:
    """
    Fit IsolationForest on ``feature_cols`` (rows with any NaN in those cols dropped).

    Returns (frame_with_time_anomaly_score, y_pred, model, scaler) or Nones if too little data.
    """
    use = [c for c in feature_cols if c in df.columns]
    if len(use) < 2 or "time" not in df.columns:
        return None, None, None, None
    # Mooring columns rarely share NaNs at identical timestamps; median-fill for IF input only.
    d = df[["time"] + use].dropna(subset=["time"]).copy()
    if len(d) < 80:
        return None, None, None, None
    for c in use:
        med = float(pd.to_numeric(d[c], errors="coerce").median())
        if np.isnan(med):
            med = 0.0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(med)
    X = d[use].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=200,
        contamination=min(0.5, max(0.01, contamination)),
        random_state=random_state,
        n_jobs=-1,
    )
    y = model.fit_predict(Xs)
    score = model.decision_function(Xs)
    out = d[["time"]].copy()
    out["anomaly"] = np.where(y == -1, 1, 0)
    out["iso_score"] = score
    for c in use:
        out[c] = d[c].values
    return out, y, model, scaler


def top_anomaly_events(out: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Lowest isolation score = most anomalous."""
    if out is None or out.empty:
        return pd.DataFrame()
    return out.nsmallest(n, "iso_score")


def build_anomaly_rank_table(
    df: pd.DataFrame,
    out: pd.DataFrame,
    feature_cols: list[str],
    *,
    n: int = 10,
) -> pd.DataFrame:
    """
    Rank top anomaly timestamps with the **driver variable** (largest |z| vs series mean/std on full ``df``).
    """
    use = [c for c in feature_cols if c in df.columns]
    if out is None or out.empty or not use:
        return pd.DataFrame()

    ref = df[use].apply(pd.to_numeric, errors="coerce")
    mu = ref.mean()
    sig = ref.std().replace(0, np.nan)

    rows_out = []
    for ix, row in out.nsmallest(n, "iso_score").iterrows():
        t = row["time"]
        zbest, vbest, val_fill = 0.0, use[0], float("nan")
        for c in use:
            if c not in row.index or pd.isna(row[c]):
                continue
            m, s = float(mu.get(c, np.nan)), float(sig.get(c, np.nan))
            if np.isnan(m) or np.isnan(s) or s == 0:
                continue
            z = (float(row[c]) - m) / s
            if abs(z) > abs(zbest):
                zbest, vbest, val_fill = z, c, float(row[c])
        raw_val = float("nan")
        if ix in df.index and vbest in df.columns:
            raw_val = float(pd.to_numeric(df.loc[ix, vbest], errors="coerce"))
        elif val_fill == val_fill:
            raw_val = val_fill
        sev = "🔴" if abs(zbest) >= 3 else ("🟡" if abs(zbest) >= 2 else "🟢")
        nice = friendly_axis_label(vbest)
        rows_out.append(
            {
                "When (UTC)": str(pd.Timestamp(t))[:19],
                "Sensor (what moved most)": nice,
                "Value at that time": round(raw_val, 5) if raw_val == raw_val else None,
                "Z vs whole mooring file": round(zbest, 2),
                "How unusual": sev,
                "_col": vbest,
            }
        )
    return pd.DataFrame(rows_out)
