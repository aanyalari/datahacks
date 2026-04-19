"""
Soft-sensor models: predict a single mooring channel from the **other** channels at the same timestamp.

Use case: a sensor (e.g. dissolved oxygen, pH) drops out — reconstruct its signal from the rest of
the suite, with honest holdout metrics + residuals + interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_CANDIDATE_TARGETS = (
    "dissolved_oxygen_mg_l",
    "ph_total",
    "pco2_uatm",
    "chl_mg_m3",
    "salinity_psu",
    "sst_c",
)


@dataclass
class SoftSensorResult:
    target: str
    feature_columns: list[str]
    train_rows: int
    valid_rows: int
    valid_mae: float
    valid_rmse: float
    valid_r2: float
    mean_baseline_mae: float
    pipeline: HistGradientBoostingRegressor
    valid_frame: pd.DataFrame  # cols: time, actual, predicted, residual
    importance: pd.DataFrame  # cols: feature, importance


def _candidate_features(df: pd.DataFrame, target: str) -> list[str]:
    skip = {"time", "mooring_id", "latitude", "longitude", target}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def train_soft_sensor(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str] | None = None,
    *,
    valid_frac: float = 0.2,
    min_rows: int = 200,
    permutation_repeats: int = 4,
) -> SoftSensorResult | None:
    """
    Train a HistGradientBoosting regressor predicting ``target`` from same-time other channels.

    Time-ordered holdout (last ``valid_frac`` of rows). Returns ``None`` when too sparse.
    """
    if "time" not in df.columns or target not in df.columns:
        return None
    feats = list(feature_cols) if feature_cols else _candidate_features(df, target)
    feats = [c for c in feats if c in df.columns and c != target]
    if len(feats) < 2:
        return None

    use = df[["time", target] + feats].copy()
    use["time"] = pd.to_datetime(use["time"], utc=True, errors="coerce")
    use = use.dropna(subset=["time", target]).sort_values("time").reset_index(drop=True)
    if len(use) < min_rows:
        return None

    n = len(use)
    cut = int(n * (1 - valid_frac))
    tr = use.iloc[:cut]
    va = use.iloc[cut:]
    if len(tr) < 80 or len(va) < 30:
        return None

    X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
    y_tr = tr[target].to_numpy(dtype=float)
    X_va = va[feats].replace([np.inf, -np.inf], np.nan)
    y_va = va[target].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=300,
        random_state=0,
    )
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_va)

    valid_mae = float(mean_absolute_error(y_va, y_hat))
    valid_rmse = float(np.sqrt(mean_squared_error(y_va, y_hat)))
    try:
        valid_r2 = float(r2_score(y_va, y_hat))
    except ValueError:
        valid_r2 = float("nan")

    mean_pred = np.full_like(y_va, fill_value=float(np.mean(y_tr)))
    mean_mae = float(mean_absolute_error(y_va, mean_pred))

    # Permutation importance on a subsample (n_jobs=1 to dodge sandbox semaphore issues).
    n_sub = min(800, len(va))
    rng = np.random.RandomState(0)
    idx = rng.choice(len(va), size=n_sub, replace=False)
    try:
        perm = permutation_importance(
            model, X_va.iloc[idx], y_va[idx], n_repeats=permutation_repeats, random_state=0, n_jobs=1
        )
        imp_df = (
            pd.DataFrame({"feature": feats, "importance": perm.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        imp_df = pd.DataFrame(columns=["feature", "importance"])

    valid_frame = pd.DataFrame(
        {
            "time": va["time"].to_numpy(),
            "actual": y_va,
            "predicted": y_hat,
            "residual": y_va - y_hat,
        }
    ).reset_index(drop=True)

    return SoftSensorResult(
        target=target,
        feature_columns=feats,
        train_rows=len(tr),
        valid_rows=len(va),
        valid_mae=valid_mae,
        valid_rmse=valid_rmse,
        valid_r2=valid_r2,
        mean_baseline_mae=mean_mae,
        pipeline=model,
        valid_frame=valid_frame,
        importance=imp_df,
    )


def candidate_soft_sensor_targets(df: pd.DataFrame) -> list[str]:
    """Sensors usually worth modeling for failure-recovery, filtered to those present in df."""
    return [c for c in DEFAULT_CANDIDATE_TARGETS if c in df.columns]
