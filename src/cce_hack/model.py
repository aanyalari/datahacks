from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


@dataclass
class TrainResult:
    pipeline: Pipeline
    feature_columns: list[str]
    target: str
    horizon_h: int
    train_mae: float
    valid_mae: float
    train_rmse: float
    valid_rmse: float
    baseline_valid_mae: float


def _rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def build_pipeline() -> Pipeline:
    # Trees only: no scaling. HistGradientBoostingRegressor accepts NaNs in X.
    return Pipeline(
        steps=[
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=300,
                    random_state=0,
                ),
            ),
        ]
    )


def train_forecaster(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    horizon_h: int,
    feature_cols: list[str],
    y_col: str,
) -> TrainResult:
    """y_col is the supervised label, e.g. sst_c_lead_24h."""

    def _xy(frame: pd.DataFrame):
        X = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = frame[y_col].to_numpy(dtype=float)
        # Do not require full rows: sparse mooring channels leave many NaN lags;
        # HGBT can use missing-value handling during tree fitting.
        m = np.isfinite(y)
        return X.loc[m], y[m]

    X_tr, y_tr = _xy(train)
    X_va, y_va = _xy(valid)

    if len(X_tr) == 0:
        raise ValueError(
            f"No training rows with finite {y_col!r}. "
            "Try another target, a shorter horizon, or a longer CSV with fewer gaps in that variable."
        )
    if len(X_va) == 0:
        raise ValueError(f"No validation rows with finite {y_col!r}.")

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    pred_tr = pipe.predict(X_tr)
    pred_va = pipe.predict(X_va)

    train_mae = float(mean_absolute_error(y_tr, pred_tr))
    valid_mae = float(mean_absolute_error(y_va, pred_va))
    train_rmse = _rmse(y_tr, pred_tr)
    valid_rmse = _rmse(y_va, pred_va)

    # Persistence: forecast y(t+H) using y(t) (same row target column)
    valid_m = valid[feature_cols + [y_col, target]].replace([np.inf, -np.inf], np.nan)
    mask = valid_m[y_col].notna() & valid_m[target].notna()
    y_true = valid_m.loc[mask, y_col].to_numpy(dtype=float)
    y_hat_persist = valid_m.loc[mask, target].to_numpy(dtype=float)
    baseline = float(mean_absolute_error(y_true, y_hat_persist)) if len(y_true) else float("nan")

    return TrainResult(
        pipeline=pipe,
        feature_columns=feature_cols,
        target=target,
        horizon_h=horizon_h,
        train_mae=train_mae,
        valid_mae=valid_mae,
        train_rmse=train_rmse,
        valid_rmse=valid_rmse,
        baseline_valid_mae=baseline,
    )
