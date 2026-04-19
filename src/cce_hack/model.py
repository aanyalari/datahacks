from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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


def gradient_boosting_feature_importances(
    result: TrainResult,
    valid: pd.DataFrame,
    y_col: str,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Feature importance for the fitted pipeline. Uses native ``feature_importances_`` when present;
    otherwise **permutation importance** on a subsample of the validation frame (slower but works on all sklearn versions).
    """
    step = result.pipeline.named_steps.get("model")
    feat = result.feature_columns
    if not feat or y_col not in valid.columns:
        return pd.DataFrame()

    if (
        step is not None
        and hasattr(step, "feature_importances_")
        and len(getattr(step, "feature_importances_")) == len(feat)
    ):
        imp = step.feature_importances_
        df = pd.DataFrame({"feature": feat, "importance": imp})
    else:
        vm = valid.replace([np.inf, -np.inf], np.nan)
        m = vm[y_col].apply(np.isfinite)
        vm = vm.loc[m]
        if len(vm) < 40:
            return pd.DataFrame()
        Xv = vm[feat]
        yv = vm[y_col].to_numpy(dtype=float)
        n_sub = min(1200, len(vm))
        rng = np.random.RandomState(0)
        idx = rng.choice(len(vm), size=n_sub, replace=False)
        r = permutation_importance(
            result.pipeline,
            Xv.iloc[idx],
            yv[idx],
            n_repeats=4,
            random_state=0,
            n_jobs=1,
        )
        df = pd.DataFrame({"feature": feat, "importance": r.importances_mean})

    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def validation_prediction_frame(
    valid: pd.DataFrame,
    result: TrainResult,
    y_col: str,
) -> pd.DataFrame:
    """
    Aligned actual vs model predictions on the validation period (time-ordered).
    Includes a **persistence** baseline (predict y(t+H) = y(t)) and a **seasonal-naive**
    baseline (y(t+H-24)) when ``result.target`` is present in ``valid``.
    """
    feat = result.feature_columns
    vm = valid.replace([np.inf, -np.inf], np.nan)
    m = vm[y_col].apply(np.isfinite)
    vm = vm.loc[m]
    if vm.empty:
        return pd.DataFrame()
    X = vm[feat]
    y_hat = result.pipeline.predict(X)
    y_true = vm[y_col].to_numpy(dtype=float)
    tcol = vm["time"] if "time" in vm.columns else pd.RangeIndex(len(vm))
    out = pd.DataFrame({"time": pd.to_datetime(tcol, utc=True, errors="coerce"), "actual": y_true, "predicted": y_hat})
    tgt = result.target
    if tgt in vm.columns:
        tgt_vals = vm[tgt].to_numpy(dtype=float)
        out["persistence"] = tgt_vals
        # seasonal-naive: y_hat(t+H) = y(t+H-24h) ≈ tgt(t-24h+H). Approximate using shift on the
        # target column inside the valid frame (time-ordered hourly).
        sn_lag = max(1, 24 - int(result.horizon_h)) if result.horizon_h <= 24 else 24
        out["seasonal_naive_24h"] = pd.Series(tgt_vals).shift(sn_lag).to_numpy()
    return out.sort_values("time").reset_index(drop=True)


def train_quantile_forecasters(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, np.ndarray]:
    """
    Fit one HistGradientBoosting **quantile** model per quantile and predict on the validation rows
    (those with finite ``y_col``). Returns ``{quantile: predictions_array}`` aligned to the same
    valid-row order used by ``validation_prediction_frame``.
    """
    vm = valid.replace([np.inf, -np.inf], np.nan)
    m = vm[y_col].apply(np.isfinite)
    valid_use = vm.loc[m]
    tm = train.replace([np.inf, -np.inf], np.nan)
    tm = tm[tm[y_col].apply(np.isfinite)]
    if len(tm) < 60 or valid_use.empty:
        return {}

    X_tr = tm[feature_cols]
    y_tr = tm[y_col].to_numpy(dtype=float)
    X_va = valid_use[feature_cols]
    out: dict[float, np.ndarray] = {}
    for q in quantiles:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(q),
            max_depth=6,
            learning_rate=0.06,
            max_iter=300,
            random_state=0,
        )
        model.fit(X_tr, y_tr)
        out[float(q)] = model.predict(X_va)
    return out


def seasonal_naive_baseline_mae(
    valid: pd.DataFrame,
    target: str,
    y_col: str,
    season_h: int = 24,
) -> float:
    """
    Compare ``y_col`` (the future value) to ``target`` shifted ``season_h`` hours ahead inside
    ``valid``. This gives the MAE of a "value 24 hours ago" forecast as a baseline alongside persistence.
    """
    vm = valid[[target, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vm) < season_h + 5:
        return float("nan")
    sn = vm[target].shift(season_h)
    keep = sn.notna()
    if keep.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(vm.loc[keep, y_col].to_numpy(dtype=float), sn.loc[keep].to_numpy(dtype=float)))


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
