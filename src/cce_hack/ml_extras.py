"""ARIMA, RandomForest + SHAP, regime classifier, optional LSTM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False

try:
    from statsmodels.tsa.arima.model import ARIMA

    _HAS_ARIMA = True
except ImportError:
    ARIMA = None
    _HAS_ARIMA = False


def arima_daily_forecast(df: pd.DataFrame, col: str, order: tuple[int, int, int] = (2, 1, 1), horizon_days: int = 30) -> dict | None:
    if not _HAS_ARIMA or col not in df.columns:
        return None
    s = df[["time", col]].dropna().sort_values("time").set_index("time")[col].resample("D").mean().interpolate(limit_direction="both")
    if len(s) < 60:
        return None
    try:
        fit = ARIMA(s, order=order).fit()
        fc = fit.get_forecast(steps=horizon_days)
        pred = fc.predicted_mean
        ci = fc.conf_int()
    except Exception:
        return None
    return {
        "history": s,
        "forecast_index": pred.index,
        "forecast_mean": pred.values,
        "ci_low": ci.iloc[:, 0].values,
        "ci_high": ci.iloc[:, 1].values,
        "aic": float(fit.aic),
    }


def random_forest_with_shap(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    max_samples: int = 4000,
    n_estimators: int = 120,
) -> dict | None:
    if not _HAS_SHAP:
        return None
    use = [c for c in features if c in df.columns and c != target]
    if target not in df.columns or len(use) < 2:
        return None
    d = df[[target] + use].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    d = d.dropna(how="all", axis=1)
    use = [c for c in use if c in d.columns]
    d = d.dropna(subset=use, how="all")
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(d[use])
    y = d[target].to_numpy()
    if len(y) < 100:
        return None
    if len(y) > max_samples:
        idx = np.random.RandomState(0).choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=0, max_depth=8, n_jobs=-1)
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X[: min(800, len(X))])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return {
        "model": rf,
        "feature_names": use,
        "shap_values": shap_vals,
        "X_sample": X[: min(800, len(X))],
        "r2_train": float(rf.score(X, y)),
    }


def regime_classifier(
    df: pd.DataFrame,
    features: list[str],
    regime_col: str = "regime_kmeans",
    test_size: float = 0.25,
) -> dict | None:
    use = [c for c in features if c in df.columns and c != regime_col]
    if regime_col not in df.columns or len(use) < 2:
        return None
    d = df[use + [regime_col]].dropna(subset=[regime_col])
    d = d.replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(d[use])
    y = d[regime_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 80:
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
    clf = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0, n_jobs=-1)),
        ]
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "pipeline": clf,
        "feature_names": use,
        "report": classification_report(y_te, pred, zero_division=0),
        "accuracy": float((pred == y_te).mean()),
    }


def lstm_sequence_forecast(
    df: pd.DataFrame,
    col: str,
    seq_len: int = 72,
    epochs: int = 15,
) -> dict | None:
    """Optional TensorFlow LSTM: next-hour target in z-score space. Returns None if TF unavailable."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        return None
    if col not in df.columns:
        return None
    s = (
        df[["time", col]]
        .dropna()
        .sort_values("time")
        .set_index("time")[col]
        .resample("h")
        .mean()
        .interpolate(limit_direction="both", limit=72)
    )
    z = (s - s.mean()) / (s.std() + 1e-9)
    vals = z.dropna().to_numpy(dtype=np.float32)
    if len(vals) < seq_len + 300:
        return None
    X, y = [], []
    for i in range(len(vals) - seq_len - 1):
        X.append(vals[i : i + seq_len])
        y.append(vals[i + seq_len])
    X = np.stack(X)[..., None]
    y = np.array(y, dtype=np.float32)
    split = int(len(X) * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    tf.keras.utils.set_random_seed(0)
    m = models.Sequential(
        [
            layers.Input(shape=(seq_len, 1)),
            layers.LSTM(40, return_sequences=False),
            layers.Dense(1),
        ]
    )
    m.compile(optimizer="adam", loss="mse")
    m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=128, verbose=0)
    pred_va = m.predict(X_va, verbose=0).ravel()
    mae = float(np.mean(np.abs(pred_va - y_va)))
    return {"model": m, "valid_mae_z": mae, "series_mean": float(s.mean()), "series_std": float(s.std())}
