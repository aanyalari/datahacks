from __future__ import annotations

import pandas as pd

from cce_hack.config import DEFAULT_HORIZON_HOURS, LAG_HOURS
from cce_hack.features import add_calendar_features, add_future_target, add_lags, train_valid_split_by_time
from cce_hack.model import TrainResult, train_forecaster


def _prune_unused_features(sup: pd.DataFrame, feat: list[str]) -> list[str]:
    """Drop feature columns that are entirely NaN (common for sparse mooring channels)."""
    out = []
    for c in feat:
        if c not in sup.columns:
            continue
        if sup[c].notna().any():
            out.append(c)
    return out


def default_lag_columns() -> list[str]:
    return [
        "sst_c",
        "sst_c_d38m",
        "sst_c_d39m",
        "wind_speed_ms",
        "ph_total",
        "salinity_psu",
        "salinity_psu_d38m",
        "salinity_psu_d39m",
        "conductivity_s_m",
        "conductivity_s_m_d38m",
        "conductivity_s_m_d39m",
        "pco2_uatm",
        "air_temp_c",
        "chl_mg_m3",
        "chl_mg_m3_d40m",
        "no3",
    ]


def build_supervised_frame(df: pd.DataFrame, target: str, horizon_h: int = DEFAULT_HORIZON_HOURS) -> tuple[pd.DataFrame, list[str], str]:
    base_cols = [c for c in default_lag_columns() if c in df.columns]
    d = add_lags(df, base_cols, LAG_HOURS)
    d = add_calendar_features(d)
    d = add_future_target(d, target, horizon_h)
    y_col = f"{target}_lead_{horizon_h}h"

    feat = []
    for c in base_cols:
        for h in LAG_HOURS:
            name = f"{c}_lag_{h}h"
            if name in d.columns:
                feat.append(name)
    feat += ["hour", "doy", "month"]

    d = d.dropna(subset=[y_col]).reset_index(drop=True)
    feat = _prune_unused_features(d, feat)
    return d, feat, y_col


def run_default_experiment(df: pd.DataFrame, target: str = "sst_c", horizon_h: int = DEFAULT_HORIZON_HOURS) -> TrainResult:
    sup, feat, y_col = build_supervised_frame(df, target, horizon_h)
    tr, va = train_valid_split_by_time(sup, valid_frac=0.2)
    return train_forecaster(tr, va, target, horizon_h, feat, y_col)
