from __future__ import annotations

import pandas as pd

from cce_hack.config import LAG_HOURS


def add_lags(df: pd.DataFrame, cols: list[str], hours: tuple[int, ...] = LAG_HOURS) -> pd.DataFrame:
    out = df.sort_values("time").copy()
    if "mooring_id" in out.columns:
        parts = []
        for _, g in out.groupby("mooring_id", sort=False):
            gg = g.set_index("time")
            for c in cols:
                if c not in gg.columns:
                    continue
                for h in hours:
                    gg[f"{c}_lag_{h}h"] = gg[c].shift(h)
            parts.append(gg.reset_index())
        return pd.concat(parts, ignore_index=True)

    gg = out.set_index("time")
    for c in cols:
        if c not in gg.columns:
            continue
        for h in hours:
            gg[f"{c}_lag_{h}h"] = gg[c].shift(h)
    return gg.reset_index()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out["time"]
    out["hour"] = t.dt.hour.astype("int16")
    out["doy"] = t.dt.dayofyear.astype("int16")
    out["month"] = t.dt.month.astype("int16")
    return out


def add_future_target(df: pd.DataFrame, target: str, horizon_h: int) -> pd.DataFrame:
    out = df.sort_values("time").copy()
    col = f"{target}_lead_{horizon_h}h"
    if "mooring_id" in out.columns:
        out[col] = out.groupby("mooring_id", sort=False)[target].shift(-horizon_h)
    else:
        out[col] = out[target].shift(-horizon_h)
    return out


def train_valid_split_by_time(df: pd.DataFrame, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    cut = int(n * (1 - valid_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
