"""Synthetic hourly mooring-like series for demo when raw files are absent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cce_hack.config import DATA_PROC


def build_synthetic_hourly(
    mooring: str = "CCE2",
    start: str = "2022-01-01",
    hours: int = 8760 * 2,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=hours, freq="h", tz="UTC")

    # Seasonal + diurnal "ocean" signals
    doy = t.dayofyear.values
    hod = t.hour.values
    seasonal = 2.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    diurnal_sst = 0.35 * np.sin(2 * np.pi * (hod - 14) / 24)

    wind_base = 6 + 3 * np.sin(2 * np.pi * (doy - 40) / 365.25)
    storms = rng.normal(0, 1.2, len(t))
    wind_speed = np.clip(wind_base + storms + 0.8 * rng.standard_normal(len(t)), 0.5, 22)

    sst = 14.5 + seasonal + diurnal_sst + 0.02 * wind_speed + 0.15 * rng.standard_normal(len(t))

    # pH covaries with temperature/upwelling proxy
    ph = 8.05 - 0.01 * (sst - 14.5) - 0.0008 * wind_speed + 0.003 * rng.standard_normal(len(t))

    sal = 33.4 + 0.15 * np.sin(2 * np.pi * doy / 365.25) + 0.05 * rng.standard_normal(len(t))

    pco2 = 420 + 35 * (14.5 - sst) + 6 * wind_speed + 5 * rng.standard_normal(len(t))

    chl = np.clip(0.4 + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365.25) + 0.05 * rng.standard_normal(len(t)), 0.05, 3)

    air_t = sst - 1.2 + 0.35 * rng.standard_normal(len(t))

    # Dissolved O₂ (mg/L) — demo-only: warmer / windier → slightly lower O₂ (not a full gas exchange model).
    do_mgl = (
        6.8
        + 0.12 * (14.5 - sst)
        - 0.04 * (wind_speed - 8.0)
        + 0.08 * rng.standard_normal(len(t))
    )
    do_mgl = np.clip(do_mgl, 3.5, 9.5)

    df = pd.DataFrame(
        {
            "time": t,
            "mooring_id": mooring,
            "air_temp_c": air_t.astype("float64"),
            "wind_speed_ms": wind_speed.astype("float64"),
            "sst_c": sst.astype("float64"),
            "salinity_psu": sal.astype("float64"),
            "ph_total": ph.astype("float64"),
            "pco2_uatm": pco2.astype("float64"),
            "chl_mg_m3": chl.astype("float64"),
            "dissolved_oxygen_mg_l": do_mgl.astype("float64"),
        }
    )

    # Realistic gaps (telemetry loss)
    drop = rng.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[drop, ["ph_total", "pco2_uatm", "chl_mg_m3", "dissolved_oxygen_mg_l"]] = np.nan

    return df


def _augment_dissolved_oxygen_if_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add ``dissolved_oxygen_mg_l`` when an older sample CSV lacks it (keeps other columns)."""
    if "dissolved_oxygen_mg_l" in df.columns:
        return df
    if "sst_c" not in df.columns:
        return df
    sst = pd.to_numeric(df["sst_c"], errors="coerce").to_numpy()
    wind = pd.to_numeric(df.get("wind_speed_ms", pd.Series(8.0, index=df.index)), errors="coerce").fillna(8.0).to_numpy()
    do_mgl = 6.8 + 0.12 * (14.5 - sst) - 0.04 * (wind - 8.0) + 0.08 * rng.standard_normal(len(df))
    out = df.copy()
    out["dissolved_oxygen_mg_l"] = np.clip(do_mgl, 3.5, 9.5)
    return out


def ensure_sample_csv(path=None) -> Path:
    path = path or (DATA_PROC / "cce_sample_hourly.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        build_synthetic_hourly().to_csv(path, index=False)
    else:
        # Older demo files had no O₂ column — add one so Mission Control / risk gauge are not blank.
        try:
            existing = pd.read_csv(path)
            if "dissolved_oxygen_mg_l" not in existing.columns and "sst_c" in existing.columns:
                rng = np.random.default_rng(42)
                _augment_dissolved_oxygen_if_missing(existing, rng).to_csv(path, index=False)
        except Exception:
            pass
    return path
