"""Load and harmonize CCE `*_combined.csv` exports under `data/raw/<variable>/`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cce_hack.config import DATA_PROC, DATA_RAW

PANEL_FILENAME = "cce_hourly_panel.csv"


def _parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _floor_hour(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("h")


def _hourly_mean(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["time"] = _floor_hour(d["time"])
    g = d.groupby(["mooring_id", "time"], as_index=False)[value_cols].mean(numeric_only=True)
    return g


def _apply_ts_qc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, qc in (("TEMP", "TEMP_QC"), ("PSAL", "PSAL_QC"), ("CNDC", "CNDC_QC")):
        if qc in df.columns:
            bad = pd.to_numeric(df[qc], errors="coerce").ne(1)
            df.loc[bad, col] = np.nan
    return df


def load_temperature_salinity_csv(path: Path) -> pd.DataFrame:
    """All CTD depths in the export (e.g. 32 / 38 / 39 m), QC-masked, hourly means."""
    df = pd.read_csv(path)
    df["time"] = _parse_time(df["TIME"])
    df = df.dropna(subset=["time"])
    df = _apply_ts_qc(df)
    df["mooring_id"] = df["station"].astype(str)
    df["depth_m"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    merged: pd.DataFrame | None = None
    for d in sorted(df["depth_m"].dropna().unique()):
        di = int(round(float(d)))
        sub = df[np.isclose(df["depth_m"], float(d))].copy()
        if sub.empty:
            continue
        sub = sub.rename(
            columns={
                "TEMP": f"sst_c_d{di}m",
                "PSAL": f"salinity_psu_d{di}m",
                "CNDC": f"conductivity_s_m_d{di}m",
            }
        )
        cols = [f"sst_c_d{di}m", f"salinity_psu_d{di}m", f"conductivity_s_m_d{di}m"]
        cols = [c for c in cols if c in sub.columns]
        h = _hourly_mean(sub[["time", "mooring_id"] + cols], cols)
        merged = h if merged is None else merged.merge(h, on=["mooring_id", "time"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["time", "mooring_id"])

    # Legacy single-depth names (32 m) for ML defaults
    if "sst_c_d32m" in merged.columns:
        merged["sst_c"] = merged["sst_c_d32m"]
    if "salinity_psu_d32m" in merged.columns:
        merged["salinity_psu"] = merged["salinity_psu_d32m"]
    if "conductivity_s_m_d32m" in merged.columns:
        merged["conductivity_s_m"] = merged["conductivity_s_m_d32m"]

    lat = pd.to_numeric(df.get("LATITUDE"), errors="coerce")
    lon = pd.to_numeric(df.get("LONGITUDE"), errors="coerce")
    if lat.notna().any() and lon.notna().any():
        merged["latitude"] = float(lat.dropna().iloc[0])
        merged["longitude"] = float(lon.dropna().iloc[0])
    return merged


def load_ph_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = _parse_time(df["TIME"])
    df = df.dropna(subset=["time"])
    df["mooring_id"] = df["station"].astype(str)
    df["depth_m"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    df = df[np.isclose(df["depth_m"], 40.0)].copy()
    out = df.rename(columns={"PH_TOT": "ph_total"})
    return out[["time", "mooring_id", "ph_total"]]


def load_nitrate_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = _parse_time(df["TIME"])
    df = df.dropna(subset=["time"])
    df["mooring_id"] = df["station"].astype(str)
    df["depth_m"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    df = df[np.isclose(df["depth_m"], 40.0)].copy()
    out = df.rename(columns={"NO3": "no3"})
    return out[["time", "mooring_id", "no3"]]


def load_chlorophyll_csv(path: Path) -> pd.DataFrame:
    """Both 20 m and 40 m fluorometer channels (40 m often sparse)."""
    df = pd.read_csv(path)
    df["time"] = _parse_time(df["TIME"])
    df = df.dropna(subset=["time"])
    df["mooring_id"] = df["station"].astype(str)
    df["depth_m"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    merged: pd.DataFrame | None = None
    for di in (20, 40):
        sub = df[np.isclose(df["depth_m"], float(di))].copy()
        if sub.empty:
            continue
        col = f"chl_mg_m3_d{di}m"
        sub[col] = pd.to_numeric(sub["CHL"], errors="coerce")
        sub.loc[sub[col] < 0, col] = np.nan
        h = _hourly_mean(sub[["time", "mooring_id", col]], [col])
        merged = h if merged is None else merged.merge(h, on=["mooring_id", "time"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["time", "mooring_id"])

    if "chl_mg_m3_d20m" in merged.columns:
        merged["chl_mg_m3"] = merged["chl_mg_m3_d20m"]
    return merged


def discover_combined_csvs(raw_dir: Path | None = None) -> dict[str, Path]:
    raw_dir = raw_dir or DATA_RAW
    out: dict[str, Path] = {}
    mapping = {
        "temperature_salinity": raw_dir / "temperature_salinity" / "temperature_salinity_combined.csv",
        "ph": raw_dir / "ph" / "ph_combined.csv",
        "nitrate": raw_dir / "nitrate" / "nitrate_combined.csv",
        "chlorophyll": raw_dir / "chlorophyll" / "chlorophyll_combined.csv",
    }
    for key, p in mapping.items():
        if p.exists():
            out[key] = p
    return out


def build_hourly_panel(raw_dir: Path | None = None) -> pd.DataFrame:
    """Return one row per mooring_id and UTC hour with biogeochem + T/S columns."""
    raw_dir = raw_dir or DATA_RAW
    paths = discover_combined_csvs(raw_dir)
    frames: list[pd.DataFrame] = []

    if "temperature_salinity" in paths:
        ts = load_temperature_salinity_csv(paths["temperature_salinity"])
        vcols = [c for c in ts.columns if c not in ("mooring_id", "time")]
        if vcols:
            frames.append(ts[["mooring_id", "time"] + vcols])
    if "ph" in paths:
        ph = load_ph_csv(paths["ph"])
        frames.append(_hourly_mean(ph, ["ph_total"]))
    if "nitrate" in paths:
        n = load_nitrate_csv(paths["nitrate"])
        frames.append(_hourly_mean(n, ["no3"]))
    if "chlorophyll" in paths:
        c = load_chlorophyll_csv(paths["chlorophyll"])
        vcols = [x for x in c.columns if x not in ("mooring_id", "time")]
        if vcols:
            frames.append(c[["mooring_id", "time"] + vcols])

    if not frames:
        raise FileNotFoundError(
            f"No CCE combined CSVs found under {raw_dir}. Expected subfolders nitrate/, chlorophyll/, ph/, temperature_salinity/."
        )

    merged = frames[0]
    for nxt in frames[1:]:
        merged = merged.merge(nxt, on=["mooring_id", "time"], how="outer")

    merged = merged.sort_values(["mooring_id", "time"]).reset_index(drop=True)
    return merged


def write_hourly_panel(path: Path | None = None, raw_dir: Path | None = None) -> Path:
    path = path or (DATA_PROC / PANEL_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_hourly_panel(raw_dir=raw_dir)
    df.to_csv(path, index=False)
    return path


def processed_panel_path() -> Path:
    return DATA_PROC / PANEL_FILENAME
