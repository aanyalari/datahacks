from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cce_hack.config import DATA_PROC, DATA_RAW, MOORING_MASTER_FILENAME, SAMPLE_CSV
from cce_hack.ingest_raw import PANEL_FILENAME, discover_combined_csvs, write_hourly_panel
from cce_hack.sample_data import ensure_sample_csv


def finalize_mooring_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize detected time column to UTC ``time`` and sort (shared by disk + upload paths)."""
    df = df.copy()
    # Friend / daily-export shape: ``station`` + ``ph`` / ``temperature`` / …
    if "mooring_id" not in df.columns and "station" in df.columns:
        df["mooring_id"] = df["station"].astype(str)
    if "ph" in df.columns and "ph_total" not in df.columns:
        df["ph_total"] = pd.to_numeric(df["ph"], errors="coerce")
    if "temperature" in df.columns and "sst_c" not in df.columns:
        df["sst_c"] = pd.to_numeric(df["temperature"], errors="coerce")
    if "salinity" in df.columns and "salinity_psu" not in df.columns:
        df["salinity_psu"] = pd.to_numeric(df["salinity"], errors="coerce")
    if "nitrate" in df.columns and "no3" not in df.columns:
        df["no3"] = pd.to_numeric(df["nitrate"], errors="coerce")
    if "chlorophyll" in df.columns and "chl_mg_m3" not in df.columns:
        df["chl_mg_m3"] = pd.to_numeric(df["chlorophyll"], errors="coerce")
    if "oxygen" in df.columns:
        df["oxygen"] = pd.to_numeric(df["oxygen"], errors="coerce")

    time_col = _detect_time_column(df.columns)
    if "unixtime" in time_col.lower() and "*1000" in time_col.lower():
        df[time_col] = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit="ms", utc=True)
    else:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.rename(columns={time_col: "time"})

    _meta = {"time", "date", "station", "mooring_id", "year", "month", "day", "latitude", "longitude"}
    for c in df.columns:
        if c in _meta or c.startswith("Unnamed"):
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            coerced = pd.to_numeric(df[c], errors="coerce")
            if int(coerced.notna().sum()) > max(5, len(df) // 200):
                df[c] = coerced
    return df


def load_mooring_from_upload(file) -> pd.DataFrame:
    """Load mooring-like CSV from a Streamlit ``UploadedFile`` or any file-like accepted by ``pandas.read_csv``."""
    df = pd.read_csv(file)
    return finalize_mooring_dataframe(df)


def load_mooring_table(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load long-format mooring CSV. Expected columns include a time column."""
    csv_path = Path(csv_path) if csv_path else pick_default_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing data file: {csv_path}. Place a CSV under data/raw/ or run sample generation."
        )
    df = pd.read_csv(csv_path)
    return finalize_mooring_dataframe(df)


def pick_default_csv() -> Path:
    """Prefer hourly panel; else daily ``mooring_master``; else build panel from ``raw/*/``; else sample."""
    panel = DATA_PROC / PANEL_FILENAME
    if panel.exists():
        return panel
    master = DATA_PROC / MOORING_MASTER_FILENAME
    if master.exists():
        return master
    raw_dir = DATA_RAW
    if raw_dir.exists() and discover_combined_csvs(raw_dir):
        return write_hourly_panel(panel, raw_dir=raw_dir)
    if raw_dir.exists():
        cands = sorted(p for p in raw_dir.glob("*.csv") if not p.name.startswith("_"))
        if cands:
            return cands[0]
    return ensure_sample_csv(SAMPLE_CSV)


def _detect_time_column(columns: pd.Index) -> str:
    lowered = {c.lower(): c for c in columns}
    for key in ("time", "timestamp", "datetime", "date"):
        if key in lowered:
            return lowered[key]
    for c in columns:
        cl = str(c).lower()
        if "time" in cl or "unixtime" in cl:
            return str(c)
    raise ValueError(f"Could not detect time column from: {list(columns)}")
