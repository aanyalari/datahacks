from __future__ import annotations

from pathlib import Path

import pandas as pd

from cce_hack.config import DATA_PROC, DATA_RAW, SAMPLE_CSV
from cce_hack.ingest_raw import PANEL_FILENAME, discover_combined_csvs, write_hourly_panel
from cce_hack.sample_data import ensure_sample_csv


def finalize_mooring_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize detected time column to UTC ``time`` and sort (shared by disk + upload paths)."""
    df = df.copy()
    time_col = _detect_time_column(df.columns)
    if "unixtime" in time_col.lower() and "*1000" in time_col.lower():
        df[time_col] = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit="ms", utc=True)
    else:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.rename(columns={time_col: "time"})
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
    """Prefer merged hourly panel; else top-level raw CSV; else build panel from `raw/*/` exports."""
    panel = DATA_PROC / PANEL_FILENAME
    if panel.exists():
        return panel
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
