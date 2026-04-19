"""
Build daily processed CSVs from OceanSITES-style combined files under data/raw/,
then merge into data/processed/mooring_master.csv.

Optional: oxygen (if data/raw/oxygen/oxygen_combined.csv exists),
CalCOFI larvae / zooplankton CSVs if present.

Run from repo root:
  python scripts/process_mooring_daily.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def to_daily(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    station_col: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["date"] = df[time_col].dt.date
    group_cols = ["date"] + ([station_col] if station_col else [])
    use_cols = [c for c in value_cols if c in df.columns]
    if not use_cols:
        raise ValueError(f"None of {value_cols} found in columns: {list(df.columns)}")
    return df.groupby(group_cols, observed=False)[use_cols].mean().reset_index()


def _print_ok(msg: str) -> None:
    print(msg)


def main() -> int:
    os.makedirs("data/processed", exist_ok=True)

    # --- pH ---
    ph_path = Path("data/raw/ph/ph_combined.csv")
    if not ph_path.exists():
        print(f"Missing required file: {ph_path}", file=sys.stderr)
        return 1
    ph = pd.read_csv(ph_path)
    ph_clean = to_daily(ph, "TIME", ["PH_TOT"], "station")
    ph_clean.columns = ["date", "station", "ph"]
    ph_clean.to_csv("data/processed/ph_daily.csv", index=False)
    _print_ok(f"pH: {ph_clean.shape} | range: {ph_clean['date'].min()} -> {ph_clean['date'].max()}")

    # --- Temperature & salinity ---
    ts_path = Path("data/raw/temperature_salinity/temperature_salinity_combined.csv")
    if not ts_path.exists():
        print(f"Missing required file: {ts_path}", file=sys.stderr)
        return 1
    ts = pd.read_csv(ts_path)
    ts_clean = to_daily(ts, "TIME", ["TEMP", "PSAL"], "station")
    ts_clean.columns = ["date", "station", "temperature", "salinity"]
    ts_clean.to_csv("data/processed/temp_salinity_daily.csv", index=False)
    _print_ok(
        f"Temp/Sal: {ts_clean.shape} | range: {ts_clean['date'].min()} -> {ts_clean['date'].max()}"
    )

    # --- Nitrate ---
    no3_path = Path("data/raw/nitrate/nitrate_combined.csv")
    if not no3_path.exists():
        print(f"Missing required file: {no3_path}", file=sys.stderr)
        return 1
    no3 = pd.read_csv(no3_path)
    no3_clean = to_daily(no3, "TIME", ["NO3"], "station")
    no3_clean.columns = ["date", "station", "nitrate"]
    no3_clean.to_csv("data/processed/nitrate_daily.csv", index=False)
    _print_ok(
        f"Nitrate: {no3_clean.shape} | range: {no3_clean['date'].min()} -> {no3_clean['date'].max()}"
    )

    # --- Chlorophyll ---
    chl_path = Path("data/raw/chlorophyll/chlorophyll_combined.csv")
    if not chl_path.exists():
        print(f"Missing required file: {chl_path}", file=sys.stderr)
        return 1
    chl = pd.read_csv(chl_path)
    chl_clean = to_daily(chl, "TIME", ["CHL"], "station")
    chl_clean.columns = ["date", "station", "chlorophyll"]
    chl_clean.to_csv("data/processed/chlorophyll_daily.csv", index=False)
    _print_ok(
        f"Chlorophyll: {chl_clean.shape} | range: {chl_clean['date'].min()} -> {chl_clean['date'].max()}"
    )

    # --- Oxygen (optional: needs oxygen_combined.csv from OPeNDAP loader) ---
    oxy_path = Path("data/raw/oxygen/oxygen_combined.csv")
    oxy_clean: pd.DataFrame | None = None
    if oxy_path.exists():
        oxy = pd.read_csv(oxy_path)
        oxy_var: str | None = None
        for c in oxy.columns:
            if str(c).upper() == "TIME" or c == "station":
                continue
            cu = str(c).upper()
            if any(k in cu for k in ("DOX2", "DOX", "OXY", "O2")):
                oxy_var = c
                break
        if oxy_var:
            oxy_clean = to_daily(oxy, "TIME", [oxy_var], "station")
            oxy_clean.columns = ["date", "station", "oxygen"]
        if oxy_clean is not None:
            oxy_clean.to_csv("data/processed/oxygen_daily.csv", index=False)
            _print_ok(
                f"Oxygen: {oxy_clean.shape} | range: {oxy_clean['date'].min()} -> {oxy_clean['date'].max()}"
            )
    else:
        _print_ok(f"Skip oxygen (not found): {oxy_path}")

    # --- CalCOFI larvae (optional) ---
    larvae_path = Path("data/raw/fish_larvae/Larvae.csv")
    if larvae_path.exists():
        larvae = pd.read_csv(larvae_path, skiprows=[1])
        larvae["date"] = pd.to_datetime(larvae["time"], errors="coerce").dt.date
        key_species = ["Engraulis mordax", "Sardinops sagax"]
        larvae_filtered = larvae[larvae["scientific_name"].isin(key_species)].copy()
        larvae_clean = larvae_filtered.groupby(["date", "scientific_name"], observed=False).agg(
            larvae_count=("larvae_count", "sum"),
            larvae_10m2=("larvae_10m2", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        ).reset_index()
        larvae_clean.to_csv("data/processed/calcofi_larvae_daily.csv", index=False)
        _print_ok(
            f"CalCOFI Larvae: {larvae_clean.shape} | range: {larvae_clean['date'].min()} -> {larvae_clean['date'].max()}"
        )
    else:
        _print_ok(f"Skip CalCOFI larvae (not found): {larvae_path}")

    # --- CalCOFI zooplankton (optional) ---
    zoo_path = Path("data/raw/zooplankton/Zooplankton.csv")
    if zoo_path.exists():
        zoo = pd.read_csv(zoo_path, skiprows=[1])
        zoo["date"] = pd.to_datetime(zoo["time"], errors="coerce").dt.date
        zoo_clean = zoo.groupby("date", observed=False).agg(
            total_plankton=("total_plankton", "mean"),
            small_plankton=("small_plankton", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        ).reset_index()
        zoo_clean.to_csv("data/processed/calcofi_zooplankton_daily.csv", index=False)
        _print_ok(
            f"Zooplankton: {zoo_clean.shape} | range: {zoo_clean['date'].min()} -> {zoo_clean['date'].max()}"
        )
    else:
        _print_ok(f"Skip zooplankton (not found): {zoo_path}")

    # --- Merge mooring daily ---
    print("\nMerging mooring data...")
    master = ph_clean.copy()
    for part in [ts_clean, no3_clean, chl_clean]:
        master = master.merge(part, on=["date", "station"], how="outer")
    if oxy_clean is not None:
        master = master.merge(oxy_clean, on=["date", "station"], how="outer")

    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values(["station", "date"]).reset_index(drop=True)
    master["mooring_id"] = master["station"].astype(str)
    # UTC midnight for Streamlit ``time`` + names expected by ``column_pick`` / dashboards
    master["time"] = pd.to_datetime(master["date"], utc=True)
    master = master.rename(
        columns={
            "ph": "ph_total",
            "temperature": "sst_c",
            "salinity": "salinity_psu",
            "nitrate": "no3",
            "chlorophyll": "chl_mg_m3",
        }
    )
    out_cols = [
        "time",
        "date",
        "station",
        "mooring_id",
        "ph_total",
        "sst_c",
        "salinity_psu",
        "no3",
        "chl_mg_m3",
    ]
    if "oxygen" in master.columns:
        out_cols.append("oxygen")
    master = master[out_cols]
    master.to_csv("data/processed/mooring_master.csv", index=False)
    _print_ok(f"Mooring master: {master.shape}")
    _print_ok(f"  Columns: {list(master.columns)}")
    _print_ok(f"  Date range: {master['date'].min()} -> {master['date'].max()}")
    _print_ok(f"  Stations: {master['station'].unique()}")
    for st in sorted(master["station"].unique()):
        sub = master.loc[master["station"] == st]
        _print_ok(
            f"  [{st}] rows={len(sub)} | {sub['date'].min().date()} -> {sub['date'].max().date()} | "
            f"sst non-null={pd.to_numeric(sub['sst_c'], errors='coerce').notna().sum()}"
        )
    print(f"\n  Sample:\n{master.head()}")
    print("\nDone. Cleaned files in data/processed/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
