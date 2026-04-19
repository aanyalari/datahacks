"""
Download OceanSITES CCE1/CCE2 NetCDF via OPeNDAP (dodsC), flatten to CSV per variable group.

Requires: pip install xarray netCDF4

Run from repo root:
  python scripts/download_oceansites_by_variable.py
  python scripts/download_oceansites_by_variable.py --variables ph --stations cce1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CCE1 = "https://dods.ndbc.noaa.gov/thredds/dodsC/oceansites/DATA/CCE1/"
CCE2 = "https://dods.ndbc.noaa.gov/thredds/dodsC/oceansites/DATA/CCE2/"

# CCE2 pH files use P_PH-15m (not 40m) for most deployments on this catalog.
CCE2_PH = [f"OS_CCE2_{i:02d}_P_PH-15m.nc" for i in range(2, 18)]
CCE2_NO3 = [f"OS_CCE2_{i:02d}_D_NO3.nc" for i in range(1, 18)]
CCE2_CHL = [f"OS_CCE2_{i:02d}_D_CHL.nc" for i in range(1, 18)]
CCE2_OXY = [f"OS_CCE2_{i:02d}_D_OXYGEN.nc" for i in range(5, 18)]
CCE2_CTD = [f"OS_CCE2_{i:02d}_D_CTD.nc" for i in range(7, 18)]

CCE2_MICROCAT = [
    "OS_CCE2_01_D_MICROCAT.nc",
    "OS_CCE2_02_D_MICROCAT-PART1.nc",
    "OS_CCE2_02_D_MICROCAT-PART2.nc",
    "OS_CCE2_02_D_MICROCAT-PART3.nc",
    "OS_CCE2_02_D_MICROCAT-PART4.nc",
    "OS_CCE2_03_D_MICROCAT-PART1.nc",
    "OS_CCE2_03_D_MICROCAT-PART2.nc",
    "OS_CCE2_03_D_MICROCAT-PART3.nc",
    "OS_CCE2_03_D_MICROCAT-PART4.nc",
    "OS_CCE2_04_D_MICROCAT-PART1.nc",
    "OS_CCE2_04_D_MICROCAT-PART2.nc",
    "OS_CCE2_04_D_MICROCAT-PART3.nc",
    "OS_CCE2_04_D_MICROCAT-PART4.nc",
]

datasets: dict[str, dict[str, list[str]]] = {
    "ph": {
        "cce1": [
            "OS_CCE1_03_P_PH-40m.nc",
            "OS_CCE1_04_P_PH-40m.nc",
            "OS_CCE1_05_P_PH-40m.nc",
            "OS_CCE1_07_P_PH-40m.nc",
            "OS_CCE1_09_P_PH-40m.nc",
            "OS_CCE1_10_P_PH-40m.nc",
            "OS_CCE1_11_P_PH-40m.nc",
            "OS_CCE1_12_P_PH-40m.nc",
            "OS_CCE1_13_P_PH-40m.nc",
            "OS_CCE1_14_P_PH-40m.nc",
            "OS_CCE1_15_P_PH-40m.nc",
            "OS_CCE1_16_P_PH-40m.nc",
            "OS_CCE1_17_P_PH-40m.nc",
        ],
        "cce2": CCE2_PH,
    },
    "temperature_salinity": {
        "cce1": [
            "OS_CCE1_02_D_MICROCAT.nc",
            "OS_CCE1_03_D_MICROCAT.nc",
            "OS_CCE1_04_D_MICROCAT.nc",
        ],
        "cce2": CCE2_MICROCAT + CCE2_CTD,
    },
    "nitrate": {
        "cce1": [
            "OS_CCE1_02_D_NO3.nc",
            "OS_CCE1_04_D_NO3.nc",
            "OS_CCE1_05_D_NO3.nc",
            "OS_CCE1_07_P_NO3.nc",
            "OS_CCE1_09_D_NO3.nc",
            "OS_CCE1_10_D_NO3.nc",
            "OS_CCE1_11_D_NO3.nc",
            "OS_CCE1_13_D_NO3.nc",
            "OS_CCE1_14_D_NO3.nc",
            "OS_CCE1_15_D_NO3.nc",
            "OS_CCE1_16_D_NO3.nc",
            "OS_CCE1_17_D_NO3.nc",
        ],
        "cce2": CCE2_NO3,
    },
    "chlorophyll": {
        "cce1": [
            "OS_CCE1_03_D_CHL.nc",
            "OS_CCE1_04_D_CHL.nc",
            "OS_CCE1_05_D_CHL.nc",
            "OS_CCE1_07_D_CHL.nc",
            "OS_CCE1_09_D_CHL.nc",
            "OS_CCE1_10_D_CHL.nc",
            "OS_CCE1_11_D_CHL.nc",
            "OS_CCE1_13_D_CHL.nc",
            "OS_CCE1_14_D_CHL.nc",
            "OS_CCE1_15_D_CHL.nc",
            "OS_CCE1_16_D_CHL.nc",
            "OS_CCE1_17_D_CHL.nc",
        ],
        "cce2": CCE2_CHL,
    },
    "oxygen": {
        "cce1": [
            "OS_CCE1_09_D_OXYGEN.nc",
            "OS_CCE1_10_D_OXYGEN.nc",
            "OS_CCE1_11_D_OXYGEN.nc",
            "OS_CCE1_12_D_OXYGEN.nc",
            "OS_CCE1_13_D_OXYGEN.nc",
            "OS_CCE1_14_D_OXYGEN.nc",
            "OS_CCE1_15_D_OXYGEN.nc",
            "OS_CCE1_16_D_OXYGEN.nc",
            "OS_CCE1_17_D_OXYGEN.nc",
        ],
        "cce2": CCE2_OXY,
    },
}


def _time_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).upper() == "TIME":
            return c
    return None


def _lat_lon_from_ds(ds: xr.Dataset) -> tuple[float, float]:
    if "LATITUDE" in ds.coords and "LONGITUDE" in ds.coords:
        try:
            la = float(np.asarray(ds["LATITUDE"].values).reshape(-1)[0])
            lo = float(np.asarray(ds["LONGITUDE"].values).reshape(-1)[0])
            return la, lo
        except Exception:
            pass
    return float("nan"), float("nan")


def load_and_concat(base_url: str, files: list[str], station: str) -> pd.DataFrame | None:
    dfs: list[pd.DataFrame] = []
    for fn in files:
        url = base_url + fn
        try:
            with xr.open_dataset(url, decode_times=True) as ds:
                ds = ds.load()
                lat, lon = _lat_lon_from_ds(ds)
                df = ds.to_dataframe().reset_index()
                df["source_file"] = fn
                df["station"] = station
                df["lat"] = lat
                df["lon"] = lon
            dfs.append(df)
            print(f"    OK {fn}  rows={len(df):,}")
        except Exception as e:
            print(f"    FAIL {fn}: {e}")
    if not dfs:
        return None
    out = pd.concat(dfs, ignore_index=True, sort=False)
    tc = _time_col(out)
    if tc:
        out = out.sort_values(tc).reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variables",
        default="all",
        help="Comma-separated keys from datasets, or 'all' (default: all).",
    )
    p.add_argument(
        "--stations",
        default="cce1,cce2",
        help="Comma-separated: cce1, cce2, or both (default both).",
    )
    args = p.parse_args()
    want_vars = set(datasets.keys()) if args.variables.strip().lower() == "all" else {v.strip() for v in args.variables.split(",") if v.strip()}
    want_st = {s.strip().lower() for s in args.stations.split(",") if s.strip()}

    for var, sources in datasets.items():
        if var not in want_vars:
            continue
        out_dir = ROOT / "data" / "raw" / var
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{var.upper()}]")
        all_dfs: list[pd.DataFrame] = []

        if "cce1" in want_st and "cce1" in sources:
            print("  CCE1:")
            df = load_and_concat(CCE1, sources["cce1"], "CCE1")
            if df is not None:
                all_dfs.append(df)

        if "cce2" in want_st and "cce2" in sources:
            print("  CCE2:")
            df = load_and_concat(CCE2, sources["cce2"], "CCE2")
            if df is not None:
                all_dfs.append(df)

        if not all_dfs:
            print("  (no data saved)")
            continue

        combined = pd.concat(all_dfs, ignore_index=True, sort=False)
        tc = _time_col(combined)
        if tc:
            combined = combined.sort_values(tc).reset_index(drop=True)

        out_path = out_dir / f"{var}_combined.csv"
        combined.to_csv(out_path, index=False)
        tmin = combined[tc].min() if tc else None
        tmax = combined[tc].max() if tc else None
        print(f"  SAVED {out_path.relative_to(ROOT)}  rows={len(combined):,}  TIME {tmin} -> {tmax}")


if __name__ == "__main__":
    main()
