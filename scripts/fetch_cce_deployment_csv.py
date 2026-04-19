"""
Download public telemetered CSVs from a CCE deployment and merge into one file
for `data/raw/` (UTC `time` column + names expected by `cce_hack`).

Sensors do not share identical timestamps (wind vs temp vs CO2, etc.). Plain
outer-merge on `time` produces mostly empty cells. This script aligns every
series to the **temperature timeline** using merge_asof (backward), so each
row is filled with the most recent value from each instrument within a
tolerance window.

Example:
  python scripts/fetch_cce_deployment_csv.py \\
    --csv-base https://mooring.ucsd.edu/cce2/cce2_19/csv/ \\
    --mooring-id CCE2 \\
    --out data/raw/cce2_19_merged.csv

Find `--csv-base`: open https://mooring.ucsd.edu/cce/, pick mooring + deployment,
then use the deployment URL with `csv/` appended.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pandas as pd


def _read_remote_csv(base: str, name: str) -> pd.DataFrame:
    url = urljoin(base, name)
    with urlopen(url, timeout=60) as resp:
        raw = resp.read()
    return pd.read_csv(BytesIO(raw))


def _time_from_first_column(df: pd.DataFrame) -> pd.Series:
    tcol = df.columns[0]
    s = pd.to_numeric(df[tcol], errors="coerce")
    return pd.to_datetime(s, unit="ms", utc=True)


def _series_table(df: pd.DataFrame, value_map: dict[str, str]) -> pd.DataFrame:
    """time + renamed numeric columns, sorted, deduped on time (last wins)."""
    out = pd.DataFrame({"time": _time_from_first_column(df)})
    for src, dst in value_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")
    return out.drop_duplicates(subset=["time"], keep="last")


def _coalesce_depth_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    """Prefer shallow sensors; fall back to deeper bins when 1 m is missing (common on CCE)."""
    s: pd.Series | None = None
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        s = v if s is None else s.combine_first(v)
    if s is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return s


def _asof_backward(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Attach right columns to left using backward as-of merge on time."""
    rcols = [c for c in right.columns if c != "time"]
    if not rcols:
        return left
    r = right[["time"] + rcols].sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return pd.merge_asof(
        left.sort_values("time"),
        r,
        on="time",
        direction="backward",
        tolerance=tolerance,
    )


def _merge_frames(base: str, tolerance: pd.Timedelta) -> pd.DataFrame:
    temp = _read_remote_csv(base, "temp.csv")
    tdf = pd.DataFrame(
        {
            "time": _time_from_first_column(temp),
            "sst_c": _coalesce_depth_columns(
                temp, ("T_C_1m", "T_C_7m", "T_C_15m", "T_C_27m", "T_C_46m", "T_C_75m")
            ),
        }
    )
    left = tdf.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")

    sal = _read_remote_csv(base, "sal.csv")
    if any(c in sal.columns for c in ("S_1m", "S_7m", "S_15m")):
        salinity = _coalesce_depth_columns(sal, ("S_1m", "S_7m", "S_15m", "S_27m", "S_46m", "S_75m"))
        sal_df = pd.DataFrame({"time": _time_from_first_column(sal), "salinity_psu": salinity})
        sal_df = sal_df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        left = _asof_backward(left, sal_df, tolerance)

    wind = _read_remote_csv(base, "wind.csv")
    if "WindSpd_m/s" in wind.columns:
        left = _asof_backward(left, _series_table(wind, {"WindSpd_m/s": "wind_speed_ms"}), tolerance)

    ph = _read_remote_csv(base, "pH.csv")
    ph_val = pd.Series(np.nan, index=ph.index, dtype=float)
    for c in ("pH_1m", "pH-int_15m", "pH-ext_15m"):
        if c in ph.columns:
            ph_val = ph_val.combine_first(pd.to_numeric(ph[c], errors="coerce"))
    ph_df = pd.DataFrame({"time": _time_from_first_column(ph), "ph_total": ph_val})
    ph_df = ph_df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    left = _asof_backward(left, ph_df, tolerance)

    co2 = _read_remote_csv(base, "co2.csv")
    if "xCO2water" in co2.columns:
        left = _asof_backward(
            left,
            _series_table(co2, {"xCO2water": "pco2_uatm"}),
            tolerance,
        )

    air = _read_remote_csv(base, "airPT.csv")
    if "AirT_C" in air.columns:
        left = _asof_backward(left, _series_table(air, {"AirT_C": "air_temp_c"}), tolerance)

    chl = _read_remote_csv(base, "chl.csv")
    if "Chl_ug/l_1m" in chl.columns or "Chl_ug/l_15m" in chl.columns:
        ch_ug = _coalesce_depth_columns(
            chl,
            ("Chl_ug/l_1m", "Chl_ug/l_15m"),
        )
        ch = pd.DataFrame({"time": _time_from_first_column(chl), "chl_mg_m3": ch_ug / 1000.0})
        ch = ch.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        left = _asof_backward(left, ch, tolerance)

    return left.sort_values("time").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch CCE deployment CSVs and write one merged UTC file.")
    p.add_argument(
        "--csv-base",
        default="https://mooring.ucsd.edu/cce2/cce2_19/csv/",
        help="URL ending with .../csv/ for the deployment",
    )
    p.add_argument("--mooring-id", default="CCE2", help="Value stored in mooring_id column")
    p.add_argument(
        "--out",
        default="data/raw/cce_merged.csv",
        help="Output path (repo-root relative or absolute)",
    )
    p.add_argument(
        "--tolerance-hours",
        type=float,
        default=6.0,
        help="merge_asof window: each temp timestamp gets latest sensor value at or before time, "
        "if within this many hours (default 6). Increase if a channel is very sparse.",
    )
    args = p.parse_args()

    base = args.csv_base if args.csv_base.endswith("/") else args.csv_base + "/"
    tol = pd.Timedelta(hours=args.tolerance_hours)

    merged = _merge_frames(base, tol)
    merged["mooring_id"] = args.mooring_id
    cols = ["time", "mooring_id", "sst_c", "salinity_psu", "wind_speed_ms", "ph_total", "pco2_uatm", "air_temp_c", "chl_mg_m3"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[cols].sort_values("time")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    n = len(merged)
    cov = {c: float(merged[c].notna().mean()) for c in cols if c not in ("time", "mooring_id")}
    print(f"Wrote {out.resolve()} ({n} rows)")
    print("Non-null fraction per column:", ", ".join(f"{k}={v:.2f}" for k, v in cov.items()))
    print("Browse CSV listing:", urljoin(base, "."))


if __name__ == "__main__":
    main()
