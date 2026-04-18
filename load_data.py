import xarray as xr
import pandas as pd
import os

CCE1 = "https://dods.ndbc.noaa.gov/thredds/dodsC/oceansites/DATA/CCE1/"
CCE2 = "https://dods.ndbc.noaa.gov/thredds/dodsC/oceansites/DATA/CCE2/"

datasets = {
    "ph": {
        "cce1": [
            "OS_CCE1_03_P_PH-40m.nc", "OS_CCE1_04_P_PH-40m.nc",
            "OS_CCE1_05_P_PH-40m.nc", "OS_CCE1_07_P_PH-40m.nc",
            "OS_CCE1_09_P_PH-40m.nc", "OS_CCE1_10_P_PH-40m.nc",
            "OS_CCE1_11_P_PH-40m.nc", "OS_CCE1_12_P_PH-40m.nc",
            "OS_CCE1_13_P_PH-40m.nc", "OS_CCE1_14_P_PH-40m.nc",
            "OS_CCE1_15_P_PH-40m.nc", "OS_CCE1_16_P_PH-40m.nc",
            "OS_CCE1_17_P_PH-40m.nc",
        ],
        "cce2": [
            "OS_CCE2_02_P_PH-15m.nc", "OS_CCE2_03_P_PH-15m.nc",
            "OS_CCE2_04_P_PH-15m.nc", "OS_CCE2_05_P_PH-15m.nc",
            "OS_CCE2_06_P_PH-15m.nc", "OS_CCE2_07_P_PH-15m.nc",
            "OS_CCE2_08_P_PH-15m.nc", "OS_CCE2_09_P_PH-15m.nc",
            "OS_CCE2_10_P_PH-15m.nc", "OS_CCE2_11_P_PH-15m.nc",
            "OS_CCE2_12_P_PH-15m.nc", "OS_CCE2_13_P_PH-15m.nc",
            "OS_CCE2_14_P_PH-15m.nc", "OS_CCE2_15_P_PH-15m.nc",
            "OS_CCE2_17_P_PH-15m.nc",
        ]
    },
    "temperature_salinity": {
        "cce1": [
            "OS_CCE1_02_D_MICROCAT.nc", "OS_CCE1_03_D_MICROCAT.nc",
            "OS_CCE1_04_D_MICROCAT.nc",
        ],
        "cce2": [
            "OS_CCE2_01_D_MICROCAT.nc",
            "OS_CCE2_02_D_MICROCAT-PART1.nc", "OS_CCE2_02_D_MICROCAT-PART2.nc",
            "OS_CCE2_03_D_MICROCAT-PART1.nc", "OS_CCE2_03_D_MICROCAT-PART2.nc",
            "OS_CCE2_04_D_MICROCAT-PART1.nc", "OS_CCE2_04_D_MICROCAT-PART2.nc",
        ]
    },
    "nitrate": {
        "cce1": [
            "OS_CCE1_02_D_NO3.nc", "OS_CCE1_04_D_NO3.nc",
            "OS_CCE1_05_D_NO3.nc", "OS_CCE1_07_P_NO3.nc",
            "OS_CCE1_09_D_NO3.nc", "OS_CCE1_10_D_NO3.nc",
            "OS_CCE1_11_D_NO3.nc", "OS_CCE1_13_D_NO3.nc",
            "OS_CCE1_14_D_NO3.nc", "OS_CCE1_15_D_NO3.nc",
            "OS_CCE1_16_D_NO3.nc", "OS_CCE1_17_D_NO3.nc",
        ],
        "cce2": [
            "OS_CCE2_01_D_NO3.nc", "OS_CCE2_02_D_NO3.nc",
            "OS_CCE2_03_D_NO3.nc", "OS_CCE2_04_D_NO3.nc",
            "OS_CCE2_05_D_NO3.nc", "OS_CCE2_06_P_NO3.nc",
            "OS_CCE2_07_D_NO3.nc", "OS_CCE2_08_D_NO3.nc",
            "OS_CCE2_09_D_NO3.nc", "OS_CCE2_10_D_NO3.nc",
            "OS_CCE2_11_D_NO3.nc", "OS_CCE2_12_D_NO3.nc",
            "OS_CCE2_13_D_NO3.nc", "OS_CCE2_14_D_NO3.nc",
            "OS_CCE2_15_D_NO3.nc", "OS_CCE2_16_D_NO3.nc",
            "OS_CCE2_17_D_NO3.nc",
        ]
    },
    "chlorophyll": {
        "cce1": [
            "OS_CCE1_03_D_CHL.nc", "OS_CCE1_04_D_CHL.nc",
            "OS_CCE1_05_D_CHL.nc", "OS_CCE1_07_D_CHL.nc",
            "OS_CCE1_09_D_CHL.nc", "OS_CCE1_10_D_CHL.nc",
            "OS_CCE1_11_D_CHL.nc", "OS_CCE1_13_D_CHL.nc",
            "OS_CCE1_14_D_CHL.nc", "OS_CCE1_15_D_CHL.nc",
            "OS_CCE1_16_D_CHL.nc", "OS_CCE1_17_D_CHL.nc",
        ],
        "cce2": [
            "OS_CCE2_01_D_CHL.nc", "OS_CCE2_02_D_CHL.nc",
            "OS_CCE2_03_D_CHL.nc", "OS_CCE2_04_D_CHL.nc",
            "OS_CCE2_05_D_CHL.nc", "OS_CCE2_06_D_CHL.nc",
            "OS_CCE2_07_D_CHL.nc", "OS_CCE2_08_D_CHL.nc",
            "OS_CCE2_09_D_CHL.nc", "OS_CCE2_10_D_CHL.nc",
            "OS_CCE2_11_D_CHL.nc", "OS_CCE2_12_D_CHL.nc",
            "OS_CCE2_13_D_CHL.nc", "OS_CCE2_14_D_CHL.nc",
            "OS_CCE2_15_D_CHL.nc", "OS_CCE2_16_D_CHL.nc",
            "OS_CCE2_17_D_CHL.nc",
        ]
    },
    # OXYGEN NOT HERE — handled separately below
}

# ─── STANDARD LOADER ───────────────────────────────────────────────
def load_and_concat(base_url, files):
    dfs = []
    for file in files:
        try:
            ds = xr.open_dataset(base_url + file)
            df = ds.to_dataframe().reset_index()
            dfs.append(df)
            print(f"    ✓ {file.split('/')[-1]} — {len(df)} rows")
        except Exception as e:
            print(f"    ✗ {file} failed: {e}")
    return pd.concat(dfs).sort_values("TIME").reset_index(drop=True) if dfs else None

# ─── OXYGEN LOADER (aggregated to daily) ───────────────────────────
def load_oxygen_aggregated(base_url, files, station_name):
    dfs = []
    for file in files:
        try:
            print(f"    Loading {file.split('/')[-1]}...")
            ds = xr.open_dataset(base_url + file)
            oxygen_vars = [v for v in ds.data_vars if "DOX" in v or "OXY" in v or "O2" in v]
            print(f"      Oxygen vars found: {oxygen_vars}")
            if not oxygen_vars:
                print(f"      ✗ No oxygen variable found, skipping")
                continue
            ds_surface = ds.isel(DEPTH=0) if "DEPTH" in ds.dims else ds
            df = ds_surface[oxygen_vars].to_dataframe().reset_index()
            df["TIME"] = pd.to_datetime(df["TIME"])
            df = df.groupby(pd.Grouper(key="TIME", freq="D")).mean().reset_index()
            df["station"] = station_name
            dfs.append(df)
            print(f"      ✓ {file.split('/')[-1]} — {len(df)} rows after daily aggregation")
        except Exception as e:
            print(f"      ✗ {file} failed: {e}")
    return pd.concat(dfs).sort_values("TIME").reset_index(drop=True) if dfs else None

# ─── CREATE FOLDER STRUCTURE ────────────────────────────────────────
for var in list(datasets.keys()) + ["oxygen"]:
    os.makedirs(f"data/raw/{var}", exist_ok=True)

# ─── LOAD + SAVE STANDARD VARIABLES ────────────────────────────────
for var, sources in datasets.items():
    print(f"\n📂 {var.upper()}")
    all_dfs = []

    if "cce1" in sources:
        print("  CCE1:")
        df = load_and_concat(CCE1, sources["cce1"])
        if df is not None:
            df["station"] = "CCE1"
            all_dfs.append(df)

    if "cce2" in sources:
        print("  CCE2:")
        df = load_and_concat(CCE2, sources["cce2"])
        if df is not None:
            df["station"] = "CCE2"
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs).sort_values("TIME").reset_index(drop=True)
        out = f"data/raw/{var}/{var}_combined.csv"
        combined.to_csv(out, index=False)
        print(f"  ✓ Saved {out} — {len(combined)} rows, {combined['TIME'].min()} → {combined['TIME'].max()}")

# ─── LOAD + SAVE OXYGEN (special handling, both stations) ──────────
print("\n📂 OXYGEN (aggregated to daily)")

cce1_oxy = load_oxygen_aggregated(CCE1, [
    "OS_CCE1_09_D_OXYGEN.nc", "OS_CCE1_10_D_OXYGEN.nc",
    "OS_CCE1_11_D_OXYGEN.nc", "OS_CCE1_12_D_OXYGEN.nc",
    "OS_CCE1_13_D_OXYGEN.nc", "OS_CCE1_14_D_OXYGEN.nc",
    "OS_CCE1_15_D_OXYGEN.nc", "OS_CCE1_16_D_OXYGEN.nc",
    "OS_CCE1_17_D_OXYGEN.nc",
], "CCE1")

# CCE2 oxygen files are small (hundreds of KB) — safe to load normally
cce2_oxy = load_oxygen_aggregated(CCE2, [
    "OS_CCE2_05_D_OXYGEN.nc", "OS_CCE2_06_D_OXYGEN.nc",
    "OS_CCE2_07_D_OXYGEN.nc", "OS_CCE2_08_D_OXYGEN.nc",
    "OS_CCE2_09_D_OXYGEN.nc", "OS_CCE2_10_D_OXYGEN.nc",
    "OS_CCE2_11_D_OXYGEN.nc", "OS_CCE2_12_D_OXYGEN.nc",
    "OS_CCE2_13_D_OXYGEN.nc", "OS_CCE2_14_D_OXYGEN.nc",
    "OS_CCE2_15_D_OXYGEN.nc", "OS_CCE2_16_D_OXYGEN.nc",
    "OS_CCE2_17_D_OXYGEN.nc",
], "CCE2")

oxygen_dfs = [df for df in [cce1_oxy, cce2_oxy] if df is not None]
if oxygen_dfs:
    oxygen_combined = pd.concat(oxygen_dfs).sort_values("TIME").reset_index(drop=True)
    oxygen_combined.to_csv("data/raw/oxygen/oxygen_combined.csv", index=False)
    print(f"  ✓ Saved oxygen_combined.csv — {len(oxygen_combined)} rows")

print("\n✅ All done! Check data/raw/ for your CSVs")