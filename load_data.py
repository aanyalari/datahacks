import xarray as xr
import pandas as pd
import os

CCE1 = "https://dods.ndbc.noaa.gov/thredds/dodsC/oceansites/DATA/CCE1/"
CCE2 = "https://dods.ndbc.noaa.gov/thredds/dodsC/oceansites/DATA/CCE2/"

# Organized by variable type
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
        ]
    },
    "temperature_salinity": {
        "cce1": [
            "OS_CCE1_02_D_MICROCAT.nc", "OS_CCE1_03_D_MICROCAT.nc",
            "OS_CCE1_04_D_MICROCAT.nc",
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
        ]
    },
    "oxygen": {
        "cce1": [
            "OS_CCE1_09_D_OXYGEN.nc", "OS_CCE1_10_D_OXYGEN.nc",
            "OS_CCE1_11_D_OXYGEN.nc", "OS_CCE1_12_D_OXYGEN.nc",
            "OS_CCE1_13_D_OXYGEN.nc", "OS_CCE1_14_D_OXYGEN.nc",
            "OS_CCE1_15_D_OXYGEN.nc", "OS_CCE1_16_D_OXYGEN.nc",
            "OS_CCE1_17_D_OXYGEN.nc",
        ]
    },
}

def load_and_concat(base_url, files, variable=None):
    dfs = []
    for file in files:
        try:
            ds = xr.open_dataset(base_url + file)
            if variable:
                cols = [v for v in ds.data_vars if variable in v or v in ["PH_TOT", "TEMP", "PSAL", "NO3", "CHL", "DOX2"]]
                df = ds[cols].to_dataframe().reset_index() if cols else ds.to_dataframe().reset_index()
            else:
                df = ds.to_dataframe().reset_index()
            dfs.append(df)
            print(f"    ✓ {file.split('/')[-1]} — {len(df)} rows")
        except Exception as e:
            print(f"    ✗ {file} failed: {e}")
    return pd.concat(dfs).sort_values("TIME").reset_index(drop=True) if dfs else None

# Create folder structure
for var in datasets:
    os.makedirs(f"data/raw/{var}", exist_ok=True)

# Download and save everything
for var, sources in datasets.items():
    print(f"\n📂 {var.upper()}")
    all_dfs = []
    
    if "cce1" in sources:
        print("  CCE1:")
        df = load_and_concat(CCE1, sources["cce1"])
        if df is not None:
            df["station"] = "CCE1"
            df["lat"] = 33.42
            df["lon"] = -122.48
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