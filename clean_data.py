import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# ─── HELPER ────────────────────────────────────────────────────────
def to_daily(df, time_col, value_cols, station_col=None):
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["date"] = df[time_col].dt.date
    group_cols = ["date"] + ([station_col] if station_col else [])
    return df.groupby(group_cols)[value_cols].mean().reset_index()

# ─── PH ────────────────────────────────────────────────────────────
ph = pd.read_csv("data/raw/ph/ph_combined.csv")
ph_clean = to_daily(ph, "TIME", ["PH_TOT"], "station")
ph_clean.columns = ["date", "station", "ph"]
ph_clean.to_csv("data/processed/ph_daily.csv", index=False)
print(f"✓ pH: {ph_clean.shape} | range: {ph_clean['date'].min()} → {ph_clean['date'].max()}")

# ─── TEMPERATURE & SALINITY ─────────────────────────────────────────
ts = pd.read_csv("data/raw/temperature_salinity/temperature_salinity_combined.csv")
ts_clean = to_daily(ts, "TIME", ["TEMP", "PSAL"], "station")
ts_clean.columns = ["date", "station", "temperature", "salinity"]
ts_clean.to_csv("data/processed/temp_salinity_daily.csv", index=False)
print(f"✓ Temp/Sal: {ts_clean.shape} | range: {ts_clean['date'].min()} → {ts_clean['date'].max()}")

# ─── NITRATE ───────────────────────────────────────────────────────
no3 = pd.read_csv("data/raw/nitrate/nitrate_combined.csv")
no3_clean = to_daily(no3, "TIME", ["NO3"], "station")
no3_clean.columns = ["date", "station", "nitrate"]
no3_clean.to_csv("data/processed/nitrate_daily.csv", index=False)
print(f"✓ Nitrate: {no3_clean.shape} | range: {no3_clean['date'].min()} → {no3_clean['date'].max()}")

# ─── CHLOROPHYLL ───────────────────────────────────────────────────
chl = pd.read_csv("data/raw/chlorophyll/chlorophyll_combined.csv")
chl_clean = to_daily(chl, "TIME", ["CHL"], "station")
chl_clean.columns = ["date", "station", "chlorophyll"]
chl_clean.to_csv("data/processed/chlorophyll_daily.csv", index=False)
print(f"✓ Chlorophyll: {chl_clean.shape} | range: {chl_clean['date'].min()} → {chl_clean['date'].max()}")

# ─── OXYGEN ────────────────────────────────────────────────────────
oxy = pd.read_csv("data/raw/oxygen/oxygen_combined.csv")
oxy_clean = to_daily(oxy, "TIME", ["DOX2"], "station")
oxy_clean.columns = ["date", "station", "oxygen"]
oxy_clean.to_csv("data/processed/oxygen_daily.csv", index=False)
print(f"✓ Oxygen: {oxy_clean.shape} | range: {oxy_clean['date'].min()} → {oxy_clean['date'].max()}")

# ─── CALCOFI LARVAE ────────────────────────────────────────────────
larvae = pd.read_csv("data/raw/fish_larvae/Larvae.csv", skiprows=[1])
larvae["date"] = pd.to_datetime(larvae["time"], errors="coerce").dt.date

key_species = ["Engraulis mordax", "Sardinops sagax"]
larvae_filtered = larvae[larvae["scientific_name"].isin(key_species)].copy()

larvae_clean = larvae_filtered.groupby(["date", "scientific_name"]).agg(
    larvae_count=("larvae_count", "sum"),
    larvae_10m2=("larvae_10m2", "mean"),
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean")
).reset_index()
larvae_clean.to_csv("data/processed/calcofi_larvae_daily.csv", index=False)
print(f"✓ CalCOFI Larvae: {larvae_clean.shape} | range: {larvae_clean['date'].min()} → {larvae_clean['date'].max()}")

# ─── CALCOFI ZOOPLANKTON ───────────────────────────────────────────
zoo = pd.read_csv("data/raw/zooplankton/Zooplankton.csv", skiprows=[1])
zoo["date"] = pd.to_datetime(zoo["time"], errors="coerce").dt.date
zoo_clean = zoo.groupby("date").agg(
    total_plankton=("total_plankton", "mean"),
    small_plankton=("small_plankton", "mean"),
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean")
).reset_index()
zoo_clean.to_csv("data/processed/calcofi_zooplankton_daily.csv", index=False)
print(f"✓ Zooplankton: {zoo_clean.shape} | range: {zoo_clean['date'].min()} → {zoo_clean['date'].max()}")

# ─── MERGE ALL MOORING DATA INTO MASTER FILE ───────────────────────
print("\nMerging mooring data...")
master = ph_clean.copy()
for df in [ts_clean, no3_clean, chl_clean, oxy_clean]:
    master = master.merge(df, on=["date", "station"], how="outer")

master["date"] = pd.to_datetime(master["date"])
master = master.sort_values(["station", "date"]).reset_index(drop=True)
master.to_csv("data/processed/mooring_master.csv", index=False)
print(f"✓ Mooring master: {master.shape}")
print(f"  Columns: {list(master.columns)}")
print(f"  Date range: {master['date'].min()} → {master['date'].max()}")
print(f"  Stations: {master['station'].unique()}")
print(f"\n  Sample:\n{master.head()}")

print("\n✅ All cleaned files in data/processed/")