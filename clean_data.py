import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# ─── HELPER ────────────────────────────────────────────────────────
BOUNDS = {
    "PH_TOT":  (6.0, 9.0),
    "TEMP":    (-2.0, 35.0),
    "PSAL":    (0.0, 42.0),
    "NO3":     (0.0, 60.0),
    "CHL":     (0.0, 100.0),
    "DOX2":    (0.0, 600.0),
}

def add_date_parts(df, date_col="date"):
    d = pd.to_datetime(df[date_col])
    df.insert(df.columns.get_loc(date_col) + 1, "year", d.dt.year)
    df.insert(df.columns.get_loc(date_col) + 2, "month", d.dt.month)
    df.insert(df.columns.get_loc(date_col) + 3, "day", d.dt.day)
    return df

def to_daily(df, time_col, value_cols, station_col=None, qc_cols=None, depth_col="DEPTH"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Normalize station names before any grouping
    if station_col and station_col in df.columns:
        df[station_col] = df[station_col].astype(str).str.strip().str.upper()

    # QC flag filtering: Argo convention 1=good, 2=probably good
    if qc_cols:
        for qc_col in qc_cols:
            if qc_col in df.columns:
                df = df[df[qc_col].isin([1, 2]) | df[qc_col].isna()]

    # Filter to shallowest depth per station per day (sensor depth can shift over time)
    if depth_col and depth_col in df.columns:
        tmp_date = df[time_col].dt.date
        if station_col and station_col in df.columns:
            min_depth = df.groupby([df[station_col], tmp_date])[depth_col].transform("min")
        else:
            min_depth = df.groupby(tmp_date)[depth_col].transform("min")
        df = df[df[depth_col] == min_depth]

    # Physical bounds filtering
    for col in value_cols:
        if col in BOUNDS:
            lo, hi = BOUNDS[col]
            df = df[df[col].isna() | df[col].between(lo, hi)]

    # Deduplication on timestamp + station before aggregating
    dup_cols = [time_col] + ([station_col] if station_col and station_col in df.columns else [])
    df = df.drop_duplicates(subset=dup_cols)

    df["date"] = df[time_col].dt.date
    group_cols = ["date"] + ([station_col] if station_col else [])
    return df.groupby(group_cols)[value_cols].mean().reset_index()

# ─── PH ────────────────────────────────────────────────────────────
ph = pd.read_csv("data/raw/ph/ph_combined.csv")
ph_clean = to_daily(ph, "TIME", ["PH_TOT"], "station")
ph_clean.columns = ["date", "station", "ph"]
ph_clean = ph_clean.dropna(subset=["ph"])
ph_clean = add_date_parts(ph_clean)
ph_clean.to_csv("data/processed/ph_daily.csv", index=False)
print(f"✓ pH: {ph_clean.shape} | range: {ph_clean['date'].min()} → {ph_clean['date'].max()}")

# ─── TEMPERATURE & SALINITY ─────────────────────────────────────────
ts = pd.read_csv("data/raw/temperature_salinity/temperature_salinity_combined.csv")
# TEMP_QC and PSAL_QC are standard Argo integer flags: 1=good, 2=probably good
ts_clean = to_daily(ts, "TIME", ["TEMP", "PSAL"], "station", qc_cols=["TEMP_QC", "PSAL_QC"])
ts_clean.columns = ["date", "station", "temperature", "salinity"]
ts_clean = add_date_parts(ts_clean)
ts_clean.to_csv("data/processed/temp_salinity_daily.csv", index=False)
print(f"✓ Temp/Sal: {ts_clean.shape} | range: {ts_clean['date'].min()} → {ts_clean['date'].max()}")

# ─── NITRATE ───────────────────────────────────────────────────────
no3 = pd.read_csv("data/raw/nitrate/nitrate_combined.csv")
no3_clean = to_daily(no3, "TIME", ["NO3"], "station")
no3_clean.columns = ["date", "station", "nitrate"]
no3_clean = no3_clean.dropna(subset=["nitrate"])
no3_clean = add_date_parts(no3_clean)
no3_clean.to_csv("data/processed/nitrate_daily.csv", index=False)
print(f"✓ Nitrate: {no3_clean.shape} | range: {no3_clean['date'].min()} → {no3_clean['date'].max()}")

# ─── CHLOROPHYLL ───────────────────────────────────────────────────
chl = pd.read_csv("data/raw/chlorophyll/chlorophyll_combined.csv")
chl_clean = to_daily(chl, "TIME", ["CHL"], "station")
chl_clean.columns = ["date", "station", "chlorophyll"]
chl_clean = chl_clean.dropna(subset=["chlorophyll"])
chl_clean = add_date_parts(chl_clean)
chl_clean.to_csv("data/processed/chlorophyll_daily.csv", index=False)
print(f"✓ Chlorophyll: {chl_clean.shape} | range: {chl_clean['date'].min()} → {chl_clean['date'].max()}")

# ─── OXYGEN ────────────────────────────────────────────────────────
oxy = pd.read_csv("data/raw/oxygen/oxygen_combined.csv")
# DOX2_QC is a float (e.g. 2.5, 3.6) — scale is unknown, so skipping QC filter.
# Physical bounds in BOUNDS dict (0–600) handle obvious sensor errors instead.
oxy_clean = to_daily(oxy, "TIME", ["DOX2"], "station")
oxy_clean.columns = ["date", "station", "oxygen"]
oxy_clean = oxy_clean.dropna(subset=["oxygen"])
oxy_clean = add_date_parts(oxy_clean)
oxy_clean.to_csv("data/processed/oxygen_daily.csv", index=False)
print(f"✓ Oxygen: {oxy_clean.shape} | range: {oxy_clean['date'].min()} → {oxy_clean['date'].max()}")

# ─── CALCOFI LARVAE ────────────────────────────────────────────────
larvae = pd.read_csv("data/raw/fish_larvae/Larvae.csv", skiprows=[1])
larvae["date"] = pd.to_datetime(larvae["time"], errors="coerce").dt.date
larvae = larvae.dropna(subset=["date"])

key_species = ["Engraulis mordax", "Sardinops sagax"]
larvae_filtered = larvae[larvae["scientific_name"].isin(key_species)].dropna(subset=["larvae_10m2"]).copy()

# larvae_count is summed (total catch), larvae_10m2 is averaged (standardized density)
# proportion_sorted < 1 means only a subsample was identified — larvae_10m2 already
# accounts for this via standard_haul_factor, so it is the correct column for analysis
larvae_clean = larvae_filtered.groupby(["date", "scientific_name"]).agg(
    larvae_count=("larvae_count", "sum"),
    larvae_10m2=("larvae_10m2", "mean"),
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean")
).reset_index()
larvae_clean = add_date_parts(larvae_clean)
larvae_clean.to_csv("data/processed/calcofi_larvae_daily.csv", index=False)
print(f"✓ CalCOFI Larvae: {larvae_clean.shape} | range: {larvae_clean['date'].min()} → {larvae_clean['date'].max()}")

# ─── CALCOFI ZOOPLANKTON ───────────────────────────────────────────
zoo = pd.read_csv("data/raw/zooplankton/Zooplankton.csv", skiprows=[1])
zoo["date"] = pd.to_datetime(zoo["time"], errors="coerce").dt.date
zoo = zoo.dropna(subset=["date"])
# NOTE: zoo 'station' is a CalCOFI cruise grid number (35.0, 40.0…), not a mooring ID.
# It cannot join with mooring_master on station. Aggregating by date only.
zoo_clean = zoo.groupby("date").agg(
    total_plankton=("total_plankton", "mean"),
    small_plankton=("small_plankton", "mean"),
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean")
).reset_index()
zoo_clean = add_date_parts(zoo_clean)
zoo_clean.to_csv("data/processed/calcofi_zooplankton_daily.csv", index=False)
print(f"✓ Zooplankton: {zoo_clean.shape} | range: {zoo_clean['date'].min()} → {zoo_clean['date'].max()}")

# ─── MERGE ALL MOORING DATA INTO MASTER FILE ───────────────────────
print("\nMerging mooring data...")
drop_dp = lambda df: df.drop(columns=[c for c in ["year", "month", "day"] if c in df.columns])
master = drop_dp(ph_clean).copy()
for df in [ts_clean, no3_clean, chl_clean, oxy_clean]:
    master = master.merge(drop_dp(df), on=["date", "station"], how="outer")

master["date"] = pd.to_datetime(master["date"])
master = master.sort_values(["station", "date"]).reset_index(drop=True)
master = add_date_parts(master)
master.to_csv("data/processed/mooring_master.csv", index=False)
print(f"✓ Mooring master: {master.shape}")
print(f"  Columns: {list(master.columns)}")
print(f"  Date range: {master['date'].min()} → {master['date'].max()}")
print(f"  Stations: {master['station'].unique()}")

# Coverage report: date range and missingness per variable
print("\n  Variable coverage:")
value_cols = ["ph", "temperature", "salinity", "nitrate", "chlorophyll", "oxygen"]
for col in value_cols:
    valid = master[col].dropna()
    if valid.empty:
        print(f"    {col}: NO DATA")
    else:
        pct_missing = 100 * master[col].isna().sum() / len(master)
        print(f"    {col}: {master.loc[valid.index, 'date'].min().date()} → {master.loc[valid.index, 'date'].max().date()} ({pct_missing:.0f}% missing)")

print(f"\n  Sample:\n{master.head()}")
print("\n✅ All cleaned files in data/processed/")
