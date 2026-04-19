import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "ocean_health"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS mooring_master (
    date        DATE             NOT NULL,
    year        SMALLINT,
    month       SMALLINT,
    day         SMALLINT,
    station     VARCHAR(20)      NOT NULL,
    ph          DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    salinity    DOUBLE PRECISION,
    nitrate     DOUBLE PRECISION,
    chlorophyll DOUBLE PRECISION,
    oxygen      DOUBLE PRECISION,
    PRIMARY KEY (date, station)
);

CREATE TABLE IF NOT EXISTS calcofi_larvae (
    date            DATE            NOT NULL,
    year            SMALLINT,
    month           SMALLINT,
    day             SMALLINT,
    scientific_name VARCHAR(100)    NOT NULL,
    larvae_count    DOUBLE PRECISION,
    larvae_10m2     DOUBLE PRECISION,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    PRIMARY KEY (date, scientific_name)
);

CREATE TABLE IF NOT EXISTS calcofi_zooplankton (
    date           DATE NOT NULL PRIMARY KEY,
    year           SMALLINT,
    month          SMALLINT,
    day            SMALLINT,
    total_plankton DOUBLE PRECISION,
    small_plankton DOUBLE PRECISION,
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_mooring_station ON mooring_master(station);
CREATE INDEX IF NOT EXISTS idx_mooring_date    ON mooring_master(date);
CREATE INDEX IF NOT EXISTS idx_mooring_ph      ON mooring_master(ph);
CREATE INDEX IF NOT EXISTS idx_larvae_date     ON calcofi_larvae(date);
CREATE INDEX IF NOT EXISTS idx_larvae_species  ON calcofi_larvae(scientific_name);
CREATE INDEX IF NOT EXISTS idx_zoo_date        ON calcofi_zooplankton(date);
"""

def to_rows(df, cols):
    df = df[cols].copy()
    return [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False)
    ]

def load_table(cur, csv_path, table, cols):
    df = pd.read_csv(csv_path)
    rows = to_rows(df, cols)
    cur.execute(f"TRUNCATE {table}")
    execute_values(cur, f"INSERT INTO {table} ({','.join(cols)}) VALUES %s", rows)
    print(f"  ✓ {table}: {len(rows)} rows loaded")

def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    cur.execute(SCHEMA)
    print("✓ Schema ready")

    load_table(cur, "data/processed/mooring_master.csv", "mooring_master",
               ["date", "year", "month", "day", "station",
                "ph", "temperature", "salinity", "nitrate", "chlorophyll", "oxygen"])

    load_table(cur, "data/processed/calcofi_larvae_daily.csv", "calcofi_larvae",
               ["date", "year", "month", "day", "scientific_name",
                "larvae_count", "larvae_10m2", "latitude", "longitude"])

    load_table(cur, "data/processed/calcofi_zooplankton_daily.csv", "calcofi_zooplankton",
               ["date", "year", "month", "day",
                "total_plankton", "small_plankton", "latitude", "longitude"])

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n✅ Database ready → {DB['host']}:{DB['port']}/{DB['dbname']}")

if __name__ == "__main__":
    main()
