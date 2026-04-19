"""Build `data/processed/cce_hourly_panel.csv` from `data/raw/*/` combined exports."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cce_hack.ingest_raw import build_hourly_panel, write_hourly_panel  # noqa: E402


def main() -> None:
    out = write_hourly_panel()
    df = pd.read_csv(out)
    print("Wrote:", out)
    print("Rows:", len(df), "Moorings:", df["mooring_id"].unique().tolist())
    print(df.describe(numeric_only=True).T.head(12))


if __name__ == "__main__":
    main()
