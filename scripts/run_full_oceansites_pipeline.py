#!/usr/bin/env python3
"""
Full pipeline: download ALL OceanSITES groups (pH, T/S, nitrate, chlorophyll, oxygen)
for CCE1+CCE2, then rebuild hourly panel + daily mooring master.

From repo root:
  python scripts/run_full_oceansites_pipeline.py

Requires: pip install -e ".[netcdf]"   OR   pip install xarray netCDF4 pandas
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    py = sys.executable
    run([py, str(ROOT / "scripts" / "download_oceansites_by_variable.py")])
    run([py, str(ROOT / "scripts" / "build_processed_panel.py")])
    run([py, str(ROOT / "scripts" / "process_mooring_daily.py")])
    print("\nNext: streamlit run Home.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
