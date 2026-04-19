"""Pick the best file column for a given measurement role (handles sparse multi-depth exports)."""

from __future__ import annotations

import pandas as pd

# Preference order: surface / canonical names first, then common depth suffixes.
_ROLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ph": ("ph_total", "pH_total", "ph", "pH", "ph_insitu"),
    "sst": (
        "sst_c",
        "temperature_c",
        "temp_c",
        "temperature",
        "t_c",
        "sst_c_d32m",
        "sst_c_d38m",
        "sst_c_d39m",
        "sst_c_d40m",
    ),
    "salinity": (
        "salinity_psu",
        "salinity",
        "practical_salinity",
        "salinity_psu_d32m",
        "salinity_psu_d38m",
        "salinity_psu_d39m",
        "salinity_psu_d40m",
    ),
    "chl": ("chl_mg_m3", "chlorophyll", "chl", "chl_mg_m3_d40m", "fluorescence"),
    "no3": ("no3", "nitrate", "nitrate_um", "NO3", "no3_uM"),
    "o2": ("dissolved_oxygen_mg_l", "do_mg_l", "oxygen_mg_l", "o2_mg_l", "o2_mgl", "oxygen"),
}


def pick_best_column(df: pd.DataFrame, role: str) -> str | None:
    """Return the candidate column with the **most** non-null numeric values in ``df``."""
    cands = _ROLE_CANDIDATES.get(role, ())
    best_c, best_n = None, 0
    for c in cands:
        if c not in df.columns:
            continue
        n = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        if n > best_n:
            best_c, best_n = c, n
    return best_c


def friendly_axis_label(col: str | None) -> str:
    """Human-readable axis / legend titles (not raw CSV tokens)."""
    if not col:
        return ""
    m = {
        "ph_total": "pH (total scale)",
        "sst_c": "Sea temperature (°C)",
        "salinity_psu": "Salinity (PSU)",
        "no3": "Nitrate (µM)",
        "chl_mg_m3": "Chlorophyll (mg/m³)",
        "wind_speed_ms": "Wind speed (m/s)",
        "conductivity_s_m": "Conductivity (S/m)",
        "pco2_uatm": "pCO₂ (µatm)",
        "dissolved_oxygen_mg_l": "Dissolved oxygen (mg/L)",
        "do_mg_l": "Dissolved oxygen (mg/L)",
        "oxygen_mg_l": "Dissolved oxygen (mg/L)",
        "o2_mg_l": "Dissolved oxygen (mg/L)",
        "oxygen": "Dissolved oxygen",
    }
    if col in m:
        return m[col]
    if col.startswith("sst_c_d") and col.endswith("m"):
        depth = col.replace("sst_c_d", "").replace("m", "")
        return f"Temperature at ~{depth} m (°C)"
    if col.startswith("salinity_psu_d") and col.endswith("m"):
        depth = col.replace("salinity_psu_d", "").replace("m", "")
        return f"Salinity at ~{depth} m (PSU)"
    if col.startswith("conductivity_s_m_d"):
        return "Conductivity (depth sensor)"
    return col.replace("_", " ")
