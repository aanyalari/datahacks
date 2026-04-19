"""Ocean acidification metrics via PyCO2SYS (assumed alkalinity for dual-parameter solve)."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import PyCO2SYS as pyco2

    _HAS_PYCO2 = True
except ImportError:
    pyco2 = None
    _HAS_PYCO2 = False


def run_co2sys_on_dataframe(
    df: pd.DataFrame,
    ta_umolkg: float = 2300.0,
    pressure_dbar: float = 40.0,
    salinity_col: str = "salinity_psu",
    temp_col: str = "sst_c",
    ph_col: str = "ph_total",
) -> pd.DataFrame | None:
    """
    Compute saturation aragonite, Revelle factor, k_aragonite, pCO2 from pH (total) + assumed TA.

    Practical salinity and in-situ temperature are required on the same timestamps as pH.
    """
    if not _HAS_PYCO2:
        return None
    need = [ph_col, temp_col, salinity_col]
    if not all(c in df.columns for c in need):
        return None
    d = df[["time"] + need].dropna().sort_values("time")
    if len(d) < 10:
        return None
    n = len(d)
    ta = np.full(n, float(ta_umolkg))
    ph = d[ph_col].to_numpy(dtype=np.float64)
    sal = d[salinity_col].to_numpy(dtype=np.float64)
    temp = d[temp_col].to_numpy(dtype=np.float64)
    pres = np.full(n, float(pressure_dbar))
    try:
        r = pyco2.sys(
            par1=ta,
            par2=ph,
            par1_type=1,
            par2_type=3,
            salinity=sal,
            temperature=temp,
            pressure=pres,
        )
    except Exception:
        return None
    out = d[["time"]].copy()
    out["saturation_aragonite"] = np.asarray(r["saturation_aragonite"], dtype=float).ravel()
    out["revelle_factor"] = np.asarray(r["revelle_factor"], dtype=float).ravel()
    out["k_aragonite"] = np.asarray(r["k_aragonite"], dtype=float).ravel()
    out["pCO2_uatm"] = np.asarray(r["pCO2"], dtype=float).ravel()
    out["dic_umolkg"] = np.asarray(r["dic"], dtype=float).ravel()
    out["assumed_TA_umolkg"] = ta_umolkg
    out["pressure_dbar"] = pressure_dbar
    return out


def ph_variability_index(df: pd.DataFrame, ph_col: str = "ph_total") -> pd.DataFrame | None:
    """Daily (max - min) pH as a simple variability index."""
    if ph_col not in df.columns:
        return None
    d = df[["time", ph_col]].dropna().sort_values("time")
    if d.empty:
        return None
    day = d.set_index("time")[ph_col].resample("D").agg(["min", "max", "mean", "std"])
    day["daily_range"] = day["max"] - day["min"]
    day = day.reset_index().rename(
        columns={"min": "ph_min", "max": "ph_max", "mean": "ph_mean", "std": "ph_std"}
    )
    return day


def omega_profile_isochemical(
    salinity: float,
    temperature_c: float,
    ta_umolkg: float,
    ph_total_surface: float,
    surface_pressure_dbar: float = 40.0,
    pressures_dbar: np.ndarray | None = None,
) -> pd.DataFrame | None:
    """
    Omega_aragonite vs pressure holding TA + DIC fixed from a surface (pH, TA) solve (isochemical ascent).
    Horizon = first depth where Omega_ar < 1 (approximate depth = P/1.02 dbar·m⁻¹).
    """
    if not _HAS_PYCO2:
        return None
    if pressures_dbar is None:
        pressures_dbar = np.linspace(0.0, 500.0, 51)
    ta = np.array([float(ta_umolkg)])
    ph0 = np.array([float(ph_total_surface)])
    sal = np.array([float(salinity)])
    temp = np.array([float(temperature_c)])
    p0 = np.array([float(surface_pressure_dbar)])
    try:
        r0 = pyco2.sys(
            par1=ta,
            par2=ph0,
            par1_type=1,
            par2_type=3,
            salinity=sal,
            temperature=temp,
            pressure=p0,
        )
        dic0 = float(np.asarray(r0["dic"], dtype=float).ravel()[0])
    except Exception:
        return None
    omegas = []
    for P in pressures_dbar:
        try:
            rp = pyco2.sys(
                par1=np.array([float(ta_umolkg)]),
                par2=np.array([dic0]),
                par1_type=1,
                par2_type=2,
                salinity=sal,
                temperature=temp,
                pressure=np.array([float(P)]),
            )
            omegas.append(float(np.asarray(rp["saturation_aragonite"], dtype=float).ravel()[0]))
        except Exception:
            omegas.append(np.nan)
    prof = pd.DataFrame({"pressure_dbar": pressures_dbar, "saturation_aragonite": omegas})
    prof["depth_m_approx"] = prof["pressure_dbar"] / 1.02
    below = prof[np.isfinite(prof["saturation_aragonite"]) & (prof["saturation_aragonite"] < 1.0)]
    prof.attrs["horizon_depth_m"] = float(below["depth_m_approx"].iloc[0]) if len(below) else float("nan")
    prof.attrs["horizon_pressure_dbar"] = float(below["pressure_dbar"].iloc[0]) if len(below) else float("nan")
    return prof
