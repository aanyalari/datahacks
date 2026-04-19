"""Mission Control: alert levels + one-line carbonate / habitat narrative from file facts."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from cce_hack.acidification_co2sys import run_co2sys_on_dataframe
from cce_hack.column_pick import pick_best_column

Alert = Literal["green", "yellow", "red", "na"]


def _latest_valid(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _mean_last_days(df: pd.DataFrame, col: str, days: float = 7.0) -> float | None:
    if col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", col]].dropna().copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"])
    if d.empty:
        return None
    t1 = d["time"].max()
    cut = t1 - pd.Timedelta(days=days)
    w = d[d["time"] >= cut]
    if w.empty:
        return None
    return float(pd.to_numeric(w[col], errors="coerce").mean())


def alert_ph(ph: float | None) -> tuple[Alert, str]:
    if ph is None or np.isnan(ph):
        return "na", "No pH"
    if ph < 7.75:
        return "red", "OA stress (pH < 7.75)"
    if ph < 7.95:
        return "yellow", "Elevated OA risk (pH < 7.95)"
    return "green", "pH within typical surface range"


def alert_o2_mg_l(o2: float | None) -> tuple[Alert, str]:
    if o2 is None or np.isnan(o2):
        return "na", "No O₂ (mg/L) column"
    if o2 < 2.0:
        return "red", "Hypoxic threshold (O₂ < 2 mg/L)"
    if o2 < 4.0:
        return "yellow", "Low oxygen (O₂ < 4 mg/L)"
    return "green", "O₂ above hypoxic threshold"


def alert_chl(chl: float | None) -> tuple[Alert, str]:
    if chl is None or np.isnan(chl):
        return "na", "No chlorophyll"
    if chl > 10.0:
        return "red", "Very high chlorophyll (>10 mg/m³)"
    if chl > 5.0:
        return "yellow", "Elevated chlorophyll (>5 mg/m³)"
    return "green", "Chlorophyll not in bloom alert range"


def alert_no3(no3: float | None) -> tuple[Alert, str]:
    if no3 is None or np.isnan(no3):
        return "na", "No nitrate"
    if no3 > 30.0:
        return "red", "Nutrient spike (NO₃ > 30 µM)"
    if no3 > 20.0:
        return "yellow", "Elevated nitrate (>20 µM)"
    return "green", "Nitrate below spike threshold"


def pick_o2_column(df: pd.DataFrame) -> str | None:
    return pick_best_column(df, "o2")


def pick_chl_column(df: pd.DataFrame) -> str | None:
    return pick_best_column(df, "chl")


def aragonite_habitat_sentence(df: pd.DataFrame) -> str:
    """
    One auto-generated habitat sentence (uses PyCO2SYS when pH+T+S exist; else heuristic).
    """
    ph_c = pick_best_column(df, "ph")
    t_c = pick_best_column(df, "sst")
    s_c = pick_best_column(df, "salinity")
    if ph_c and t_c and s_c:
        co2 = run_co2sys_on_dataframe(df, salinity_col=s_c, temp_col=t_c, ph_col=ph_c)
    else:
        co2 = run_co2sys_on_dataframe(df)
    if co2 is not None and len(co2):
        om = float(co2["saturation_aragonite"].dropna().iloc[-1])
        ph = (
            float(pd.to_numeric(df[ph_c], errors="coerce").dropna().iloc[-1])
            if ph_c and df[ph_c].notna().any()
            else float("nan")
        )
        if om < 1.0:
            return (
                f"Aragonite saturation (Ω_ar) is about **{om:.2f}** in the latest co-located pH/T/S window — "
                "below 1.0 suggests undersaturation; **pteropod and calcifying habitat stress** is plausible and merits follow-up with full carbonate observations."
            )
        return (
            f"Latest Ω_aragonite ≈ **{om:.2f}** from pH + T + S with assumed alkalinity — above 1.0 suggests surface waters are not strongly undersaturated "
            "in this simplified check (still not a substitute for bottle TA/pH pairs)."
        )

    ph = _latest_valid(df, ph_c) if ph_c else None
    t = _latest_valid(df, t_c) if t_c else None
    if ph is not None and t is not None:
        if ph < 7.85 and t > 12:
            return (
                "Without a full carbonate solve, **low pH with warm surface water** suggests elevated acidification stress; "
                "aragonite saturation **may** approach undersaturation — treat as a flag to run PyCO2SYS with measured alkalinity when available."
            )
    if ph is not None:
        return f"Latest pH is **{ph:.3f}**; Ω_aragonite cannot be estimated here without reliable salinity, temperature, and alkalinity on the same timestamps."
    return "Insufficient pH / T / S overlap in this file to auto-estimate aragonite saturation — add co-located carbonate parameters for a stronger statement."


def alert_rows_for_mission(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows for a compact mission table."""
    o2c = pick_o2_column(df)
    chlc = pick_chl_column(df)
    ph_c = pick_best_column(df, "ph")
    no3_c = pick_best_column(df, "no3")
    rows = []
    ph = _latest_valid(df, ph_c) if ph_c else None
    a, msg = alert_ph(ph)
    rows.append({"Variable": "pH", "Latest": ph, "Alert": a, "Detail": msg})
    o2 = _latest_valid(df, o2c) if o2c else None
    a, msg = alert_o2_mg_l(o2)
    rows.append({"Variable": "O₂" + (f" (`{o2c}`)" if o2c else ""), "Latest": o2, "Alert": a, "Detail": msg})
    chl = _latest_valid(df, chlc) if chlc else None
    a, msg = alert_chl(chl)
    rows.append({"Variable": "Chlorophyll" + (f" (`{chlc}`)" if chlc else ""), "Latest": chl, "Alert": a, "Detail": msg})
    no3 = _latest_valid(df, no3_c) if no3_c else None
    a, msg = alert_no3(no3)
    rows.append({"Variable": "Nitrate", "Latest": no3, "Alert": a, "Detail": msg})
    return rows
