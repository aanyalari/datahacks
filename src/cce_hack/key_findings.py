"""Deterministic 'so what' bullets for demos (no LLM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cce_hack.acidification_co2sys import run_co2sys_on_dataframe
from cce_hack.column_pick import pick_best_column
from cce_hack.mission_alerts import pick_chl_column, pick_o2_column


def _series_slope_per_year(s: pd.Series, t: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce")
    m = s.notna() & t.notna()
    if int(m.sum()) < 30:
        return None
    days = (t[m] - t[m].min()).dt.total_seconds() / 86400.0
    y = s[m].to_numpy(dtype=float)
    if np.nanstd(y) == 0:
        return None
    coef = np.polyfit(days.to_numpy(), y, 1)
    return float(coef[0] * 365.25)


def key_findings_mission(df: pd.DataFrame) -> list[str]:
    bullets: list[str] = []
    if "time" not in df.columns:
        return ["No time column — cannot summarize trends."]

    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")

    ph_c = pick_best_column(d, "ph")
    if ph_c:
        sl = _series_slope_per_year(d[ph_c], d["time"])
        if sl is not None:
            if sl < -0.005:
                bullets.append(
                    f"pH declined about **{abs(sl):.3f} units per year** over the visible record — directionally consistent with regional acidification pressure (verify with QA flags)."
                )
            elif sl > 0.005:
                bullets.append(
                    f"pH increased about **{sl:.3f} units per year** in this slice — could reflect seasonal dominance or sensor drift; cross-check maintenance logs."
                )
            else:
                bullets.append("pH trend over the record is **near flat** after a simple linear fit — no dramatic acidification signal in this window alone.")

    o2c = pick_o2_column(d)
    if o2c:
        low = (pd.to_numeric(d[o2c], errors="coerce") < 2.0).sum()
        if int(low) > 0:
            bullets.append(
                f"**{int(low)}** hourly samples register O₂ below **2 mg/L** in this file — count hypoxic exposure events for biology-focused storylines."
            )
        else:
            bullets.append("No samples in this slice fall below the **2 mg/L** hypoxia demo threshold (or O₂ is absent).")

    chlc = pick_chl_column(d)
    if chlc:
        bloom = (pd.to_numeric(d[chlc], errors="coerce") > 5.0).sum()
        if int(bloom) > 0:
            bullets.append(
                f"**{int(bloom)}** samples exceed **5 mg/m³** chlorophyll (demo bloom threshold) — useful for narrating bloom seasonality vs wind/upwelling."
            )

    co2 = run_co2sys_on_dataframe(d)
    if co2 is not None and co2["saturation_aragonite"].notna().any():
        om = float(co2["saturation_aragonite"].dropna().iloc[-1])
        bullets.append(f"Latest Ω_aragonite (pH+T+S, assumed TA) ≈ **{om:.2f}** — values persistently below 1.0 would strengthen a carbonate-stress storyline.")

    if not bullets:
        bullets.append("Add co-located pH, nitrate, and biology columns to unlock richer auto-findings.")

    return bullets[:5]


def key_findings_analytics(df: pd.DataFrame) -> list[str]:
    bullets: list[str] = []
    d = df.copy()
    if "time" not in d.columns:
        return ["No time column."]
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"])

    sst_c = pick_best_column(d, "sst")
    sal_c = pick_best_column(d, "salinity")
    if sst_c and sal_c:
        bullets.append(
            "The **T–S diagram** fingerprints water masses: clusters correspond to different density structures that carry distinct nutrients and biology."
        )

    chlc = pick_chl_column(d)
    if chlc and "no3" in d.columns:
        bullets.append(
            "**Chlorophyll + nitrate** together help separate bloom accumulation from nitrate-fueled growth — look for periods where Chl rises without NO₃ collapse (recycling) vs both move (new production)."
        )

    if pick_best_column(d, "ph"):
        bullets.append(
            "**pH trends** here are from sensors; pairing with Ω_aragonite (carbonate tab) connects chemistry to shellfish/pteropod habitat risk narratives."
        )

    co2 = run_co2sys_on_dataframe(d)
    if co2 is not None:
        om = co2["saturation_aragonite"].dropna()
        if len(om) > 10:
            delta = float(om.iloc[-1] - om.iloc[:30].mean())
            bullets.append(
                f"Ω_aragonite moved by **{delta:+.2f}** from the early-window mean to the latest estimate — small shifts matter near saturation = 1."
            )

    bullets.append("**Hovmöller / radar** compress time: they make seasonality and depth structure visible without reading every line plot.")

    return bullets[:5]
