"""Short, data-derived captions for under-chart context (no static essays)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cce_hack.column_pick import friendly_axis_label, pick_best_column
from cce_hack.mission_alerts import pick_chl_column, pick_o2_column


def _sort_time(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "time" not in df.columns:
        return df
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time")


def insight_mooring_window(df: pd.DataFrame) -> str:
    d = _sort_time(df)
    if d.empty:
        return "No rows in the current sidebar time window."
    t0, t1 = d["time"].iloc[0], d["time"].iloc[-1]
    n = len(d)
    mid = ""
    if "mooring_id" in d.columns:
        u = d["mooring_id"].dropna().astype(str).unique().tolist()
        if u:
            mid = f" **Mooring:** {', '.join(u[:3])}."
    return f"**{n:,}** samples from **{str(t0.date())}** to **{str(t1.date())}** (UTC).{mid}"


def insight_headline_metrics(df: pd.DataFrame) -> str:
    """One line from largest |latest − 30d mean| among core roles."""
    if df is None or df.empty or "time" not in df.columns:
        return ""
    best = None
    best_mag = 0.0
    for role in ("ph", "sst", "salinity", "o2", "no3", "chl"):
        c = pick_o2_column(df) if role == "o2" else pick_best_column(df, role)
        if not c or c not in df.columns:
            continue
        sub = df[["time", c]].copy()
        sub["time"] = pd.to_datetime(sub["time"], utc=True, errors="coerce")
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=["time", c]).sort_values("time")
        if len(sub) < 5:
            continue
        latest = float(sub[c].iloc[-1])
        t1 = sub["time"].iloc[-1]
        w = sub[sub["time"] >= t1 - pd.Timedelta(days=30)]
        if len(w) < 3:
            mu = float(sub[c].mean())
        else:
            mu = float(w[c].mean())
        mag = abs(latest - mu)
        if mag > best_mag:
            best_mag = mag
            lab = friendly_axis_label(c)
            best = (lab, latest, mu, latest - mu)
    if not best:
        return "Not enough overlapping numeric columns to summarize deltas."
    lab, latest, mu, delta = best
    direction = "above" if delta > 0 else "below"
    return (
        f"Largest move vs 30-day mean: **{lab}** is **{abs(delta):.4g}** {direction} its recent average "
        f"(latest **{latest:.4g}**, 30d mean **{mu:.4g}**)."
    )


def insight_multisensor_normalized(df: pd.DataFrame, *, max_days: int = 90) -> str:
    """
    Explains the 0-1 overlay: which pair moved most similarly in the plotted window.
    """
    use: list[tuple[str, str]] = []
    o2c = pick_o2_column(df)
    chlc = pick_chl_column(df)
    for lab, role in [
        ("pH", "ph"),
        ("Temperature", "sst"),
        ("Salinity", "salinity"),
        ("O₂", "o2"),
        ("Nitrate", "no3"),
        ("Chlorophyll", "chl"),
    ]:
        c = o2c if role == "o2" else (chlc if role == "chl" else pick_best_column(df, role))
        if c and c in df.columns:
            use.append((lab, c))
    if len(use) < 2:
        return "Need at least two sensors with data to compare co-movement."
    d = df[["time"] + [c for _, c in use]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    t1 = d["time"].max()
    d = d[d["time"] >= t1 - pd.Timedelta(days=max_days)]
    if len(d) < 15:
        return "Very few points in the recent window — widen **Time window** in the sidebar."
    norm = {}
    for lab, c in use:
        s = pd.to_numeric(d[c], errors="coerce")
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if hi <= lo or np.isnan(lo):
            continue
        norm[lab] = (s - lo) / (hi - lo)
    names = list(norm.keys())
    if len(names) < 2:
        return "After normalization, not enough variation to compare sensors."
    best_r, pair = -2.0, (names[0], names[1])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = norm[names[i]].to_numpy()
            b = norm[names[j]].to_numpy()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                continue
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            if r > best_r:
                best_r, pair = r, (names[i], names[j])
    if best_r < -1.5:
        return (
            "**What this chart is:** each line is a different sensor rescaled to **0–1** over the window so you can see "
            "**timing** (up together = shared forcing; one spikes alone = local process or sensor glitch). "
            "Correlations are weak here — use **Analytics** for raw units."
        )
    return (
        "**What this chart is:** each sensor is min–max scaled to **0–1** for this window only — units are removed on purpose so you "
        "compare **when** things move, not absolute chemistry. "
        f"**Tightest co-movement:** *{pair[0]}* vs *{pair[1]}* (Pearson **r ≈ {best_r:.2f}** on overlapping points)."
    )


def insight_ts_pair(df: pd.DataFrame, col_a: str | None, col_b: str | None) -> str:
    if not col_a or not col_b or col_a not in df.columns or col_b not in df.columns:
        return ""
    d = df[["time", col_a, col_b]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    for c in (col_a, col_b):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[col_a, col_b])
    if len(d) < 12:
        return ""
    r = float(np.corrcoef(d[col_a].to_numpy(), d[col_b].to_numpy())[0, 1])
    la, lb = friendly_axis_label(col_a), friendly_axis_label(col_b)
    strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.35 else "weak"
    sign = "same-direction" if r > 0 else "opposite-direction"
    return f"**{la}** vs **{lb}**: {strength} **{sign}** linkage in this date filter (**r ≈ {r:.2f}**, n={len(d):,})."


def insight_ph_aragonite(co2: pd.DataFrame | None) -> str:
    if co2 is None or co2.empty or "saturation_aragonite" not in co2.columns:
        return ""
    s = pd.to_numeric(co2["saturation_aragonite"], errors="coerce").dropna()
    if s.empty:
        return ""
    latest = float(s.iloc[-1])
    lo = float(s.quantile(0.1))
    risk = "Ω_aragonite dipped near or below **1** in part of this window — carbonate stress for aragonite shell-builders." if (s < 1.05).any() else "Ω_aragonite stayed above **1.05** everywhere in this window — lower immediate aragonite undersaturation risk in this reconstruction."
    return f"Latest Ω_aragonite ≈ **{latest:.2f}** (10th percentile in window ≈ **{lo:.2f}**). {risk}"


def insight_analytics_physical(df: pd.DataFrame, sst_c: str | None, sal_c: str | None) -> str:
    if not sst_c or not sal_c:
        return "Add overlapping temperature + salinity columns for a T–S water-mass readout."
    d = df[["time", sst_c, sal_c]].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d[sst_c] = pd.to_numeric(d[sst_c], errors="coerce")
    d[sal_c] = pd.to_numeric(d[sal_c], errors="coerce")
    d = d.dropna(subset=[sst_c, sal_c])
    if len(d) < 20:
        return "Too few T–S points in this filter for a stable cluster read."
    span_t = float(d[sst_c].max() - d[sst_c].min())
    span_s = float(d[sal_c].max() - d[sal_c].min())
    return (
        f"In this window, temperature spans **{span_t:.2f} °C** and salinity **{span_s:.3f} PSU** — "
        "compact clusters on the T–S plot usually mean recurring water masses; a long streak suggests advection or seasonal progression."
    )
