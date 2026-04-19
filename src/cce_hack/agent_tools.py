"""Deterministic summaries for mooring data — agents must ground answers in this JSON only."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _pct(x: float) -> float:
    return round(100.0 * float(x), 1)


def _col_dates(df: pd.DataFrame, col: str) -> tuple[str, str]:
    m = df[col].notna()
    if not m.any():
        return "—", "—"
    tt = pd.to_datetime(df.loc[m, "time"], utc=True, errors="coerce")
    return str(tt.min())[:10], str(tt.max())[:10]


def build_context_package(
    df: pd.DataFrame,
    max_cols: int = 12,
    stats_max_rows: int = 35_000,
) -> dict[str, Any]:
    """
    Single JSON-serializable dict for LLM agents (compact).

    Coverage uses the full dataframe; mean/std/min/max use the last ``stats_max_rows``
    rows so huge files do not slow Python down before Ollama even starts.
    """
    if "time" not in df.columns:
        return {"error": "Data needs a time column."}
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    t0, t1 = t.min(), t.max()
    numeric = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("mooring_id", "latitude", "longitude")
    ][:max_cols]

    work = df if len(df) <= stats_max_rows else df.tail(stats_max_rows)

    coverage: dict[str, Any] = {}
    for c in numeric:
        s = df[c]
        fd, ld = _col_dates(df, c)
        coverage[c] = {
            "filled_percent": _pct(s.notna().mean()),
            "first_date": fd,
            "last_date": ld,
        }

    def block(df_src: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
        out = {}
        for c in cols:
            if c not in df_src.columns:
                continue
            s = pd.to_numeric(df_src[c], errors="coerce").dropna()
            if len(s) < 3:
                out[c] = {"note": "too few values"}
                continue
            out[c] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
            }
        return out

    physical_cols = [c for c in ("sst_c", "salinity_psu", "conductivity_s_m") if c in df.columns]
    phys_extra = [c for c in df.columns if c.startswith("sst_c_d") or c.startswith("salinity_psu_d")]
    bio_cols = [c for c in ("ph_total", "no3", "chl_mg_m3", "chl_mg_m3_d40m") if c in df.columns]

    mooring = None
    if "mooring_id" in df.columns and df["mooring_id"].notna().any():
        mooring = str(df["mooring_id"].dropna().iloc[0])

    note = (
        "Temperature/salinity may stop years before pH or nitrate if the export file ends earlier. "
        "Do not invent numbers; only interpret the summaries above."
    )
    if len(df) > stats_max_rows:
        note += f" Numeric summaries (mean/std/min/max) use the last {stats_max_rows:,} rows for speed."

    return {
        "mooring_id": mooring,
        "rows": int(len(df)),
        "time_range_utc": {"start": str(t0)[:19], "end": str(t1)[:19]},
        "coverage_by_column": coverage,
        "physical_summary": block(work, physical_cols + phys_extra[:6]),
        "biogeochem_summary": block(work, bio_cols),
        "note_for_agents": note,
    }


def context_json_for_prompt(pkg: dict[str, Any], indent: int = 0, max_chars: int = 4200) -> str:
    """Keep prompt small: less input tokens = faster on CPU."""
    s = json.dumps(pkg, indent=indent, default=str)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 80] + "\n/* truncated for speed; stats blocks are complete */"


def instant_report_markdown(pkg: dict[str, Any]) -> str:
    """
    Judge-ready narrative built **only** from the JSON package — no LLM, no wait.
    Use beside optional Ollama text so the demo always has a story even when AI is slow or offline.
    """
    if "error" in pkg:
        return f"**Cannot build instant report:** {pkg['error']}"

    lines: list[str] = []
    title = "Instant mooring brief"
    mid = pkg.get("mooring_id")
    if mid:
        title += f" — **{mid}**"
    lines.append(f"### {title}")
    tr = pkg.get("time_range_utc") or {}
    lines.append(
        f"- **Rows:** {pkg.get('rows', 0):,}  \n"
        f"- **Time span (UTC):** `{tr.get('start', '—')}` → `{tr.get('end', '—')}`"
    )

    cov = pkg.get("coverage_by_column") or {}
    if cov:
        lines.append("\n#### Data availability")
        worst = sorted(cov.items(), key=lambda kv: kv[1].get("filled_percent", 0))[:4]
        for col, info in worst:
            fp = info.get("filled_percent", 0)
            fd, ld = info.get("first_date", "—"), info.get("last_date", "—")
            lines.append(f"- **{col}:** {fp:.0f}% filled (first `{fd}`, last `{ld}`)")

    def _stat_lines(block: dict[str, Any], heading: str) -> None:
        if not block:
            return
        lines.append(f"\n#### {heading}")
        for col, stats in block.items():
            if isinstance(stats, dict) and stats.get("note"):
                lines.append(f"- **{col}:** {stats['note']}")
                continue
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- **{col}:** mean {stats.get('mean')}, std {stats.get('std')}, "
                f"range [{stats.get('min')}, {stats.get('max')}]"
            )

    _stat_lines(pkg.get("physical_summary") or {}, "Physical (from file)")
    _stat_lines(pkg.get("biogeochem_summary") or {}, "Biogeochemistry (from file)")

    note = pkg.get("note_for_agents")
    if note:
        lines.append(f"\n*{note}*")

    lines.append(
        "\n---\n*This block is 100% deterministic: it mirrors the JSON summary. "
        "Optional AI below only rephrases the same facts.*"
    )
    return "\n".join(lines)
