"""Coverage heatmap + downloadable markdown QA brief."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cce_hack.agent_tools import build_context_package
from cce_hack.plot_theme import apply_plotly, plotly_theme_kwargs
from cce_hack.streamlit_shell import CHART_H_FULL, friendly_column_label_plain


def monthly_coverage_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, list[str]] | tuple[None, None]:
    """Rows = month periods (YYYY-MM), columns = sensors, values = % rows with data that month."""
    if "time" not in df.columns or not numeric_cols:
        return None, None
    d = df[["time"] + numeric_cols].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"])
    if d.empty:
        return None, None
    d["month"] = d["time"].dt.to_period("M").astype(str)
    months = sorted(d["month"].unique())
    mat = np.zeros((len(months), len(numeric_cols)))
    for j, col in enumerate(numeric_cols):
        for i, m in enumerate(months):
            sub = d[d["month"] == m]
            if sub.empty:
                mat[i, j] = np.nan
                continue
            filled = pd.to_numeric(sub[col], errors="coerce").notna().mean()
            mat[i, j] = 100.0 * float(filled)
    out = pd.DataFrame(mat, index=months, columns=numeric_cols)
    return out, months


def coverage_heatmap_figure(mat: pd.DataFrame) -> go.Figure:
    z = mat.to_numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[friendly_column_label_plain(c) for c in mat.columns],
            y=mat.index.astype(str).tolist(),
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title="% filled"),
        )
    )
    fig.update_layout(
        title="Monthly data coverage (% of rows with a value)",
        xaxis_title="Sensor / column",
        yaxis_title="Month (UTC)",
        height=CHART_H_FULL,
        **{k: v for k, v in plotly_theme_kwargs().items() if k != "margin"},
    )
    return apply_plotly(fig)


def build_data_quality_markdown_report(df: pd.DataFrame) -> str:
    pkg = build_context_package(df)
    lines = ["# Mooring data quality summary", ""]
    if "error" in pkg:
        return lines[0] + "\n\n" + str(pkg["error"])
    lines.append(f"- **Rows:** {pkg.get('rows', 0):,}")
    tr = pkg.get("time_range_utc") or {}
    lines.append(f"- **UTC window:** `{tr.get('start')}` → `{tr.get('end')}`")
    if pkg.get("mooring_id"):
        lines.append(f"- **Mooring:** `{pkg['mooring_id']}`")
    lines.append("")
    lines.append("## Per-column coverage")
    for col, info in (pkg.get("coverage_by_column") or {}).items():
        lines.append(
            f"- **{col}:** {info.get('filled_percent', 0):.1f}% filled "
            f"({info.get('first_date')} → {info.get('last_date')})"
        )
    lines.append("")
    lines.append("*Generated deterministically from the loaded CSV.*")
    return "\n".join(lines)
