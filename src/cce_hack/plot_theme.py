"""Shared Plotly layout defaults for Streamlit dashboards (light default for demos)."""

from __future__ import annotations

import streamlit as st

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,22,38,0.92)",
    font=dict(family="Segoe UI, system-ui, sans-serif", color="#c8d7f0", size=13),
    title_font=dict(size=16, color="#f0f6ff"),
    margin=dict(l=52, r=24, t=48, b=44),
    colorway=["#2dd4bf", "#60a5fa", "#fb923c", "#c084fc", "#4ade80", "#fbbf24"],
)

PLOTLY_LIGHT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Segoe UI, system-ui, sans-serif", color="#0f172a", size=13),
    title_font=dict(size=16, color="#0f172a"),
    margin=dict(l=52, r=24, t=48, b=44),
    colorway=["#0d9488", "#2563eb", "#ea580c", "#7c3aed", "#15803d", "#b45309"],
)

# Back-compat alias
PLOTLY_BASE = PLOTLY_DARK


def current_plotly_theme() -> str:
    return str(st.session_state.get("cce_chart_theme", "light")).lower()


def plotly_theme_kwargs() -> dict:
    return PLOTLY_LIGHT if current_plotly_theme() == "light" else PLOTLY_DARK


def apply_plotly(fig):
    fig.update_layout(**plotly_theme_kwargs())
    return fig
