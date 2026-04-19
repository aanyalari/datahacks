"""Analysis lab — STL, carbonate, coupling, dim reduction, gallery, ML (one module per run)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from cce_hack.dynamic_insights import insight_mooring_window
from cce_hack.streamlit_shell import inject_theme_css, page_config, render_global_sidebar
from cce_hack.ui_advanced import (
    render_acidification_tab,
    render_cross_tab,
    render_dimred_tab,
    render_fancy_tab,
    render_ml_advanced_tab,
    render_temporal_tab,
)

page_config(title="Analysis Lab — CCE")
inject_theme_css()
df = render_global_sidebar()

st.title("Analysis lab")
st.caption(
    "**Same CSV as Analytics** — Analytics is the judge-facing story; this page is optional **one-at-a-time** tools "
    "(coupling, spectra, UMAP, naive extrapolation, etc.). Pick one module so the demo stays fast."
)
st.caption(insight_mooring_window(df))

lab_pick = st.selectbox(
    "Module",
    [
        "Temporal decomposition & rolling stats",
        "Carbonate system & pH",
        "Coupling, Granger, wavelets",
        "PCA / clustering / UMAP",
        "Gallery (Hovmöller, radar, pairplot)",
        "ML & forecasting",
    ],
)

if lab_pick == "Temporal decomposition & rolling stats":
    render_temporal_tab(df)
elif lab_pick == "Carbonate system & pH":
    render_acidification_tab(df)
elif lab_pick == "Coupling, Granger, wavelets":
    render_cross_tab(df)
elif lab_pick == "PCA / clustering / UMAP":
    render_dimred_tab(df)
elif lab_pick == "Gallery (Hovmöller, radar, pairplot)":
    render_fancy_tab(df)
else:
    render_ml_advanced_tab(df)
