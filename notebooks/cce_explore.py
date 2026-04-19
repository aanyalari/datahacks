"""Marimo notebook: CCE mooring EDA + supervised setup.

Run from repo root:

  pip install -e .
  marimo edit notebooks/cce_explore.py
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    return mo, plt


@app.cell
def _(mo):
    mo.md(r"""
    # CCE mooring exploration (Marimo)

    Reactive notebook for **Data Analytics**: coverage, gaps, simple seasonality checks.

    **Dataset:** [CCE mooring array](https://mooring.ucsd.edu/cce/) (Scripps / SIO collaborators). Uses the same CSV loader as `streamlit_app.py`.
    """)
    return


@app.cell
def _():
    from cce_hack.data import load_mooring_table, pick_default_csv
    from cce_hack.sample_data import ensure_sample_csv

    path = pick_default_csv()
    if not path.exists():
        ensure_sample_csv(path)
    df = load_mooring_table(path)
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## Sample rows
    """)
    return


@app.cell
def _(df, mo):
    mo.md("## Missingness (fraction NaN)")
    num = df.select_dtypes(include="number")
    return


@app.cell
def _(df, mo, plt):
    mo.md("## Median SST by hour-of-day (UTC)")
    fig, ax = plt.subplots(figsize=(9, 3))
    if "sst_c" in df.columns:
        g = df.assign(hour=df["time"].dt.hour).groupby("hour")["sst_c"].median()
        ax.plot(g.index, g.values, marker="o")
        ax.set_xlabel("hour")
        ax.set_ylabel("median SST (°C)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hackathon checklist
    - Swap synthetic CSV for a real deployment export under `data/raw/`.
    - Document joins if you add CalCOFI, Spray, or reanalysis context.
    - Keep ML validation time-ordered (no leakage across t + H).
    """)
    return


if __name__ == "__main__":
    app.run()
