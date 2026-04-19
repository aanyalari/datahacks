# CCE Mooring Lab — analytics, quality, and forecasting

This repository supports a **California Current Ecosystem (CCE) mooring** workflow around the **[Scripps mooring program](https://mooring.ucsd.edu/cce/)**: exploratory analysis, daily panels, Streamlit dashboards, and optional ML / LLM-assisted narrative. The default branch is **`main`** and includes the full application, packaged utilities, and curated sample data for demos.

## What is in this repository

| Area | Description |
|------|-------------|
| **Streamlit app** | `Home.py` (Mission Control) plus multipage analytics under `pages/` (coverage, climatology, FFT, data quality, CalCOFI tie-ins, and more). |
| **Python package** | `src/cce_hack/` — loading, column selection, ingest from raw subfolders, processed panels, and model helpers. |
| **Data** | Versioned **combined** CSVs under `data/raw/` (by variable) and **daily** summaries under `data/processed/`. Large local drops stay gitignored; see `.gitignore`. |
| **Scripts** | `scripts/` — e.g. `fetch_cce_deployment_csv.py`, `generate_sample_data.py`, `build_processed_panel.py`, OceanSITES helpers. |
| **Notebooks** | Marimo EDA: `notebooks/cce_explore.py`. |
| **Documentation** | Sphinx sources under `docs/` (methodology + API). |

**Requirements:** Python **3.10+** (3.11+ recommended). Use a virtual environment; do **not** commit API keys or secrets.

---

## Quick start (Streamlit)

```bash
git clone https://github.com/aanyalari/datahacks.git
cd datahacks
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
streamlit run Home.py
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
streamlit run Home.py
```

The browser opens **Home** (Mission Control). Other screens appear in the sidebar under **Analytics**, **AI Predictions**, **Analysis Lab**, **Data Quality**, and related pages.

### Tips

- **Time window:** If charts look empty, set the sidebar range to **All data** or widen the window; merged moorings can have sparse or non-overlapping sensors.
- **Optional LLM features:** For narrative buttons, add a [Google AI Studio](https://aistudio.google.com) (Gemini) or [Groq](https://console.groq.com) key in the sidebar, or set `GOOGLE_API_KEY` / `GROQ_API_KEY`. The rest of the app runs **without** keys.
- **Your own CSV:** Upload in the sidebar, or place a merged `*.csv` under `data/raw/` (the loader picks a sensible default; see package `cce_hack.data`).

### Troubleshooting

| Issue | What to try |
|--------|-------------|
| `pip install` errors | Upgrade `pip`; confirm Python ≥ 3.10. |
| Windows activation errors | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate `.venv` again. |
| Empty plots | Use **All data**; confirm timestamps and numeric columns exist. |
| Odd page titles | Streamlit derives names from `pages/` filenames; emoji prefixes are optional. |

---

## Optional: sample CSV for a minimal offline demo

If you want a tiny synthetic file instead of the bundled data:

```bash
python scripts/generate_sample_data.py
```

That writes under `data/raw/` (ignored for new blobs except placeholders where configured). The app can also fall back to built-in sample logic when no file is found.

---

## Rebuilding processed panels (optional)

If you refresh raw extracts locally, you can rebuild hourly/daily panels using the project scripts (see `scripts/build_processed_panel.py` and related pipeline scripts). Close programs that lock output CSVs before re-running.

---

## Marimo and Sphinx

**Marimo:**

```bash
marimo edit notebooks/cce_explore.py
```

**Sphinx** (install doc extras first):

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

---

## Real CCE telemetry

### Option A — automated merge from deployment `csv/` (recommended)

```bash
python scripts/fetch_cce_deployment_csv.py ^
  --csv-base https://mooring.ucsd.edu/cce2/cce2_19/csv/ ^
  --mooring-id CCE2 ^
  --out data/raw/cce2_19_merged.csv
```

On PowerShell, use a single line or backtick continuation instead of `^` (cmd.exe). The script downloads channel CSVs, normalizes time to UTC `time`, and merges with **`merge_asof` (backward)** plus shallow-to-deep column fallbacks so sparse sensors do not blank the whole table. Tune `--tolerance-hours` if needed.

**Note:** `pco2_uatm` in merged output may be derived from `xCO2water` (dry mole fraction); treat it as a documented proxy if you use it in models or figures.

### Option B — manual download

Use the [CCE project page](https://mooring.ucsd.edu/cce/), open a deployment, append **`csv/`** to the URL, and save files into `data/raw/` if you prefer not to run the fetch script.

### Option C — OceanSITES NetCDF

QC products are available via OceanSITES / THREDDS (links from the mooring site). Use `xarray` to subset and export (`pip install ".[netcdf]"`).

---

## Contributing and Git workflow

- Work on a **feature branch**, open a **pull request** into `main`, and keep commits focused.
- **Never commit:** `.venv/`, `.streamlit/secrets.toml`, `.env`, or large private raw dumps ignored by `.gitignore`.

---

## Prize / eligibility framing

Use **CCE moorings** as the required **Scripps** dataset for the Scripps Challenge. Any extra datasets should support one clear thesis (see `docs/methodology.rst`).

## License

Hackathon project code: **MIT** unless your team chooses otherwise.
