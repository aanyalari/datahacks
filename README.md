# MooringMind

**Ocean mooring insights, instantly.**

An interactive **Streamlit dashboard** for exploring and forecasting conditions in the **California Current Ecosystem (CCE) mooring array** (Scripps / SIO). It turns multi-sensor mooring telemetry into:

- **Mission Control** health checks and headline metrics
- **Analytics** (coverage, climatology, spectral / FFT, anomalies)
- **Data Quality** views for gaps, sensor issues, and sanity checks
- **AI Predictions**: simple forecasting and “soft sensor” modelling
- Optional **narrative insights** (works offline; keys only enable the LLM buttons)

Primary reference: **[CCE Mooring Array](https://mooring.ucsd.edu/cce/)**.

---

## App screenshots

Add screenshots to make the project instantly understandable on GitHub. Save images into `assets/screenshots/` using these filenames:

- `assets/screenshots/01-home.png` — Mission Control (Home)
- `assets/screenshots/02-analytics.png` — Analytics page (climatology / FFT / coverage)
- `assets/screenshots/03-ai.png` — AI Predictions page (forecast / anomalies / soft sensor)
- `assets/screenshots/04-quality.png` — Data Quality page

Once you add them, they will render here automatically:

![Mission Control](assets/screenshots/01-home.png)
![Analytics](assets/screenshots/02-analytics.png)
![AI Predictions](assets/screenshots/03-ai.png)
![Data Quality](assets/screenshots/04-quality.png)

---

## Run locally

**Requirements:** Python **3.10+** (3.11+ recommended).

```bash
git clone https://github.com/aanyalari/datahacks.git
cd datahacks
```

## Installation

This project is packaged as `cce-hack` under `src/`. The recommended install is **editable** so changes to the package are picked up instantly while you develop the app.

- **Recommended (editable install)**:

```bash
pip install -U pip
pip install -e .
```

- **From requirements (alternative)**:

```bash
pip install -U pip
pip install -r requirements.txt
```

- **Optional extras**:
  - **Docs**: `pip install -r requirements-docs.txt`
  - **NetCDF/OceanSITES**: `pip install -e ".[netcdf]"`

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
streamlit run Home.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
streamlit run Home.py
```

---

## Data inputs

You can use the repo’s included CSVs, or bring your own:

- **Upload a CSV** in the app sidebar, or
- Drop a merged `*.csv` under `data/raw/` (the loader picks a default)

If charts look empty, set the sidebar time window to **All data** (mooring sensors often do not overlap perfectly in time).

### Expected CSV shape (quick guide)

The app works best with:

- a timestamp column named `time` (UTC recommended), and
- sensor columns as numeric fields (temperature, salinity, pH, oxygen, nitrate, chlorophyll, wind, etc.)

If your raw files are separate per sensor, use the fetch/merge script below (or merge them yourself into one “wide” table).

### Optional: generate a tiny sample file

```bash
python scripts/generate_sample_data.py
```

### Optional: fetch and merge a deployment directly from the mooring site

```bash
python scripts/fetch_cce_deployment_csv.py ^
  --csv-base https://mooring.ucsd.edu/cce2/cce2_19/csv/ ^
  --mooring-id CCE2 ^
  --out data/raw/cce2_19_merged.csv
```

On PowerShell, use a single line or backtick continuation instead of `^` (cmd.exe).

---

## Features (what to try in the app)

### What you can do with MooringMind

- **Mission Control (Home)**:
  - quick “health” view of the selected time window
  - headline sensor tiles (latest vs a trailing baseline)
  - map view of reference sites
- **Analytics**:
  - coverage and gaps across variables
  - climatology / seasonality views
  - spectral structure (FFT / periodicity)
  - multi-sensor comparisons in real units
- **Data Quality**:
  - missingness summaries and “is this usable?” checks
  - quick outlier/sanity scans that help spot sensor problems early
- **AI Predictions**:
  - forecasting experiments using lagged sensor history + calendar features
  - anomaly detection with plain-English explanations
  - “soft sensor” models to estimate harder-to-measure variables from easier ones
- **Ecosystem context** (if you use the included pages/data):
  - CalCOFI comparisons (ship surveys vs high-frequency mooring telemetry)
  - species observation tie-ins (optional pages)

### Pages overview

- **Home**: Mission Control entry point (`Home.py`)
- **Analytics**: core plots + story-style tabs (`pages/1_📊_Analytics.py`)
- **AI Predictions**: forecast/anomaly/soft sensor tabs (`pages/2_🤖_AI_Predictions.py`)
- **Analysis Lab**: one-at-a-time advanced modules (`pages/3_🧪_Analysis_Lab.py`)
- **Data Quality**: completeness, gaps, and QA checks (`pages/4_📋_Data_Quality.py`)

---

## Optional AI narrative buttons

The app runs **fully without** API keys. Keys only unlock narrative / explanation buttons.

- Set `GOOGLE_API_KEY` / `GROQ_API_KEY` as environment variables, or use the sidebar inputs.
- Never commit `.env` or `.streamlit/secrets.toml` (both are ignored by default).

---

## Project layout (for contributors)

- `Home.py`: Streamlit entry point
- `pages/`: multipage dashboard screens
- `src/cce_hack/`: reusable package (loading, ingest, features, pipeline helpers)
- `scripts/`: data fetch / processing utilities
- `data/raw/`, `data/processed/`: example inputs and derived panels (large local drops stay gitignored)
- `docs/`: Sphinx docs (optional)
- `notebooks/`: Marimo EDA (optional)

---

## Dev extras

### Common scripts

- `scripts/fetch_cce_deployment_csv.py`: download a deployment’s `csv/` directory and merge channels into a single wide table
- `scripts/build_processed_panel.py`: build processed panels (when raw folders are present)
- `scripts/process_mooring_daily.py`: daily summary pipeline utilities
- `scripts/download_oceansites_by_variable.py`: OceanSITES helper (optional)

Marimo:

```bash
marimo edit notebooks/cce_explore.py
```

Sphinx:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
```

---

## License

MIT (hackathon project code), unless your team decides otherwise.
