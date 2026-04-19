# CCE Mooring — dual-track (Analytics + ML)

Primary dataset: **[California Current Ecosystem (CCE) Mooring Array](https://mooring.ucsd.edu/cce/)** (Scripps / SIO collaborators). This repo includes:

- **Streamlit** judge demo (`streamlit_app.py`)
- **Marimo** reactive EDA (`notebooks/cce_explore.py`)
- **Sphinx** methodology + API docs (`docs/`)
- A small **Python package** (`src/cce_hack/`) for loading, feature construction, and forecasting

## For your team — run the Streamlit app

**Requirements:** Python **3.10 or newer** (3.11+ recommended). Do **not** commit API keys; use the sidebar or environment variables only on your machine.

### 1. Clone and open a terminal in the repo folder

```bash
git clone https://github.com/aanyalari/datahacks.git
cd datahacks
git checkout feature/cce-mooring-streamlit
```

The **Streamlit Mission Control** app and package live on branch `feature/cce-mooring-streamlit`. Older experiment files may remain on `main`.

### 2. Create a virtual environment and install

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 3. (Optional) Sample CSV for a quick offline demo

If you do not have real mooring CSVs yet:

```bash
python scripts/generate_sample_data.py
```

That writes a small merged-style CSV under `data/raw/` (gitignored except placeholders). The app can also use built-in sample logic when no file is present—see the sidebar in the app.

### 4. Start the app

```bash
streamlit run streamlit_app.py
```

Your browser should open to **Mission Control** (home). Other tabs live under **Analytics**, **AI Predictions**, **Analysis Lab**, and **Data Quality** in the Streamlit sidebar.

### Tips

- **Time window:** If charts or KPIs look empty, set the sidebar time range to **All data** (or widen the window). Merged moorings often have sparse columns or non-overlapping sensors.
- **Optional Claude features:** For AI narrative helpers, add an [Anthropic](https://www.anthropic.com/) API key in the sidebar **or** set `ANTHROPIC_API_KEY` in your environment. The rest of the app works **offline** without it.
- **Your own data:** Upload a CSV in the sidebar, or place a merged `*.csv` in `data/raw/` (see “Real CCE data” below).

### Troubleshooting

| Issue | What to try |
|--------|----------------|
| `pip install` errors | Upgrade pip; use Python 3.10+ |
| Permission errors on Windows | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate `.venv` again |
| Empty plots | All-data time window; confirm the CSV has numeric columns and timestamps |
| Page names look odd | Streamlit uses the `pages/` filenames; emoji prefixes are optional |

---

## Maintainer — push this project to GitHub

1. On GitHub: **New repository** → choose a name (e.g. `datahacks`) → **do not** add a README/license if you already have them locally → Create.
2. In the project folder (first time only), or to add a feature branch:

```bash
git init
git add .
git commit -m "Initial commit: CCE Mooring Streamlit lab"
git remote add origin https://github.com/aanyalari/datahacks.git
git checkout -b feature/cce-mooring-streamlit
git push -u origin feature/cce-mooring-streamlit
```

Use SSH if you prefer: `git@github.com:aanyalari/datahacks.git`. To open a **pull request** into `main`, use GitHub’s “Compare & pull request” after the push.

**Never push:** `.venv/`, `.streamlit/secrets.toml`, `.env`, or large raw datasets under `data/raw/` (they are gitignored by default).

---

## Quickstart (full stack)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .
python scripts/generate_sample_data.py
streamlit run streamlit_app.py
```

Marimo:

```bash
marimo edit notebooks/cce_explore.py
```

Sphinx:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`.

## Real CCE data

### Option A — automated (recommended)

Telemetry CSVs for each deployment live under `.../<deployment>/csv/` on [mooring.ucsd.edu](https://mooring.ucsd.edu/cce/). From the repo root:

```bash
python scripts/fetch_cce_deployment_csv.py ^
  --csv-base https://mooring.ucsd.edu/cce2/cce2_19/csv/ ^
  --mooring-id CCE2 ^
  --out data/raw/cce2_19_merged.csv
```

(Use `^` line continuation in **cmd.exe**; in **PowerShell** use backtick `` ` `` or one line.)

That script downloads `temp.csv`, `sal.csv`, `wind.csv`, `pH.csv`, `co2.csv`, `airPT.csv`, `chl.csv`, normalizes time to **UTC** `time`, and writes one merged file. It **does not** rely on every sensor sharing the same timestamp: it aligns channels to the temperature timeline with **`merge_asof` (backward)** and uses **shallow-to-deep fallbacks** (e.g. `T_C_7m` when `T_C_1m` is empty), which fixes the “mostly blank spreadsheet” effect from a naive outer join. Use `--tolerance-hours` if a very slow sensor needs a wider window. Close Excel before re-running if the CSV is open. Adjust `cce2` / `cce2_19` in the URL for other moorings or deployments.

**Note:** `pco2_uatm` in the merged file is filled from `xCO2water` (dry mole fraction, µmol/mol). That is related to, but not identical to, seawater **pCO₂ µatm** — fine as a model feature if you state the caveat in your writeup.

### Option B — manual in the browser

1. Open [CCE project page](https://mooring.ucsd.edu/cce/) and click a mooring (e.g. CCE2) and a deployment (e.g. CCE2-19).
2. In the address bar, append **`csv/`** to the deployment URL (e.g. `https://mooring.ucsd.edu/cce2/cce2_19/csv/`).
3. Save individual files (`temp.csv`, `wind.csv`, …) into `data/raw/` if you prefer not to merge.

The Streamlit app loads the **first** `*.csv` in `data/raw/` whose name does **not** start with `_`. For a single-variable file, the loader recognizes `UnixTime*1000_GMT` as epoch-milliseconds UTC.

### Option C — archival QC (NetCDF)

Quality-controlled products are distributed via **OceanSITES** THREDDS (see links on the CCE page). Use `xarray` to subset and export to CSV if you extend the loader (`pip install ".[netcdf]"`).

## Prize / eligibility framing

Use **CCE moorings** as the required **Scripps** dataset for the Scripps Challenge. Additional curated or public datasets should support one coherent thesis (see `docs/methodology.rst`).

## License

Hackathon project code: MIT unless your team chooses otherwise.
