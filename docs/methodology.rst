Methodology
===========

Data ingestion
--------------

The reference workflow expects a **tabular CSV** with a detectable UTC timestamp column (``time``, ``timestamp``, ``datetime``, etc.).

Place curated mooring exports under ``data/raw/``. If no raw file exists, the repository generates a **synthetic hourly** series under ``data/processed/cce_sample_hourly.csv`` so dashboards run out of the box.

For archival, quality-controlled products, the mooring site references **OceanSITES** THREDDS holdings for CCE1/CCE2; extending ingestion to NetCDF is a natural next step (optional dependency: ``xarray``).

Analytics layer
---------------

Exploratory analysis emphasizes:

- coverage and gaps (telemetry loss is expected),
- covariance among physical and biogeochemical channels,
- diurnal and seasonal structure.

The Marimo notebook (``notebooks/cce_explore.py``) is the “lab notebook” surface; Streamlit is the judge-facing interactive demo.

Machine learning layer
----------------------

We formulate **H-hour ahead** prediction for a scalar target (for example ``sst_c``) using:

- lagged observations of multiple mooring channels,
- calendar features (hour, day-of-year, month),
- a **time-based train/validation split** (last 20% of time).

The model is a ``HistGradientBoostingRegressor`` inside a standardized pipeline. The baseline is **persistence**: predict the future value using the current observation :math:`\hat{y}(t+H)=y(t)` evaluated against the true :math:`y(t+H)` (aligned rows).

Combining external datasets
---------------------------

If you add global reanalysis or satellite fields, subset a regional box and merge on time with documented aggregation (nearest hour / daily mean). Avoid **future leakage** when forecasting.
