"""iNaturalist observation fetch (metadata only — no images)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

# iNaturalist: place_id **14** is California, US. (ID **5** is Washington, DC — not the California Current.)
INAT_PLACE_CALIFORNIA: int = 14

# Common name -> scientific name (API `taxon_name`)
INAT_SPECIES: dict[str, str] = {
    "Northern Anchovy": "Engraulis mordax",
    "Pacific Sardine": "Sardinops sagax",
    "Humboldt Squid": "Dosidicus gigas",
    "Blue Whale": "Balaenoptera musculus",
}

_INAT_COLUMNS = ["date", "lat", "lon", "species", "common_name"]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_INAT_COLUMNS)


def _parse_observation(rec: dict[str, Any]) -> dict[str, Any] | None:
    obs_on = rec.get("observed_on") or rec.get("observed_on_string")
    if not obs_on:
        return None
    lat = rec.get("latitude")
    lon = rec.get("longitude")
    if lat is None or lon is None:
        loc = rec.get("location")
        if isinstance(loc, str) and "," in loc:
            parts = loc.split(",")
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
            except (TypeError, ValueError):
                return None
        else:
            gj = rec.get("geojson") or {}
            coords = gj.get("coordinates")
            if coords and len(coords) >= 2:
                lon, lat = float(coords[0]), float(coords[1])
            else:
                return None
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None

    taxon = rec.get("taxon") or {}
    sci = taxon.get("name") or rec.get("species_guess") or ""
    # API uses preferred_common_name; tolerate taxon.common_name dict shape
    common = taxon.get("preferred_common_name")
    if not common:
        cn = taxon.get("common_name")
        if isinstance(cn, dict):
            common = cn.get("name")
        elif isinstance(cn, str):
            common = cn
    if not common:
        common = ""

    return {
        "date": str(obs_on)[:10],
        "lat": lat_f,
        "lon": lon_f,
        "species": str(sci),
        "common_name": str(common),
    }


def fetch_inaturalist_observations(
    taxon_name: str,
    *,
    place_id: int = INAT_PLACE_CALIFORNIA,
    days_back: int = 365,
    quality_grade: str = "research",
    timeout_s: float = 45.0,
) -> pd.DataFrame:
    """
    GET https://api.inaturalist.org/v1/observations for one taxon.

    Returns columns: date, lat, lon, species, common_name (empty frame on failure).
    """
    try:
        import requests
    except ImportError:
        return _empty_frame()

    d2 = date.today()
    d1 = d2 - timedelta(days=int(days_back))
    params = {
        "taxon_name": taxon_name,
        "place_id": place_id,
        "quality_grade": quality_grade,
        "per_page": 200,
        "order_by": "observed_on",
        "order": "desc",
        "d1": d1.isoformat(),
        "d2": d2.isoformat(),
    }
    url = "https://api.inaturalist.org/v1/observations"
    try:
        r = requests.get(url, params=params, timeout=timeout_s)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return _empty_frame()

    rows: list[dict[str, Any]] = []
    for rec in payload.get("results") or []:
        parsed = _parse_observation(rec if isinstance(rec, dict) else {})
        if parsed:
            rows.append(parsed)

    if not rows:
        return _empty_frame()
    return pd.DataFrame(rows)


def fetch_all_species_observations(
    species: dict[str, str],
    *,
    place_id: int = INAT_PLACE_CALIFORNIA,
    days_back: int = 365,
    quality_grade: str = "research",
) -> pd.DataFrame:
    """Fetch each scientific name; tag rows with display common name from ``species`` keys."""
    parts: list[pd.DataFrame] = []
    for common, sci in species.items():
        sub = fetch_inaturalist_observations(
            sci,
            place_id=place_id,
            days_back=days_back,
            quality_grade=quality_grade,
        )
        if sub.empty:
            continue
        sub = sub.copy()
        sub["common_name"] = common
        sub["species"] = sci
        parts.append(sub)
    if not parts:
        return _empty_frame()
    out = pd.concat(parts, ignore_index=True)
    return out[_INAT_COLUMNS]


def synthetic_species_observations(
    anomaly_times: pd.Series | np.ndarray | list,
    *,
    n: int = 200,
    days_back: int = 365,
    species_keys: list[str] | None = None,
    seed: int = 42,
    lat_bounds: tuple[float, float] = (33.0, 35.0),
    lon_bounds: tuple[float, float] = (-123.0, -119.0),
    window_days: int = 14,
    reduction: float = 0.6,
) -> pd.DataFrame:
    """
    ~``n`` synthetic sightings near the California coast.

    During ±``window_days`` around each anomaly timestamp, keep probability ``1 - reduction``.
    """
    rng = np.random.default_rng(seed)
    keys = species_keys or list(INAT_SPECIES.keys())
    sci = [INAT_SPECIES[k] for k in keys]

    d2 = date.today()
    d1 = d2 - timedelta(days=int(days_back))
    span_days = max((d2 - d1).days, 1)

    ats_utc: list[pd.Timestamp] = []
    for t in pd.to_datetime(pd.Series(anomaly_times), utc=True, errors="coerce").dropna():
        ats_utc.append(pd.Timestamp(t).tz_convert("UTC"))

    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = n * 80
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        off = int(rng.integers(0, span_days + 1))
        day_date = d1 + timedelta(days=off)
        day = day_date.isoformat()
        ts = pd.Timestamp(day_date, tz="UTC")

        in_window = False
        for at in ats_utc:
            if abs((ts - at).total_seconds()) <= window_days * 86400:
                in_window = True
                break
        if in_window and rng.random() < reduction:
            continue

        lat = rng.uniform(lat_bounds[0], lat_bounds[1])
        lon = rng.uniform(lon_bounds[0], lon_bounds[1])
        i = int(rng.integers(0, len(keys)))
        rows.append(
            {
                "date": day,
                "lat": float(lat),
                "lon": float(lon),
                "species": sci[i],
                "common_name": keys[i],
            }
        )

    return pd.DataFrame(rows, columns=_INAT_COLUMNS)
