from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SAMPLE_CSV = DATA_PROC / "cce_sample_hourly.csv"
MOORING_MASTER_FILENAME = "mooring_master.csv"

DEFAULT_MOORING = "CCE2"
DEFAULT_HORIZON_HOURS = 24
LAG_HOURS = (1, 6, 12, 24, 48, 72)

# Reference mooring positions (judge map — hardcoded; independent of CSV lat/lon).
MOORING_SITES: dict[str, dict[str, float]] = {
    "CCE1": {"latitude": 33.45, "longitude": -122.8},
    "CCE2": {"latitude": 34.3, "longitude": -120.7},
}

# Free-tier defaults (override with env GEMINI_MODEL / GROQ_MODEL).
# 1.5 Flash works with google-generativeai<0.8 (see pyproject pins for langchain/grpc compatibility).
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
