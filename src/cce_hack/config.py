from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SAMPLE_CSV = DATA_PROC / "cce_sample_hourly.csv"

DEFAULT_MOORING = "CCE2"
DEFAULT_HORIZON_HOURS = 24
LAG_HOURS = (1, 6, 12, 24, 48, 72)

# Reference mooring positions (judge map — hardcoded; independent of CSV lat/lon).
MOORING_SITES: dict[str, dict[str, float]] = {
    "CCE1": {"latitude": 33.45, "longitude": -122.8},
    "CCE2": {"latitude": 34.3, "longitude": -120.7},
}

# Default Anthropic model id (override with env ANTHROPIC_MODEL).
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
