"""Generate synthetic demo CSV under data/processed/."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cce_hack.sample_data import build_synthetic_hourly  # noqa: E402


def main() -> None:
    out = ROOT / "data" / "processed" / "cce_sample_hourly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_synthetic_hourly().to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
