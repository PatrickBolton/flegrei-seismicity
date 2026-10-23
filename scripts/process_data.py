"""Turn the raw downloads in data/raw/ into clean tables in data/processed/.

Run scripts/download_data.py first. Then, from the repository root:
    python scripts/process_data.py
"""

from pathlib import Path

from src.catalogue import load_catalogue
from src.gps import load_all_stations

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def process_gps() -> None:
    df = load_all_stations(RAW_DIR / "gps_21_stations")

    # Cheap checks that would catch a parsing mistake or a changed file format.
    assert not df.duplicated(["station", "date"]).any(), "duplicate station-days"
    assert not df.isna().any().any(), "missing values"

    path = PROCESSED_DIR / "gps_daily.parquet"
    df.to_parquet(path, index=False)
    print(
        f"GPS: {len(df):,} rows, {df['station'].nunique()} stations, "
        f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d} -> {path.relative_to(ROOT)}"
    )


def process_catalogue() -> None:
    df = load_catalogue(RAW_DIR / "catalogue")

    # Every event must have an ID and a time; location and magnitude may be missing.
    assert df["event_id"].notna().all(), "missing event IDs"
    assert df["time"].notna().all(), "unparsed times"

    path = PROCESSED_DIR / "catalogue.parquet"
    df.to_parquet(path, index=False)
    print(
        f"Catalogue: {len(df):,} events, "
        f"{df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d} -> {path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    process_gps()
    process_catalogue()
