"""Turn the raw downloads in data/raw/ into clean tables in data/processed/.

Run scripts/download_data.py first. Then, from the repository root:
    python scripts/process_data.py
"""

from pathlib import Path

from src.catalogue import drop_located_duplicates, flag_shared_time, load_catalogue
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
    raw = load_catalogue(RAW_DIR / "catalogue")
    df = flag_shared_time(drop_located_duplicates(raw))

    # Every event must have an ID and a time; location and magnitude may be missing.
    assert df["event_id"].notna().all(), "missing event IDs"
    assert df["time"].notna().all(), "unparsed times"
    # As of Oct 2026 every event is an earthquake. If other types (e.g. explosions) appear
    # in a new download, look at them before deciding to drop them.
    assert (df["type"] == "earthquake").all(), f"new event types: {set(df['type'])}"
    # One EventID (1057) is reused for two different events, so the ID alone is not unique.
    assert not df.duplicated(["time", "event_id"]).any(), "duplicate (time, event_id)"

    path = PROCESSED_DIR / "catalogue.parquet"
    df.to_parquet(path, index=False)
    print(
        f"Catalogue: {len(df):,} events ({len(raw) - len(df)} located duplicates dropped, "
        f"{df['shared_time'].sum()} flagged shared_time), "
        f"{df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d} -> {path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    process_gps()
    process_catalogue()
