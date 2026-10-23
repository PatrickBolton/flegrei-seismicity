"""Load the INGV-OV earthquake catalogue for Campi Flegrei (GOSSIP, doi:10.13127/gossip).

Each raw file (e.g. flegrei_2025.csv) is one year of events with a header row:
#EventID, Time, Latitude, Longitude, Depth, MD, MD Error, Area, Type, Level.

`Time` is UTC: the GOSSIP web page labels the same column "Tempo (UTC)".
`MD` is the duration magnitude (Md); small negative values are normal for tiny events.
Location (lat, lon, depth) and magnitude are blank for some events; they load as NaN.
"""

from pathlib import Path

import pandas as pd

# Raw header -> tidy snake_case names.
RENAME = {
    "#EventID": "event_id",
    "Time": "time",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Depth": "depth_km",
    "MD": "md",
    "MD Error": "md_error",
    "Area": "area",
    "Type": "type",
    "Level": "level",
}


def load_year(path: Path) -> pd.DataFrame:
    """Load one yearly file into a table with a UTC `time` column."""
    df = pd.read_csv(path).rename(columns=RENAME)
    # Times look like "2025-01-01 02:49:16.009" with 0-3 decimals; "ISO8601" accepts all of them.
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    return df


def load_catalogue(folder: Path) -> pd.DataFrame:
    """Load every yearly file in `folder` into one table (one row per event), sorted by time."""
    paths = sorted(folder.glob("flegrei_[0-9][0-9][0-9][0-9].csv"))
    if not paths:
        raise FileNotFoundError(f"No catalogue files in {folder}")

    df = pd.concat([load_year(p) for p in paths], ignore_index=True)
    for col in ["area", "type", "level"]:
        df[col] = df[col].astype("category")
    return df.sort_values("time", ignore_index=True)


def drop_located_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop located events that repeat another event's time, latitude, longitude and depth.

    Such rows are the same event entered twice under different IDs. Keep the copy with the
    larger magnitude; a missing magnitude counts as the smallest.
    """
    key = ["time", "latitude", "longitude", "depth_km"]
    located = df["latitude"].notna()
    is_copy = (
        df[located]
        .sort_values("md", ascending=False, na_position="last")
        .duplicated(key)
    )
    return df.drop(is_copy[is_copy].index).reset_index(drop=True)


def flag_shared_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add `shared_time`: True when another event has exactly the same time.

    After drop_located_duplicates these are almost all unlocated events (mostly 2008-2009).
    They may be separate events stamped with a common time, or duplicates: we can't tell.
    """
    return df.assign(shared_time=df["time"].duplicated(keep=False))
