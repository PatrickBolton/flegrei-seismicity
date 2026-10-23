"""Load the Campi Flegrei GPS time series (De Martino et al., doi:10.5281/zenodo.22259337).

Each raw file (e.g. RITE_2000_2025.txt) has seven space-separated columns and no header:
MJD, North, East, Up, and the uncertainties of North, East and Up, all in metres.
See data/raw/gps_21_stations/Time_Series_Format.txt.

Values are displacements from an arbitrary reference position: only changes are meaningful.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Note the order: North comes before East in these files.
COLUMNS = ["mjd", "north", "east", "up", "sigma_north", "sigma_east", "sigma_up"]

# Modified Julian Date counts days since midnight UTC on this date.
MJD_EPOCH = pd.Timestamp("1858-11-17", tz="UTC")


def mjd_to_datetime(mjd: pd.Series) -> pd.Series:
    """Convert Modified Julian Dates to UTC timestamps."""
    return MJD_EPOCH + pd.to_timedelta(mjd, unit="D")


def load_station(path: Path) -> pd.DataFrame:
    """Load one station file into a table with a UTC `date` column."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df.insert(0, "date", mjd_to_datetime(df.pop("mjd")))
    df.insert(1, "station", path.name.split("_")[0])  # "RITE_2000_2025.txt" -> "RITE"
    return df


def load_station_coordinates(path: Path) -> pd.DataFrame:
    """Station latitude and longitude, from data/reference/gps_stations.csv (see its README)."""
    return pd.read_csv(path).set_index("station")


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees (works on arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_all_stations(folder: Path) -> pd.DataFrame:
    """Load every station file in `folder` into one tidy table (one row per station per day)."""
    # Station IDs are four upper-case letters or digits; this skips Time_Series_Format.txt.
    paths = sorted(folder.glob("[A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9]_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No station files in {folder}")

    df = pd.concat([load_station(p) for p in paths], ignore_index=True)
    df["station"] = df["station"].astype("category")
    return df.sort_values(["station", "date"], ignore_index=True)
