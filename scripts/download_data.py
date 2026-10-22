"""Download the raw data for this project into data/raw/.

So far this fetches the INGV-OV GOSSIP earthquake catalogue for Campi Flegrei
(doi:10.13127/gossip, CC BY 4.0). GOSSIP publishes one static CSV per year.

Usage (from the repository root):
    python scripts/download_data.py
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_DIR = ROOT / "data" / "raw" / "catalogue"

GOSSIP_URL = "https://terremoti.ov.ingv.it/gossip/flegrei/{year}/events.csv"
FIRST_YEAR = 2005  # start of the period studied in arXiv 2503.20129
EXPECTED_HEADER = "#EventID,Time,Latitude,Longitude,Depth,MD"


def download_year(year: int) -> dict:
    """Download one year of the catalogue and return a record describing it."""
    url = GOSSIP_URL.format(year=year)
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # stop on 404, 500, etc.

    # Guard against saving an error page instead of the catalogue.
    if not response.text.startswith(EXPECTED_HEADER):
        raise ValueError(f"{url} does not look like a GOSSIP catalogue CSV")

    path = CATALOGUE_DIR / f"flegrei_{year}.csv"
    path.write_bytes(response.content)

    n_events = response.text.count("\n") - 1  # minus the header line
    print(f"{year}: {n_events:6d} events -> {path.relative_to(ROOT)}")
    return {
        "url": url,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
    }


def download_catalogue() -> None:
    CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now(timezone.utc).year

    sources = {}
    for year in range(FIRST_YEAR, this_year + 1):
        sources[f"flegrei_{year}.csv"] = download_year(year)
        time.sleep(1)  # be polite to the INGV server

    # Record where each file came from and when, since data/raw/ is not in git.
    sources_path = CATALOGUE_DIR / "SOURCES.json"
    sources_path.write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Wrote {sources_path.relative_to(ROOT)}")


if __name__ == "__main__":
    download_catalogue()
