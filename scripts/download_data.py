"""Download the raw data for this project into data/raw/.

Fetches:
- the INGV-OV GOSSIP earthquake catalogue for Campi Flegrei
  (doi:10.13127/gossip), one static CSV per year;
- GPS time series of 21 stations (doi:10.5281/zenodo.22259337);
- GPS vertical time series of station RITE (doi:10.5281/zenodo.17132077).

All data is CC BY 4.0. Each folder gets a SOURCES.json recording where every
file came from and when, since data/raw/ is not committed to git.

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
RAW_DIR = ROOT / "data" / "raw"

GOSSIP_URL = "https://terremoti.ov.ingv.it/gossip/flegrei/{year}/events.csv"
FIRST_YEAR = 2005  # start of the period studied in arXiv 2503.20129
EXPECTED_HEADER = "#EventID,Time,Latitude,Longitude,Depth,MD"

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
ZENODO_RECORDS = {
    "gps_21_stations": 22259337,  # De Martino et al. (2026)
    "gps_rite": 17132077,  # De Martino et al. (2025)
}


def fetch(url: str) -> bytes:
    """Download a URL and return its content, stopping on HTTP errors."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # stop on 404, 500, etc.
    return response.content


def source_record(url: str, content: bytes) -> dict:
    """Describe a downloaded file for SOURCES.json."""
    return {
        "url": url,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def write_sources(folder: Path, sources: dict) -> None:
    path = folder / "SOURCES.json"
    path.write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Wrote {path.relative_to(ROOT)}")


def download_catalogue() -> None:
    folder = RAW_DIR / "catalogue"
    folder.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now(timezone.utc).year

    sources = {}
    for year in range(FIRST_YEAR, this_year + 1):
        url = GOSSIP_URL.format(year=year)
        content = fetch(url)

        # Guard against saving an error page instead of the catalogue.
        if not content.decode("utf-8").startswith(EXPECTED_HEADER):
            raise ValueError(f"{url} does not look like a GOSSIP catalogue CSV")

        name = f"flegrei_{year}.csv"
        (folder / name).write_bytes(content)
        sources[name] = source_record(url, content)

        n_events = content.count(b"\n") - 1  # minus the header line
        print(f"{year}: {n_events:6d} events -> {name}")
        time.sleep(1)  # be polite to the INGV server

    write_sources(folder, sources)


def download_zenodo_record(record_id: int, folder: Path) -> None:
    """Download every file in a Zenodo record and check it against Zenodo's MD5."""
    folder.mkdir(parents=True, exist_ok=True)

    # The Zenodo API lists a record's files, with download links and checksums.
    record = json.loads(fetch(ZENODO_API.format(record_id=record_id)))
    print(f"Zenodo {record_id}: {record['metadata']['title']}")

    sources = {}
    for file in record["files"]:
        name, url = file["key"], file["links"]["self"]
        content = fetch(url)

        # Zenodo gives checksums as "md5:<hex>". A mismatch means a broken download.
        expected_md5 = file["checksum"].removeprefix("md5:")
        if hashlib.md5(content).hexdigest() != expected_md5:
            raise ValueError(f"{name}: MD5 does not match Zenodo's checksum")

        (folder / name).write_bytes(content)
        sources[name] = source_record(url, content)
        sources[name]["doi"] = record["doi"]
        print(f"  {name} ({len(content):,} bytes, checksum OK)")

    write_sources(folder, sources)


if __name__ == "__main__":
    download_catalogue()
    for folder_name, record_id in ZENODO_RECORDS.items():
        download_zenodo_record(record_id, RAW_DIR / folder_name)
