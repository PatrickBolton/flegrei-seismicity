"""Faint background map features (coastline, lakes, place names) for longitude-latitude plots.

The features come from OpenStreetMap (© OpenStreetMap contributors, ODbL), downloaded by
scripts/download_data.py into data/raw/osm/. Any figure that uses them must show the credit.
"""

import json
from pathlib import Path

import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd

CREDIT = "Map data © OpenStreetMap contributors (ODbL)"

# Names worth labelling on a caldera-scale map, as spelled in OpenStreetMap.
DEFAULT_LABELS = ["Pozzuoli", "Bagnoli", "Baia", "Quarto", "Monte di Procida", "Solfatara",
                  "Monte Nuovo", "Astroni", "Lago d'Averno", "Lago Miseno", "Lago del Fusaro",
                  "Bonifica di Agnano"]


def load_osm_features(path: Path) -> dict:
    """Read the Overpass JSON into lines and labelled points.

    Returns {"coastline": [array], "lakes": [(name, array)], "points": DataFrame}, where each
    array has columns longitude, latitude. Coastline pieces are kept as separate lines.
    """
    elements = json.loads(Path(path).read_text())["elements"]

    def line(geometry):
        return np.array([[p["lon"], p["lat"]] for p in geometry])

    coastline, lakes, points = [], [], []
    for e in elements:
        tags = e.get("tags", {})
        if tags.get("natural") == "coastline":
            coastline.append(line(e["geometry"]))
        elif tags.get("natural") == "water" and e["type"] == "way":
            lakes.append((tags["name"], line(e["geometry"])))
        elif tags.get("natural") == "water" and e["type"] == "relation":
            for member in e["members"]:  # e.g. Lago del Fusaro: drawn as its outer rings
                if member.get("role") == "outer" and "geometry" in member:
                    lakes.append((tags["name"], line(member["geometry"])))
        elif e["type"] == "node":
            points.append({"name": tags.get("name"), "kind": tags.get("natural") or tags.get("place"),
                           "longitude": e["lon"], "latitude": e["lat"]})
    return {"coastline": coastline, "lakes": lakes, "points": pd.DataFrame(points)}


def draw_basemap(ax, features: dict, labels: list[str] = DEFAULT_LABELS, colour: str = "0.55",
                 offsets: dict | None = None) -> None:
    """Draw lakes underneath the data, and the coastline and place names faintly on top of it.

    The coastline and labels go on top, with a thin white outline, so they stay visible through
    dense clusters of points. `offsets` moves chosen labels by (x, y) points to avoid collisions.
    """
    offsets = offsets or {}
    halo = [patheffects.withStroke(linewidth=2, foreground="white")]
    for xy in features["coastline"]:
        ax.plot(xy[:, 0], xy[:, 1], color=colour, linewidth=0.8, zorder=3, path_effects=halo)

    lake_label_at = {}
    for name, xy in features["lakes"]:
        ax.fill(xy[:, 0], xy[:, 1], color="0.93", edgecolor=colour, linewidth=0.6, zorder=0)
        lake_label_at[name] = xy.mean(axis=0)

    points = features["points"].drop_duplicates("name").set_index("name")
    for name in labels:
        if name in points.index:
            x, y = points.loc[name, ["longitude", "latitude"]]
        elif name in lake_label_at:
            x, y = lake_label_at[name]
        else:
            continue
        label = "Agnano" if name == "Bonifica di Agnano" else name
        ax.annotate(label, (x, y), xytext=offsets.get(name, (0, 0)), textcoords="offset points",
                    fontsize=7.5, color="0.45", style="italic", ha="center", va="center",
                    zorder=4, path_effects=halo)

    ax.text(0.01, 0.01, CREDIT, transform=ax.transAxes, fontsize=6.5, color="0.5",
            ha="left", va="bottom")
