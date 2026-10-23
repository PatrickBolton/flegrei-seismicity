"""Magnitude of completeness (Mc) and Gutenberg-Richter tools for the earthquake catalogue.

Mc is the smallest magnitude above which the network detects practically every event.
Below it, events go missing, and how many depends on how many stations were running.
"""

import numpy as np
import pandas as pd

# GOSSIP reports Md to 0.1, except July 2021 to early September 2023, when it used 0.01.
BIN_WIDTH = 0.1

# Md comes from signal duration, apparently in whole seconds, so small magnitudes take
# discrete values: -1.11, -0.76, -0.49, -0.27, -0.08, 0.09, 0.23, 0.36, 0.48, 0.58, ...
# Below about 0.5 these are more than 0.1 apart, so counts per 0.1 bin are unreliable there.
# Only search for Mc at or above this magnitude.
MC_SEARCH_MIN = 0.5


def bin_magnitudes(md: pd.Series, width: float = BIN_WIDTH) -> pd.Series:
    """Round magnitudes to the nearest bin centre (default 0.1), so all years share one grid."""
    # The final round() removes float noise such as 0.30000000000000004.
    return (np.round(md / width) * width).round(6)


def frequency_magnitude(md: pd.Series, width: float = BIN_WIDTH) -> pd.Series:
    """Number of events in each magnitude bin, including empty bins between min and max."""
    binned = bin_magnitudes(md.dropna(), width)
    counts = binned.value_counts().sort_index()
    all_bins = np.round(np.arange(counts.index.min(), counts.index.max() + width / 2, width), 6)
    return counts.reindex(all_bins, fill_value=0).rename_axis("md_bin").rename("count")


def cumulative_counts(md: pd.Series) -> pd.Series:
    """N(>=M): the number of events with magnitude at least M, for each distinct magnitude M.

    Unlike counts per bin, this doesn't depend on how bins line up with the discrete Md values.
    """
    counts = md.dropna().value_counts().sort_index()
    return counts[::-1].cumsum()[::-1].rename_axis("md").rename("n_at_least")
