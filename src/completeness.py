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


def mc_maxc(md: pd.Series, min_mag: float = MC_SEARCH_MIN, width: float = BIN_WIDTH) -> float:
    """Mc by maximum curvature: the most common magnitude bin at or above `min_mag`.

    If the result equals `min_mag`, the peak may lie lower, so it only says Mc <= min_mag.
    Woessner & Wiemer (2005) add +0.2 because this method tends to underestimate Mc;
    callers apply that themselves, and it isn't meaningful when the result is `min_mag`.
    """
    counts = frequency_magnitude(md, width)
    return float(counts[counts.index >= min_mag - 1e-9].idxmax())


def b_value_mle(md: pd.Series, mc: float, width: float = BIN_WIDTH) -> tuple[float, float, int]:
    """Gutenberg-Richter b-value of events with M >= mc, by maximum likelihood.

    Aki (1965), with Utsu's correction for binned magnitudes: the lower edge is mc - width/2.
    The uncertainty is from Shi & Bolt (1982). Returns (b, b_sigma, number of events used).
    """
    m = bin_magnitudes(md.dropna(), width)
    m = m[m >= mc - 1e-9]
    n = len(m)
    if n < 2:
        return np.nan, np.nan, n
    b = np.log10(np.e) / (m.mean() - (mc - width / 2))
    b_sigma = 2.3 * b**2 * np.sqrt(((m - m.mean()) ** 2).sum() / (n * (n - 1)))
    return float(b), float(b_sigma), n


def b_stability(md: pd.Series, min_mag: float = MC_SEARCH_MIN, max_mag: float = 1.5,
                width: float = BIN_WIDTH, span: float = 0.5) -> pd.DataFrame:
    """The table behind mc_mbs: for each trial Mc, b, its uncertainty, and b averaged above.

    b_ave is the mean of b at Mc, Mc + width, ..., Mc + span (Woessner & Wiemer 2005).
    """
    rows = []
    for mc in np.round(np.arange(min_mag, max_mag + width / 2, width), 6):
        b, b_sigma, n = b_value_mle(md, mc, width)
        rows.append({"mc": mc, "b": b, "b_sigma": b_sigma, "n": n})
    table = pd.DataFrame(rows).set_index("mc")
    k = int(round(span / width)) + 1  # number of trial Mc values averaged
    table["b_ave"] = table["b"][::-1].rolling(k).mean()[::-1]
    return table


def mc_mbs(md: pd.Series, min_mag: float = MC_SEARCH_MIN, width: float = BIN_WIDTH,
           span: float = 0.5, min_events: int = 50) -> float:
    """Mc by b-value stability (Cao & Gao 2002; Woessner & Wiemer 2005).

    Above the true Mc, b no longer changes when Mc is raised. Mc is the lowest trial value
    where |b_ave - b| <= b_sigma. Returns NaN if no trial value with at least `min_events`
    events passes. As with mc_maxc, a result equal to `min_mag` only means Mc <= min_mag.
    """
    table = b_stability(md, min_mag, min_mag + 1.5, width, span).dropna()
    table = table[table["n"] >= min_events]
    stable = table[(table["b_ave"] - table["b"]).abs() <= table["b_sigma"]]
    return float(stable.index[0]) if len(stable) else np.nan


def bootstrap(md: pd.Series, statistic, n_boot: int = 500, seed: int = 42) -> np.ndarray:
    """Apply `statistic` to `n_boot` resamples of the magnitudes (drawn with replacement)."""
    rng = np.random.default_rng(seed)
    values = md.dropna().to_numpy()
    return np.array([statistic(pd.Series(rng.choice(values, size=len(values))))
                     for _ in range(n_boot)])


def _windows(n_events: int, size: int, step: int):
    """Start and stop positions of windows of `size` consecutive events, moving by `step`."""
    for start in range(0, n_events - size + 1, step):
        yield start, start + size


def mc_over_time(events: pd.DataFrame, size: int = 1000, step: int = 250) -> pd.DataFrame:
    """MAXC Mc in moving windows of `size` events (with a magnitude), in time order.

    `events` needs `time` and `md` columns. Each row describes one window: its first, middle
    (median) and last event time, and its Mc. Windows overlap, so neighbouring rows aren't
    independent.
    """
    ev = events.dropna(subset=["md"]).sort_values("time", ignore_index=True)
    rows = []
    for start, stop in _windows(len(ev), size, step):
        w = ev.iloc[start:stop]
        rows.append({"start": w["time"].iloc[0], "mid": w["time"].median(),
                     "end": w["time"].iloc[-1], "mc_maxc": mc_maxc(w["md"])})
    return pd.DataFrame(rows)


def b_value_over_time(events: pd.DataFrame, mc: float = MC_SEARCH_MIN, size: int = 200,
                      step: int = 50) -> pd.DataFrame:
    """b-value in moving windows of `size` events with M >= mc, in time order.

    Same layout as mc_over_time, with columns b and b_sigma instead of mc_maxc.
    """
    ev = events.dropna(subset=["md"])
    ev = ev[bin_magnitudes(ev["md"]) >= mc - 1e-9].sort_values("time", ignore_index=True)
    rows = []
    for start, stop in _windows(len(ev), size, step):
        w = ev.iloc[start:stop]
        b, b_sigma, _ = b_value_mle(w["md"], mc)
        rows.append({"start": w["time"].iloc[0], "mid": w["time"].median(),
                     "end": w["time"].iloc[-1], "b": b, "b_sigma": b_sigma})
    return pd.DataFrame(rows)


def cumulative_counts(md: pd.Series) -> pd.Series:
    """N(>=M): the number of events with magnitude at least M, for each distinct magnitude M.

    Unlike counts per bin, this doesn't depend on how bins line up with the discrete Md values.
    """
    counts = md.dropna().value_counts().sort_index()
    return counts[::-1].cumsum()[::-1].rename_axis("md").rename("n_at_least")
