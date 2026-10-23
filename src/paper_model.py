"""The model of Godano et al. (2025), arXiv 2503.20129, in open code.

The paper links the cumulative number of earthquakes N to the ground uplift delta (cm) at
GPS station RITE with the differential equation

    dN/d(delta) = beta * delta**alpha * N * ln(N)                      (their Eq. 4)

whose solution is ln(ln N) = beta / (alpha + 1) * delta**(alpha + 1) + C (Eq. 6). They fit
ln(ln N) = k1 * delta**k2 (Eq. 2, i.e. C = 0) by non-linear least squares and report
k1 = 0.76, k2 = 0.2, so alpha = k2 - 1 = -0.8 and beta = k1 * k2 = 0.15. They also fit a plain
exponential N = N0 * exp(b * delta) (Eq. 1: N0 = 35.25, b = 0.035). They leave out the first
30 earthquakes, which "appear to be delta independent".
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.series import station_daily

PAPER_PARAMS = {"k1": 0.76, "k2": 0.2, "alpha": -0.8, "beta": 0.15, "N0": 35.25, "b": 0.035}
SKIP_FIRST = 30  # events left out at the start, as in the paper


def uplift_cm(gps: pd.DataFrame, station: str = "RITE", zero: str = "2005-01",
              smooth_days: int = 15) -> pd.Series:
    """Daily vertical displacement in cm relative to the mean of month `zero`.

    Gaps are interpolated and the series is smoothed with a centred rolling median. Centring
    uses future days: fine for describing the curve, not for forecasting features.
    """
    up = station_daily(gps, station)
    cm = ((up - up[zero].mean()) * 100).interpolate(limit_area="inside")
    return cm.rolling(smooth_days, center=True, min_periods=smooth_days // 3).median()


def event_uplift_table(events: pd.DataFrame, delta: pd.Series, start: str, end: str) -> pd.DataFrame:
    """For each event in [start, end), in time order: its cumulative number N and delta that day."""
    selected = events[(events["time"] >= start) & (events["time"] < end)].sort_values("time")
    table = pd.DataFrame({
        "time": selected["time"].to_numpy(),
        "N": np.arange(1, len(selected) + 1),
        "delta": delta.reindex(selected["time"].dt.floor("D")).to_numpy(),
    })
    return table.dropna()


def _fit_range(table: pd.DataFrame, skip_first: int) -> pd.DataFrame:
    # delta must be positive for delta**k2.
    part = table.iloc[skip_first:]
    return part[part["delta"] > 0]


def double_log(delta, k1, k2, c=0.0):
    """ln(ln N) predicted by the paper's solution."""
    return k1 * np.asarray(delta, dtype=float) ** k2 + c


def predict_n(delta, k1, k2, c=0.0):
    """N predicted by the paper's solution: exp(exp(k1 * delta**k2 + c))."""
    return np.exp(np.exp(double_log(delta, k1, k2, c)))


def fit_double_log(table: pd.DataFrame, with_constant: bool = False,
                   skip_first: int = SKIP_FIRST) -> dict:
    """Least-squares fit of ln(ln N) = k1 * delta**k2 (+ c), as in the paper.

    Returns k1, k2, c and the implied alpha = k2 - 1 and beta = k1 * k2.
    """
    part = _fit_range(table, skip_first)
    y = np.log(np.log(part["N"].to_numpy(dtype=float)))
    if with_constant:
        (k1, k2, c), _ = curve_fit(double_log, part["delta"], y, p0=[0.76, 0.2, 0.0], maxfev=20000)
    else:
        (k1, k2), _ = curve_fit(lambda d, k1, k2: double_log(d, k1, k2), part["delta"], y,
                                p0=[0.76, 0.2])
        c = 0.0
    return {"k1": k1, "k2": k2, "c": c, "alpha": k2 - 1, "beta": k1 * k2, "events": len(part)}


def fit_exponential(table: pd.DataFrame, skip_first: int = SKIP_FIRST) -> dict:
    """Least-squares fit of N = N0 * exp(b * delta) (the paper's Eq. 1)."""
    part = _fit_range(table, skip_first)
    (n0, b), _ = curve_fit(lambda d, n0, b: n0 * np.exp(b * d), part["delta"], part["N"],
                           p0=[35.0, 0.035])
    return {"N0": n0, "b": b, "events": len(part)}
