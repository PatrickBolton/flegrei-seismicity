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
import statsmodels.api as sm
from scipy.optimize import curve_fit

from src.series import on_weekday, station_daily, trailing_rate

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


# ---- The same model as a weekly forecast ----
#
# Written in time, Eq. 4 gives the rate: dN/dt = beta * delta**alpha * N * ln(N) * d(delta)/dt.
# For one week, the expected number of events is therefore
#
#     mu = beta * delta**alpha * N * ln(N) * (uplift during the week)
#
# with delta and N at the start of the week. Taking logs, log(mu) = log(beta) + alpha*log(delta)
# + log(N ln N * uplift): a Poisson regression on log(delta) with offset log(N ln N * uplift),
# which fits beta and alpha directly to weekly counts by maximum likelihood.

RATE_WINDOW_DAYS = 28  # window of the uplift rate used for the weekly uplift
MIN_DELTA_CM = 5.0  # weeks with less uplift than this are left out of fits (delta**alpha blows up)


def add_uplift_features(weekly: pd.DataFrame, gps: pd.DataFrame, station: str = "RITE",
                        zero: str = "2005-01") -> pd.DataFrame:
    """Add to the weekly table what the rate model needs, all in cm:

    - N_before: events counted (column "count") in all earlier weeks;
    - delta_start: uplift at the start of the week (7-day mean up to the Sunday before);
    - uplift_prev: uplift per week over the 28 days up to the Sunday before (known when the
      forecast is made);
    - uplift_oracle: uplift per week over the 28 days up to the week's own Sunday. This uses the
      week being forecast, so a forecast with it is an upper bound, not a real forecast.
    """
    up = station_daily(gps, station)
    cm = (up - up[zero].mean()) * 100
    rate_per_week = trailing_rate(cm, RATE_WINDOW_DAYS) * 7 / 365.25
    level = cm.rolling(7, min_periods=3).mean()  # trailing: no future days

    out = weekly.copy()
    out["N_before"] = out["count"].cumsum().shift(1, fill_value=0)
    out["delta_start"] = on_weekday(level, out.index, -1)
    out["uplift_prev"] = on_weekday(rate_per_week, out.index, -1)
    out["uplift_oracle"] = on_weekday(rate_per_week, out.index, 6)
    return out


def rate_offset(n_before, uplift):
    """log(N ln N * uplift); -inf where the model can't be applied (N <= 1 or no uplift)."""
    n = np.asarray(n_before, dtype=float)
    u = np.asarray(uplift, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((n > 1) & (u > 0), np.log(n * np.log(np.where(n > 1, n, 2)) * u), -np.inf)


def fit_rate_model(history: pd.DataFrame) -> dict:
    """Fit log(beta) and alpha by Poisson regression on past weeks.

    Past weeks are fully observed, so their actual uplift (uplift_oracle) is used.
    """
    offset = rate_offset(history["N_before"], history["uplift_oracle"])
    usable = np.isfinite(offset) & (history["delta_start"] > MIN_DELTA_CM).to_numpy()
    x = sm.add_constant(np.log(history.loc[usable, "delta_start"].to_numpy()))
    fit = sm.GLM(history.loc[usable, "count"].to_numpy(dtype=float), x,
                 family=sm.families.Poisson(), offset=offset[usable]).fit()
    log_beta, alpha = fit.params
    return {"beta": float(np.exp(log_beta)), "alpha": float(alpha), "weeks": int(usable.sum())}


def rate_expected(current: pd.Series, beta: float, alpha: float, uplift_column: str) -> float:
    """Expected events in the week: beta * delta**alpha * N ln N * uplift (0 if no uplift)."""
    offset = rate_offset([current["N_before"]], [current[uplift_column]])[0]
    if not np.isfinite(offset) or current["delta_start"] <= 0:
        return 0.0
    return float(beta * current["delta_start"] ** alpha * np.exp(offset))


def make_rate_forecaster(uplift_column: str = "uplift_prev", params: dict | None = None):
    """Weekly forecaster from the rate form of the paper's model.

    With `params` (beta, alpha) fixed, e.g. PAPER_PARAMS; otherwise refitted on the history
    before every test week. uplift_column "uplift_prev" is a real forecast (the last 28 days'
    uplift rate carried forward); "uplift_oracle" uses the week's actual uplift (upper bound).
    """
    def rate_model(history: pd.DataFrame, current: pd.Series) -> float:
        p = params if params is not None else fit_rate_model(history)
        return rate_expected(current, p["beta"], p["alpha"], uplift_column)
    source = "paper" if params is not None else "refit"
    rate_model.__name__ = f"rate_{source}_{uplift_column.removeprefix('uplift_')}"
    return rate_model
