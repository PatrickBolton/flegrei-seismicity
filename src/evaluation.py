"""Weekly forecasting targets, walk-forward evaluation, and scores.

Every forecast for week w is made from data before week w only (expanding window, no
shuffling). Weeks run Monday to Sunday (UTC) and are labelled by their Monday, as in
src/series.py.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.completeness import MC_MAIN, bin_magnitudes
from src.series import event_counts

M_LARGE = 3.0  # "larger events" for the probability target
LAMBDA_FLOOR = 0.1  # smallest expected weekly count allowed (see poisson_deviance)
P_CLIP = 1e-3  # probabilities are kept in [P_CLIP, 1 - P_CLIP] for log-loss


def weekly_targets(events: pd.DataFrame, start: str, end: str, mc: float = MC_MAIN) -> pd.DataFrame:
    """One row per week: events with M >= mc, events with M >= M_LARGE, and whether any.

    Columns: count, count_large, has_large. Weeks with no events are kept as zeros.
    """
    m = bin_magnitudes(events["md"])
    large = events[m >= M_LARGE - 1e-9]
    table = pd.DataFrame({
        "count": event_counts(events, "W", mc=mc, start=start, end=end),
        "count_large": event_counts(large, "W", start=start, end=end),
    })
    table["has_large"] = table["count_large"] > 0
    return table


def walk_forward(weekly: pd.DataFrame, forecast: Callable[[pd.DataFrame], float],
                 test_start: str, test_end: str) -> pd.Series:
    """Forecast each test week from the weeks before it only.

    `forecast` receives the history (all rows of `weekly` before the test week) and returns
    one number: an expected count, or a probability, depending on the forecaster.
    """
    test_weeks = weekly.loc[test_start:test_end].index
    values = [forecast(weekly.loc[weekly.index < week]) for week in test_weeks]
    return pd.Series(values, index=test_weeks, name=getattr(forecast, "__name__", "forecast"))


def poisson_deviance(y: pd.Series, mean: pd.Series) -> float:
    """Mean Poisson deviance of count forecasts (lower is better; 0 is perfect).

    Forecasts of 0 would give infinite deviance whenever an event happens, so expected
    counts are floored at LAMBDA_FLOOR. The same floor applies to every forecaster.
    """
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mean, dtype=float), LAMBDA_FLOOR)
    y_log = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (y_log - (y - mu))))


def brier_score(happened: pd.Series, p: pd.Series) -> float:
    """Mean squared error of probability forecasts (lower is better)."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(happened, dtype=float)) ** 2))


def log_loss(happened: pd.Series, p: pd.Series) -> float:
    """Mean negative log-likelihood of probability forecasts (lower is better)."""
    p = np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)
    y = np.asarray(happened, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(happened: pd.Series, p: pd.Series,
                      bins: tuple = (0, 0.05, 0.1, 0.2, 0.3, 1.0)) -> pd.DataFrame:
    """Group weeks by forecast probability; compare mean forecast with observed frequency.

    For a well-calibrated forecaster, mean_forecast and observed_share are close in every bin.
    """
    groups = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(happened, dtype=float)}).groupby(
        pd.cut(np.asarray(p), bins, include_lowest=True), observed=True)
    return pd.DataFrame({"weeks": groups.size(), "mean_forecast": groups["p"].mean(),
                         "observed_share": groups["y"].mean()}).rename_axis("forecast_bin")


def count_scores(y: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """Poisson deviance and mean absolute error for each count forecaster."""
    return pd.DataFrame({
        name: {"poisson_deviance": poisson_deviance(y, f), "mae": float(np.mean(np.abs(y - f)))}
        for name, f in forecasts.items()
    }).T


def probability_scores(happened: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """Brier score, log-loss and mean forecast probability for each probability forecaster."""
    return pd.DataFrame({
        name: {"brier": brier_score(happened, p), "log_loss": log_loss(happened, p),
               "mean_p": float(np.mean(p))}
        for name, p in forecasts.items()
    }).T
