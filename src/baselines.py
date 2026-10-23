"""Simple forecasters that use past seismicity only: the bar later models must beat.

Each forecaster is a function (history, current) -> number, for src.evaluation.walk_forward.
`history` is the weekly table up to (not including) the week being forecast; `current` holds
that week's known features. These baselines use history only and ignore `current`.
Functions named make_* build a forecaster with chosen settings.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---- Expected number of events next week (M >= Mc) ----


def persistence(history: pd.DataFrame, current: pd.Series | None = None) -> float:
    """Next week = this week."""
    return float(history["count"].iloc[-1])


def make_rolling_mean(weeks: int):
    """Next week = the mean of the last `weeks` weeks."""
    def rolling_mean(history: pd.DataFrame, current: pd.Series | None = None) -> float:
        return float(history["count"].iloc[-weeks:].mean())
    rolling_mean.__name__ = f"rolling_mean_{weeks}w"
    return rolling_mean


def make_trend(weeks: int):
    """Poisson regression of weekly counts on time over the last `weeks` weeks, extrapolated
    one week ahead: log(expected count) = a + b * week. Captures a growing (or falling) rate.
    """
    def trend(history: pd.DataFrame, current: pd.Series | None = None) -> float:
        y = history["count"].iloc[-weeks:].to_numpy(dtype=float)
        if y.sum() == 0:
            return 0.0
        t = np.arange(len(y), dtype=float)
        fit = sm.GLM(y, sm.add_constant(t), family=sm.families.Poisson()).fit()
        a, b = fit.params
        return float(np.exp(a + b * len(y)))
    trend.__name__ = f"trend_{weeks}w"
    return trend


# ---- Probability of at least one M >= 3 event next week ----


def make_climatology(weeks: int | None = None):
    """Share of past weeks with an M >= 3 event: all history, or the last `weeks` weeks."""
    def climatology(history: pd.DataFrame, current: pd.Series | None = None) -> float:
        past = history if weeks is None else history.iloc[-weeks:]
        return float(past["has_large"].mean())
    climatology.__name__ = "climatology_all" if weeks is None else f"climatology_{weeks}w"
    return climatology


def make_from_count(count_forecast, share_weeks: int = 156):
    """Turn an expected count into P(M >= 3): 1 - exp(-expected * share).

    `share` is the fraction of past events (M >= Mc) that reached M >= 3, over the last
    `share_weeks` weeks. Using the observed share avoids assuming a Gutenberg-Richter b-value,
    which phase 4 found underpredicts large events.
    """
    def from_count(history: pd.DataFrame, current: pd.Series | None = None) -> float:
        past = history.iloc[-share_weeks:]
        share = past["count_large"].sum() / max(past["count"].sum(), 1)
        return float(1 - np.exp(-count_forecast(history, current) * share))
    from_count.__name__ = f"from_{count_forecast.__name__}"
    return from_count
