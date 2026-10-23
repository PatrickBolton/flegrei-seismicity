"""Turn the event and GPS tables into regular time series (daily, weekly or monthly, UTC).

Phase 4 uses these for plots; phase 6 builds features and targets from the same functions.
Weeks run Monday to Sunday and are labelled by their Monday; months by their first day.
"""

import pandas as pd

from src.completeness import bin_magnitudes


GRID_STEP = {"D": "D", "W": "7D", "M": "MS"}


def period_start(time: pd.Series, freq: str = "W") -> pd.Series:
    """The start of the day ("D"), Monday-to-Sunday week ("W") or month ("M") of each time."""
    day = time.dt.floor("D")
    if freq == "D":
        return day
    if freq == "W":
        return day - pd.to_timedelta(day.dt.weekday, unit="D")
    if freq == "M":
        return day - pd.to_timedelta(day.dt.day - 1, unit="D")
    raise ValueError(f"freq must be 'D', 'W' or 'M', not {freq!r}")


def event_counts(events: pd.DataFrame, freq: str = "W", mc: float | None = None,
                 start: str | None = None, end: str | None = None) -> pd.Series:
    """Number of events per day, week or month, including periods with no events.

    With `mc`, only events with magnitude (binned to 0.1) >= mc count; events without a
    magnitude are then left out. The grid runs from the period containing `start` to the one
    containing `end` (default: first and last event in `events`), so that quiet periods
    appear as zeros rather than being missing.
    """
    if mc is not None:
        events = events[bin_magnitudes(events["md"]) >= mc - 1e-9]
    periods = period_start(events["time"], freq)

    first = pd.Series(pd.Timestamp(start, tz="UTC") if start else events["time"].min())
    last = pd.Series(pd.Timestamp(end, tz="UTC") if end else events["time"].max())
    grid = pd.date_range(period_start(first, freq).iloc[0], period_start(last, freq).iloc[0],
                         freq=GRID_STEP[freq])

    counts = periods.value_counts().reindex(grid, fill_value=0)
    return counts.rename_axis("period_start").rename("events")


def station_daily(gps: pd.DataFrame, station: str, component: str = "up") -> pd.Series:
    """One station's displacement (metres) on a daily grid from its first to last day.

    Days without a GPS solution are NaN, not filled: how to fill is a feature decision.
    """
    s = gps.loc[gps["station"] == station].set_index("date")[component]
    grid = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(grid).rename_axis("date").rename(station)


def on_weekday(daily: pd.Series, weeks: pd.DatetimeIndex, offset_days: int = 6) -> pd.Series:
    """Sample a daily series once per week: the value on Monday + `offset_days` of each week.

    The result is indexed by the weeks' Mondays, like event_counts(..., "W"). offset_days=6
    (Sunday) uses data up to the end of the week, so it can describe the week just finished
    when forecasting the next one; offset_days=3 (Thursday) is the middle of the week.
    """
    values = daily.reindex(weeks + pd.Timedelta(days=offset_days))
    values.index = weeks
    return values


def trailing_rate(daily: pd.Series, window_days: int = 90, min_fraction: float = 0.5) -> pd.Series:
    """Rate of change per year: the least-squares slope over the `window_days` up to each day.

    The window is trailing (it ends on the day it is labelled with), so the rate uses no
    future data and can later serve as a forecasting feature. Days with fewer than
    `min_fraction` of the window observed get NaN. Daily GPS noise (~10 mm vertical) is too
    large to difference day to day; a fitted slope averages it out.
    """
    t = pd.Series((daily.index - daily.index[0]).days, index=daily.index, dtype=float)
    observed = daily.notna()
    t = t.where(observed)  # use only days with data in every sum

    roll = lambda s: s.rolling(window_days, min_periods=1).sum()  # noqa: E731
    n = roll(observed.astype(float))
    st, sy = roll(t), roll(daily)
    stt, sty = roll(t * t), roll(t * daily)

    slope_per_day = (n * sty - st * sy) / (n * stt - st**2)
    slope_per_day = slope_per_day.where(n >= min_fraction * window_days)
    return (slope_per_day * 365.25).rename(daily.name)
