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
