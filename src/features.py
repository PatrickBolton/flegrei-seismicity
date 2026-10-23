"""Weekly features for forecasting models, each known before the week starts.

Every feature for the week starting on Monday w uses only events and GPS data up to the
Sunday before (w - 1 day). check_no_lookahead() verifies this by recomputing features from
truncated data. Features are grouped so models can be trained with or without GPS:

- SEISMIC_FEATURES: from past earthquakes only.
- GPS_FEATURES: from past ground deformation only.
- COMBINED_FEATURES: mix both (inspired by the paper's model: N x uplift rate).
"""

import numpy as np
import pandas as pd

from src.completeness import MC_MAIN, bin_magnitudes
from src.evaluation import weekly_targets
from src.paper_model import PAPER_PARAMS, add_uplift_features, rate_expected
from src.series import event_counts, on_weekday, station_daily, trailing_rate

# Stations within ~3.5 km of the seismic centre with long records (notebook 02, plot 6);
# ACAE is left out because it stops in Nov 2021. Far stations (9-10 km) barely move.
NEAR_STATIONS = ["RITE", "SOLO", "PIS1", "STRZ", "ARFE", "IPPO", "ASTR"]
FAR_STATIONS = ["LICO", "FRUL", "NAMM", "MAFE"]
# Each near station's uplift relative to RITE is estimated on 2017-2019 only, before the
# test period (2020-2025), so that test-week features never use later data.
SCALE_PERIOD = ("2017-01-01", "2019-12-31")
WEEKS_SINCE_CAP = 260  # "weeks since the last M >= 3" is capped at 5 years

SEISMIC_FEATURES = ["log_count_1w", "log_mean_2w", "log_mean_4w", "log_mean_12w", "log_max_4w",
                    "log_count_m2_4w", "log_count_m25_4w", "weeks_since_large", "log_N"]
GPS_FEATURES = ["delta_rite", "rate_rite_7d", "rate_rite_14d", "rate_rite_28d", "rate_rite_90d",
                "accel_rite", "rate_near_28d", "rate_far_28d"]
COMBINED_FEATURES = ["log_N_x_uplift", "log_paper_mu"]
ALL_FEATURES = SEISMIC_FEATURES + GPS_FEATURES + COMBINED_FEATURES


def _seismic(weekly: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Features from earlier weeks' counts. shift(1) makes week w see weeks up to w - 1."""
    c = weekly["count"]
    m = bin_magnitudes(events["md"])
    start, end = weekly.index[0].strftime("%Y-%m-%d"), weekly.index[-1].strftime("%Y-%m-%d")
    m2 = event_counts(events[m >= 2.0 - 1e-9], "W", start=start, end=end).reindex(weekly.index, fill_value=0)
    m25 = event_counts(events[m >= 2.5 - 1e-9], "W", start=start, end=end).reindex(weekly.index, fill_value=0)

    # Weeks since the most recent earlier week with an M >= 3 event.
    week_number = pd.Series(np.arange(len(weekly)), index=weekly.index)
    last_large = week_number.where(weekly["has_large"]).ffill().shift(1)
    weeks_since = (week_number - last_large).fillna(WEEKS_SINCE_CAP).clip(upper=WEEKS_SINCE_CAP)

    return pd.DataFrame({
        "log_count_1w": np.log1p(c.shift(1)),
        "log_mean_2w": np.log1p(c.rolling(2).mean().shift(1)),
        "log_mean_4w": np.log1p(c.rolling(4).mean().shift(1)),
        "log_mean_12w": np.log1p(c.rolling(12).mean().shift(1)),
        "log_max_4w": np.log1p(c.rolling(4).max().shift(1)),
        "log_count_m2_4w": np.log1p(m2.rolling(4).sum().shift(1)),
        "log_count_m25_4w": np.log1p(m25.rolling(4).sum().shift(1)),
        "weeks_since_large": weeks_since,
    }, index=weekly.index)


def _station_rate_28d(gps: pd.DataFrame, station: str) -> pd.Series:
    """Trailing 28-day vertical rate (cm/year) of one station, on a daily grid.

    Empty if the station has no data yet (e.g. ASTR before July 2016).
    """
    if not (gps["station"] == station).any():
        return pd.Series(dtype=float)
    return trailing_rate(station_daily(gps, station) * 100, 28)


def station_scales(gps: pd.DataFrame) -> pd.Series:
    """Each near station's uplift relative to RITE's over SCALE_PERIOD (RITE = 1).

    These are fixed constants describing the shape of the uplift (one source, one pattern).
    They use 2017-2019, so features of test weeks (2020 on) never depend on later data.
    """
    def change_cm(station):
        cm = station_daily(gps, station)[SCALE_PERIOD[0]:SCALE_PERIOD[1]] * 100
        return cm.iloc[-30:].mean() - cm.iloc[:30].mean()
    rite = change_cm("RITE")
    return pd.Series({s: change_cm(s) / rite for s in NEAR_STATIONS}, name="scale")


def _gps(weekly: pd.DataFrame, gps: pd.DataFrame, scales: pd.Series) -> pd.DataFrame:
    """Uplift features sampled on the Sunday before each week (on_weekday offset -1)."""
    rite_cm = station_daily(gps, "RITE") * 100
    rite_cm = rite_cm - rite_cm["2005-01"].mean()
    rates = {days: trailing_rate(rite_cm, days) for days in (7, 14, 28, 90)}
    sunday_before = lambda daily: on_weekday(daily, weekly.index, -1)  # noqa: E731

    # Near stations: if station i moves s_i times as much as RITE, its rate is about s_i * S,
    # where S is the shared rate in "RITE units". The least-squares estimate from the stations
    # reporting that day is S = sum(s_i * rate_i) / sum(s_i^2): stations that move little get
    # little weight, so their noise isn't amplified.
    near_rates = pd.concat({s: _station_rate_28d(gps, s) for s in NEAR_STATIONS}, axis=1)
    weights = near_rates.notna() * scales[near_rates.columns]
    near = (near_rates * scales[near_rates.columns]).sum(axis=1, min_count=1) / (weights * scales[near_rates.columns]).sum(axis=1)
    far = pd.concat([_station_rate_28d(gps, s) for s in FAR_STATIONS], axis=1).mean(axis=1)

    rate28 = sunday_before(rates[28])
    return pd.DataFrame({
        "delta_rite": sunday_before(rite_cm.rolling(7, min_periods=3).mean()),
        "rate_rite_7d": sunday_before(rates[7]),
        "rate_rite_14d": sunday_before(rates[14]),
        "rate_rite_28d": rate28,
        "rate_rite_90d": sunday_before(rates[90]),
        "accel_rite": rate28 - rate28.shift(4),  # change in 28-day rate over 4 weeks
        "rate_near_28d": sunday_before(near),
        "rate_far_28d": sunday_before(far),
    }, index=weekly.index)


def build_features(events: pd.DataFrame, gps: pd.DataFrame, start: str, end: str,
                   mc: float = MC_MAIN, scales: pd.Series | None = None) -> pd.DataFrame:
    """Weekly table with targets (count, count_large, has_large) and all features.

    `scales` (from station_scales) defaults to those estimated from `gps`.
    """
    scales = scales if scales is not None else station_scales(gps)
    weekly = add_uplift_features(weekly_targets(events, start, end, mc), gps)
    table = pd.concat([weekly, _seismic(weekly, events), _gps(weekly, gps, scales)], axis=1)
    table["log_N"] = np.log1p(table["N_before"])

    uplift = table["uplift_prev"].clip(lower=0)  # cm/week, the last 28 days' rate
    table["log_N_x_uplift"] = np.log1p(table["N_before"] * uplift)
    paper_mu = [rate_expected(row, PAPER_PARAMS["beta"], PAPER_PARAMS["alpha"], "uplift_prev")
                for _, row in table.iterrows()]
    table["log_paper_mu"] = np.log1p(paper_mu)
    return table


def check_no_lookahead(events: pd.DataFrame, gps: pd.DataFrame, start: str, end: str,
                       weeks: list[str], full: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recompute features with all data from week w onwards deleted; compare week w's row.

    If a feature used data from its own week or later, the two values would differ. Returns,
    for each checked week, the largest absolute difference over all features (should be 0).
    The station scales are the fixed 2017-2019 constants in both cases (see station_scales).
    """
    scales = station_scales(gps)
    full = full if full is not None else build_features(events, gps, start, end, scales=scales)
    rows = {}
    for week in weeks:
        cut = pd.Timestamp(week, tz="UTC")
        truncated = build_features(events[events["time"] < cut], gps[gps["date"] < cut],
                                   start, week, scales=scales)
        diff = (full.loc[cut, ALL_FEATURES] - truncated.loc[cut, ALL_FEATURES]).abs()
        rows[week] = {"max_abs_difference": float(diff.max()), "features_compared": int(diff.notna().sum())}
    return pd.DataFrame(rows).T
