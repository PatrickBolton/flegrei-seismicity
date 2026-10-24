"""Named forecasting experiments with cached results (phase 7).

A *setting* fixes how models are trained and what they forecast: training start, sliding
window, horizon in weeks, and the magnitude threshold for "larger events". forecasters()
returns, for a setting, the count and probability forecasters compared in phase 7 (with and
without GPS). run() walks each one forward over the test weeks and caches the result in
data/processed/forecasts/ (git-ignored), keyed by the setting, the forecaster and a
fingerprint of the feature table: a changed table means a new cache file. After changing
model code, delete that folder to recompute everything.
"""

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import walk_forward
from src.features import ALL_FEATURES, SEISMIC_FEATURES
from src.models import (make_classifier_forecaster, make_glm_forecaster,
                        make_lgbm_forecaster, share_large)
from src.paper_model import PAPER_PARAMS, make_rate_forecaster

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "forecasts"


@dataclass(frozen=True)
class Setting:
    """How models are trained and what they forecast."""
    name: str = "main"
    train_start: str = "2013-01-07"
    count_window: int | None = None  # count models: None = expanding window
    prob_window: int | None = 156  # probability models: sliding 3-year window (phase 6, 4b)
    horizon: int = 1  # weeks
    threshold: str = ""  # "" = M >= 3; "25" = M >= 2.5

    @property
    def count_target(self) -> str:
        return "count" if self.horizon == 1 else f"count_h{self.horizon}"

    @property
    def prob_target(self) -> str:
        base = f"has_large{self.threshold}"
        return base if self.horizon == 1 else f"{base}_h{self.horizon}"

    @property
    def large_column(self) -> str:
        return f"count_large{self.threshold}"


MAIN = Setting()
SENSITIVITY = [
    replace(MAIN, name="start_2005", train_start="2005-01-03"),
    replace(MAIN, name="start_2018", train_start="2018-01-01"),
    replace(MAIN, name="count_window_156", count_window=156),
    replace(MAIN, name="prob_window_expanding", prob_window=None),
    replace(MAIN, name="horizon_4", horizon=4),
    replace(MAIN, name="threshold_2.5", threshold="25"),
]
GPS_CORRECTED = "rate_rite_minus_far_28d"  # the noise-corrected GPS rate (phase 6, 4b)


# ---- Baselines that follow the setting's horizon and threshold ----

def _complete(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """History without the last horizon - 1 weeks, whose multi-week targets aren't known yet."""
    return history.iloc[:-(horizon - 1)] if horizon > 1 else history


def make_rolling_mean(weeks: int, horizon: int = 1):
    """Expected events over the horizon: horizon x the mean weekly count of the last `weeks`."""
    def rolling_mean(history, current=None):
        return float(horizon * history["count"].iloc[-weeks:].mean())
    rolling_mean.__name__ = f"rolling_mean_{weeks}w"
    return rolling_mean


def make_climatology(weeks: int, horizon: int, target: str):
    """Share of the last `weeks` complete periods in which the event happened."""
    def climatology(history, current=None):
        return float(_complete(history, horizon)[target].iloc[-weeks:].mean())
    climatology.__name__ = f"climatology_{weeks}w"
    return climatology


def make_bar(horizon: int, large_column: str):
    """The phase 5 bar: P = 1 - exp(-expected count x share of events reaching the threshold)."""
    mean_4w = make_rolling_mean(4, horizon)

    def bar(history, current=None):
        return float(1 - np.exp(-mean_4w(history) * share_large(history, column=large_column)))
    bar.__name__ = "bar"
    return bar


def make_paper_rate(horizon: int):
    """The paper's weekly rate model (forecast uplift), times the horizon."""
    weekly = make_rate_forecaster("uplift_prev", PAPER_PARAMS)

    def paper_rate(history, current):
        return horizon * weekly(history, current)
    paper_rate.__name__ = "paper_rate"
    return paper_rate


# ---- The forecasters compared in phase 7 ----

def forecasters(s: Setting, include_main_only: bool = True) -> dict[str, dict]:
    """Factories (functions returning a fresh forecaster) for a setting, by kind.

    Each forecaster gets a fresh instance, because the models remember their yearly tuning.
    """
    glm = lambda feats, name: (lambda: make_glm_forecaster(  # noqa: E731
        feats, name, train_start=s.train_start, window_weeks=s.count_window,
        horizon=s.horizon, target=s.count_target))
    recal = lambda feats, name: (lambda: make_classifier_forecaster(  # noqa: E731
        feats, name, "logistic", window_weeks=s.prob_window, C=1.0,
        fallback=make_bar(s.horizon, s.large_column), train_start=s.train_start,
        horizon=s.horizon, target=s.prob_target))
    counts = {
        "rolling_mean_4w": lambda: make_rolling_mean(4, s.horizon),
        "glm_seismic": glm(SEISMIC_FEATURES, "glm_seismic"),
        "glm_all": glm(ALL_FEATURES, "glm_all"),
    }
    probs = {
        "climatology_52w": lambda: make_climatology(52, s.horizon, s.prob_target),
        "bar": lambda: make_bar(s.horizon, s.large_column),
        "recalibrated": recal(["log_mean_4w"], "recalibrated"),
        "recalibrated_gps": recal(["log_mean_4w", GPS_CORRECTED], "recalibrated_gps"),
    }
    if include_main_only:
        counts["paper_rate"] = lambda: make_paper_rate(s.horizon)
        counts["lgbm_seismic"] = lambda: make_lgbm_forecaster(SEISMIC_FEATURES, "lgbm_seismic")
    return {"counts": counts, "probs": probs}


def _fingerprint(table: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()).hexdigest()[:10]


def run(table: pd.DataFrame, s: Setting, start: str, end: str,
        include_main_only: bool = True) -> dict[str, pd.DataFrame]:
    """Walk every forecaster of a setting forward over [start, end], with caching.

    Returns {"counts": DataFrame, "probs": DataFrame}, one column per forecaster.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(table)
    out = {}
    for kind, factories in forecasters(s, include_main_only).items():
        columns = {}
        for name, factory in factories.items():
            path = CACHE_DIR / f"{s.name}__{kind}__{name}__{start}__{end}__{fp}.parquet"
            if path.exists():
                columns[name] = pd.read_parquet(path)[name]
            else:
                columns[name] = walk_forward(table, factory(), start, end).rename(name)
                columns[name].to_frame().to_parquet(path)
        out[kind] = pd.DataFrame(columns)
    return out
