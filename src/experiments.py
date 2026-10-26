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
    return {kind: _run_cached(table, factories, f"{s.name}__{kind}", start, end)
            for kind, factories in forecasters(s, include_main_only).items()}


def _run_cached(table: pd.DataFrame, factories: dict, prefix: str, start: str, end: str) -> pd.DataFrame:
    """walk_forward each factory's forecaster over [start, end], caching each result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(table)
    columns = {}
    for name, factory in factories.items():
        path = CACHE_DIR / f"{prefix}__{name}__{start}__{end}__{fp}.parquet"
        if path.exists():
            columns[name] = pd.read_parquet(path)[name]
        else:
            columns[name] = walk_forward(table, factory(), start, end).rename(name)
            columns[name].to_frame().to_parquet(path)
    return pd.DataFrame(columns)


# ---- Physics baselines (Part 2): ETAS, rate-and-state, and the hybrid ----

PHYSICS_MODELS = {"etas": "physics_etas", "rs": "physics_rs", "rs_etas": "physics_rs_etas"}
PHYSICS_SIMS = 2000  # simulations per week for the expected count of ETAS and the hybrid


def run_physics(table: pd.DataFrame, events: pd.DataFrame, uplift: pd.DataFrame, start: str, end: str,
                n_sims: int = PHYSICS_SIMS) -> dict[str, pd.DataFrame]:
    """Walk the physics forecasters forward over [start, end] (one-week horizon), with caching.

    events and uplift are src.physics tables (physics_events, uplift_series). Returns
    {"counts": one column per model, "fits": parameters at every refit, "weeks": each week's
    background / triggered / no-cascade / expected count}.
    """
    from src.physics import make_physics_forecaster  # here, so Part 1 runs don't need it

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(table)[:5] + _fingerprint(events)[:5] + _fingerprint(uplift)[:5]
    counts, fits, weeks = {}, [], []
    for model, name in PHYSICS_MODELS.items():
        stem = CACHE_DIR / f"physics__{name}__{n_sims}__{start}__{end}__{fp}"
        paths = {k: stem.with_name(f"{stem.name}__{k}.parquet") for k in ["counts", "fits", "weeks"]}
        if not all(p.exists() for p in paths.values()):
            forecaster = make_physics_forecaster(model, events, uplift, n_sims=n_sims)
            walk_forward(table, forecaster, start, end).rename(name).to_frame().to_parquet(paths["counts"])
            pd.DataFrame(forecaster.log["fits"]).to_parquet(paths["fits"])
            pd.DataFrame(forecaster.log["weeks"]).to_parquet(paths["weeks"])
        counts[name] = pd.read_parquet(paths["counts"])[name]
        fits.append(pd.read_parquet(paths["fits"]).assign(model=model))
        weeks.append(pd.read_parquet(paths["weeks"]).assign(model=model))
    return {"counts": pd.DataFrame(counts), "fits": pd.concat(fits, ignore_index=True),
            "weeks": pd.concat(weeks, ignore_index=True)}


# ---- ML with the physics forecast as an input (Part 2, milestone 5) ----

PHYSICS_FEATURES = {"etas": "log_physics_etas", "hybrid": "log_physics_rs_etas"}


def add_physics_features(table: pd.DataFrame, physics_counts: pd.DataFrame) -> pd.DataFrame:
    """The table plus log(physics forecast) per week (floored at LAMBDA_FLOOR, like the scores).
    Each value is the walk-forward forecast made before its week, so it is a legitimate
    feature of that week; weeks without a physics forecast are left missing."""
    from src.evaluation import LAMBDA_FLOOR

    out = table.copy()
    for column, name in [(PHYSICS_FEATURES["etas"], "physics_etas"), (PHYSICS_FEATURES["hybrid"], "physics_rs_etas")]:
        out[column] = np.log(np.maximum(physics_counts[name], LAMBDA_FLOOR)).reindex(out.index)
    return out


def ml_physics_forecasters() -> dict:
    """The Part 1 GLM and LightGBM (seismic features) with a physics forecast added, either as
    one more feature or (GLM) as a fixed offset that the GLM corrects. Main setting only."""
    factories = {}
    for short, column in PHYSICS_FEATURES.items():
        factories[f"glm_seismic+{short}"] = (
            lambda c=column, n=f"glm_seismic+{short}": make_glm_forecaster(SEISMIC_FEATURES + [c], n))
        factories[f"glm_offset_{short}"] = (
            lambda c=column, n=f"glm_offset_{short}": make_glm_forecaster(SEISMIC_FEATURES, n, offset=c))
        factories[f"lgbm_seismic+{short}"] = (
            lambda c=column, n=f"lgbm_seismic+{short}": make_lgbm_forecaster(SEISMIC_FEATURES + [c], n))
    return factories


def run_ml_physics(table: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Walk the ml_physics_forecasters forward over [start, end], with caching. `table` must
    contain the physics feature columns (add_physics_features)."""
    return _run_cached(table, ml_physics_forecasters(), "mlphysics__counts", start, end)
