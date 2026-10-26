"""Machine-learning forecasters for src.evaluation.walk_forward.

Each make_* function returns a forecaster(history, current) -> expected count (or
probability). Models are refitted before every test week on the history only, from
TRAIN_START. Hyperparameters are re-tuned once a year, before the first test week of each
calendar year, by a grid search over time-ordered folds (see tune()); the best setting is
used for the whole year. Test weeks are never used for tuning.

The forecasters also take (for phase 7's sensitivity analyses) a different training start,
a sliding window (`window_weeks`), a longer `horizon` in weeks, and the `target` column to
forecast. With horizon h, a training week's target covers that week and the h - 1 weeks
after it, so the last h - 1 weeks before the forecast are dropped from training: their
targets would overlap the weeks being forecast (see training_rows).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import poisson_deviance

TRAIN_START = "2013-01-07"
TUNE_FOLDS = 3  # validation years per tuning
TUNE_WEEKS = 52  # weeks per validation fold
GLM_ALPHAS = (1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # strength of the L2 penalty
SEED = 42

# LightGBM settings searched each year (36 combinations). Small trees and strong minimum
# leaf sizes, because there are only ~360-700 training weeks. A first grid (100-300 trees,
# 4-15 leaves) was nearly always tuned to its smallest values, so it was extended downwards.
LGBM_GRID = {
    "n_estimators": [50, 100, 300],  # number of trees
    "num_leaves": [2, 4, 8],  # leaves per tree: 2 = one split per tree, so no interactions
    "min_child_samples": [10, 30],  # fewest training weeks in a leaf
    "reg_lambda": [0.0, 5.0],  # L2 penalty on leaf values
}
LGBM_FIXED = {"objective": "poisson", "learning_rate": 0.05, "random_state": SEED, "n_jobs": 1,
              "verbose": -1}

# Our own Poisson deviance (with its 0.1 floor), so tuning and testing use the same score.
# greater_is_better=False makes scikit-learn maximise its negative, i.e. minimise it.
POISSON_SCORER = make_scorer(poisson_deviance, greater_is_better=False)


def training_rows(history: pd.DataFrame, features: list[str], train_start: str = TRAIN_START,
                  window_weeks: int | None = None, horizon: int = 1,
                  dropna: bool = True) -> pd.DataFrame:
    """The training weeks for a forecast made after `history`.

    From `train_start`, minus the last horizon - 1 weeks (their targets aren't complete yet),
    optionally only the last `window_weeks` of those, and without weeks with a missing
    feature (unless dropna=False).
    """
    train = history.loc[train_start:]
    if horizon > 1:
        train = train.iloc[:-(horizon - 1)]
    if window_weeks is not None:
        train = train.iloc[-window_weeks:]
    return train.dropna(subset=features) if dropna else train


def _training_rows(history: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """History from TRAIN_START, without weeks where any feature is missing."""
    return training_rows(history, features)


def _fill_missing(current: pd.Series, features: list[str], train: pd.DataFrame) -> pd.DataFrame:
    """The current week's features as one row; a missing value takes the training median."""
    row = current[features].astype(float)
    return row.fillna(train[features].median()).to_frame().T


def tune(estimator, param_grid: dict, train: pd.DataFrame, features: list[str],
         target: str = "count", horizon: int = 1) -> GridSearchCV:
    """Grid search over time-ordered folds; returns the fitted search (best_params_, cv_results_).

    TimeSeriesSplit makes TUNE_FOLDS folds from the end of `train`: each validation fold is
    TUNE_WEEKS consecutive weeks, and its model is trained only on the weeks before it. For
    tuning at the start of 2023, the folds validate on roughly 2020, 2021 and 2022. (The
    default k-fold would mix past and future: never use it for time series.) Each candidate's
    score is its mean Poisson deviance over the folds. refit=False: the forecaster refits
    weekly anyway. With horizon h, a gap of h - 1 weeks separates each fold's training and
    validation weeks, so no training target overlaps the validation weeks. With a short history
    (e.g. training from 2018), fewer folds are used, so each still has a year of training.
    """
    n_splits = min(TUNE_FOLDS, (len(train) - horizon + 1) // TUNE_WEEKS - 1)
    if n_splits >= 2:
        cv = TimeSeriesSplit(n_splits=n_splits, test_size=TUNE_WEEKS, gap=horizon - 1)
    else:  # TimeSeriesSplit needs 2+ folds: give the single split explicitly
        n = len(train)
        cv = [(np.arange(n - TUNE_WEEKS - (horizon - 1)), np.arange(n - TUNE_WEEKS, n))]
    search = GridSearchCV(estimator, param_grid, scoring=POISSON_SCORER, refit=False, n_jobs=-1, cv=cv)
    return search.fit(train[features], train[target])


def fit_glm(train: pd.DataFrame, features: list[str], alpha: float, target: str = "count"):
    """Poisson regression on standardised features (log link, L2 penalty `alpha`)."""
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))
    return model.fit(train[features], train[target])


class OffsetPoissonRegressor(RegressorMixin, BaseEstimator):
    """Poisson regression with a fixed offset (Part 2): the last column of X is log(E), and the
    model is log(mean) = log(E) + b0 + b . x on the other (standardised) columns, so it learns a
    multiplicative correction to E. scikit-learn has no offset option, but the Poisson
    likelihood with offset E equals that of the rate y / E with sample weights E:
    sum(y log(E mu) - E mu) = sum(E ((y / E) log mu - mu)) + constant."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        exposure = np.exp(X[:, -1])
        self.model_ = make_pipeline(StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=1000))
        self.model_.fit(X[:, :-1], np.asarray(y, dtype=float) / exposure, poissonregressor__sample_weight=exposure)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.exp(X[:, -1]) * self.model_.predict(X[:, :-1])


def make_glm_forecaster(features: list[str], name: str, alphas=GLM_ALPHAS,
                        train_start: str = TRAIN_START, window_weeks: int | None = None,
                        horizon: int = 1, target: str = "count", offset: str | None = None):
    """Poisson GLM forecaster, refitted every week, penalty re-tuned every year.

    offset: a column holding log(E) for a fixed offset (OffsetPoissonRegressor), e.g. the log
    of a physics model's forecast; the GLM then learns only a correction to E.
    For reporting, forecaster.chosen holds the penalty chosen per year and
    forecaster.tuning the grid-search results per year.
    """
    chosen, tuning = {}, {}
    columns = features + [offset] if offset else features
    grid = {"alpha" if offset else "poissonregressor__alpha": list(alphas)}

    def new_model(alpha=1.0):
        if offset:
            return OffsetPoissonRegressor(alpha=alpha)
        return make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))

    def glm(history: pd.DataFrame, current: pd.Series) -> float:
        train = training_rows(history, columns, train_start, window_weeks, horizon)
        year = current.name.year
        if year not in chosen:
            search = tune(new_model(), grid, train, columns, target, horizon)
            chosen[year] = next(iter(search.best_params_.values()))
            tuning[year] = search.cv_results_
        model = new_model(chosen[year]).fit(train[columns], train[target])
        return float(model.predict(_fill_missing(current, columns, train))[0])

    glm.__name__ = name
    glm.chosen, glm.tuning = chosen, tuning
    return glm


def make_lgbm_forecaster(features: list[str], name: str, grid: dict = LGBM_GRID):
    """LightGBM forecaster with a Poisson objective, refitted every week, tuned every year.

    Unlike the GLM, LightGBM handles missing feature values itself (a missing value is sent
    down whichever branch fitted best in training), so weeks with GPS gaps are kept and
    nothing is filled in. No standardisation is needed: trees only use the order of values.
    forecaster.chosen holds the settings chosen per year, forecaster.tuning the searches.
    """
    chosen, tuning = {}, {}

    def lgbm(history: pd.DataFrame, current: pd.Series) -> float:
        train = history.loc[TRAIN_START:]
        year = current.name.year
        if year not in chosen:
            search = tune(LGBMRegressor(**LGBM_FIXED), grid, train, features)
            chosen[year] = search.best_params_
            tuning[year] = search.cv_results_
        model = LGBMRegressor(**LGBM_FIXED, **chosen[year]).fit(train[features], train["count"])
        return float(model.predict(current[features].astype(float).to_frame().T)[0])

    lgbm.__name__ = name
    lgbm.chosen, lgbm.tuning = chosen, tuning
    return lgbm


def fit_lgbm(train: pd.DataFrame, features: list[str], params: dict) -> LGBMRegressor:
    """A LightGBM model with the given settings, fitted on `train` (for inspection, e.g. SHAP)."""
    return LGBMRegressor(**LGBM_FIXED, **params).fit(train[features], train["count"])


# ---- P(at least one M >= 3 next week) ----
#
# Weeks with an M >= 3 are rare and recent: the training data has its 1st in December 2019,
# its 10th only in September 2023. So the direct classifiers (a) start only once the training
# data holds MIN_POSITIVE_WEEKS such weeks, using the 52-week climatology before that, and
# (b) use fixed, conservative settings: yearly tuning is impossible when the validation years
# (2020-2022) contain only a handful of positive weeks.

MIN_POSITIVE_WEEKS = 10
CLIMATOLOGY_WEEKS = 52


def share_large(history: pd.DataFrame, weeks: int = 156, column: str = "count_large") -> float:
    """Fraction of events (M >= Mc) in the last `weeks` weeks that reached M >= 3 (or the
    threshold counted in `column`)."""
    past = history.iloc[-weeks:]
    return float(past[column].sum() / max(past["count"].sum(), 1))


def _make_classifier(kind: str, C: float = 0.1):
    if kind == "logistic":
        # Standardised features and an L2 penalty (default C = 0.1, fairly strong). Classes are not
        # re-weighted: that would help catch rare positive weeks but distort the probabilities,
        # and probabilities are what we score.
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    if kind == "lgbm":
        return LGBMClassifier(n_estimators=100, num_leaves=2, min_child_samples=10,
                              learning_rate=0.05, random_state=SEED, n_jobs=1, verbose=-1)
    raise ValueError(kind)


def make_classifier_forecaster(features: list[str], name: str, kind: str = "logistic",
                               window_weeks: int | None = None, C: float = 0.1, fallback=None,
                               train_start: str = TRAIN_START, horizon: int = 1,
                               target: str = "has_large"):
    """Direct forecaster of P(M >= 3 next week): logistic regression or LightGBM.

    Refitted every week, on weeks from TRAIN_START or, with `window_weeks`, on the last
    `window_weeks` weeks only (a sliding window, for a target whose relation to the features
    may change). Until the training data has MIN_POSITIVE_WEEKS weeks with an M >= 3 it uses
    `fallback(history, current)`, by default the 52-week climatology. C is the logistic
    model's inverse penalty (larger = weaker).
    forecaster.state["active_from"] is the first test week where the classifier was used.
    """
    state = {"active_from": None}

    def classifier(history: pd.DataFrame, current: pd.Series) -> float:
        # The logistic model can't take missing values; LightGBM can.
        train = training_rows(history, features, train_start, window_weeks, horizon,
                              dropna=(kind == "logistic"))
        if train[target].sum() < MIN_POSITIVE_WEEKS:
            if fallback is not None:
                return float(fallback(history, current))
            complete = history.iloc[:-(horizon - 1)] if horizon > 1 else history
            return float(complete[target].iloc[-CLIMATOLOGY_WEEKS:].mean())
        if state["active_from"] is None:
            state["active_from"] = current.name
        model = _make_classifier(kind, C).fit(train[features], train[target].astype(int))
        x = current[features].astype(float).to_frame().T
        if kind == "logistic":
            x = _fill_missing(current, features, train)
        return float(model.predict_proba(x)[0, 1])

    classifier.__name__ = name
    classifier.state = state
    return classifier


def tuning_table(search_results: dict, param: str) -> pd.DataFrame:
    """Mean Poisson deviance of each candidate in each validation fold, from cv_results_."""
    folds = [k for k in search_results if k.startswith("split") and k.endswith("_test_score")]
    table = pd.DataFrame({f"fold {i + 1}": -np.asarray(search_results[k]) for i, k in enumerate(folds)})
    table.index = pd.Index(search_results[f"param_{param}"], name=param.split("__")[-1])
    table["mean"] = table.mean(axis=1)
    return table


def glm_coefficients(train: pd.DataFrame, features: list[str], alpha: float) -> pd.Series:
    """Coefficients of a GLM fitted on `train`, per standard deviation of each feature.

    A coefficient c means: one standard deviation more of that feature multiplies the
    expected count by exp(c), other features held fixed.
    """
    model = fit_glm(train, features, alpha)
    return pd.Series(model[-1].coef_, index=features, name="coefficient")
