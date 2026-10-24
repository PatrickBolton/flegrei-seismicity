"""Machine-learning forecasters for src.evaluation.walk_forward.

Each make_* function returns a forecaster(history, current) -> expected count (or
probability). Models are refitted before every test week on the history only, from
TRAIN_START. Hyperparameters are re-tuned once a year, before the first test week of each
calendar year, by a grid search over time-ordered folds (see tune()); the best setting is
used for the whole year. Test weeks are never used for tuning.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import poisson_deviance

TRAIN_START = "2013-01-07"
TUNE_FOLDS = 3  # validation years per tuning
TUNE_WEEKS = 52  # weeks per validation fold
GLM_ALPHAS = (1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # strength of the L2 penalty

# Our own Poisson deviance (with its 0.1 floor), so tuning and testing use the same score.
# greater_is_better=False makes scikit-learn maximise its negative, i.e. minimise it.
POISSON_SCORER = make_scorer(poisson_deviance, greater_is_better=False)


def _training_rows(history: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """History from TRAIN_START, without weeks where any feature is missing."""
    return history.loc[TRAIN_START:].dropna(subset=features)


def _fill_missing(current: pd.Series, features: list[str], train: pd.DataFrame) -> pd.DataFrame:
    """The current week's features as one row; a missing value takes the training median."""
    row = current[features].astype(float)
    return row.fillna(train[features].median()).to_frame().T


def tune(estimator, param_grid: dict, train: pd.DataFrame, features: list[str]) -> GridSearchCV:
    """Grid search over time-ordered folds; returns the fitted search (best_params_, cv_results_).

    TimeSeriesSplit makes TUNE_FOLDS folds from the end of `train`: each validation fold is
    TUNE_WEEKS consecutive weeks, and its model is trained only on the weeks before it. For
    tuning at the start of 2023, the folds validate on roughly 2020, 2021 and 2022. (The
    default k-fold would mix past and future: never use it for time series.) Each candidate's
    score is its mean Poisson deviance over the folds. refit=False: the forecaster refits
    weekly anyway.
    """
    search = GridSearchCV(estimator, param_grid, scoring=POISSON_SCORER, refit=False, n_jobs=-1,
                          cv=TimeSeriesSplit(n_splits=TUNE_FOLDS, test_size=TUNE_WEEKS))
    return search.fit(train[features], train["count"])


def fit_glm(train: pd.DataFrame, features: list[str], alpha: float):
    """Poisson regression on standardised features (log link, L2 penalty `alpha`)."""
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))
    return model.fit(train[features], train["count"])


def make_glm_forecaster(features: list[str], name: str, alphas=GLM_ALPHAS):
    """Poisson GLM forecaster, refitted every week, penalty re-tuned every year.

    For reporting, forecaster.chosen holds the penalty chosen per year and
    forecaster.tuning the grid-search results per year.
    """
    chosen, tuning = {}, {}
    grid = {"poissonregressor__alpha": list(alphas)}

    def glm(history: pd.DataFrame, current: pd.Series) -> float:
        train = _training_rows(history, features)
        year = current.name.year
        if year not in chosen:
            pipeline = make_pipeline(StandardScaler(), PoissonRegressor(max_iter=1000))
            search = tune(pipeline, grid, train, features)
            chosen[year] = search.best_params_["poissonregressor__alpha"]
            tuning[year] = search.cv_results_
        model = fit_glm(train, features, chosen[year])
        return float(model.predict(_fill_missing(current, features, train))[0])

    glm.__name__ = name
    glm.chosen, glm.tuning = chosen, tuning
    return glm


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
