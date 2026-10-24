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
TARGET_COLUMNS = ["count", "count_large", "has_large"]  # hidden from the week being forecast
# Any column starting with these is a target (e.g. count_h4, has_large25) and is hidden too.
TARGET_PREFIXES = ("count", "has_large")


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


def add_horizon_targets(weekly: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Targets over `weeks` weeks starting at each week: count_h{weeks}, count_large_h{weeks},
    has_large_h{weeks}. Missing where the table ends before the period does."""
    out = weekly.copy()
    ahead = lambda s: s.rolling(weeks).sum().shift(-(weeks - 1))  # noqa: E731
    out[f"count_h{weeks}"] = ahead(out["count"])
    out[f"count_large_h{weeks}"] = ahead(out["count_large"])
    out[f"has_large_h{weeks}"] = (out[f"count_large_h{weeks}"] > 0).astype(float).where(
        out[f"count_large_h{weeks}"].notna())
    return out


def add_threshold_targets(weekly: pd.DataFrame, events: pd.DataFrame, magnitude: float,
                          suffix: str) -> pd.DataFrame:
    """Targets for another "larger event" threshold: count_large{suffix}, has_large{suffix}."""
    m = bin_magnitudes(events["md"])
    start, end = weekly.index[0].strftime("%Y-%m-%d"), weekly.index[-1].strftime("%Y-%m-%d")
    counts = event_counts(events[m >= magnitude - 1e-9], "W", start=start, end=end)
    out = weekly.copy()
    out[f"count_large{suffix}"] = counts.reindex(out.index, fill_value=0)
    out[f"has_large{suffix}"] = out[f"count_large{suffix}"] > 0
    return out


def walk_forward(weekly: pd.DataFrame, forecast: Callable[[pd.DataFrame, pd.Series], float],
                 test_start: str, test_end: str) -> pd.Series:
    """Forecast each test week from the weeks before it only.

    `forecast(history, current)` receives the history (all rows of `weekly` before the test
    week) and `current`, the test week's own row with the target columns removed: only
    feature columns that were known when the forecast is made (or, for "oracle" features,
    deliberately marked as such). It returns one number: an expected count or a probability.
    """
    test_weeks = weekly.loc[test_start:test_end].index
    targets = [c for c in weekly.columns if c.startswith(TARGET_PREFIXES)]
    values = [forecast(weekly.loc[weekly.index < week], weekly.loc[week].drop(targets))
              for week in test_weeks]
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


def weekly_poisson_deviance(y: pd.Series, mean: pd.Series) -> pd.Series:
    """Poisson deviance of each week separately (same floor as poisson_deviance)."""
    yv = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mean, dtype=float), LAMBDA_FLOOR)
    y_log = np.where(yv > 0, yv * np.log(np.where(yv > 0, yv, 1) / mu), 0.0)
    return pd.Series(2 * (y_log - (yv - mu)), index=y.index)


def paired_difference(per_week_a: pd.Series, per_week_b: pd.Series, block_weeks: int = 8,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of (score A - score B) over weeks, with a 95% block-bootstrap interval.

    Both forecasters are scored on the same weeks, so we look at their per-week difference.
    Neighbouring weeks are correlated (swarms last several weeks), so we resample blocks of
    `block_weeks` consecutive weeks rather than single weeks. If the interval excludes 0,
    the difference is unlikely to be luck. Negative means A has the lower (better) score.
    """
    d = (per_week_a - per_week_b).to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_weeks + 1, size=(n_boot, int(np.ceil(n / block_weeks))))
    idx = (starts[:, :, None] + np.arange(block_weeks)).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_difference": float(d.mean()), "ci_low": float(low), "ci_high": float(high)}


def brier_decomposition(happened: pd.Series, p: pd.Series, bins: int = 5) -> dict:
    """Split the Brier score into reliability - resolution + uncertainty (Murphy, 1973).

    Weeks are grouped into `bins` groups of similar forecast (by quantile). Reliability: how
    far each group's mean forecast is from how often the event happened in it (lower is
    better: probabilities are right). Resolution: how far each group's frequency is from the
    overall frequency (higher is better: the forecast separates riskier weeks from safer
    ones). Uncertainty: base rate * (1 - base rate), the same for every forecaster. With
    binned forecasts the identity is approximate.
    """
    df = pd.DataFrame({"p": np.asarray(p, dtype=float), "y": np.asarray(happened, dtype=float)})
    df["group"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("group")
    size, base = g.size(), df["y"].mean()
    return {"reliability": float((size * (g["p"].mean() - g["y"].mean()) ** 2).sum() / len(df)),
            "resolution": float((size * (g["y"].mean() - base) ** 2).sum() / len(df)),
            "uncertainty": float(base * (1 - base))}


def ranking_scores(happened: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """ROC AUC and the Brier decomposition for each probability forecaster.

    AUC: the probability that a randomly chosen week with the event got a higher forecast
    than a randomly chosen week without it (0.5 = no better than random ranking, 1 = perfect).
    It ignores whether the probabilities themselves are right.
    """
    from sklearn.metrics import roc_auc_score
    return pd.DataFrame({
        name: {"auc": roc_auc_score(np.asarray(happened, dtype=int), np.asarray(p, dtype=float)),
               **brier_decomposition(happened, p)}
        for name, p in forecasts.items()
    }).T


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


# ---- Full predictive distributions for counts (phase 7) ----
#
# A count forecast so far is one number, the expected count mu. The negative binomial (NB)
# turns it into a whole distribution with variance mu + alpha * mu^2: alpha = 0 is the Poisson
# distribution, larger alpha means more spread, as swarms produce. alpha is fitted on how
# much past counts scattered around the model's own past forecasts.

ALPHA_BOUNDS = (1e-4, 20.0)


def _nb(mu, alpha):
    """scipy's negative binomial with mean mu and variance mu + alpha * mu^2."""
    from scipy.stats import nbinom
    r = 1.0 / np.asarray(alpha, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), LAMBDA_FLOOR)
    return nbinom(r, r / (r + mu))


def fit_dispersion(y, mu) -> float:
    """Maximum-likelihood alpha for counts y around forecasts mu."""
    from scipy.optimize import minimize_scalar
    y = np.asarray(y, dtype=float)
    result = minimize_scalar(lambda log_a: -_nb(mu, np.exp(log_a)).logpmf(y).sum(),
                             bounds=np.log(ALPHA_BOUNDS), method="bounded")
    return float(np.exp(result.x))


def rolling_dispersion(y: pd.Series, mu: pd.Series, test_start: str, weeks: int = 104) -> pd.Series:
    """alpha for each week from `test_start`, fitted on the `weeks` weeks before it only."""
    out = {}
    for week in y.loc[test_start:].index:
        past = y.index < week
        out[week] = fit_dispersion(y[past].iloc[-weeks:], mu[past].iloc[-weeks:])
    return pd.Series(out, name="alpha")


def distribution_scores(y: pd.Series, mu: pd.Series, alpha, seed: int = 0) -> pd.DataFrame:
    """Per-week scores of NB forecasts (alpha = 0 or tiny for Poisson):

    - log_score: -log P(observed count); lower is better.
    - rps: ranked probability score, sum over k of (P(count <= k) - [observed <= k])^2: the
      Brier score summed over all thresholds; lower is better.
    - pit: where the observed count falls in the forecast distribution (randomised within
      the count's probability step). For calibrated forecasts it is uniform on [0, 1].
    - in_80, in_95: whether the count lies in the central 80% / 95% prediction interval.
    """
    yv = np.asarray(y, dtype=float)
    alpha = np.broadcast_to(np.maximum(np.asarray(alpha, dtype=float), ALPHA_BOUNDS[0]), yv.shape)
    dist = _nb(mu, alpha)
    rng = np.random.default_rng(seed)
    upper = int(max(yv.max(), np.max(dist.ppf(0.9999)))) + 1
    k = np.arange(upper + 1)[:, None]
    cdf = dist.cdf(k)  # (k, weeks)
    rps = ((cdf - (yv[None, :] <= k)) ** 2).sum(axis=0)
    below = np.where(yv > 0, dist.cdf(yv - 1), 0.0)
    pit = below + rng.uniform(size=len(yv)) * dist.pmf(yv)
    return pd.DataFrame({
        "log_score": -dist.logpmf(yv), "rps": rps, "pit": pit,
        "in_80": (yv >= dist.ppf(0.10)) & (yv <= dist.ppf(0.90)),
        "in_95": (yv >= dist.ppf(0.025)) & (yv <= dist.ppf(0.975)),
    }, index=y.index)


def nb_probability_at_least_one(mu, share, alpha):
    """P(at least one larger event) when the count is NB(mu, alpha) and each event is larger
    with probability `share` independently: the larger events are then NB(mu * share, alpha)
    (thinning keeps alpha), so P = 1 - (1 + alpha * mu * share)^(-1 / alpha)."""
    lam = np.maximum(np.asarray(mu, dtype=float), LAMBDA_FLOOR) * np.asarray(share, dtype=float)
    a = np.maximum(np.asarray(alpha, dtype=float), ALPHA_BOUNDS[0])
    return 1 - (1 + a * lam) ** (-1 / a)


# ---- Probability forecasts: complete scoring and recalibration (phase 7) ----

def recalibrate(p: pd.Series, happened: pd.Series, start: str, window: int = 156,
                min_positive: int = 10) -> pd.Series:
    """Walk-forward Platt recalibration of a probability forecast.

    For each week from `start`, fit P(event) = sigmoid(a + b * logit(p)) on the previous
    `window` weeks' (forecast, outcome) pairs, and apply it to that week's forecast. Until
    the window holds `min_positive` weeks with the event, the raw forecast is kept.
    """
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(p.astype(float), P_CLIP, 1 - P_CLIP)
    logit = np.log(pc / (1 - pc))
    out = {}
    for week in p.loc[start:].index:
        past = p.index < week
        x, y = logit[past].iloc[-window:], happened[past].iloc[-window:].astype(int)
        if y.sum() < min_positive or y.nunique() < 2:
            out[week] = float(pc[week])
            continue
        model = LogisticRegression(C=1e6, max_iter=1000).fit(x.to_frame(), y)
        out[week] = float(model.predict_proba(pd.DataFrame({x.name: [logit[week]]}))[0, 1])
    return pd.Series(out, name=p.name)


def probability_report(happened: pd.Series, forecasts: dict[str, pd.Series],
                       reference: str) -> pd.DataFrame:
    """Brier, log-loss, Brier skill score against `reference`, ROC AUC, precision-recall AUC,
    and the Brier decomposition, for each forecaster.

    Brier skill score: 1 - Brier / Brier(reference); 0 = no better than the reference, 1 =
    perfect. Precision-recall AUC (average precision) suits rare events: its no-skill value
    is the base rate, not 0.5.
    """
    from sklearn.metrics import average_precision_score
    y = np.asarray(happened, dtype=int)
    ref = brier_score(happened, forecasts[reference])
    rows = {}
    for name, p in forecasts.items():
        b = brier_score(happened, p)
        rows[name] = {"brier": b, "log_loss": log_loss(happened, p), "brier_skill": 1 - b / ref,
                      **ranking_scores(happened, {name: p}).loc[name].to_dict(),
                      "pr_auc": average_precision_score(y, np.asarray(p, dtype=float))}
    table = pd.DataFrame(rows).T
    return table[["brier", "brier_skill", "log_loss", "auc", "pr_auc", "reliability", "resolution", "uncertainty"]]


def detectable_difference(per_week_a: pd.Series, per_week_b: pd.Series, block_weeks: int = 8,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """paired_difference plus what the comparison could detect.

    - se: standard error of the mean difference, from the same block bootstrap.
    - mde_80: the minimum detectable effect, 2.8 x se: a true difference this large would be
      found significant (5% level, two-sided) about 80% of the time (1.96 + 0.84 = 2.8).
    - best_case_gain: -ci_low when A could be better (ci_low < 0), else 0: the largest
      improvement by A that the data still allow, at the 95% level.
    """
    d = (per_week_a - per_week_b).to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_weeks + 1, size=(n_boot, int(np.ceil(n / block_weeks))))
    idx = (starts[:, :, None] + np.arange(block_weeks)).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    se = float(means.std(ddof=1))
    return {"mean_difference": float(d.mean()), "ci_low": float(low), "ci_high": float(high),
            "se": se, "mde_80": 2.8 * se, "best_case_gain": float(max(-low, 0.0))}
