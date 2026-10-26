"""Physics baselines: ETAS, rate-and-state (RS), and the RS + ETAS hybrid of Hainzl et al.

Hainzl, Dahm & Tramelli (2026, Commun. Earth Environ. 7, 244) model the seismicity at Campi
Flegrei as ETAS (earthquakes triggering earthquakes) whose background rate comes from a
rate-and-state model driven by the uplift:

    lambda(t) = mu(t) + sum over earlier events i of K * 10**(alpha * (m_i - Mcut)) * (c + t - t_i)**(-p)

with the sum limited to events in the last TMAX_TRIGGER days, and

    mu(t) = r * exp(S / Asig) / (1 + integral of exp(S / Asig) dt / ta)          (RS)
    S(t)  = uplift(t) + (Asig / ta) * (t - t0)        (uplift in m, plus background loading)

integrated from the start of the uplift series (1905). Time is in days (MJD), as in their code.

Adapted from their code (Hainzl, S. 2026, Zenodo, doi:10.5281/zenodo.18413119, CC BY 4.0),
`parameter_estimation.py`. Changes: the RS rate is computed in log space, the triggering sums
use precomputed event pairs instead of a loop, and fits use bounds and several starting points.
Their discretisation is kept, so the log-likelihoods match theirs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MCUT = 0.45  # their magnitude cut: Md >= 0.5 at 0.1 resolution
TMAX_TRIGGER = 100.0  # days: aftershocks are only triggered for this long (their TmaxTrig)
FIT_START_YEAR = 2005.0


# ---- Their files (data/raw/hainzl/) ----

def load_uplift(folder: Path) -> pd.DataFrame:
    """Their weekly uplift series since 1905: columns mjd, year (decimal), uplift (m)."""
    return pd.read_csv(folder / "gps.dat", sep=r"\s+", comment="#", names=["mjd", "year", "uplift"])


def load_catalogue(folder: Path, mcut: float = MCUT) -> pd.DataFrame:
    """Their catalogue, events with m >= mcut: columns mjd, year (decimal), m."""
    cat = pd.read_csv(folder / "campi-flegrei.kat", sep=r"\s+", comment="#",
                      names=["id", "mjd", "year", "lat", "lon", "depth_km", "m"])
    return cat.loc[cat["m"] >= mcut, ["mjd", "year", "m"]].reset_index(drop=True)


def year_to_mjd(year: float, uplift: pd.DataFrame) -> float:
    """Decimal year to MJD with the linear relation in the uplift file (as their code does)."""
    slope, intercept = np.polyfit(uplift["mjd"], uplift["year"], 1)
    return (year - intercept) / slope


# ---- Rate-and-state background ----

def log_rs_rate(t, uplift, r, asig, ta):
    """log of the RS seismicity rate (events/day) at the times t of the uplift series.

    Heimisson & Segall (2018), as in Hainzl et al.: the integral is their right-hand sum on the
    grid t. Everything stays in log space, so exp(S / asig) never overflows.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(uplift, dtype=float) + (asig / ta) * (t - t[0])
    dt = np.diff(t, prepend=2 * t[0] - t[1])  # first step = second step, as in their code
    log_k = s / asig
    log_integral = np.logaddexp.accumulate(log_k + np.log(dt))
    return np.log(r) + log_k - np.logaddexp(0.0, log_integral - np.log(ta))


# ---- Likelihood of ETAS, RS and the hybrid ----

MODELS = {  # parameters of each model
    "etas": ["mu", "K", "alpha", "c", "p"],
    "rs": ["r", "asig", "ta"],
    "rs_etas": ["r", "asig", "ta", "K", "alpha", "c", "p"],
}


def prepare(events: pd.DataFrame, uplift: pd.DataFrame, t1: float, t2: float,
            tmax: float = TMAX_TRIGGER) -> dict:
    """Everything the likelihood needs for a fit window [t1, t2] (days), computed once.

    `events` are all events up to t2, including those before t1: they can still trigger events
    in the window. `uplift` is the whole series (RS integrates over its full history).
    """
    events = events[events["mjd"] <= t2].sort_values("mjd")
    t, m = events["mjd"].to_numpy(), events["m"].to_numpy()
    target = np.flatnonzero((t >= t1) & (t <= t2))  # events whose rate enters the likelihood
    # Pairs (i, j): event j triggers event i if t_i - tmax <= t_j < t_i (strictly earlier).
    lo = np.searchsorted(t, t[target] - tmax, side="left")
    hi = np.searchsorted(t, t[target], side="left")
    n_pairs = hi - lo
    pair_i = np.repeat(np.arange(len(target)), n_pairs)  # index into target
    pair_j = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(target) else np.array([], int)
    grid = uplift[uplift["mjd"] <= t2]
    return {
        "t": t, "m": m, "t1": t1, "t2": t2, "tmax": tmax,
        "target": target, "pair_i": pair_i, "pair_j": pair_j,
        "lag": t[target][pair_i] - t[pair_j],
        "contributes": (t > t1 - tmax) & (t <= t2),  # events whose aftershocks fall in the window
        "grid_t": grid["mjd"].to_numpy(), "grid_uplift": grid["uplift"].to_numpy(),
        "grid_window": ((grid["mjd"] >= t1) & (grid["mjd"] <= t2)).to_numpy(),
    }


def triggering_integral(t, productivity, c, p, t1, t2, tmax=TMAX_TRIGGER):
    """Expected number of aftershocks in [t1, t2] of events at t (truncated after tmax days)."""
    start = np.clip(t1 - t, 0.0, None)
    end = np.minimum(t2 - t, tmax)
    if p == 1:
        return np.sum(productivity * (np.log(c + end) - np.log(c + start)))
    return np.sum(productivity * ((c + end) ** (1 - p) - (c + start) ** (1 - p)) / (1 - p))


def intensities(params: dict, data: dict, model: str):
    """Background and total rate (events/day) at each event in the window, and the expected
    number of events in the window (the integral of the rate)."""
    t, target = data["t"], data["target"]
    if model == "etas":
        background = np.full(len(target), params["mu"])
        expected = params["mu"] * (data["t2"] - data["t1"])
    else:
        rate = np.exp(log_rs_rate(data["grid_t"], data["grid_uplift"],
                                  params["r"], params["asig"], params["ta"]))
        background = np.interp(t[target], data["grid_t"], rate)
        w = data["grid_window"]
        expected = np.trapezoid(rate[w], data["grid_t"][w])
    total = background
    if model != "rs":
        K, alpha, c, p = (params[k] for k in ["K", "alpha", "c", "p"])
        productivity = K * 10.0 ** (alpha * (data["m"] - MCUT))
        kernel = productivity[data["pair_j"]] * (c + data["lag"]) ** (-p)
        total = background + np.bincount(data["pair_i"], weights=kernel, minlength=len(target))
        cont = data["contributes"]
        expected += triggering_integral(t[cont], productivity[cont], c, p,
                                        data["t1"], data["t2"], data["tmax"])
    return background, total, expected


def negative_log_likelihood(params: dict, data: dict, model: str) -> float:
    """-log L = expected number of events in the window - sum of log rates at the events."""
    _, total, expected = intensities(params, data, model)
    with np.errstate(divide="ignore"):  # a zero rate at an event (extreme parameters) -> 1e10
        nll = expected - np.sum(np.log(total))
    return float(nll) if np.isfinite(nll) else 1e10



def expected_counts(params: dict, data: dict, model: str, edges) -> pd.DataFrame:
    """Expected events between consecutive `edges` (days, inside the fit window), split into
    background and triggered. Triggering uses all observed events up to each interval's end, so
    this is the fitted model's account of the past (in sample), not a forecast."""
    edges = np.asarray(edges, dtype=float)
    if model == "etas":
        background = params["mu"] * np.diff(edges)
    else:
        rate = np.exp(log_rs_rate(data["grid_t"], data["grid_uplift"],
                                  params["r"], params["asig"], params["ta"]))
        cumulative = np.concatenate([[0.0], np.cumsum(np.diff(data["grid_t"]) * (rate[1:] + rate[:-1]) / 2)])
        background = np.diff(np.interp(edges, data["grid_t"], cumulative))
    triggered = np.zeros(len(edges) - 1)
    if model != "rs":
        K, alpha, c, p = (params[k] for k in ["K", "alpha", "c", "p"])
        t, m = data["t"], data["m"]
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            cont = (t > a - data["tmax"]) & (t <= b)
            triggered[k] = triggering_integral(t[cont], K * 10.0 ** (alpha * (m[cont] - MCUT)), c, p,
                                               a, b, data["tmax"])
    return pd.DataFrame({"start": edges[:-1], "end": edges[1:], "background": background, "triggered": triggered})


def branching_ratio(params: dict, b: float, tmax: float = TMAX_TRIGGER) -> float:
    """Average number of direct aftershocks per event: K * (Omori kernel integrated over tmax days)
    * E[10**(alpha (m - Mcut))], where the last factor is b / (b - alpha) for Gutenberg-Richter
    magnitudes. Near 1, sequences barely die out."""
    K, alpha, c, p = (params[k] for k in ["K", "alpha", "c", "p"])
    kernel = np.log((c + tmax) / c) if p == 1 else ((c + tmax) ** (1 - p) - c ** (1 - p)) / (1 - p)
    return K * kernel * b / (b - alpha)

# ---- Maximum-likelihood fits ----

BOUNDS = {  # physically sensible ranges; fits work with log10 of each parameter
    "mu": (1e-5, 100.0),  # background events/day
    "r": (1e-9, 100.0),  # RS rate scale, events/day
    "asig": (1e-3, 10.0),  # A sigma', m of uplift
    "ta": (10.0, 1e10),  # RS relaxation time, days
    "K": (1e-4, 5.0),  # productivity
    "alpha": (1e-3, 3.0),  # magnitude scaling of productivity
    "c": (1e-6, 1.0),  # Omori c, days
    "p": (0.5, 3.0),  # Omori exponent
}
START = {"mu": 0.1, "r": 0.01, "asig": 1.0, "ta": 1e5, "K": 0.01, "alpha": 0.9, "c": 0.01, "p": 1.1}  # theirs


def fit(data: dict, model: str, n_random: int = 4, seed: int = 42, start: dict | None = None) -> dict:
    """Maximum-likelihood fit from their starting point (or `start`) plus n_random random ones.

    Returns the best parameters, -log L, AIC, and every start's result (to see if they agree).
    """
    names = MODELS[model]
    bounds = np.log10([BOUNDS[n] for n in names])
    rng = np.random.default_rng(seed)
    starts = [np.log10([(start or START)[n] for n in names])]
    # Random starts: uniform in log10, away from the edges (the middle 60% of each range).
    width = bounds[:, 1] - bounds[:, 0]
    starts += [rng.uniform(bounds[:, 0] + 0.2 * width, bounds[:, 1] - 0.2 * width)
               for _ in range(n_random)]

    def objective(x):
        return negative_log_likelihood(dict(zip(names, 10.0 ** x)), data, model)

    runs = []
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        runs.append({"nll": res.fun, **dict(zip(names, 10.0 ** res.x)), "converged": res.success})
    runs = pd.DataFrame(runs)
    best = runs.loc[runs["nll"].idxmin()]
    params = {n: float(best[n]) for n in names}
    return {"params": params, "nll": float(best["nll"]), "aic": 2 * float(best["nll"]) + 2 * len(names),
            "runs": runs}


# ---- Simulation ----

def _omori_delays(rng, c, p, start, end):
    """Random delays from the Omori density (c + tau)**(-p), truncated to [start, end]."""
    u = rng.uniform(size=len(start))
    if p == 1:
        return (c + start) * ((c + end) / (c + start)) ** u - c
    a, b = (c + start) ** (1 - p), (c + end) ** (1 - p)
    return (a + u * (b - a)) ** (1 / (1 - p)) - c


def simulate_etas(params: dict, t_start: float, t_end: float, b: float, rng: np.random.Generator,
                  background=None, history: pd.DataFrame | None = None, mmax: float = 5.0,
                  tmax: float = TMAX_TRIGGER) -> pd.DataFrame:
    """Simulate events with m >= MCUT in (t_start, t_end], generation by generation.

    background: (grid_t, rate) with the rate (events/day) constant between grid points;
    None = the constant params["mu"]. history: earlier events (mjd, m) that can still trigger.
    Magnitudes follow Gutenberg-Richter with this b, truncated at mmax.
    """
    if background is None:
        background = (np.array([t_start, t_end]), np.array([params["mu"], 0.0]))
    grid_t, rate = background
    cells = np.clip(np.diff(np.clip(grid_t, t_start, t_end)), 0, None) * rate[:-1]
    n_background = rng.poisson(cells.sum())
    cell = rng.choice(len(cells), size=n_background, p=cells / cells.sum()) if n_background else np.array([], int)
    lo, hi = np.maximum(grid_t[cell], t_start), np.minimum(grid_t[cell + 1], t_end)
    times = rng.uniform(lo, hi)

    def magnitudes(n):
        # Inverse CDF of the exponential (Gutenberg-Richter) distribution truncated at mmax.
        beta = b * np.log(10)
        return MCUT - np.log(1 - rng.uniform(size=n) * (1 - np.exp(-beta * (mmax - MCUT)))) / beta

    out_t, out_m = [times], [magnitudes(n_background)]
    parents_t, parents_m = out_t[0], out_m[0]
    if history is not None:
        recent = history[(history["mjd"] > t_start - tmax) & (history["mjd"] <= t_start)]
        parents_t = np.concatenate([recent["mjd"].to_numpy(), parents_t])
        parents_m = np.concatenate([recent["m"].to_numpy(), parents_m])
    K, alpha, c, p = (params[k] for k in ["K", "alpha", "c", "p"])
    while len(parents_t):
        start = np.clip(t_start - parents_t, 0.0, None)  # history: only aftershocks after t_start
        end = np.minimum(t_end - parents_t, tmax)
        productivity = K * 10.0 ** (alpha * (parents_m - MCUT))
        if p == 1:
            expected = productivity * np.log((c + end) / (c + start))
        else:
            expected = productivity * ((c + end) ** (1 - p) - (c + start) ** (1 - p)) / (1 - p)
        n_children = rng.poisson(np.clip(expected, 0, None))
        idx = np.repeat(np.arange(len(parents_t)), n_children)
        parents_t = parents_t[idx] + _omori_delays(rng, c, p, start[idx], end[idx])
        parents_m = magnitudes(len(parents_t))
        out_t.append(parents_t)
        out_m.append(parents_m)
    sim = pd.DataFrame({"mjd": np.concatenate(out_t), "m": np.concatenate(out_m)})
    return sim.sort_values("mjd", ignore_index=True)
