"""Evaluation metrics (spec Sec 17, 26.1, 26.5-26.7, 26.12-26.15, 26.17).

Pure numpy / pandas / scipy functions: no database access and no config reads.
Callers pass every threshold explicitly, so each rule is visible where it is
applied (src/reports/diagnostics.py).

Conventions
  y            realized outcome of the contract (1 = resolved YES)
  delta_brier  Brier_market - Brier_model; positive = the model is better
  game_id      the resampling unit (26.1). A game contributes several rows (both
               contracts, several snapshot offsets). Bootstraps resample GAMES,
               with each game's rows kept together, so duplicating rows can
               never narrow a confidence interval.
  decisions    backtest_decisions rows (see src/storage/schemas.py for the P&L
               semantics). p_market is the YES contract's market probability;
               side-specific values are derived by `side_view`:
                 p_market_side = p_market (YES) or 1 - p_market (NO)
                 p_model_side  = quoted_price + predicted_edge   (by definition
                                 predicted_edge = p_model(side) - quoted_price)
                 realized_edge = side_won - quoted_price
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

EPS = 1e-6
INSUFFICIENT = "INSUFFICIENT SAMPLE"
NAN = float("nan")


def num(x: Any) -> np.ndarray:
    """Float array; nullable extension types (Int32, boolean, Float64) become NaN-aware floats."""
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.to_numeric(x, errors="coerce").astype("float64").to_numpy()
    return np.asarray(x, dtype=float)


def flag(x: pd.Series) -> np.ndarray:
    """Boolean array; missing values are False."""
    return x.astype("boolean").fillna(False).to_numpy(dtype=bool)


# ---------------------------------------------------------------------------
# Proper scoring rules (Sec 17, 26.1)
# ---------------------------------------------------------------------------

def brier(y: Any, p: Any) -> float:
    y, p = num(y), num(p)
    return float(np.mean((p - y) ** 2)) if y.size else NAN


def log_loss(y: Any, p: Any, eps: float = EPS) -> float:
    y, p = num(y), np.clip(num(p), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))) if y.size else NAN


def bss(brier_model: float, brier_market: float) -> float:
    """Brier skill score vs the market: 1 - Brier_model / Brier_market (26.1)."""
    if not np.isfinite(brier_market) or brier_market <= 0 or not np.isfinite(brier_model):
        return NAN
    return float(1 - brier_model / brier_market)


def wilson_interval(k: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return NAN, NAN
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def logistic_regression(X: Any, y: Any, *, groups: Any = None, max_iter: int = 100,
                        tol: float = 1e-10) -> dict[str, Any]:
    """Maximum-likelihood logistic regression by Newton/IRLS with step halving.

    X must include an intercept column if one is wanted. With `groups`, standard
    errors are cluster-robust by group (sandwich estimator; games as clusters).
    Returns NaNs (converged = False) on separation, constant y or singular X.
    """
    X, y = num(X), num(y)
    if X.ndim == 1:
        X = X[:, None]
    k = X.shape[1]
    nan_out = {"coef": np.full(k, NAN), "se": np.full(k, NAN), "z": np.full(k, NAN),
               "p": np.full(k, NAN), "converged": False, "n": int(len(y))}
    if len(y) <= k or np.all(y == y[0]):
        return nan_out

    def loglik(b: np.ndarray) -> float:
        eta = X @ b
        return float(np.sum(y * eta - np.logaddexp(0, eta)))

    beta, ll, converged = np.zeros(k), loglik(np.zeros(k)), False
    for _ in range(max_iter):
        mu = expit(X @ beta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return nan_out
        t = 1.0
        while t > 1e-6:
            cand = beta + t * step
            ll_c = loglik(cand)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12:
                break
            t /= 2
        beta, ll = cand, ll_c
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    if not converged or not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 50:
        return nan_out
    mu = expit(X @ beta)
    try:
        hinv = np.linalg.inv(X.T @ (X * np.clip(mu * (1 - mu), 1e-12, None)[:, None]))
    except np.linalg.LinAlgError:
        return nan_out
    if groups is None:
        cov = hinv
    else:
        codes, uniq = pd.factorize(pd.Series(groups).to_numpy())
        g = len(uniq)
        scores = X * (y - mu)[:, None]
        s = np.column_stack([np.bincount(codes, weights=scores[:, j], minlength=g) for j in range(k)])
        cov = hinv @ (s.T @ s) @ hinv * (g / (g - 1) if g > 1 else NAN)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return {"coef": beta, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)), "converged": True,
            "n": int(len(y))}


# ---------------------------------------------------------------------------
# Game-level bootstrap (26.1: the GAME is the unit, not the snapshot)
# ---------------------------------------------------------------------------

def _game_sums(game_ids: Any, *columns: np.ndarray) -> np.ndarray:
    """(n_games, n_columns) per-game sums."""
    codes, uniq = pd.factorize(pd.Series(game_ids).to_numpy())
    g = len(uniq)
    return np.column_stack([np.bincount(codes, weights=c, minlength=g) for c in columns]) if g else \
        np.zeros((0, len(columns)))


def _game_draws(g: int, n: int, rng: np.random.Generator, max_cells: int = 1_000_000):
    """Yield (start, idx): `n` resamples of `g` games drawn uniformly with replacement
    (Generator.integers), in chunks of rows so memory stays at ~max_cells indices."""
    step = max(1, max_cells // max(g, 1))
    for start in range(0, n, step):
        yield start, rng.integers(0, g, size=(min(step, n - start), g))


def _bootstrap_sums(sums: np.ndarray, n: int, rng: np.random.Generator, max_cells: int = 1_000_000) -> np.ndarray:
    """(n, n_columns) column sums over `n` resamples of games drawn with replacement.

    Exact resampling: each resample draws n_games game indices uniformly
    (Generator.integers) and sums the drawn games' per-game totals. Chunked so
    memory stays at ~max_cells indices regardless of n x n_games.
    """
    g, k = sums.shape
    out = np.empty((n, k))
    cols = [np.ascontiguousarray(sums[:, j]) for j in range(k)]
    for start, idx in _game_draws(g, n, rng, max_cells):
        for j in range(k):
            out[start:start + len(idx), j] = cols[j][idx].sum(axis=1)
    return out


def _ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        return NAN, NAN
    lo, hi = np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def paired_game_bootstrap(df: pd.DataFrame, *, n: int = 10_000, seed: int = 0, ci: float = 0.95,
                          y: str = "y", p_model: str = "p_model", p_market: str = "p_market",
                          game: str = "game_id") -> dict[str, Any]:
    """Paired bootstrap of delta_brier = Brier_market - Brier_model over games (26.1).

    Each resample draws games with replacement and recomputes the pooled
    per-observation Brier delta over all rows of the drawn games.
    p_value is one-sided: the share of resamples with delta <= 0.
    """
    d = df[[game, y, p_model, p_market]].dropna()
    yv, pm, pk = num(d[y]), num(d[p_model]), num(d[p_market])
    se_mod, se_mk = (pm - yv) ** 2, (pk - yv) ** 2
    sums = _game_sums(d[game], se_mk - se_mod, np.ones(len(d)), se_mod, se_mk)
    n_games = int(sums.shape[0])
    out: dict[str, Any] = {"n_games": n_games, "n_obs": int(len(d)), "n_resamples": int(n), "ci_level": ci,
                           "seed": seed, "brier_model": brier(yv, pm), "brier_market": brier(yv, pk)}
    out["delta_brier"] = out["brier_market"] - out["brier_model"]
    out["bss"] = bss(out["brier_model"], out["brier_market"])
    out["log_loss_model"], out["log_loss_market"] = log_loss(yv, pm), log_loss(yv, pk)
    out["delta_log_loss"] = out["log_loss_market"] - out["log_loss_model"]
    if n_games == 0 or n <= 0:
        return out | {"mean_delta": NAN, "ci_low": NAN, "ci_high": NAN, "p_value": NAN,
                      "bss_ci_low": NAN, "bss_ci_high": NAN}
    boot = _bootstrap_sums(sums, n, np.random.default_rng(seed))
    delta = boot[:, 0] / boot[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        bss_b = boot[:, 0] / boot[:, 3]  # 1 - model/market = (market - model) / market
    lo, hi = _ci(delta, ci)
    blo, bhi = _ci(bss_b, ci)
    return out | {"mean_delta": float(np.mean(delta)), "ci_low": lo, "ci_high": hi,
                  "p_value": float(np.mean(delta <= 0)), "bss_ci_low": blo, "bss_ci_high": bhi}


def game_bootstrap_mean_ci(values: Any, game_ids: Any, *, n: int = 10_000, seed: int = 0,
                           ci: float = 0.95) -> tuple[float, float]:
    """Percentile CI for a pooled per-row mean, resampling games (not rows)."""
    v = num(values)
    ok = np.isfinite(v)
    if not ok.any() or n <= 0:
        return NAN, NAN
    sums = _game_sums(np.asarray(game_ids)[ok], v[ok], np.ones(int(ok.sum())))
    boot = _bootstrap_sums(sums, n, np.random.default_rng(seed))
    return _ci(boot[:, 0] / boot[:, 1], ci)


# ---------------------------------------------------------------------------
# Calibration (26.13) and uncertainty coverage (26.12)
# ---------------------------------------------------------------------------

def calibration_table(y: Any, p: Any, n_bins: int = 10) -> dict[str, Any]:
    """Equal-width bins over [0, 1] (>= 10 bins, 26.13).

    Per bin: mean predicted, empirical frequency, n, Wilson 95% CI. Summary:
    ECE (n-weighted mean |gap|), MCE (max |gap| over non-empty bins), and the
    calibration slope / intercept of the logistic recalibration
    y ~ a + b * logit(p) (perfect calibration: a = 0, b = 1).
    """
    y, p = num(y), num(p)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    bins = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1) if p.size else np.array([], int)
    rows = []
    for b in range(n_bins):
        m = bins == b
        nb = int(m.sum())
        mean_p = float(p[m].mean()) if nb else NAN
        freq = float(y[m].mean()) if nb else NAN
        lo, hi = wilson_interval(float(y[m].sum()), nb)
        rows.append({"bin": b, "lower": b / n_bins, "upper": (b + 1) / n_bins, "n": nb,
                     "mean_predicted": mean_p, "empirical_frequency": freq, "ci_low": lo, "ci_high": hi,
                     "abs_gap": abs(mean_p - freq) if nb else NAN})
    table = pd.DataFrame(rows)
    n = int(y.size)
    filled = table[table["n"] > 0]
    ece = float((filled["n"] * filled["abs_gap"]).sum() / n) if n else NAN
    mce = float(filled["abs_gap"].max()) if n else NAN
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    fit = logistic_regression(np.column_stack([np.ones(n), z]), y) if n else None
    intercept, slope = (fit["coef"] if fit else (NAN, NAN))
    return {"n": n, "n_bins": n_bins, "ece": ece, "mce": mce, "slope": float(slope),
            "intercept": float(intercept), "table": table}


def _weighted_recalibration_slopes(w: np.ndarray, z: np.ndarray, y: np.ndarray, start: tuple[float, float],
                                   max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Slope of y ~ a + b * z fitted by Newton for each row of weights `w` (resamples x observations),
    starting at `start`. NaN where it does not converge or |coef| > 50 (as logistic_regression)."""
    m = w.shape[0]
    a, b = np.full(m, float(start[0])), np.full(m, float(start[1]))
    zz = z * z
    converged = np.zeros(m, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(max_iter):
            r = np.multiply.outer(b, z)                 # in place from here: two (m x rows) arrays per step
            r += a[:, None]
            expit(r, out=r)
            v = 1 - r
            v *= r
            v *= w
            np.subtract(y, r, out=r)
            r *= w
            g0, g1 = r.sum(axis=1), r @ z
            h00, h01, h11 = v.sum(axis=1), v @ z, v @ zz
            det = h00 * h11 - h01 * h01
            da, db = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
            a, b = a + da, b + db
            converged = np.isfinite(da) & np.isfinite(db) & (np.maximum(np.abs(da), np.abs(db)) < tol)
            if (converged | ~np.isfinite(a + b)).all():
                break
    ok = converged & np.isfinite(a) & np.isfinite(b) & (np.maximum(np.abs(a), np.abs(b)) <= 50)
    return np.where(ok, b, NAN)


def calibration_slope_bootstrap(y: Any, p: Any, game_ids: Any, *, n: int = 10_000, seed: int = 0,
                                ci: float = 0.95, max_cells: int = 1_000_000) -> dict[str, Any]:
    """Calibration slope (and intercept) of y ~ a + b * logit(p), as calibration_table, with a
    percentile CI from resampling GAMES with replacement (26.13; the game is the unit, as in 26.1).

    The draws are the other game bootstraps' (default_rng(seed), _game_draws). Each resample refits
    the recalibration on every row of the drawn games, a game drawn k times weighing k, by Newton
    from the full-sample fit. Resamples that do not converge are left out of the CI; n_valid counts
    the ones kept.
    """
    y, p = num(y), num(p)
    gid = pd.Series(game_ids).to_numpy()
    ok = np.isfinite(y) & np.isfinite(p) & pd.notna(gid)
    y, gid = y[ok], gid[ok]
    pc = np.clip(p[ok], EPS, 1 - EPS)
    z = np.log(pc / (1 - pc))
    codes, uniq = pd.factorize(gid)
    g, rows = len(uniq), len(y)
    fit = logistic_regression(np.column_stack([np.ones(rows), z]), y) if rows else None
    intercept, slope = fit["coef"] if fit else (NAN, NAN)
    out: dict[str, Any] = {"slope": float(slope), "intercept": float(intercept), "ci_low": NAN, "ci_high": NAN,
                           "ci_level": ci, "n_resamples": int(n), "n_valid": 0, "seed": seed, "unit": "game",
                           "n_games": int(g), "n_obs": int(rows)}
    if not g or n <= 0 or not (np.isfinite(slope) and np.isfinite(intercept)):
        return out
    slopes = np.full(n, NAN)
    block = max(1, max_cells // rows)
    for start, idx in _game_draws(g, n, np.random.default_rng(seed), max_cells):
        m = len(idx)
        counts = np.bincount((idx + (np.arange(m) * g)[:, None]).ravel(), minlength=m * g).reshape(m, g)
        for s in range(0, m, block):
            slopes[start + s:start + min(s + block, m)] = _weighted_recalibration_slopes(
                counts[s:s + block][:, codes].astype(float), z, y, (intercept, slope))
    lo, hi = _ci(slopes, ci)
    return out | {"ci_low": lo, "ci_high": hi, "n_valid": int(np.isfinite(slopes).sum())}


def interval_coverage(lower: Any, upper: Any, target: Any) -> float:
    """Share of rows with lower <= target <= upper (26.12). Rows with a missing target are ignored."""
    lo, hi, t = num(lower), num(upper), num(target)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(t)
    return float(np.mean((lo[ok] <= t[ok] + 1e-12) & (t[ok] <= hi[ok] + 1e-12))) if ok.any() else NAN


def binned_frequency_target(y: Any, p: Any, min_obs_per_bin: int, min_bins: int = 10) -> np.ndarray | None:
    """Proxy coverage target when the true probability is unknown (26.12).

    Sort by p, cut into floor(N / min_obs_per_bin) equal-count bins, and give
    each observation its bin's empirical outcome frequency. None if fewer than
    `min_bins` bins of `min_obs_per_bin` observations can be formed.
    """
    y, p = num(y), num(p)
    n = y.size
    k = n // max(int(min_obs_per_bin), 1)
    if k < min_bins or not (np.isfinite(y).all() and np.isfinite(p).all()):
        return None
    order = np.argsort(p, kind="stable")
    bins = np.empty(n, dtype=int)
    bins[order] = (np.arange(n) * k) // n
    freq = np.bincount(bins, weights=y, minlength=k) / np.bincount(bins, minlength=k)
    return freq[bins]


def _coverable_rows(lower: np.ndarray, upper: np.ndarray, m_low: float, m_high: float) -> int:
    """Most rows of one group whose p could lie in [lower, upper] when mean(p) lies in [m_low, m_high].

    All rows fit when the band meets [mean(lower), mean(upper)]. When the mean must exceed
    mean(upper), take m = m_low (the most favourable value): covered rows add at most
    upper_i, uncovered rows at most 1, so a set S fits iff sum_S (1 - upper_i) <= n (1 - m);
    the largest S takes the rows with the largest upper first. Symmetrically below: a set S
    fits iff sum_S lower_i <= n m, smallest lower first.
    """
    n = lower.size
    tol = 1e-12 * max(n, 1)
    if m_low > upper.mean():
        cost, budget = np.sort(1.0 - upper), n * (1.0 - m_low)
    elif m_high < lower.mean():
        cost, budget = np.sort(lower), n * m_high
    else:
        return n
    return int(np.searchsorted(np.cumsum(cost), budget + tol, side="right"))


def coverage_upper_bound(y: Any, p: Any, lower: Any, upper: Any, clusters: Any = None, *,
                         min_obs_per_bin: int, alpha: float) -> dict[str, Any] | None:
    """Upper confidence bound on per-observation interval coverage from outcomes alone (26.12).

    Why a bound and not an estimate: each observation has one Bernoulli outcome, so the
    outcomes identify the group mean of p_true but not its spread around that mean
    (Var(y | group) = m (1 - m) for every spread). Coverage of p_true depends on the
    spread, so no outcome-based statistic measures it without bias. The mean does limit
    it from above: when a group's mean lies outside most of its intervals, some rows must
    lie outside theirs.

    Method (parameters from config, fixed before measurement):
    - k = floor(N / min_obs_per_bin) equal-count bins of p (outcome-free grouping);
    - per bin, a Wilson interval for the mean outcome at level 1 - alpha / k (Bonferroni
      over bins), with effective size n^2 / sum_c n_c^2 over `clusters` (games): exact when
      rows of one cluster are perfectly correlated, conservative otherwise;
    - per bin, the most rows whose p_true could lie inside their interval for any bin mean
      in that Wilson interval (`_coverable_rows`, exact);
    - bound = coverable rows / N.

    With probability >= 1 - alpha (normal approximation inside Wilson), the share of rows
    with lower <= p_true <= upper is <= bound. It can show under-coverage; it can never
    show adequate coverage and cannot detect over-coverage. None if N < min_obs_per_bin
    or inputs are not finite.
    """
    y, p, lo, hi = num(y), num(p), num(lower), num(upper)
    n = y.size
    k = n // max(int(min_obs_per_bin), 1)
    if k < 1 or not all(np.isfinite(a).all() for a in (y, p, lo, hi)):
        return None
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    cl = pd.factorize(pd.Series(np.asarray(clusters, dtype=object)), use_na_sentinel=False)[0] \
        if clusters is not None else np.arange(n)
    order = np.argsort(p, kind="stable")
    bins = np.empty(n, dtype=int)
    bins[order] = (np.arange(n) * k) // n
    z = float(stats.norm.ppf(1 - alpha / (2 * k)))
    rows, covered = [], 0
    for b in range(k):
        m = bins == b
        nb = int(m.sum())
        n_eff = nb * nb / float((np.bincount(cl[m]) ** 2).sum())
        freq = float(y[m].mean())
        ci_lo, ci_hi = wilson_interval(freq * n_eff, n_eff, z)
        c = _coverable_rows(lo[m], hi[m], ci_lo, ci_hi)
        covered += c
        rows.append({"bin": b, "n": nb, "n_eff": n_eff, "mean_p": float(p[m].mean()), "frequency": freq,
                     "ci_low": ci_lo, "ci_high": ci_hi, "mean_lower": float(lo[m].mean()),
                     "mean_upper": float(hi[m].mean()), "coverable_share": c / nb})
    table = pd.DataFrame(rows)
    return {"bound": covered / n, "n": int(n), "n_bins": int(k), "alpha": float(alpha), "z": z,
            "bins_binding": int((table["coverable_share"] < 1).sum()), "table": table}


# ---------------------------------------------------------------------------
# Decisions: side view, edge buckets (26.6), monotonicity (26.7)
# ---------------------------------------------------------------------------

def side_view(decisions: pd.DataFrame) -> pd.DataFrame:
    """Decisions with side-oriented probabilities and realized edge (see module docstring)."""
    d = decisions.copy()
    is_no = d["side"].eq("no").fillna(False).to_numpy(dtype=bool) if "side" in d else np.zeros(len(d), bool)
    p_market = num(d["p_market"])
    d["p_market_side"] = np.where(is_no, 1 - p_market, p_market)
    d["p_model_side"] = num(d["quoted_price"]) + num(d["predicted_edge"])
    side_won = num(d["side_won"]) if "side_won" in d else np.full(len(d), NAN)
    if "outcome" in d:
        outcome = num(d["outcome"])
        side_won = np.where(np.isfinite(side_won), side_won, np.where(is_no, 1 - outcome, outcome))
    d["side_won"] = side_won
    d["realized_edge"] = side_won - num(d["quoted_price"])
    d["is_trade"] = d["decision"].ne("PASS").fillna(False).to_numpy(dtype=bool)
    return d


def bucket_labels(edges: Sequence[float]) -> list[str]:
    pct = [f"{100 * e:g}" for e in edges]
    return [f"<{pct[0]}%", *[f"{a}-{b}%" for a, b in zip(pct, pct[1:])], f">{pct[-1]}%"]


def assign_buckets(values: Any, edges: Sequence[float]) -> np.ndarray:
    """Bucket rank per value: 0 for < edges[0], i for [edges[i-1], edges[i]), len(edges) for >= edges[-1]; -1 if NaN."""
    v = num(values)
    rank = np.searchsorted(np.asarray(edges, float), v, side="right")
    return np.where(np.isfinite(v), rank, -1)


def _ratio(num_: float, den: float) -> float:
    return float(num_ / den) if den and np.isfinite(den) and den != 0 else NAN


def edge_buckets(df: pd.DataFrame, edges: Sequence[float], min_games: int, *, n_boot: int = 10_000,
                 seed: int = 0, ci: float = 0.95) -> pd.DataFrame:
    """Edge-bucket validation (26.6) over ALL observations (trades and passes) of one scenario.

    Bucket key = predicted_edge of the side considered. Buckets are fixed by
    `edges` and never merged; a bucket with fewer than `min_games` distinct
    games is labeled INSUFFICIENT SAMPLE. realized_edge = mean(side_won -
    quoted_price); its CI resamples games. ROI is over trades only.
    """
    d = side_view(df)
    d = d[np.isfinite(num(d["predicted_edge"])) & np.isfinite(d["realized_edge"].to_numpy())]
    ranks = assign_buckets(d["predicted_edge"], edges)
    rows = []
    for r, label in enumerate(bucket_labels(edges)):
        g = d[ranks == r]
        t = g[g["is_trade"]]
        n_games = int(g["game_id"].nunique())
        lo, hi = game_bootstrap_mean_ci(g["realized_edge"], g["game_id"], n=n_boot, seed=seed + r, ci=ci) \
            if len(g) else (NAN, NAN)
        stake = float(num(t["stake"]).sum()) if len(t) else 0.0
        rows.append({
            "bucket": label, "rank": r,
            "lower": edges[r - 1] if r > 0 else None, "upper": edges[r] if r < len(edges) else None,
            "observations": int(len(g)), "n_games": n_games, "trades": int(len(t)),
            "mean_p_market": _mean(g["p_market_side"]), "mean_p_model": _mean(g["p_model_side"]),
            "mean_p_conservative": _mean(g["p_conservative"]) if "p_conservative" in g else NAN,
            "empirical_frequency": _mean(g["side_won"]), "mean_predicted_edge": _mean(g["predicted_edge"]),
            "realized_edge": _mean(g["realized_edge"]), "realized_edge_ci_low": lo, "realized_edge_ci_high": hi,
            "gross_roi": _ratio(float(num(t["gross_pnl"]).sum()), stake) if len(t) else NAN,
            "net_roi": _ratio(float(num(t["net_pnl"]).sum()), stake) if len(t) else NAN,
            "sufficient": n_games >= min_games,
            "label": "OK" if n_games >= min_games else INSUFFICIENT,
        })
    return pd.DataFrame(rows)


def _mean(s: Any) -> float:
    v = num(s)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else NAN


def edge_monotonicity(buckets: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, Any]:
    """26.7 diagnostics: Spearman(bucket rank, realized edge) over sufficiently-sampled
    buckets, and logistic side_won ~ p_market_side + predicted_edge with
    game-clustered standard errors."""
    ok = buckets[buckets["sufficient"].astype(bool)] if len(buckets) else buckets
    rho = rho_p = NAN
    if len(ok) >= 3 and np.ptp(num(ok["realized_edge"])) > 0:
        res = stats.spearmanr(num(ok["rank"]), num(ok["realized_edge"]))
        rho, rho_p = float(res.statistic), float(res.pvalue)
    d = side_view(decisions)
    d = d[np.isfinite(num(d["predicted_edge"])) & np.isfinite(d["side_won"].to_numpy())
          & np.isfinite(d["p_market_side"].to_numpy())]
    X = np.column_stack([np.ones(len(d)), num(d["p_market_side"]), num(d["predicted_edge"])])
    fit = logistic_regression(X, num(d["side_won"]), groups=d["game_id"].to_numpy())
    return {"n_sufficient_buckets": int(len(ok)), "spearman": rho, "spearman_p": rho_p,
            "n_obs": int(len(d)), "converged": fit["converged"],
            "coef_intercept": float(fit["coef"][0]), "coef_market": float(fit["coef"][1]),
            "coef_edge": float(fit["coef"][2]), "se_edge": float(fit["se"][2]), "p_edge": float(fit["p"][2]),
            "se_market": float(fit["se"][1]), "p_market": float(fit["p"][1]),
            "se_method": "cluster-robust by game"}


# ---------------------------------------------------------------------------
# Trading evaluation (Sec 17, 26.8, 26.9)
# ---------------------------------------------------------------------------

def _ordered(decisions: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("snapshot_time", "observation_id") if c in decisions]
    return decisions.sort_values(keys, kind="stable") if keys else decisions


def max_drawdown(pnl: Any) -> float:
    """Largest peak-to-trough fall of cumulative P&L (starting from 0), as a positive amount."""
    c = np.concatenate([[0.0], np.cumsum(num(pnl))])
    return float(np.max(np.maximum.accumulate(c) - c))


def longest_losing_streak(pnl: Any) -> int:
    best = cur = 0
    for v in num(pnl):
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return int(best)


def trading_summary(decisions: pd.DataFrame) -> dict[str, Any]:
    """Sec 17 trading evaluation for ONE scenario's decisions (trades and passes).

    pnl_before_fees = gross (neither fees nor slippage), pnl_after_fees =
    gross - fees, pnl_after_fees_slippage = net (26.8). With an exit leg, fees
    and slippage are both legs', so gross - fees - slippage = net.
    Drawdown and streaks follow snapshot_time order.
    """
    d = _ordered(side_view(decisions)) if len(decisions) else side_view(decisions)
    t = d[d["is_trade"]]
    reasons = d.loc[~d["is_trade"], "pass_reason"].value_counts().to_dict() if "pass_reason" in d else {}
    out: dict[str, Any] = {"observations": int(len(d)), "trades": int(len(t)), "passes": int(len(d) - len(t)),
                           "pass_reasons": {str(k): int(v) for k, v in reasons.items()}}
    gross, fees = float(num(t["gross_pnl"]).sum()), float(num(t["calculated_fee"]).sum()) + _exit_sum(t, "exit_fee")
    slip = float(num(t["slippage_cost"]).sum()) + _exit_sum(t, "exit_slippage_cost")
    net, stake = float(num(t["net_pnl"]).sum()), float(num(t["stake"]).sum())
    out |= {
        "win_rate": _mean(num(t["net_pnl"]) > 0) if len(t) else NAN,
        "avg_entry_probability": _mean(t["execution_price"]),
        "avg_quoted_price": _mean(t["quoted_price"]),
        "avg_model_probability": _mean(t["p_model_side"]),
        "avg_conservative_probability": _mean(t["p_conservative"]) if "p_conservative" in t else NAN,
        "realized_outcome_probability": _mean(t["side_won"]),
        "avg_predicted_edge": _mean(t["predicted_edge"]),
        "avg_realized_edge": _mean(t["realized_edge"]),
        "contracts": float(num(t["quantity"]).sum()), "stake": stake,
        "gross_pnl": gross, "fees": fees, "slippage": slip, "net_pnl": net,
        "gross_roi": _ratio(gross, stake), "roi": _ratio(net, stake),
        "pnl_before_fees": gross, "pnl_after_fees": gross - fees, "pnl_after_fees_slippage": net,
        "max_drawdown": max_drawdown(t["net_pnl"]) if len(t) else 0.0,
        "longest_losing_streak": longest_losing_streak(t["net_pnl"]) if len(t) else 0,
        "exit_statuses": _counts(t, "exit_status"), "exit_fallbacks": _counts(t, "exit_reason"),
    }
    return out


def _exit_sum(t: pd.DataFrame, col: str) -> float:
    """Exit-leg total; held trades carry no exit leg (NULL), so they add nothing."""
    return float(np.nansum(num(t[col]))) if col in t else 0.0


def _counts(t: pd.DataFrame, col: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in t[col].value_counts().items()} if col in t else {}


# ---------------------------------------------------------------------------
# P&L concentration (26.14)
# ---------------------------------------------------------------------------

def _prob_band(p: np.ndarray) -> np.ndarray:
    lo = np.clip(np.floor(p * 10), 0, 9) / 10
    return np.array([f"{a:.1f}-{a + 0.1:.1f}" if np.isfinite(a) else "unknown" for a in lo], dtype=object)


def _group_table(t: pd.DataFrame, key: pd.Series | np.ndarray) -> pd.DataFrame:
    g = pd.DataFrame({"group": np.asarray(key, dtype=object), "net_pnl": num(t["net_pnl"]),
                      "stake": num(t["stake"])})
    g["group"] = g["group"].where(pd.notna(g["group"]), "unknown").astype(str)
    out = g.groupby("group", sort=True).agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
                                            stake=("stake", "sum")).reset_index()
    out["roi"] = out["net_pnl"] / out["stake"].where(out["stake"] != 0)
    pos = out["net_pnl"].clip(lower=0).sum()
    out["share_of_positive"] = out["net_pnl"] / pos if pos > 0 else NAN
    return out.sort_values("net_pnl", ascending=False, kind="stable").reset_index(drop=True)


def concentration(trades: pd.DataFrame, *, top10_max: float, team_max: float,
                  edges: Sequence[float]) -> dict[str, Any]:
    """26.14 diagnostics over one scenario's trades.

    Trade shares (largest / top 5 / top 10) are of the total positive trade
    P&L. Group shares (best team, best month) are of the sum of positive
    group P&L. team = yes_team of the contract (team_backed is informational).
    Flags: top-10 share > top10_max, best-team share > team_max.
    """
    t = side_view(trades)
    t = t[t["is_trade"]]
    net = num(t["net_pnl"])
    month = pd.to_datetime(t["official_date"] if "official_date" in t else t["snapshot_time"]).dt.strftime("%Y-%m")
    is_no = t["side"].eq("no").fillna(False).to_numpy(dtype=bool)
    backed = np.where(is_no, t["away_team"].where(t["yes_side"].eq("home"), t["home_team"]), t["yes_team"]) \
        if {"home_team", "away_team", "yes_side", "yes_team"} <= set(t.columns) else None
    keys: dict[str, Any] = {
        "season": t.get("season"), "period": t.get("period_id"), "month": month, "team": t.get("yes_team"),
        "team_backed": backed, "market_probability_band": _prob_band(num(t["p_market_side"])),
        "predicted_edge_band": np.array(bucket_labels(edges), dtype=object)[
            np.clip(assign_buckets(t["predicted_edge"], edges), 0, None)] if len(t) else np.array([], object),
        "home_away": t.get("yes_side"), "availability_state": t.get("availability_state"),
    }
    groups = {name: _group_table(t, key) for name, key in keys.items() if key is not None}
    pos = np.sort(net[net > 0])[::-1]
    pos_total = float(pos.sum())
    share = (lambda k: float(pos[:k].sum() / pos_total) if pos_total > 0 else NAN)
    order = np.argsort(-net, kind="stable")
    top10 = order[:10]
    rest = np.setdiff1d(np.arange(len(t)), top10)
    stake = num(t["stake"])

    def best(name: str) -> tuple[Any, float]:
        g = groups.get(name)
        if g is None or g.empty or not np.isfinite(g["share_of_positive"].iloc[0]):
            return None, NAN
        return g["group"].iloc[0], float(g["share_of_positive"].iloc[0])

    best_team, team_share = best("team")
    best_month, month_share = best("month")
    top10_share = share(10)
    flags = []
    if np.isfinite(top10_share) and top10_share > top10_max:
        flags.append(f"top 10 trades = {top10_share:.1%} of positive P&L (> {top10_max:.0%})")
    if np.isfinite(team_share) and team_share > team_max:
        flags.append(f"team {best_team} = {team_share:.1%} of positive team P&L (> {team_max:.0%})")
    return {
        "trades": int(len(t)), "total_net_pnl": float(net.sum()), "positive_pnl": pos_total,
        "largest_trade": float(net.max()) if len(t) else NAN, "largest_trade_share": share(1),
        "top5_share": share(5), "top10_share": top10_share,
        "best_team": best_team, "best_team_share": team_share, "best_month": best_month,
        "best_month_share": month_share,
        "flag_top10": bool(np.isfinite(top10_share) and top10_share > top10_max),
        "flag_team": bool(np.isfinite(team_share) and team_share > team_max), "flags": flags,
        "ex_top10": {"trades": int(len(rest)), "net_pnl": float(net[rest].sum()),
                     "roi": _ratio(float(net[rest].sum()), float(stake[rest].sum()))},
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# Edge decay (26.15)
# ---------------------------------------------------------------------------

def edge_decay(trades: pd.DataFrame, period_order: Sequence[str], ci: float = 0.95) -> dict[str, Any]:
    """Per-period predicted edge, realized edge, net ROI over trades, and OLS
    realized_edge ~ period ordinal (unweighted, one point per period with trades).

    DECAY WARNING if the slope's CI upper bound < 0, or the two most recent
    periods with trades both have realized edge below the aggregate.
    """
    t = side_view(trades)
    t = t[t["is_trade"]]
    ordinal = {pid: i for i, pid in enumerate(period_order)}
    rows = []
    for pid in period_order:
        g = t[t["period_id"] == pid]
        if not len(g):
            continue
        rows.append({"period_id": pid, "ordinal": ordinal[pid], "trades": int(len(g)),
                     "predicted_edge": _mean(g["predicted_edge"]), "realized_edge": _mean(g["realized_edge"]),
                     "net_roi": _ratio(float(num(g["net_pnl"]).sum()), float(num(g["stake"]).sum()))})
    per = pd.DataFrame(rows, columns=["period_id", "ordinal", "trades", "predicted_edge", "realized_edge", "net_roi"])
    aggregate = _mean(t["realized_edge"])
    k = len(per)
    slope = se = lo = hi = NAN
    if k >= 3:
        x, yv = num(per["ordinal"]), num(per["realized_edge"])
        sxx = float(((x - x.mean()) ** 2).sum())
        if sxx > 0:
            slope = float(((x - x.mean()) * (yv - yv.mean())).sum() / sxx)
            resid = yv - (yv.mean() + slope * (x - x.mean()))
            se = float(math.sqrt((resid ** 2).sum() / (k - 2) / sxx))
            tcrit = float(stats.t.ppf(1 - (1 - ci) / 2, k - 2))
            lo, hi = slope - tcrit * se, slope + tcrit * se
    recent = bool(k >= 2 and (per["realized_edge"].iloc[-2:] < aggregate).all())
    return {"n_periods": k, "periods": per, "aggregate_realized_edge": aggregate, "slope": slope, "slope_se": se,
            "slope_ci_low": lo, "slope_ci_high": hi, "recent_two_underperform": recent,
            "decay_warning": bool((np.isfinite(hi) and hi < 0) or recent)}


# ---------------------------------------------------------------------------
# Multiple testing (26.5) and the market-shuffle test (26.17)
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues: Any) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values. NaN p-values stay NaN and don't count toward m."""
    p = num(pvalues)
    q = np.full(p.shape, NAN)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = ranked
    q[ok] = out
    return q


def shuffle_test(df: pd.DataFrame, *, n_perm: int = 1000, seed: int = 0, group_cols: Sequence[str] | None = None,
                 y: str = "y", p_model: str = "p_model", p_market: str = "p_market",
                 period: str = "period_id") -> dict[str, Any]:
    """Market-shuffle sanity test (26.17) — an EVALUATION-LEVEL shuffle.

    Predictions (p_model, p_market) are held fixed; models are NOT refit.
    Outcomes are permuted within groups (default: period x p_market decile),
    which preserves each group's market calibration while breaking any link
    between the model's residual and the outcome. The statistic is the pooled
    delta Brier (market - model). Empirical p = (1 + #perm >= observed) / (1 + n_perm).

    Uses (y - a)^2 - (y - b)^2 = 2y(b - a) + a^2 - b^2: only sum(y_perm * w)
    changes under permutation, with w = p_model - p_market.
    """
    cols = [y, p_model, p_market] + [c for c in (group_cols or [period]) if c in df.columns]
    d = df[list(dict.fromkeys(cols))].dropna(subset=[y, p_model, p_market]).reset_index(drop=True)
    n = len(d)
    method = ("evaluation-level shuffle: outcomes permuted within groups; predictions held fixed; "
              "models are NOT refit")
    if n == 0:
        return {"observed_delta_brier": NAN, "p_value": NAN, "n_perm": n_perm, "n_groups": 0, "n_obs": 0,
                "perm_mean": NAN, "perm_sd": NAN, "method": method}
    yv, pm, pk = num(d[y]), num(d[p_model]), num(d[p_market])
    if group_cols is None:
        decile = pd.qcut(pd.Series(pk).rank(method="first"), 10, labels=False) if n >= 10 else np.zeros(n)
        keys = [d[period].astype(str).to_numpy() if period in d else np.zeros(n), np.asarray(decile)]
    else:
        keys = [d[c].astype(str).to_numpy() for c in group_cols]
    codes = pd.MultiIndex.from_arrays(keys).factorize()[0] if len(keys) > 1 else pd.factorize(keys[0])[0]
    w = pm - pk
    const = float(np.mean(pk ** 2 - pm ** 2))
    observed = float(2 * np.mean(yv * w) + const)
    rng = np.random.default_rng(seed)
    total = np.zeros(n_perm)
    for g in np.unique(codes):
        idx = np.flatnonzero(codes == g)
        yg, wg = yv[idx], w[idx]
        if len(idx) < 2 or np.ptp(yg) == 0:
            total += float(yg @ wg)
            continue
        step = max(1, 2_000_000 // len(idx))
        for s in range(0, n_perm, step):
            m = min(step, n_perm - s)
            total[s:s + m] += rng.permuted(np.tile(yg, (m, 1)), axis=1) @ wg
    perm = 2 * total / n + const
    p = (1 + int(np.sum(perm >= observed - 1e-12))) / (1 + n_perm)
    return {"observed_delta_brier": observed, "p_value": float(p), "n_perm": int(n_perm),
            "n_groups": int(len(np.unique(codes))), "n_obs": n, "perm_mean": float(perm.mean()),
            "perm_sd": float(perm.std()), "method": method}
