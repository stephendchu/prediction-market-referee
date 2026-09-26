"""Probability calibration, Layer C (spec Sec 7, 26.13).

Calibrators are fit only on the calibration split: the latest
`model.calibration_fraction` of a training window, which the base model never
saw. Methods, simplest first: none, platt (expit(a + b*logit p)), isotonic.

Selection (`choose_calibration`) scores every method by K-fold cross-validation
inside the calibration split (contiguous folds in time order, cut on game
boundaries), so isotonic is not rewarded for fitting its own data, then refits
the chosen method on the whole split. Ties go to the simpler method, where a
"tie" is a CV Brier within `tie_se` game-clustered standard errors of the best
(the one-standard-error rule; `tie_se=0` is a plain argmin).

The time-split helpers used by the model layer (`tail_split`,
`contiguous_folds`) live here too: rows arrive sorted by snapshot_time and a
game is never divided between two sides of a split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from ..pricing import EPS, expit, logit

METHODS = ("none", "platt", "isotonic")  # simplicity order: ties go left
PLATT_BOUNDS = ((-20.0, 20.0), (0.01, 20.0))  # slope > 0 keeps the map increasing


# ---------------------------------------------------------------------------
# Game-boundary splits on time-ordered rows
# ---------------------------------------------------------------------------

def game_codes(groups: Sequence | None, n: int) -> np.ndarray:
    """Per row, its game's rank by first appearance (rows are time-ordered)."""
    if groups is None:
        return np.arange(n)
    g = np.asarray(groups)
    if len(g) != n:
        raise ValueError(f"groups has {len(g)} entries for {n} rows")
    codes, _ = pd.factorize(g, sort=False)
    if (codes < 0).any():
        raise ValueError("groups contain nulls")
    return codes


def _rows_before(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(codes)
    return counts, np.cumsum(counts) - counts


def tail_split(groups: Sequence | None, n: int, fraction: float) -> np.ndarray:
    """Mask of the latest ~`fraction` of rows, cut on a game boundary.

    Games are ordered by their first row and cut at the game boundary nearest
    (1 - fraction) of the rows (ties: the larger tail); a game whose rows
    straddle the cut goes wholly to the side of its first row.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    codes = game_codes(groups, n)
    counts, _ = _rows_before(codes)
    head_rows = np.cumsum(counts)[:-1]  # rows in the head if it ends after game g
    if len(head_rows) == 0:
        raise ValueError(f"too few games ({len(counts)}) to hold out a {fraction:.0%} tail")
    last_head = int(np.argmin(np.abs(head_rows - (1 - fraction) * n)))
    return codes > last_head


def contiguous_folds(groups: Sequence | None, n: int, k: int) -> np.ndarray:
    """Fold id per row: k contiguous time blocks of ~equal rows, whole games only."""
    codes = game_codes(groups, n)
    counts, before = _rows_before(codes)
    k = max(1, min(k, len(counts)))
    fold_of_game = np.minimum(k - 1, (k * before) // max(n, 1))
    return fold_of_game[codes]


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

@dataclass
class Calibrator:
    method: str
    a: float = 0.0
    b: float = 1.0
    iso: IsotonicRegression | None = None
    cv_scores: dict[str, float] = field(default_factory=dict)
    cv_se: dict[str, float] = field(default_factory=dict)

    def transform(self, p_raw: Any) -> np.ndarray:
        p = np.clip(np.asarray(p_raw, dtype=float), EPS, 1 - EPS)
        if self.method == "none":
            return p
        if self.method == "platt":
            out = expit(self.a + self.b * logit(p))
        elif self.method == "isotonic":
            out = self.iso.predict(p)
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        return np.clip(out, EPS, 1 - EPS)

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method}
        if self.method == "platt":
            out |= {"a": self.a, "b": self.b}
        elif self.method == "isotonic":
            out["n_steps"] = int(len(self.iso.X_thresholds_))
        return out


def _platt_loss(theta: np.ndarray, z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = theta[0] + theta[1] * z
    resid = expit(eta) - y
    n = len(y)
    loss = np.sum(np.logaddexp(0.0, eta) - y * eta) / n
    return loss, np.array([resid.sum(), (resid * z).sum()]) / n


def fit_calibrator(method: str, p_raw: Any, y: Any) -> Calibrator:
    p = np.clip(np.asarray(p_raw, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    if method == "none":
        return Calibrator("none")
    if method == "platt":
        res = minimize(_platt_loss, np.array([0.0, 1.0]), args=(logit(p), y), jac=True,
                       method="L-BFGS-B", bounds=PLATT_BOUNDS)
        return Calibrator("platt", a=float(res.x[0]), b=float(res.x[1]))
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        return Calibrator("isotonic", iso=iso.fit(p, y))
    raise ValueError(f"unknown calibration method {method!r}")


def cross_val_calibrated(method: str, p_raw: Any, y: Any, *, groups: Sequence | None = None,
                         k: int = 5) -> np.ndarray:
    """Out-of-fold calibrated predictions over contiguous game-boundary folds."""
    p = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    folds = contiguous_folds(groups, len(p), k)
    out = np.empty(len(p))
    if folds.max() == 0:  # a single fold: nothing to hold out
        out[:] = fit_calibrator("none", p, y).transform(p)
        return out
    for f in np.unique(folds):
        test = folds == f
        out[test] = fit_calibrator(method, p[~test], y[~test]).transform(p[test])
    return out


def clustered_se(d: np.ndarray, codes: np.ndarray) -> float:
    """Standard error of mean(d) with rows clustered by game."""
    n, n_games = len(d), int(codes.max()) + 1
    if n_games < 2:
        return 0.0
    per_game = np.bincount(codes, weights=d - d.mean(), minlength=n_games)
    return float(np.sqrt(np.sum(per_game ** 2) * n_games / (n_games - 1)) / n)


def choose_calibration(p_raw: Any, y: Any, methods: Sequence[str] = METHODS, *,
                       groups: Sequence | None = None, k: int = 5,
                       tie_se: float = 1.0) -> tuple[str, Calibrator, dict[str, float]]:
    """Pick the calibration method by CV Brier on the calibration split.

    Returns (method, calibrator refit on the whole split, {method: CV Brier}).
    The simplest method whose CV Brier is within `tie_se` clustered standard
    errors (of its paired loss difference) of the best method's wins.
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown calibration methods {sorted(unknown)}")
    ordered = [m for m in METHODS if m in methods] or ["none"]
    p = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    codes = game_codes(groups, len(p))
    losses = {m: (cross_val_calibrated(m, p, y, groups=groups, k=k) - y) ** 2 for m in ordered}
    scores = {m: float(losses[m].mean()) for m in ordered}
    best = min(ordered, key=lambda m: scores[m])  # min() keeps the first (simplest) on exact ties
    se = {m: clustered_se(losses[m] - losses[best], codes) for m in ordered}
    chosen = next(m for m in ordered if scores[m] <= scores[best] + tie_se * se[m])
    cal = fit_calibrator(chosen, p, y)
    cal.cv_scores, cal.cv_se = scores, se
    return chosen, cal, scores
