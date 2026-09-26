"""Model uncertainty: p_model -> [lower_bound, upper_bound] (spec Sec 13, 26.12).

Not a constant haircut and never tuned on P&L. The interval is symmetric in
logit space around logit(p_model):

    sd_total     = scale * sqrt(sd_ensemble^2 + sd_calibration^2) * (1 + missing_share)
    lower, upper = expit(logit(p_model) -/+ z * sd_total),  z = Phi^-1(0.5 + nominal/2)

Components (all fixed before any evaluation data is seen):
  sd_ensemble     per row: std across game-level bootstrap refits of the base
                  model (fit split only) of logit(calibrated member prediction).
                  Isotonic calibration is piecewise constant, so under it this
                  component is lumpy (members inside one step agree exactly).
  sd_calibration  one constant per model from the calibration split, in
                  probability units, then per row in logit units by the delta
                  method sd_p / (p(1-p)), capped at SD_CAL_LOGIT_MAX:
                    sd_p^2 = max(0, excess) + sampling
                    excess   = sum_b n_b[(freq_b - pred_b)^2 - pred_b(1-pred_b)/n_b] / N
                               (squared calibration error beyond binomial noise, 10 equal-count bins)
                    sampling = sum_i p_i(1-p_i) / N^2
                               (variance of the calibration-in-the-large check itself; keeps the
                               width non-zero when the excess estimate is 0 — the market baseline's
                               only component — and shrinks as calibration data grows)
  missing_share   per row: share of non-market feature families whose features are all NaN.
  scale           tuned on the calibration split so that the interval covers the BINNED
                  FREQUENCY PROXY at exactly the nominal rate: target_i = empirical outcome
                  frequency of row i's equal-count p_model bin. Bins hold >= review.
                  coverage_bin_min_obs rows when there are at least 10 such bins' worth of rows
                  (n_bins = max(10, N // min_obs)); with fewer rows, 10 bins of N/10. Found by
                  bisection (log scale) on [0.01, 100]. Uses outcomes only: real data has no
                  p_true, and nothing here may ever see a synthetic p_true.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..pricing import EPS, expit, logit
from .calibration import Calibrator, game_codes

SD_CAL_LOGIT_MAX = 1.0  # at scale 1 a 90% band is then already +/-1.6 logits (0.5 -> [0.16, 0.84])
SCALE_BOUNDS = (0.01, 100.0)
CALIBRATION_BINS = 10
EXEMPT_FAMILIES = ("market",)


def z_value(nominal_coverage: float) -> float:
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be in (0, 1)")
    return float(norm.ppf(0.5 + nominal_coverage / 2))


def total_sd(sd_ensemble: Any, sd_calibration: Any, missing_share: Any, scale: float) -> np.ndarray:
    ens = np.asarray(sd_ensemble, dtype=float)
    cal = np.asarray(sd_calibration, dtype=float)
    return scale * np.sqrt(ens ** 2 + cal ** 2) * (1 + np.asarray(missing_share, dtype=float))


def interval(p_model: Any, sd_ensemble: Any, sd_calibration: Any, missing_share: Any,
             scale: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) in [0, 1] with lower <= p_model <= upper exactly."""
    p = np.asarray(p_model, dtype=float)
    half = z * total_sd(sd_ensemble, sd_calibration, missing_share, scale)
    zp = logit(p)
    flat = half <= 0  # the expit(logit(p)) round trip is not exact
    lower = np.clip(np.where(flat, p, np.minimum(expit(zp - half), p)), 0.0, 1.0)
    upper = np.clip(np.where(flat, p, np.maximum(expit(zp + half), p)), 0.0, 1.0)
    return lower, upper


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def equal_count_bins(p: Any, n_bins: int) -> np.ndarray:
    """Bin id per row: n_bins groups of (near-)equal count by rank of p."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    n_bins = max(1, min(n_bins, n))
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(p, kind="stable")] = np.arange(n)
    return (rank * n_bins) // max(n, 1)


def calibration_sd(p: Any, y: Any, n_bins: int = CALIBRATION_BINS) -> float:
    """sd_p of the calibration component, in probability units (module docstring)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n == 0:
        raise ValueError("no calibration rows")
    bins = equal_count_bins(p, n_bins)
    n_b = np.bincount(bins).astype(float)
    freq = np.bincount(bins, weights=y) / n_b
    pred = np.bincount(bins, weights=p) / n_b
    excess = np.sum(n_b * (freq - pred) ** 2 - pred * (1 - pred)) / n
    sampling = np.sum(p * (1 - p)) / n ** 2
    return float(np.sqrt(max(0.0, excess) + sampling))


def calibration_sd_logit(sd_p: float, p_model: Any) -> np.ndarray:
    p = np.clip(np.asarray(p_model, dtype=float), EPS, 1 - EPS)
    return np.minimum(sd_p / (p * (1 - p)), SD_CAL_LOGIT_MAX)


def missing_share(X: pd.DataFrame | None, families: dict[str, str] | None) -> np.ndarray:
    """Per row, the share of non-market families whose features are all NaN."""
    if X is None or families is None:
        return np.zeros(0 if X is None else len(X))
    unknown = [c for c in X.columns if c not in families]
    if unknown:
        raise ValueError(f"features without a family: {unknown}")
    by_family: dict[str, list[str]] = {}
    for col in X.columns:
        if families[col] not in EXEMPT_FAMILIES:
            by_family.setdefault(families[col], []).append(col)
    if not by_family:
        return np.zeros(len(X))
    nan = X.isna()
    all_missing = np.column_stack([nan[cols].all(axis=1).to_numpy() for cols in by_family.values()])
    return all_missing.mean(axis=1)


def ensemble_sd(member_logits: np.ndarray) -> np.ndarray:
    """Per-row std (ddof=1) across members; rows = axis 1."""
    m = np.asarray(member_logits, dtype=float)
    if m.ndim != 2 or m.shape[0] < 2:
        return np.zeros(m.shape[-1] if m.ndim else 0)
    return m.std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Scale tuning on the binned frequency proxy
# ---------------------------------------------------------------------------

def proxy_n_bins(n: int, min_bin_obs: int) -> int:
    return max(1, min(n, max(CALIBRATION_BINS, n // max(min_bin_obs, 1))))


def proxy_targets(p_model: Any, y: Any, min_bin_obs: int) -> np.ndarray:
    """Per row, the empirical outcome frequency of its equal-count p_model bin."""
    p = np.asarray(p_model, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = equal_count_bins(p, proxy_n_bins(len(p), min_bin_obs))
    freq = np.bincount(bins, weights=y) / np.bincount(bins)
    return freq[bins]


def proxy_coverage(p_model: Any, y: Any, lower: Any, upper: Any, min_bin_obs: int) -> float:
    target = proxy_targets(p_model, y, min_bin_obs)
    return float(np.mean((np.asarray(lower) <= target) & (target <= np.asarray(upper))))


def tune_scale(p_model: Any, y: Any, sd_ensemble: Any, sd_calibration: Any, missing_share: Any,
               z: float, nominal_coverage: float, min_bin_obs: int, *,
               bounds: tuple[float, float] = SCALE_BOUNDS, iters: int = 60) -> float:
    """Smallest scale in `bounds` whose proxy coverage reaches nominal (bisection in log scale)."""
    p = np.asarray(p_model, dtype=float)
    target = proxy_targets(p, y, min_bin_obs)

    def covered(scale: float) -> float:
        lo, hi = interval(p, sd_ensemble, sd_calibration, missing_share, scale, z)
        return float(np.mean((lo <= target) & (target <= hi)))

    lo, hi = bounds
    if covered(lo) >= nominal_coverage:
        return lo
    if covered(hi) < nominal_coverage:
        return hi
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if covered(mid) >= nominal_coverage:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Fitted uncertainty model
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyModel:
    members: list[Any]                 # bootstrap refits; empty for the market baseline
    calibrator: Calibrator             # the fitted model's calibrator, applied to each member
    families: dict[str, str] | None    # None: no missing-information penalty
    sd_calibration_p: float
    z: float
    nominal_coverage: float
    scale: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def components(self, X: pd.DataFrame | None, p_market: Any, p_model: Any) -> dict[str, np.ndarray]:
        p_model = np.asarray(p_model, dtype=float)
        if self.members:
            logits = np.vstack([logit(self.calibrator.transform(m.predict_raw(X, p_market)))
                                for m in self.members])
            sd_ens = ensemble_sd(logits)
        else:
            sd_ens = np.zeros(len(p_model))
        ms = missing_share(X, self.families) if self.families is not None else np.zeros(len(p_model))
        return {"sd_ensemble": sd_ens,
                "sd_calibration": calibration_sd_logit(self.sd_calibration_p, p_model),
                "missing_share": ms}

    def bounds(self, X: pd.DataFrame | None, p_market: Any,
               p_model: Any) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        comps = self.components(X, p_market, p_model)
        lower, upper = interval(p_model, comps["sd_ensemble"], comps["sd_calibration"],
                                comps["missing_share"], self.scale, self.z)
        return lower, upper, comps

    def component_json(self, comps: dict[str, np.ndarray]) -> list[str]:
        cols = ("sd_ensemble", "sd_calibration", "missing_share")
        return [json.dumps({"sd_ensemble": float(e), "sd_calibration": float(c), "missing_share": float(m),
                            "scale": self.scale, "z": self.z}, separators=(",", ":"))
                for e, c, m in zip(*(comps[k] for k in cols))]


def bootstrap_rows(groups: Sequence | None, n: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one game-level bootstrap resample (games drawn with replacement)."""
    codes = game_codes(groups, n)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.cumsum(counts) - counts
    pick = rng.integers(0, len(counts), len(counts))
    return np.concatenate([order[starts[g]:starts[g] + counts[g]] for g in pick])


def fit_uncertainty(make_member: Callable[[int], Any] | None,
                    X_fit: pd.DataFrame | None, p_fit: np.ndarray, y_fit: np.ndarray,
                    groups_fit: Sequence | None,
                    X_cal: pd.DataFrame | None, p_cal: np.ndarray, y_cal: np.ndarray,
                    p_model_cal: np.ndarray, calibrator: Calibrator,
                    families: dict[str, str] | None, *, ensemble_size: int,
                    nominal_coverage: float, min_bin_obs: int, seed: int) -> UncertaintyModel:
    """Bootstrap ensemble on the fit split; calibration sd and scale on the calibration split.

    `p_model_cal` must be out of sample for both the base model and the
    calibrator (out-of-fold calibrated predictions), or the calibration
    component and the scale are understated.
    """
    rng = np.random.default_rng(seed)
    members: list[Any] = []
    if make_member is not None:
        for b in range(ensemble_size):
            idx = bootstrap_rows(groups_fit, len(p_fit), rng)
            X_b = X_fit.iloc[idx] if X_fit is not None else None
            members.append(make_member(seed + 1 + b).fit(X_b, p_fit[idx], y_fit[idx]))
    um = UncertaintyModel(members, calibrator, families, calibration_sd(p_model_cal, y_cal),
                          z_value(nominal_coverage), nominal_coverage)
    comps = um.components(X_cal, p_cal, p_model_cal)
    um.scale = tune_scale(p_model_cal, y_cal, comps["sd_ensemble"], comps["sd_calibration"],
                          comps["missing_share"], um.z, nominal_coverage, min_bin_obs)
    lower, upper = interval(p_model_cal, comps["sd_ensemble"], comps["sd_calibration"],
                            comps["missing_share"], um.scale, um.z)
    um.diagnostics = {
        "proxy_coverage_cal": proxy_coverage(p_model_cal, y_cal, lower, upper, min_bin_obs),
        "proxy_n_bins": proxy_n_bins(len(p_model_cal), min_bin_obs),
        "scale_at_bound": um.scale in SCALE_BOUNDS,
        "ensemble_size": len(members),
        "mean_sd_ensemble_cal": float(comps["sd_ensemble"].mean()),
        "mean_sd_calibration_cal": float(comps["sd_calibration"].mean()),
        "mean_missing_share_cal": float(comps["missing_share"].mean()),
        "mean_width_cal": float(np.mean(upper - lower)),
    }
    return um
