"""Layer B residual models and the fit/predict unit (spec Sec 7, 13, 26.12, 26.13).

    z_fair = logit(p_market) + f(X),     p_model_raw = expit(z_fair)

The market logit is a fixed offset (coefficient 1, never learned), so with no
signal in X the model collapses to the market. Two residual learners:

  ResidualLogistic  intercept + L2-penalised linear f on standardized features,
                    each first clipped to its training-window quantiles
                    (model.logistic.winsorize) so a value outside the range the
                    window saw is not extrapolated linearly; mean-imputed, with
                    a 0/1 missing indicator for each feature that had NaNs in
                    training. Objective
                        sum_i NLL_i + (l2 / 2) * ||beta||^2      (b0 unpenalised)
                    i.e. sklearn's C = 1 / l2; fit by L-BFGS-B with the
                    analytic gradient (sklearn has no offset).
  ResidualLGBM      gradient-boosted trees with init_score = logit(p_market);
                    NaN handled natively; deterministic, fixed seed.

`fit_model` is the unit the walk-forward orchestration calls once per training
window. On time-ordered rows it:
  1. splits on a game boundary into fit (earliest 1 - calibration_fraction) and
     calibration (latest calibration_fraction);
  2. fits the base model on the fit split. When the config gives a complexity
     grid (model.logistic.l2_grid, model.lgbm.n_estimators_grid) each value is
     fitted and scored on the calibration split, and the simplest value (most
     penalty / fewest trees) within model.complexity_tie_se game-clustered SEs
     of the best Brier wins - chosen inside the training window, never on the
     period being predicted;
  3. predicts the calibration split (out of sample for the base model) and
     chooses + fits the calibrator there (calibration.py);
  4. fits the uncertainty model (uncertainty.py): bootstrap refits on the fit
     split; calibration component and interval scale from the calibration
     split's out-of-fold calibrated predictions.
  5. does NOT refit the base model on the full window: the calibrator and the
     interval scale were estimated for this exact fitted model, and a refit on
     more data would be a different (sharper) model they were never checked
     against. The cost is that the latest calibration_fraction of each window
     only informs calibration and uncertainty, not the residual.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import stable_hash
from ..pricing import expit, logit
from .baseline import MarketBaseline, as_outcome, as_prob
from .calibration import (
    METHODS,
    Calibrator,
    choose_calibration,
    clustered_se,
    cross_val_calibrated,
    fit_calibrator,
    game_codes,
    tail_split,
)
from .uncertainty import UncertaintyModel, fit_uncertainty

MODEL_TYPES = ("market", "logistic_residual", "lgbm_residual")


def align_features(X: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """X's columns reordered to `names` as a float matrix; any other column set is an error."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    cols = list(X.columns)
    if len(set(cols)) != len(cols):
        raise ValueError("X has duplicate column names")
    unseen, missing = sorted(set(cols) - set(names)), sorted(set(names) - set(cols))
    if unseen or missing:
        raise ValueError(f"feature mismatch: unseen {unseen}, missing {missing}")
    return X[list(names)].to_numpy(dtype=float)


def _check_rows(X: pd.DataFrame, p_market: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    p = as_prob(p_market)
    if len(X) != len(p):
        raise ValueError(f"X has {len(X)} rows, p_market {len(p)}")
    return p, (None if y is None else as_outcome(y, len(p)))


# ---------------------------------------------------------------------------
# Regularised logistic regression with a fixed market offset
# ---------------------------------------------------------------------------

def offset_logistic_loss(theta: np.ndarray, Z: np.ndarray, offset: np.ndarray, y: np.ndarray,
                         l2: float) -> tuple[float, np.ndarray]:
    """(objective, gradient) / n for theta = [b0, beta...]; the penalty skips b0."""
    b0, beta = theta[0], theta[1:]
    eta = offset + b0 + Z @ beta
    resid = expit(eta) - y
    n = len(y)
    loss = np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * l2 * beta @ beta
    grad = np.concatenate(([resid.sum()], Z.T @ resid + l2 * beta))
    return loss / n, grad / n


class ResidualLogistic:
    model_type = "logistic_residual"

    def __init__(self, l2: float = 1.0, winsorize: Sequence[float] | None = None) -> None:
        self.l2 = float(l2)
        self.winsorize = None if winsorize is None else (float(winsorize[0]), float(winsorize[1]))

    def fit(self, X: pd.DataFrame, p_market: Any, y: Any) -> ResidualLogistic:
        p, y = _check_rows(X, p_market, y)
        self.feature_names_ = list(X.columns)
        A = align_features(X, self.feature_names_)
        nan_any, nan_all = np.isnan(A).any(axis=0), np.isnan(A).all(axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            if self.winsorize is not None:
                lo, hi = np.nanquantile(A, self.winsorize[0], axis=0), np.nanquantile(A, self.winsorize[1], axis=0)
            else:
                lo, hi = np.full(A.shape[1], -np.inf), np.full(A.shape[1], np.inf)
            C = np.clip(A, lo, hi)
            mean, std = np.nanmean(C, axis=0), np.nanstd(C, axis=0)
        keep = np.isfinite(std) & (std > 0)
        self.kept_ = [f for f, k in zip(self.feature_names_, keep) if k]
        self.mean_, self.std_ = mean[keep], std[keep]
        self.lo_, self.hi_ = lo[keep], hi[keep]
        self.missing_ = [f for f, a, b in zip(self.feature_names_, nan_any, nan_all) if a and not b]
        Z = self._design(A)
        res = minimize(offset_logistic_loss, np.zeros(Z.shape[1] + 1), args=(Z, logit(p), y, self.l2),
                       jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-9})
        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:1 + len(self.kept_)]
        self.missing_coef_ = res.x[1 + len(self.kept_):]
        self.converged_ = bool(res.success)
        return self

    def _design(self, A: np.ndarray) -> np.ndarray:
        pos = {f: i for i, f in enumerate(self.feature_names_)}
        vals = (np.clip(A[:, [pos[f] for f in self.kept_]], self.lo_, self.hi_) - self.mean_) / self.std_
        vals = np.nan_to_num(vals, nan=0.0)
        miss = np.isnan(A[:, [pos[f] for f in self.missing_]]).astype(float)
        return np.hstack([vals, miss])

    def _terms(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row logit contribution of each original feature (indicator folded in)."""
        Z = self._design(align_features(X, self.feature_names_))
        k = len(self.kept_)
        out = np.zeros((len(X), len(self.feature_names_)))
        pos = {f: i for i, f in enumerate(self.feature_names_)}
        for j, f in enumerate(self.kept_):
            out[:, pos[f]] += Z[:, j] * self.coef_[j]
        for j, f in enumerate(self.missing_):
            out[:, pos[f]] += Z[:, k + j] * self.missing_coef_[j]
        return out

    def predict_raw(self, X: pd.DataFrame, p_market: Any) -> np.ndarray:
        p, _ = _check_rows(X, p_market)
        return expit(logit(p) + self.intercept_ + self._terms(X).sum(axis=1))

    def contributions(self, X: pd.DataFrame, p_market: Any) -> pd.DataFrame:
        p, _ = _check_rows(X, p_market)
        out = pd.DataFrame(self._terms(X), columns=self.feature_names_, index=X.index)
        out["market"] = logit(p)
        out["intercept"] = self.intercept_
        return out

    def params(self) -> dict[str, Any]:
        """Coefficients per unit of the original feature (intercept at x = 0, all present)."""
        per_unit = dict(zip(self.kept_, (self.coef_ / self.std_).tolist()))
        return {
            "l2": self.l2,
            "winsorize": list(self.winsorize) if self.winsorize else None,
            "clip": {f: [float(a), float(b)] for f, a, b in zip(self.kept_, self.lo_, self.hi_)},
            "intercept": float(self.intercept_ - np.sum(self.coef_ * self.mean_ / self.std_)),
            "intercept_standardized": self.intercept_,
            "coef": {f: per_unit.get(f, 0.0) for f in self.feature_names_},
            "coef_standardized": dict(zip(self.kept_, self.coef_.tolist())),
            "missing_coef": dict(zip(self.missing_, self.missing_coef_.tolist())),
            "dropped_constant": [f for f in self.feature_names_ if f not in self.kept_],
            "converged": self.converged_,
        }


# ---------------------------------------------------------------------------
# LightGBM residual with init_score = logit(p_market)
# ---------------------------------------------------------------------------

_LGB_ALIASES = {"min_child_samples": "min_data_in_leaf", "subsample": "bagging_fraction",
                "subsample_freq": "bagging_freq", "colsample_bytree": "feature_fraction",
                "reg_lambda": "lambda_l2", "reg_alpha": "lambda_l1"}


class ResidualLGBM:
    model_type = "lgbm_residual"

    def __init__(self, *, seed: int = 0, num_threads: int = 4, **params: Any) -> None:
        self.seed, self.num_threads, self.config = int(seed), int(num_threads), dict(params)

    def _train_params(self) -> tuple[dict[str, Any], int]:
        cfg = dict(self.config)
        rounds = int(cfg.pop("n_estimators", cfg.pop("num_iterations", 100)))
        params = {_LGB_ALIASES.get(k, k): v for k, v in cfg.items()}
        params |= {"objective": "binary", "boost_from_average": False, "deterministic": True,
                   "force_row_wise": True, "seed": self.seed, "num_threads": self.num_threads,
                   "verbosity": -1}
        return params, rounds

    def fit(self, X: pd.DataFrame, p_market: Any, y: Any) -> ResidualLGBM:
        p, y = _check_rows(X, p_market, y)
        self.feature_names_ = list(X.columns)
        params, rounds = self._train_params()
        data = lgb.Dataset(align_features(X, self.feature_names_), label=y, init_score=logit(p),
                           params={"verbosity": -1})
        self.booster_ = lgb.train(params, data, num_boost_round=rounds)
        return self

    def _raw(self, X: pd.DataFrame, **kw: Any) -> np.ndarray:
        return self.booster_.predict(align_features(X, self.feature_names_), **kw)

    def predict_raw(self, X: pd.DataFrame, p_market: Any) -> np.ndarray:
        p, _ = _check_rows(X, p_market)
        return expit(logit(p) + self._raw(X, raw_score=True))

    def contributions(self, X: pd.DataFrame, p_market: Any) -> pd.DataFrame:
        p, _ = _check_rows(X, p_market)
        c = self._raw(X, pred_contrib=True)
        out = pd.DataFrame(c[:, :-1], columns=self.feature_names_, index=X.index)
        out["market"] = logit(p)
        out["intercept"] = c[:, -1]
        return out

    def params(self) -> dict[str, Any]:
        gain = self.booster_.feature_importance(importance_type="gain")
        return {"config": self.config, "seed": self.seed, "num_trees": self.booster_.num_trees(),
                "gain": dict(zip(self.feature_names_, gain.tolist()))}


# ---------------------------------------------------------------------------
# The fit/predict unit
# ---------------------------------------------------------------------------

def _base_factories(model_type: str, cfg: dict[str, Any], num_threads: int) -> tuple[str | None, list]:
    """(grid parameter name, [(value, factory), ...]) ordered simplest first."""
    mcfg = cfg.get("model") or {}
    if model_type == "market":
        return None, [(None, lambda seed: MarketBaseline())]
    if model_type == "logistic_residual":
        lc = mcfg.get("logistic") or {}
        grid = sorted({float(v) for v in lc.get("l2_grid") or [lc.get("l2", 1.0)]}, reverse=True)
        win = lc.get("winsorize")
        return "l2", [(v, (lambda v: lambda seed: ResidualLogistic(l2=v, winsorize=win))(v)) for v in grid]
    if model_type == "lgbm_residual":
        params = dict(mcfg.get("lgbm") or {})
        grid = sorted({int(v) for v in params.pop("n_estimators_grid", None) or [params.get("n_estimators", 100)]})
        return "n_estimators", [(v, (lambda v: lambda seed: ResidualLGBM(
            seed=seed, num_threads=num_threads, **(params | {"n_estimators": v})))(v)) for v in grid]
    raise ValueError(f"model_type must be one of {MODEL_TYPES}")


def choose_complexity(options: list, X_fit: pd.DataFrame | None, p_fit: np.ndarray, y_fit: np.ndarray,
                      X_cal: pd.DataFrame | None, p_cal: np.ndarray, y_cal: np.ndarray, cal_groups: np.ndarray,
                      *, seed: int, tie_se: float) -> tuple[int, Any, dict[str, Any]]:
    """Index of the simplest option within `tie_se` clustered SEs of the best calibration-split
    Brier, its fitted base model, and the scores. `options` is ordered simplest first."""
    fitted = [make(seed).fit(X_fit, p_fit, y_fit) for _, make in options]
    losses = [(m.predict_raw(X_cal, p_cal) - y_cal) ** 2 for m in fitted]
    scores = [float(l.mean()) for l in losses]
    best = int(np.argmin(scores))
    codes = game_codes(cal_groups, len(y_cal))
    se = [clustered_se(l - losses[best], codes) for l in losses]
    chosen = next(i for i in range(len(options)) if scores[i] <= scores[best] + tie_se * se[i])
    return chosen, fitted[chosen], {"scores": scores, "se": se, "best": best}


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


@dataclass
class FittedModel:
    model_type: str
    base: Any
    calibrator: Calibrator
    uncertainty: UncertaintyModel
    features: list[str]
    info: dict[str, Any] = field(default_factory=dict)

    def _X(self, X: pd.DataFrame | None) -> pd.DataFrame | None:
        if self.model_type == "market":
            return None
        if X is None:
            raise ValueError("this model needs features")
        return X

    def predict(self, X: pd.DataFrame | None, p_market: Any) -> pd.DataFrame:
        """One row per input row; satisfies the PREDICTIONS table checks."""
        p = as_prob(p_market)
        Xa = self._X(X)
        p_raw = self.base.predict_raw(Xa, p)
        p_model = self.calibrator.transform(p_raw)
        lower, upper, comps = self.uncertainty.bounds(Xa, p, p_model)
        index = X.index if isinstance(X, pd.DataFrame) and len(X) == len(p) else None
        return pd.DataFrame({
            "p_market": p, "p_model_raw": p_raw, "p_model": p_model,
            "lower_bound": lower, "upper_bound": upper,
            "p_conservative_yes": lower, "p_conservative_no": 1.0 - upper,
            "uncertainty_json": self.uncertainty.component_json(comps),
        }, index=index)

    def contributions(self, X: pd.DataFrame | None, p_market: Any) -> pd.DataFrame:
        """Logit contributions of the raw model plus "calibration"; each row sums to logit(p_model)."""
        p = as_prob(p_market)
        Xa = self._X(X)
        out = self.base.contributions(Xa, p)
        if Xa is None and isinstance(X, pd.DataFrame) and len(X) == len(p):
            out.index = X.index
        p_raw = self.base.predict_raw(Xa, p)
        out["calibration"] = logit(self.calibrator.transform(p_raw)) - logit(p_raw)
        return out

    def summary(self) -> dict[str, Any]:
        return dict(self.info)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(path: str | Path) -> FittedModel:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, FittedModel):
            raise TypeError(f"{path} does not hold a FittedModel")
        return obj


def fit_model(model_type: str, X: pd.DataFrame | None, p_market: Any, y: Any,
              groups: Sequence, families: dict[str, str] | None, cfg: dict[str, Any], *,
              seed: int, num_threads: int = 4) -> FittedModel:
    """Fit base model, calibrator and uncertainty model on one time-ordered training window.

    `cfg` is the sport config (sections model, uncertainty, review are read).
    """
    grid_param, options = _base_factories(model_type, cfg, num_threads)
    mcfg, ucfg, rcfg = cfg.get("model") or {}, cfg.get("uncertainty") or {}, cfg.get("review") or {}
    is_market = model_type == "market"
    p = as_prob(p_market)
    y = as_outcome(y, len(p))
    if is_market:
        X, features, families = None, [], None
    else:
        if X is None or len(X) != len(p):
            raise ValueError("X must be a DataFrame with one row per p_market entry")
        features = list(X.columns)
        if families is not None:
            unknown = [c for c in features if c not in families]
            if unknown:
                raise ValueError(f"features without a family: {unknown}")
            families = {c: families[c] for c in features}
    groups = np.asarray(groups)
    cal = tail_split(groups, len(p), float(mcfg.get("calibration_fraction", 0.25)))
    fit = ~cal
    X_fit = X.iloc[np.flatnonzero(fit)] if X is not None else None
    X_cal = X.iloc[np.flatnonzero(cal)] if X is not None else None

    if len(options) > 1:
        k, base, sel = choose_complexity(options, X_fit, p[fit], y[fit], X_cal, p[cal], y[cal], groups[cal],
                                         seed=seed, tie_se=float(mcfg.get("complexity_tie_se", 1.0)))
        complexity = {"param": grid_param, "grid": [v for v, _ in options], "chosen": options[k][0],
                      "brier_raw_cal": dict(zip(map(str, (v for v, _ in options)), sel["scores"])),
                      "se_vs_best": dict(zip(map(str, (v for v, _ in options)), sel["se"]))}
    else:
        k, complexity = 0, ({"param": grid_param, "grid": [options[0][0]], "chosen": options[0][0]}
                            if grid_param else None)
        base = options[0][1](seed).fit(X_fit, p[fit], y[fit])
    make_base = options[k][1]
    p_raw_cal = base.predict_raw(X_cal, p[cal])

    if is_market:
        method, calibrator, scores = "none", fit_calibrator("none", p_raw_cal, y[cal]), {}
        p_model_cal = calibrator.transform(p_raw_cal)
    else:
        tie_se = float(mcfg.get("calibration_tie_se", 1.0))
        method, calibrator, scores = choose_calibration(
            p_raw_cal, y[cal], mcfg.get("calibration_methods", METHODS), groups=groups[cal], tie_se=tie_se)
        p_model_cal = cross_val_calibrated(method, p_raw_cal, y[cal], groups=groups[cal])

    nominal = float(ucfg.get("nominal_coverage", 0.90))
    min_bin_obs = int(rcfg.get("coverage_bin_min_obs", 500))
    unc = fit_uncertainty(
        None if is_market else make_base, X_fit, p[fit], y[fit], groups[fit],
        X_cal, p[cal], y[cal], p_model_cal, calibrator, families,
        ensemble_size=int(ucfg.get("ensemble_size", 20)), nominal_coverage=nominal,
        min_bin_obs=min_bin_obs, seed=seed)

    config_hash = stable_hash({"model_type": model_type, "model": mcfg, "uncertainty": ucfg,
                               "coverage_bin_min_obs": min_bin_obs, "seed": seed})
    info = {
        "model_type": model_type,
        "n_fit_obs": int(fit.sum()), "n_cal_obs": int(cal.sum()),
        "n_games": int(game_codes(groups, len(p)).max() + 1),
        "n_fit_games": int(len(np.unique(groups[fit]))), "n_cal_games": int(len(np.unique(groups[cal]))),
        "calibration_method": method,
        "calibration_scores": scores,
        "calibration_se": dict(calibrator.cv_se),
        "calibration_params": calibrator.params(),
        "uncertainty": {"scale": unc.scale, "z": unc.z, "nominal_coverage": nominal,
                        "sd_calibration_p": unc.sd_calibration_p, **unc.diagnostics},
        "features": features,
        "params": base.params(),
        "complexity": complexity,
        "config_hash": config_hash,
        "seed": seed,
        "brier_model_cal": _brier(p_model_cal, y[cal]),     # out-of-fold calibrated
        "brier_raw_cal": _brier(p_raw_cal, y[cal]),         # out of sample for the base model
        "brier_market_cal": _brier(p[cal], y[cal]),
        "refit_on_full_window": False,
    }
    return FittedModel(model_type, base, calibrator, unc, features, info)
