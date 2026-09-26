"""Uncertainty intervals: components, proxy-tuned scale, coverage (spec Sec 13, 26.12)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_sport_config
from src.pricing import expit, logit
from src.models.residual import fit_model
from src.models.uncertainty import (
    SD_CAL_LOGIT_MAX,
    bootstrap_rows,
    calibration_sd,
    calibration_sd_logit,
    interval,
    missing_share,
    proxy_coverage,
    proxy_n_bins,
    proxy_targets,
    tune_scale,
    z_value,
)

FAMILIES = {"starter_diff": "starter", "offense_diff": "offense", "bullpen_diff": "bullpen",
            "park": "environment", "noise": "negative_control"}
FEATURES = list(FAMILIES)


def simulate_games(n_games: int, seed: int, miss: float = 0.15) -> pd.DataFrame:
    """Two mirrored rows per game. The market is calibrated on its own information and
    misses r(features); p_true = expit(logit p_market + r) = E[y | market, features]."""
    rng = np.random.default_rng(seed)
    x = {k: rng.normal(size=n_games) for k in FAMILIES}
    z_mkt = 0.08 + rng.normal(0, 0.45, n_games)
    z_true = z_mkt + 0.30 * x["starter_diff"] + 0.20 * x["offense_diff"] + 0.10 * x["bullpen_diff"]
    y = (rng.random(n_games) < expit(z_true)).astype(int)
    X = pd.DataFrame(x)
    X.loc[rng.random(n_games) < miss, "starter_diff"] = np.nan
    game = np.arange(n_games)
    home = X.assign(p=expit(z_mkt), y=y, p_true=expit(z_true), game=game, side=0)
    away = (-X).assign(p=1 - expit(z_mkt), y=1 - y, p_true=1 - expit(z_true), game=game, side=1)
    df = pd.concat([home, away]).sort_values(["game", "side"]).reset_index(drop=True)
    df["game_id"] = "g" + df["game"].astype(str)
    return df


def assert_prediction_invariants(out: pd.DataFrame) -> None:
    """The PREDICTIONS table checks (src/storage/schemas.py), evaluated in pandas."""
    for col in ("p_market", "p_model_raw", "p_model"):
        assert out[col].between(0, 1).all(), col
    assert (out.lower_bound >= 0).all() and (out.upper_bound <= 1).all()
    assert (out.lower_bound <= out.p_model).all() and (out.p_model <= out.upper_bound).all()
    assert (out.p_conservative_yes == out.lower_bound).all()
    assert ((out.p_conservative_no - (1 - out.upper_bound)).abs() < 1e-12).all()


# ---------------------------------------------------------------------------
# The interval as a pure function
# ---------------------------------------------------------------------------

def test_z_for_nominal_coverage():
    assert z_value(0.90) == pytest.approx(1.644854, abs=1e-6)
    assert z_value(0.95) == pytest.approx(1.959964, abs=1e-6)


def test_interval_contains_p_model_and_stays_in_bounds():
    p = np.array([1e-6, 0.001, 0.02, 0.3, 0.5, 0.7, 0.98, 0.999, 1 - 1e-6])
    for sd in (0.0, 0.05, 1.0, 50.0):
        lo, hi = interval(p, sd, sd, 0.5, 1.0, 1.645)
        assert np.all((0 <= lo) & (lo <= p) & (p <= hi) & (hi <= 1))
    lo, hi = interval(p, 0.0, 0.0, 0.0, 1.0, 1.645)
    assert np.array_equal(lo, p) and np.array_equal(hi, p)


def test_interval_is_symmetric_in_logit_space_and_matches_the_formula():
    lo, hi = interval(np.array([0.6]), 0.3, 0.4, 0.25, 2.0, 1.645)
    half = 1.645 * 2.0 * np.hypot(0.3, 0.4) * 1.25
    assert logit(hi)[0] - logit(0.6) == pytest.approx(half)
    assert logit(0.6) - logit(lo)[0] == pytest.approx(half)


def test_interval_widens_with_missing_share_and_ensemble_spread():
    p = np.full(5, 0.55)
    ms = np.linspace(0, 1, 5)
    lo, hi = interval(p, 0.1, 0.05, ms, 1.0, 1.645)
    assert np.all(np.diff(hi - lo) > 0)
    ens = np.linspace(0, 0.4, 5)
    lo, hi = interval(p, ens, 0.05, 0.0, 1.0, 1.645)
    assert np.all(np.diff(hi - lo) > 0)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def test_calibration_sd_separates_calibrated_from_miscalibrated():
    rng = np.random.default_rng(1)
    p_true = expit(rng.normal(0, 0.6, 3_000))
    y = (rng.random(3_000) < p_true).astype(int)
    floor = np.sqrt(np.sum(p_true * (1 - p_true))) / 3_000  # sampling term alone
    calibrated = calibration_sd(p_true, y)
    overconfident = calibration_sd(expit(2 * logit(p_true)), y)
    assert calibrated < 2.5 * floor
    assert overconfident > 5 * floor
    assert calibration_sd(np.full(3_000, 0.5), (np.arange(3_000) % 2)) == pytest.approx(
        np.sqrt(0.25 / 3_000), rel=1e-9)  # zero excess: width never collapses to 0


def test_calibration_sd_delta_method_and_cap():
    p = np.array([0.5, 0.2, 1e-4])
    out = calibration_sd_logit(0.02, p)
    assert out[0] == pytest.approx(0.02 / 0.25)
    assert out[1] == pytest.approx(0.02 / 0.16)
    assert out[2] == SD_CAL_LOGIT_MAX


def test_missing_share_counts_whole_families_and_skips_market():
    fam = {"sp_era": "starter", "sp_k": "starter", "off": "offense", "mkt_move": "market"}
    X = pd.DataFrame({
        "sp_era": [1.0, np.nan, np.nan, np.nan],
        "sp_k":   [1.0, 2.0,    np.nan, np.nan],
        "off":    [0.1, 0.2,    0.3,    np.nan],
        "mkt_move": [np.nan] * 4,
    })
    assert missing_share(X, fam).tolist() == [0.0, 0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="family"):
        missing_share(X.assign(extra=1.0), fam)


def test_bootstrap_resamples_whole_games():
    groups = np.repeat([f"g{i}" for i in range(300)], [2, 3] * 150)
    idx = bootstrap_rows(groups, len(groups), np.random.default_rng(2))
    drawn = pd.Series(groups[idx]).value_counts()
    sizes = pd.Series(groups).value_counts()
    assert all(drawn[g] % sizes[g] == 0 for g in drawn.index)  # a game comes in whole copies
    assert drawn.index.size < 300  # with replacement: some games are left out


# ---------------------------------------------------------------------------
# Scale tuning on the binned frequency proxy
# ---------------------------------------------------------------------------

def test_proxy_bins_follow_min_obs_with_a_floor_of_ten():
    assert proxy_n_bins(3_000, 500) == 10
    assert proxy_n_bins(50_000, 500) == 100
    assert proxy_n_bins(7, 500) == 7
    p = np.linspace(0.1, 0.9, 1_000)
    y = (np.arange(1_000) % 3 == 0).astype(int)
    t = proxy_targets(p, y, 500)
    assert len(np.unique(t)) <= 10
    assert t[:100].mean() == pytest.approx(y[:100].mean())


def test_tune_scale_hits_nominal_proxy_coverage():
    rng = np.random.default_rng(3)
    n = 3_000
    p_model = expit(rng.normal(0, 0.5, n))
    y = (rng.random(n) < expit(logit(p_model) + rng.normal(0, 0.3, n))).astype(int)
    ens, cal, ms = rng.uniform(0.02, 0.2, n), calibration_sd_logit(0.01, p_model), rng.choice([0, 0.2], n)
    z = z_value(0.90)
    scale = tune_scale(p_model, y, ens, cal, ms, z, 0.90, 500)
    lo, hi = interval(p_model, ens, cal, ms, scale, z)
    assert proxy_coverage(p_model, y, lo, hi, 500) == pytest.approx(0.90, abs=0.01)
    lo, hi = interval(p_model, ens, cal, ms, scale * 0.9, z)
    assert proxy_coverage(p_model, y, lo, hi, 500) < 0.90  # the smallest such scale


@pytest.fixture(scope="module")
def coverage_run():
    cfg = load_sport_config("mlb")
    train, test = simulate_games(6_000, 41), simulate_games(4_000, 42)
    model = fit_model("logistic_residual", train[FEATURES], train.p, train.y, train.game_id,
                      FAMILIES, cfg, seed=5)
    return model, test, model.predict(test[FEATURES], test.p)


def test_proxy_tuned_interval_covers_true_probability(coverage_run):
    model, test, out = coverage_run
    u = model.summary()["uncertainty"]
    assert u["proxy_coverage_cal"] == pytest.approx(0.90, abs=0.01)
    assert not u["scale_at_bound"]
    covered = (out.lower_bound <= test.p_true) & (test.p_true <= out.upper_bound)
    print(f"proxy coverage (cal) {u['proxy_coverage_cal']:.4f}; p_true coverage (held out) {covered.mean():.4f};"
          f" scale {u['scale']:.3f}; mean width {np.mean(out.upper_bound - out.lower_bound):.4f}")
    # observed 0.9337 with these seeds; the nominal band is 0.87-0.93 (26.12)
    assert 0.85 <= covered.mean() <= 0.97
    # the missing-information penalty applies exactly to rows whose starter is unknown (1 of 5 families)
    share = out.uncertainty_json.map(lambda s: json.loads(s)["missing_share"])
    assert np.array_equal(share.to_numpy(), np.where(test.starter_diff.isna(), 0.2, 0.0))


def test_invariants_hold_at_extreme_probabilities(coverage_run):
    model, test, _ = coverage_run
    X = test[FEATURES].head(6)
    p = np.array([0.001, 0.999, 1e-5, 1 - 1e-5, 0.5, 0.001])
    out = model.predict(X, p)
    assert_prediction_invariants(out)
    assert (out.upper_bound - out.lower_bound).gt(0).all()
