"""Calibration methods, CV-based selection and game-boundary splits (spec Sec 7 Layer C, 26.13)."""

from __future__ import annotations

import numpy as np
import pytest

from src.pricing import EPS, expit, logit
from src.models.calibration import (
    choose_calibration,
    clustered_se,
    contiguous_folds,
    cross_val_calibrated,
    fit_calibrator,
    game_codes,
    tail_split,
)


def outcomes(n: int, seed: int, sharpen: float = 1.0):
    """p_true ~ expit(N(0, 0.8)); p_raw = expit(sharpen * logit p_true); two rows per game."""
    rng = np.random.default_rng(seed)
    p_true = expit(rng.normal(0, 0.8, n))
    y = (rng.random(n) < p_true).astype(int)
    groups = np.repeat(np.arange((n + 1) // 2), 2)[:n].astype(str)
    return expit(sharpen * logit(p_true)), y, groups


def brier(p, y) -> float:
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_tail_split_takes_the_latest_rows_on_a_game_boundary():
    groups = np.array(["a", "a", "b", "b", "b", "c", "c", "d", "d", "e"])
    cal = tail_split(groups, len(groups), 0.25)
    assert cal.tolist() == [False] * 7 + [True] * 3  # nearest boundary to 7.5 rows: after c
    for g in np.unique(groups):
        assert len(set(cal[groups == g])) == 1  # no game straddles
    with pytest.raises(ValueError):
        tail_split(["a", "a", "a"], 3, 0.25)


def test_a_straddling_game_goes_wholly_to_the_earlier_side():
    groups = np.array(["a", "b", "a", "c", "d"])  # 'a' rows interleave with 'b'
    cal = tail_split(groups, 5, 0.5)
    assert not cal[groups == "a"].any()


def test_contiguous_folds_are_time_blocks_of_whole_games():
    groups = np.repeat(np.arange(50), 2).astype(str)
    folds = contiguous_folds(groups, 100, 5)
    assert np.all(np.diff(folds) >= 0)  # contiguous in time order
    assert np.bincount(folds).tolist() == [20] * 5
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    assert game_codes(["x", "y", "x"], 3).tolist() == [0, 1, 0]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def test_platt_repairs_overconfident_probabilities():
    p_raw, y, groups = outcomes(20_000, seed=1, sharpen=2.0)
    fit, test = slice(0, 10_000), slice(10_000, None)
    cal = fit_calibrator("platt", p_raw[fit], y[fit])
    assert cal.b == pytest.approx(0.5, abs=0.05)
    assert abs(cal.a) < 0.05
    assert brier(cal.transform(p_raw[test]), y[test]) < brier(p_raw[test], y[test]) - 0.005
    method, _, scores = choose_calibration(p_raw[fit], y[fit], groups=groups[fit])
    assert method != "none"
    assert scores["platt"] < scores["none"] - 0.005


def test_none_is_chosen_for_well_calibrated_input():
    for seed in (2, 3, 4):
        p_raw, y, groups = outcomes(3_000, seed=seed)
        method, cal, scores = choose_calibration(p_raw, y, groups=groups)
        assert method == "none", (seed, scores, cal.cv_se)
        assert np.array_equal(cal.transform(p_raw), p_raw)


def test_cv_scoring_does_not_reward_isotonic_for_fitting_its_own_data():
    p_raw, y, groups = outcomes(3_000, seed=5)
    in_sample = brier(fit_calibrator("isotonic", p_raw, y).transform(p_raw), y)
    assert in_sample < brier(p_raw, y)  # in sample, isotonic always looks better ...
    _, _, scores = choose_calibration(p_raw, y, groups=groups)
    assert scores["isotonic"] > scores["none"]  # ... out of fold it does not
    oof = cross_val_calibrated("isotonic", p_raw, y, groups=groups)
    assert scores["isotonic"] == pytest.approx(brier(oof, y))


def test_tie_rule_prefers_the_simpler_method():
    p_raw, y, groups = outcomes(4_000, seed=6, sharpen=1.3)
    strict, _, scores = choose_calibration(p_raw, y, groups=groups, tie_se=0.0)
    assert strict == min(scores, key=scores.get)
    assert choose_calibration(p_raw, y, groups=groups, tie_se=1e9)[0] == "none"
    assert choose_calibration(p_raw, y, ["platt", "isotonic"], groups=groups, tie_se=1e9)[0] == "platt"


@pytest.mark.parametrize("method", ["none", "platt", "isotonic"])
def test_transform_is_monotone_and_bounded(method):
    p_raw, y, _ = outcomes(2_000, seed=7, sharpen=1.5)
    cal = fit_calibrator(method, p_raw, y)
    grid = np.array([0.0, 1e-12, 1e-4, 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99, 1 - 1e-12, 1.0])
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= EPS and out.max() <= 1 - EPS
    assert cal.method == method


def test_unknown_method_is_rejected():
    p_raw, y, _ = outcomes(100, seed=8)
    with pytest.raises(ValueError):
        choose_calibration(p_raw, y, ["none", "beta"])


def test_clustered_se_counts_games_not_rows():
    rng = np.random.default_rng(9)
    per_game = rng.normal(size=500)
    d = np.repeat(per_game, 2)  # both rows of a game share one value
    codes = np.repeat(np.arange(500), 2)
    naive = d.std(ddof=1) / np.sqrt(len(d))
    game_level = per_game.std(ddof=1) / np.sqrt(500)
    assert clustered_se(d, codes) == pytest.approx(game_level, rel=1e-9)
    assert clustered_se(d, codes) > 1.3 * naive
