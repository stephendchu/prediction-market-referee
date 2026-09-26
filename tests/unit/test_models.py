"""Residual models and the fit/predict unit (spec Sec 7, 13, 26.12)."""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.config import load_sport_config
from src.pricing import expit, logit
from src.models.baseline import MarketBaseline
from src.models.residual import (
    FittedModel,
    ResidualLGBM,
    ResidualLogistic,
    fit_model,
    offset_logistic_loss,
)

FAMILIES = {"starter_diff": "starter", "offense_diff": "offense", "bullpen_diff": "bullpen",
            "park": "environment", "noise": "negative_control"}
FEATURES = list(FAMILIES)


def simulate_games(n_games: int, seed: int, *, signal: str = "linear", miss: float = 0.15) -> pd.DataFrame:
    """Two mirrored rows per game (home, away), time-ordered. The market is calibrated on
    its own information and misses a residual r(features); p_true = expit(logit p + r)."""
    rng = np.random.default_rng(seed)
    x = {k: rng.normal(size=n_games) for k in FAMILIES}
    z_mkt = 0.08 + rng.normal(0, 0.45, n_games)
    if signal == "linear":
        r = 0.30 * x["starter_diff"] + 0.20 * x["offense_diff"] + 0.10 * x["bullpen_diff"]
    elif signal == "interaction":  # odd in x, so the mirrored away row is consistent; no linear part
        r = 0.45 * (x["starter_diff"] ** 2 - 1) * x["offense_diff"]
    else:
        r = np.zeros(n_games)
    z_true = z_mkt + r
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
    assert out.notna().all().all()


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def planted(n: int, seed: int, beta: float = 0.5, missing: float = 0.0, miss_effect: float = 0.0):
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(0, 0.6, n))
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    miss = rng.random(n) < missing
    eta = logit(p) + beta * np.where(miss, 0.0, x1) + miss_effect * miss
    y = (rng.random(n) < expit(eta)).astype(int)
    X = pd.DataFrame({"x1": np.where(miss, np.nan, x1), "x2": x2})
    return X, p, y


# ---------------------------------------------------------------------------
# Market baseline
# ---------------------------------------------------------------------------

def test_market_baseline_returns_the_market():
    p = np.array([0.2, 0.55, 0.9])
    m = MarketBaseline().fit(None, p, [0, 1, 1])
    assert np.array_equal(m.predict_raw(None, p), p)
    with pytest.raises(ValueError):
        m.predict_raw(None, [0.0, 0.5])


# ---------------------------------------------------------------------------
# ResidualLogistic
# ---------------------------------------------------------------------------

def test_logistic_recovers_planted_residual_coefficient():
    X, p, y = planted(20_000, seed=1, beta=0.5)
    prm = ResidualLogistic(l2=1.0).fit(X, p, y).params()
    assert prm["coef"]["x1"] == pytest.approx(0.5, abs=0.06)
    assert abs(prm["coef"]["x2"]) < 0.05
    assert abs(prm["intercept"]) < 0.05
    assert prm["converged"]


def test_logistic_without_signal_collapses_to_the_market():
    X, p, y = planted(20_000, seed=2, beta=0.0)
    m = ResidualLogistic().fit(X, p, y)
    pred = m.predict_raw(X, p)
    assert np.max(np.abs(logit(pred) - logit(p))) < 0.1
    assert np.max(np.abs(pred - p)) < 0.02
    assert all(abs(c) < 0.05 for c in m.params()["coef"].values())


def test_offset_logistic_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    Z, off = rng.normal(size=(200, 4)), rng.normal(size=200)
    y = (rng.random(200) < 0.4).astype(float)
    theta, h = rng.normal(size=5), 1e-6
    _, grad = offset_logistic_loss(theta, Z, off, y, 2.5)
    num = np.array([(offset_logistic_loss(theta + h * e, Z, off, y, 2.5)[0]
                     - offset_logistic_loss(theta - h * e, Z, off, y, 2.5)[0]) / (2 * h) for e in np.eye(5)])
    assert np.allclose(grad, num, rtol=1e-6, atol=1e-9)
    # the penalty skips the intercept: with Z = 0 the b0 gradient is the pure residual sum
    _, g0 = offset_logistic_loss(np.array([0.3, 1, 1, 1, 1.0]), np.zeros_like(Z), off, y, 100.0)
    assert g0[0] == pytest.approx(np.mean(expit(off + 0.3) - y))


def test_missing_indicator_is_learned_and_folded_into_its_feature():
    X, p, y = planted(30_000, seed=4, beta=0.5, missing=0.3, miss_effect=0.8)
    m = ResidualLogistic().fit(X, p, y)
    assert m.missing_ == ["x1"]  # x2 has no NaN in training: no indicator
    prm = m.params()
    assert prm["missing_coef"]["x1"] == pytest.approx(0.8, abs=0.12)
    assert prm["coef"]["x1"] == pytest.approx(0.5, abs=0.06)
    contrib = m.contributions(X, p)
    assert list(contrib.columns) == ["x1", "x2", "market", "intercept"]
    pred = m.predict_raw(X, p)
    assert np.all(np.isfinite(pred))
    assert np.allclose(contrib.sum(axis=1), logit(pred))
    # NaN in a feature that never had NaN in training: mean imputation, contribution 0
    X_nan = X.head(5).assign(x2=np.nan)
    assert np.allclose(m.contributions(X_nan, p[:5])["x2"], 0.0)


def test_logistic_aligns_columns_by_name_and_rejects_mismatches():
    X, p, y = planted(5_000, seed=5)
    m = ResidualLogistic().fit(X, p, y)
    assert np.array_equal(m.predict_raw(X[["x2", "x1"]], p), m.predict_raw(X, p))
    with pytest.raises(ValueError, match="unseen"):
        m.predict_raw(X.assign(x3=1.0), p)
    with pytest.raises(ValueError, match="missing"):
        m.predict_raw(X[["x1"]], p)
    with pytest.raises(ValueError):
        m.predict_raw(X.rename(columns={"x2": "x9"}), p)


def test_constant_feature_is_dropped():
    X, p, y = planted(3_000, seed=6)
    m = ResidualLogistic().fit(X.assign(c=2.0), p, y)
    assert m.params()["dropped_constant"] == ["c"]
    assert np.allclose(m.contributions(X.assign(c=5.0), p)["c"], 0.0)


# ---------------------------------------------------------------------------
# ResidualLGBM
# ---------------------------------------------------------------------------

LGBM = load_sport_config("mlb")["model"]["lgbm"]


def test_lgbm_learns_a_nonlinear_residual_the_linear_model_cannot():
    train, test = simulate_games(6_000, 21, signal="interaction"), simulate_games(3_000, 22, signal="interaction")
    lgbm = ResidualLGBM(seed=1, **LGBM).fit(train[FEATURES], train.p, train.y)
    lin = ResidualLogistic().fit(train[FEATURES], train.p, train.y)
    b_lgbm = brier(lgbm.predict_raw(test[FEATURES], test.p), test.y)
    b_lin = brier(lin.predict_raw(test[FEATURES], test.p), test.y)
    b_mkt = brier(test.p, test.y)
    assert b_lgbm < b_mkt - 0.002
    assert b_lgbm < b_lin - 0.001


def with_priced_feature(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Replace bullpen_diff by a noisy copy of the market logit: information the market already prices."""
    noise = np.random.default_rng(seed).normal(0, 0.3, df.game.max() + 1)[df.game]
    return df.assign(bullpen_diff=logit(df.p.to_numpy()) + np.where(df.side == 0, noise, -noise))


def test_lgbm_with_no_residual_signal_tracks_the_market_through_init_score():
    train = with_priced_feature(simulate_games(6_000, 23, signal="none"), 1)
    test = with_priced_feature(simulate_games(3_000, 24, signal="none"), 2)
    m = ResidualLGBM(seed=1, **LGBM).fit(train[FEATURES], train.p, train.y)
    z = logit(m.predict_raw(test[FEATURES], test.p))
    z_mkt = logit(test.p.to_numpy())
    # observed: mean |dz| ~0.12 (~3pp at p = 0.5, all noise, with the configured params)
    assert np.mean(np.abs(z - z_mkt)) < 0.15
    assert np.corrcoef(z, z_mkt)[0, 1] > 0.9
    assert abs(np.mean(z - z_mkt)) < 0.02
    # the same booster fit WITHOUT the offset learns what the market already prices; adding the
    # market back at predict time then double counts it
    params, rounds = m._train_params()
    plain = lgb.train(params, lgb.Dataset(train[FEATURES].to_numpy(), label=train.y), num_boost_round=rounds)
    z_double = z_mkt + plain.predict(test[FEATURES].to_numpy(), raw_score=True)
    assert np.mean(np.abs(z_double - z_mkt)) > 2 * np.mean(np.abs(z - z_mkt))
    assert brier(expit(z_double), test.y) > brier(expit(z), test.y)


def test_lgbm_contributions_sum_to_the_prediction_and_fits_are_deterministic():
    train = simulate_games(2_000, 25)
    a = ResidualLGBM(seed=3, **LGBM).fit(train[FEATURES], train.p, train.y)
    b = ResidualLGBM(seed=3, **LGBM).fit(train[FEATURES], train.p, train.y)
    pa = a.predict_raw(train[FEATURES], train.p)
    assert np.array_equal(pa, b.predict_raw(train[FEATURES], train.p))
    c = a.contributions(train[FEATURES[::-1]], train.p)
    assert set(FEATURES) | {"market", "intercept"} == set(c.columns)
    assert np.allclose(c.sum(axis=1), logit(pa), atol=1e-9)


# ---------------------------------------------------------------------------
# fit_model end to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    cfg = load_sport_config("mlb")
    train, test = simulate_games(6_000, 31), simulate_games(1_500, 32)
    models, t0 = {}, time.perf_counter()
    for mt in ("market", "logistic_residual", "lgbm_residual"):
        models[mt] = fit_model(mt, train[FEATURES], train.p, train.y, train.game_id, FAMILIES, cfg, seed=7)
    return {"models": models, "elapsed": time.perf_counter() - t0, "train": train, "test": test, "cfg": cfg}


def test_fit_model_all_types_within_time_budget(fitted):
    print(f"fit_model x3 on 6,000 games: {fitted['elapsed']:.1f}s")
    assert fitted["elapsed"] < 60


@pytest.mark.parametrize("mt", ["market", "logistic_residual", "lgbm_residual"])
def test_predict_output_passes_prediction_invariants(fitted, mt):
    test = fitted["test"]
    out = fitted["models"][mt].predict(test[FEATURES], test.p)
    assert list(out.columns) == ["p_market", "p_model_raw", "p_model", "lower_bound", "upper_bound",
                                 "p_conservative_yes", "p_conservative_no", "uncertainty_json"]
    assert len(out) == len(test) and out.index.equals(test.index)
    assert_prediction_invariants(out)
    assert (out.upper_bound - out.lower_bound).gt(0).all()  # no degenerate intervals, market included
    comp = json.loads(out.uncertainty_json.iloc[0])
    assert set(comp) == {"sd_ensemble", "sd_calibration", "missing_share", "scale", "z"}
    assert comp["z"] == pytest.approx(1.6449, abs=1e-4)


@pytest.mark.parametrize("mt", ["market", "logistic_residual", "lgbm_residual"])
def test_save_load_round_trip_gives_identical_predictions(fitted, mt, tmp_path):
    test, model = fitted["test"], fitted["models"][mt]
    loaded = FittedModel.load(model.save(tmp_path / f"{mt}.pkl"))
    pd.testing.assert_frame_equal(loaded.predict(test[FEATURES], test.p), model.predict(test[FEATURES], test.p))


def test_market_model_is_the_uncalibrated_market_with_calibration_width_only(fitted):
    test, m = fitted["test"], fitted["models"]["market"]
    out = m.predict(test[FEATURES], test.p)
    assert np.array_equal(out.p_model.to_numpy(), test.p.to_numpy())
    comps = out.uncertainty_json.map(json.loads)
    assert all(c["sd_ensemble"] == 0 and c["missing_share"] == 0 and c["sd_calibration"] > 0 for c in comps)
    assert m.summary()["calibration_method"] == "none"


def test_residual_models_beat_the_market_out_of_sample(fitted):
    test = fitted["test"]
    b_mkt = brier(test.p, test.y)
    for mt in ("logistic_residual", "lgbm_residual"):
        assert brier(fitted["models"][mt].predict(test[FEATURES], test.p).p_model, test.y) < b_mkt - 0.002, mt


def test_summary_reports_the_split_and_the_choices(fitted):
    train = fitted["train"]
    s = fitted["models"]["logistic_residual"].summary()
    assert s["n_fit_obs"] + s["n_cal_obs"] == len(train)
    assert s["n_cal_obs"] == pytest.approx(0.25 * len(train), abs=2)
    assert s["n_fit_games"] + s["n_cal_games"] == s["n_games"] == 6_000  # no game on both sides
    assert s["calibration_method"] in ("none", "platt", "isotonic")
    assert set(s["calibration_scores"]) == {"none", "platt", "isotonic"}
    assert s["uncertainty"]["proxy_coverage_cal"] == pytest.approx(0.90, abs=0.01)
    assert s["features"] == FEATURES
    assert s["brier_model_cal"] < s["brier_market_cal"]
    assert s["refit_on_full_window"] is False
    assert s["config_hash"] == fitted["models"]["logistic_residual"].summary()["config_hash"]
    assert s["config_hash"] != fitted["models"]["lgbm_residual"].summary()["config_hash"]


def test_fitted_contributions_sum_to_logit_p_model(fitted):
    test = fitted["test"].head(200)
    for m in fitted["models"].values():
        c = m.contributions(test[FEATURES], test.p)
        assert np.allclose(c.sum(axis=1), logit(m.predict(test[FEATURES], test.p).p_model), atol=1e-9)


def test_fitted_model_rejects_unknown_features(fitted):
    test = fitted["test"]
    with pytest.raises(ValueError, match="unseen"):
        fitted["models"]["logistic_residual"].predict(test[FEATURES].assign(extra=0.0), test.p)


def test_fit_model_rejects_features_without_a_family(fitted):
    train = fitted["train"]
    with pytest.raises(ValueError, match="family"):
        fit_model("logistic_residual", train[FEATURES].assign(extra=0.0), train.p, train.y,
                  train.game_id, FAMILIES, fitted["cfg"], seed=1)
