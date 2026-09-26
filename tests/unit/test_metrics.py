"""Hand-checkable examples for src/backtest/metrics.py (Sec 17, 26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as M


def test_brier_log_loss_bss_small_numbers():
    y, p = [1, 0, 1, 0], [0.8, 0.3, 0.6, 0.5]
    assert M.brier(y, p) == pytest.approx((0.04 + 0.09 + 0.16 + 0.25) / 4)     # 0.135
    assert M.log_loss([1, 0], [0.8, 0.3]) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert M.bss(0.18, 0.20) == pytest.approx(0.10)
    assert M.bss(0.22, 0.20) == pytest.approx(-0.10)
    assert np.isnan(M.bss(0.1, 0.0))


def test_wilson_interval_known_value():
    lo, hi = M.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert all(np.isnan(M.wilson_interval(0, 0)))


def test_calibration_ece_hand_example():
    # bin [0.2, 0.3): p = 0.25 x2, outcomes 0, 0 -> gap 0.25; bin [0.8, 0.9): p = 0.85 x2, outcomes 0, 0 -> gap 0.85
    # bin [0.5, 0.6): p = 0.55 x4, outcomes 1,1,1,1 -> gap 0.45; bin [0.1, 0.2): p=0.15 x2, outcomes 0,0 -> gap 0.15
    y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    p = [0.25, 0.25, 0.85, 0.85, 0.55, 0.55, 0.55, 0.55, 0.15, 0.15]
    c = M.calibration_table(y, p, 10)
    expected = (2 * 0.25 + 2 * 0.85 + 4 * 0.45 + 2 * 0.15) / 10
    assert c["ece"] == pytest.approx(expected)                                   # 0.43
    assert c["mce"] == pytest.approx(0.85)
    t = c["table"].set_index("bin")
    assert t["n"].sum() == 10
    assert (t["ci_low"].dropna() <= t["empirical_frequency"].dropna()).all()


def test_calibration_slope_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 20_000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(float)
    over = 1 / (1 + np.exp(-2 * z))                                              # logits doubled
    c = M.calibration_table(y, over, 10)
    assert c["slope"] == pytest.approx(0.5, abs=0.05)
    assert c["intercept"] == pytest.approx(0.0, abs=0.05)


def _cal_games(n_games, rng, rows_per_game=1, scale=1.0):
    """One outcome per game; rows_per_game=2 adds the complementary (away) contract, as in the review's two
    contracts per game. The model's logits are the truth's times `scale` (true calibration slope 1 / scale)."""
    z = rng.normal(0, 1, n_games)
    won = (rng.random(n_games) < 1 / (1 + np.exp(-z))).astype(float)
    zz = np.column_stack([z, -z][:rows_per_game]).ravel()
    y = np.column_stack([won, 1 - won][:rows_per_game]).ravel()
    return y, 1 / (1 + np.exp(-scale * zz)), np.repeat(np.arange(n_games), rows_per_game)


def test_calibration_slope_bootstrap_point_matches_calibration_table():
    y, p, g = _cal_games(3000, np.random.default_rng(5), rows_per_game=2, scale=1.5)
    b = M.calibration_slope_bootstrap(y, p, g, n=200, seed=0, ci=0.95)
    c = M.calibration_table(y, p, 10)
    assert b["slope"] == pytest.approx(c["slope"], rel=1e-9) and b["intercept"] == pytest.approx(c["intercept"],
                                                                                                abs=1e-9)
    assert b["ci_low"] < b["slope"] < b["ci_high"]
    assert b["slope"] == pytest.approx(1 / 1.5, abs=0.08)
    assert b["n_games"] == 3000 and b["n_obs"] == 6000 and b["unit"] == "game"


def test_calibration_slope_bootstrap_resamples_games_not_rows():
    """Five identical rows per game (five snapshot offsets) must not narrow the CI; a row bootstrap would."""
    y, p, g = _cal_games(1500, np.random.default_rng(6), scale=1.2)
    once = M.calibration_slope_bootstrap(y, p, g, n=400, seed=3, ci=0.95)
    five = M.calibration_slope_bootstrap(np.repeat(y, 5), np.repeat(p, 5), np.repeat(g, 5), n=400, seed=3, ci=0.95)
    assert five["n_games"] == 1500 and five["n_obs"] == 7500
    assert five["slope"] == pytest.approx(once["slope"], rel=1e-7)
    assert (five["ci_low"], five["ci_high"]) == pytest.approx((once["ci_low"], once["ci_high"]), rel=1e-6)
    rows = M.calibration_slope_bootstrap(np.repeat(y, 5), np.repeat(p, 5), np.arange(7500), n=400, seed=3, ci=0.95)
    width = lambda r: r["ci_high"] - r["ci_low"]
    assert width(rows) < 0.6 * width(five)                                       # ~sqrt(5) narrower


def test_calibration_slope_bootstrap_seed_resamples_and_level_are_used_and_recorded():
    y, p, g = _cal_games(2000, np.random.default_rng(7), rows_per_game=2)
    a = M.calibration_slope_bootstrap(y, p, g, n=300, seed=1, ci=0.95)
    assert a == M.calibration_slope_bootstrap(y, p, g, n=300, seed=1, ci=0.95)  # deterministic given the seed
    assert (a["n_resamples"], a["seed"], a["ci_level"]) == (300, 1, 0.95) and a["n_valid"] == 300
    b = M.calibration_slope_bootstrap(y, p, g, n=300, seed=2, ci=0.95)
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])
    c = M.calibration_slope_bootstrap(y, p, g, n=301, seed=1, ci=0.95)
    assert c["n_resamples"] == 301 and (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])
    half = M.calibration_slope_bootstrap(y, p, g, n=300, seed=1, ci=0.50)
    ratio = (half["ci_high"] - half["ci_low"]) / (a["ci_high"] - a["ci_low"])
    assert ratio == pytest.approx(0.6745 / 1.96, abs=0.08)
    none = M.calibration_slope_bootstrap(y, p, g, n=0, seed=1, ci=0.95)
    assert np.isfinite(none["slope"]) and np.isnan(none["ci_low"]) and np.isnan(none["ci_high"])


def test_calibration_slope_bootstrap_ci_agrees_with_cluster_robust_se():
    y, p, g = _cal_games(4000, np.random.default_rng(8), rows_per_game=2, scale=0.8)
    b = M.calibration_slope_bootstrap(y, p, g, n=1000, seed=4, ci=0.95)
    z = np.log(p / (1 - p))
    se = M.logistic_regression(np.column_stack([np.ones(len(z)), z]), y, groups=g)["se"][1]
    assert b["ci_high"] - b["ci_low"] == pytest.approx(2 * 1.96 * se, rel=0.15)


def test_benjamini_hochberg_known_example():
    q = M.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])
    q = M.benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(q[1]) and q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.02)


def test_drawdown_and_losing_streak():
    pnl = [5, -2, -3, 4, -1, -1, -1, 10]
    assert M.max_drawdown(pnl) == pytest.approx(5)                               # 5 -> 0
    assert M.max_drawdown([-4, 1]) == pytest.approx(4)                           # from the 0 start
    assert M.longest_losing_streak(pnl) == 3
    assert M.longest_losing_streak([1, 2]) == 0


def test_spearman_in_edge_monotonicity():
    buckets = pd.DataFrame({"rank": [0, 1, 2, 3], "realized_edge": [-0.02, 0.01, 0.00, 0.05],
                            "sufficient": [True, True, True, True]})
    rng = np.random.default_rng(1)
    n = 400
    dec = pd.DataFrame({"game_id": np.arange(n) // 2, "side": "yes", "decision": "PASS",
                        "p_market": rng.uniform(0.3, 0.7, n), "quoted_price": 0.5,
                        "predicted_edge": rng.normal(0, 0.05, n), "side_won": rng.integers(0, 2, n)})
    m = M.edge_monotonicity(buckets, dec)
    assert m["spearman"] == pytest.approx(0.8)                                   # ranks 1,3,2,4 vs 1,2,3,4
    assert m["n_sufficient_buckets"] == 4 and np.isfinite(m["se_edge"])
    two = buckets.assign(sufficient=[True, True, False, False])
    assert np.isnan(M.edge_monotonicity(two, dec)["spearman"])


def _games(n_games, rng, signal=0.0):
    p_mkt = rng.uniform(0.3, 0.7, n_games)
    y = (rng.random(n_games) < p_mkt).astype(float)
    p_model = np.clip(p_mkt + signal * (y - p_mkt) + rng.normal(0, 0.02, n_games), 0.01, 0.99)
    return pd.DataFrame({"game_id": np.arange(n_games), "y": y, "p_market": p_mkt, "p_model": p_model,
                         "period_id": np.where(np.arange(n_games) < n_games // 2, "A", "B")})


def test_bootstrap_resamples_games_not_rows():
    df = _games(400, np.random.default_rng(2), signal=0.1)
    once = M.paired_game_bootstrap(df, n=2000, seed=0)
    # the same games observed at five snapshot offsets: five identical rows per game
    five = pd.concat([df] * 5, ignore_index=True)
    rep = M.paired_game_bootstrap(five, n=2000, seed=0)
    assert rep["n_games"] == 400 and rep["n_obs"] == 2000
    assert rep["mean_delta"] == pytest.approx(once["mean_delta"])
    width = lambda r: r["ci_high"] - r["ci_low"]
    assert width(rep) == pytest.approx(width(once), rel=1e-9)                   # not narrowed by sqrt(5)
    # a row-level bootstrap would have shrunk it ~2.2x; guard against that regression explicitly
    as_rows = M.paired_game_bootstrap(five.assign(game_id=np.arange(len(five))), n=2000, seed=0)
    assert width(as_rows) < 0.6 * width(rep)


def test_bootstrap_detects_planted_signal_and_reports_p():
    rng = np.random.default_rng(3)
    good = M.paired_game_bootstrap(_games(3000, rng, signal=0.15), n=2000, seed=1)
    assert good["ci_low"] > 0 and good["p_value"] < 0.01 and good["bss"] > 0
    null = M.paired_game_bootstrap(_games(3000, rng, signal=-0.1), n=2000, seed=1)
    assert null["ci_high"] < 0 and null["p_value"] > 0.99


def test_shuffle_p_near_one_when_model_equals_market():
    df = _games(2000, np.random.default_rng(4))
    df["p_model"] = df["p_market"]
    s = M.shuffle_test(df, n_perm=300, seed=0)
    assert s["observed_delta_brier"] == pytest.approx(0.0, abs=1e-12)
    assert s["p_value"] == pytest.approx(1.0)
    assert "NOT refit" in s["method"]


def test_shuffle_p_small_for_planted_signal():
    s = M.shuffle_test(_games(2000, np.random.default_rng(5), signal=0.1), n_perm=500, seed=0)
    assert s["p_value"] <= 2 / 501
    assert s["observed_delta_brier"] > s["perm_mean"] + 3 * s["perm_sd"]


def test_shuffle_p_uniformish_without_signal():
    s = M.shuffle_test(_games(2000, np.random.default_rng(6), signal=0.0), n_perm=500, seed=0)
    assert 0.01 < s["p_value"] < 0.99


def _decisions(edges_by_game, rng):
    rows = []
    for g, e in enumerate(edges_by_game):
        won = int(rng.random() < 0.5 + e)
        rows.append({"game_id": g, "observation_id": f"o{g}", "snapshot_time": pd.Timestamp("2025-06-01") +
                     pd.Timedelta(minutes=g), "side": "yes", "decision": "BUY_YES" if e >= 0.04 else "PASS",
                     "p_market": 0.5, "quoted_price": 0.5, "predicted_edge": e, "side_won": won,
                     "stake": 50.0 if e >= 0.04 else np.nan, "gross_pnl": 100 * (won - 0.5) if e >= 0.04 else np.nan,
                     "net_pnl": 100 * (won - 0.5) - 1 if e >= 0.04 else np.nan})
    return pd.DataFrame(rows)


def test_edge_buckets_label_insufficient_sample():
    rng = np.random.default_rng(7)
    edges = [0.0, 0.02, 0.04, 0.06, 0.08]
    e = np.concatenate([np.full(300, -0.01), np.full(300, 0.01), np.full(10, 0.03), np.full(250, 0.05),
                        np.full(5, 0.07), np.full(250, 0.10)])
    b = M.edge_buckets(_decisions(e, rng), edges, min_games=200, n_boot=200, seed=0)
    assert list(b["bucket"]) == ["<0%", "0-2%", "2-4%", "4-6%", "6-8%", ">8%"]
    assert list(b["n_games"]) == [300, 300, 10, 250, 5, 250]
    assert list(b["label"]) == ["OK", "OK", M.INSUFFICIENT, "OK", M.INSUFFICIENT, "OK"]
    assert list(b["sufficient"]) == [True, True, False, True, False, True]
    assert list(b["trades"]) == [0, 0, 0, 250, 5, 250]
    # buckets are never merged: a bucket with no rows still appears
    empty = M.edge_buckets(_decisions(np.full(50, 0.01), rng), edges, min_games=200, n_boot=50)
    assert len(empty) == 6 and empty.loc[empty["bucket"] == ">8%", "observations"].item() == 0
    assert (empty["label"] == M.INSUFFICIENT).all()


def test_edge_bucket_boundaries():
    edges = [0.0, 0.02, 0.04]
    np.testing.assert_array_equal(M.assign_buckets([-0.001, 0.0, 0.0199, 0.02, 0.04, np.nan], edges),
                                  [0, 1, 1, 2, 3, -1])


def test_trading_summary_pnl_layers():
    d = pd.DataFrame({"observation_id": ["a", "b", "c"], "snapshot_time": pd.to_datetime(["2025-01-01"] * 3),
                      "decision": ["BUY_YES", "BUY_NO", "PASS"], "pass_reason": [None, None, "below_min_edge"],
                      "side": ["yes", "no", "yes"], "p_market": 0.5, "quoted_price": [0.4, 0.5, 0.5],
                      "predicted_edge": [0.1, 0.1, 0.0], "execution_price": [0.41, 0.52, np.nan],
                      "quantity": [100, 100, np.nan], "stake": [41.0, 52.0, np.nan], "side_won": [1, 0, 1],
                      "gross_pnl": [60.0, -50.0, np.nan], "calculated_fee": [1.0, 1.0, np.nan],
                      "slippage_cost": [1.0, 2.0, np.nan], "net_pnl": [58.0, -53.0, np.nan]})
    t = M.trading_summary(d)
    assert (t["trades"], t["passes"]) == (2, 1)
    assert t["pass_reasons"] == {"below_min_edge": 1}
    assert t["pnl_before_fees"] == pytest.approx(10.0)
    assert t["pnl_after_fees"] == pytest.approx(8.0)
    assert t["pnl_after_fees_slippage"] == pytest.approx(5.0)
    assert t["roi"] == pytest.approx(5.0 / 93.0)
    assert t["win_rate"] == pytest.approx(0.5)
    assert t["max_drawdown"] == pytest.approx(53.0)


def test_concentration_flags():
    n = 30
    net = np.r_[np.full(10, 100.0), np.full(20, 1.0)]
    t = pd.DataFrame({"decision": "BUY_YES", "side": "yes", "p_market": 0.5, "quoted_price": 0.5,
                      "predicted_edge": 0.05, "side_won": 1, "net_pnl": net, "gross_pnl": net, "stake": 50.0,
                      "yes_team": ["NYY"] * 10 + [f"T{i}" for i in range(20)], "period_id": "P",
                      "official_date": pd.Timestamp("2025-06-01")})
    c = M.concentration(t, top10_max=0.40, team_max=0.20, edges=[0.0, 0.04])
    assert c["top10_share"] == pytest.approx(1000 / 1020)
    assert c["flag_top10"] and c["flag_team"] and c["best_team"] == "NYY"
    assert c["ex_top10"]["net_pnl"] == pytest.approx(20.0)
    assert set(c["groups"]) >= {"period", "month", "team", "market_probability_band", "predicted_edge_band"}
    assert c["groups"]["team"]["group"].iloc[0] == "NYY"


def test_edge_decay_warning():
    rows = []
    for i, e in enumerate([0.08, 0.06, 0.04, 0.02, 0.00]):
        for j in range(40):
            won = 1 if j < int(40 * (0.5 + e)) else 0
            rows.append({"period_id": f"P{i}", "decision": "BUY_YES", "side": "yes", "p_market": 0.5,
                         "quoted_price": 0.5, "predicted_edge": 0.05, "side_won": won,
                         "net_pnl": 100 * (won - 0.5), "stake": 50.0})
    d = M.edge_decay(pd.DataFrame(rows), [f"P{i}" for i in range(5)])
    assert d["n_periods"] == 5 and d["slope"] < 0 and d["slope_ci_high"] < 0
    assert d["recent_two_underperform"] and d["decay_warning"]


def test_interval_coverage_and_binned_target():
    assert M.interval_coverage([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.15, 0.35, 0.4]) == pytest.approx(2 / 3)
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < p).astype(float)
    target = M.binned_frequency_target(y, p, 500)
    assert target is not None and len(np.unique(target)) == 10
    assert M.binned_frequency_target(y[:4000], p[:4000], 500) is None           # only 8 bins


# -- 26.12 outcome upper bound on coverage --------------------------------------------

def test_coverable_rows_hand_examples():
    lo, hi = np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.6, 0.7])
    # mean(p) >= 0.8 needs sum(p) >= 2.4: one covered row (the 0.7 cap) + two rows at 1 reach 2.7; two covered reach 2.3
    assert M._coverable_rows(lo, hi, 0.8, 0.9) == 1
    # mean(p) <= 0.2 needs sum(p) <= 0.6: one covered row (at 0.4) fits; two need >= 0.9
    assert M._coverable_rows(lo, hi, 0.1, 0.2) == 1
    assert M._coverable_rows(lo, hi, 0.45, 0.55) == 3                 # the band meets [mean(lo), mean(hi)]


def _calibrated_world(n, half, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, n)
    p_true = np.clip(p + rng.normal(0, 0.05, n), 0, 1)
    y = (rng.random(n) < p_true).astype(float)
    return y, p, p_true, np.clip(p - half, 0, 1), np.clip(p + half, 0, 1)


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_coverage_bound_is_valid_but_blind_to_width_when_calibrated(factor):
    """Intervals sized to cover p_true 90% of the time, then scaled by a known factor (true coverage about
    0.59 / 0.90 / 0.999). The bin means sit inside the intervals, so the bound is about 1 in all three: it
    holds (>= truth) and cannot tell the too-narrow intervals from the right ones. That is the limitation."""
    y, p, p_true, lo, hi = _calibrated_world(6000, factor * 1.6449 * 0.05, seed=31)
    true_cov = M.interval_coverage(lo, hi, p_true)
    assert true_cov == pytest.approx({0.5: 0.59, 1.0: 0.90, 2.0: 0.999}[factor], abs=0.03)
    b = M.coverage_upper_bound(y, p, lo, hi, min_obs_per_bin=500, alpha=0.05)
    assert b["n_bins"] == 12 and b["bound"] >= true_cov
    assert b["bound"] >= 0.87                                         # never shows the x0.5 under-coverage


def test_coverage_bound_shows_a_known_shift():
    """p_model near 0.3, p_true = p_model + 0.4, intervals +/-0.05: true coverage is 0. Each bin's frequency
    (about 0.7) is far above every interval, so at most about (1 - 0.64) / (1 - 0.35) of rows can be covered."""
    rng = np.random.default_rng(32)
    n = 6000
    p = rng.uniform(0.25, 0.35, n)
    p_true = p + 0.4
    y = (rng.random(n) < p_true).astype(float)
    lo, hi = p - 0.05, p + 0.05
    assert M.interval_coverage(lo, hi, p_true) == 0.0
    b = M.coverage_upper_bound(y, p, lo, hi, min_obs_per_bin=500, alpha=0.05)
    assert 0.0 <= b["bound"] < 0.7 and b["bins_binding"] == b["n_bins"] == 12
    assert set(b["table"].columns) >= {"bin", "n", "n_eff", "frequency", "ci_low", "ci_high", "coverable_share"}


def test_coverage_bound_alpha_validity_in_the_extremal_world():
    """The bound is sharp here: every interval is [0.45, 0.55]; 60% of rows have p_true 0.55 (covered) and
    40% have p_true 1 (the most an uncovered row can add). True coverage 0.6, and the bound approaches it.
    Across repetitions it falls below the truth in at most about alpha of them."""
    n, reps, alpha = 2000, 300, 0.05
    rng = np.random.default_rng(33)
    lo, hi, p = np.full(n, 0.45), np.full(n, 0.55), np.full(n, 0.5)
    below, bounds = 0, []
    for _ in range(reps):
        p_true = np.where(rng.random(n) < 0.6, 0.55, 1.0)
        true_cov = float(np.mean(p_true <= 0.55))
        y = (rng.random(n) < p_true).astype(float)
        b = M.coverage_upper_bound(y, p, lo, hi, min_obs_per_bin=500, alpha=alpha)["bound"]
        below += b < true_cov
        bounds.append(b)
    assert below <= 1.5 * alpha * reps
    assert 0.6 < np.median(bounds) < 0.87                             # informative: it would fail 26.12


def test_coverage_bound_clusters_shrink_effective_size():
    """Each observation duplicated under one cluster id (a game's two perfectly dependent rows): the effective
    size per bin halves, the Wilson intervals widen, and the bound can only rise."""
    rng = np.random.default_rng(34)
    n = 3000
    p = rng.uniform(0.25, 0.35, n)
    y = (rng.random(n) < p + 0.4).astype(float)
    lo, hi = p - 0.05, p + 0.05
    dup = lambda a: np.repeat(a, 2)
    flat = M.coverage_upper_bound(dup(y), dup(p), dup(lo), dup(hi), min_obs_per_bin=500, alpha=0.05)
    clus = M.coverage_upper_bound(dup(y), dup(p), dup(lo), dup(hi), np.repeat(np.arange(n), 2),
                                  min_obs_per_bin=500, alpha=0.05)
    assert np.allclose(clus["table"]["n_eff"], clus["table"]["n"] / 2)
    assert np.allclose(flat["table"]["n_eff"], flat["table"]["n"])
    assert clus["bound"] > flat["bound"]


def test_coverage_bound_refuses_small_or_bad_input():
    y, p = np.array([0, 1] * 200, dtype=float), np.full(400, 0.5)
    lo, hi = p - 0.1, p + 0.1
    assert M.coverage_upper_bound(y, p, lo, hi, min_obs_per_bin=500, alpha=0.05) is None
    assert M.coverage_upper_bound(y, p, lo, hi, min_obs_per_bin=100, alpha=0.05)["n_bins"] == 4
    bad = hi.copy()
    bad[0] = np.nan
    assert M.coverage_upper_bound(y, p, lo, bad, min_obs_per_bin=100, alpha=0.05) is None
    for a in (0.0, 1.0):
        with pytest.raises(ValueError):
            M.coverage_upper_bound(y, p, lo, hi, min_obs_per_bin=100, alpha=a)
