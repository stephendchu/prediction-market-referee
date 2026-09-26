"""`quant review` (Sec 26-27) and the Sec 17 evaluation report on small synthetic frames.

Builders below construct ReviewInputs directly (no database) with a planted
residual signal; one test runs load_review_inputs against the in-memory DB.
"""

from __future__ import annotations

import copy
import io
import json
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from factories import frame, observation
from src.backtest.walk_forward import HoldoutLocked, evaluation_periods, holdout_period, load_periods
from src.config import load_sport_config
from src.reports import diagnostics as D

SCENARIOS = {"pen_0": 0.0, "pen_25": 0.25, "pen_50": 0.5, "pen_100": 1.0}
TEAMS = [f"T{i:02d}" for i in range(30)]


def small_cfg(**review) -> dict:
    cfg = copy.deepcopy(load_sport_config("mlb"))
    cfg["review"] |= {"bucket_min_games": 30, "bootstrap_resamples": 300, "shuffle_permutations": 200,
                      "coverage_bin_min_obs": 100} | review
    return cfg


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_predictions(cfg: dict, *, games_per_period: int = 300, signal: float = 0.7, miscal: float = 1.0,
                     seed: int = 1, periods=None) -> pd.DataFrame:
    """Two complementary contracts per game. p_true -> outcome; market = truth + noise;
    model moves `signal` of the way from market to truth; `miscal` scales model logits."""
    rng = np.random.default_rng(seed)
    periods = periods or evaluation_periods(load_periods(cfg, "synthetic"))
    rows = []
    for per in periods:
        n = games_per_period
        days = (per.end - per.start).days
        dates = [per.start + timedelta(days=int(d)) for d in rng.integers(0, days + 1, n)]
        z_true = rng.normal(0, 0.6, n)
        z_mkt = z_true + rng.normal(0, 0.3, n)
        z_mod = z_mkt + signal * (z_true - z_mkt)
        p_true, p_mkt = _sigmoid(z_true), np.clip(_sigmoid(z_mkt), 0.03, 0.97)
        p_mod = np.clip(_sigmoid(miscal * z_mod), 0.01, 0.99)
        home_won = (rng.random(n) < p_true).astype(int)
        h, a = rng.integers(0, 30, n), rng.integers(0, 29, n)
        a = np.where(a >= h, a + 1, a)
        for side in ("home", "away"):
            yes_home = side == "home"
            gid = [f"g-{per.period_id}-{i}" for i in range(n)]
            rows.append(pd.DataFrame({
                "observation_id": [f"{g}-{side}@T-30" for g in gid], "game_id": gid,
                "contract_id": [f"{g}-{side}" for g in gid], "period_id": per.period_id,
                "official_date": dates, "season": [d.year for d in dates],
                "home_team": [TEAMS[i] for i in h], "away_team": [TEAMS[i] for i in a],
                "yes_team": [TEAMS[i] for i in (h if yes_home else a)], "yes_side": side,
                "p_market": p_mkt if yes_home else 1 - p_mkt, "p_model": p_mod if yes_home else 1 - p_mod,
                "p_true": p_true if yes_home else 1 - p_true,
                "outcome": home_won if yes_home else 1 - home_won,
            }))
    p = pd.concat(rows, ignore_index=True)
    half = float(np.quantile(np.abs(p["p_model"] - p["p_true"]), 0.90))
    p["lower_bound"] = np.clip(p["p_model"] - half, 0, 1)
    p["upper_bound"] = np.clip(p["p_model"] + half, 0, 1)
    p["p_conservative_yes"], p["p_conservative_no"] = p["lower_bound"], 1 - p["upper_bound"]
    p["quote_age_s"], p["spread"], p["size_verified"] = 60.0, 0.02, True
    p["snapshot_offset_min"], p["is_canonical"] = 30, True
    p["liquidity_tercile"], p["availability_state"] = 2, "complete"
    p["model_run_id"] = "mr-" + p["period_id"]
    p["snapshot_time"] = pd.to_datetime(p["official_date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=22, minutes=35)
    return p


def make_decisions(preds: pd.DataFrame, *, minimum_edge: float = 0.04, offset: int = 30) -> pd.DataFrame:
    """backtest_decisions semantics (schemas.BACKTEST_DECISIONS): side chosen at pen_25 by
    net_edge; the same trades re-priced for every scenario."""
    p = preds
    bid, ask = p["p_market"] - 0.01, p["p_market"] + 0.01
    spread = (ask - bid).to_numpy()
    fee = lambda q, x: 0.07 * q * x * (1 - x)
    quoted = {"yes": ask.to_numpy(), "no": (1 - bid).to_numpy()}
    model = {"yes": p["p_model"].to_numpy(), "no": 1 - p["p_model"].to_numpy()}
    cons = {"yes": p["lower_bound"].to_numpy(), "no": 1 - p["upper_bound"].to_numpy()}
    base = {s: quoted[s] + 0.25 * spread for s in quoted}
    net = {s: cons[s] - base[s] - fee(1, base[s]) for s in quoted}
    is_yes = net["yes"] >= net["no"]
    pick = lambda d: np.where(is_yes, d["yes"], d["no"])
    q_price, p_side, p_cons, net_edge = pick(quoted), pick(model), pick(cons), pick(net)
    trade = net_edge >= minimum_edge
    qty = np.where(trade, np.floor(100 / pick(base)), np.nan)
    side_won = np.where(is_yes, p["outcome"], 1 - p["outcome"])
    out = []
    for name, pen in SCENARIOS.items():
        ex = np.where(trade, q_price + pen * spread, np.nan)
        slip, cost = qty * (ex - q_price), fee(qty, ex)
        gross = qty * (side_won - q_price)
        out.append(pd.DataFrame({
            "backtest_run_id": f"bt-{offset}", "scenario": name, "model_run_id": p["model_run_id"],
            "observation_id": p["observation_id"], "game_id": p["game_id"], "contract_id": p["contract_id"],
            "period_id": p["period_id"], "snapshot_time": p["snapshot_time"],
            "decision": np.where(trade, np.where(is_yes, "BUY_YES", "BUY_NO"), "PASS"),
            "pass_reason": np.where(trade, None, "below_min_edge"), "side": np.where(is_yes, "yes", "no"),
            "p_market": p["p_market"], "p_model": p["p_model"], "p_conservative": p_cons,
            "predicted_edge": p_side - q_price, "net_edge": net_edge, "quoted_price": q_price,
            "spread_penalty_frac": pen, "execution_price": ex, "quantity": qty, "stake": qty * ex,
            "fee_schedule_id": np.where(trade, "taker-test", None),
            "maker_or_taker": np.where(trade, "taker", None), "calculated_fee": cost, "slippage_cost": slip,
            "depth_verified": trade, "outcome": p["outcome"], "side_won": side_won, "gross_pnl": gross,
            "net_pnl": gross - slip - cost, "is_synthetic": True,
            **{c: p[c] for c in ("official_date", "season", "home_team", "away_team", "yes_team", "yes_side",
                                 "liquidity_tercile", "availability_state", "quote_age_s", "spread",
                                 "size_verified", "is_canonical")},
            "snapshot_offset_min": offset,
        }))
    return pd.concat(out, ignore_index=True)


def make_registry(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg_at = pd.Timestamp("2026-01-01T00:00Z")
    hyps = pd.DataFrame({"hypothesis_id": ["H-market", "H-starter", "H-nc"],
                         "family": ["market", "starter", "negative_control"], "registered_at": reg_at,
                         "git_commit": "a" * 40})
    rows = []
    for i, per in enumerate(evaluation_periods(load_periods(cfg, "synthetic"))):
        for fam, hid, fs, d, p in (("hypothesis", "H-starter", "starter_v1", 0.002, 0.01),
                                   ("negative_control", "H-nc", "nc_random", -0.0001 * (i % 2), 0.6),
                                   ("ablation", "H-starter", "drop_starter", 0.001, 0.2)):
            rows.append({"experiment_id": f"e-{fam}-{per.period_id}", "experiment_family": fam, "hypothesis_id": hid,
                         "label": fam, "feature_set_id": fs, "model_type": "logistic_residual",
                         "train_period": "expanding", "test_period": per.period_id, "n_games": 300,
                         "metrics_json": json.dumps({"reference": "market", "candidate": fs, "n_obs": 600,
                                                     "brier_ref": 0.24, "brier_cand": 0.24 - d, "delta_brier": d,
                                                     "logloss_ref": 0.67, "logloss_cand": 0.67 - d,
                                                     "ece_ref": 0.02, "ece_cand": 0.02, "roi_ref": 0.0,
                                                     "roi_cand": 0.01}),
                         "p_value": p, "git_commit": "a" * 40, "created_at": reg_at + pd.Timedelta(days=10)})
    return hyps, pd.DataFrame(rows)


def make_inputs(cfg: dict, preds: pd.DataFrame | None = None, **over) -> D.ReviewInputs:
    preds = make_predictions(cfg) if preds is None else preds
    hyps, exps = make_registry(cfg)
    fields = dict(
        sport="mlb", dataset="synthetic", is_synthetic=True, periods=load_periods(cfg, "synthetic"), predictions=preds,
        decisions=make_decisions(preds, minimum_edge=cfg["decision"]["minimum_edge"]),
        experiments=exps, hypotheses=hyps,
        holdout_manifest={"status": "LOCKED", "holdout": "2026H2", "holdout_evaluated": False},
        leakage_audit={name: 0 for name in D.LEAKAGE_SQL},
        model_selection={"feature_families": ["market", "starter"], "chosen": "logistic_residual"},
        backtest_run={"backtest_run_id": "bt-30", "snapshot_offset_min": 30, "model_type": "logistic_residual",
                      "created_at": "2026-02-01T00:00:00+00:00", "git_commit": "a" * 40},
        fee_schedules=pd.DataFrame({"fee_schedule_id": ["taker-test"], "verified": [False]}),
    )
    return D.ReviewInputs(**(fields | over))


def by_id(review: dict) -> dict[str, dict]:
    return {c["id"]: c for c in review["criteria"]}


@pytest.fixture(scope="module")
def cfg():
    return small_cfg()


@pytest.fixture(scope="module")
def good_review(cfg):
    return D.build_review(make_inputs(cfg), cfg)


# ---------------------------------------------------------------------------
# Structure and status
# ---------------------------------------------------------------------------

def test_every_criterion_present_with_valid_state(good_review):
    crit = by_id(good_review)
    for cid in D.CRITERIA_IDS:
        assert cid in crit, cid
        assert crit[cid]["state"] in D.STATES
        assert {"id", "title", "state", "value", "threshold", "detail"} <= set(crit[cid])
    assert good_review["status"] in D.STATUSES


def test_holdout_unevaluated_keeps_good_model_inconclusive(good_review):
    crit = by_id(good_review)
    assert crit["26.1"]["state"] == D.INSUFFICIENT and crit["26.1"]["evaluated"] is False
    assert crit["26.1-wf"]["state"] == D.PASS          # planted signal is real out of sample
    assert crit["26.3"]["state"] == D.PASS
    assert crit["26.11"]["state"] == D.PASS
    assert crit["26.13"]["state"] == D.PASS
    assert crit["26.17"]["state"] == D.PASS
    assert good_review["status"] == "INCONCLUSIVE"
    assert any("26.1 " in r for r in good_review["status_reasons"])


def test_coverage_uses_generator_truth_on_synthetic(good_review):
    c = by_id(good_review)["26.12"]
    assert c["value"]["governing_target"] == "p_true"
    assert c["state"] == D.PASS and 0.87 <= c["value"]["coverage"] <= 0.93


def test_severe_miscalibration_rejects(cfg):
    review = D.build_review(make_inputs(cfg, make_predictions(cfg, miscal=2.5)), cfg)
    c = by_id(review)["26.13"]
    assert c["state"] == D.FAIL
    assert c["value"]["slope_model"] < cfg["review"]["calibration_slope_band"][0]
    assert c["value"]["slope_ci_high"] < cfg["review"]["calibration_slope_band"][0]   # the whole CI, not the point
    assert "wholly outside" in c["detail"]
    assert review["status"] == "REJECTED"
    assert any(r.startswith("26.13 FAIL") for r in review["status_reasons"])


def test_leakage_count_rejects(cfg):
    audit = {name: 0 for name in D.LEAKAGE_SQL} | {"feature_available_after_cutoff": 3}
    review = D.build_review(make_inputs(cfg, leakage_audit=audit), cfg)
    assert by_id(review)["26.11"]["state"] == D.FAIL
    assert review["status"] == "REJECTED"


def test_missing_leakage_audit_is_insufficient_not_pass(cfg):
    review = D.build_review(make_inputs(cfg, leakage_audit={}), cfg)
    assert by_id(review)["26.11"]["state"] == D.INSUFFICIENT
    assert review["status"] != "RESEARCH-PROMISING"


def test_no_signal_fails_shuffle_and_period_rules(cfg):
    review = D.build_review(make_inputs(cfg, make_predictions(cfg, signal=0.0, seed=3)), cfg)
    crit = by_id(review)
    assert crit["26.17"]["state"] == D.FAIL
    assert crit["26.1-wf"]["state"] == D.FAIL
    assert review["status"] in ("INCONCLUSIVE", "REJECTED")


def test_negative_control_that_works_rejects(cfg):
    hyps, exps = make_registry(cfg)
    nc = exps["experiment_family"] == "negative_control"
    exps.loc[nc, "metrics_json"] = json.dumps({"delta_brier": 0.003})
    exps.loc[nc, "p_value"] = 0.001
    review = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    assert by_id(review)["26.16"]["state"] == D.FAIL
    assert review["status"] == "REJECTED"


def test_unregistered_family_fails_26_4(cfg):
    hyps, exps = make_registry(cfg)
    ms = {"feature_families": ["market", "starter", "bullpen"]}
    review = D.build_review(make_inputs(cfg, model_selection=ms, experiments=exps, hypotheses=hyps), cfg)
    c = by_id(review)["26.4"]
    assert c["state"] == D.FAIL and "bullpen" in c["value"]["missing"]


def test_hypothesis_registered_after_testing_fails_26_4(cfg):
    hyps, exps = make_registry(cfg)
    hyps.loc[hyps["family"] == "starter", "registered_at"] = pd.Timestamp("2026-06-01T00:00Z")
    review = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    assert by_id(review)["26.4"]["state"] == D.FAIL


def test_holdout_experiment_fails_26_3(cfg):
    hyps, exps = make_registry(cfg)
    extra = exps.iloc[[0]].assign(experiment_id="e-peek", test_period="2026H2")
    review = D.build_review(make_inputs(cfg, experiments=pd.concat([exps, extra]), hypotheses=hyps), cfg)
    c = by_id(review)["26.3"]
    assert c["state"] == D.FAIL and c["value"]["experiments_on_holdout"] == 1
    assert "e-peek" not in set(review["supporting"]["26.5"]["registry"].query("q_value.notna()")["experiment_id"])


def test_edge_buckets_label_insufficient_sample(cfg):
    review = D.build_review(make_inputs(small_cfg(bucket_min_games=10_000)), small_cfg(bucket_min_games=10_000))
    b = review["informational"]["edge_buckets"]
    assert (b["label"] == D.INSUFFICIENT).all()
    crit = by_id(review)
    assert crit["26.6"]["state"] == D.INSUFFICIENT and crit["26.7"]["state"] == D.INSUFFICIENT


# ---------------------------------------------------------------------------
# decide_status (26.19) directly
# ---------------------------------------------------------------------------

CLEAN, DIRTY_RUN = "a" * 40, "a" * 40 + "-dirty"


def _states(**over: str) -> dict[str, dict]:
    crit = {cid: {"id": cid, "state": D.PASS} for cid in D.CRITERIA_IDS}
    crit["26.1"] |= {"evaluated": True, "value": {"bss": 0.01}}
    crit["26.18"]["state"] = D.INSUFFICIENT
    for cid, st in over.items():
        crit[cid.replace("_", ".")]["state"] = st
    return crit


def test_decide_status_mapping():
    kw = {"baseline_trades": 500, "bucket_min_games": 200, "run_commit": CLEAN, "dirty_inputs": ()}
    assert D.decide_status(_states(), **kw)[0] == "RESEARCH-PROMISING"
    assert D.decide_status(_states(**{"26_18": D.PASS}), **kw)[0] == "PROSPECTIVELY-VALIDATED"
    assert D.decide_status(_states(**{"26_12": D.INSUFFICIENT}), **kw)[0] == "INCONCLUSIVE"
    assert D.decide_status(_states(**{"26_3": D.FAIL}), **kw)[0] == "INCONCLUSIVE"
    for cid in ("26_13", "26_11", "26_16"):
        assert D.decide_status(_states(**{cid: D.FAIL}), **kw)[0] == "REJECTED"
    # 26.9 FAIL rejects only with enough baseline trades
    assert D.decide_status(_states(**{"26_9": D.FAIL}), **kw)[0] == "REJECTED"
    assert D.decide_status(_states(**{"26_9": D.FAIL}), baseline_trades=50, bucket_min_games=200,
                           run_commit=CLEAN, dirty_inputs=())[0] == \
        "INCONCLUSIVE"
    holdout_bad = _states()
    holdout_bad["26.1"]["value"] = {"bss": -0.001}
    assert D.decide_status(holdout_bad, **kw)[0] == "REJECTED"


# ---------------------------------------------------------------------------
# Holdout guard
# ---------------------------------------------------------------------------

def test_holdout_period_rows_raise(cfg):
    hold = holdout_period(load_periods(cfg, "synthetic"))
    preds = make_predictions(cfg, games_per_period=40, periods=[hold])
    with pytest.raises(HoldoutLocked):
        D.build_review(make_inputs(cfg, preds), cfg)
    with pytest.raises(HoldoutLocked):
        D.evaluation_report(make_inputs(cfg, preds), cfg)


def test_holdout_dated_rows_raise_even_if_mislabeled(cfg):
    preds = make_predictions(cfg, games_per_period=40)
    preds.loc[:5, "official_date"] = pd.Timestamp("2026-08-01").date()   # period_id still says evaluation
    with pytest.raises(HoldoutLocked):
        D.build_review(make_inputs(cfg, preds), cfg)


def test_holdout_rows_in_sensitivity_raise(cfg):
    preds = make_predictions(cfg, games_per_period=40)
    hold = holdout_period(load_periods(cfg, "synthetic"))
    sens = make_decisions(make_predictions(cfg, games_per_period=20, periods=[hold]), offset=60)
    with pytest.raises(HoldoutLocked):
        D.build_review(make_inputs(cfg, preds, sensitivity={60: sens}), cfg)


# ---------------------------------------------------------------------------
# Artifacts and rendering
# ---------------------------------------------------------------------------

def test_artifacts_json_has_no_nan_and_md_ends_with_status(good_review, tmp_path, cfg):
    inputs = make_inputs(cfg)
    paths = D.write_review_artifacts(good_review, tmp_path, experiments=inputs.experiments,
                                     hypotheses=inputs.hypotheses, holdout_manifest=inputs.holdout_manifest)
    assert {p.name for p in paths.values()} == {"mlb_review.json", "mlb_review.md", "experiment_registry.json",
                                                "holdout_manifest.json"}
    for p in paths.values():
        if p.suffix == ".json":
            text = p.read_text()
            json.loads(text, parse_constant=lambda c: pytest.fail(f"{p.name} contains {c}"))
            assert "NaN" not in text and "Infinity" not in text
    data = json.loads(paths["review_json"].read_text())
    assert data["status"] == good_review["status"]
    assert {c["id"] for c in data["criteria"]} >= set(D.CRITERIA_IDS)
    md = paths["review_md"].read_text().splitlines()
    assert md[-1] == f"STATUS: {good_review['status']}"
    assert D.BANNER in "\n".join(md[:5]) and D.BANNER in "\n".join(md[-5:])
    reg = json.loads(paths["experiment_registry"].read_text())
    assert len(reg["experiments"]) + len(reg["negative_controls"]["experiments"]) == len(inputs.experiments)
    assert all(e["q_value"] is not None for e in reg["experiments"])
    assert all("q_value" not in e and e["control_q"] is not None for e in reg["negative_controls"]["experiments"])


def test_printed_output_ends_with_status_and_banners(good_review):
    console = Console(record=True, file=io.StringIO(), width=120)
    D.print_review(good_review, console)
    lines = [ln for ln in console.export_text().splitlines() if ln.strip()]
    assert lines[-1] == f"STATUS: {good_review['status']}"
    text = "\n".join(lines)
    assert text.count(D.BANNER) == 2
    assert text.index(D.BANNER) < text.index("Sec 26 criteria")


def test_printed_review_lists_the_adr_0020_flags(good_review):
    rf = {"event": D.FLAG_RAISE, "recorded": [{"flag_id": "rf-1", "event": "raise", "backtest_run_id": "bt-a"}],
          "open": [{"flag_id": "rf-1", "backtest_run_id": "bt-a", "git_commit": DIRTY_RUN, "run_created_at": "t"}]}
    console = Console(record=True, file=io.StringIO(), width=200)
    D.print_review(good_review | {"review_flags": rf}, console)
    lines = [ln for ln in console.export_text().splitlines() if ln.strip()]
    text = "\n".join(lines)
    assert "Review flags (ADR 0002):" in text and "this review raised rf-1 (bt-a)" in text
    assert "open flags: 1" in text and "open: rf-1: 26.16 FAIL on bt-a" in text
    # freeze refuses; registration runs once, for the clearing run
    assert "`quant freeze` refuses, and hypothesis registration runs once, for the clearing run" in text
    assert "registration refuse" not in text
    assert lines[-1] == f"STATUS: {good_review['status']}"
    console = Console(record=True, file=io.StringIO(), width=200)
    D.print_review(good_review, console)
    assert "Review flags" not in console.export_text()


def test_no_banner_on_live_dataset(cfg):
    review = D.build_review(make_inputs(cfg, make_predictions(cfg, games_per_period=60), is_synthetic=False,
                                        dataset="live"), cfg)
    console = Console(record=True, file=io.StringIO(), width=120)
    D.print_review(review, console)
    assert D.BANNER not in console.export_text()
    assert review["banner"] is None
    # p_true present but not synthetic: the outcome upper bound governs coverage, and it can only fail
    c12 = by_id(review)["26.12"]
    assert c12["value"]["governing_target"] == "outcome_upper_bound"
    assert c12["state"] in (D.FAIL, D.INSUFFICIENT)


def test_evaluation_report_prints(cfg):
    inputs = make_inputs(cfg, make_predictions(cfg, games_per_period=80))
    rep = D.evaluation_report(inputs, cfg)
    assert list(rep["trading"]) == ["pen_0", "pen_25", "pen_50", "pen_100"]
    assert len(rep["prediction_quality"]["per_period"]) == 6
    console = Console(record=True, file=io.StringIO(), width=160)
    D.print_evaluation(rep, console)
    out = console.export_text()
    assert "Prediction quality" in out and "Model-edge buckets" in out and out.count(D.BANNER) == 2
    json.loads(D.to_json(rep))


def test_scenarios_reprice_same_trades(good_review):
    tr = good_review["informational"]["trading"]
    assert len({t["trades"] for t in tr.values()}) == 1
    nets = [tr[s]["net_pnl"] for s in ("pen_0", "pen_25", "pen_50", "pen_100")]
    assert nets == sorted(nets, reverse=True)


def test_unverified_size_fails_26_10_in_an_executable_run(cfg):
    preds = make_predictions(cfg, games_per_period=100)
    preds.loc[preds.index[::3], "size_verified"] = False
    c = by_id(D.build_review(make_inputs(cfg, preds), cfg))["26.10"]
    assert c["state"] == D.FAIL
    n = int((~preds["size_verified"]).sum())
    assert c["value"]["size_unverified"] == n
    assert f"'predictions: displayed size unverified in an executable run': {n}" in c["detail"]


# -- candle-derived quotes are degraded, never executable ---------------

def candle_inputs(cfg, **over) -> D.ReviewInputs:
    preds = make_predictions(cfg, games_per_period=100)
    preds["size_verified"], preds["quote_quality"] = False, D.CANDLE_DEGRADED
    inputs = make_inputs(cfg, preds, quote_quality=D.CANDLE_DEGRADED, **over)
    inputs.decisions["quote_quality"] = D.CANDLE_DEGRADED
    return inputs


def test_candle_run_26_10_is_insufficient_never_pass(cfg):
    review = D.build_review(candle_inputs(cfg), cfg)
    c = by_id(review)["26.10"]
    assert c["state"] == D.INSUFFICIENT
    assert "not evaluable: candle_degraded quotes carry no displayed size" in c["detail"]
    assert review["quote_quality"] == D.CANDLE_DEGRADED


def test_candle_run_opens_with_degraded_execution_banner(cfg, tmp_path):
    review = D.build_review(candle_inputs(cfg), cfg)
    assert review["execution_banner"] == D.DEGRADED_EXECUTION
    md = D.render_markdown(review).splitlines()
    assert md[2] == f"> **{D.DEGRADED_EXECUTION}**"                 # first thing after the title
    assert "- Quote quality: candle_degraded" in md
    console = Console(record=True, file=io.StringIO(), width=200)
    D.print_review(review, console)
    text = console.export_text()
    assert text.count(D.DEGRADED_EXECUTION) == 2
    assert text.index(D.DEGRADED_EXECUTION) < text.index(D.BANNER) < text.index("Sec 26 criteria")
    rep = D.evaluation_report(candle_inputs(cfg), cfg)
    console = Console(record=True, file=io.StringIO(), width=200)
    D.print_evaluation(rep, console)
    out = console.export_text()
    assert out.count(D.DEGRADED_EXECUTION) == 2 and out.index(D.DEGRADED_EXECUTION) < out.index("Prediction quality")


def test_executable_run_has_no_execution_banner(good_review):
    assert good_review["execution_banner"] is None and good_review["quote_quality"] == D.EXECUTABLE
    assert D.DEGRADED_EXECUTION not in D.render_markdown(good_review)


def test_decide_status_caps_non_executable_at_inconclusive():
    kw = {"baseline_trades": 500, "bucket_min_games": 200, "run_commit": CLEAN, "dirty_inputs": ()}
    for states in (_states(), _states(**{"26_18": D.PASS})):
        assert D.decide_status(states, **kw)[0] in ("RESEARCH-PROMISING", "PROSPECTIVELY-VALIDATED")
        status, reasons = D.decide_status(states, **kw, quote_quality=D.CANDLE_DEGRADED)
        assert status == "INCONCLUSIVE"
        assert reasons == ["capped at INCONCLUSIVE: quote_quality is candle_degraded, not executable"]
    # the cap never hides a rejection
    assert D.decide_status(_states(**{"26_11": D.FAIL}), **kw, quote_quality=D.CANDLE_DEGRADED)[0] == "REJECTED"


@pytest.mark.parametrize("frame_name", ["predictions", "decisions", "sensitivity"])
def test_mixed_quote_quality_is_refused(cfg, frame_name):
    inputs = make_inputs(cfg)
    inputs.predictions["quote_quality"] = D.EXECUTABLE
    inputs.decisions["quote_quality"] = D.EXECUTABLE
    inputs.sensitivity = {60: inputs.decisions.copy()}
    target = inputs.sensitivity[60] if frame_name == "sensitivity" else getattr(inputs, frame_name)
    target.loc[target.index[0], "quote_quality"] = D.CANDLE_DEGRADED
    with pytest.raises(D.MixedQuoteQuality, match="candle_degraded"):
        D.build_review(inputs, cfg)
    with pytest.raises(D.MixedQuoteQuality):
        D.evaluation_report(inputs, cfg)


def test_unknown_quote_quality_is_refused(cfg):
    with pytest.raises(ValueError, match="quote_quality must be one of"):
        D.build_review(make_inputs(cfg, quote_quality="close_enough"), cfg)


def test_stale_quote_fails_26_10(cfg):
    preds = make_predictions(cfg, games_per_period=60)
    preds.loc[0, "quote_age_s"] = 301.0
    assert by_id(D.build_review(make_inputs(cfg, preds), cfg))["26.10"]["state"] == D.FAIL


def test_paper_trading_insufficient_then_evaluated(cfg):
    base = make_inputs(cfg)
    assert by_id(D.build_review(base, cfg))["26.18"]["state"] == D.INSUFFICIENT
    rng = np.random.default_rng(5)
    n = 400
    ts = pd.Timestamp("2026-03-01T22:00Z") + pd.to_timedelta(rng.integers(0, 100, n), unit="D")
    p_true = rng.uniform(0.3, 0.7, n)
    mid = np.clip(p_true + rng.normal(0, 0.05, n), 0.05, 0.95)
    outcome = (rng.random(n) < p_true).astype(int)
    pt = pd.DataFrame({"paper_trade_id": [f"pt{i}" for i in range(n)], "decision": "BUY_YES",
                       "freeze_manifest_hash": "h", "decision_ts": ts, "bid": mid - 0.01, "ask": mid + 0.01,
                       "model_probability": p_true, "hypothetical_size": 100.0})
    ps = pd.DataFrame({"paper_trade_id": pt["paper_trade_id"], "result": np.where(outcome == 1, "yes", "no"),
                       "outcome": outcome, "net_pnl": 100 * (outcome - (mid + 0.01))})
    c = by_id(D.build_review(make_inputs(cfg, paper_trades=pt, paper_settlements=ps), cfg))["26.18"]
    assert c["state"] in (D.PASS, D.FAIL)
    assert c["value"]["settled_trades"] == n and c["value"]["days"] >= cfg["review"]["paper_min_days"]


def test_paper_bss_covers_passes_and_frequency_counts_qualifying_days(cfg):
    """26.18: the paper Brier covers every qualifying decision (passes settle with their outcome), not only
    the BUYs selected on edge; trade count and frequency use BUYs over days with a qualifying slate."""
    M = D.M
    rng = np.random.default_rng(7)
    n_buy, n_pass = 300, 500
    n = n_buy + n_pass
    day = pd.Timestamp("2026-03-01T22:00Z") + pd.to_timedelta(np.arange(n) % 90, unit="D")
    p_true = rng.uniform(0.3, 0.7, n)
    mid = np.clip(p_true + rng.normal(0, 0.05, n), 0.05, 0.95)
    outcome = (rng.random(n) < p_true).astype(int)
    decision = np.where(np.arange(n) < n_buy, "BUY_YES", "PASS")
    pt = pd.DataFrame({"paper_trade_id": [f"pt{i}" for i in range(n)], "decision": decision,
                       "freeze_manifest_hash": "h", "decision_ts": day, "bid": mid - 0.01, "ask": mid + 0.01,
                       "model_probability": p_true, "hypothetical_size": np.where(decision == "PASS", np.nan, 100.0)})
    # a non-qualifying slate on 30 extra days: logged, settled, but outside 26.18
    extra = pt.head(30).assign(paper_trade_id=[f"nq{i}" for i in range(30)], freeze_manifest_hash=None,
                               decision_ts=pd.Timestamp("2026-07-01T22:00Z") + pd.to_timedelta(np.arange(30), "D"))
    pt = pd.concat([pt, extra], ignore_index=True)
    out = np.concatenate([outcome, outcome[:30]])
    ps = pd.DataFrame({"paper_trade_id": pt["paper_trade_id"], "result": np.where(out == 1, "yes", "no"),
                       "outcome": out, "net_pnl": np.where(pt["decision"] == "PASS", np.nan, 100 * (out - pt["ask"]))})
    v = by_id(D.build_review(make_inputs(cfg, paper_trades=pt, paper_settlements=ps), cfg))["26.18"]["value"]
    assert v["settled_trades"] == n_buy and v["settled_decisions"] == n
    y = pd.Series(outcome, dtype=float)
    expected_bss = M.bss(M.brier(y, pd.Series(p_true)), M.brier(y, pd.Series(mid)))
    buys_only = M.bss(M.brier(y[:n_buy], pd.Series(p_true[:n_buy])), M.brier(y[:n_buy], pd.Series(mid[:n_buy])))
    assert v["paper_bss"] == pytest.approx(expected_bss, abs=1e-12)
    assert abs(expected_bss - buys_only) > 1e-6   # the two definitions differ on this data
    assert v["trades_per_day_paper"] == pytest.approx(n_buy / 90)


def test_full_size_review_runtime(tmp_path):
    """~20k predictions at production review settings stays well under 30 s."""
    import time
    full = copy.deepcopy(load_sport_config("mlb"))
    preds = make_predictions(full, games_per_period=1700, seed=11)
    assert len(preds) >= 20_000
    inputs = make_inputs(full, preds)
    t0 = time.perf_counter()
    review = D.build_review(inputs, full)
    D.write_review_artifacts(review, tmp_path, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                             holdout_manifest=inputs.holdout_manifest)
    D.print_review(review, Console(file=io.StringIO(), width=120))
    assert time.perf_counter() - t0 < 30


# ---------------------------------------------------------------------------
# Loader against the in-memory database
# ---------------------------------------------------------------------------

def test_load_review_inputs_from_db(syn_db):
    cfg = load_sport_config("mlb")
    hold_date = pd.Timestamp("2026-07-15").date()
    hold_start = pd.Timestamp("2026-07-15T23:05Z")
    obs = frame(
        observation(outcome=1),
        observation(offset=60, outcome=1, is_canonical=False),
        observation(observation_id="SYN-HOLD@T-30", game_id="syn-mlb-hold", contract_id="SYN-HOLD",
                    official_date=hold_date, season=2026, event_start_time=hold_start,
                    snapshot_time=hold_start - timedelta(minutes=30), quote_ts=hold_start - timedelta(minutes=31),
                    quote_available_at=hold_start - timedelta(minutes=31), outcome=0),
    )
    syn_db.insert("observations", obs)

    def pred(oid, period):
        return {"model_run_id": "mr-1", "observation_id": oid, "period_id": period, "p_market": 0.56,
                "p_model": 0.60, "lower_bound": 0.55, "upper_bound": 0.65, "p_conservative_yes": 0.55,
                "p_conservative_no": 0.35, "is_synthetic": True}
    syn_db.insert("predictions", frame(pred("SYN-NYY@T-30", "2025H1"), pred("SYN-NYY@T-60", "2025H1"),
                                       pred("SYN-HOLD@T-30", "2026H2")))

    def run(rid, off, created):
        return {"backtest_run_id": rid, "model_type": "logistic_residual", "snapshot_offset_min": off,
                "config_json": json.dumps({"model_type": "logistic_residual"}), "config_hash": "c",
                "quote_quality": "executable", "dataset": "synthetic", "created_at": pd.Timestamp(created),
                "is_synthetic": True}
    # one invocation: the engine writes every offset with one created_at (ADR 0001 N1)
    syn_db.insert("backtest_runs", frame(run("bt-30", 30, "2026-02-01T00:00Z"), run("bt-60", 60, "2026-02-01T00:00Z")))

    def dec(rid, oid, period, snap):
        return {"backtest_run_id": rid, "scenario": "pen_25", "model_run_id": "mr-1", "observation_id": oid,
                "game_id": "g", "contract_id": "c", "period_id": period, "snapshot_time": snap, "decision": "PASS",
                "pass_reason": "below_min_edge", "side": "yes", "p_market": 0.56, "p_model": 0.60,
                "predicted_edge": 0.03, "quoted_price": 0.57, "spread_penalty_frac": 0.25, "outcome": 1,
                "side_won": 1, "exit_rule": "hold_to_settlement", "is_synthetic": True}
    syn_db.insert("backtest_decisions", frame(
        dec("bt-30", "SYN-NYY@T-30", "2025H1", obs["snapshot_time"][0]),
        dec("bt-30", "SYN-HOLD@T-30", "2026H2", obs["snapshot_time"][2]),
        dec("bt-60", "SYN-NYY@T-60", "2025H1", obs["snapshot_time"][1])))

    inputs = D.load_review_inputs(syn_db, cfg, holdout_manifest={"status": "LOCKED"},
                                  leakage_audit=D.run_leakage_audit(syn_db),
                                  truth={"SYN-NYY@T-30": 0.58})
    assert inputs.backtest_run["backtest_run_id"] == "bt-30"
    assert inputs.backtest_run["model_type"] == "logistic_residual"
    assert list(inputs.predictions["observation_id"]) == ["SYN-NYY@T-30"]          # holdout row dropped
    assert list(inputs.decisions["observation_id"]) == ["SYN-NYY@T-30"]
    assert inputs.predictions["p_true"].iloc[0] == pytest.approx(0.58)
    assert inputs.predictions["availability_state"].iloc[0] == "no_features"
    assert set(inputs.sensitivity) == {60} and len(inputs.sensitivity[60]) == 1
    assert all(v == 0 for v in inputs.leakage_audit.values())
    assert inputs.observation_counts["canonical"] == 1                             # holdout excluded
    review = D.build_review(inputs, cfg)                                           # tiny sample: runs, no raise
    assert review["status"] in D.STATUSES
    assert by_id(review)["26.2"]["state"] == D.INSUFFICIENT


def test_loader_keeps_quote_quality_classes_apart(syn_db):
    """An executable and a candle_degraded run in one database never share a review."""
    from src.backtest.registry import record_experiments
    cfg = load_sport_config("mlb")
    candle = dict(bid_size=None, ask_size=None, size_verified=False, quote_source="candle")
    syn_db.insert("observations", frame(
        observation(outcome=1),
        observation(observation_id="SYN-C@T-30", game_id="syn-mlb-2", contract_id="SYN-C", outcome=0, **candle)))
    syn_db.insert("predictions", frame(*[
        {"model_run_id": "mr-1", "observation_id": oid, "period_id": "2025H1", "p_market": 0.56, "p_model": 0.60,
         "lower_bound": 0.55, "upper_bound": 0.65, "p_conservative_yes": 0.55, "p_conservative_no": 0.35,
         "is_synthetic": True} for oid in ("SYN-NYY@T-30", "SYN-C@T-30")]))
    syn_db.insert("backtest_runs", frame(*[
        {"backtest_run_id": rid, "model_type": "logistic_residual", "snapshot_offset_min": 30,
         "config_json": "{}", "config_hash": "c", "quote_quality": q, "dataset": "synthetic",
         "created_at": pd.Timestamp(created), "is_synthetic": True}
        for rid, q, created in (("bt-exec", "executable", "2026-02-01T00:00Z"),
                                ("bt-candle", "candle_degraded", "2026-02-01T00:05Z"),
                                ("bt-mixed", "executable", "2026-01-01T00:00Z"))]))
    snap = observation()["snapshot_time"]
    syn_db.insert("backtest_decisions", frame(*[
        {"backtest_run_id": rid, "scenario": "pen_25", "model_run_id": "mr-1", "observation_id": oid,
         "game_id": "g", "contract_id": "c", "period_id": "2025H1", "snapshot_time": snap, "decision": "PASS",
         "pass_reason": "below_min_edge", "side": "yes", "p_market": 0.56, "p_model": 0.60, "predicted_edge": 0.03,
         "quoted_price": 0.57, "spread_penalty_frac": 0.25, "outcome": 1, "side_won": 1,
         "exit_rule": "hold_to_settlement", "is_synthetic": True}
        for rid, oid in (("bt-exec", "SYN-NYY@T-30"), ("bt-candle", "SYN-C@T-30"),
                         ("bt-mixed", "SYN-NYY@T-30"), ("bt-mixed", "SYN-C@T-30"))]))
    metrics = json.dumps({"reference": "market", "candidate": "m", "n_obs": 10, "brier_ref": 0.25,
                          "brier_cand": 0.24, "delta_brier": 0.25 - 0.24})
    record_experiments(syn_db, [
        {"hypothesis_id": "H1", "experiment_family": "market", "label": "CONFIRMATORY", "feature_set_id": "fs",
         "feature_set_hash": "fsh", "model_type": "logistic_residual", "config_hash": "c", "train_period": "2023H1",
         "test_period": "2025H1", "n_games": 10, "metrics_json": metrics, "p_value": 0.5, "quote_quality": q}
        for q in ("executable", "candle_degraded")])

    latest = D.load_review_inputs(syn_db, cfg)                                  # newest T-30 run is the candle one
    assert latest.backtest_run["backtest_run_id"] == "bt-candle"
    assert latest.quote_quality == D.CANDLE_DEGRADED
    assert set(latest.experiments["quote_quality"]) == {D.CANDLE_DEGRADED}
    assert latest.observation_counts["canonical"] == 1
    assert D.build_review(latest, cfg)["execution_banner"] == D.DEGRADED_EXECUTION

    ex = D.load_review_inputs(syn_db, cfg, quote_quality=D.EXECUTABLE)
    assert ex.backtest_run["backtest_run_id"] == "bt-exec" and ex.quote_quality == D.EXECUTABLE
    assert list(ex.predictions["observation_id"]) == ["SYN-NYY@T-30"]
    assert set(ex.experiments["quote_quality"]) == {D.EXECUTABLE}
    assert ex.observation_counts["canonical"] == 1
    assert D.build_review(ex, cfg)["execution_banner"] is None

    mixed = D.load_review_inputs(syn_db, cfg, backtest_run_id="bt-mixed")
    with pytest.raises(D.MixedQuoteQuality, match="candle_degraded"):
        D.build_review(mixed, cfg)


# ---------------------------------------------------------------------------
# Referee fixes
# ---------------------------------------------------------------------------

def _rebuild_truth(preds: pd.DataFrame, p_true_home: pd.Series, rng) -> pd.DataFrame:
    """Replace the generator truth with per-game `p_true_home` (indexed by game_id) and redraw outcomes;
    the away contract is the complement, as in make_predictions."""
    p = preds.copy()
    home = (p["yes_side"] == "home").to_numpy()
    won = pd.Series((rng.random(len(p_true_home)) < p_true_home.to_numpy()).astype(int), index=p_true_home.index)
    pt, w = p["game_id"].map(p_true_home), p["game_id"].map(won)
    p["p_true"], p["outcome"] = np.where(home, pt, 1 - pt), np.where(home, w, 1 - w)
    return p


def _set_intervals(p: pd.DataFrame, half) -> pd.DataFrame:
    p["lower_bound"], p["upper_bound"] = np.clip(p["p_model"] - half, 0, 1), np.clip(p["p_model"] + half, 0, 1)
    p["p_conservative_yes"], p["p_conservative_no"] = p["lower_bound"], 1 - p["upper_bound"]
    return p


def _home_model(p: pd.DataFrame) -> pd.Series:
    home = p[p["yes_side"] == "home"]
    return pd.Series(home["p_model"].to_numpy(), index=home["game_id"].to_numpy())


def test_26_12_off_synthetic_never_passes_intervals_tuned_to_the_binned_proxy():
    """The old proxy's failure, reproduced where p_true is known but does not govern (not synthetic).

    p_true scatters around p_model per game (sd 0.08), so bin frequencies track p_model; intervals are
    sized to cover the binned-frequency target 90% of the time, as uncertainty.py tunes them. True coverage
    is far below the band. The retired proxy read 0.90 and PASSED; the outcome upper bound cannot see the
    spread, so it must not PASS (INSUFFICIENT), and it must stay above the truth (it is a bound)."""
    cfg = small_cfg(coverage_bin_min_obs=500)
    rng = np.random.default_rng(12)
    p = make_predictions(cfg, games_per_period=1000, seed=11)
    pm = _home_model(p)
    p = _rebuild_truth(p, pd.Series(np.clip(pm.to_numpy() + rng.normal(0, 0.08, len(pm)), 0.01, 0.99),
                                    index=pm.index), rng)
    proxy = D.M.binned_frequency_target(p["outcome"], p["p_model"], 500)
    p = _set_intervals(p, float(np.quantile(np.abs(proxy - p["p_model"]), 0.90)))
    c = by_id(D.build_review(make_inputs(cfg, p, is_synthetic=False, dataset="live"), cfg))["26.12"]
    assert c["value"]["coverage_true"] < 0.6                        # known: the intervals badly under-cover
    assert c["state"] != D.PASS
    assert c["value"]["governing_target"] == "outcome_upper_bound" and c["value"]["coverage"] is None
    assert c["value"]["coverage_upper_bound"] >= c["value"]["coverage_true"] and c["value"]["bound_holds"]


def test_26_12_off_synthetic_bound_fails_gross_miscalibration():
    """p_true = 0.5 + 3 (p_model - 0.5) with +/-0.02 intervals: the outer bins' outcome frequencies lie so far
    outside their intervals that most rows must be uncovered. The bound shows it and 26.12 FAILs."""
    cfg = small_cfg(coverage_bin_min_obs=500)
    rng = np.random.default_rng(22)
    p = make_predictions(cfg, games_per_period=1000, seed=21)
    pm = _home_model(p)
    p = _set_intervals(_rebuild_truth(p, np.clip(0.5 + 3 * (pm - 0.5), 0.01, 0.99), rng), 0.02)
    c = by_id(D.build_review(make_inputs(cfg, p, is_synthetic=False, dataset="live"), cfg))["26.12"]
    assert c["state"] == D.FAIL
    v = c["value"]
    assert v["coverage_true"] <= v["coverage_upper_bound"] < cfg["uncertainty"]["coverage_band"][0]
    assert v["bound_bins_binding"] > 0 and "under-coverage shown by outcomes" in c["detail"]


def test_26_12_synthetic_reports_the_bound_next_to_the_truth(good_review):
    c = by_id(good_review)["26.12"]
    v = c["value"]
    assert v["governing_target"] == "p_true" and v["coverage"] == v["coverage_true"]
    assert v["coverage_upper_bound"] is not None and v["bound_holds"] is True
    assert "coverage_binned_proxy" not in v
    per = good_review["supporting"]["26.12"]["per_period"]
    assert {"coverage_true", "coverage_upper_bound"} <= set(per.columns)
    assert (per["coverage_upper_bound"] >= per["coverage_true"]).all()


def test_zero_trades_are_insufficient_in_26_8_and_26_9(cfg):
    preds = make_predictions(cfg)
    review = D.build_review(make_inputs(cfg, preds, decisions=make_decisions(preds, minimum_edge=10.0)), cfg)
    crit = by_id(review)
    assert crit["26.8"]["state"] == D.INSUFFICIENT and crit["26.9"]["state"] == D.INSUFFICIENT
    assert crit["26.9"]["value"]["baseline_trades"] == 0
    assert crit["26.9"]["detail"].startswith("no baseline trades")
    assert review["status"] == "INCONCLUSIVE"


def test_losing_baseline_with_trades_still_fails_26_9(cfg):
    preds = make_predictions(cfg)
    dec = make_decisions(preds, minimum_edge=cfg["decision"]["minimum_edge"])
    traded = dec["decision"] != "PASS"
    assert traded.sum() > 0
    dec.loc[traded, "net_pnl"] = -abs(dec.loc[traded, "net_pnl"]) - 1.0
    dec.loc[traded, "gross_pnl"] = dec.loc[traded, "net_pnl"]
    c9 = by_id(D.build_review(make_inputs(cfg, preds, decisions=dec), cfg))["26.9"]
    assert c9["state"] == D.FAIL and c9["value"]["baseline_trades"] > 0


def test_26_9_insufficient_changes_no_status():
    """26.9 is in PROMISING_REQUIRES, so FAIL and INSUFFICIENT both block RESEARCH-PROMISING; FAIL rejects
    only with >= bucket_min_games baseline trades, which zero trades never reach. Same status either way."""
    for kw in ({"baseline_trades": 0, "bucket_min_games": 200, "run_commit": CLEAN, "dirty_inputs": ()},
               {"baseline_trades": 0, "bucket_min_games": 1, "run_commit": CLEAN, "dirty_inputs": ()}):
        fail = D.decide_status(_states(**{"26_9": D.FAIL}), **kw)
        insufficient = D.decide_status(_states(**{"26_9": D.INSUFFICIENT}), **kw)
        assert fail[0] == insufficient[0] == "INCONCLUSIVE"
        assert insufficient[1] == ["not RESEARCH-PROMISING: 26.9 INSUFFICIENT SAMPLE"]
    assert "26.9" in D.PROMISING_REQUIRES


def test_26_2_tiny_positive_bss_periods_do_not_count_as_improved():
    """The 26.2 floor: "a period counts as improved only if its BSS >= 0.001". A model 2% of the way from the
    market to the truth beats the market in every period by a BSS far below the floor; sign-only counting
    PASSED it."""
    cfg = small_cfg()
    assert cfg["review"]["period_bss_floor"] == 0.001
    review = D.build_review(make_inputs(cfg, make_predictions(cfg, signal=0.02)), cfg)
    c, tbl = by_id(review)["26.2"], review["supporting"]["26.2"]["per_period"]
    assert (tbl["bss"] > 0).sum() >= 5                               # sign-only: improved almost everywhere
    assert c["state"] == D.FAIL and c["threshold"]["period_bss_floor"] == 0.001
    assert c["value"]["periods_improved"] == int((tbl["bss"] >= 0.001).sum()) < 5
    assert "improved (BSS >= 0.001)" in c["detail"]


def test_26_2_floor_is_inclusive():
    """A period whose BSS equals the floor counts; the next float above it does not."""
    base_cfg = small_cfg()
    preds = make_predictions(base_cfg)
    bss = sorted(D.build_review(make_inputs(base_cfg, preds), base_cfg)["supporting"]["26.2"]["per_period"]["bss"])
    fifth = bss[-5]                                                   # 5 of 6 periods needed
    at = small_cfg(period_bss_floor=fifth)
    above = small_cfg(period_bss_floor=float(np.nextafter(fifth, np.inf)))
    c_at = by_id(D.build_review(make_inputs(at, preds), at))["26.2"]
    c_above = by_id(D.build_review(make_inputs(above, preds), above))["26.2"]
    assert c_at["value"]["periods_improved"] == 5 and c_at["state"] == D.PASS
    assert c_above["value"]["periods_improved"] == 4 and c_above["state"] == D.FAIL


def test_26_4_registration_at_the_experiment_instant_is_not_before_testing(cfg):
    """`quant train` stamps its registration and its experiments with one timestamp; strict order fails it."""
    hyps, exps = make_registry(cfg)
    hyps["registered_at"] = pd.to_datetime(exps["created_at"]).min()
    c = by_id(D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg))["26.4"]
    assert c["state"] == D.FAIL
    assert set(c["value"]["missing"]) == set(c["value"]["registered_at_reference_instant"]) == {
        "market", "starter", "negative_control"}
    assert "same instant as the reference" in c["detail"]
    hyps["registered_at"] = pd.to_datetime(exps["created_at"]).min() - pd.Timedelta(microseconds=1)
    assert by_id(D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg))["26.4"]["state"] == D.PASS


def _c264_from_db(db) -> dict:
    from types import SimpleNamespace
    from src.backtest.registry import hypotheses_with_status
    inputs = SimpleNamespace(hypotheses=hypotheses_with_status(db), experiments=db.df("SELECT * FROM experiments"),
                             model_selection={"feature_families": []}, backtest_run={})
    return D._c264(inputs)[0]


def _hyp_file(tmp_path):
    import yaml
    h = {"hypothesis_id": "H_T", "family": "starter", "description": "d", "mechanism": "m",
         "features": [{"name": "f", "definition": "fd"}], "expected_direction": "positive", "minimum_sample": 10,
         "evaluation_periods": ["2023H2"], "primary_metric": "delta_brier", "label": "CONFIRMATORY"}
    path = tmp_path / "h.yaml"
    path.write_text(yaml.safe_dump({"hypotheses": [h]}))
    return path


def _exp(created_at) -> dict:
    metrics = json.dumps({"reference": "market", "candidate": "m", "n_obs": 10, "brier_ref": 0.25,
                          "brier_cand": 0.24, "delta_brier": 0.01})
    return {"hypothesis_id": "H_T", "experiment_family": "primary", "label": "CONFIRMATORY", "feature_set_id": "fs",
            "feature_set_hash": "fsh", "model_type": "logistic_residual", "config_hash": "c", "train_period": "2023H1",
            "test_period": "2023H2", "n_games": 10, "metrics_json": metrics, "p_value": 0.5,
            "quote_quality": "executable", "created_at": created_at}


def test_26_4_db_same_timestamp_as_train_fails(syn_db, tmp_path):
    """The root cause, reproduced: registration and experiments share train's `now`."""
    from datetime import datetime, timezone
    from src.backtest.registry import load_hypotheses, record_experiments, register_hypotheses
    now = datetime.now(timezone.utc)
    register_hypotheses(syn_db, load_hypotheses(_hyp_file(tmp_path)), now=now)
    record_experiments(syn_db, [_exp(now)])
    c = _c264_from_db(syn_db)
    assert c["state"] == D.FAIL and c["value"]["registered_at_reference_instant"] == ["starter"]


def test_26_4_db_register_now_then_experiments_passes(syn_db, tmp_path):
    from datetime import datetime, timezone
    from src.backtest.registry import record_experiments, register_now
    n, reg = register_now(syn_db, _hyp_file(tmp_path))
    assert n == 1
    record_experiments(syn_db, [_exp(datetime.now(timezone.utc))])
    c = _c264_from_db(syn_db)
    assert c["state"] == D.PASS and c["value"]["registered_at_reference_instant"] == []


# -- evaluator versioning -------------------------------------------------------------------

def test_referee_version_and_config_hash_are_stamped(good_review, cfg, tmp_path):
    h = D.referee_config_hash(cfg)
    assert good_review["referee_version"] == D.REFEREE_VERSION == "referee-v2"
    assert good_review["referee_config_hash"] == h and len(h) == 16
    assert f"- Referee: referee-v2, config hash {h}" in D.render_markdown(good_review).splitlines()
    console = Console(record=True, file=io.StringIO(), width=160)
    D.print_review(good_review, console)
    assert f"Referee referee-v2  config hash {h}" in console.export_text()
    inputs = make_inputs(cfg)
    paths = D.write_review_artifacts(good_review, tmp_path, experiments=inputs.experiments,
                                     hypotheses=inputs.hypotheses, holdout_manifest=inputs.holdout_manifest)
    data = json.loads(paths["review_json"].read_text())
    assert data["referee_version"] == "referee-v2" and data["referee_config_hash"] == h


def test_referee_config_hash_is_stable_and_covers_the_review_section(cfg):
    h = D.referee_config_hash(cfg)
    reordered = copy.deepcopy(cfg)
    reordered["review"] = dict(reversed(list(cfg["review"].items())))
    assert D.referee_config_hash(reordered) == h
    other = copy.deepcopy(cfg)
    other["decision"]["minimum_edge"] = 0.5                           # not in the review section: not covered
    assert D.referee_config_hash(other) == h
    for key, value in (("period_bss_floor", 0.002), ("coverage_bound_alpha", 0.1),
                       ("fdr_families", ["ablation", "hypothesis"])):
        changed = copy.deepcopy(cfg)
        changed["review"][key] = value
        assert D.referee_config_hash(changed) != h, key


# -- 26.5: experiment rows without a commit (spec Sec 20 tracks git_commit per experiment) --------

@pytest.mark.parametrize("missing", [None, np.nan, "", "  ", "unavailable: not a git repository"])
def test_26_5_flags_an_experiment_row_without_a_commit(cfg, missing):
    hyps, exps = make_registry(cfg)
    exps["git_commit"] = exps["git_commit"].astype(object)
    exps.loc[exps.index[0], "git_commit"] = missing
    c = by_id(D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg))["26.5"]
    assert c["state"] == D.FAIL and c["value"]["missing_git_commit"] == 1
    assert "1 experiment(s) have no git commit" in c["detail"]


def test_26_5_counts_rows_without_a_commit_even_on_the_holdout(cfg):
    """The count covers every experiment row, including one tested on the holdout (26.3 fails it too)."""
    hyps, exps = make_registry(cfg)
    extra = exps.iloc[[0]].assign(experiment_id="e-peek", test_period="2026H2", git_commit=None)
    c = by_id(D.build_review(make_inputs(cfg, experiments=pd.concat([exps, extra]), hypotheses=hyps), cfg))["26.5"]
    assert c["state"] == D.FAIL and c["value"]["missing_git_commit"] == 1


def test_26_5_no_git_commit_column_flags_every_row(cfg):
    hyps, exps = make_registry(cfg)
    c = by_id(D.build_review(make_inputs(cfg, experiments=exps.drop(columns="git_commit"), hypotheses=hyps),
                             cfg))["26.5"]
    assert c["state"] == D.FAIL and c["value"]["missing_git_commit"] == len(exps)


def test_26_5_passes_when_every_experiment_has_a_commit(good_review):
    c = by_id(good_review)["26.5"]
    assert c["state"] == D.PASS and c["value"]["missing_git_commit"] == 0


def test_26_5_reports_missing_p_values_and_commits_together(cfg):
    hyps, exps = make_registry(cfg)
    exps.loc[exps.index[0], "p_value"] = np.nan
    exps["git_commit"] = exps["git_commit"].astype(object)
    exps.loc[exps.index[1], "git_commit"] = None
    c = by_id(D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg))["26.5"]
    assert c["state"] == D.FAIL and "lack a p-value" in c["detail"] and "have no git commit" in c["detail"]


def test_a_missing_commit_fails_26_5_and_caps_the_run(cfg):
    """26.5 is still not in PROMISING_REQUIRES or the REJECTED list. The status is capped by the feeding-row
    rule instead (superseding this test's earlier "changes no status")."""
    assert "26.5" not in D.PROMISING_REQUIRES
    hyps, exps = make_registry(cfg)
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    flagged = D.build_review(make_inputs(cfg, experiments=exps.assign(git_commit=None), hypotheses=hyps), cfg)
    assert by_id(flagged)["26.5"]["state"] == D.FAIL and flagged["status"] == "INCONCLUSIVE"
    assert not _feed_caps(base)
    assert f"experiments: {len(exps)} of {len(exps)} rows" in _feed_caps(flagged)[0]


# -- 26.13: calibration slope with a game-bootstrap CI ---------------------------

def _slope_ci(review: dict) -> tuple[float, float, float]:
    v = by_id(review)["26.13"]["value"]
    return v["slope_model"], v["slope_ci_low"], v["slope_ci_high"]


def test_26_13_reports_the_slope_with_a_game_bootstrap_ci(good_review, cfg):
    c = by_id(good_review)["26.13"]
    slope, lo, hi = _slope_ci(good_review)
    assert lo < slope < hi and c["state"] == D.PASS
    rc = cfg["review"]
    assert c["threshold"]["ci_level"] == rc["ci_level"] == 0.95
    assert c["threshold"]["resamples"] == rc["bootstrap_resamples"] and c["threshold"]["unit"] == "game"
    assert f"slope {slope:.3f} (95% CI {lo:.3f}–{hi:.3f})" in c["detail"]
    assert c["value"]["slope_ci_level"] == 0.95 and c["value"]["slope_resamples_valid"] == rc["bootstrap_resamples"]


def test_26_13_ci_is_the_game_bootstrap_under_the_review_seed_and_resamples(cfg):
    """The review's CI is exactly calibration_slope_bootstrap over games, with review.seed,
    review.bootstrap_resamples and review.ci_level."""
    inputs = make_inputs(cfg)
    preds, *_ = D._prepare(inputs)
    rc = cfg["review"]
    want = D.M.calibration_slope_bootstrap(preds["y"], preds["p_model"], preds["game_id"],
                                           n=int(rc["bootstrap_resamples"]), seed=int(rc["seed"]),
                                           ci=float(rc["ci_level"]))
    got = _slope_ci(D.build_review(inputs, cfg))
    assert got == (want["slope"], want["ci_low"], want["ci_high"])
    reseeded = small_cfg(seed=int(rc["seed"]) + 1)
    assert _slope_ci(D.build_review(make_inputs(reseeded), reseeded))[1:] != got[1:]


def test_26_13_point_outside_the_band_with_an_overlapping_ci_does_not_fail(cfg):
    """A point estimate outside the band FAILed under referee-v2's point rule; the CI rule FAILs only
    when the whole CI lies outside the band."""
    preds = make_predictions(cfg, miscal=1.3)
    slope, lo, hi = _slope_ci(D.build_review(make_inputs(cfg, preds), cfg))
    assert lo < slope < hi
    for band in ([0.0, (lo + slope) / 2], [(slope + hi) / 2, 5.0]):   # point outside, CI overlapping
        c = small_cfg(calibration_slope_band=band)
        crit = by_id(D.build_review(make_inputs(c, preds), c))["26.13"]
        assert not band[0] <= crit["value"]["slope_model"] <= band[1]
        assert crit["state"] == D.PASS, band
        assert "point estimate outside" in crit["detail"]


def test_26_13_whole_ci_outside_the_band_fails_on_either_side_and_touching_does_not(cfg):
    preds = make_predictions(cfg, miscal=1.3)
    _, lo, hi = _slope_ci(D.build_review(make_inputs(cfg, preds), cfg))
    cases = {(float(np.nextafter(hi, np.inf)), 5.0): D.FAIL,        # CI wholly below the band
             (0.0, float(np.nextafter(lo, -np.inf))): D.FAIL,       # CI wholly above the band
             (hi, 5.0): D.PASS, (0.0, lo): D.PASS}                  # CI touches the band: overlaps
    for band, state in cases.items():
        c = small_cfg(calibration_slope_band=list(band))
        review = D.build_review(make_inputs(c, preds), c)
        assert by_id(review)["26.13"]["state"] == state, band
        assert (review["status"] == "REJECTED") == (state == D.FAIL), band


def test_26_13_without_a_ci_is_insufficient_not_pass(cfg):
    c = small_cfg(bootstrap_resamples=0)
    crit = by_id(D.build_review(make_inputs(c), c))["26.13"]
    assert crit["state"] == D.INSUFFICIENT and "could not be estimated" in crit["detail"]


# -- 26.5 / 26.16: negative controls are not in the BH-FDR set ------------------

def test_fdr_families_are_a_config_value():
    assert load_sport_config("mlb")["review"]["fdr_families"] == ["ablation", "hypothesis", "model"]


def test_negative_controls_get_no_fdr_q_and_fdr_families_keep_theirs(cfg, good_review):
    hyps, exps = make_registry(cfg)
    reg = D.experiment_registry(exps, fdr_q=0.1, fdr_families=cfg["review"]["fdr_families"])
    nc = reg["experiment_family"] == "negative_control"
    assert nc.sum() == 6 and reg.loc[nc, "q_value"].isna().all() and not reg.loc[nc, "success"].any()
    for fam in ("hypothesis", "ablation"):
        rows = reg["experiment_family"] == fam
        np.testing.assert_allclose(reg.loc[rows, "q_value"], D.M.benjamini_hochberg(reg.loc[rows, "p_value"]))
    c = by_id(good_review)["26.5"]
    assert c["state"] == D.PASS
    assert (c["value"]["fdr_families"], c["value"]["fdr_experiments"], c["value"]["negative_controls"]) == (2, 12, 6)
    assert c["threshold"]["fdr_families"] == ["ablation", "hypothesis", "model"]
    assert not set(good_review["supporting"]["26.5"]["registry"]["experiment_family"]) & {"negative_control"}


def test_an_experiment_family_outside_the_fdr_set_fails_26_5_closed(cfg):
    """"exploratory" was this test's example until ADR 0001 Q1 gave that family its own section; any other
    family outside the set still fails closed."""
    hyps, exps = make_registry(cfg)
    extra = exps[exps["experiment_family"] == "hypothesis"].assign(
        experiment_family="adhoc", experiment_id=lambda d: d["experiment_id"] + "-x")
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    review = D.build_review(make_inputs(cfg, experiments=pd.concat([exps, extra]), hypotheses=hyps), cfg)
    c = by_id(review)["26.5"]
    assert c["state"] == D.FAIL and c["value"]["unassigned_families"] == ["adhoc"]
    assert "adhoc" in c["detail"] and "fail-closed" in c["detail"]
    reg = review["supporting"]["26.5"]["registry"]
    assert reg.loc[reg["experiment_family"] == "adhoc", "q_value"].isna().all()
    assert review["status"] == base["status"]                          # 26.5 moves no status
    narrow = small_cfg(fdr_families=["hypothesis"])                    # the set is read from config
    c = by_id(D.build_review(make_inputs(narrow, experiments=exps, hypotheses=hyps), narrow))["26.5"]
    assert c["state"] == D.FAIL and c["value"]["unassigned_families"] == ["ablation"]


def test_26_5_fails_closed_when_the_fdr_set_names_the_controls_or_holds_no_experiment(cfg):
    hyps, exps = make_registry(cfg)
    named = small_cfg(fdr_families=["ablation", "hypothesis", "negative_control"])
    review = D.build_review(make_inputs(named, experiments=exps, hypotheses=hyps), named)
    c = by_id(review)["26.5"]
    assert c["state"] == D.FAIL and "names negative_control" in c["detail"]
    controls = review["negative_controls"]["experiments"]
    assert len(controls) == 6 and controls["control_q"].notna().all()              # still controls, not FDR rows
    assert "negative_control" not in set(review["supporting"]["26.5"]["registry"]["experiment_family"])
    only = exps[exps["experiment_family"] == "negative_control"]
    c = by_id(D.build_review(make_inputs(cfg, experiments=only, hypotheses=hyps), cfg))["26.5"]
    assert c["state"] == D.FAIL and "no experiment in an FDR family" in c["detail"]


def test_negative_controls_have_their_own_section(good_review, cfg, tmp_path):
    sec = good_review["negative_controls"]
    assert sec["label"] == "Negative controls (sanity checks, not hypotheses)"
    rows = sec["experiments"]
    assert len(rows) == 6 and set(rows["experiment_family"]) == {"negative_control"}
    np.testing.assert_allclose(rows["control_q"], D.M.benjamini_hochberg(rows["p_value"]))
    assert "q_value" not in rows and "success" not in rows
    md = D.render_markdown(good_review)
    assert "## Negative controls (sanity checks, not hypotheses)" in md.splitlines()
    console = Console(record=True, file=io.StringIO(), width=200)
    D.print_review(good_review, console)
    assert "Negative controls (sanity checks, not hypotheses)" in console.export_text()
    inputs = make_inputs(cfg)
    paths = D.write_review_artifacts(good_review, tmp_path, experiments=inputs.experiments,
                                     hypotheses=inputs.hypotheses, holdout_manifest=inputs.holdout_manifest)
    reg = json.loads(paths["experiment_registry"].read_text())
    assert reg["negative_controls"]["label"] == sec["label"] and len(reg["negative_controls"]["experiments"]) == 6
    assert {e["experiment_family"] for e in reg["experiments"]} == {"hypothesis", "ablation"}
    assert [f["experiment_family"] for f in reg["fdr_families"]] == ["ablation", "hypothesis"]


def test_the_registry_artifact_reads_the_fdr_set_from_the_review_config(tmp_path):
    narrow = small_cfg(fdr_families=["hypothesis"])
    inputs = make_inputs(narrow)
    review = D.build_review(inputs, narrow)
    paths = D.write_review_artifacts(review, tmp_path, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                                     holdout_manifest=inputs.holdout_manifest)
    reg = json.loads(paths["experiment_registry"].read_text())
    assert reg["fdr_family_set"] == ["hypothesis"] and reg["unassigned_families"] == ["ablation"]
    assert [f["experiment_family"] for f in reg["fdr_families"]] == ["hypothesis"]
    assert all(e["q_value"] is None for e in reg["experiments"] if e["experiment_family"] == "ablation")


def test_26_16_reads_bh_within_the_controls_and_its_verdict_is_unchanged(cfg):
    """26.16's q is BH within the non-holdout negative-control rows, as before; it now lives in the
    controls section as control_q, a 26.16 statistic."""
    hyps, exps = make_registry(cfg)
    review = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    nc = exps[exps["experiment_family"] == "negative_control"]
    t16 = review["supporting"]["26.16"]["negative_controls"]
    assert t16["pooled_q"].tolist() == [float(np.nanmin(D.M.benjamini_hochberg(nc["p_value"])))]
    assert by_id(review)["26.16"]["state"] == D.PASS
    exps.loc[nc.index, "metrics_json"] = json.dumps({"delta_brier": 0.003})
    for p, state in ((0.001, D.FAIL), (0.5, D.PASS)):
        exps.loc[nc.index, "p_value"] = p
        assert by_id(D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg))["26.16"]["state"] \
            == state, p


@pytest.mark.parametrize("pooled", [True, False])
def test_26_16_compares_control_q_not_the_raw_p_with_fdr_q(cfg, pooled):
    """Seven control rows, one with p 0.02: its control_q is 0.02 * 7 = 0.14 (0.02 * 6 = 0.12 without the pooled
    row), above fdr_q 0.10, so no control works. The raw p, 0.02, would say one does."""
    hyps, exps = make_registry(cfg)
    nc = exps["experiment_family"] == "negative_control"
    exps.loc[nc, "metrics_json"] = json.dumps({"delta_brier": 0.003})           # improves in every period
    if pooled:
        exps = pd.concat([exps, exps[nc].iloc[:1].assign(experiment_id="e-negative_control-pooled",
                                                         test_period="pooled", p_value=0.02)], ignore_index=True)
    else:
        exps.loc[exps.index[nc][0], "p_value"] = 0.02
    review = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    t16 = review["supporting"]["26.16"]["negative_controls"]
    assert t16["q_source"].iloc[0] == ("pooled row" if pooled else "min over period rows (no pooled row recorded)")
    assert t16["pooled_q"].iloc[0] == pytest.approx(0.02 * (7 if pooled else 6))
    assert t16["pooled_q"].iloc[0] > cfg["review"]["fdr_q"] > 0.02
    assert t16["periods_improved"].iloc[0] == t16["periods"].iloc[0] == 6
    assert by_id(review)["26.16"]["state"] == D.PASS


def test_a_control_tested_on_the_holdout_gets_no_control_q_and_does_not_move_26_16(cfg):
    """A control row whose test period is the holdout is listed (flagged) but, as before, is outside BH and 26.16."""
    hyps, exps = make_registry(cfg)
    hold = holdout_period(load_periods(cfg, "synthetic"))
    held = exps[exps["experiment_family"] == "negative_control"].iloc[:1].assign(
        experiment_id="e-negative_control-holdout", test_period=hold.period_id, p_value=0.0001,
        metrics_json=json.dumps({"delta_brier": 0.003}))
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    review = D.build_review(make_inputs(cfg, experiments=pd.concat([exps, held], ignore_index=True),
                                        hypotheses=hyps), cfg)
    rows = review["negative_controls"]["experiments"]
    on_hold = rows["tested_on_holdout"].astype(bool)
    assert on_hold.sum() == 1 and rows.loc[on_hold, "control_q"].isna().all()
    np.testing.assert_allclose(rows.loc[~on_hold, "control_q"], base["negative_controls"]["experiments"]["control_q"])
    pd.testing.assert_frame_equal(review["supporting"]["26.16"]["negative_controls"],
                                  base["supporting"]["26.16"]["negative_controls"])
    for k in ("state", "detail"):
        assert by_id(review)["26.16"][k] == by_id(base)["26.16"][k], k


# -- 26.4 / 26.5 labels: feature families vs FDR families -----------------------

def test_26_4_counts_feature_families_and_26_5_counts_fdr_families(good_review, cfg, tmp_path):
    crit = by_id(good_review)
    c4, c5 = crit["26.4"], crit["26.5"]
    assert "feature_families_used" in c4["value"] and "families_used" not in c4["value"]
    assert c4["detail"] == "all 3 feature families registered before earliest experiment created_at"
    assert "feature_families" in good_review["supporting"]["26.4"] and "families" not in good_review["supporting"]["26.4"]
    assert c5["title"] == "Experiment registry with BH-FDR q-values per FDR family"
    assert "fdr_families" in c5["value"] and "families" not in c5["value"]
    assert c5["detail"].startswith("q computed within 2 FDR families")
    hyps, exps = make_registry(cfg)
    ms = {"feature_families": ["market", "starter", "bullpen"]}
    bad = by_id(D.build_review(make_inputs(cfg, model_selection=ms, experiments=exps, hypotheses=hyps), cfg))["26.4"]
    assert bad["detail"].startswith("feature families without a prior hypothesis: ['bullpen']")
    console = Console(record=True, file=io.StringIO(), width=240)
    D.print_review(good_review, console)
    assert "fdr_families=2" in console.export_text()
    inputs = make_inputs(cfg)
    paths = D.write_review_artifacts(good_review, tmp_path, experiments=inputs.experiments,
                                     hypotheses=inputs.hypotheses, holdout_manifest=inputs.holdout_manifest)
    reg = json.loads(paths["experiment_registry"].read_text())
    assert "fdr_families" in reg and "families" not in reg


# -- ADR 0001 Q4: a -dirty row is listed but has no provenance for acceptance -----------------------

@pytest.mark.parametrize("commit", [DIRTY_RUN, None, "", "unavailable: not a git repository"])
def test_decide_status_caps_a_run_without_a_clean_commit_at_inconclusive(commit):
    kw = {"baseline_trades": 500, "bucket_min_games": 200, "dirty_inputs": ()}
    for states in (_states(), _states(**{"26_18": D.PASS})):
        assert D.decide_status(states, **kw, run_commit=CLEAN)[0] in ("RESEARCH-PROMISING", "PROSPECTIVELY-VALIDATED")
        status, reasons = D.decide_status(states, **kw, run_commit=commit)
        assert status == "INCONCLUSIVE" and len(reasons) == 1
        assert reasons[0].startswith("capped at INCONCLUSIVE: the run under review has no clean commit")
        assert "ADR 0001 Q4" in reasons[0]
    # ADR 0002 item 3: the commit cap holds a rejection back (it was REJECTED under ADR 0001 Q4)
    assert D.decide_status(_states(**{"26_11": D.FAIL}), **kw, run_commit=commit)[0] == "INCONCLUSIVE"
    # both caps are named when both apply
    status, reasons = D.decide_status(_states(), **kw, run_commit=commit, quote_quality=D.CANDLE_DEGRADED)
    assert status == "INCONCLUSIVE" and len(reasons) == 2
    assert any("not executable" in r for r in reasons) and any("ADR 0001 Q4" in r for r in reasons)


# -- ADR 0002 item 3: no terminal status without a clean commit; evidence retained -----------------

NOT_CLEAN = [DIRTY_RUN, None, "", "unavailable: not a git repository"]
KW = {"baseline_trades": 500, "bucket_min_games": 200, "dirty_inputs": ()}


def _holdout_bad() -> dict[str, dict]:
    st = _states()
    st["26.1"]["value"] = {"bss": -0.001}
    return st


REJECTING = {"26.11": lambda: _states(**{"26_11": D.FAIL}), "26.13": lambda: _states(**{"26_13": D.FAIL}),
             "26.16": lambda: _states(**{"26_16": D.FAIL}), "26.9": lambda: _states(**{"26_9": D.FAIL}),
             "26.1": _holdout_bad,
             "all": lambda: _states(**{"26_11": D.FAIL, "26_13": D.FAIL, "26_16": D.FAIL, "26_9": D.FAIL})}


@pytest.mark.parametrize("commit", NOT_CLEAN, ids=["dirty", "none", "empty", "unavailable"])
@pytest.mark.parametrize("rule", list(REJECTING))
def test_a_run_without_a_clean_commit_is_never_rejected_and_keeps_the_reasons(commit, rule):
    states = REJECTING[rule]()
    status, clean_reasons = D.decide_status(states, **KW, run_commit=CLEAN)
    assert status == "REJECTED" and clean_reasons
    status, reasons = D.decide_status(states, **KW, run_commit=commit)
    assert status == "INCONCLUSIVE"
    assert reasons[0].startswith("capped at INCONCLUSIVE: the run under review has no clean commit")
    # evidence retained: every reason a clean commit would reject on, word for word, in order
    assert [r for r in reasons if r.startswith(D.HELD_BACK)] == [D.HELD_BACK + r for r in clean_reasons]


@pytest.mark.parametrize("commit", NOT_CLEAN, ids=["dirty", "none", "empty", "unavailable"])
def test_a_run_without_a_clean_commit_has_no_terminal_status(commit):
    seen = {D.decide_status(st, **KW, run_commit=commit, quote_quality=qq)[0]
            for st in (_states(), _states(**{"26_18": D.PASS}), *[f() for f in REJECTING.values()])
            for qq in (D.EXECUTABLE, D.CANDLE_DEGRADED)}
    assert seen == {"INCONCLUSIVE"}


# NaN, blank or padded is not clean either (NaN was truthy, pd.NA raised).
NAN_BLANK = [np.nan, pd.NA, "  ", "\t", " " + CLEAN, CLEAN + "\n", 0]
NAN_BLANK_IDS = ["nan", "pd.NA", "blank", "tab", "leading", "trailing", "int"]


@pytest.mark.parametrize("commit", NAN_BLANK, ids=NAN_BLANK_IDS)
@pytest.mark.parametrize("rule", list(REJECTING))
def test_a_run_with_a_nan_blank_or_padded_commit_is_capped_and_keeps_the_reasons(commit, rule):
    states = REJECTING[rule]()
    clean_reasons = D.decide_status(states, **KW, run_commit=CLEAN)[1]
    status, reasons = D.decide_status(states, **KW, run_commit=commit)
    assert status == "INCONCLUSIVE"
    assert reasons[0].startswith("capped at INCONCLUSIVE: the run under review has no clean commit")
    assert [r for r in reasons if r.startswith(D.HELD_BACK)] == [D.HELD_BACK + r for r in clean_reasons]


@pytest.mark.parametrize("commit", NAN_BLANK, ids=NAN_BLANK_IDS)
def test_a_negative_control_failure_on_a_nan_blank_or_padded_commit_raises_the_flag(commit):
    assert D.negative_control_flag({"26.16": {"state": D.FAIL}}, commit) == D.FLAG_RAISE
    assert D.negative_control_flag({"26.16": {"state": D.PASS}}, commit) is None


def test_the_quote_quality_cap_still_never_hides_a_rejection_on_a_clean_commit():
    for f in REJECTING.values():
        assert D.decide_status(f(), **KW, run_commit=CLEAN, quote_quality=D.CANDLE_DEGRADED)[0] == "REJECTED"
    status, reasons = D.decide_status(_states(**{"26_11": D.FAIL}), **KW, run_commit=DIRTY_RUN,
                                      quote_quality=D.CANDLE_DEGRADED)
    assert status == "INCONCLUSIVE"
    assert [r.split(":")[0] for r in reasons[:2]] == ["capped at INCONCLUSIVE"] * 2
    assert reasons[2] == D.HELD_BACK + "26.11 FAIL (leakage)"


@pytest.mark.parametrize("commit", NOT_CLEAN, ids=["dirty", "none", "empty", "unavailable"])
def test_a_negative_control_failure_without_a_clean_commit_names_the_blocking_flag(commit):
    status, reasons = D.decide_status(_states(**{"26_16": D.FAIL}), **KW, run_commit=commit)
    flagged = [r for r in reasons if r.startswith(D.BLOCKING_FLAG)]
    assert status == "INCONCLUSIVE" and len(flagged) == 1 and reasons[-1] == flagged[0]
    assert "quant freeze" in flagged[0] and "hypothesis registration" in flagged[0] and "ADR 0002" in flagged[0]
    assert "hypothesis registration runs once, for the clearing run" in flagged[0]
    assert "registration refuse" not in flagged[0]
    # no flag without a 26.16 FAIL, and none on a clean commit (that run is REJECTED instead)
    for st, c in ((_states(**{"26_11": D.FAIL}), commit), (_states(**{"26_16": D.INSUFFICIENT}), commit),
                  (_states(**{"26_16": D.FAIL}), CLEAN)):
        assert not [r for r in D.decide_status(st, **KW, run_commit=c)[1] if r.startswith(D.BLOCKING_FLAG)]


@pytest.mark.parametrize(("state", "commit", "event"), [
    (D.FAIL, DIRTY_RUN, D.FLAG_RAISE), (D.FAIL, None, D.FLAG_RAISE), (D.FAIL, "", D.FLAG_RAISE),
    (D.FAIL, "unavailable: not a git repository", D.FLAG_RAISE),
    (D.PASS, CLEAN, D.FLAG_CLEAR),
    (D.FAIL, CLEAN, None), (D.INSUFFICIENT, CLEAN, None), (D.PASS, DIRTY_RUN, None),
    (D.INSUFFICIENT, DIRTY_RUN, None), (None, DIRTY_RUN, None), (None, CLEAN, None)])
def test_negative_control_flag(state, commit, event):
    crit = _states()
    if state is None:
        crit.pop("26.16")
    else:
        crit["26.16"]["state"] = state
    assert D.negative_control_flag(crit, commit) == event


def test_decide_status_requires_the_run_commit():
    with pytest.raises(TypeError):
        D.decide_status(_states(), baseline_trades=500, bucket_min_games=200, dirty_inputs=())


@pytest.mark.parametrize("commit", [DIRTY_RUN, None])
def test_the_review_reads_the_commit_of_the_run_under_review(cfg, commit):
    clean = D.build_review(make_inputs(cfg), cfg)
    assert not any("ADR 0001 Q4" in r for r in clean["status_reasons"])
    run = make_inputs(cfg).backtest_run | {"git_commit": commit}
    review = D.build_review(make_inputs(cfg, backtest_run=run), cfg)
    assert review["status"] == "INCONCLUSIVE"
    assert [r for r in review["status_reasons"] if "ADR 0001 Q4" in r]
    missing = make_inputs(cfg).backtest_run
    missing.pop("git_commit")
    assert any("ADR 0001 Q4" in r for r in D.build_review(make_inputs(cfg, backtest_run=missing), cfg)["status_reasons"])


def _dirty(exps: pd.DataFrame, family: str, n: int = 1, **over) -> tuple[pd.DataFrame, list[str]]:
    exps = exps.copy()
    exps["git_commit"] = exps["git_commit"].astype(object)
    idx = exps.index[exps["experiment_family"] == family][:n]
    exps.loc[idx, "git_commit"] = DIRTY_RUN
    for k, v in over.items():
        exps.loc[idx, k] = v
    return exps, exps.loc[idx, "experiment_id"].tolist()


@pytest.mark.parametrize("family", ["hypothesis", "ablation"])
def test_26_5_fails_when_a_row_in_the_bh_set_is_dirty(cfg, family):
    hyps, exps = make_registry(cfg)
    dirty, ids = _dirty(exps, family, n=2)
    review = D.build_review(make_inputs(cfg, experiments=dirty, hypotheses=hyps), cfg)
    c = by_id(review)["26.5"]
    assert c["state"] == D.FAIL and c["value"]["dirty_git_commit"] == 2 and c["value"]["missing_git_commit"] == 0
    assert "2 experiment(s) in the BH-FDR set ran from uncommitted code (-dirty)" in c["detail"]
    assert "ADR 0001 Q4" in c["detail"]


def test_26_5_keeps_dirty_rows_listed(cfg, tmp_path):
    """Successes and failures stay visible: a -dirty row is never dropped from the listing to make 26.5 pass."""
    hyps, exps = make_registry(cfg)
    dirty, ids = _dirty(exps, "hypothesis", n=3)
    inputs = make_inputs(cfg, experiments=dirty, hypotheses=hyps)
    review = D.build_review(inputs, cfg)
    reg = review["supporting"]["26.5"]["registry"]
    assert set(ids) <= set(reg["experiment_id"]) and len(reg) == int((dirty["experiment_family"] != "negative_control").sum())
    assert (reg.set_index("experiment_id").loc[ids, "git_commit"] == DIRTY_RUN).all()
    c = by_id(review)["26.5"]
    assert c["value"]["fdr_experiments"] == 12 and c["value"]["successes"] + c["value"]["failures"] == 12
    paths = D.write_review_artifacts(review, tmp_path, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                                     holdout_manifest=inputs.holdout_manifest)
    listed = {e["experiment_id"]: e for e in json.loads(paths["experiment_registry"].read_text())["experiments"]}
    assert all(listed[i]["git_commit"] == DIRTY_RUN for i in ids)


def test_26_5_counts_a_dirty_bh_row_tested_on_the_holdout(cfg):
    """ADR 0001 interpretation 4: the count covers the BH-set rows tested on the holdout too (26.3 fails them)."""
    hyps, exps = make_registry(cfg)
    peek = exps[exps["experiment_family"] == "hypothesis"].iloc[[0]].assign(
        experiment_id="e-peek", test_period="2026H2", git_commit=DIRTY_RUN)
    c = by_id(D.build_review(make_inputs(cfg, experiments=pd.concat([exps, peek]), hypotheses=hyps), cfg))["26.5"]
    assert c["state"] == D.FAIL and c["value"]["dirty_git_commit"] == 1


def test_26_5_dirty_count_is_the_bh_set_only(cfg, good_review):
    """Q4 names the BH set: a -dirty negative control or exploratory row does not fail 26.5 by this rule."""
    assert by_id(good_review)["26.5"]["value"]["dirty_git_commit"] == 0
    hyps, exps = make_registry(cfg)
    dirty, _ = _dirty(exps, "negative_control", n=2)
    c = by_id(D.build_review(make_inputs(cfg, experiments=dirty, hypotheses=hyps), cfg))["26.5"]
    assert c["state"] == D.PASS and c["value"]["dirty_git_commit"] == 0


def test_a_dirty_row_fails_26_5_and_adds_only_the_feeding_row_cap(cfg):
    """This supersedes the test's earlier "changes no status": as for a dirty run, the cap line
    replaces the not-RESEARCH-PROMISING reasons (nothing here would reject, so nothing is held back)."""
    hyps, exps = make_registry(cfg)
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    dirty, _ = _dirty(exps, "hypothesis")
    flagged = D.build_review(make_inputs(cfg, experiments=dirty, hypotheses=hyps), cfg)
    assert by_id(flagged)["26.5"]["state"] == D.FAIL and flagged["status"] == "INCONCLUSIVE"
    assert base["status"] == "INCONCLUSIVE" and not _feed_caps(base)
    assert flagged["status_reasons"] == _feed_caps(flagged) and len(flagged["status_reasons"]) == 1


# -- ADR 0001 Q1: the exploratory family is outside the BH set, in its own section ------------------

def _with_exploratory(cfg, **over) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hyps, exps = make_registry(cfg)
    extra = exps[exps["experiment_family"] == "hypothesis"].assign(
        experiment_family="exploratory", experiment_id=lambda d: d["experiment_id"] + "-x", p_value=0.001)
    extra["git_commit"] = extra["git_commit"].astype(object)
    for k, v in over.items():
        extra[k] = v
    return hyps, exps, pd.concat([exps, extra], ignore_index=True)


def _states_and_details(review: dict, skip=("26.5",)) -> list[tuple]:
    return [(c["id"], c["state"], c["detail"]) for c in review["criteria"] if c["id"] not in skip]


@pytest.mark.parametrize("over", [{}, {"p_value": np.nan}, {"git_commit": None}, {"git_commit": DIRTY_RUN}],
                         ids=["plain", "no-p", "no-commit", "dirty"])
def test_exploratory_rows_do_not_fail_26_5_or_count_toward_any_criterion(cfg, over):
    hyps, exps, withx = _with_exploratory(cfg, **over)
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    review = D.build_review(make_inputs(cfg, experiments=withx, hypotheses=hyps), cfg)
    c, c0 = by_id(review)["26.5"], by_id(base)["26.5"]
    assert c0["state"] == c["state"] == D.PASS
    assert c["value"]["exploratory_experiments"] == 6 and c0["value"]["exploratory_experiments"] == 0
    drop = lambda v: {k: x for k, x in v.items() if k != "exploratory_experiments"}
    assert drop(c["value"]) == drop(c0["value"])
    assert _states_and_details(review) == _states_and_details(base)
    assert (review["status"], review["status_reasons"]) == (base["status"], base["status_reasons"])


def test_exploratory_rows_get_no_q_and_do_not_dilute_the_bh_set(cfg):
    hyps, exps, withx = _with_exploratory(cfg)
    rc = cfg["review"]
    reg = D.experiment_registry(withx, fdr_q=rc["fdr_q"], fdr_families=rc["fdr_families"])
    x = reg["experiment_family"] == "exploratory"
    assert (reg.loc[x, "fdr_role"] == "exploratory").all()
    assert reg.loc[x, "q_value"].isna().all() and not reg.loc[x, "success"].any()
    ref = D.experiment_registry(exps, fdr_q=rc["fdr_q"], fdr_families=rc["fdr_families"])
    pd.testing.assert_series_equal(reg.loc[~x, "q_value"].reset_index(drop=True), ref["q_value"])


def test_26_5_fails_closed_when_the_fdr_set_names_exploratory(cfg):
    hyps, exps, withx = _with_exploratory(cfg)
    named = small_cfg(fdr_families=["ablation", "hypothesis", "exploratory"])
    review = D.build_review(make_inputs(named, experiments=withx, hypotheses=hyps), named)
    c = by_id(review)["26.5"]
    assert c["state"] == D.FAIL and "names exploratory" in c["detail"]
    rows = review["exploratory"]["experiments"]
    assert len(rows) == 6 and "q_value" not in rows                     # still exploratory, never BH-adjusted
    assert "exploratory" not in set(review["supporting"]["26.5"]["registry"]["experiment_family"])


def test_exploratory_rows_have_their_own_section_labeled_exploratory(cfg, tmp_path):
    hyps, exps, withx = _with_exploratory(cfg)
    inputs = make_inputs(cfg, experiments=withx, hypotheses=hyps)
    review = D.build_review(inputs, cfg)
    sec = review["exploratory"]
    assert sec["label"].startswith("EXPLORATORY") and "never counted toward any acceptance criterion" in sec["label"]
    rows = sec["experiments"]
    assert len(rows) == 6 and set(rows["experiment_family"]) == {"exploratory"}
    assert "q_value" not in rows and "success" not in rows and rows["p_value"].notna().all()
    assert "exploratory" not in set(review["supporting"]["26.5"]["registry"]["experiment_family"])
    assert f"## {sec['label']}" in D.render_markdown(review).splitlines()
    console = Console(record=True, file=io.StringIO(), width=240)
    D.print_review(review, console)
    assert "EXPLORATORY" in console.export_text()
    paths = D.write_review_artifacts(review, tmp_path, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                                     holdout_manifest=inputs.holdout_manifest)
    reg = json.loads(paths["experiment_registry"].read_text())
    assert reg["exploratory"]["label"] == sec["label"] and len(reg["exploratory"]["experiments"]) == 6
    assert {e["experiment_family"] for e in reg["experiments"]} == {"hypothesis", "ablation"}
    assert reg["unassigned_families"] == []


def test_the_exploratory_section_is_present_and_empty_without_exploratory_rows(good_review):
    sec = good_review["exploratory"]
    assert sec["label"].startswith("EXPLORATORY") and len(sec["experiments"]) == 0


# -- ADR 0001 Q2: a diagnostic line when the slope CI excludes 1.0, no state change ----------------

def _slope_stub(monkeypatch, lo, hi, slope=None):
    slope = (lo + hi) / 2 if slope is None else slope
    monkeypatch.setattr(D.M, "calibration_slope_bootstrap", lambda *a, **k: {
        "slope": slope, "ci_low": lo, "ci_high": hi, "ci_level": 0.95, "n_valid": 300})


@pytest.mark.parametrize("ci, side", [((0.80, float(np.nextafter(1.0, 0))), "below 1: model overconfident"),
                                      ((float(np.nextafter(1.0, 2)), 1.20), "above 1: model underconfident")])
def test_26_13_says_when_the_slope_ci_excludes_1(cfg, monkeypatch, ci, side):
    _slope_stub(monkeypatch, *ci)
    c = by_id(D.build_review(make_inputs(cfg), cfg))["26.13"]
    assert c["state"] == D.PASS and c["value"]["slope_ci_excludes_1"] is True
    diag = [p for p in c["detail"].split("; ") if p.startswith("DIAGNOSTIC")]
    assert len(diag) == 1 and "slope CI excludes 1.0" in diag[0] and side in diag[0] and "no state change" in diag[0]


@pytest.mark.parametrize("ci", [(0.80, 1.0), (1.0, 1.2), (0.9, 1.1)])
def test_26_13_no_diagnostic_when_the_ci_touches_or_covers_1(cfg, monkeypatch, ci):
    _slope_stub(monkeypatch, *ci)
    c = by_id(D.build_review(make_inputs(cfg), cfg))["26.13"]
    assert c["state"] == D.PASS and c["value"]["slope_ci_excludes_1"] is False and "DIAGNOSTIC" not in c["detail"]


def test_26_13_diagnostic_moves_no_state_or_status(cfg, monkeypatch):
    base = D.build_review(make_inputs(cfg), cfg)
    _slope_stub(monkeypatch, 0.6, 0.9, 0.75)
    diag = D.build_review(make_inputs(cfg), cfg)
    assert by_id(diag)["26.13"]["state"] == by_id(base)["26.13"]["state"] == D.PASS
    assert (diag["status"], diag["status_reasons"]) == (base["status"], base["status_reasons"])
    _slope_stub(monkeypatch, 0.2, 0.4, 0.3)                              # wholly outside [0.5, 1.5]: the band rule
    c = by_id(D.build_review(make_inputs(cfg), cfg))["26.13"]
    assert c["state"] == D.FAIL and c["value"]["slope_ci_excludes_1"] is True and "DIAGNOSTIC" in c["detail"]


def test_26_13_diagnostic_on_real_miscalibration(cfg):
    """No stub: a model 30% too extreme has a slope CI below 1 but inside the band."""
    c = by_id(D.build_review(make_inputs(cfg, make_predictions(cfg, miscal=1.3)), cfg))["26.13"]
    assert c["value"]["slope_ci_high"] < 1.0 and c["state"] == D.PASS
    assert c["value"]["slope_ci_excludes_1"] is True and "overconfident" in c["detail"]


def test_26_13_without_a_ci_has_no_diagnostic(cfg):
    c = small_cfg(bootstrap_resamples=0)
    crit = by_id(D.build_review(make_inputs(c), c))["26.13"]
    assert crit["state"] == D.INSUFFICIENT and crit["value"]["slope_ci_excludes_1"] is None
    assert "DIAGNOSTIC" not in crit["detail"]


# -- a row feeding a clean run without a clean commit ---
# caps the run at INCONCLUSIVE, the same as a run without one. Feeding rows: the experiment rows the review
# reads in every fdr_role but exploratory, the hypotheses rows, and the model runs the run and its
# sensitivity runs decided with (plus any the run's config names).

FEED_CAP = "capped at INCONCLUSIVE: rows feeding the run under review have no clean commit"
FEED_TAIL = "; retrain from a clean tree"
ROW_NOT_CLEAN = [DIRTY_RUN, None, np.nan, "", "  ", "unavailable: not a git repository"]
ROW_IDS = ["dirty", "none", "nan", "empty", "blank", "unavailable"]
DIRTY_INPUTS = ["experiments: 1 of 18 rows (negative_control 1)"]


def _feed_caps(review_or_reasons) -> list[str]:
    reasons = review_or_reasons["status_reasons"] if isinstance(review_or_reasons, dict) else review_or_reasons
    return [r for r in reasons if r.startswith(FEED_CAP)]


def _with_commit(df: pd.DataFrame, commit, mask) -> pd.DataFrame:
    df = df.copy()
    df["git_commit"] = df["git_commit"].astype(object)
    df.loc[mask, "git_commit"] = commit
    return df


def _model_runs(*ids: str, commit=CLEAN) -> pd.DataFrame:
    return pd.DataFrame({"table": "model_runs", "row_id": list(ids), "git_commit": [commit] * len(ids)})


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
@pytest.mark.parametrize(("family", "role"), [("hypothesis", "fdr"), ("ablation", "fdr"),
                                              ("negative_control", "negative_control")])
def test_a_not_clean_experiment_row_caps_a_clean_run(cfg, family, role, commit):
    hyps, exps = make_registry(cfg)
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    assert not _feed_caps(base) and base["feeding_rows"]["not_clean"] == []
    row = exps.index[exps["experiment_family"] == family][:1]
    review = D.build_review(make_inputs(cfg, experiments=_with_commit(exps, commit, row), hypotheses=hyps), cfg)
    caps = _feed_caps(review)
    assert review["status"] == "INCONCLUSIVE" and len(caps) == 1
    assert f"(experiments: 1 of {len(exps)} rows ({role} 1)); " in caps[0] and caps[0].endswith(FEED_TAIL)
    listed = review["feeding_rows"]["not_clean"]
    assert [(r["table"], r["role"], r["row_id"]) for r in listed] == [
        ("experiments", role, exps.loc[row[0], "experiment_id"])]


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
def test_a_not_clean_hypotheses_row_caps_a_clean_run(cfg, commit):
    hyps, exps = make_registry(cfg)
    review = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=_with_commit(hyps, commit, hyps.index[:1])),
                            cfg)
    caps = _feed_caps(review)
    assert review["status"] == "INCONCLUSIVE" and len(caps) == 1 and f"(hypotheses: 1 of {len(hyps)} rows); " in caps[0]
    assert [(r["table"], r["row_id"]) for r in review["feeding_rows"]["not_clean"]] == [("hypotheses", "H-market")]


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
def test_a_not_clean_model_run_caps_a_clean_run(cfg, commit):
    runs = _model_runs("mr-1", "mr-2")
    clean = D.build_review(make_inputs(cfg, feeding_runs=runs), cfg)
    assert not _feed_caps(clean)
    assert clean["feeding_rows"]["checked"] == {"experiments": 18, "hypotheses": 3, "model_runs": 2}
    review = D.build_review(make_inputs(cfg, feeding_runs=_with_commit(runs, commit, runs.index[1:])), cfg)
    caps = _feed_caps(review)
    assert review["status"] == "INCONCLUSIVE" and len(caps) == 1 and "(model_runs: 1 of 2 rows); " in caps[0]
    assert [(r["table"], r["row_id"]) for r in review["feeding_rows"]["not_clean"]] == [("model_runs", "mr-2")]


@pytest.mark.parametrize("commit", [" " + CLEAN, CLEAN + "\n", "\t"], ids=["leading", "trailing", "tab"])
def test_a_padded_commit_on_a_feeding_row_is_not_clean(cfg, commit):
    """One definition of clean for every caller; a padded commit is not a bare commit id."""
    hyps, exps = make_registry(cfg)
    review = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=_with_commit(hyps, commit, hyps.index[:1])),
                            cfg)
    assert [(r["table"], r["row_id"]) for r in review["feeding_rows"]["not_clean"]] == [("hypotheses", "H-market")]
    assert review["status"] == "INCONCLUSIVE" and len(_feed_caps(review)) == 1


@pytest.mark.parametrize("source", ["experiments", "hypotheses", "feeding_runs"])
def test_a_feeding_frame_without_a_git_commit_column_has_no_clean_row(cfg, source):
    """Fail-closed: a row whose commit cannot be read is not clean."""
    hyps, exps = make_registry(cfg)
    frames = {"experiments": exps, "hypotheses": hyps, "feeding_runs": _model_runs("mr-1", "mr-2")}
    frames[source] = frames[source].drop(columns="git_commit")
    review = D.build_review(make_inputs(cfg, **frames), cfg)
    assert len(review["feeding_rows"]["not_clean"]) == len(frames[source]) and len(_feed_caps(review)) == 1


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
def test_exploratory_rows_are_not_feeding_rows(cfg, commit):
    """ADR 0001 Q1: exploratory rows are never counted toward any acceptance criterion, this cap included."""
    hyps, exps, withx = _with_exploratory(cfg, git_commit=commit)
    review = D.build_review(make_inputs(cfg, experiments=withx, hypotheses=hyps), cfg)
    assert not _feed_caps(review) and review["feeding_rows"]["not_clean"] == []
    assert review["feeding_rows"]["checked"]["experiments"] == len(exps) == len(withx) - 6


@pytest.mark.parametrize("kind", ["unassigned", "holdout"])
def test_unassigned_and_holdout_tested_rows_feed_the_run(cfg, kind):
    hyps, exps = make_registry(cfg)
    extra = exps[exps["experiment_family"] == "hypothesis"].iloc[[0]].assign(experiment_id="e-extra",
                                                                             git_commit=DIRTY_RUN)
    extra = extra.assign(experiment_family="mystery") if kind == "unassigned" else extra.assign(test_period="2026H2")
    review = D.build_review(make_inputs(cfg, experiments=pd.concat([exps, extra], ignore_index=True),
                                        hypotheses=hyps), cfg)
    assert [r["row_id"] for r in review["feeding_rows"]["not_clean"]] == ["e-extra"]
    assert len(_feed_caps(review)) == 1


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
def test_row_14_dirty_controls_under_a_clean_run_are_inconclusive_not_rejected(cfg, commit):
    hyps, exps = make_registry(cfg)
    nc = exps["experiment_family"] == "negative_control"
    exps.loc[nc, "metrics_json"] = json.dumps({"delta_brier": 0.003})
    exps.loc[nc, "p_value"] = 0.001
    clean = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps), cfg)
    assert clean["status"] == "REJECTED" and "26.16 FAIL (a negative control works)" in clean["status_reasons"]
    review = D.build_review(make_inputs(cfg, experiments=_with_commit(exps, commit, nc), hypotheses=hyps), cfg)
    reasons = review["status_reasons"]
    assert by_id(review)["26.16"]["state"] == D.FAIL and review["status"] == "INCONCLUSIVE"
    assert reasons[0].startswith(FEED_CAP) and f"(negative_control {int(nc.sum())})" in reasons[0]
    assert reasons[1:] == [D.HELD_BACK + r for r in clean["status_reasons"]]
    # the run's own commit is clean: no run cap, and no blocking flag (the flag reads the run's commit only)
    assert not [r for r in reasons if r.startswith(D.BLOCKING_FLAG) or "ADR 0001 Q4" in r]


@pytest.mark.parametrize("rule", list(REJECTING))
def test_dirty_inputs_hold_back_every_rejection_of_a_clean_run(rule):
    states = REJECTING[rule]()
    status, clean_reasons = D.decide_status(states, **KW, run_commit=CLEAN)
    assert status == "REJECTED"
    status, reasons = D.decide_status(states, **(KW | {"dirty_inputs": DIRTY_INPUTS}), run_commit=CLEAN)
    assert status == "INCONCLUSIVE"
    assert reasons == [f"{FEED_CAP} ({DIRTY_INPUTS[0]}){FEED_TAIL}",
                       *[D.HELD_BACK + r for r in clean_reasons]]


def test_dirty_inputs_leave_no_terminal_status():
    seen = {D.decide_status(st, **(KW | {"dirty_inputs": DIRTY_INPUTS}), run_commit=c, quote_quality=qq)[0]
            for st in (_states(), _states(**{"26_18": D.PASS}), *[f() for f in REJECTING.values()])
            for qq in (D.EXECUTABLE, D.CANDLE_DEGRADED) for c in [CLEAN, *NOT_CLEAN]}
    assert seen == {"INCONCLUSIVE"}
    two = [*DIRTY_INPUTS, "model_runs: 2 of 6 rows"]
    for st in (_states(), _states(**{"26_18": D.PASS})):
        assert D.decide_status(st, **KW, run_commit=CLEAN)[0] in ("RESEARCH-PROMISING", "PROSPECTIVELY-VALIDATED")
        assert D.decide_status(st, **(KW | {"dirty_inputs": two}), run_commit=CLEAN) == (
            "INCONCLUSIVE", [f"{FEED_CAP} ({two[0]}; {two[1]}){FEED_TAIL}"])


def test_every_cap_is_named_in_order_and_the_flag_reads_the_run_commit_only():
    status, reasons = D.decide_status(_states(**{"26_16": D.FAIL}), **(KW | {"dirty_inputs": DIRTY_INPUTS}),
                                      run_commit=DIRTY_RUN, quote_quality=D.CANDLE_DEGRADED)
    assert status == "INCONCLUSIVE" and len(reasons) == 5
    assert "not executable" in reasons[0] and "ADR 0001 Q4" in reasons[1] and reasons[2].startswith(FEED_CAP)
    assert reasons[3] == D.HELD_BACK + "26.16 FAIL (a negative control works)"
    assert reasons[4].startswith(D.BLOCKING_FLAG)
    clean_run = D.decide_status(_states(**{"26_16": D.FAIL}), **(KW | {"dirty_inputs": DIRTY_INPUTS}),
                                run_commit=CLEAN)[1]
    assert not [r for r in clean_run if r.startswith(D.BLOCKING_FLAG)]


def test_decide_status_requires_the_feeding_rows_verdict():
    with pytest.raises(TypeError):
        D.decide_status(_states(), baseline_trades=500, bucket_min_games=200, run_commit=CLEAN)


def test_the_review_json_lists_the_not_clean_feeding_rows(cfg, tmp_path):
    hyps, exps = make_registry(cfg)
    nc = exps.index[exps["experiment_family"] == "negative_control"][:2]
    inputs = make_inputs(cfg, experiments=_with_commit(exps, DIRTY_RUN, nc), hypotheses=hyps,
                         feeding_runs=pd.concat([_model_runs("mr-1"), _model_runs("mr-9", commit=None)]))
    review = D.build_review(inputs, cfg)
    paths = D.write_review_artifacts(review, tmp_path, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                                     holdout_manifest=inputs.holdout_manifest)
    fr = json.loads(paths["review_json"].read_text())["feeding_rows"]
    assert fr["checked"] == {"experiments": len(exps), "hypotheses": 3, "model_runs": 2}
    assert sorted((r["table"], r["row_id"], r["git_commit"]) for r in fr["not_clean"]) == sorted(
        [("experiments", i, DIRTY_RUN) for i in exps.loc[nc, "experiment_id"]] + [("model_runs", "mr-9", None)])
    assert _feed_caps(review) == [f"{FEED_CAP} (experiments: 2 of {len(exps)} rows (negative_control 2); "
                                  f"model_runs: 1 of 2 rows){FEED_TAIL}"]


# the loader: which model runs feed the run under review

def _feeding_world(db, *, commit, named: tuple[str, ...]) -> None:
    db.insert("observations", frame(observation(outcome=1), observation(offset=60, outcome=1, is_canonical=False)))
    db.insert("model_runs", frame(*[
        {"model_run_id": m, "model_type": "logistic_residual", "period_id": "2025H1", "feature_set_id": "fs",
         "feature_set_hash": "fsh", "config_hash": "c", "predict_from": pd.Timestamp("2025-01-01T00:00Z"),
         "dataset": "synthetic", "git_commit": c, "created_at": pd.Timestamp("2026-01-01T00:00Z"),
         "is_synthetic": True} for m, c in (("mr-1", CLEAN), ("mr-2", commit), ("mr-other", DIRTY_RUN))]))
    db.insert("predictions", frame(*[
        {"model_run_id": m, "observation_id": oid, "period_id": "2025H1", "p_market": 0.56, "p_model": 0.60,
         "lower_bound": 0.55, "upper_bound": 0.65, "p_conservative_yes": 0.55, "p_conservative_no": 0.35,
         "is_synthetic": True} for m, oid in (("mr-1", "SYN-NYY@T-30"), ("mr-2", "SYN-NYY@T-60"))]))
    runs = [{"backtest_run_id": rid, "model_type": "logistic_residual", "snapshot_offset_min": off,
             "config_json": json.dumps({"model_type": "logistic_residual", "model_run_ids": list(named)}),
             "config_hash": "c", "quote_quality": "executable", "dataset": "synthetic", "git_commit": CLEAN,
             "created_at": pd.Timestamp("2026-02-01T00:00Z"), "is_synthetic": True}
            for rid, off in (("bt-30", 30), ("bt-60", 60))]
    db.insert("backtest_runs", frame(*runs))
    db.insert("backtest_decisions", frame(*[
        {"backtest_run_id": rid, "scenario": "pen_25", "model_run_id": m, "observation_id": f"SYN-NYY@T-{off}",
         "game_id": "syn-mlb-1", "contract_id": "SYN-NYY", "period_id": "2025H1",
         "snapshot_time": observation(offset=off)["snapshot_time"], "decision": "PASS",
         "pass_reason": "below_min_edge", "side": "yes", "p_market": 0.56, "p_model": 0.60, "predicted_edge": 0.03,
         "quoted_price": 0.57, "spread_penalty_frac": 0.25, "outcome": 1, "side_won": 1,
         "exit_rule": "hold_to_settlement", "is_synthetic": True}
        for rid, off, m in (("bt-30", 30, "mr-1"), ("bt-60", 60, "mr-2"))]))


def _loaded_runs(inputs) -> list[tuple]:
    fr = inputs.feeding_runs
    return sorted((t, r, None if pd.isna(c) else c) for t, r, c in zip(fr["table"], fr["row_id"], fr["git_commit"]))


def test_the_loader_reads_the_model_runs_of_the_run_and_its_sensitivity_runs(syn_db):
    """mr-2 decided only at T-60 (a sensitivity run of the same invocation); mr-gone is named in the run's config
    and absent from model_runs, so it has no commit; mr-other fed no run here and is not read."""
    _feeding_world(syn_db, commit=CLEAN, named=("mr-1", "mr-gone"))
    inputs = D.load_review_inputs(syn_db, load_sport_config("mlb"))
    assert _loaded_runs(inputs) == [("model_runs", "mr-1", CLEAN), ("model_runs", "mr-2", CLEAN),
                                    ("model_runs", "mr-gone", None)]


@pytest.mark.parametrize(("commit", "named", "bad"), [
    (CLEAN, ("mr-1",), []),
    (DIRTY_RUN, ("mr-1",), ["mr-2"]),
    (None, ("mr-1",), ["mr-2"]),
    (CLEAN, ("mr-1", "mr-gone"), ["mr-gone"]),
], ids=["clean", "dirty-sensitivity-model", "no-commit", "named-but-missing"])
def test_a_loaded_review_caps_on_a_not_clean_model_run(syn_db, commit, named, bad):
    _feeding_world(syn_db, commit=commit, named=named)
    cfg = load_sport_config("mlb")
    review = D.build_review(D.load_review_inputs(syn_db, cfg), cfg)
    assert [r["row_id"] for r in review["feeding_rows"]["not_clean"]] == bad
    assert len(_feed_caps(review)) == (1 if bad else 0)


# -- run lineage. `quant train` records the
# model runs, experiments and hypotheses of each invocation. A run whose feeding model runs all have lineage is
# checked against its own rows: the feeding-rows cap and 26.16 read only them. 26.5 stays global. Any other run
# keeps the whole-log check, labelled LEGACY_LINEAGE_FALLBACK.

LINEAGE_RUNS = ("mr-1", "mr-2")
FEED_TAIL_LINEAGE = "retrain from a clean tree"


def _lineage(exps: pd.DataFrame, hyps: pd.DataFrame, *, runs=LINEAGE_RUNS, invocation="ti-1") -> pd.DataFrame:
    rows = ([("model_run", r) for r in runs] + [("experiment", e) for e in exps["experiment_id"]]
            + [("hypothesis", h) for h in hyps["hypothesis_id"]])
    return pd.DataFrame([{"invocation_id": invocation, "kind": k, "row_id": r} for k, r in rows])


def _other_controls(exps: pd.DataFrame, **over) -> pd.DataFrame:
    """Another invocation's negative controls: new ids and their own candidate."""
    nc = exps[exps["experiment_family"] == "negative_control"].copy()
    return nc.assign(experiment_id="other-" + nc["experiment_id"], feature_set_id="nc_other", **over)


def _three(cfg, exps, hyps, other) -> dict[str, dict]:
    """The registry alone with its lineage ("base"), then the registry plus `other` reviewed three ways: `other`
    outside the run's lineage, inside it, and with no lineage at all ("legacy")."""
    allx = pd.concat([exps, other], ignore_index=True)
    runs = _model_runs(*LINEAGE_RUNS)
    ways = {"outside": _lineage(exps, hyps), "inside": _lineage(allx, hyps), "legacy": _lineage(exps, hyps).iloc[:0]}
    out = {k: D.build_review(make_inputs(cfg, experiments=allx, hypotheses=hyps, feeding_runs=runs, lineage=lin), cfg)
           for k, lin in ways.items()}
    out["base"] = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps, feeding_runs=runs,
                                             lineage=ways["outside"]), cfg)
    return out


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
def test_an_unrelated_dirty_control_cannot_cap_a_clean_run_and_the_same_control_in_its_lineage_does(cfg, commit):
    hyps, exps = make_registry(cfg)
    other = _other_controls(exps, git_commit=commit)
    r = _three(cfg, exps, hyps, other)
    base = r["base"]
    assert [r[k]["lineage"]["mode"] for k in ("base", "outside", "inside", "legacy")] == [
        D.LINEAGE, D.LINEAGE, D.LINEAGE, D.LEGACY_LINEAGE_FALLBACK]
    assert not _feed_caps(base)
    assert r["outside"]["status_reasons"] == base["status_reasons"] and r["outside"]["status"] == base["status"]
    assert r["outside"]["feeding_rows"] == base["feeding_rows"]
    for k in ("inside", "legacy"):
        caps = _feed_caps(r[k])
        assert r[k]["status"] == "INCONCLUSIVE" and len(caps) == 1, k
        assert f"(negative_control {len(other)})" in caps[0], k
        assert sorted(x["row_id"] for x in r[k]["feeding_rows"]["not_clean"]) == sorted(other["experiment_id"]), k


def test_an_unrelated_working_control_cannot_fail_a_clean_runs_26_16_and_the_same_control_in_its_lineage_does(cfg):
    hyps, exps = make_registry(cfg)
    other = _other_controls(exps, metrics_json=json.dumps({"delta_brier": 0.003}), p_value=0.001)
    r = _three(cfg, exps, hyps, other)
    assert by_id(r["base"])["26.16"]["state"] == D.PASS and r["base"]["status"] != "REJECTED"
    assert by_id(r["outside"])["26.16"] == by_id(r["base"])["26.16"]
    assert r["outside"]["status"] == r["base"]["status"]
    assert set(r["outside"]["negative_controls"]["experiments"]["feature_set_id"]) == {"nc_random"}
    for k in ("inside", "legacy"):
        c16 = by_id(r[k])["26.16"]
        assert c16["state"] == D.FAIL and c16["value"]["working"] == ["nc_other"], k
        assert r[k]["status"] == "REJECTED" and "26.16 FAIL (a negative control works)" in r[k]["status_reasons"], k


def test_another_invocations_lineage_beside_the_runs_is_not_read(cfg):
    """The unrelated dirty, working control has its own invocation and model run in the lineage frame: the scope
    takes only the feeding runs' invocation, so it neither caps the run nor fails its 26.16."""
    hyps, exps = make_registry(cfg)
    other = _other_controls(exps, git_commit=DIRTY_RUN, metrics_json=json.dumps({"delta_brier": 0.003}),
                            p_value=0.001)
    lin = pd.concat([_lineage(exps, hyps), _lineage(other, hyps.iloc[:0], runs=("mr-other",), invocation="ti-other")],
                    ignore_index=True)
    runs = _model_runs(*LINEAGE_RUNS)
    review = D.build_review(make_inputs(cfg, experiments=pd.concat([exps, other], ignore_index=True),
                                        hypotheses=hyps, feeding_runs=runs, lineage=lin), cfg)
    base = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=hyps, feeding_runs=runs,
                                      lineage=_lineage(exps, hyps)), cfg)
    assert review["lineage"]["mode"] == D.LINEAGE and review["lineage"]["invocations"] == ["ti-1"]
    assert review["lineage"]["rows"] == base["lineage"]["rows"]
    assert not _feed_caps(review) and by_id(review)["26.16"] == by_id(base)["26.16"]
    assert review["status_reasons"] == base["status_reasons"] and review["status"] == base["status"]


@pytest.mark.parametrize("commit", ROW_NOT_CLEAN, ids=ROW_IDS)
def test_an_unrelated_dirty_hypothesis_cannot_cap_a_clean_run_and_one_in_its_lineage_does(cfg, commit):
    hyps, exps = make_registry(cfg)
    extra = hyps.iloc[[0]].assign(hypothesis_id="H-other", git_commit=commit)
    allh = pd.concat([hyps, extra], ignore_index=True)
    runs = _model_runs(*LINEAGE_RUNS)
    outside = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=allh, feeding_runs=runs,
                                         lineage=_lineage(exps, hyps)), cfg)
    inside = D.build_review(make_inputs(cfg, experiments=exps, hypotheses=allh, feeding_runs=runs,
                                        lineage=_lineage(exps, allh)), cfg)
    assert outside["lineage"]["mode"] == inside["lineage"]["mode"] == D.LINEAGE
    assert not _feed_caps(outside) and outside["feeding_rows"]["checked"]["hypotheses"] == len(hyps)
    caps = _feed_caps(inside)
    assert inside["status"] == "INCONCLUSIVE" and len(caps) == 1 and f"(hypotheses: 1 of {len(allh)} rows)" in caps[0]


def test_26_5_stays_global_under_lineage(cfg):
    """Spec 26.5: every test in a family counts, whichever invocation logged it."""
    hyps, exps = make_registry(cfg)
    other = exps[exps["experiment_family"] == "hypothesis"].iloc[[0]].assign(experiment_id="other-h",
                                                                             git_commit=DIRTY_RUN)
    r = _three(cfg, exps, hyps, other)
    assert r["outside"]["lineage"]["mode"] == D.LINEAGE and not _feed_caps(r["outside"])
    for k in ("outside", "inside", "legacy"):
        c5 = by_id(r[k])["26.5"]
        assert c5["state"] == D.FAIL and c5["value"]["dirty_git_commit"] == 1, k
        assert c5 == by_id(r["legacy"])["26.5"], k


@pytest.mark.parametrize("case", ["no-lineage", "partial", "experiment-not-read", "no-feeding-runs"])
def test_a_run_without_whole_lineage_takes_the_whole_log_labelled(cfg, case):
    hyps, exps = make_registry(cfg)
    other = _other_controls(exps, git_commit=DIRTY_RUN)
    allx = pd.concat([exps, other], ignore_index=True)
    runs, lin = _model_runs(*LINEAGE_RUNS), _lineage(exps, hyps)
    if case == "no-lineage":
        lin = lin.iloc[:0]
    elif case == "partial":
        lin = _lineage(exps, hyps, runs=LINEAGE_RUNS[:1])
    elif case == "experiment-not-read":
        lin = pd.concat([lin, pd.DataFrame([{"invocation_id": "ti-1", "kind": "experiment", "row_id": "e-gone"}])])
    else:
        runs = runs.iloc[:0]
    review = D.build_review(make_inputs(cfg, experiments=allx, hypotheses=hyps, feeding_runs=runs, lineage=lin), cfg)
    assert review["lineage"]["mode"] == D.LEGACY_LINEAGE_FALLBACK and review["lineage"]["reason"]
    caps = _feed_caps(review)
    assert len(caps) == 1 and f"(negative_control {len(other)})" in caps[0]


def test_the_default_review_inputs_have_no_lineage(cfg):
    review = D.build_review(make_inputs(cfg), cfg)
    assert review["lineage"]["mode"] == D.LEGACY_LINEAGE_FALLBACK


def test_the_feeding_cap_names_the_retrain(cfg):
    hyps, exps = make_registry(cfg)
    r = _three(cfg, exps, hyps, _other_controls(exps, git_commit=DIRTY_RUN))
    assert _feed_caps(r["inside"])[0].endswith(FEED_TAIL_LINEAGE)


@pytest.mark.parametrize("mode", ["lineage", "legacy"])
def test_the_lineage_mode_is_in_the_json_the_markdown_and_the_console(cfg, tmp_path, mode):
    hyps, exps = make_registry(cfg)
    lin = _lineage(exps, hyps) if mode == "lineage" else _lineage(exps, hyps).iloc[:0]
    inputs = make_inputs(cfg, experiments=exps, hypotheses=hyps, feeding_runs=_model_runs(*LINEAGE_RUNS), lineage=lin)
    review = D.build_review(inputs, cfg)
    label = D.LINEAGE if mode == "lineage" else D.LEGACY_LINEAGE_FALLBACK
    paths = D.write_review_artifacts(review, tmp_path, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                                     holdout_manifest=inputs.holdout_manifest)
    assert json.loads(paths["review_json"].read_text())["lineage"]["mode"] == label
    assert f"- Lineage: {label}" in paths["review_md"].read_text()
    buf = io.StringIO()
    D.print_review(review, Console(file=buf, width=200))
    assert f"Lineage {label}" in buf.getvalue()
    if mode == "lineage":
        assert review["lineage"]["invocations"] == ["ti-1"]
        assert review["lineage"]["rows"] == {"model_run": 2, "experiment": len(exps), "hypothesis": len(hyps)}


# the loader: the lineage of the model runs that feed the run under review

def _lineage_rows(invocation: str, commit: str, **kinds: tuple[str, ...]) -> pd.DataFrame:
    return frame(*[{"invocation_id": invocation, "kind": k, "row_id": r, "dataset": "synthetic", "git_commit": commit,
                    "created_at": pd.Timestamp("2026-01-01T00:00Z"), "is_synthetic": True}
                   for k, ids in kinds.items() for r in ids])


@pytest.mark.parametrize(("covered", "mode"), [(("mr-1", "mr-2"), "LINEAGE"), (("mr-1",), "LEGACY_LINEAGE_FALLBACK")],
                         ids=["whole", "partial"])
def test_the_loader_reads_the_lineage_of_the_feeding_runs_only(syn_db, covered, mode):
    _feeding_world(syn_db, commit=CLEAN, named=("mr-1",))
    syn_db.append_tables({"train_lineage": pd.concat([
        _lineage_rows("ti-1", CLEAN, model_run=covered, hypothesis=("H-1",)),
        _lineage_rows("ti-other", DIRTY_RUN, model_run=("mr-other",), experiment=("e-other",))], ignore_index=True)})
    cfg = load_sport_config("mlb")
    inputs = D.load_review_inputs(syn_db, cfg)
    assert sorted(zip(inputs.lineage["invocation_id"], inputs.lineage["kind"], inputs.lineage["row_id"])) == sorted(
        [("ti-1", "model_run", m) for m in covered] + [("ti-1", "hypothesis", "H-1")])
    assert D.build_review(inputs, cfg)["lineage"]["mode"] == mode

