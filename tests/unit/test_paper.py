"""Paper trading (spec Sec 15, 21): sizing, pricing, reruns, exits, settlement, guards.

Research only: nothing here places an order. Two worlds are used.

hand world   the factories' game (NYY YES, bid 0.55 / ask 0.57, 180 contracts at the ask) plus a
             T-10 exit quote (0.64 / 0.66, 400 at the bid), priced by a stub model whose
             prediction is FORCED to p_market + 0.14 (conservative YES = p_market + 0.10). An honest
             model on synthetic data finds no trade (best net edge about -0.045), so every BUY in
             this file exists only because its edge is forced. Numbers are computed by hand
             (taker fee = 0.07 x C x P x (1 - P), rounded up to the next cent):
               entry  0.57 + 0.25 x 0.02 = 0.575; floor(min(100 / 0.575, 0.10 x 180)) = 18 contracts,
                      stake 10.35, fee ceil(0.07 x 18 x 0.575 x 0.425 = 0.3079) = 0.31, slippage 0.09
               exit   0.64 - 0.25 x 0.02 = 0.635 for min(18, 0.10 x 400) = 18, proceeds 11.43,
                      fee ceil(0.07 x 18 x 0.635 x 0.365 = 0.2920) = 0.30, slippage 0.09
trained      a small synthetic season (2026-03-01..04-06) trained once per module by `run_training`;
             used where the real model must be exercised (prices, reruns, look-ahead, qualification).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factories import GAME_DATE, START, feature_definition, feature_value, frame, observation, paper_trade
from src.backtest import holdout as H
from src.config import REPO_ROOT, load_sport_config
from src.features.mlb import FeatureLeakError, write_features
from src.models.residual import FittedModel
from src.models.train import run_training
from src.portfolio import paper as P
from src.sports.synthetic import SyntheticSource
from src.storage.database import connect
from src.storage.ingest import ingest
from src.storage.schemas import DEGRADED_EXECUTION

ENTRY_SNAP = START - timedelta(minutes=30)       # the canonical snapshot of every hand-world contract
RUN_AT = START - timedelta(minutes=20)           # a paper run after the T-30 snapshot, before first pitch
SETTLED_AT = START + timedelta(hours=3)          # when the hand-world outcomes become known
SELECTION = {"selected": "logistic_residual", "holdout_models": ["mr-hand"], "strategy_family": "outcome_residual",
             "quote_quality": "executable", "feature_set_hash": "fsh"}
FORCED = 0.14                                    # p_model = p_market + FORCED
BAND = 0.04                                      # lower/upper bound = p_model -/+ BAND
DEEP = dict(ask_size=10_000.0, bid_size=10_000.0)  # a book the 10% depth cap never binds on


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    """No test reads or writes the repo's data/ or its committed freeze manifest."""
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(P, "manifest_path", lambda sport: tmp_path / "no_freeze_manifest.json")


@pytest.fixture
def forced(monkeypatch):
    """FittedModel.predict, except that a model whose info carries `forced` FORCES its edge:
    p_model = p_market + forced with a +/- BAND interval. Honest models are untouched."""
    honest = FittedModel.predict

    def predict(self, X, p_market):
        shift = self.info.get("forced")
        if shift is None:
            return honest(self, X, p_market)
        p = np.asarray(p_market, dtype=float)
        pm = p + shift
        return pd.DataFrame({"p_market": p, "p_model_raw": pm, "p_model": pm, "lower_bound": pm - BAND,
                             "upper_bound": pm + BAND, "p_conservative_yes": pm - BAND,
                             "p_conservative_no": 1.0 - (pm + BAND), "uncertainty_json": None})

    monkeypatch.setattr(FittedModel, "predict", predict)


# ---------------------------------------------------------------------------
# The hand world
# ---------------------------------------------------------------------------

EXIT_QUOTE = dict(market_yes_bid=0.64, market_yes_ask=0.66, spread=0.02, bid_size=400.0, ask_size=300.0,
                  p_mid=0.65, market_last=0.65)
WIDE = dict(observation_id="SYN-BOS@T-30", game_id="syn-mlb-2", contract_id="SYN-BOS", home_team="BOS",
            away_team="NYM", yes_team="BOS", market_yes_bid=0.40, market_yes_ask=0.52, spread=0.12, p_mid=0.46)
OTHER_SIDE = dict(observation_id="SYN-TB@T-30", contract_id="SYN-TB", yes_team="TB", yes_side="away",
                  market_yes_bid=0.43, market_yes_ask=0.45, spread=0.02, p_mid=0.44)


def add_model(db, tmp_path, *, run_id="mr-hand", through=ENTRY_SNAP - timedelta(days=30), features=(),
              shift=FORCED):
    """A stub logistic_residual final-period model (its edge is forced) and its model_runs row."""
    model = FittedModel("logistic_residual", None, None, None, list(features), {"forced": shift})
    path = model.save(tmp_path / "models" / f"{run_id}.pkl")
    db.insert("model_runs", frame({
        "model_run_id": run_id, "model_type": "logistic_residual", "period_id": "2025H1", "feature_set_id": "fs",
        "feature_set_hash": "fsh", "config_hash": "c", "train_data_through": through,
        "predict_from": through + timedelta(days=1), "artifact_path": str(path), "dataset": db.dataset,
        "created_at": through + timedelta(hours=1), "is_synthetic": db.is_synthetic}))


def hand_world(db, tmp_path, *, exit_quote=True, **model):
    rows = [observation(), observation(observation_id="SYN-BOS@T-30", **{k: v for k, v in WIDE.items()
                                                                          if k != "observation_id"})]
    if exit_quote:
        rows.append(observation(offset=10, **EXIT_QUOTE))
    if not db.is_synthetic:                     # the same hand rows, labelled as live order-book quotes
        rows = [r | {"is_synthetic": False, "quote_source": "orderbook"} for r in rows]
    db.insert("observations", frame(*rows))
    add_model(db, tmp_path, **model)


def settle(db, contract_id, outcome, at=SETTLED_AT):
    """The game ends: every snapshot of `contract_id` learns its outcome, known from `at`."""
    db.con.execute("UPDATE observations SET outcome = ?, outcome_available_at = ? WHERE contract_id = ?",
                   [outcome, at, contract_id])


def exit_cfg():
    cfg = load_sport_config("mlb")
    cfg["execution"] |= {"exit_rule": "exit_at_offset", "exit_offset_min": 10}
    return cfg


def by_contract(df):
    return {r["contract_id"]: r for r in df.to_dict(orient="records")}


def settled_by_trade(db):
    t = db.df("SELECT paper_trade_id, contract_id FROM paper_trades")
    s = db.df("SELECT * FROM paper_settlements").merge(t, on="paper_trade_id")
    return by_contract(s)


# ---------------------------------------------------------------------------
# 1. Fractional Kelly and the caps (decide_slate on hand-built slate rows; edge is forced)
# ---------------------------------------------------------------------------
# Deep book (10,000 at the ask): the flat fill at stake 100 is floor(100 / 0.575) = 173 contracts,
# fee ceil(0.07 x 173 x 0.575 x 0.425 = 2.9594) = 2.96, so the all-in price is q = 0.575 + 2.96 / 173.
Q_DEEP = 0.575 + 2.96 / 173


def f_star(p, q=Q_DEEP):
    return (p - q) / (1 - q)


def slate_row(p_cons_yes, game="syn-mlb-1", team="NYY", **over):
    """A priced slate row: the factories' quote with a FORCED conservative YES probability."""
    o = observation(game_id=game, contract_id=f"SYN-{team}", observation_id=f"SYN-{team}@T-30", yes_team=team, **over)
    pm = p_cons_yes + BAND
    return o | {"p_market": o["p_mid"], "p_model": pm, "p_model_raw": pm, "lower_bound": p_cons_yes,
                "upper_bound": pm + BAND, "p_conservative_yes": p_cons_yes, "p_conservative_no": 1 - (pm + BAND),
                "uncertainty_json": None, "outcome": None, "depth_json": None, "model_run_id": "mr-hand",
                "period_id": "2025H1"}


def paper_cfg(**paper):
    cfg = load_sport_config("mlb")
    cfg["paper"] |= paper
    return cfg


def decide(rows, cfg, logged=()):
    out = P.decide_slate(pd.DataFrame(rows), cfg, frame(*logged) if logged else pd.DataFrame())
    return [(r["observation_id"], d, s) for r, d, s in out]


def logged_buy(game, stake, day=GAME_DATE, **over):
    return paper_trade(paper_trade_id=f"pt-{game}", game_id=game, stake=stake, official_date=day, **over)


def test_uncapped_kelly_below_the_position_cap():
    """Sizing: stake = kelly_fraction x f* x bankroll when no cap binds (capped_by None).
    Forced edge, p_cons 0.64: f* = (0.64 - q) / (1 - q) = 0.11741; 0.25 x f* x 10000 = 293.52 < 500."""
    [(_, d, s)] = decide([slate_row(0.64, **DEEP)], paper_cfg(max_position_frac=0.05, max_game_frac=0.05))
    assert s["requested_stake"] == pytest.approx(0.25 * f_star(0.64) * 10_000, abs=1e-9)
    assert s["requested_stake"] == pytest.approx(293.52, abs=0.01)
    assert s["allowed_stake"] == s["requested_stake"] and s["capped_by"] is None
    # re-decided at 293.52: floor(293.52 / 0.575) = 510 contracts, stake 293.25, fee ceil(8.7238) = 8.73
    assert (d["decision"], d["quantity"], d["calculated_fee"]) == ("BUY_YES", 510.0, 8.73)
    assert d["stake"] == pytest.approx(293.25, abs=1e-9)


def test_position_cap_binds():
    """Sizing: Kelly above max_position_frac x bankroll is cut to it (capped_by 'position').
    Forced edge, p_cons 0.66: 0.25 x f* x 10000 = 416.11 > 100."""
    [(_, d, s)] = decide([slate_row(0.66, **DEEP)], paper_cfg())
    assert 0.25 * f_star(0.66) * 10_000 == pytest.approx(416.11, abs=0.01)
    assert (s["requested_stake"], s["allowed_stake"], s["capped_by"]) == (100.0, 100.0, "position")
    assert (d["decision"], d["quantity"], d["calculated_fee"]) == ("BUY_YES", 173.0, 2.96)
    assert d["stake"] == pytest.approx(99.475, abs=1e-9)                         # 173 x 0.575


def test_game_cap_binds():
    """Sizing: the per-game cap cuts the stake (capped_by 'game'). A logged BUY in the same game
    already makes a new one a same_game_position pass, so the game cap is exercised through config:
    position cap 500, game cap 200. Forced edge, p_cons 0.66: requested 416.11, allowed 200."""
    [(_, d, s)] = decide([slate_row(0.66, **DEEP)], paper_cfg(max_position_frac=0.05, max_game_frac=0.02))
    assert s["requested_stake"] == pytest.approx(0.25 * f_star(0.66) * 10_000, abs=1e-9)
    assert (s["allowed_stake"], s["capped_by"]) == (200.0, "game")
    assert (d["decision"], d["quantity"], d["calculated_fee"]) == ("BUY_YES", 347.0, 5.94)  # floor(200 / 0.575)
    assert d["stake"] == pytest.approx(199.525, abs=1e-9)


def test_logged_buy_in_the_game_blocks_a_second_position():
    """One position per game across runs: an earlier logged BUY in the game -> same_game_position,
    unsized. A logged PASS in the game does not block (its stake is nothing)."""
    [(_, d, s)] = decide([slate_row(0.66, **DEEP)], paper_cfg(), [logged_buy("syn-mlb-1", 10.0)])
    assert (d["decision"], d["pass_reason"]) == ("PASS", "same_game_position")
    assert s == {"requested_stake": None, "allowed_stake": None, "capped_by": None}
    passed = logged_buy("syn-mlb-1", None, decision="PASS", pass_reason="below_min_edge")
    [(_, d, _)] = decide([slate_row(0.66, **DEEP)], paper_cfg(), [passed])
    assert d["decision"] == "BUY_YES"


def test_daily_cap_binds_across_games_in_one_slate():
    """Sizing: the daily cap is shared by the slate's games, filled in net-edge order, and a BUY
    commits its filled stake (not its allowance). Daily cap 150. Forced edges: NYY (p 0.66, listed
    second) is first: allowed 100, fills 173 for 99.475; BOS (p 0.65) gets 150 - 99.475 = 50.525,
    fills floor(50.525 / 0.575) = 87 for 50.025. Output keeps the input order."""
    rows = [slate_row(0.65, game="syn-mlb-2", team="BOS", **DEEP), slate_row(0.66, **DEEP)]
    out = decide(rows, paper_cfg(max_daily_new_frac=0.015))
    assert [oid for oid, _, _ in out] == ["SYN-BOS@T-30", "SYN-NYY@T-30"]
    (_, bos, bos_s), (_, nyy, nyy_s) = out
    assert (nyy_s["allowed_stake"], nyy_s["capped_by"], nyy["quantity"]) == (100.0, "position", 173.0)
    assert bos_s["requested_stake"] == 100.0 and bos_s["capped_by"] == "daily"
    assert bos_s["allowed_stake"] == pytest.approx(150 - 99.475, abs=1e-9)
    assert (bos["decision"], bos["quantity"]) == ("BUY_YES", 87.0)
    assert bos["stake"] == pytest.approx(50.025, abs=1e-9)


def test_daily_cap_counts_earlier_logged_stakes_of_the_same_day_only():
    """The book is seeded from logged BUYs of the same official date: 250 + 200 logged today leave
    500 - 450 = 50 (capped_by 'daily'); a logged BUY on another date takes nothing from today.
    Forced edge, p_cons 0.66: stake 50 fills floor(50 / 0.575) = 86 contracts for 49.45."""
    today = [logged_buy("syn-mlb-8", 250.0), logged_buy("syn-mlb-9", 200.0)]
    yesterday = [logged_buy("syn-mlb-7", 300.0, day=GAME_DATE - timedelta(days=1))]
    [(_, d, s)] = decide([slate_row(0.66, **DEEP)], paper_cfg(), today + yesterday)
    assert (s["requested_stake"], s["allowed_stake"], s["capped_by"]) == (100.0, 50.0, "daily")
    assert (d["decision"], d["quantity"]) == ("BUY_YES", 86.0)
    assert d["stake"] == pytest.approx(49.45, abs=1e-9)


@pytest.mark.parametrize("third_stake, allowed", [(50.0, 0.0), (47.0, 3.0)])
def test_no_headroom_is_an_exposure_cap_pass(third_stake, allowed):
    """exposure_cap: no daily headroom left (500 logged), or too little to buy the 10-contract
    minimum (3 left: floor(3 / 0.575) = 5). Forced edge; the pass records what was asked for."""
    logged = [logged_buy("syn-mlb-8", 250.0), logged_buy("syn-mlb-9", 200.0), logged_buy("syn-mlb-7", third_stake)]
    [(_, d, s)] = decide([slate_row(0.66, **DEEP)], paper_cfg(), logged)
    assert (d["decision"], d["pass_reason"]) == ("PASS", "exposure_cap")
    assert (s["requested_stake"], s["capped_by"]) == (100.0, "daily")
    assert s["allowed_stake"] == pytest.approx(allowed, abs=1e-9)
    assert d.get("stake") is None and d.get("quantity") is None


def test_kelly_stake_too_small_to_fill_is_its_own_pass():
    """kelly_size_below_minimum: no cap binds, but the Kelly stake buys fewer than 10 contracts.
    Forced edge, p_cons 0.66, kelly_fraction 0.003: 0.003 x 0.16644 x 10000 = 4.99 -> 8 contracts."""
    [(_, d, s)] = decide([slate_row(0.66, **DEEP)], paper_cfg(kelly_fraction=0.003))
    assert (d["decision"], d["pass_reason"]) == ("PASS", "kelly_size_below_minimum")
    assert s["capped_by"] is None
    assert s["requested_stake"] == pytest.approx(0.003 * f_star(0.66) * 10_000, abs=1e-9)
    assert s["requested_stake"] == pytest.approx(4.99, abs=0.01)


def test_flat_pass_is_not_sized():
    """A row the backtest rule passes (spread 0.12 > 0.10) is not sized, even with forced edge."""
    wide = {k: v for k, v in WIDE.items() if k not in ("observation_id", "game_id", "contract_id", "yes_team")}
    [(_, d, s)] = decide([slate_row(0.80, game="syn-mlb-2", team="BOS", **wide)], paper_cfg())
    assert (d["decision"], d["pass_reason"]) == ("PASS", "wide_spread")
    assert s == {"requested_stake": None, "allowed_stake": None, "capped_by": None}


def test_filled_stake_never_exceeds_the_allowance_or_the_caps():
    """For every BUY: stake <= allowed_stake <= requested_stake, and committed stakes (seeded ones
    included) stay within the game and daily caps. 40 random slates with forced edges, random caps."""
    rng = np.random.default_rng(20260924)
    buys = 0
    for _ in range(40):
        paper = dict(kelly_fraction=float(rng.uniform(0.05, 0.5)), max_position_frac=float(rng.uniform(0.002, 0.03)),
                     max_game_frac=float(rng.uniform(0.002, 0.03)), max_daily_new_frac=float(rng.uniform(0.005, 0.05)))
        cfg = paper_cfg(**paper)
        rows = [slate_row(float(rng.uniform(0.60, 0.85)), game=f"g{i}", team=f"T{i}",
                          ask_size=float(rng.choice([180.0, 2_000.0, 10_000.0]))) for i in range(6)]
        seeded = float(rng.uniform(0, 0.03)) * 10_000
        out = decide(rows, cfg, [logged_buy("g-earlier", seeded)])
        by_game, day = {"g-earlier": seeded}, seeded
        for _, d, s in out:
            if d["decision"] == "PASS":
                continue
            buys += 1
            assert d["stake"] <= s["allowed_stake"] + 1e-9 <= s["requested_stake"] + 2e-9
            assert s["requested_stake"] <= paper["max_position_frac"] * 10_000 + 1e-9
            by_game[d["game_id"]] = by_game.get(d["game_id"], 0.0) + d["stake"]
            day += d["stake"]
        assert max(by_game.values()) <= max(paper["max_game_frac"] * 10_000, seeded) + 1e-9
        assert day <= max(paper["max_daily_new_frac"] * 10_000, seeded) + 1e-9
    assert buys >= 20                               # the sweep actually exercised BUYs


# ---------------------------------------------------------------------------
# The trained world (built once per module; each test gets its own copy)
# ---------------------------------------------------------------------------

TRAINED_DAY = date(2026, 4, 6)
TRAINED_NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)


def small_season():
    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= {"seasons": [2026], "through_date": TRAINED_DAY.isoformat()}
    cfg["walk_forward"]["periods"] = [
        {"id": "P0", "role": "train_only", "start": date(2026, 3, 1), "end": date(2026, 3, 29)},
        {"id": "P1", "role": "evaluation", "start": date(2026, 3, 30), "end": date(2026, 4, 2)},
        {"id": "PH", "role": "holdout", "start": date(2026, 4, 3), "end": date(2026, 4, 30)},
    ]
    cfg["model"]["min_train_games"] = 30
    cfg["uncertainty"]["ensemble_size"] = 5
    return cfg


@pytest.fixture(scope="module")
def trained_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper_trained")
    cfg = small_season()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUANT_DATA_DIR", str(root / "data"))          # model artifacts land here
        db = connect("mlb", "synthetic", path=root / "world.duckdb", logs=root / "build_logs")
        try:
            ingest(db, SyntheticSource(cfg), date(2026, 1, 1), TRAINED_DAY)
            write_features(db, cfg, TRAINED_DAY)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # expit overflow on synthetic extremes
                selection = dict(run_training(db, cfg, now=TRAINED_NOW, threads=2).selection)
        finally:
            db.close()
    return SimpleNamespace(path=root / "world.duckdb", cfg=cfg, selection=selection)


@pytest.fixture
def trained(trained_file, tmp_path):
    path = tmp_path / "world.duckdb"
    shutil.copy(trained_file.path, path)
    db = connect("mlb", "synthetic", path=path, logs=tmp_path / "logs")
    yield SimpleNamespace(db=db, cfg=copy.deepcopy(trained_file.cfg), selection=dict(trained_file.selection))
    db.close()


def quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kw)


# ---------------------------------------------------------------------------
# 2. On-the-fly prices equal what `quant train` stored
# ---------------------------------------------------------------------------

def test_on_the_fly_prices_equal_the_stored_final_period_predictions(trained):
    """The paper slate is priced by the stored final-period model on the stored features: the same
    numbers `quant train` wrote to predictions, for the T-30 slate and every other offset (1e-9)."""
    db, sel = trained.db, trained.selection
    sm = P.slate_model(db, sel)
    assert sm.run["period_id"] == "PH" and sm.run["model_type"] == sel["selected"] == "logistic_residual"
    stored = db.df("SELECT * FROM predictions WHERE model_run_id = ?", [sm.run["model_run_id"]])
    obs = P._typed(db.df(f"SELECT {P._OBS_COLS} FROM observations o WHERE o.observation_id IN "
                         f"(SELECT unnest(?::VARCHAR[]))", [stored["observation_id"].tolist()]))
    assert len(obs) == len(stored) == 94 and obs["snapshot_offset_min"].nunique() == 5
    slate = quiet(P.load_slate, db, trained.cfg, sel, TRAINED_DAY, now=TRAINED_NOW)
    everything = quiet(P.price_rows, obs, P.model_features(db, obs, sm.model), sm)
    for rows, n in ((slate.rows, 18), (everything, 94)):
        m = rows.merge(stored, on="observation_id", suffixes=("", "_stored"), validate="one_to_one")
        assert len(m) == n
        for c in ("p_model", "lower_bound", "upper_bound", "p_conservative_yes", "p_conservative_no"):
            assert np.abs(m[c] - m[f"{c}_stored"]).max() <= 1e-9, c


# ---------------------------------------------------------------------------
# 3. Reruns are idempotent
# ---------------------------------------------------------------------------

def test_rerunning_a_day_logs_nothing_new(trained):
    """One paper decision per snapshot: a second run of the same day logs no row and counts the
    first run's decisions as already logged. Honest model: every decision is a pass."""
    db, cfg, sel = trained.db, trained.cfg, trained.selection
    first = quiet(P.run_paper, db, cfg, sel, TRAINED_DAY, now=TRAINED_NOW)
    assert (len(first.decisions), first.already_logged) == (18, 0)
    assert set(first.decisions["pass_reason"]) == {"below_min_edge"}
    second = quiet(P.run_paper, db, cfg, sel, TRAINED_DAY, now=TRAINED_NOW + timedelta(hours=1))
    assert (len(second.decisions), second.already_logged) == (0, 18)
    logged = db.df("SELECT paper_trade_id, observation_id FROM paper_trades")
    assert len(logged) == 18 and logged["paper_trade_id"].is_unique
    assert (logged["paper_trade_id"] == logged["observation_id"].map(
        lambda o: P.paper_trade_id("synthetic", o))).all()


def test_paper_trade_id_is_deterministic_across_processes():
    """paper_trade_id depends only on (dataset, observation_id): the same in a fresh interpreter
    with a different hash seed, and different for another dataset."""
    here = P.paper_trade_id("synthetic", "SYN-NYY@T-30")
    code = "from src.portfolio.paper import paper_trade_id; print(paper_trade_id('synthetic', 'SYN-NYY@T-30'))"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
                         env=os.environ | {"PYTHONHASHSEED": "12345"})
    assert out.stdout.strip() == here and here.startswith("pt-")
    assert P.paper_trade_id("live", "SYN-NYY@T-30") != here
    assert P.paper_trade_id("synthetic", "SYN-NYY@T-10") != here


def test_rerun_with_a_forced_buy_does_not_double_the_position(syn_db, tmp_path, forced):
    """A rerun logs only new contracts; the other side of a game already bought is a
    same_game_position pass. Edge is forced (both sides would be BUY_YES on their own)."""
    hand_world(syn_db, tmp_path)
    cfg = load_sport_config("mlb")
    first = P.run_paper(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT)
    assert by_contract(first.decisions)["SYN-NYY"]["decision"] == "BUY_YES"
    syn_db.insert("observations", frame(observation(**OTHER_SIDE)))
    second = P.run_paper(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT + timedelta(minutes=1))
    assert second.already_logged == 2
    [row] = second.decisions.to_dict(orient="records")
    assert (row["contract_id"], row["decision"], row["pass_reason"]) == ("SYN-TB", "PASS", "same_game_position")
    tb = second.slate.rows[second.slate.rows["contract_id"] == "SYN-TB"]
    [alone] = P.flat_decisions(tb, P.entry_policy(cfg)).to_dict(orient="records")
    assert alone["decision"] == "BUY_YES"                         # it passed only because the game is held
    trades = syn_db.df("SELECT * FROM paper_trades")
    assert len(trades) == 3 and (trades["decision"] != "PASS").sum() == 1


# ---------------------------------------------------------------------------
# 4. Exits and settlement (hand world; edge is forced)
# ---------------------------------------------------------------------------

def test_forced_buy_is_logged_with_the_backtest_fill(syn_db, tmp_path, forced):
    """The logged BUY carries the backtest's fill at the flat-stake size cap: 18 at 0.575, stake
    10.35, fee 0.31, slippage 0.09 (all hand-computed), Kelly 415.5 capped to 100 by position."""
    hand_world(syn_db, tmp_path)
    res = P.run_paper(syn_db, load_sport_config("mlb"), SELECTION, GAME_DATE, now=RUN_AT)
    t = by_contract(syn_db.df("SELECT * FROM paper_trades"))["SYN-NYY"]
    assert (t["decision"], t["side"], t["hypothetical_size"], t["entry_price"]) == ("BUY_YES", "yes", 18.0, 0.575)
    assert (t["stake"], t["calculated_fee"], t["slippage_cost"]) == pytest.approx((10.35, 0.31, 0.09), abs=1e-9)
    assert t["net_edge"] == pytest.approx(0.66 - 0.575 - 0.31 / 18, abs=1e-9)
    assert (t["requested_stake"], t["allowed_stake"], t["capped_by"]) == (100.0, 100.0, "position")
    assert t["decision_ts"] == ENTRY_SNAP and t["train_data_through"] < t["decision_ts"]
    assert by_contract(res.decisions)["SYN-BOS"]["pass_reason"] == "wide_spread"


@pytest.mark.parametrize("outcome, gross, net", [(1, 18 * (1 - 0.57), 18 - 10.35 - 0.31),
                                                  (0, 18 * (0 - 0.57), 0 - 10.35 - 0.31)])
def test_hold_to_settlement_round_trip(syn_db, tmp_path, forced, outcome, gross, net):
    """Held to settlement: net = 18 x side_won - stake 10.35 - fee 0.31 (7.34 won, -10.66 lost),
    gross at the quoted ask 0.57. Settled only once outcome_available_at <= now, and only once."""
    hand_world(syn_db, tmp_path)
    P.run_paper(syn_db, load_sport_config("mlb"), SELECTION, GAME_DATE, now=RUN_AT)
    settle(syn_db, "SYN-NYY", outcome)
    assert P.run_settlement(syn_db, now=SETTLED_AT - timedelta(microseconds=1)).empty
    P.run_settlement(syn_db, now=SETTLED_AT)
    s = settled_by_trade(syn_db)["SYN-NYY"]
    assert (s["outcome"], s["side_won"], s["held_quantity"]) == (outcome, outcome, 18.0)
    assert s["net_pnl"] == pytest.approx(net, abs=1e-9) and s["net_pnl"] == pytest.approx([-10.66, 7.34][outcome])
    assert s["pnl"] == pytest.approx(gross, abs=1e-9)
    assert s["fees"] == pytest.approx(0.31, abs=1e-9) and pd.isna(s["exit_proceeds"])
    assert P.run_settlement(syn_db, now=SETTLED_AT + timedelta(days=1)).empty
    assert syn_db.scalar("SELECT count(*) FROM paper_settlements") == 1


def test_exit_at_offset_round_trip(syn_db, tmp_path, forced):
    """T-30 entry, sold into the T-10 bid: proceeds 11.43, exit fee 0.30. The settlement waits for
    the exit leg, then net = 11.43 - 10.35 - 0.31 - 0.30 = 0.47 whatever the outcome (YES lost here);
    gross 18 x (0.64 - 0.57) = 1.26 = net + both fees + both slippages (0.61 + 0.18)."""
    hand_world(syn_db, tmp_path)
    cfg = exit_cfg()
    P.run_paper(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT)
    t = by_contract(syn_db.df("SELECT * FROM paper_trades"))["SYN-NYY"]
    assert (t["exit_rule"], t["exit_offset_min"]) == ("exit_at_offset", 10)
    settle(syn_db, "SYN-NYY", 0)
    settle(syn_db, "SYN-BOS", 0)
    early = P.run_settlement(syn_db, now=SETTLED_AT)             # the exit leg is not recorded yet
    assert list(early["decision"]) == ["PASS"]
    x = P.run_exits(syn_db, cfg, now=START - timedelta(minutes=5)).iloc[0]
    assert (x["exit_status"], x["exit_observation_id"], x["exit_quantity"], x["held_quantity"]) == \
        ("exited", "SYN-NYY@T-10", 18.0, 0.0)
    assert (x["exit_quoted_price"], x["exit_price"]) == pytest.approx((0.64, 0.635), abs=1e-9)
    assert (x["exit_proceeds"], x["exit_fee"], x["exit_slippage_cost"]) == pytest.approx((11.43, 0.30, 0.09), abs=1e-9)
    assert P.run_exits(syn_db, cfg, now=START).empty                # sold once
    P.run_settlement(syn_db, now=SETTLED_AT)
    s = settled_by_trade(syn_db)["SYN-NYY"]
    assert s["net_pnl"] == pytest.approx(11.43 - 10.35 - 0.31 - 0.30, abs=1e-9)
    assert s["net_pnl"] == pytest.approx(0.47, abs=1e-9)
    assert s["pnl"] == pytest.approx(1.26, abs=1e-9)
    assert s["fees"] == pytest.approx(0.61, abs=1e-9) and s["exit_proceeds"] == pytest.approx(11.43, abs=1e-9)
    assert s["net_pnl"] == pytest.approx(s["pnl"] - 0.31 - 0.30 - 0.09 - 0.09, abs=1e-9)
    assert (s["held_quantity"], s["side_won"]) == (0.0, 0)


def test_missing_exit_quote_waits_for_the_game_then_holds(syn_db, tmp_path, forced):
    """No T-10 observation: the exit waits while the game is unresolved, then the position is
    recorded as held (no_valid_exit_quote) and settles like a held one: 18 - 10.35 - 0.31 = 7.34."""
    hand_world(syn_db, tmp_path, exit_quote=False)
    cfg = exit_cfg()
    P.run_paper(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT)
    assert P.run_exits(syn_db, cfg, now=START + timedelta(hours=1)).empty
    settle(syn_db, "SYN-NYY", 1)
    x = P.run_exits(syn_db, cfg, now=SETTLED_AT).iloc[0]
    assert (x["exit_status"], x["exit_reason"], x["held_quantity"]) == ("held", "no_valid_exit_quote", 18.0)
    assert pd.isna(x["exit_quantity"]) and pd.isna(x["exit_fee"])
    P.run_settlement(syn_db, now=SETTLED_AT)
    s = settled_by_trade(syn_db)["SYN-NYY"]
    assert s["net_pnl"] == pytest.approx(7.34, abs=1e-9) and s["fees"] == pytest.approx(0.31, abs=1e-9)


def test_passes_settle_with_the_outcome_only(syn_db, tmp_path, forced):
    """Every decision is settled, passes included, but a pass has no side, fees or P&L."""
    hand_world(syn_db, tmp_path)
    P.run_paper(syn_db, load_sport_config("mlb"), SELECTION, GAME_DATE, now=RUN_AT)
    settle(syn_db, "SYN-BOS", 1)
    out = P.run_settlement(syn_db, now=SETTLED_AT)
    [s] = out.to_dict(orient="records")                            # NYY is unresolved: not settled
    assert (s["decision"], s["outcome"], s["result"]) == ("PASS", 1, "yes")
    for c in ("side", "side_won", "pnl", "net_pnl", "fees", "held_quantity", "exit_proceeds"):
        assert s[c] is None or pd.isna(s[c]), c


def test_a_logged_fill_that_disagrees_with_its_fee_schedule_cannot_settle(syn_db):
    """Settlement rebuilds the entry from its fee schedule: a stored fee that the schedule does not
    produce (0.17, not ceil(0.07 x 10 x 0.575 x 0.425) = 0.18) is refused, not settled."""
    syn_db.insert("observations", frame(observation(outcome=1, outcome_available_at=SETTLED_AT)))
    syn_db.append_log("paper_trades", frame(paper_trade(calculated_fee=0.17)), ts=RUN_AT)
    with pytest.raises(ValueError, match="stored calculated_fee 0.17"):
        P.run_settlement(syn_db, now=SETTLED_AT)
    assert syn_db.scalar("SELECT count(*) FROM paper_settlements") == 0


# ---------------------------------------------------------------------------
# 5. Guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("through, refused", [
    (ENTRY_SNAP - timedelta(microseconds=1), False),  # control: trained strictly before the snapshot
    (ENTRY_SNAP, True),                                # trained through the snapshot itself
    (ENTRY_SNAP + timedelta(hours=4), True),           # trained on this game's own outcome
])
def test_model_trained_through_a_slate_snapshot_is_refused(syn_db, tmp_path, forced, through, refused):
    """LookAheadModel unless train_data_through < every slate snapshot; nothing is logged."""
    hand_world(syn_db, tmp_path, through=through)
    cfg = load_sport_config("mlb")
    if refused:
        with pytest.raises(P.LookAheadModel, match="mr-hand"):
            P.run_paper(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT)
        assert syn_db.scalar("SELECT count(*) FROM paper_trades") == 0
    else:
        assert len(P.run_paper(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT).decisions) == 2


def test_look_ahead_guard_on_the_trained_model(trained):
    """The real final-period model passes the guard; the same run claiming data through the
    slate's first snapshot is refused."""
    db, sel = trained.db, trained.selection
    slate = quiet(P.load_slate, db, trained.cfg, sel, TRAINED_DAY, now=TRAINED_NOW)
    first = slate.rows["snapshot_time"].min()
    sm = slate.model
    assert sm.run["train_data_through"] < first
    leaky = P.SlateModel(sm.run | {"train_data_through": first}, sm.model)
    with pytest.raises(P.LookAheadModel, match=sm.run["model_run_id"]):
        quiet(P.price_rows, slate.rows, slate.X, leaky)


@pytest.mark.parametrize("snapshot_shift, leaks", [
    (timedelta(0), False),                 # control: the real snapshot
    (timedelta(minutes=-5), False),        # boundary: available exactly at the snapshot is allowed
    (timedelta(minutes=-5, microseconds=-1), True),
])
def test_feature_available_after_the_snapshot_is_refused(syn_db, tmp_path, forced, snapshot_shift, leaks):
    """FeatureLeakError when a feature's available_at is later than the snapshot it would price.
    The DB refuses to store such a row, so the observation's snapshot is moved earlier instead:
    the feature is available at T-35; pricing it at T-35 is allowed, a microsecond earlier is not."""
    hand_world(syn_db, tmp_path, features=["starter_k_minus_bb"])
    syn_db.insert("feature_definitions", frame(feature_definition()))
    syn_db.insert("feature_values", frame(feature_value(ENTRY_SNAP, ENTRY_SNAP - timedelta(minutes=5))))
    model = P.slate_model(syn_db, SELECTION).model
    obs = P._typed(syn_db.df(f"SELECT {P._OBS_COLS} FROM observations o WHERE o.observation_id = 'SYN-NYY@T-30'"))
    obs["snapshot_time"] = obs["snapshot_time"] + snapshot_shift
    if leaks:
        with pytest.raises(FeatureLeakError, match="starter_k_minus_bb for SYN-NYY@T-30"):
            P.model_features(syn_db, obs, model)
    else:
        assert P.model_features(syn_db, obs, model)["starter_k_minus_bb"].tolist() == [0.12]


def test_trained_features_leak_check(trained):
    """On the trained model's real features: the slate prices; moved one microsecond before its
    latest feature, the first slate observation is refused."""
    db, sel = trained.db, trained.selection
    sm = P.slate_model(db, sel)
    obs = P._typed(db.df(P._SLATE_SQL, [TRAINED_DAY, 30, "executable"]))
    assert P.model_features(db, obs, sm.model).shape == (18, len(sm.model.features))
    oid = obs["observation_id"].iloc[0]
    latest = db.scalar("SELECT max(available_at) FROM feature_values WHERE observation_id = ? AND feature_name "
                       "IN (SELECT unnest(?::VARCHAR[]))", [oid, list(sm.model.features)])
    latest = pd.Timestamp(latest)
    latest = latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")
    obs.loc[0, "snapshot_time"] = latest - pd.Timedelta(microseconds=1)
    with pytest.raises(FeatureLeakError, match=oid):
        P.model_features(db, obs, sm.model)


@pytest.mark.parametrize("over, message", [
    ({"strategy_family": "market_behavior"}, "model selection was trained as 'market_behavior'"),
    ({"quote_quality": "midpoint"}, "unknown quote_quality 'midpoint'"),
])
def test_mismatched_selection_is_refused(syn_db, tmp_path, forced, over, message):
    """The slate is priced only by a selection trained as the configured strategy family on a
    known quote class; otherwise nothing is logged."""
    hand_world(syn_db, tmp_path)
    with pytest.raises(ValueError, match=message):
        P.run_paper(syn_db, load_sport_config("mlb"), SELECTION | over, GAME_DATE, now=RUN_AT)
    assert syn_db.scalar("SELECT count(*) FROM paper_trades") == 0


def test_candle_selection_prices_only_candle_rows(syn_db, tmp_path, forced):
    """One quote class, never mixed: a candle_degraded selection sees only candle
    observations and says so; an executable one never sees them."""
    hand_world(syn_db, tmp_path)
    candle = dict(observation_id="SYN-SEA@T-30", game_id="syn-mlb-3", contract_id="SYN-SEA", home_team="SEA",
                  away_team="OAK", yes_team="SEA", quote_source="candle", size_verified=False)
    syn_db.insert("observations", frame(observation(**candle)))
    cfg = load_sport_config("mlb")
    degraded = P.load_slate(syn_db, cfg, SELECTION | {"quote_quality": "candle_degraded"}, GAME_DATE, now=RUN_AT)
    assert degraded.rows["contract_id"].tolist() == ["SYN-SEA"] and DEGRADED_EXECUTION in degraded.notes
    executable = P.load_slate(syn_db, cfg, SELECTION, GAME_DATE, now=RUN_AT)
    assert sorted(executable.rows["contract_id"]) == ["SYN-BOS", "SYN-NYY"]
    assert DEGRADED_EXECUTION not in executable.notes


# ---------------------------------------------------------------------------
# 6. Qualification (26.18)
# ---------------------------------------------------------------------------

def test_synthetic_paper_trades_never_carry_a_freeze_hash(trained, tmp_path):
    """No frozen manifest: every decision is LOCKED with no freeze hash and its P&L is withheld;
    and the synthetic configuration itself can never be frozen."""
    db, cfg, sel = trained.db, trained.cfg, trained.selection
    res = quiet(P.run_paper, db, cfg, sel, TRAINED_DAY, now=TRAINED_NOW)
    q = res.qualification
    assert (q.status, q.manifest_hash, q.reasons) == ("LOCKED", None, ["no frozen manifest (26.3)"])
    logged = db.df("SELECT manifest_status, freeze_manifest_hash FROM paper_trades")
    assert len(logged) == 18 and set(logged["manifest_status"]) == {"LOCKED"}
    assert logged["freeze_manifest_hash"].isna().all()
    assert f"{P.WITHHELD}: no frozen manifest (26.3)" in res.notes
    path = tmp_path / "attempted_freeze.json"
    with pytest.raises(H.SyntheticFreeze):
        H.freeze(path, H.current_manifest(db, cfg, sel, TRAINED_NOW), now=TRAINED_NOW)
    assert not path.exists()


def test_a_qualifying_run_stamps_its_manifest_hash_on_every_decision(live_db, tmp_path, forced, monkeypatch):
    """Plumbing: when qualification() grants a hash, every row logged by that run carries it and
    its status, and the withheld note is not added. (Forced edge; qualification is stubbed, and the
    hand world sits in a live-dataset database, because a synthetic row may never carry a hash.)"""
    hand_world(live_db, tmp_path)
    monkeypatch.setattr(P, "qualification", lambda *a, **k: P.Qualification("FROZEN", "mh-test", []))
    res = P.run_paper(live_db, load_sport_config("mlb"), SELECTION, GAME_DATE, now=RUN_AT)
    logged = live_db.df("SELECT decision, manifest_status, freeze_manifest_hash FROM paper_trades")
    assert len(logged) == 2 and set(logged["decision"]) == {"BUY_YES", "PASS"}
    assert set(logged["manifest_status"]) == {"FROZEN"} and set(logged["freeze_manifest_hash"]) == {"mh-test"}
    assert not any(n.startswith(P.WITHHELD) for n in res.notes)


def test_a_hand_written_synthetic_freeze_does_not_qualify(trained, tmp_path, monkeypatch):
    """Synthetic data never qualifies for 26.18: not even under a FROZEN manifest that matches the
    current configuration (written by hand, with a pinned git commit), which freeze() would refuse."""
    db, cfg, sel = trained.db, trained.cfg, trained.selection
    monkeypatch.setattr(H, "git_commit", lambda: "0" * 40)
    current = H.current_manifest(db, cfg, sel, TRAINED_NOW)
    assert current["dataset"] == "synthetic"
    path = tmp_path / "hand_frozen_manifest.json"
    path.write_text(json.dumps(current | {"status": "FROZEN"}))
    monkeypatch.setattr(P, "manifest_path", lambda sport: path)
    res = quiet(P.run_paper, db, cfg, sel, TRAINED_DAY, now=TRAINED_NOW)
    assert res.qualification.manifest_hash is None
    assert db.df("SELECT freeze_manifest_hash FROM paper_trades")["freeze_manifest_hash"].isna().all()
