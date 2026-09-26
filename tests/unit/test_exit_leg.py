"""The exit leg. Sell at the bid, pay a second fee, round-trip P&L.

The acceptance tests compute every number by hand (taker rate 0.07, fee rounded
up to the next cent per order; see src/backtest/execution.py). They passed
before the paper module was designed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from factories import START, frame, observation
from src.backtest.engine import Policy, decide, run_backtest
from src.backtest.execution import (
    ExitRule,
    Fill,
    NoExit,
    Position,
    QuadraticFeeSchedule,
    Quote,
    bid_depth_levels,
    bid_price,
    bid_top_size,
    settle,
    simulate_exit,
    simulate_fill,
)
from src.backtest.metrics import trading_summary
from src.config import load_sport_config
from src.storage.database import ConstraintViolation
from src.storage.schemas import BACKTEST_DECISIONS

ENTRY_T = START - timedelta(minutes=30)
EXIT_T = START - timedelta(minutes=10)
ENTRY_Q = Quote(yes_bid=0.55, yes_ask=0.57, bid_size=250.0, ask_size=180.0)  # factories.observation's quote
EXIT_Q = Quote(yes_bid=0.64, yes_ask=0.66, bid_size=400.0, ask_size=300.0)   # the market moved up by T-10
EXIT_AT_10 = ExitRule("exit_at_offset", 10)


def entry(side="yes", quote=ENTRY_Q) -> Fill:
    f = simulate_fill(side, quote, penalty_frac=0.25, stake_dollars=100.0, max_fraction_top=0.10, min_contracts=10,
                      when=ENTRY_T)
    assert isinstance(f, Fill)
    return f


def close(pos: Position, quote: Quote | None, when=EXIT_T, penalty=0.25) -> Position:
    return pos.close(quote, when, penalty_frac=penalty, max_fraction_top=0.10)


# -- acceptance: hand-computed round trips --------------------------------------

def test_round_trip_entry_at_ask_exit_at_bid_both_fees():
    """Buy 18 YES at the ask, sell all 18 into the bid at T-10.

    entry  base 0.57 + 0.25 x 0.02 = 0.575; qty floor(min(100 / 0.575, 0.10 x 180)) = 18
           cost 18 x 0.575 = 10.35; fee ceil-cent(0.07 x 18 x 0.575 x 0.425 = 0.3079) = 0.31
    exit   base 0.64 - 0.25 x 0.02 = 0.635; qty min(18, 0.10 x 400 = 40) = 18
           proceeds 18 x 0.635 = 11.43; fee ceil-cent(0.07 x 18 x 0.635 x 0.365 = 0.2920) = 0.30
    net    11.43 - 10.35 - 0.31 - 0.30 = 0.47, whatever the game's outcome
    gross  18 x (0.64 - 0.57) = 1.26 = 0.47 + 0.09 + 0.09 slippage + 0.61 fees
    """
    e = entry()
    assert (e.quantity, e.execution_price, e.stake, e.fee) == pytest.approx((18, 0.575, 10.35, 0.31))
    pos = close(Position(e, ENTRY_T, EXIT_AT_10), EXIT_Q)
    x = pos.exit
    assert (x.quoted_price, x.execution_price, x.quantity) == pytest.approx((0.64, 0.635, 18))
    assert (x.proceeds, x.fee, x.slippage_cost) == pytest.approx((11.43, 0.30, 0.09))
    assert x.fee_schedule.fee_schedule_id == "quadratic-taker-v1"
    assert pos.exit_status == "exited" and pos.exit_reason is None and pos.held_quantity == 0
    for won in (0, 1, None):
        assert pos.pnl(won)["net_pnl"] == pytest.approx(0.47)
        assert pos.pnl(won)["gross_pnl"] == pytest.approx(1.26)


def test_round_trip_held_to_settlement():
    """Same entry, held: net = payout - 10.35 - 0.31 (7.34 on a win, -10.66 on a loss); no exit fee."""
    pos = Position(entry(), ENTRY_T)
    assert pos.exit_status == "held" and pos.exit is None
    assert pos.pnl(1) == pytest.approx({"gross_pnl": 7.74, "net_pnl": 7.34})
    assert pos.pnl(0) == pytest.approx({"gross_pnl": -10.26, "net_pnl": -10.66})
    assert pos.pnl(None) == {"gross_pnl": None, "net_pnl": None}
    assert pos.pnl(1)["net_pnl"] == pytest.approx(settle(pos.entry, 1)["net_pnl"])  # one P&L formula


def test_round_trip_no_side_partial_exit():
    """Buy 25 NO, sell only 10 into thin NO bids (NO bids are the YES asks), hold 15.

    entry  no_ask 1 - 0.55 = 0.45; base 0.455; qty floor(min(219.8, 0.10 x bid_size 250)) = 25
           cost 11.375; fee ceil-cent(0.07 x 25 x 0.455 x 0.545 = 0.4340) = 0.44
    exit   no_bid 1 - 0.42 = 0.58; base 0.575; qty min(25, 0.10 x ask_size 100) = 10
           proceeds 5.75; fee ceil-cent(0.07 x 10 x 0.575 x 0.425 = 0.1711) = 0.18
    net    NO wins: 5.75 + 15 - 11.375 - 0.44 - 0.18 = 8.755; NO loses: 5.75 - 11.375 - 0.44 - 0.18 = -6.245
    """
    e = entry("no")
    assert (e.quantity, e.stake, e.fee) == pytest.approx((25, 11.375, 0.44))
    pos = close(Position(e, ENTRY_T, EXIT_AT_10), Quote(0.40, 0.42, bid_size=500.0, ask_size=100.0))
    assert (pos.exit.quoted_price, pos.exit.execution_price, pos.exit.quantity) == pytest.approx((0.58, 0.575, 10))
    assert (pos.exit.proceeds, pos.exit.fee) == pytest.approx((5.75, 0.18))
    assert pos.exit_status == "partial" and pos.exit_reason == "insufficient_exit_depth"
    assert pos.held_quantity == 15
    assert pos.pnl(1) == pytest.approx({"gross_pnl": 9.55, "net_pnl": 8.755})
    assert pos.pnl(0)["net_pnl"] == pytest.approx(-6.245)
    assert pos.pnl(None) == {"gross_pnl": None, "net_pnl": None}  # 15 contracts still await settlement


def test_exit_fee_comes_from_the_schedule_in_force_at_exit():
    """A 00:15 UTC start puts T-30 and T-10 on different UTC dates; each leg pays its own date's schedule.

    exit fee under the new schedule: ceil-cent(0.035 x 18 x 0.635 x 0.365 = 0.1460) = 0.15 (old rate: 0.30)
    """
    start = datetime(2025, 6, 2, 0, 15, tzinfo=timezone.utc)
    t_in, t_out = start - timedelta(minutes=30), start - timedelta(minutes=10)
    reg = (QuadraticFeeSchedule(fee_schedule_id="old", maker_or_taker="taker", effective_from=datetime(2021, 1, 1).date(),
                                effective_to=t_in.date(), params={"rate": 0.07}),
           QuadraticFeeSchedule(fee_schedule_id="new", maker_or_taker="taker", effective_from=t_out.date(),
                                params={"rate": 0.035}))
    e = simulate_fill("yes", ENTRY_Q, penalty_frac=0.25, stake_dollars=100.0, max_fraction_top=0.10,
                      min_contracts=10, when=t_in, registry=reg)
    pos = Position(e, t_in, EXIT_AT_10).close(EXIT_Q, t_out, penalty_frac=0.25, max_fraction_top=0.10, registry=reg)
    assert (e.fee_schedule.fee_schedule_id, e.fee) == ("old", pytest.approx(0.31))
    assert (pos.exit.fee_schedule.fee_schedule_id, pos.exit.fee) == ("new", pytest.approx(0.15))
    assert pos.pnl(1)["net_pnl"] == pytest.approx(11.43 - 10.35 - 0.31 - 0.15)
    assert pos.record(1)["exit_fee_schedule_effective_date"] == t_out.date()


# -- sell-side prices and sizes --------------------------------------------------

def test_sale_crosses_the_bid_of_the_side_held():
    assert bid_price("yes", EXIT_Q) == pytest.approx(0.64)
    assert bid_price("no", EXIT_Q) == pytest.approx(1 - 0.66)
    assert bid_top_size("yes", EXIT_Q) == 400.0 and bid_top_size("no", EXIT_Q) == 300.0


def test_exit_walks_the_bids_best_first():
    book = json.dumps({"yes_bids": [[0.62, 20], [0.64, 10]], "yes_asks": [[0.66, 5], [0.70, 7]]})
    q = Quote(0.64, 0.66, depth_json=book)
    assert bid_depth_levels("yes", q) == [(0.64, 10.0), (0.62, 20.0)]
    assert [v for lv in bid_depth_levels("no", q) for v in lv] == pytest.approx([0.34, 5.0, 0.30, 7.0])
    x = simulate_exit("yes", q, 18, penalty_frac=0.0, max_fraction_top=0.10, when=EXIT_T)
    assert x.quantity == 18 and x.depth_verified
    assert x.execution_price == pytest.approx((10 * 0.64 + 8 * 0.62) / 18)  # VWAP below the best bid
    assert x.proceeds == pytest.approx(11.36)
    capped = simulate_exit("no", q, 18, penalty_frac=0.0, max_fraction_top=0.10, when=EXIT_T)
    assert capped.quantity == 12  # displayed NO bids hold 12 contracts


def test_candle_exit_has_no_size_cap_and_is_unverified():
    x = simulate_exit("yes", Quote(0.64, 0.66), 18, penalty_frac=0.25, max_fraction_top=0.10, when=EXIT_T)
    assert x.quantity == 18 and not x.depth_verified


def test_exit_price_is_never_above_the_bid():
    for k in (0.0, 0.25, 1.0):
        x = simulate_exit("yes", EXIT_Q, 18, penalty_frac=k, max_fraction_top=0.10, when=EXIT_T)
        assert x.execution_price == pytest.approx(0.64 - k * 0.02) and x.execution_price <= x.quoted_price


# -- fallbacks: held to settlement, reason recorded -----------------------------

@pytest.mark.parametrize("quote, when, reason", [
    (None, EXIT_T, "no_valid_exit_quote"),
    (Quote(0.66, 0.64, 400.0, 300.0), EXIT_T, "no_valid_exit_quote"),      # crossed
    (EXIT_Q, ENTRY_T, "exit_not_after_entry"),
    (Quote(0.012, 0.03, 400.0, 300.0), EXIT_T, "no_exit_price"),           # 0.012 - 0.0045 <= 0.01
    (Quote(0.64, 0.66, bid_size=5.0, ask_size=300.0), EXIT_T, "insufficient_exit_depth"),  # 0.10 x 5 < 1
])
def test_failed_exit_is_held_with_its_reason(quote, when, reason):
    pos = close(Position(entry(), ENTRY_T, EXIT_AT_10), quote, when)
    assert pos.exit is None and pos.exit_status == "held" and pos.exit_reason == reason
    assert pos.exit_time == when  # the planned exit is recorded, not hidden
    assert pos.pnl(1) == pytest.approx(Position(entry(), ENTRY_T).pnl(1))


def test_no_exit_reports_its_reason():
    assert simulate_exit("yes", Quote(None, 0.5), 5, penalty_frac=0, max_fraction_top=0.1, when=EXIT_T) == \
        NoExit("yes", "no_valid_exit_quote")


def test_hold_rule_never_closes_and_positions_close_once():
    with pytest.raises(ValueError, match="held to settlement"):
        close(Position(entry(), ENTRY_T), EXIT_Q)
    pos = close(Position(entry(), ENTRY_T, EXIT_AT_10), EXIT_Q)
    with pytest.raises(ValueError, match="already closed"):
        close(pos, EXIT_Q)


# -- exit rule and open-position tracking ---------------------------------------

def test_exit_rule_validation_and_config():
    assert ExitRule.from_config({}) == ExitRule("hold_to_settlement")
    assert ExitRule.from_config({"execution": {"exit_rule": "hold_to_settlement", "exit_offset_min": 10}}).offset_min \
        is None
    assert ExitRule.from_config({"execution": {"exit_rule": "exit_at_offset", "exit_offset_min": 10}}) == EXIT_AT_10
    assert EXIT_AT_10.exit_time(START) == EXIT_T and ExitRule().exit_time(START) is None
    for bad in (("exit_at_offset", None), ("exit_at_offset", 0), ("exit_at_offset", 10.0),
                ("hold_to_settlement", 10), ("sell_on_signal", None)):
        with pytest.raises(ValueError):
            ExitRule(*bad)
    assert load_sport_config("mlb")["execution"]["exit_rule"] == "hold_to_settlement"  # the default


def test_position_is_open_from_entry_until_exit_or_settlement():
    settled = START + timedelta(hours=4)
    partial = close(Position(entry(), ENTRY_T, EXIT_AT_10), Quote(0.64, 0.66, bid_size=100.0, ask_size=300.0))
    assert partial.exit.quantity == 10 and partial.held_quantity == 8
    held = Position(entry(), ENTRY_T)
    for t, n_partial, n_held in ((ENTRY_T - timedelta(seconds=1), 0, 0), (ENTRY_T, 18, 18),
                                 (EXIT_T - timedelta(seconds=1), 18, 18), (EXIT_T, 8, 18), (settled, 0, 0)):
        assert partial.open_quantity(t, settled) == n_partial
        assert held.open_quantity(t, settled) == n_held
    assert held.open_quantity(settled + timedelta(days=1)) == 18  # unsettled: still open


# -- the shared decision path (engine) ------------------------------------------

POL = Policy(minimum_edge=0.04, maximum_spread=0.10, min_contracts=10, stake=100.0, max_fraction_top=0.10,
             maker_or_taker="taker", baseline=0.25, scenarios=(0.0, 0.25, 0.5, 1.0))
POL_EXIT = Policy(**{**POL.__dict__, "exit_rule": EXIT_AT_10})
ENTRY_KEYS = ("decision", "pass_reason", "side", "quoted_price", "execution_price", "quantity", "stake",
              "calculated_fee", "net_edge", "predicted_edge")


def entry_row(outcome=1) -> dict:
    return {"model_run_id": "mr-1", "observation_id": "SYN-NYY@T-30", "period_id": "2025H1", "game_id": "syn-mlb-1",
            "contract_id": "SYN-NYY", "snapshot_time": ENTRY_T, "snapshot_offset_min": 30,
            "event_start_time": START, "p_market": 0.56, "p_model": 0.70, "p_conservative_yes": 0.66,
            "p_conservative_no": 0.26, "market_yes_bid": 0.55, "market_yes_ask": 0.57, "bid_size": 250.0,
            "ask_size": 180.0, "depth_json": None, "spread": 0.02, "outcome": outcome}


def exit_row(**over) -> dict:
    return {"observation_id": "SYN-NYY@T-10", "snapshot_time": EXIT_T, "market_yes_bid": 0.64,
            "market_yes_ask": 0.66, "bid_size": 400.0, "ask_size": 300.0, "depth_json": None} | over


def by_scenario(rows):
    return {r["scenario"]: r for r in rows}


def test_engine_round_trip_matches_the_hand_computation():
    r = by_scenario(decide(entry_row(), POL_EXIT, exit_row()))["pen_25"]
    assert r["decision"] == "BUY_YES" and r["exit_rule"] == "exit_at_offset"
    assert r["exit_status"] == "exited" and r["exit_observation_id"] == "SYN-NYY@T-10"
    assert (r["exit_price"], r["exit_quantity"], r["exit_fee"], r["held_quantity"]) == pytest.approx((0.635, 18, 0.30, 0))
    assert (r["net_pnl"], r["gross_pnl"]) == pytest.approx((0.47, 1.26))
    assert set(r) <= set(BACKTEST_DECISIONS.column_names)
    held = by_scenario(decide(entry_row(), POL))["pen_25"]
    assert held["exit_rule"] == "hold_to_settlement" and held["exit_status"] == "held"
    assert held["exit_reason"] is None and held["net_pnl"] == pytest.approx(7.34)


def test_each_scenario_exits_at_its_own_penalty():
    rows = by_scenario(decide(entry_row(), POL_EXIT, exit_row()))
    for name, k in (("pen_0", 0.0), ("pen_50", 0.5), ("pen_100", 1.0)):
        assert rows[name]["exit_price"] == pytest.approx(0.64 - k * 0.02)
        assert rows[name]["execution_price"] == pytest.approx(0.57 + k * 0.02)


def test_engine_missing_exit_quote_is_held_and_says_so():
    r = by_scenario(decide(entry_row(), POL_EXIT, None))["pen_25"]
    assert r["exit_status"] == "held" and r["exit_reason"] == "no_valid_exit_quote"
    assert r["exit_snapshot_time"] == EXIT_T and r["net_pnl"] == pytest.approx(7.34)


def test_exit_quote_never_reaches_the_entry_decision():
    """Leakage guard: a T-10 quote exists at T-30 decision time only in hindsight."""
    base = by_scenario(decide(entry_row(), POL))
    for q in (exit_row(market_yes_bid=0.97, market_yes_ask=0.98), exit_row(market_yes_bid=0.02, market_yes_ask=0.03)):
        moved = by_scenario(decide(entry_row(), POL_EXIT, q))
        for s in base:
            assert {k: moved[s].get(k) for k in ENTRY_KEYS} == {k: base[s].get(k) for k in ENTRY_KEYS}


def test_unsettled_exit_has_pnl_but_unsettled_hold_does_not():
    r = by_scenario(decide(entry_row(outcome=None), POL_EXIT, exit_row()))["pen_25"]
    assert r["net_pnl"] == pytest.approx(0.47)
    assert by_scenario(decide(entry_row(outcome=None), POL))["pen_25"]["net_pnl"] is None


def test_trading_summary_counts_both_legs():
    rows = [r for r in decide(entry_row(), POL_EXIT, exit_row()) if r["scenario"] == "pen_25"]
    s = trading_summary(pd.DataFrame(rows).reindex(columns=BACKTEST_DECISIONS.column_names))
    assert s["fees"] == pytest.approx(0.61) and s["slippage"] == pytest.approx(0.18)
    assert s["gross_pnl"] - s["fees"] - s["slippage"] == pytest.approx(s["net_pnl"]) == pytest.approx(0.47)
    assert s["exit_statuses"] == {"exited": 1} and s["exit_fallbacks"] == {}


# -- through the database -------------------------------------------------------

WHEN = datetime(2026, 2, 1, tzinfo=timezone.utc)


def seed(syn_db, *, with_exit_quote: bool) -> None:
    rows = [observation(outcome=0)]
    if with_exit_quote:
        rows.append(observation(offset=10, market_yes_bid=0.64, market_yes_ask=0.66, spread=0.02, bid_size=400.0,
                                ask_size=300.0, p_mid=0.65, market_last=0.65, outcome=0))
    syn_db.insert("observations", frame(*rows))
    syn_db.insert("model_runs", frame({
        "model_run_id": "mr-1", "model_type": "logistic_residual", "period_id": "2025H1", "feature_set_id": "fs",
        "feature_set_hash": "fsh", "config_hash": "c", "predict_from": pd.Timestamp("2025-01-01T00:00Z"),
        "dataset": "synthetic", "created_at": WHEN, "is_synthetic": True}))
    syn_db.insert("predictions", frame({
        "model_run_id": "mr-1", "observation_id": "SYN-NYY@T-30", "period_id": "2025H1", "p_market": 0.56,
        "p_model": 0.70, "lower_bound": 0.66, "upper_bound": 0.74, "p_conservative_yes": 0.66,
        "p_conservative_no": 0.26, "is_synthetic": True}))


def exit_cfg():
    cfg = load_sport_config("mlb")
    cfg["execution"] = {**cfg["execution"], "exit_rule": "exit_at_offset", "exit_offset_min": 10}
    return cfg


def test_config_edits_do_not_leak_to_later_loads():
    """exit_cfg() once edited the cached config, so every later test silently backtested with exit_at_offset."""
    exit_cfg()["execution"]["exit_offset_min"] = 99
    assert load_sport_config("mlb")["execution"]["exit_rule"] == "hold_to_settlement"
    assert ExitRule.from_config(load_sport_config("mlb")) == ExitRule()


@pytest.mark.parametrize("with_exit_quote", [True, False])
def test_backtest_records_the_round_trip(syn_db, with_exit_quote):
    seed(syn_db, with_exit_quote=with_exit_quote)
    run_backtest(syn_db, exit_cfg(), {"selected": "logistic_residual"}, now=WHEN)
    d = syn_db.df("SELECT * FROM backtest_decisions WHERE scenario = 'pen_25' AND observation_id = 'SYN-NYY@T-30'")
    r = d.iloc[0]
    assert r["decision"] == "BUY_YES" and r["exit_rule"] == "exit_at_offset" and r["outcome"] == 0
    if with_exit_quote:
        assert r["exit_status"] == "exited" and r["exit_observation_id"] == "SYN-NYY@T-10"
        assert (r["net_pnl"], r["exit_fee"], r["calculated_fee"]) == pytest.approx((0.47, 0.30, 0.31))
    else:
        assert r["exit_status"] == "held" and r["exit_reason"] == "no_valid_exit_quote"
        assert r["net_pnl"] == pytest.approx(-10.66)  # held and lost
    runs = syn_db.df("SELECT config_json FROM backtest_runs WHERE snapshot_offset_min = 30")
    assert json.loads(runs["config_json"][0])["execution"]["exit_rule"] == "exit_at_offset"


def test_t10_entry_cannot_exit_at_t10(syn_db):
    seed(syn_db, with_exit_quote=True)
    syn_db.insert("predictions", frame({
        "model_run_id": "mr-1", "observation_id": "SYN-NYY@T-10", "period_id": "2025H1", "p_market": 0.65,
        "p_model": 0.80, "lower_bound": 0.76, "upper_bound": 0.84, "p_conservative_yes": 0.76,
        "p_conservative_no": 0.16, "is_synthetic": True}))
    run_backtest(syn_db, exit_cfg(), {"selected": "logistic_residual"}, now=WHEN)
    r = syn_db.df("SELECT * FROM backtest_decisions WHERE scenario = 'pen_25' AND observation_id = 'SYN-NYY@T-10'")
    assert r["decision"][0] == "BUY_YES"
    assert r["exit_status"][0] == "held" and r["exit_reason"][0] == "exit_not_after_entry"


# -- database invariants: each rule rejects a bad row and accepts the good one ---

def trade(**over) -> dict:
    row = {"backtest_run_id": "b1", "scenario": "pen_25", "observation_id": "o1", "game_id": "g1",
           "contract_id": "c1", "period_id": "p1", "snapshot_time": ENTRY_T, "decision": "BUY_YES", "side": "yes",
           "p_market": 0.56, "p_model": 0.70, "quoted_price": 0.57, "spread_penalty_frac": 0.25,
           "execution_price": 0.575, "quantity": 18.0, "fee_schedule_id": "quadratic-taker-v1",
           "fee_schedule_effective_date": ENTRY_T.date(), "maker_or_taker": "taker", "calculated_fee": 0.31,
           "exit_rule": "exit_at_offset", "exit_status": "exited", "exit_observation_id": "o1@T-10",
           "exit_snapshot_time": EXIT_T, "exit_quoted_price": 0.64, "exit_price": 0.635, "exit_quantity": 18.0,
           "held_quantity": 0.0, "exit_proceeds": 11.43, "exit_fee_schedule_id": "quadratic-taker-v1",
           "exit_fee_schedule_effective_date": EXIT_T.date(), "exit_fee": 0.30, "exit_slippage_cost": 0.09,
           "net_pnl": 0.47, "is_synthetic": True}
    return row | over


HELD = dict(exit_status="held", exit_reason="no_valid_exit_quote", exit_observation_id=None,
            exit_quoted_price=None, exit_price=None,
            exit_quantity=None, held_quantity=18.0, exit_proceeds=None, exit_fee_schedule_id=None,
            exit_fee_schedule_effective_date=None, exit_fee=None, exit_slippage_cost=None)


@pytest.mark.parametrize("bad, good, rule", [
    (trade(exit_fee=None), trade(), "exit_fee IS NOT NULL"),                            # the second fee is required
    (trade(exit_price=0.645), trade(), "exit_price <= exit_quoted_price"),              # never sells above the bid
    (trade(exit_snapshot_time=ENTRY_T), trade(), "exit_snapshot_time > snapshot_time"),  # exit after entry
    (trade(exit_snapshot_time=None), trade(), "exit_snapshot_time IS NOT NULL"),
    (trade(held_quantity=1.0), trade(), "abs(held_quantity"),                           # quantities add up
    (trade(held_quantity=None), trade(), "held_quantity IS NOT NULL"),                  # NULL can't slip through
    (trade(**HELD | {"exit_reason": None}), trade(**HELD), "OR exit_reason IS NOT NULL"),     # fallback never hidden
    (trade(**HELD | {"exit_fee": 0.30}), trade(**HELD), "exit_fee IS NULL"),                  # held pays no exit fee
    (trade(exit_rule="hold_to_settlement"), trade(**HELD | {"exit_reason": None, "exit_snapshot_time": None},
                                                   exit_rule="hold_to_settlement"), "exit_rule <> 'hold_to_settlement'"),
    (trade(exit_status=None), trade(), "(decision = 'PASS') = (exit_status IS NULL)"),
    (trade(exit_status="partial", exit_reason="insufficient_exit_depth"),              # partial but all sold
     trade(exit_status="partial", exit_reason="insufficient_exit_depth", exit_quantity=10.0, held_quantity=8.0,
           exit_proceeds=6.35), "(exit_status = 'exited') = (held_quantity = 0)"),
])
def test_exit_invariants(syn_db, bad, good, rule):
    with pytest.raises(ConstraintViolation) as exc:
        syn_db.insert("backtest_decisions", frame(bad))
    assert rule in exc.value.check, exc.value.check
    syn_db.insert("backtest_decisions", frame(good))
    assert syn_db.count("backtest_decisions") == 1
