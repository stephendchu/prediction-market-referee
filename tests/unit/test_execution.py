"""Fee schedules (26.8), executable-price fills and book walks (26.9), settlement P&L."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.backtest.execution import (
    FEE_SCHEDULES,
    FeeSchedule,
    Fill,
    NoFeeSchedule,
    NoFill,
    QuadraticFeeSchedule,
    Quote,
    depth_levels,
    estimated_net_edge,
    fee_schedule_for,
    fee_schedule_rows,
    net_edge,
    quoted_price,
    scenario_name,
    settle,
    side_won,
    simulate_fill,
    spread,
    top_size,
)
from src.storage.schemas import FEE_SCHEDULES as FEE_TABLE

WHEN = datetime(2025, 6, 1, 22, 35, tzinfo=timezone.utc)
Q = Quote(yes_bid=0.55, yes_ask=0.57, bid_size=250.0, ask_size=180.0)  # factories.observation's quote
BOOK = json.dumps({"yes_bids": [[0.55, 50], [0.54, 100]], "yes_asks": [[0.57, 40], [0.58, 60], [0.60, 500]]})


def fill(side="yes", quote=Q, penalty=0.25, stake=100.0, **kw):
    return simulate_fill(side, quote, penalty_frac=penalty, stake_dollars=stake, max_fraction_top=0.10,
                         min_contracts=10, when=WHEN, **kw)


def two_versions(**v2):
    v1 = QuadraticFeeSchedule(fee_schedule_id="t-v1", maker_or_taker="taker", effective_from=date(2021, 1, 1),
                              effective_to=date(2024, 12, 31), params={"rate": 0.07})
    new = dict(fee_schedule_id="t-v2", maker_or_taker="taker", effective_from=date(2025, 1, 1),
               params={"rate": 0.05}) | v2
    return (v1, QuadraticFeeSchedule(**new))


# -- fee schedules ------------------------------------------------------------

def test_taker_fee_rounds_up_to_the_cent():
    taker = fee_schedule_for(WHEN)
    assert taker.fee_schedule_id == "quadratic-taker-v1"
    assert taker.fee(100, 0.57) == 1.72  # 0.07 * 100 * 0.57 * 0.43 = 1.7157
    assert taker.fee(1, 0.50) == 0.02    # 0.0175 -> 2 cents
    assert taker.fee(0, 0.50) == 0.0


def test_fee_on_an_exact_cent_is_not_bumped_by_float_noise():
    assert 0.07 * 100 * 0.5 * 0.5 * 100 > 175  # 175.00000000000003 in floating point
    assert fee_schedule_for(WHEN).fee(100, 0.50) == 1.75


def test_maker_fee_uses_the_maker_schedule():
    maker = fee_schedule_for(WHEN, "maker")
    assert maker.fee_schedule_id == "quadratic-maker-v1"
    assert maker.fee(100, 0.57) == 0.43  # 0.0175 * 100 * 0.2451 = 0.4289


def test_shipped_schedules_are_marked_unverified():
    assert all(not s.verified and "unverified" in s.source_note for s in FEE_SCHEDULES)


def test_dated_versions_resolve_by_date():
    reg = two_versions()
    assert fee_schedule_for(date(2024, 12, 31), registry=reg).fee_schedule_id == "t-v1"
    assert fee_schedule_for(date(2025, 1, 1), registry=reg).fee_schedule_id == "t-v2"
    # 22:00 New York on Dec 31 is Jan 1 UTC: the UTC date decides
    ny = datetime(2024, 12, 31, 22, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert fee_schedule_for(ny, registry=reg).fee_schedule_id == "t-v2"
    assert fee_schedule_for(pd.Timestamp("2024-06-01T23:00Z"), registry=reg).fee_schedule_id == "t-v1"


def test_fills_are_charged_under_the_schedule_in_force_at_the_trade():
    reg = two_versions()
    old = simulate_fill("yes", Q, penalty_frac=0.0, stake_dollars=57.0, max_fraction_top=1.0, min_contracts=1,
                        when=date(2024, 6, 1), registry=reg)
    new = simulate_fill("yes", Q, penalty_frac=0.0, stake_dollars=57.0, max_fraction_top=1.0, min_contracts=1,
                        when=date(2025, 6, 1), registry=reg)
    assert old.quantity == new.quantity == 100
    assert (old.fee_schedule.fee_schedule_id, old.fee) == ("t-v1", 1.72)
    assert (new.fee_schedule.fee_schedule_id, new.fee) == ("t-v2", 1.23)  # 0.05 * 24.51 = 1.2255


def test_missing_or_ambiguous_schedule_raises():
    with pytest.raises(NoFeeSchedule, match="no taker"):
        fee_schedule_for(date(2020, 12, 31))
    overlapping = two_versions(effective_from=date(2024, 12, 1))
    with pytest.raises(NoFeeSchedule, match="ambiguous"):
        fee_schedule_for(date(2024, 12, 15), registry=overlapping)
    assert fee_schedule_for(date(2024, 11, 30), registry=overlapping).fee_schedule_id == "t-v1"


def test_fee_schedule_is_an_interface_with_validated_versions():
    with pytest.raises(TypeError):
        FeeSchedule(fee_schedule_id="x", maker_or_taker="taker", effective_from=date(2021, 1, 1), formula="f")
    with pytest.raises(ValueError, match="maker_or_taker"):
        QuadraticFeeSchedule(fee_schedule_id="x", maker_or_taker="both", effective_from=date(2021, 1, 1),
                             params={"rate": 0.07})
    with pytest.raises(ValueError, match="effective_to"):
        two_versions(effective_to=date(2024, 1, 1))
    with pytest.raises(TypeError):
        FEE_SCHEDULES[0].params["rate"] = 0.0  # the shipped registry can't be mutated in place


def test_fee_schedule_rows_load_into_the_table(syn_db):
    rows = fee_schedule_rows()
    assert list(rows.columns) == FEE_TABLE.column_names
    assert syn_db.insert("fee_schedules", rows) == len(FEE_SCHEDULES)
    stored = syn_db.df("SELECT * FROM fee_schedules ORDER BY fee_schedule_id")
    assert list(stored["fee_schedule_id"]) == ["quadratic-maker-v1", "quadratic-taker-v1"]
    assert json.loads(stored.iloc[1]["params_json"]) == {"rate": 0.07}
    assert not stored["verified"].any()


# -- quotes and fills -----------------------------------------------------------

def test_quoted_prices_are_executable_asks_never_the_midpoint():
    assert quoted_price("yes", Q) == 0.57
    assert quoted_price("no", Q) == pytest.approx(0.45)  # no_ask = 1 - yes_bid
    assert spread(Q) == pytest.approx(0.02)
    assert (top_size("yes", Q), top_size("no", Q)) == (180.0, 250.0)  # YES eats asks, NO eats bids


def test_yes_fill_at_baseline_penalty_capped_by_top_of_book():
    f = fill("yes")
    assert isinstance(f, Fill)
    assert f.execution_price == pytest.approx(0.575)  # 0.57 + 0.25 * 0.02
    assert f.quantity == 18                            # min(100 / 0.575, 10% of 180)
    assert f.stake == pytest.approx(10.35)
    assert f.slippage_cost == pytest.approx(0.09)
    assert f.fee == 0.31                               # 0.07 * 18 * 0.575 * 0.425 = 0.3079
    assert f.depth_verified
    rec = f.record()
    assert (rec["fee_schedule_id"], rec["maker_or_taker"], rec["fee_schedule_effective_date"]) == (
        "quadratic-taker-v1", "taker", date(2021, 1, 1))
    assert rec["calculated_fee"] == 0.31


def test_no_fill_buys_at_one_minus_the_yes_bid():
    f = fill("no")
    assert f.quoted_price == pytest.approx(0.45)
    assert f.execution_price == pytest.approx(0.455)
    assert f.quantity == 25  # 10% of the 250 YES bids


@pytest.mark.parametrize("penalty,price", [(0.0, 0.57), (0.25, 0.575), (0.5, 0.58), (1.0, 0.59)])
def test_spread_penalty_scenarios(penalty, price):
    f = fill("yes", penalty=penalty)
    assert f.execution_price == pytest.approx(price)
    assert f.execution_price >= f.quoted_price
    assert scenario_name(penalty) == f"pen_{int(penalty * 100)}"


def test_depth_walk_gives_volume_weighted_price():
    f = fill("yes", Quote(0.55, 0.57, 50.0, 40.0, BOOK), penalty=0.0)
    assert f.quantity == 175                              # floor(100 / 0.57), not the 10% top-of-book cap
    assert f.execution_price == pytest.approx(102.6 / 175)  # 40 @ .57 + 60 @ .58 + 75 @ .60
    assert f.slippage_cost == pytest.approx(2.85)          # 102.6 - 175 * 0.57
    assert f.depth_verified


def test_depth_walk_for_no_uses_the_yes_bids():
    levels = depth_levels("no", Quote(0.55, 0.57, depth_json=BOOK))
    assert levels == [pytest.approx((0.45, 50.0)), pytest.approx((0.46, 100.0))]
    f = fill("no", Quote(0.55, 0.57, 50.0, 40.0, BOOK), penalty=0.0, stake=30.0)
    assert f.quantity == 66                                   # floor(30 / 0.45)
    assert f.execution_price == pytest.approx((50 * 0.45 + 16 * 0.46) / 66)


def test_depth_walk_never_beats_the_penalized_price():
    book = json.dumps({"yes_bids": [[0.55, 500]], "yes_asks": [[0.57, 500]]})
    f = fill("yes", Quote(0.55, 0.57, depth_json=book), penalty=0.25)
    assert f.execution_price == pytest.approx(0.575)  # VWAP 0.57 < base 0.575


def test_depth_caps_quantity_at_displayed_contracts():
    thin = json.dumps({"yes_bids": [[0.55, 5]], "yes_asks": [[0.57, 12]]})
    f = fill("yes", Quote(0.55, 0.57, depth_json=thin), penalty=0.0)
    assert (f.quantity, f.execution_price) == (12, pytest.approx(0.57))
    nf = fill("no", Quote(0.55, 0.57, depth_json=thin), penalty=0.0)
    assert nf == NoFill("no", "insufficient_depth")  # 5 displayed < 10 minimum


def test_thin_top_of_book_is_insufficient_depth():
    assert fill("yes", Quote(0.55, 0.57, 250.0, 50.0)) == NoFill("yes", "insufficient_depth")  # 10% of 50 = 5


def test_candles_without_size_fill_the_stake_unverified():
    f = fill("yes", Quote(0.55, 0.57))
    assert f.quantity == 173  # floor(100 / 0.575)
    assert not f.depth_verified
    nan_sizes = Quote.from_row({"market_yes_bid": 0.55, "market_yes_ask": 0.57, "bid_size": math.nan,
                                "ask_size": math.nan, "depth_json": None})
    assert fill("yes", nan_sizes) == f  # NaN size means unknown, not zero


def test_quote_from_row_accepts_observation_and_market_quote_columns():
    obs = Quote.from_row(pd.Series({"market_yes_bid": 0.55, "market_yes_ask": 0.57, "bid_size": 250,
                                    "ask_size": 180, "market_last": 0.61}))
    mq = Quote.from_row({"yes_bid": 0.55, "yes_ask": 0.57, "bid_size": 250, "ask_size": 180})
    assert obs == mq == Q


@pytest.mark.parametrize("quote,side", [
    (Quote(0.0, 0.57, 10, 10), "yes"),      # one-sided
    (Quote(0.57, 0.57, 10, 10), "yes"),     # locked
    (Quote(0.58, 0.57, 10, 10), "no"),      # crossed
    (Quote(math.nan, 0.57, 10, 10), "no"),  # missing bid
    (Quote(0.97, 0.99, 900, 900), "yes"),   # ask at 0.99
    (Quote(0.94, 0.98, 900, 900), "yes"),   # 0.98 + 0.25 * 0.04 = 0.99
    (Quote(0.005, 0.02, 900, 900), "no"),   # no_ask = 0.995
])
def test_unpriceable_quotes_are_no_price(quote, side):
    assert fill(side, quote) == NoFill(side, "no_price")
    assert estimated_net_edge(0.5, quote, side, 0.25, 100.0, WHEN) is None


# -- settlement and edges -------------------------------------------------------

def test_settlement_matches_backtest_decisions_semantics():
    f = fill("yes")  # 18 @ quoted 0.57, executed 0.575, fee 0.31
    win, loss = settle(f, 1), settle(f, 0)
    assert win["gross_pnl"] == pytest.approx(7.74)     # 18 * (1 - 0.57)
    assert win["net_pnl"] == pytest.approx(7.34)       # 7.74 - 0.09 slippage - 0.31 fee
    assert loss["gross_pnl"] == pytest.approx(-10.26)  # 18 * (0 - 0.57)
    assert loss["net_pnl"] == pytest.approx(-10.66)
    assert win["net_pnl"] == pytest.approx(f.quantity * (1 - f.execution_price) - f.fee)
    with pytest.raises(ValueError):
        settle(f, 2)


def test_side_won_flips_for_no():
    assert (side_won("yes", 1), side_won("yes", 0), side_won("no", 1), side_won("no", 0)) == (1, 0, 0, 1)


def test_net_edge_per_contract():
    assert net_edge(0.62, 0.575, 0.31, 18) == pytest.approx(0.62 - 0.575 - 0.31 / 18)
    with pytest.raises(ValueError):
        net_edge(0.62, 0.575, 0.31, 0)


def test_estimated_net_edge_ignores_the_size_cap():
    # 173 contracts at 0.575; fee = ceil(0.07 * 173 * 0.575 * 0.425 = 2.9594) = 2.96
    assert estimated_net_edge(0.62, Q, "yes", 0.25, 100.0, WHEN) == pytest.approx(0.62 - 0.575 - 2.96 / 173)
    # NO: p_conservative_no = 0.50 vs 0.455; 219 contracts, fee ceil(0.07 * 219 * 0.455 * 0.545) = 3.81
    assert estimated_net_edge(0.50, Q, "no", 0.25, 100.0, WHEN) == pytest.approx(0.50 - 0.455 - 3.81 / 219)
    assert estimated_net_edge(0.62, Q, "yes", 0.25, 0.5, WHEN) is None  # stake buys no contract
