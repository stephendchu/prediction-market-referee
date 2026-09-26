"""Stored fills rebuilt from their records: Fill.from_record, ExitFill.from_record,
fee_schedule_by_id. A logged paper leg is settled only if its stored fee, cost and slippage are
exactly what its named, dated fee schedule produces at the time of the trade.

Hand numbers (taker fee = 0.07 x C x P x (1 - P), rounded up to the next cent):
  YES entry  ask 0.57 + 0.25 x 0.02 = 0.575, min(100 / 0.575, 0.10 x 180) -> 18 contracts,
             stake 10.35, fee ceil(30.79c) = 0.31, slippage 18 x 0.005 = 0.09
  NO entry   1 - bid 0.55 + 0.005 = 0.455, min(100 / 0.455, 0.10 x 250) -> 25 contracts,
             stake 11.375, fee ceil(43.39c) = 0.44
  YES exit   bid 0.64 - 0.005 = 0.635 for 18, proceeds 11.43, fee ceil(29.20c) = 0.30
  NO exit    1 - ask 0.66 - 0.005 = 0.335 for min(25, 0.10 x 300) = 25, proceeds 8.375, fee ceil(38.99c) = 0.39
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.backtest.execution import (FEE_SCHEDULES, ExitFill, ExitRule, Fill, NoFeeSchedule, Position, Quote,
                                    QuadraticFeeSchedule, fee_schedule_by_id, simulate_fill)

ENTRY_T = datetime(2025, 6, 1, 22, 35, tzinfo=timezone.utc)   # T-30 of a 23:05 UTC game
EXIT_T = ENTRY_T + timedelta(minutes=20)                        # T-10
ENTRY_QUOTE = Quote(0.55, 0.57, 250.0, 180.0)
EXIT_QUOTE = Quote(0.64, 0.66, 400.0, 300.0)
TAKER, MAKER = FEE_SCHEDULES


def entry(side="yes", when=ENTRY_T, registry=FEE_SCHEDULES):
    return simulate_fill(side, ENTRY_QUOTE, penalty_frac=0.25, stake_dollars=100.0, max_fraction_top=0.10,
                         min_contracts=10, when=when, registry=registry)


def closed(side="yes"):
    pos = Position(entry(side), ENTRY_T, ExitRule("exit_at_offset", 10))
    return pos.close(EXIT_QUOTE, EXIT_T, ref="SYN-NYY@T-10", penalty_frac=0.25, max_fraction_top=0.10)


def exit_record(side="yes", **over):
    return {k: v for k, v in closed(side).record(None).items()} | over


def rebuild_exit(rec, side="yes", **kw):
    return ExitFill.from_record(rec, side=side, penalty_frac=0.25, when=kw.pop("when", EXIT_T), **kw)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side, qty, price, stake, fee", [("yes", 18.0, 0.575, 10.35, 0.31),
                                                           ("no", 25.0, 0.455, 11.375, 0.44)])
def test_entry_fill_round_trips_exactly(side, qty, price, stake, fee):
    """Fill.from_record(fill.record()) is the same fill, and records the same row."""
    f = entry(side)
    assert (f.quantity, f.execution_price, f.fee) == (qty, pytest.approx(price, abs=1e-12), fee)
    assert f.stake == pytest.approx(stake, abs=1e-9)
    rec = f.record()
    back = Fill.from_record(rec, when=ENTRY_T)
    assert back == f and back.record() == rec
    assert back.fee_schedule is TAKER


def test_entry_fill_round_trips_through_a_dataframe_row():
    """A record read back from a table (numpy scalars, a Timestamp effective date) rebuilds the same fill."""
    f = entry()
    row = pd.DataFrame([f.record() | {"fee_schedule_effective_date": pd.Timestamp("2021-01-01")}]).iloc[0].to_dict()
    assert Fill.from_record(row, when=pd.Timestamp(ENTRY_T)) == f


@pytest.mark.parametrize("side, qty, price, proceeds, fee", [("yes", 18.0, 0.635, 11.43, 0.30),
                                                              ("no", 25.0, 0.335, 8.375, 0.39)])
def test_exit_fill_round_trips_exactly(side, qty, price, proceeds, fee):
    """ExitFill.from_record(Position.record()) is the same sale."""
    pos = closed(side)
    x = pos.exit
    assert (pos.exit_status, x.quantity, x.fee) == ("exited", qty, fee)
    assert x.execution_price == pytest.approx(price, abs=1e-12) and x.proceeds == pytest.approx(proceeds, abs=1e-9)
    back = rebuild_exit(pos.record(None), side=side)
    assert back == x and back.fee_schedule is TAKER


def test_exit_record_naming_its_side_round_trips():
    """paper_exits rows carry the side sold; a matching side is accepted."""
    assert rebuild_exit(exit_record(side="yes") | {"side": "yes"}) == closed().exit


def test_fee_schedule_by_id_returns_the_registry_schedule():
    """A stored id resolves to the registry's own object (not a copy)."""
    assert fee_schedule_by_id("quadratic-taker-v1") is TAKER
    assert fee_schedule_by_id("quadratic-maker-v1") is MAKER


# ---------------------------------------------------------------------------
# A stored entry that does not match its schedule is refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fee_schedule_id", ["quadratic-taker-v2", "", "QUADRATIC-TAKER-V1"])
def test_unknown_fee_schedule_id_is_refused(fee_schedule_id):
    """NoFeeSchedule (a LookupError) when the stored id names no registry schedule."""
    rec = entry().record() | {"fee_schedule_id": fee_schedule_id}
    with pytest.raises(NoFeeSchedule, match="0 fee schedules with id"):
        Fill.from_record(rec, when=ENTRY_T)
    assert issubclass(NoFeeSchedule, LookupError)


def test_ambiguous_fee_schedule_id_is_refused():
    """Two registry schedules with the stored id: NoFeeSchedule, not the first match."""
    with pytest.raises(NoFeeSchedule, match="2 fee schedules with id 'quadratic-taker-v1'"):
        Fill.from_record(entry().record(), when=ENTRY_T, registry=(TAKER, TAKER))


def test_maker_schedule_under_a_taker_label_is_refused():
    """The stored role must be the schedule's role."""
    rec = entry().record() | {"fee_schedule_id": "quadratic-maker-v1"}
    with pytest.raises(ValueError, match="quadratic-maker-v1 is a maker schedule, not taker"):
        Fill.from_record(rec, when=ENTRY_T)


def test_consistent_relabel_to_maker_is_refused_by_the_fee():
    """Relabelling a taker fill as maker (id and role) fails on the fee: maker charges
    ceil(0.0175 x 18 x 0.575 x 0.425 = 7.70c) = 0.08, not the stored 0.31."""
    rec = entry().record() | {"fee_schedule_id": "quadratic-maker-v1", "maker_or_taker": "maker"}
    with pytest.raises(ValueError, match=r"stored calculated_fee 0\.31 disagrees with 0\.08"):
        Fill.from_record(rec, when=ENTRY_T)


@pytest.mark.parametrize("fee", ["drop", None, float("nan"), 0.32, 0.30, 0.31 + 2e-9])
def test_dropped_or_altered_entry_fee_is_refused(fee):
    """calculated_fee must be present and equal the schedule's 0.31 (tolerance 1e-9)."""
    rec = entry().record()
    if fee == "drop":
        del rec["calculated_fee"]
    else:
        rec["calculated_fee"] = fee
    with pytest.raises(ValueError, match="stored calculated_fee .* recomputed under quadratic-taker-v1"):
        Fill.from_record(rec, when=ENTRY_T)


def test_fee_within_float_noise_is_accepted():
    """Control for the test above: 0.31 + 5e-10 is float noise, not a different fee."""
    assert Fill.from_record(entry().record() | {"calculated_fee": 0.31 + 5e-10}, when=ENTRY_T) == entry()


@pytest.mark.parametrize("key, value", [("stake", 10.36), ("stake", None), ("slippage_cost", 0.0),
                                        ("slippage_cost", 0.1)])
def test_altered_stake_or_slippage_is_refused(key, value):
    """stake = quantity x price and slippage = quantity x (price - quoted) must match the record."""
    with pytest.raises(ValueError, match=f"stored {key} "):
        Fill.from_record(entry().record() | {key: value}, when=ENTRY_T)


@pytest.mark.parametrize("eff", [date(2021, 1, 2), date(2020, 12, 31), None, pd.NaT, "drop"])
def test_mismatched_or_missing_effective_date_is_refused(eff):
    """The stored effective date must be the named schedule's effective_from (2021-01-01)."""
    rec = entry().record()
    if eff == "drop":
        del rec["fee_schedule_effective_date"]
    else:
        rec["fee_schedule_effective_date"] = eff
    with pytest.raises(ValueError, match="quadratic-taker-v1: stored effective date .* is not 2021-01-01"):
        Fill.from_record(rec, when=ENTRY_T)


def test_schedule_not_in_force_at_the_trade_time_is_refused():
    """A record naming a schedule that had been replaced by its trade date is refused; the same
    record at a date inside its window is accepted (a registry with a dated rate change)."""
    old = QuadraticFeeSchedule(fee_schedule_id="taker-old", maker_or_taker="taker", effective_from=date(2021, 1, 1),
                               effective_to=date(2024, 12, 31), params={"rate": 0.07})
    new = QuadraticFeeSchedule(fee_schedule_id="taker-new", maker_or_taker="taker", effective_from=date(2025, 1, 1),
                               params={"rate": 0.07})
    registry = (old, new, MAKER)
    in_window = datetime(2024, 6, 1, 22, 35, tzinfo=timezone.utc)
    rec = entry(when=in_window, registry=registry).record()
    assert rec["fee_schedule_id"] == "taker-old"
    assert Fill.from_record(rec, when=in_window, registry=registry).fee_schedule is old
    with pytest.raises(ValueError, match="taker-old was not in force on 2025-06-01"):
        Fill.from_record(rec, when=ENTRY_T, registry=registry)


def test_trade_time_before_any_schedule_is_refused():
    """No schedule in force at all (before 2021): NoFeeSchedule."""
    with pytest.raises(NoFeeSchedule, match="no taker fee schedule for 2020-06-01"):
        Fill.from_record(entry().record(), when=datetime(2020, 6, 1, tzinfo=timezone.utc))


def test_trade_time_is_read_as_a_utc_date():
    """`when` is converted to UTC before its date is taken: 2021-01-01 00:30 UTC is 2020-12-31 in
    New York, still inside the schedule; 2020-12-31 23:30 UTC is not."""
    ny = timezone(timedelta(hours=-5))
    assert Fill.from_record(entry().record(), when=datetime(2020, 12, 31, 19, 30, tzinfo=ny)) == entry()
    with pytest.raises(NoFeeSchedule):
        Fill.from_record(entry().record(), when=datetime(2020, 12, 31, 18, 30, tzinfo=ny))


@pytest.mark.parametrize("over, message", [({"quantity": 0.0}, "a fill needs quantity > 0"),
                                           ({"quantity": -18.0}, "a fill needs quantity > 0"),
                                           ({"side": "maybe"}, "side must be yes|no")])
def test_impossible_fill_is_refused(over, message):
    with pytest.raises(ValueError, match=message):
        Fill.from_record(entry().record() | over, when=ENTRY_T)


# ---------------------------------------------------------------------------
# A stored exit that does not match its schedule is refused
# ---------------------------------------------------------------------------

def test_exit_with_an_unknown_fee_schedule_id_is_refused():
    with pytest.raises(NoFeeSchedule, match="0 fee schedules with id 'quadratic-taker-v0'"):
        rebuild_exit(exit_record(exit_fee_schedule_id="quadratic-taker-v0"))


@pytest.mark.parametrize("key, value", [("exit_fee", 0.31), ("exit_fee", None), ("exit_fee", "drop"),
                                        ("exit_proceeds", 11.52), ("exit_proceeds", None),
                                        ("exit_slippage_cost", 0.0)])
def test_dropped_or_altered_exit_amounts_are_refused(key, value):
    """exit_fee (0.30), exit_proceeds (11.43) and exit_slippage_cost (0.09) must match the schedule."""
    rec = exit_record()
    if value == "drop":
        del rec[key]
    else:
        rec[key] = value
    with pytest.raises(ValueError, match=f"stored {key} "):
        rebuild_exit(rec)


@pytest.mark.parametrize("eff", [date(2021, 1, 2), None])
def test_exit_effective_date_mismatch_is_refused(eff):
    with pytest.raises(ValueError, match="stored effective date .* is not 2021-01-01"):
        rebuild_exit(exit_record(exit_fee_schedule_effective_date=eff))


def test_exit_record_of_the_other_side_is_refused():
    """A sale recorded for YES cannot close a NO position."""
    with pytest.raises(ValueError, match="exit record sold 'yes', position holds 'no'"):
        rebuild_exit(exit_record(side="yes") | {"side": "yes"}, side="no")


def test_exit_under_the_wrong_liquidity_role_is_refused():
    """The position's role (maker) must be the stored schedule's (taker)."""
    with pytest.raises(ValueError, match="quadratic-taker-v1 is a taker schedule, not maker"):
        rebuild_exit(exit_record(), maker_or_taker="maker")


def test_exit_schedule_not_in_force_is_refused():
    with pytest.raises(NoFeeSchedule, match="no taker fee schedule for 2020-06-01"):
        rebuild_exit(exit_record(), when=datetime(2020, 6, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize("over, message", [({"exit_quantity": 0.0}, "a sale needs quantity > 0"),
                                           ({"exit_quantity": -18.0}, "a sale needs quantity > 0"),
                                           ({"side": "maybe"}, "exit record sold 'maybe'")])
def test_impossible_exit_is_refused(over, message):
    """No contracts sold is not a sale (a held position has no exit fill to rebuild)."""
    with pytest.raises(ValueError, match=message):
        rebuild_exit(exit_record() | over)
