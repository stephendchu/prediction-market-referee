"""Database-level leakage guards.

Each test inserts a row that uses information from after the snapshot and
asserts the database rejects it, then inserts the same row without the leak and
asserts it is accepted. So a test fails if the guard is missing, and also if the
guard rejects everything.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from factories import (START, feature_definition, feature_value, frame, observation, paper_exit, paper_settlement,
                       paper_trade, provenance)

from src.storage.database import ConstraintViolation

SNAP = START - timedelta(minutes=30)


def rejected_then_accepted(db, table, bad, good, rule):
    with pytest.raises(ConstraintViolation) as exc:
        db.insert(table, frame(bad))
    assert rule in exc.value.check, exc.value.check
    before = db.count(table)
    db.insert(table, frame(good))
    assert db.count(table) == before + 1


def test_future_feature_value_is_rejected(syn_db):
    syn_db.insert("feature_definitions", frame(feature_definition()))
    rejected_then_accepted(
        syn_db, "feature_values",
        bad=feature_value(SNAP, available_at=SNAP + timedelta(seconds=1)),
        good=feature_value(SNAP, available_at=SNAP),
        rule="available_at <= cutoff_time",
    )


def test_feature_published_between_quote_and_snapshot_is_rejected(syn_db):
    # 26.11: the cutoff is the quote time, not the snapshot time.
    quote_ts = SNAP - timedelta(minutes=3)
    syn_db.insert("feature_definitions", frame(feature_definition()))
    rejected_then_accepted(
        syn_db, "feature_values",
        bad=feature_value(SNAP, available_at=quote_ts + timedelta(minutes=1), cutoff_time=quote_ts),
        good=feature_value(SNAP, available_at=quote_ts, cutoff_time=quote_ts),
        rule="available_at <= cutoff_time",
    )


def test_cutoff_cannot_be_after_the_snapshot(syn_db):
    syn_db.insert("feature_definitions", frame(feature_definition()))
    rejected_then_accepted(
        syn_db, "feature_values",
        bad=feature_value(SNAP, available_at=SNAP, cutoff_time=SNAP + timedelta(minutes=1)),
        good=feature_value(SNAP, available_at=SNAP),
        rule="cutoff_time <= snapshot_time",
    )


def test_feature_value_without_availability_time_is_rejected(syn_db):
    syn_db.insert("feature_definitions", frame(feature_definition()))
    rejected_then_accepted(
        syn_db, "feature_values",
        bad=feature_value(SNAP, available_at=pd.NaT),
        good=feature_value(SNAP, available_at=pd.NaT, value=None),  # missing, not leaked
        rule="value IS NULL OR available_at IS NOT NULL",
    )


def test_unregistered_feature_is_rejected(syn_db):
    row = feature_value(SNAP, available_at=SNAP, feature_name="not_registered")
    with pytest.raises(ConstraintViolation) as exc:
        syn_db.insert("feature_values", frame(row))
    assert "foreign-key" in exc.value.check
    syn_db.insert("feature_definitions", frame(feature_definition("not_registered")))
    syn_db.insert("feature_values", frame(row))


def test_quote_seen_after_snapshot_is_rejected(syn_db):
    late = SNAP + timedelta(seconds=1)
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(quote_ts=late, quote_available_at=late, quote_age_s=0.0),
        good=observation(),
        rule="quote_available_at <= snapshot_time",
    )


def test_consensus_seen_after_snapshot_is_rejected(syn_db):
    rejected_then_accepted(
        syn_db, "observations",
        # the factory quote is at SNAP - 1 min; consensus after it (but before the snapshot) is a leak
        bad=observation(p_consensus=0.58, consensus_available_at=SNAP - timedelta(seconds=30)),
        good=observation(p_consensus=0.58, consensus_available_at=SNAP - timedelta(minutes=4)),
        rule="consensus_available_at <= quote_ts",
    )


def test_last_trade_cannot_be_the_market_probability(syn_db):
    row = observation()
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(p_mid=row["market_last"]),  # 0.61, outside the 0.55/0.57 quote
        good=row,
        rule="p_mid >= market_yes_bid AND p_mid <= market_yes_ask",
    )


def test_one_sided_quote_is_not_an_observation(syn_db):
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(market_yes_bid=0.0, spread=0.57, p_mid=0.285),
        good=observation(),
        rule="market_yes_bid > 0",
    )


def test_snapshot_time_must_equal_start_minus_offset(syn_db):
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(snapshot_time=START - timedelta(minutes=25)),
        good=observation(),
        rule="snapshot_time = event_start_time - to_minutes(snapshot_offset_min)",
    )


def test_only_the_t30_snapshot_is_canonical(syn_db):
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(offset=60, is_canonical=True),
        good=observation(offset=60, is_canonical=False),
        rule="is_canonical = (snapshot_offset_min = 30)",
    )


def test_model_trained_on_data_after_its_first_prediction_is_rejected(syn_db):
    base = {
        "model_run_id": "m1", "model_type": "logistic_residual", "period_id": "2025H1",
        "feature_set_id": "fs", "feature_set_hash": "h", "config_hash": "c",
        "predict_from": SNAP, "dataset": "synthetic", "created_at": START, "is_synthetic": True,
    }
    rejected_then_accepted(
        syn_db, "model_runs",
        bad=base | {"train_data_through": SNAP},
        good=base | {"train_data_through": SNAP - timedelta(hours=20)},
        rule="train_data_through < predict_from",
    )


def test_observed_data_cannot_be_available_before_we_received_it(live_db):
    ts = START - timedelta(hours=1)
    quote = {"contract_id": "SYN-MLBGAME-X-NYY", "quote_ts": ts, "quote_source": "listing",
             "yes_bid": 0.55, "yes_ask": 0.57}
    rejected_then_accepted(
        live_db, "market_quotes",
        bad=quote | provenance(ts, synthetic=False, ingest_timestamp=ts + timedelta(seconds=5)),
        good=quote | provenance(ts, synthetic=False, ingest_timestamp=ts),
        rule="available_at >= ingest_timestamp",
    )


def logged_rejected_then_accepted(db, table, bad, good, rule):
    """append_log twin of rejected_then_accepted: the persistent logs are checked before any write."""
    with pytest.raises(ConstraintViolation) as exc:
        db.append_log(table, frame(bad))
    assert rule in exc.value.check, exc.value.check
    before = db.count(table)
    db.append_log(table, frame(good))
    assert db.count(table) == before + 1


LATE = SNAP + timedelta(seconds=30)
LIVE = {"dataset": "live", "is_synthetic": False}


@pytest.mark.parametrize("bad, rule", [
    # Sec 21: prices observed at decision time only, never a later or closing price
    ({"quote_ts": LATE, "quote_available_at": LATE}, "quote_available_at <= decision_ts"),
    ({"quoted_price": 0.60, "entry_price": 0.605}, "abs(quoted_price - ask) < 1e-9"),
    ({"side": "no", "decision": "BUY_NO", "quoted_price": 0.43, "conservative_probability": 0.30},
     "abs(quoted_price - (1 - bid)) < 1e-9"),
    ({"p_market": 0.58}, "abs(p_market - (bid + ask) / 2) < 1e-9"),
    ({"decision_ts": START, "created_at": START}, "decision_ts < event_start_time"),
    # the model's training outcomes must end before the decision
    ({"train_data_through": SNAP}, "train_data_through < decision_ts"),
    # a BUY carries its whole fill, and its stake never exceeds what the caps allowed
    ({"calculated_fee": None}, "AND calculated_fee IS NOT NULL"),
    ({"stake": 6.5, "hypothetical_size": 11.3}, "stake <= allowed_stake + 1e-9"),
    ({"conservative_probability": 0.30}, "side = 'yes' AND conservative_probability = p_conservative_yes"),
    ({"freeze_manifest_hash": "abc"}, "freeze_manifest_hash IS NULL OR manifest_status <> 'LOCKED'"),
])
def test_paper_trade_invariants(syn_db, bad, rule):
    logged_rejected_then_accepted(syn_db, "paper_trades", bad=paper_trade(**bad), good=paper_trade(), rule=rule)


def test_live_paper_trade_is_logged_before_the_game_starts(live_db):
    # prospective: a live row logged at or after first pitch is not a paper trade
    rule = "is_synthetic OR (created_at >= decision_ts AND created_at < event_start_time)"
    logged_rejected_then_accepted(live_db, "paper_trades", bad=paper_trade(created_at=START, **LIVE),
                                  good=paper_trade(created_at=SNAP + timedelta(minutes=5), **LIVE), rule=rule)
    with pytest.raises(ConstraintViolation) as exc:          # nor one back-dated before its snapshot
        live_db.append_log("paper_trades", frame(paper_trade(paper_trade_id="pt-2", created_at=SNAP - timedelta(seconds=1),
                                                             **LIVE)))
    assert rule in exc.value.check


def test_synthetic_paper_trade_never_carries_a_freeze_hash(syn_db, live_db):
    # 26.18: only live data qualifies, even under a hand-written FROZEN manifest
    frozen = {"manifest_status": "FROZEN", "freeze_manifest_hash": "abc"}
    logged_rejected_then_accepted(syn_db, "paper_trades", bad=paper_trade(**frozen),
                                  good=paper_trade(**frozen | {"freeze_manifest_hash": None}),
                                  rule="NOT is_synthetic OR freeze_manifest_hash IS NULL")
    # a live row may carry the hash (the fixtures share one log directory, hence the filter)
    live_db.append_log("paper_trades", frame(paper_trade(created_at=SNAP + timedelta(minutes=5), **LIVE, **frozen)))
    assert live_db.df("SELECT count(*) AS n FROM paper_trades WHERE NOT is_synthetic")["n"].item() == 1


def test_paper_pass_carries_no_fill(syn_db):
    passed = dict(decision="PASS", pass_reason="exposure_cap", hypothetical_size=0.0, entry_price=None, stake=None,
                  calculated_fee=None)
    logged_rejected_then_accepted(syn_db, "paper_trades", bad=paper_trade(**passed | {"stake": 5.75}),
                                  good=paper_trade(**passed),
                                  rule="decision <> 'PASS' OR (hypothetical_size = 0 AND entry_price IS NULL")


EXIT_T10 = START - timedelta(minutes=10)


@pytest.mark.parametrize("bad, rule", [
    # the exit sells into the bid of the side held at the exit snapshot, never a later or closing price
    ({"exit_quote_ts": EXIT_T10 + timedelta(seconds=5), "exit_quote_available_at": EXIT_T10 + timedelta(seconds=5)},
     "exit_quote_available_at <= exit_snapshot_time"),
    ({"exit_quoted_price": 0.62, "exit_price": 0.615}, "abs(exit_quoted_price - exit_bid) < 1e-9"),
    ({"exit_price": 0.61}, "exit_price <= exit_quoted_price"),
    ({"exit_fee": None}, "exit_fee IS NOT NULL"),
    ({"exit_status": "held", "held_quantity": 10.0, "exit_reason": "no_valid_exit_quote"},
     "exit_status <> 'held' OR (exit_quantity IS NULL"),
])
def test_paper_exit_invariants(syn_db, bad, rule):
    logged_rejected_then_accepted(syn_db, "paper_exits", bad=paper_exit(**bad), good=paper_exit(), rule=rule)


def test_live_paper_exit_is_not_recorded_before_its_snapshot(live_db):
    logged_rejected_then_accepted(live_db, "paper_exits", bad=paper_exit(created_at=EXIT_T10 - timedelta(minutes=1), **LIVE),
                                  good=paper_exit(**LIVE), rule="is_synthetic OR created_at >= exit_snapshot_time")


@pytest.mark.parametrize("bad, rule", [
    ({"settled_at": START + timedelta(hours=2)}, "settled_at >= outcome_available_at"),
    ({"outcome_available_at": None}, "outcome_available_at IS NOT NULL AND settled_at"),
    ({"result": "no"}, "(outcome = 1 AND result = 'yes') OR (outcome = 0 AND result = 'no')"),
    ({"side_won": 0}, "side_won IS NULL OR (side = 'yes' AND side_won = outcome)"),
    ({"net_pnl": None}, "AND net_pnl IS NOT NULL"),
])
def test_paper_settlement_invariants(syn_db, bad, rule):
    logged_rejected_then_accepted(syn_db, "paper_settlements", bad=paper_settlement(**bad), good=paper_settlement(),
                                  rule=rule)


def test_quote_without_displayed_size_is_not_an_observation(syn_db):
    # 26.10: a valid quote has nonzero displayed size. Only historical candles,
    # which carry no size at all, may enter, and they are flagged unverified.
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(ask_size=0.0),
        good=observation(),
        rule="NOT size_verified OR (bid_size > 0 AND ask_size > 0)",
    )
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(observation_id="o-unsized", bid_size=None, ask_size=None, size_verified=False),
        good=observation(observation_id="o-candle", bid_size=None, ask_size=None, size_verified=False,
                         quote_source="candle"),
        rule="size_verified OR quote_source = 'candle'",
    )
    # candle quotes are labelled candle_degraded, and nothing else is
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(observation_id="o-candle-exec", bid_size=None, ask_size=None, size_verified=False,
                        quote_source="candle", quote_quality="executable"),
        good=observation(observation_id="o-candle-exec", bid_size=None, ask_size=None, size_verified=False,
                         quote_source="candle", quote_quality="candle_degraded"),
        rule="(quote_quality = 'candle_degraded') = (quote_source = 'candle')",
    )
    rejected_then_accepted(
        syn_db, "observations",
        bad=observation(observation_id="o-book-degraded", quote_quality="candle_degraded"),
        good=observation(observation_id="o-book-degraded"),
        rule="(quote_quality = 'candle_degraded') = (quote_source = 'candle')",
    )
