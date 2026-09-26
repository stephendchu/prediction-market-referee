"""Minimal valid rows per table. Tests override one field to break one rule."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pandas as pd

START = pd.Timestamp("2025-06-01T23:05:00Z")  # a game's scheduled start
GAME_DATE = date(2025, 6, 1)


def provenance(ts: pd.Timestamp = START - timedelta(hours=2), synthetic: bool = True, **over) -> dict:
    row = {
        "source": "synthetic" if synthetic else "exchange",
        "source_timestamp": ts,
        "ingest_timestamp": ts,
        "available_at": ts,
        "availability_basis": "synthetic" if synthetic else "observed",
        "collector_version": "test",
        "ingest_run_id": "run-test",
        "is_synthetic": synthetic,
    }
    return row | over


def observation(offset: int = 30, **over) -> dict:
    snap = START - timedelta(minutes=offset)
    quote_ts = snap - timedelta(minutes=1)
    row = {
        "observation_id": f"SYN-NYY@T-{offset}",
        "game_id": "syn-mlb-1",
        "contract_id": "SYN-NYY",
        "sport": "mlb",
        "season": 2025,
        "official_date": GAME_DATE,
        "home_team": "NYY",
        "away_team": "TB",
        "yes_team": "NYY",
        "yes_side": "home",
        "event_start_time": START,
        "snapshot_offset_min": offset,
        "snapshot_time": snap,
        "is_canonical": offset == 30,
        "quote_ts": quote_ts,
        "quote_available_at": quote_ts,
        "quote_age_s": 60.0,
        "quote_source": "synthetic",
        "market_yes_bid": 0.55,
        "market_yes_ask": 0.57,
        "spread": 0.57 - 0.55,
        "bid_size": 250.0,
        "ask_size": 180.0,
        "size_verified": True,
        "p_mid": 0.56,
        "market_last": 0.61,
        "build_run_id": "build-test",
        "is_synthetic": True,
    }
    row |= over
    # the label follows the quote source unless a test overrides it
    row.setdefault("quote_quality", "candle_degraded" if row["quote_source"] == "candle" else "executable")
    return row


def feature_definition(name: str = "starter_k_minus_bb", **over) -> dict:
    row = {
        "feature_name": name,
        "family": "starter",
        "dtype": "float",
        "description": "test feature",
        "availability_rule": "max(available_at of inputs)",
        "version": "test",
    }
    return row | over


def feature_value(snapshot_time: pd.Timestamp, available_at: pd.Timestamp, cutoff_time: pd.Timestamp | None = None,
                  **over) -> dict:
    row = {
        "observation_id": "SYN-NYY@T-30",
        "feature_name": "starter_k_minus_bb",
        "value": 0.12,
        "available_at": available_at,
        "snapshot_time": snapshot_time,
        "cutoff_time": snapshot_time if cutoff_time is None else cutoff_time,
        "build_run_id": "build-test",
        "is_synthetic": True,
    }
    return row | over


def paper_trade(**over) -> dict:
    """A valid BUY_YES paper decision at the T-30 snapshot of `observation()`: 10 contracts at
    ask 0.57 + 0.25 x 0.02 spread = 0.575, taker fee ceil(0.07 x 10 x 0.575 x 0.425) = 0.18."""
    snap = START - timedelta(minutes=30)
    quote_ts = snap - timedelta(minutes=1)
    row = {
        "paper_trade_id": "pt-1", "created_at": snap, "sport": "mlb", "dataset": "synthetic",
        "game_id": "syn-mlb-1", "contract_id": "SYN-NYY", "observation_id": "SYN-NYY@T-30",
        "official_date": GAME_DATE, "event_start_time": START, "snapshot_offset_min": 30,
        "quote_quality": "executable", "period_id": "2025H1", "model_run_id": "mr-1",
        "model_type": "logistic_residual", "train_data_through": snap - timedelta(days=30),
        "manifest_status": "LOCKED", "freeze_manifest_hash": None, "config_hash": "cfg",
        "decision_ts": snap, "quote_ts": quote_ts, "quote_available_at": quote_ts,
        "bid": 0.55, "ask": 0.57, "bid_size": 250.0, "ask_size": 180.0, "depth_json": None,
        "p_market": 0.56, "model_probability": 0.66, "p_conservative_yes": 0.64, "p_conservative_no": 0.30,
        "conservative_probability": 0.64, "predicted_edge": 0.09, "net_edge": 0.047,
        "decision": "BUY_YES", "pass_reason": None, "side": "yes", "hypothetical_size": 10.0,
        "quoted_price": 0.57, "spread_penalty_frac": 0.25, "entry_price": 0.575, "stake": 5.75,
        "fee_schedule_id": "quadratic-taker-v1", "fee_schedule_effective_date": date(2021, 1, 1),
        "maker_or_taker": "taker", "calculated_fee": 0.18, "slippage_cost": 0.05, "depth_verified": True,
        "requested_stake": 10.0, "allowed_stake": 6.0, "capped_by": None,
        "exit_rule": "hold_to_settlement", "exit_offset_min": None, "is_synthetic": True,
    }
    return row | over


def paper_exit(**over) -> dict:
    """A valid full exit of `paper_trade(exit_rule='exit_at_offset', exit_offset_min=10)` at T-10:
    YES bid 0.60 less 0.25 x 0.02 = 0.595 for 10 contracts, fee ceil(0.07 x 10 x 0.595 x 0.405) = 0.17."""
    snap = START - timedelta(minutes=10)
    quote_ts = snap - timedelta(seconds=20)
    row = {
        "paper_trade_id": "pt-1", "created_at": snap, "side": "yes", "exit_rule": "exit_at_offset",
        "exit_status": "exited", "exit_reason": None, "exit_observation_id": "SYN-NYY@T-10",
        "exit_snapshot_time": snap, "exit_quote_ts": quote_ts, "exit_quote_available_at": quote_ts,
        "exit_bid": 0.60, "exit_ask": 0.62, "exit_quoted_price": 0.60, "exit_price": 0.595,
        "exit_quantity": 10.0, "held_quantity": 0.0, "exit_proceeds": 5.95,
        "exit_fee_schedule_id": "quadratic-taker-v1", "exit_fee_schedule_effective_date": date(2021, 1, 1),
        "exit_fee": 0.17, "exit_slippage_cost": 0.05, "exit_depth_verified": True, "spread_penalty_frac": 0.25,
        "dataset": "synthetic", "is_synthetic": True,
    }
    return row | over


def paper_settlement(**over) -> dict:
    """A valid settlement of `paper_trade()` held to settlement, YES won: 10 x 1 - 5.75 - 0.18."""
    row = {
        "paper_trade_id": "pt-1", "settled_at": START + timedelta(hours=4), "outcome": 1,
        "outcome_available_at": START + timedelta(hours=3), "result": "yes", "decision": "BUY_YES",
        "side": "yes", "side_won": 1, "pnl": 4.25, "fees": 0.18, "net_pnl": 4.07, "held_quantity": 10.0,
        "exit_proceeds": None, "dataset": "synthetic", "is_synthetic": True,
    }
    return row | over


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def locked_registration(holdout, **over) -> dict:
    """A LOCKED holdout registration entry for the walk-forward period `holdout`."""
    from src.backtest import holdout as H

    text = "the final holdout period"
    entry = {"name": "final_holdout", "dataset": "live", "family": "outcome_residual", "rule_text": text,
             "rule_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
             "period_id": holdout.period_id, "registered_at": "2026-01-01T00:00:00+00:00",
             "status": "LOCKED", "lock": {"locked_at": "2026-01-02T00:00:00+00:00",
                                          "start": holdout.start.isoformat(), "end": holdout.end.isoformat()}}
    entry |= over
    entry["entry_hash"] = H.entry_hash(entry)
    return entry
