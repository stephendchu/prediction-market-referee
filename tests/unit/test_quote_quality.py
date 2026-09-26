"""Executable and candle_degraded quotes never share a training set or a backtest run."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from factories import GAME_DATE, START, frame, observation, provenance
from src.backtest.engine import run_backtest
from src.config import load_sport_config
from src.features.observations import build_observations
from src.models.train import load_train_data
from src.storage.schemas import CANDLE_DEGRADED, DEGRADED_EXECUTION, EXECUTABLE

WHEN = datetime(2026, 2, 1, tzinfo=timezone.utc)
CANDLE = dict(bid_size=None, ask_size=None, size_verified=False, quote_source="candle")


def two_class_observations(syn_db, *, executable: bool = True) -> None:
    rows = [observation(observation_id="SYN-C@T-30", game_id="syn-mlb-2", contract_id="SYN-C", outcome=0, **CANDLE)]
    if executable:
        rows.append(observation(outcome=1))
    syn_db.insert("observations", frame(*rows))


def test_the_builder_labels_candle_quotes_degraded(syn_db):
    """Through the real builder: an order-book quote is executable, a candle quote candle_degraded."""
    game = "syn-mlb-9"
    syn_db.insert("games", frame({
        "game_id": game, "sport": "mlb", "season": 2025, "official_date": GAME_DATE, "scheduled_start": START,
        "home_team": "NYY", "away_team": "TB", "home_win": True, "outcome_available_at": START + pd.Timedelta(hours=4),
        **provenance()}))
    syn_db.insert("contracts", frame(*[
        {"contract_id": cid, "game_id": game, "sport": "mlb", "market_type": "ml", "yes_side": side,
         "yes_team": team, "result": result, **provenance()}
        for cid, side, team, result in (("SYN-9-NYY", "home", "NYY", "yes"), ("SYN-9-TB", "away", "TB", "no"))]))
    q_ts = START - pd.Timedelta(minutes=31)
    syn_db.insert("market_quotes", frame(
        {"contract_id": "SYN-9-NYY", "quote_ts": q_ts, "quote_source": "orderbook", "yes_bid": 0.55, "yes_ask": 0.57,
         "bid_size": 200.0, "ask_size": 150.0, **provenance(q_ts)},
        {"contract_id": "SYN-9-TB", "quote_ts": q_ts, "quote_source": "candle", "yes_bid": 0.43, "yes_ask": 0.45,
         "bid_size": None, "ask_size": None, **provenance(q_ts)}))
    build_observations(syn_db, load_sport_config("mlb"))
    obs = syn_db.df("SELECT contract_id, quote_source, size_verified, quote_quality FROM observations "
                    "ORDER BY contract_id").set_index("contract_id")
    assert obs.loc["SYN-9-NYY", "quote_quality"] == EXECUTABLE and bool(obs.loc["SYN-9-NYY", "size_verified"])
    assert obs.loc["SYN-9-TB", "quote_quality"] == CANDLE_DEGRADED and not bool(obs.loc["SYN-9-TB", "size_verified"])


def test_training_data_is_one_quote_class(syn_db):
    two_class_observations(syn_db)
    ex, candle = load_train_data(syn_db), load_train_data(syn_db, CANDLE_DEGRADED)
    assert list(ex.obs["observation_id"]) == ["SYN-NYY@T-30"] and ex.quote_quality == EXECUTABLE
    assert list(candle.obs["observation_id"]) == ["SYN-C@T-30"] and candle.quote_quality == CANDLE_DEGRADED
    assert ex.fingerprint != candle.fingerprint
    with pytest.raises(ValueError, match="quote_quality must be one of"):
        load_train_data(syn_db, "mixed")


def test_candle_only_data_is_not_trained_as_executable(syn_db):
    two_class_observations(syn_db, executable=False)
    with pytest.raises(RuntimeError, match="no executable observations.*--quote-quality candle_degraded"):
        load_train_data(syn_db)


def test_backtest_run_is_one_quote_class(syn_db):
    cfg = load_sport_config("mlb")
    two_class_observations(syn_db)
    syn_db.insert("model_runs", frame({
        "model_run_id": "mr-1", "model_type": "logistic_residual", "period_id": "2025H1", "feature_set_id": "fs",
        "feature_set_hash": "fsh", "config_hash": "c", "predict_from": pd.Timestamp("2025-01-01T00:00Z"),
        "dataset": "synthetic", "created_at": WHEN, "is_synthetic": True}))
    syn_db.insert("predictions", frame(*[
        {"model_run_id": "mr-1", "observation_id": oid, "period_id": "2025H1", "p_market": 0.56, "p_model": 0.70,
         "lower_bound": 0.66, "upper_bound": 0.74, "p_conservative_yes": 0.66, "p_conservative_no": 0.26,
         "is_synthetic": True} for oid in ("SYN-NYY@T-30", "SYN-C@T-30")]))

    seen = {}
    for quality, oid in ((EXECUTABLE, "SYN-NYY@T-30"), (CANDLE_DEGRADED, "SYN-C@T-30")):
        res = run_backtest(syn_db, cfg, {"selected": "logistic_residual", "quote_quality": quality}, now=WHEN)
        assert set(res.runs["quote_quality"]) == {quality}
        dec = syn_db.df("SELECT * FROM backtest_decisions")
        dec = dec[dec["backtest_run_id"].isin(res.runs["backtest_run_id"])]  # runs accumulate (ADR 0001 N1)
        assert set(dec["observation_id"]) == {oid}                      # the other class never enters the run
        assert (res.notes[:1] == [DEGRADED_EXECUTION]) == (quality == CANDLE_DEGRADED)
        seen[quality] = set(res.runs["backtest_run_id"])
    assert not seen[EXECUTABLE] & seen[CANDLE_DEGRADED]


def test_backtest_refuses_unknown_quote_quality(syn_db):
    with pytest.raises(ValueError, match="unknown quote_quality"):
        run_backtest(syn_db, load_sport_config("mlb"), {"selected": "logistic_residual", "quote_quality": "x"})
