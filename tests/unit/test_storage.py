"""DuckDB schema, dataset isolation, checked writes, and Parquet-backed views."""

from __future__ import annotations

from datetime import timedelta

import pytest
from factories import GAME_DATE, START, frame, observation, provenance

from src.storage.database import ConstraintViolation, DatasetMismatch, SchemaMismatch, connect
from src.storage.schemas import (
    DERIVED_TABLES,
    PERSISTENT_TABLES,
    PROVENANCE,
    SCHEMA_VERSION,
)


def game(**over) -> dict:
    row = {"game_id": "syn-mlb-1", "sport": "mlb", "season": 2025, "official_date": GAME_DATE,
           "scheduled_start": START, "home_team": "NYY", "away_team": "TB"}
    return row | provenance() | over


def test_schema_creates_every_table_and_log_view(syn_db):
    names = set(syn_db.df("SELECT table_name FROM information_schema.tables")["table_name"])
    expected = {t.name for t in (*DERIVED_TABLES, *PERSISTENT_TABLES)}
    assert expected <= names
    assert syn_db.scalar("SELECT value FROM db_meta WHERE key = 'schema_version'") == SCHEMA_VERSION


def test_reconnect_is_idempotent(tmp_path):
    path = tmp_path / "mlb_synthetic.duckdb"
    connect("mlb", "synthetic", path=path, logs=tmp_path / "logs").close()
    db = connect("mlb", "synthetic", path=path, logs=tmp_path / "logs")
    assert db.count("db_meta") == 5
    db.close()


def test_schema_version_change_demands_a_rebuild(tmp_path):
    path = tmp_path / "mlb_synthetic.duckdb"
    db = connect("mlb", "synthetic", path=path, logs=tmp_path / "logs")
    db.con.execute("UPDATE db_meta SET value = 'db-0.0.0' WHERE key = 'schema_version'")
    db.close()
    with pytest.raises(SchemaMismatch):
        connect("mlb", "synthetic", path=path, logs=tmp_path / "logs")


def test_a_database_file_belongs_to_one_dataset(tmp_path):
    path = tmp_path / "db.duckdb"
    connect("mlb", "synthetic", path=path, logs=tmp_path / "logs").close()
    with pytest.raises(DatasetMismatch):
        connect("mlb", "live", path=path, logs=tmp_path / "logs")


def test_synthetic_database_rejects_real_rows(syn_db):
    with pytest.raises(ConstraintViolation) as exc:
        syn_db.insert("games", frame(game() | provenance(synthetic=False)))
    assert exc.value.check == "is_synthetic = true"
    syn_db.insert("games", frame(game()))


def test_live_database_rejects_synthetic_rows(live_db):
    with pytest.raises(ConstraintViolation) as exc:
        live_db.insert("games", frame(game()))
    assert exc.value.check == "is_synthetic = false"
    live_db.insert("games", frame(game() | provenance(synthetic=False)))


def test_synthetic_flag_and_availability_basis_must_agree(syn_db):
    with pytest.raises(ConstraintViolation) as exc:
        syn_db.insert("games", frame(game(availability_basis="observed")))
    assert "availability_basis = 'synthetic'" in exc.value.check


def test_unknown_or_missing_columns_are_rejected(syn_db):
    with pytest.raises(ConstraintViolation, match="unknown"):
        syn_db.insert("games", frame(game(surprise=1)))
    bad = game()
    del bad["home_team"]
    with pytest.raises(ConstraintViolation, match="missing"):
        syn_db.insert("games", frame(bad))
    with pytest.raises(ConstraintViolation, match="NOT NULL"):
        syn_db.insert("games", frame(game(away_team=None)))


def test_replace_is_atomic(syn_db):
    syn_db.insert("games", frame(game()))
    with pytest.raises(ConstraintViolation):
        syn_db.replace("games", frame(game(away_team="NYY")), "game_id = ?", ["syn-mlb-1"])
    assert syn_db.scalar("SELECT away_team FROM games") == "TB"
    syn_db.replace("games", frame(game(away_team="BOS")), "game_id = ?", ["syn-mlb-1"])
    assert syn_db.scalar("SELECT away_team FROM games") == "BOS"


def test_outcome_needs_an_availability_time_and_must_match_the_score(syn_db):
    with pytest.raises(ConstraintViolation, match="outcome_available_at"):
        syn_db.insert("games", frame(game(home_win=True, home_score=5, away_score=3)))
    with pytest.raises(ConstraintViolation, match="home_score > away_score"):
        syn_db.insert("games", frame(game(home_win=True, home_score=2, away_score=3,
                                          outcome_available_at=START + timedelta(hours=3))))
    syn_db.insert("games", frame(game(home_win=True, home_score=5, away_score=3,
                                      outcome_available_at=START + timedelta(hours=3))))


def test_prediction_bounds_follow_26_12(syn_db):
    row = {"model_run_id": "m1", "observation_id": "o1", "period_id": "p1", "p_market": 0.55,
           "p_model": 0.58, "lower_bound": 0.53, "upper_bound": 0.63, "p_conservative_yes": 0.53,
           "p_conservative_no": 1 - 0.63, "is_synthetic": True}
    with pytest.raises(ConstraintViolation, match="p_conservative_yes = lower_bound"):
        syn_db.insert("predictions", frame(row | {"p_conservative_yes": 0.55}))
    with pytest.raises(ConstraintViolation, match="lower_bound <= p_model"):
        syn_db.insert("predictions", frame(row | {"lower_bound": 0.59, "p_conservative_yes": 0.59}))
    syn_db.insert("predictions", frame(row))


def test_backtest_trade_needs_a_fee_schedule_and_cannot_beat_the_ask(syn_db):
    row = {"backtest_run_id": "b1", "scenario": "pen_25", "observation_id": "o1", "game_id": "g1",
           "contract_id": "c1", "period_id": "p1", "snapshot_time": START, "decision": "BUY_YES",
           "side": "yes", "p_market": 0.56, "p_model": 0.63, "quoted_price": 0.57,
           "spread_penalty_frac": 0.25, "execution_price": 0.575, "quantity": 10.0,
           "fee_schedule_id": "fee-test", "fee_schedule_effective_date": GAME_DATE,
           "maker_or_taker": "taker", "calculated_fee": 0.18, "exit_rule": "hold_to_settlement",
           "exit_status": "held", "held_quantity": 10.0, "is_synthetic": True}
    with pytest.raises(ConstraintViolation, match="fee_schedule_id IS NOT NULL"):
        syn_db.insert("backtest_decisions", frame(row | {"fee_schedule_id": None}))
    with pytest.raises(ConstraintViolation, match="execution_price >= quoted_price"):
        syn_db.insert("backtest_decisions", frame(row | {"execution_price": 0.56}))  # midpoint fill
    syn_db.insert("backtest_decisions", frame(row))
    passed = {k: row[k] for k in ("backtest_run_id", "game_id", "contract_id", "period_id", "snapshot_time",
                                  "p_market", "p_model", "spread_penalty_frac", "exit_rule", "is_synthetic")}
    syn_db.insert("backtest_decisions", frame(passed | {"scenario": "pen_25", "observation_id": "o2",
                                                        "decision": "PASS", "pass_reason": "below_min_edge"}))


def test_every_normalized_source_table_carries_provenance():
    prov = {c.name for c in PROVENANCE}
    sourced = [t for t in DERIVED_TABLES if "source" in t.column_names]
    assert len(sourced) >= 12
    for t in sourced:
        assert prov <= set(t.column_names), t.name


def test_observation_row_keeps_last_trade_but_prices_from_the_quote(syn_db):
    syn_db.insert("observations", frame(observation()))
    last, p, bid, ask = syn_db.con.execute(
        "SELECT market_last, p_mid, market_yes_bid, market_yes_ask FROM observations").fetchone()
    assert last == 0.61 and bid <= p <= ask


def test_persistent_log_is_append_only_parquet(syn_db, tmp_path):
    row = {"hypothesis_id": "H1", "family": "starter", "description": "d", "mechanism": "m",
           "feature_definitions_json": "[]", "expected_direction": "+", "minimum_sample": 200,
           "evaluation_periods_json": "[]", "primary_metric": "brier", "label": "CONFIRMATORY",
           "definition_hash": "h", "registered_at": START, "dataset": "synthetic", "is_synthetic": True}
    first = syn_db.append_log("hypotheses", frame(row))
    second = syn_db.append_log("hypotheses", frame(row | {"hypothesis_id": "H2"}))
    assert first != second and first.exists() and second.exists()
    assert sorted(syn_db.df("SELECT hypothesis_id FROM hypotheses")["hypothesis_id"]) == ["H1", "H2"]


