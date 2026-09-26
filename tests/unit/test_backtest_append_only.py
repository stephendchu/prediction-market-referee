"""ADR 0001 N1: a backtest run id is never reused, and no run overwrites another.

`bt-54815509fa927a44` was overwritten by a rerun on a later commit, and the clean-commit row the
referee had read was gone. Each `quant backtest` invocation now gets its own ids (the hash takes
the run's commit and time), and backtest_runs and backtest_decisions only grow.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from factories import frame, observation
from src.backtest import holdout as H
from src.backtest.engine import run_backtest
from src.backtest.execution import fee_schedule_rows
from src.config import load_sport_config
from src.reports import diagnostics as D
from src.storage.database import ConstraintViolation
from src.storage.schemas import ALL_TABLES

APPEND_ONLY = ("backtest_runs", "backtest_decisions")
WHEN = datetime(2026, 2, 1, tzinfo=timezone.utc)
CLEAN_A, CLEAN_B = "a" * 40, "b" * 40


def bt_run(rid: str, off: int = 30, created: str = "2026-02-01T00:00Z", **over) -> dict:
    return {"backtest_run_id": rid, "model_type": "logistic_residual", "snapshot_offset_min": off,
            "config_json": json.dumps({"model_type": "logistic_residual"}), "config_hash": "c",
            "quote_quality": "executable", "dataset": "synthetic", "git_commit": CLEAN_A,
            "created_at": pd.Timestamp(created), "is_synthetic": True} | over


def bt_decision(rid: str, oid: str = "SYN-NYY@T-30", off: int = 30) -> dict:
    return {"backtest_run_id": rid, "scenario": "pen_25", "model_run_id": "mr-1", "observation_id": oid,
            "game_id": "syn-mlb-1", "contract_id": "SYN-NYY", "period_id": "2025H1",
            "snapshot_time": observation(offset=off)["snapshot_time"], "decision": "PASS",
            "pass_reason": "below_min_edge", "side": "yes", "p_market": 0.56, "p_model": 0.60,
            "predicted_edge": 0.03, "quoted_price": 0.57, "spread_penalty_frac": 0.25, "outcome": 1,
            "side_won": 1, "exit_rule": "hold_to_settlement", "is_synthetic": True}


# ---------------------------------------------------------------------------
# storage: the two tables only grow
# ---------------------------------------------------------------------------

def test_the_append_only_tables_are_the_backtest_and_train_tables():
    """The train tables joined in ADR 0002 item 4 (tests/unit/test_train_append_only.py), then
    walk_forward_periods (tests/unit/test_walk_forward_append_only.py) and train_lineage."""
    assert {t.name for t in ALL_TABLES.values() if t.append_only} == set(APPEND_ONLY) | {
        "model_runs", "predictions", "walk_forward_periods", "train_lineage"}


@pytest.mark.parametrize("table", APPEND_ONLY)
def test_replace_refuses_an_append_only_table(syn_db, table):
    syn_db.insert("backtest_runs", frame(bt_run("bt-1")))
    syn_db.insert("backtest_decisions", frame(bt_decision("bt-1")))
    empty = pd.DataFrame(columns=ALL_TABLES[table].column_names)
    with pytest.raises(ValueError, match=f"{table} is append-only.*ADR 0001"):
        syn_db.replace(table, empty, "backtest_run_id = ?", ["bt-1"])
    assert syn_db.count(table) == 1


@pytest.mark.parametrize("table", APPEND_ONLY)
def test_replace_tables_refuses_an_append_only_table_before_emptying_anything(syn_db, table):
    syn_db.insert("backtest_runs", frame(bt_run("bt-1")))
    syn_db.insert("backtest_decisions", frame(bt_decision("bt-1")))
    syn_db.replace_tables({"fee_schedules": fee_schedule_rows()})
    n_fees = syn_db.count("fee_schedules")
    frames = {"fee_schedules": fee_schedule_rows(), table: pd.DataFrame(columns=ALL_TABLES[table].column_names)}
    with pytest.raises(ValueError, match=f"{table} is append-only.*ADR 0001"):
        syn_db.replace_tables(frames)
    assert syn_db.count(table) == 1 and syn_db.count("fee_schedules") == n_fees


def test_append_tables_keeps_the_rows_already_there(syn_db):
    syn_db.append_tables({"backtest_runs": frame(bt_run("bt-1")), "backtest_decisions": frame(bt_decision("bt-1"))})
    syn_db.append_tables({"backtest_runs": frame(bt_run("bt-2")), "backtest_decisions": frame(bt_decision("bt-2"))})
    assert sorted(syn_db.df("SELECT backtest_run_id FROM backtest_runs")["backtest_run_id"]) == ["bt-1", "bt-2"]
    assert sorted(syn_db.df("SELECT backtest_run_id FROM backtest_decisions")["backtest_run_id"]) == ["bt-1", "bt-2"]


def test_a_repeated_run_id_raises_and_writes_nothing(syn_db):
    """The run row is unique; the second write is refused whole, decisions included, and the first stays."""
    first = bt_run("bt-1", config_hash="first")
    syn_db.append_tables({"backtest_runs": frame(first), "backtest_decisions": frame(bt_decision("bt-1"))})
    again = {"backtest_runs": frame(bt_run("bt-1", config_hash="second")),
             "backtest_decisions": frame(bt_decision("bt-1", oid="SYN-NYY@T-60", off=60))}
    with pytest.raises(ConstraintViolation):
        syn_db.append_tables(again)
    runs = syn_db.df("SELECT * FROM backtest_runs")
    assert list(runs["config_hash"]) == ["first"]
    assert list(syn_db.df("SELECT observation_id FROM backtest_decisions")["observation_id"]) == ["SYN-NYY@T-30"]


def test_append_tables_rolls_back_every_table_on_a_failure(syn_db):
    """A bad decisions frame after a good runs frame leaves neither behind."""
    bad = pd.DataFrame([bt_decision("bt-1") | {"decision": "HOLD"}])      # not a DECISIONS value
    with pytest.raises(ConstraintViolation):
        syn_db.append_tables({"backtest_runs": frame(bt_run("bt-1")), "backtest_decisions": bad})
    assert syn_db.count("backtest_runs") == 0 and syn_db.count("backtest_decisions") == 0


# ---------------------------------------------------------------------------
# engine: one id per invocation
# ---------------------------------------------------------------------------

def seed(db) -> None:
    db.insert("observations", frame(observation(outcome=1), observation(offset=60, outcome=1, is_canonical=False)))
    db.insert("model_runs", frame({
        "model_run_id": "mr-1", "model_type": "logistic_residual", "period_id": "2025H1", "feature_set_id": "fs",
        "feature_set_hash": "fsh", "config_hash": "c", "predict_from": pd.Timestamp("2025-01-01T00:00Z"),
        "dataset": "synthetic", "created_at": WHEN, "is_synthetic": True}))
    db.insert("predictions", frame(*[
        {"model_run_id": "mr-1", "observation_id": oid, "period_id": "2025H1", "p_market": 0.56, "p_model": 0.70,
         "lower_bound": 0.66, "upper_bound": 0.74, "p_conservative_yes": 0.66, "p_conservative_no": 0.26,
         "is_synthetic": True} for oid in ("SYN-NYY@T-30", "SYN-NYY@T-60")]))


def backtest(db, commit: str | None, when: datetime, monkeypatch):
    monkeypatch.setattr(H, "git_commit", lambda: commit)
    return run_backtest(db, load_sport_config("mlb"), {"selected": "logistic_residual"}, now=when)


@pytest.mark.parametrize("first, second", [
    ((CLEAN_A, WHEN), (CLEAN_B, WHEN)),                              # the overwrite N1 names: a new commit
    ((CLEAN_A, WHEN), (CLEAN_A + H.DIRTY, WHEN)),                    # the same HEAD, now with uncommitted edits
    ((CLEAN_A + H.DIRTY, WHEN), (CLEAN_A + H.DIRTY, WHEN + timedelta(seconds=1))),  # two dirty reruns
    ((CLEAN_A, WHEN), (CLEAN_A, WHEN + timedelta(microseconds=1))),  # a plain rerun
], ids=["new-commit", "clean-then-dirty", "dirty-rerun", "same-commit-rerun"])
def test_every_invocation_gets_new_ids_and_keeps_the_last_one(syn_db, monkeypatch, first, second):
    seed(syn_db)
    a = backtest(syn_db, *first, monkeypatch)
    n_decisions = syn_db.count("backtest_decisions")
    b = backtest(syn_db, *second, monkeypatch)
    ids_a, ids_b = set(a.runs["backtest_run_id"]), set(b.runs["backtest_run_id"])
    assert len(ids_a) == len(ids_b) == len(a.runs) and not ids_a & ids_b
    runs = syn_db.df("SELECT backtest_run_id, git_commit FROM backtest_runs")
    assert set(runs["backtest_run_id"]) == ids_a | ids_b                      # the earlier invocation survives
    assert dict(zip(runs["backtest_run_id"], runs["git_commit"])) == (
        {i: first[0] for i in ids_a} | {i: second[0] for i in ids_b})
    assert syn_db.count("backtest_decisions") == 2 * n_decisions
    kept = syn_db.df("SELECT DISTINCT backtest_run_id FROM backtest_decisions")["backtest_run_id"]
    seeded = syn_db.df("SELECT backtest_run_id FROM backtest_runs WHERE snapshot_offset_min IN (30, 60)")
    assert len(seeded) == 4 and set(kept) == set(seeded["backtest_run_id"])   # T-30 and T-60 of both invocations


def test_the_same_invocation_twice_raises_and_overwrites_nothing(syn_db, monkeypatch):
    """Same commit and same timestamp: the id repeats, and the write is refused, not applied."""
    seed(syn_db)
    a = backtest(syn_db, CLEAN_A, WHEN, monkeypatch)
    before = syn_db.df("SELECT * FROM backtest_decisions ORDER BY ALL")
    with pytest.raises(ConstraintViolation):
        backtest(syn_db, CLEAN_A, WHEN, monkeypatch)
    assert set(syn_db.df("SELECT backtest_run_id FROM backtest_runs")["backtest_run_id"]) == set(a.runs["backtest_run_id"])
    pd.testing.assert_frame_equal(syn_db.df("SELECT * FROM backtest_decisions ORDER BY ALL"), before)


def test_fee_schedules_are_still_replaced(syn_db, monkeypatch):
    seed(syn_db)
    backtest(syn_db, CLEAN_A, WHEN, monkeypatch)
    backtest(syn_db, CLEAN_A, WHEN + timedelta(seconds=1), monkeypatch)
    assert syn_db.count("fee_schedules") == len(fee_schedule_rows())


# ---------------------------------------------------------------------------
# referee: the latest invocation, never mixed with another
# ---------------------------------------------------------------------------

def review_world(db, runs: list[dict]) -> None:
    db.insert("observations", frame(observation(outcome=1), observation(offset=60, outcome=1, is_canonical=False)))
    db.insert("predictions", frame(*[
        {"model_run_id": "mr-1", "observation_id": oid, "period_id": "2025H1", "p_market": 0.56, "p_model": 0.60,
         "lower_bound": 0.55, "upper_bound": 0.65, "p_conservative_yes": 0.55, "p_conservative_no": 0.35,
         "is_synthetic": True} for oid in ("SYN-NYY@T-30", "SYN-NYY@T-60")]))
    db.insert("backtest_runs", frame(*runs))
    db.insert("backtest_decisions", frame(*[
        bt_decision(r["backtest_run_id"], f"SYN-NYY@T-{r['snapshot_offset_min']}", r["snapshot_offset_min"])
        for r in runs]))


def test_the_review_reads_the_latest_invocation(syn_db):
    review_world(syn_db, [bt_run("bt-old", created="2026-02-01T00:00Z", git_commit=CLEAN_A),
                          bt_run("bt-new", created="2026-02-02T00:00Z", git_commit=CLEAN_B)])
    inputs = D.load_review_inputs(syn_db, load_sport_config("mlb"))
    assert inputs.backtest_run["backtest_run_id"] == "bt-new"
    assert inputs.backtest_run["git_commit"] == CLEAN_B


def test_sensitivity_comes_from_the_same_invocation(syn_db):
    review_world(syn_db, [bt_run("bt-30"), bt_run("bt-60", 60)])
    inputs = D.load_review_inputs(syn_db, load_sport_config("mlb"))
    assert inputs.backtest_run["backtest_run_id"] == "bt-30"
    assert set(inputs.sensitivity) == {60}
    assert list(inputs.sensitivity[60]["backtest_run_id"]) == ["bt-60"]


def test_sensitivity_matches_a_missing_commit_to_a_missing_commit(syn_db):
    """Without git both rows carry None: that is one invocation, not two."""
    review_world(syn_db, [bt_run("bt-30", git_commit=None), bt_run("bt-60", 60, git_commit=None)])
    assert set(D.load_review_inputs(syn_db, load_sport_config("mlb")).sensitivity) == {60}


@pytest.mark.parametrize("field, other", [
    ("created_at", pd.Timestamp("2026-01-31T00:00Z")),                # an older invocation
    ("git_commit", CLEAN_B),
    ("git_commit", None),
    ("config_hash", "c-other"),
    ("model_type", "lightgbm_residual"),
])
def test_sensitivity_never_mixes_in_another_invocation(syn_db, field, other):
    """The T-60 row differs from the T-30 run in one invocation field: no sensitivity run, not a stale one."""
    review_world(syn_db, [bt_run("bt-30"), bt_run("bt-60", 60, **{field: other})])
    inputs = D.load_review_inputs(syn_db, load_sport_config("mlb"))
    assert inputs.backtest_run["backtest_run_id"] == "bt-30"
    assert inputs.sensitivity == {}


def test_a_run_without_a_commit_does_not_match_one_with_a_commit(syn_db):
    review_world(syn_db, [bt_run("bt-30", git_commit=None), bt_run("bt-60", 60, git_commit=CLEAN_B)])
    assert D.load_review_inputs(syn_db, load_sport_config("mlb")).sensitivity == {}


def test_sensitivity_never_mixes_in_another_quote_class(syn_db):
    review_world(syn_db, [bt_run("bt-30"), bt_run("bt-60", 60, quote_quality="candle_degraded")])
    assert D.load_review_inputs(syn_db, load_sport_config("mlb")).sensitivity == {}


def test_a_named_run_takes_sensitivity_from_its_own_invocation(syn_db):
    review_world(syn_db, [bt_run("bt-30-old"), bt_run("bt-60-old", 60),
                          bt_run("bt-30-new", created="2026-02-02T00:00Z", git_commit=CLEAN_B),
                          bt_run("bt-60-new", 60, created="2026-02-02T00:00Z", git_commit=CLEAN_B)])
    inputs = D.load_review_inputs(syn_db, load_sport_config("mlb"), backtest_run_id="bt-30-old")
    assert list(inputs.sensitivity[60]["backtest_run_id"]) == ["bt-60-old"]
