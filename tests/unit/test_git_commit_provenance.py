"""Every run, hypothesis and experiment row records the commit of the code that wrote it (spec Sec 20
"Track experiments: ... git_commit"; 26.4 "git commit, registration timestamp").

holdout.git_commit() is patched to a fixed value, so no test depends on the repo's HEAD. Before this
file, train.py and engine.py wrote git_commit None on every row.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from factories import feature_definition, feature_value, frame
from src.backtest import engine
from src.backtest import holdout as H
from src.backtest.engine import Policy, run_backtest
from src.backtest.registry import load_hypotheses, register_hypotheses, register_now
from src.config import load_sport_config
from src.features.mlb import write_features
from src.models.train import run_training
from src.sports.synthetic import SyntheticSource
from src.storage.database import connect
from src.storage.ingest import ingest

COMMIT = "c0ffee" + "0" * 34
DAY = date(2026, 4, 6)
NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)
HYPS = "config/mlb_hypotheses.yaml"
COMMIT_SQL = {
    "ingest_runs": "SELECT git_commit FROM ingest_runs",
    "hypotheses": "SELECT git_commit FROM hypotheses",
    "model_runs": "SELECT git_commit FROM model_runs",
    "experiments": "SELECT git_commit FROM experiments",
    "backtest_runs": "SELECT git_commit FROM backtest_runs",
}


def small_season():
    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= {"seasons": [2026], "through_date": DAY.isoformat()}
    cfg["walk_forward"]["periods"] = [
        {"id": "P0", "role": "train_only", "start": date(2026, 3, 1), "end": date(2026, 3, 29)},
        {"id": "P1", "role": "evaluation", "start": date(2026, 3, 30), "end": date(2026, 4, 2)},
        {"id": "PH", "role": "holdout", "start": date(2026, 4, 3), "end": date(2026, 4, 30)},
    ]
    cfg["model"]["min_train_games"] = 30
    cfg["uncertainty"]["ensemble_size"] = 5
    cfg["review"]["bootstrap_resamples"] = 200
    return cfg


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """ingest -> features -> train -> backtest on one small synthetic season, with HEAD = COMMIT."""
    root = tmp_path_factory.mktemp("provenance")
    cfg = small_season()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUANT_DATA_DIR", str(root / "data"))
        mp.setattr(H, "git_commit", lambda: COMMIT)
        db = connect("mlb", "synthetic", path=root / "world.duckdb", logs=root / "logs")
        try:
            ingest(db, SyntheticSource(cfg), date(2026, 1, 1), DAY)
            write_features(db, cfg, DAY)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # expit overflow on synthetic extremes
                selection = dict(run_training(db, cfg, now=NOW, threads=2).selection)
                run_backtest(db, cfg, selection, now=NOW)
            tables = {t: db.df(sql)["git_commit"] for t, sql in COMMIT_SQL.items()}
        finally:
            db.close()
    return SimpleNamespace(tables=tables)


@pytest.mark.parametrize("table", list(COMMIT_SQL))
def test_every_row_records_the_commit(world, table):
    col = world.tables[table]
    assert len(col) > 0, f"{table}: the world wrote no rows, so this test would prove nothing"
    assert col.tolist() == [COMMIT] * len(col), f"{table}: {col.value_counts(dropna=False).to_dict()}"


def test_register_now_records_the_commit(syn_db, monkeypatch):
    monkeypatch.setattr(H, "git_commit", lambda: COMMIT)
    n, _ = register_now(syn_db, HYPS)
    stored = syn_db.df(COMMIT_SQL["hypotheses"])["git_commit"]
    assert n == len(stored) > 0 and stored.tolist() == [COMMIT] * n


def test_without_git_the_rows_stay_null(syn_db, monkeypatch):
    """git unavailable: git_commit() is None and the row keeps NULL; the referee flags it (26.5).
    This is the call `quant train` makes: a run is recorded whatever the tree, and only the
    registration step and a freeze refuse (tests/unit/test_dirty_tree.py)."""
    monkeypatch.setattr(H, "git_commit", lambda: None)
    register_hypotheses(syn_db, load_hypotheses(HYPS), git_commit=H.git_commit(), now=NOW)
    stored = syn_db.df(COMMIT_SQL["hypotheses"])["git_commit"]
    assert len(stored) > 0 and stored.isna().all()


def test_h3_primary_metric_rows_record_the_commit(syn_db, monkeypatch):
    """engine._record_h3 (H_MLB_003's fee-band metric): no other test reaches its recording path,
    because the shipped yaml disables H_MLB_003. Here it is ACTIVE on a hand-made decision frame."""
    monkeypatch.setattr(H, "git_commit", lambda: COMMIT)
    hyps = load_hypotheses(HYPS)
    for h in hyps:
        if h["hypothesis_id"] == engine.H3:
            for k in ("status", "status_reason", "status_decision"):
                h.pop(k, None)
    register_hypotheses(syn_db, hyps, now=NOW - timedelta(days=1))
    cfg = small_season()
    pol = Policy.from_config(cfg)
    snap = pd.Timestamp("2026-03-30T22:00Z")
    rows, band = [], []
    for g in range(8):
        for side, inside in (("a", 1.0), ("b", 0.0)):
            oid = f"O{g}{side}"
            rows.append({"observation_id": oid, "game_id": f"G{g}", "period_id": "P1",
                         "scenario": engine.scenario_name(pol.baseline), "decision": "BUY_YES",
                         "net_pnl": float((g % 3) - 1 + inside), "quantity": 10.0})
            band.append(feature_value(snap, snap - timedelta(hours=1), observation_id=oid,
                                      feature_name=engine.H3_FEATURE, value=inside))
    syn_db.insert("feature_definitions", frame(feature_definition(engine.H3_FEATURE)))
    syn_db.insert("feature_values", frame(*band))
    notes: list[str] = []
    n = engine._record_h3(syn_db, cfg, pd.DataFrame(rows), pol, "logistic_residual", "cfg", notes, NOW)
    stored = syn_db.df(COMMIT_SQL["experiments"])["git_commit"]
    assert n == len(stored) == 2, notes                     # P1 and pooled
    assert stored.tolist() == [COMMIT, COMMIT]
