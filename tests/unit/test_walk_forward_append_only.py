"""walk_forward_periods is append-only, so an old period_id can never name another window.

Before this, every `quant train` replaced the table with the config's periods. An edited config could give
an old period_id new dates, and every model run and prediction already stored under that id would then point
at a window it was never trained or scored on. Now, as for backtest runs (ADR 0001 N1) and trains (ADR 0002
item 4), the table only grows: a period_id, once written, keeps its role, start and end for good; a config
that changes them is refused before anything is written; new ids are appended with the train's own rows.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from src.backtest import holdout as H
from src.backtest import walk_forward as W
from src.config import load_sport_config
from src.features.mlb import write_features
from src.models import train as T
from src.sports.synthetic import SyntheticSource
from src.storage.database import ConstraintViolation, connect
from src.storage.ingest import ingest
from src.storage.schemas import ALL_TABLES

PERIODS_SQL = "SELECT * FROM walk_forward_periods ORDER BY period_id"
COUNTS_SQL = ("SELECT (SELECT count(*) FROM model_runs) AS model_runs, (SELECT count(*) FROM predictions) AS predictions, "
              "(SELECT count(*) FROM hypotheses) AS hypotheses, (SELECT count(*) FROM hypothesis_status) AS status, "
              "(SELECT count(*) FROM walk_forward_periods) AS periods")
CLEAN = "c" * 40


def cfg_with(periods: list[dict]) -> dict:
    return {"walk_forward": {"periods": periods}}


P0 = {"id": "P0", "role": "train_only", "start": date(2026, 3, 1), "end": date(2026, 3, 29)}
P1 = {"id": "P1", "role": "evaluation", "start": date(2026, 3, 30), "end": date(2026, 4, 1)}
PH = {"id": "PH", "role": "holdout", "start": date(2026, 4, 3), "end": date(2026, 4, 30)}
P2 = {"id": "P2", "role": "evaluation", "start": date(2026, 4, 2), "end": date(2026, 4, 2)}  # the gap before PH
BASE = [P0, P1, PH]


def periods(ps: list[dict]) -> list[W.Period]:
    return W.load_periods(cfg_with(ps), "synthetic")


def written(db) -> pd.DataFrame:
    return db.df(PERIODS_SQL)


# ---------------------------------------------------------------------------
# storage: the table only grows
# ---------------------------------------------------------------------------

def test_walk_forward_periods_is_append_only():
    assert ALL_TABLES["walk_forward_periods"].append_only


def test_replace_refuses_walk_forward_periods(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    before = written(syn_db)
    moved = W.period_rows(periods([P0, P1 | {"end": date(2026, 4, 2)}, PH]))
    with pytest.raises(ValueError, match="walk_forward_periods is append-only"):
        syn_db.replace("walk_forward_periods", moved, "period_id = ?", ["P1"])
    with pytest.raises(ValueError, match="walk_forward_periods is append-only"):
        syn_db.replace_tables({"walk_forward_periods": moved})
    pd.testing.assert_frame_equal(written(syn_db), before)


def test_a_repeated_period_id_raises_and_keeps_the_first_row(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    before = written(syn_db)
    moved = W.period_rows(periods([P0, P1 | {"end": date(2026, 4, 2)}, PH]))
    with pytest.raises(ConstraintViolation):
        syn_db.append_tables({"walk_forward_periods": moved[moved["period_id"] == "P1"]})
    pd.testing.assert_frame_equal(written(syn_db), before)


# ---------------------------------------------------------------------------
# new_period_rows: what a train may append (reads only)
# ---------------------------------------------------------------------------

def test_on_an_empty_table_every_period_is_new(syn_db):
    new = W.new_period_rows(syn_db, periods(BASE))
    pd.testing.assert_frame_equal(new, W.period_rows(periods(BASE)))
    assert syn_db.count("walk_forward_periods") == 0  # reads only


def test_the_same_periods_again_are_not_new(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    assert W.new_period_rows(syn_db, periods(BASE)).empty


# The role case adds P2 so the config still has an evaluation period and loads (26.2).
@pytest.mark.parametrize("change, extra", [
    ({"start": date(2026, 3, 31)}, []), ({"end": date(2026, 4, 2)}, []), ({"role": "train_only"}, [P2]),
], ids=["start", "end", "role"])
def test_a_written_period_id_with_another_window_or_role_is_refused(syn_db, change, extra):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    before = written(syn_db)
    with pytest.raises(W.PeriodChanged) as exc:
        W.new_period_rows(syn_db, periods([P0, P1 | change, *extra, PH]))
    msg = str(exc.value)
    assert "P1" in msg and "append-only" in msg and "new period_id" in msg
    assert "2026-03-30" in msg and "2026-04-01" in msg and "evaluation" in msg  # what was written
    pd.testing.assert_frame_equal(written(syn_db), before)


def test_the_holdout_cannot_be_relabelled(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    later = {"id": "PH2", "role": "holdout", "start": date(2026, 5, 1), "end": date(2026, 5, 31)}
    with pytest.raises(W.PeriodChanged, match="PH"):
        W.new_period_rows(syn_db, periods([P0, P1, PH | {"role": "evaluation"}, later]))


def test_every_changed_period_is_named(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    with pytest.raises(W.PeriodChanged) as exc:
        W.new_period_rows(syn_db, periods([P0 | {"end": date(2026, 3, 28)}, P1 | {"start": date(2026, 3, 31)}, PH]))
    assert "P0" in str(exc.value) and "P1" in str(exc.value)


def test_a_new_period_id_is_new_and_the_written_ones_are_not(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    new = W.new_period_rows(syn_db, periods([P0, P1, P2, PH]))
    assert list(new["period_id"]) == ["P2"]
    # PH's position moved from 2 to 3; its ordinal is the definition's, not the window, so that is allowed
    assert new["definition_hash"].iloc[0] == W.definition_hash(periods([P0, P1, P2, PH]))


def test_a_period_dropped_from_the_config_is_not_an_error(syn_db):
    syn_db.append_tables({"walk_forward_periods": W.period_rows(periods(BASE))})
    assert W.new_period_rows(syn_db, periods([P1, PH])).empty


# ---------------------------------------------------------------------------
# quant train: refuses first, appends with its own rows
# ---------------------------------------------------------------------------

DAY = date(2026, 4, 6)
NOW = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)


def season(ps: list[dict]) -> dict:
    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= {"seasons": [2026], "through_date": DAY.isoformat()}
    cfg["walk_forward"]["periods"] = ps
    cfg["model"]["min_train_games"] = 30
    cfg["uncertainty"]["ensemble_size"] = 5
    cfg["review"]["bootstrap_resamples"] = 200
    return cfg


def counts(db) -> dict:
    return db.df(COUNTS_SQL).iloc[0].to_dict()


def model_files(db) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in T.model_dir(db).glob("*.pkl")}


def with_a_new_hypothesis(tmp_path) -> str:
    """The config hypotheses plus one new id: a train that reaches registration writes a row."""
    raw = yaml.safe_load(T.hypotheses_path("mlb").read_text())
    env = next(h for h in raw["hypotheses"] if h["hypothesis_id"] == "H_MLB_L1_ENV")
    raw["hypotheses"].append({**env, "hypothesis_id": "H_TEST_WF"})
    path = tmp_path / "hyps.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def refuse_model_runs(real, frames):
    """append_tables, except that a write including model_runs is refused before anything is written, as
    if another train had written the same model_run_id in between."""
    if "model_runs" in frames:
        raise ConstraintViolation("model_runs", "model_run_id is unique (ADR 0002 item 4)", detail="test: refused")
    return real(frames)


@pytest.fixture(scope="module")
def trains(tmp_path_factory):
    """ingest -> features -> train (BASE) -> train (BASE again) -> train (P2 added) -> train (P1 moved)
    -> train whose period check is bypassed (a concurrent writer) -> train with a new period whose own
    rows are refused -> state after each."""
    root = tmp_path_factory.mktemp("wf_append_only")
    out = SimpleNamespace()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUANT_DATA_DIR", str(root / "data"))
        mp.setattr(H, "git_commit", lambda: CLEAN)
        db = connect("mlb", "synthetic", path=root / "world.duckdb", logs=root / "logs")
        try:
            ingest(db, SyntheticSource(season(BASE)), date(2026, 1, 1), DAY)
            write_features(db, season(BASE), DAY)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # expit overflow on synthetic extremes
                T.run_training(db, season(BASE), now=NOW, threads=2)
                out.first = written(db)
                T.run_training(db, season(BASE), now=NOW + timedelta(hours=1), threads=2)
                out.again = written(db)
                out.added_run = T.run_training(db, season([P0, P1, P2, PH]), now=NOW + timedelta(hours=2), threads=2)
                out.added = written(db)
                out.before_moved, out.files_before_moved = counts(db), model_files(db)
                try:
                    T.run_training(db, season([P0, P1 | {"end": date(2026, 4, 2)}, P2 | {"start": date(2026, 4, 3),
                                   "end": date(2026, 4, 3)}, PH | {"start": date(2026, 4, 4)}]),
                                   hyps_path=with_a_new_hypothesis(root), now=NOW + timedelta(hours=3), threads=2)
                    out.moved_error = None
                except W.PeriodChanged as exc:
                    out.moved_error = exc
                out.after_moved, out.periods_after_moved = counts(db), written(db)
                out.files_after_moved = model_files(db)
                moved = W.period_rows(periods([P0, P1 | {"end": date(2026, 4, 2)}, P2 | {"start": date(2026, 4, 3),
                                                "end": date(2026, 4, 3)}, PH | {"start": date(2026, 4, 4)}]))
                with pytest.MonkeyPatch.context() as race:
                    race.setattr(T, "new_period_rows", lambda db_, ps: moved[moved["period_id"] == "P1"],
                                 raising=False)
                    try:
                        T.run_training(db, season(BASE), now=NOW + timedelta(hours=4), threads=2)
                        out.race_error = None
                    except ConstraintViolation as exc:
                        out.race_error = exc
                out.after_race, out.periods_after_race = counts(db), written(db)
                with pytest.MonkeyPatch.context() as refused:
                    real = db.append_tables
                    refused.setattr(db, "append_tables", lambda frames: refuse_model_runs(real, frames))
                    try:  # P2b: a new id (P2's window, P2 dropped), so the check passes and it is appended
                        T.run_training(db, season([P0, P1, P2 | {"id": "P2b"}, PH]), now=NOW + timedelta(hours=5),
                                       threads=2)
                        out.refused_error = None
                    except ConstraintViolation as exc:
                        out.refused_error = exc
                out.after_refused, out.periods_after_refused = counts(db), written(db)
        finally:
            db.close()
    return out


def test_the_first_train_writes_the_configured_periods(trains):
    assert list(trains.first["period_id"]) == ["P0", "P1", "PH"]
    assert set(trains.first["definition_hash"]) == {W.definition_hash(periods(BASE))}


def test_a_second_train_on_the_same_periods_writes_none(trains):
    pd.testing.assert_frame_equal(trains.again, trains.first)


def test_a_train_with_a_new_period_appends_only_that_one(trains):
    assert len(trains.added_run.model_runs) > 0
    assert list(trains.added["period_id"]) == ["P0", "P1", "P2", "PH"]
    old = trains.added[trains.added["period_id"] != "P2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, trains.first)  # the written rows keep their ordinal and hash
    p2 = trains.added[trains.added["period_id"] == "P2"].iloc[0]
    assert p2["definition_hash"] == W.definition_hash(periods([P0, P1, P2, PH])) != trains.first["definition_hash"][0]


def test_a_train_that_moves_a_written_period_is_refused_before_anything_is_written(trains):
    assert isinstance(trains.moved_error, W.PeriodChanged), trains.moved_error
    msg = str(trains.moved_error)
    assert all(p in msg for p in ("P1", "P2", "PH")) and "append-only" in msg
    assert trains.after_moved == trains.before_moved  # no hypotheses, status, model runs, predictions or periods
    assert trains.files_after_moved == trains.files_before_moved  # and no model file
    pd.testing.assert_frame_equal(trains.periods_after_moved, trains.added)


def test_the_periods_are_appended_with_the_trains_rows_in_one_transaction(trains):
    """A writer that got a period id in between the check and the append: the whole train rolls back."""
    assert isinstance(trains.race_error, ConstraintViolation), trains.race_error
    for t in ("model_runs", "predictions", "periods"):
        assert trains.after_race[t] == trains.after_moved[t], t
    pd.testing.assert_frame_equal(trains.periods_after_race, trains.added)


def test_a_train_whose_own_rows_are_refused_leaves_no_new_period(trains):
    """The new period ids are not written ahead of the train's rows: refusing the
    model runs leaves no period behind that no train used."""
    assert isinstance(trains.refused_error, ConstraintViolation), trains.refused_error
    assert "P2b" not in set(trains.periods_after_refused["period_id"])
    for t in ("model_runs", "predictions", "periods"):
        assert trains.after_refused[t] == trains.after_race[t], t
