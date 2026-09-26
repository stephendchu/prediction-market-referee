"""ADR 0002 item 3: a negative-control failure (26.16 FAIL) on a run without a clean commit raises a
blocking flag; `quant freeze` refuses until a review of a later run from a clean commit has 26.16 PASS.
While a flag is open, hypothesis registration runs once, for the clearing run.

The flag is a persistent append-only log (review_flags), so it survives a reconnect and a re-ingest.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from src.backtest import holdout as H
from src.backtest import review_flags as RF
from src.backtest.registry import hypothesis_rows, load_hypotheses, register_now
from src.config import CONFIG_DIR, load_sport_config
from src.reports import diagnostics as D
from src.storage.database import ConstraintViolation, connect

CLEAN, DIRTY = "c" * 40, "c" * 40 + "-dirty"
T0 = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def review(run_id: str, commit: str | None, state: str | None, *, at: datetime = T0,
           dataset: str = "synthetic") -> dict:
    crit = [{"id": cid, "state": D.PASS} for cid in D.CRITERIA_IDS if cid != "26.16"]
    if state is not None:
        crit.append({"id": "26.16", "state": state})
    return {"dataset": dataset, "is_synthetic": dataset == "synthetic", "referee_version": D.REFEREE_VERSION,
            "criteria": crit,
            "backtest_run": {"backtest_run_id": run_id, "created_at": at.isoformat(), "git_commit": commit,
                             "config_hash": "cfg0"}}


def flags(db) -> pd.DataFrame:
    return db.df("SELECT * FROM review_flags ORDER BY recorded_at, event DESC")


def open_ids(db) -> list[str]:
    return sorted(RF.open_flags(db)["backtest_run_id"])


# -- raising ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("commit", [DIRTY, None, "", "unavailable: not a git repository"],
                         ids=["dirty", "none", "empty", "unavailable"])
def test_a_dirty_negative_control_failure_raises_one_flag(syn_db, commit):
    out = RF.record_review(syn_db, review("bt-a", commit, D.FAIL))
    assert out["event"] == RF.FLAG_RAISE
    f = flags(syn_db)
    assert len(f) == 1 and len(out["recorded"]) == 1
    row = f.iloc[0]
    assert (row.event, row.criterion, row.backtest_run_id, row.flag_id) == \
        ("raise", "26.16", "bt-a", RF.flag_id("synthetic", "bt-a"))
    assert row.dataset == "synthetic" and bool(row.is_synthetic) and row.referee_version == D.REFEREE_VERSION
    assert pd.Timestamp(row.run_created_at) == pd.Timestamp(T0)
    assert open_ids(syn_db) == ["bt-a"]
    assert [o["backtest_run_id"] for o in out["open"]] == ["bt-a"]


def test_reviewing_the_same_flagged_run_again_does_not_raise_a_second_row(syn_db):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    out = RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    assert out["event"] == RF.FLAG_RAISE and out["recorded"] == []
    assert len(flags(syn_db)) == 1 and open_ids(syn_db) == ["bt-a"]
    # a second dirty run is its own flag
    RF.record_review(syn_db, review("bt-b", DIRTY, D.FAIL, at=T0 + timedelta(hours=1)))
    assert open_ids(syn_db) == ["bt-a", "bt-b"]


@pytest.mark.parametrize(("commit", "state"), [(CLEAN, D.FAIL), (DIRTY, D.PASS), (DIRTY, D.INSUFFICIENT),
                                               (DIRTY, None), (CLEAN, D.INSUFFICIENT)])
def test_no_flag_without_a_dirty_negative_control_failure(syn_db, commit, state):
    out = RF.record_review(syn_db, review("bt-a", commit, state))
    assert out["event"] is None and out["recorded"] == [] and out["open"] == []
    assert len(flags(syn_db)) == 0


# -- clearing --------------------------------------------------------------------------------------

def test_a_later_clean_run_with_26_16_pass_clears_every_earlier_flag(syn_db):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    RF.record_review(syn_db, review("bt-b", None, D.FAIL, at=T0 + timedelta(minutes=5)))
    out = RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    assert out["event"] == RF.FLAG_CLEAR and out["open"] == []
    assert open_ids(syn_db) == []
    clears = flags(syn_db).query("event == 'clear'")
    assert sorted(clears["flag_id"]) == sorted([RF.flag_id("synthetic", "bt-a"), RF.flag_id("synthetic", "bt-b")])
    assert set(clears["backtest_run_id"]) == {"bt-c"} and set(clears["git_commit"]) == {CLEAN}
    assert len(out["recorded"]) == 2
    # nothing is deleted: the raise rows are still there
    assert len(flags(syn_db).query("event == 'raise'")) == 2
    # a second clean review clears nothing more
    again = RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    assert again["event"] == RF.FLAG_CLEAR and again["recorded"] == []
    assert len(flags(syn_db)) == 4


@pytest.mark.parametrize(("commit", "state", "dt"), [
    (CLEAN, D.INSUFFICIENT, 1),      # no controls cannot show the controls fail
    (CLEAN, D.FAIL, 1),              # REJECTED on a clean commit; the flag stays
    (DIRTY, D.PASS, 1),              # not a clean rerun
    (CLEAN, D.PASS, -1),             # a clean run created before the flagged run
    (CLEAN, D.PASS, 0),              # ... or at the same time
], ids=["insufficient", "clean-fail", "dirty-pass", "earlier", "same-time"])
def test_the_flag_stays_open_unless_a_later_clean_run_passes_26_16(syn_db, commit, state, dt):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    out = RF.record_review(syn_db, review("bt-z", commit, state, at=T0 + timedelta(hours=dt)))
    assert out["recorded"] == []
    assert open_ids(syn_db) == ["bt-a"]
    assert [o["backtest_run_id"] for o in out["open"]] == ["bt-a"]


def test_a_clean_pass_clears_only_the_flags_raised_on_earlier_runs(syn_db):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL, at=T0))
    RF.record_review(syn_db, review("bt-late", DIRTY, D.FAIL, at=T0 + timedelta(hours=3)))
    RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    assert open_ids(syn_db) == ["bt-late"]


def test_open_flags_reads_this_dataset_only(syn_db):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    other = review("bt-x", DIRTY, D.FAIL, dataset="live")
    other["is_synthetic"] = True  # the synthetic database's CHECK; only the dataset differs
    RF.record_review(syn_db, other)
    assert open_ids(syn_db) == ["bt-a"]


def test_flags_persist_across_a_reconnect(tmp_path):
    db = connect("mlb", "synthetic", path=tmp_path / "a.duckdb", logs=tmp_path / "logs")
    RF.record_review(db, review("bt-a", DIRTY, D.FAIL))
    db.close()
    # a rebuilt database (a re-ingest) over the same logs still has the flag
    db = connect("mlb", "synthetic", path=tmp_path / "b.duckdb", logs=tmp_path / "logs")
    try:
        assert open_ids(db) == ["bt-a"]
        with pytest.raises(RF.ReviewBlocked):
            RF.require_unblocked(db, "freeze")
    finally:
        db.close()


# -- the schema keeps a raise dirty and a clear clean ----------------------------------------------

def _row(**over) -> pd.DataFrame:
    row = {"flag_id": "rf-0", "event": "raise", "criterion": "26.16", "backtest_run_id": "bt-a",
           "run_created_at": pd.Timestamp(T0), "git_commit": DIRTY, "config_hash": "cfg0", "detail": "d",
           "recorded_at": pd.Timestamp(T0), "referee_version": D.REFEREE_VERSION, "dataset": "synthetic",
           "is_synthetic": True}
    return pd.DataFrame([row | over])


@pytest.mark.parametrize("over", [
    {"event": "raise", "git_commit": CLEAN},
    {"event": "clear", "git_commit": DIRTY},
    {"event": "clear", "git_commit": None},
    {"event": "clear", "git_commit": ""},
    {"event": "clear", "git_commit": "unavailable: not a git repository"},
    {"event": "open"},
    {"criterion": "26.17"},
], ids=["raise-clean", "clear-dirty", "clear-none", "clear-empty", "clear-unavailable", "event", "criterion"])
def test_the_schema_refuses_a_row_that_breaks_the_flag_rules(syn_db, over):
    with pytest.raises(ConstraintViolation):
        syn_db.append_log("review_flags", _row(**over))
    assert len(flags(syn_db)) == 0


@pytest.mark.parametrize("over", [{"event": "raise", "git_commit": DIRTY}, {"event": "raise", "git_commit": None},
                                  {"event": "clear", "git_commit": CLEAN}])
def test_the_schema_accepts_well_formed_rows(syn_db, over):
    syn_db.append_log("review_flags", _row(**over))
    assert len(flags(syn_db)) == 1


# -- what the flag blocks --------------------------------------------------------------------------

def test_require_unblocked_names_the_open_flags(syn_db):
    RF.require_unblocked(syn_db, "freeze")  # nothing open: no error
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    with pytest.raises(RF.ReviewBlocked) as exc:
        RF.require_unblocked(syn_db, "freeze")
    msg = str(exc.value)
    assert "freeze" in msg and "bt-a" in msg and "ADR 0002" in msg and "26.16" in msg
    assert isinstance(exc.value, RuntimeError)
    RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    RF.require_unblocked(syn_db, "freeze")


# -- NaN, blank or padded commits are not clean, in Python and in the schema -------

NOT_BARE = [float("nan"), pd.NA, "  ", " " + CLEAN]
NOT_BARE_IDS = ["nan", "pd.NA", "blank", "padded"]


@pytest.mark.parametrize("commit", NOT_BARE, ids=NOT_BARE_IDS)
def test_a_negative_control_failure_on_a_nan_or_blank_commit_raises_a_flag(syn_db, commit):
    out = RF.record_review(syn_db, review("bt-a", commit, D.FAIL))
    assert out["event"] == D.FLAG_RAISE and open_ids(syn_db) == ["bt-a"]


@pytest.mark.parametrize("commit", NOT_BARE, ids=NOT_BARE_IDS)
def test_a_pass_on_a_nan_or_blank_commit_clears_nothing(syn_db, commit):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    out = RF.record_review(syn_db, review("bt-b", commit, D.PASS, at=T0 + timedelta(hours=1)))
    assert out["event"] is None and out["recorded"] == [] and open_ids(syn_db) == ["bt-a"]


@pytest.mark.parametrize("commit", ["  ", "\t", " " + CLEAN, CLEAN + "\n"], ids=["blank", "tab", "leading", "trailing"])
def test_the_schema_keeps_blank_and_padded_commits_off_a_clear_and_allows_them_on_a_raise(syn_db, commit):
    with pytest.raises(ConstraintViolation):
        syn_db.append_log("review_flags", _row(event="clear", git_commit=commit))
    syn_db.append_log("review_flags", _row(event="raise", git_commit=commit))
    assert list(flags(syn_db)["event"]) == ["raise"]


# -- registration while a flag is open: the clearing run's only -------------------

HYPS = CONFIG_DIR / "mlb_hypotheses.yaml"


def written(db) -> tuple[int, int]:
    """(hypotheses rows, hypothesis_status rows) of the database."""
    return len(db.df("SELECT * FROM hypotheses")), len(db.df("SELECT * FROM hypothesis_status"))


def hyps_file(tmp_path, name: str, *, new_ids=(), nc: str | None = "ACTIVE", pause: str | None = None):
    """The config hypotheses, edited: `new_ids` copies of H_MLB_L1_ENV under new ids; the negative
    control dropped (nc=None) or given another status; `pause` PAUSED."""
    raw = yaml.safe_load(HYPS.read_text())
    hyps = [h for h in raw["hypotheses"] if nc is not None or h["family"] != "negative_control"]
    env = next(h for h in hyps if h["hypothesis_id"] == "H_MLB_L1_ENV")
    hyps += [{**env, "hypothesis_id": i} for i in new_ids]
    for h in hyps:
        if (h["family"] == "negative_control" and nc != "ACTIVE") or h["hypothesis_id"] == pause:
            h.update(status=nc if h["family"] == "negative_control" else "PAUSED", status_reason="test",
                     status_decision="test")
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"hypotheses": hyps}, sort_keys=False))
    return path


def test_the_clearing_run_can_register_while_a_flag_is_open(syn_db):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    n, reg = register_now(syn_db, HYPS)
    assert n == len(load_hypotheses(HYPS)) == written(syn_db)[0] == len(reg)
    assert open_ids(syn_db) == ["bt-a"]  # registering clears nothing
    with pytest.raises(RF.ReviewBlocked, match="freeze refused"):  # the freeze stays refused
        RF.require_unblocked(syn_db, "freeze")


def test_a_second_registration_while_the_flag_is_open_refuses(syn_db, tmp_path):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    register_now(syn_db, HYPS)
    before = written(syn_db)
    for path in (hyps_file(tmp_path, "new", new_ids=("H_TEST_NEW",)), HYPS):  # new rows, or none at all
        with pytest.raises(RF.ReviewBlocked) as exc:
            register_now(syn_db, path)
        msg = str(exc.value)
        assert msg.startswith("register hypotheses refused: 1 open blocking flag(s)") and "bt-a" in msg
        assert "the one registration allowed while a flag is open, for the clearing run, was used at " in msg
        assert written(syn_db) == before  # refused before anything is written
    RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    assert open_ids(syn_db) == []
    assert register_now(syn_db, hyps_file(tmp_path, "new2", new_ids=("H_TEST_NEW",)))[0] == 1


def test_a_status_change_after_the_flag_is_the_one_registration(syn_db, tmp_path):
    register_now(syn_db, HYPS)  # before any flag
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    n_hyp, n_status = written(syn_db)
    assert register_now(syn_db, hyps_file(tmp_path, "pause", pause="H_MLB_L1_ENV"))[0] == 0
    assert written(syn_db) == (n_hyp, n_status + 1)
    with pytest.raises(RF.ReviewBlocked, match="was used at"):
        register_now(syn_db, hyps_file(tmp_path, "new", new_ids=("H_TEST_NEW",)))
    assert written(syn_db) == (n_hyp, n_status + 1)


def test_a_registration_before_the_flag_does_not_use_it(syn_db, tmp_path):
    register_now(syn_db, HYPS)
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    assert register_now(syn_db, hyps_file(tmp_path, "new", new_ids=("H_TEST_NEW",)))[0] == 1


def test_a_registration_that_writes_nothing_does_not_use_it(syn_db, tmp_path):
    register_now(syn_db, HYPS)  # before any flag
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    before = written(syn_db)
    assert register_now(syn_db, HYPS)[0] == 0  # every definition and status unchanged: no row
    assert written(syn_db) == before
    assert register_now(syn_db, hyps_file(tmp_path, "new", new_ids=("H_TEST_NEW",)))[0] == 1


def test_rows_of_another_dataset_do_not_use_it(syn_db):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    syn_db.append_log("hypotheses", hypothesis_rows(load_hypotheses(HYPS)[:1], dataset="live", is_synthetic=True,
                                                    git_commit=CLEAN, now=datetime.now(timezone.utc)))
    assert register_now(syn_db, HYPS)[0] == len(load_hypotheses(HYPS))


def test_each_new_flag_allows_one_more_registration(syn_db, tmp_path):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    register_now(syn_db, HYPS)
    # the run that registration served was not clean: its review raises a second flag
    RF.record_review(syn_db, review("bt-b", DIRTY, D.FAIL, at=T0 + timedelta(hours=1)))
    assert open_ids(syn_db) == ["bt-a", "bt-b"]
    assert register_now(syn_db, hyps_file(tmp_path, "one", new_ids=("H_TEST_ONE",)))[0] == 1
    with pytest.raises(RF.ReviewBlocked, match="2 open blocking flag"):
        register_now(syn_db, hyps_file(tmp_path, "two", new_ids=("H_TEST_ONE", "H_TEST_TWO")))


@pytest.mark.parametrize("nc", [None, "PAUSED", "DISABLED"], ids=["absent", "paused", "disabled"])
def test_a_registration_whose_run_cannot_clear_the_flag_refuses(syn_db, tmp_path, nc):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    with pytest.raises(RF.ReviewBlocked) as exc:
        register_now(syn_db, hyps_file(tmp_path, "nc", nc=nc))
    msg = str(exc.value)
    assert msg.startswith("register hypotheses refused: 1 open blocking flag(s)")
    assert "no ACTIVE negative_control hypothesis" in msg and "26.16 cannot PASS" in msg
    assert written(syn_db) == (0, 0)
    # the one registration is still there for a run that can clear the flag
    assert register_now(syn_db, HYPS)[0] == len(load_hypotheses(HYPS))


def test_without_an_open_flag_registration_is_unrestricted(syn_db, tmp_path):
    assert register_now(syn_db, hyps_file(tmp_path, "nc", nc=None))[0] > 0
    assert register_now(syn_db, hyps_file(tmp_path, "one", nc=None, new_ids=("H_TEST_ONE",)))[0] == 1
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    assert register_now(syn_db, hyps_file(tmp_path, "two", nc=None, new_ids=("H_TEST_TWO",)))[0] == 1
    assert register_now(syn_db, hyps_file(tmp_path, "three", nc=None, new_ids=("H_TEST_THREE",)))[0] == 1


def test_a_dirty_tree_is_refused_before_the_flag_is_read(syn_db, monkeypatch):
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    monkeypatch.setattr(H, "git_commit", lambda: DIRTY)
    with pytest.raises(H.DirtyTree):
        register_now(syn_db, HYPS)
    assert written(syn_db) == (0, 0)


def test_quant_freeze_refuses_while_a_flag_is_open(syn_db, monkeypatch):
    from src import cli
    from src.cli import app

    calls = []
    monkeypatch.setattr(cli, "_open", lambda sport: (load_sport_config("mlb"), syn_db))
    monkeypatch.setattr(H, "freeze", lambda *a, **k: calls.append(a) or {"manifest_hash": "h"})
    monkeypatch.setattr(H, "current_manifest", lambda *a, **k: {})
    monkeypatch.setattr(cli, "_selection", lambda db: None)
    RF.record_review(syn_db, review("bt-a", DIRTY, D.FAIL))
    res = CliRunner().invoke(app, ["freeze", "--sport", "mlb"])
    assert res.exit_code == 1 and "ReviewBlocked" in res.output and "bt-a" in res.output
    assert calls == []
    RF.record_review(syn_db, review("bt-c", CLEAN, D.PASS, at=T0 + timedelta(hours=1)))
    res = CliRunner().invoke(app, ["freeze", "--sport", "mlb"])
    assert res.exit_code == 0 and len(calls) == 1


# -- quant review records and shows the flag -------------------------------------------------------

def test_quant_review_records_the_flag_and_shows_it(syn_db, monkeypatch, tmp_path):
    from src import cli
    from src.cli import app

    rev = review("bt-a", DIRTY, D.FAIL) | {"status": "INCONCLUSIVE", "status_reasons": ["r"]}
    seen = {}
    monkeypatch.setattr(cli, "_open", lambda sport: (load_sport_config("mlb"), syn_db))
    monkeypatch.setattr(cli, "_review_inputs", lambda db, cfg: type("I", (), {
        "experiments": None, "hypotheses": None, "holdout_manifest": {}})())
    monkeypatch.setattr(D, "build_review", lambda inputs, cfg: rev)
    monkeypatch.setattr(D, "write_review_artifacts", lambda r, out, **k: seen.setdefault("written", r))
    monkeypatch.setattr(D, "print_review", lambda r, console: seen.setdefault("printed", r))
    res = CliRunner().invoke(app, ["review", "--sport", "mlb", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert open_ids(syn_db) == ["bt-a"]
    for key in ("written", "printed"):
        rf = seen[key]["review_flags"]
        assert rf["event"] == RF.FLAG_RAISE and [o["backtest_run_id"] for o in rf["open"]] == ["bt-a"]


def test_the_review_text_lists_the_flags():
    rf = {"event": RF.FLAG_RAISE, "recorded": [{"flag_id": "rf-1", "event": "raise", "backtest_run_id": "bt-a"}],
          "open": [{"flag_id": "rf-1", "backtest_run_id": "bt-a", "git_commit": DIRTY,
                    "run_created_at": T0.isoformat()}]}
    md = D.render_markdown(_minimal_review() | {"review_flags": rf})
    assert "## Review flags (ADR 0002)" in md and "rf-1" in md and "bt-a" in md
    assert md.rstrip("\n").splitlines()[-1].startswith("STATUS: ")
    assert "Review flags" not in D.render_markdown(_minimal_review())


def _minimal_review() -> dict:
    return {"sport": "mlb", "dataset": "synthetic", "generated_at": "t", "baseline_scenario": "pen_25",
            "holdout_period": {"period_id": "h", "start": None, "end": None, "note": "n"},
            "criteria": [{"id": "26.1", "title": "t", "state": D.PASS, "value": None, "detail": ""}],
            "status_reasons": ["r"], "supporting": {}, "informational": {}, "status": "INCONCLUSIVE"}
