"""Code that is not committed: a run from it is recorded as `<commit>-dirty`, and it
is never registered or frozen.

git_commit() is tested on throwaway repositories. The rest of the suite runs with it pinned to a
clean commit (tests/conftest.py), so that no test depends on the state of this checkout.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src import config
from src.backtest import holdout as H
from src.backtest.registry import register_now
from src.config import CONFIG_DIR

REAL_GIT_COMMIT = H.git_commit  # bound at import, before conftest pins it for the session
DIRTY = "abc1234" + "0" * 33 + "-dirty"
NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


def git(root, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", *args],
                          cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with one commit: a tracked file and an ignored directory. REPO_ROOT points at it."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "tracked.py").write_text("x = 1\n")
    (root / ".gitignore").write_text("ignored/\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "one")
    monkeypatch.setattr(config, "REPO_ROOT", root)
    return SimpleNamespace(root=root, head=git(root, "rev-parse", "HEAD"))


def test_the_function_under_test_is_the_real_one():
    assert REAL_GIT_COMMIT.__module__ == "src.backtest.holdout" and REAL_GIT_COMMIT.__name__ == "git_commit"


# ------------------------------------------------------------------ recording: git_commit()

def test_a_clean_tree_is_its_commit(repo):
    assert REAL_GIT_COMMIT() == repo.head and len(repo.head) == 40


@pytest.mark.parametrize("change", ["edit", "stage", "delete"])
def test_a_change_to_a_tracked_file_tags_the_commit_dirty(repo, change):
    f = repo.root / "tracked.py"
    if change == "delete":
        f.unlink()
    else:
        f.write_text("x = 2\n")
        if change == "stage":
            git(repo.root, "add", "tracked.py")
    assert REAL_GIT_COMMIT() == repo.head + "-dirty"


def test_untracked_and_ignored_files_do_not_count(repo):
    """As `git describe --dirty`: a new file changes a run only through a tracked file that uses it."""
    (repo.root / "new_module.py").write_text("y = 1\n")
    (repo.root / "ignored").mkdir()
    (repo.root / "ignored" / "rows.parquet").write_text("")
    assert REAL_GIT_COMMIT() == repo.head


def test_a_touched_but_unchanged_file_is_clean(repo):
    """A new mtime with the same content is not a change (the index is refreshed before comparing)."""
    f = repo.root / "tracked.py"
    t = f.stat().st_mtime + 100
    os.utime(f, (t, t))
    assert REAL_GIT_COMMIT() == repo.head


@pytest.mark.parametrize("failure", ["exit", "raise"])
def test_a_tree_whose_state_is_unknown_counts_as_dirty(repo, monkeypatch, failure):
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if "status" in cmd:
            if failure == "raise":
                raise subprocess.TimeoutExpired(cmd, 5)
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: index file corrupt")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    assert REAL_GIT_COMMIT() == repo.head + "-dirty"


def test_no_repository_or_no_commit_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(config, "REPO_ROOT", plain)
    assert REAL_GIT_COMMIT() is None
    git(plain, "init", "-q")
    assert REAL_GIT_COMMIT() is None


# ------------------------------------------------------------------ refusals: hypothesis registration

def refused(exc: pytest.ExceptionInfo) -> None:
    assert exc.type.__name__ == "DirtyTree"
    assert "ADR 0001 Q3" in str(exc.value)


@pytest.mark.parametrize("commit", [DIRTY, None], ids=["dirty", "no-commit"])
def test_hypotheses_are_not_registered_from_uncommitted_code(syn_db, monkeypatch, commit):
    monkeypatch.setattr(H, "git_commit", lambda: commit)
    with pytest.raises(RuntimeError, match="clean tree") as exc:
        register_now(syn_db, CONFIG_DIR / "mlb_hypotheses.yaml")
    refused(exc)
    assert len(syn_db.df("SELECT * FROM hypotheses")) == 0
    assert len(syn_db.df("SELECT * FROM hypothesis_status")) == 0


def test_hypotheses_register_from_a_clean_tree(syn_db, monkeypatch):
    monkeypatch.setattr(H, "git_commit", lambda: "abc1234" + "0" * 33)
    n, _ = register_now(syn_db, CONFIG_DIR / "mlb_hypotheses.yaml")
    assert n > 0
    assert set(syn_db.df("SELECT git_commit FROM hypotheses")["git_commit"]) == {"abc1234" + "0" * 33}


# ---- `quant train` on a dirty tree: it trains (tagged -dirty) but refuses the registration step, loudly.

TRAIN_DAY = datetime(2026, 4, 6, 12, tzinfo=timezone.utc)


def _small_season() -> dict:
    from datetime import date
    cfg = config.load_sport_config("mlb")
    cfg["synthetic"] |= {"seasons": [2026], "through_date": "2026-04-06"}
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
def dirty_train(tmp_path_factory):
    """ingest -> features -> train on one small synthetic season, from a dirty tree."""
    import warnings
    from datetime import date

    from src.features.mlb import write_features
    from src.models.train import run_training
    from src.sports.synthetic import SyntheticSource
    from src.storage.database import connect
    from src.storage.ingest import ingest

    root = tmp_path_factory.mktemp("dirty_train")
    cfg = _small_season()
    lines: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QUANT_DATA_DIR", str(root / "data"))
        mp.setattr(H, "git_commit", lambda: DIRTY)
        db = connect("mlb", "synthetic", path=root / "world.duckdb", logs=root / "logs")
        try:
            ingest(db, SyntheticSource(cfg), date(2026, 1, 1), date(2026, 4, 6))
            write_features(db, cfg, date(2026, 4, 6))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res = run_training(db, cfg, now=TRAIN_DAY, threads=2, log=lines.append)
            out = SimpleNamespace(
                res=res, log=lines, hypotheses=db.df("SELECT * FROM hypotheses"),
                status=db.df("SELECT * FROM hypothesis_status"),
                experiments=db.df("SELECT git_commit FROM experiments"),
                model_runs=db.df("SELECT git_commit FROM model_runs"))
        finally:
            db.close()
    return out


def test_dirty_train_trains_and_tags_its_rows_dirty(dirty_train):
    assert len(dirty_train.model_runs) > 0 and set(dirty_train.model_runs["git_commit"]) == {DIRTY}
    assert len(dirty_train.experiments) > 0 and set(dirty_train.experiments["git_commit"]) == {DIRTY}


def test_dirty_train_registers_no_hypothesis_and_no_status(dirty_train):
    assert len(dirty_train.hypotheses) == 0, "a dirty train registered hypotheses (ADR 0001 Q3)"
    assert len(dirty_train.status) == 0
    assert dirty_train.res.hypotheses_registered == 0


def test_dirty_train_says_it_refused_the_registration(dirty_train):
    """Not a silent skip: the refusal is printed while training and kept in the run's notes, and it
    names what was not registered."""
    said = [m for m in dirty_train.log if "registration refused" in m]
    noted = [n for n in dirty_train.res.notes if "registration refused" in n]
    assert said and noted and said[0] == noted[0]
    assert "ADR 0001 Q3" in said[0] and DIRTY in said[0]
    from src.backtest.registry import load_hypotheses
    registered_ids = [h["hypothesis_id"] for h in load_hypotheses(CONFIG_DIR / "mlb_hypotheses.yaml")]
    assert all(h in said[0] for h in registered_ids), said[0]


def test_dirty_train_still_refuses_an_edited_definition(syn_db, monkeypatch, tmp_path):
    """The refusal skips only the writes: the hypotheses file is still validated against the registry,
    so an edited definition still stops the run, from any tree."""
    import yaml

    from src.backtest.registry import HypothesisChanged, load_hypotheses, register_hypotheses
    from src.models.train import run_training
    hyps = load_hypotheses(CONFIG_DIR / "mlb_hypotheses.yaml")
    register_hypotheses(syn_db, hyps, git_commit="abc1234" + "0" * 33, now=NOW)
    raw = yaml.safe_load((CONFIG_DIR / "mlb_hypotheses.yaml").read_text())
    first = raw["hypotheses"][0]
    first["description"] = str(first["description"]) + " (edited)"
    edited = tmp_path / "edited.yaml"
    edited.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(H, "git_commit", lambda: DIRTY)
    n_before = len(syn_db.df("SELECT * FROM hypotheses"))
    with pytest.raises(HypothesisChanged):
        run_training(syn_db, _small_season(), hyps_path=edited, now=NOW)
    assert len(syn_db.df("SELECT * FROM hypotheses")) == n_before


# ---- a commit that is missing, NaN, blank or not a bare string is not clean (fail-closed)

NOT_CLEAN = [None, float("nan"), pd.NA, pd.NaT, "", " ", "\t\n", "　", DIRTY, " " + DIRTY + " ", DIRTY + "\n",
             H.GIT_UNAVAILABLE, " " + H.GIT_UNAVAILABLE, "abc1234 ", " abc1234", " abc1234", 0, 1.5,
             b"abc1234", ["abc1234"]]
NOT_CLEAN_IDS = ["none", "nan", "pd.NA", "NaT", "empty", "space", "tab-newline", "ideographic-space", "dirty",
                 "padded-dirty", "dirty-newline", "unavailable", "padded-unavailable", "trailing-space",
                 "leading-space", "nbsp", "int", "float", "bytes", "list"]
CLEAN_COMMITS = ["abc1234" + "0" * 33, "7e57" * 10, "a"]


@pytest.mark.parametrize("commit", NOT_CLEAN, ids=NOT_CLEAN_IDS)
def test_a_missing_blank_padded_or_non_string_commit_is_not_clean(commit):
    assert H.is_clean_commit(commit) is False
    assert isinstance(H.tree_state(commit), str) and H.tree_state(commit)


@pytest.mark.parametrize("commit", CLEAN_COMMITS)
def test_a_bare_commit_id_is_clean(commit):
    assert H.is_clean_commit(commit) is True


@pytest.mark.parametrize("commit", [None, float("nan"), pd.NA, "", "  ", "\t"], ids=["none", "nan", "pd.NA", "empty",
                                                                                       "blank", "tab"])
def test_a_missing_or_blank_commit_reads_as_no_commit(commit):
    assert H.tree_state(commit).startswith("no git commit")


def test_the_sql_mirror_agrees_with_is_clean_commit():
    """schemas._NOT_CLEAN (the review_flags CHECKs) is is_clean_commit in SQL: the same answer for every
    string value, and NULL is not clean."""
    import duckdb

    from src.storage.schemas import _NOT_CLEAN
    con = duckdb.connect()
    values = [c for c in NOT_CLEAN if isinstance(c, str)] + CLEAN_COMMITS + [None]
    for v in values:
        not_clean = con.execute("SELECT " + _NOT_CLEAN + " FROM (SELECT CAST(? AS VARCHAR) AS git_commit)",
                                [v]).fetchone()[0]
        assert not_clean is (not H.is_clean_commit(v)), repr(v)


@pytest.mark.parametrize("commit", [float("nan"), pd.NA, "  "], ids=["nan", "pd.NA", "blank"])
def test_require_clean_tree_refuses_a_nan_or_blank_commit(monkeypatch, commit):
    monkeypatch.setattr(H, "git_commit", lambda: commit)
    with pytest.raises(H.DirtyTree):
        H.require_clean_tree("register a holdout")


@pytest.mark.parametrize("commit", [float("nan"), pd.NA, "  ", " " + "abc1234" + "0" * 33],
                         ids=["nan", "pd.NA", "blank", "padded"])
def test_a_freeze_refuses_a_nan_or_blank_manifest_commit(tmp_path, commit):
    manifest = {k: "x" for k in H.FROZEN_FIELDS} | {"dataset": "live", "git_commit": commit}
    path = tmp_path / "freeze.json"
    with pytest.raises(H.DirtyTree):
        H.freeze(path, manifest)
    assert not path.exists()
