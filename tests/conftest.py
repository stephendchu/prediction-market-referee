"""Shared fixtures: a pinned git commit and throwaway databases (row builders live in factories.py)."""

from __future__ import annotations

import pytest

from src.backtest import holdout
from src.storage.database import connect

PINNED_COMMIT = "7e57" * 10


@pytest.fixture(autouse=True, scope="session")
def _pinned_git_commit():
    """Every test sees one clean commit, never this working tree's HEAD or whether it is dirty.

    A freeze and both registration steps refuse a dirty tree, so without the pin a
    test that registers would pass or fail with the state of the checkout it runs in. A test that
    needs another commit patches holdout.git_commit itself; tests/unit/test_dirty_tree.py tests the
    real one.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(holdout, "git_commit", lambda: PINNED_COMMIT)
        yield


@pytest.fixture
def syn_db(tmp_path):
    db = connect("mlb", "synthetic", path=":memory:", logs=tmp_path / "logs")
    yield db
    db.close()


@pytest.fixture
def live_db(tmp_path):
    db = connect("mlb", "live", path=":memory:", logs=tmp_path / "logs")
    yield db
    db.close()
