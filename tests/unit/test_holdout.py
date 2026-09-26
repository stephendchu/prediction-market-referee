"""FINAL_HOLDOUT freeze manifest (26.3): LOCKED -> FROZEN -> UNLOCKED, never re-frozen."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.backtest import holdout as H
from src.backtest.walk_forward import HoldoutLocked, definition_hash, holdout_period, load_periods
from tests.factories import locked_registration

CFG = {"walk_forward": {"live_periods": [
    {"id": "P1", "role": "train_only", "start": "2023-03-01", "end": "2023-06-30"},
    {"id": "P2", "role": "evaluation", "start": "2023-07-01", "end": "2023-11-30"},
    {"id": "P3", "role": "holdout", "start": "2024-03-01", "end": "2024-06-30"},
]}}
PERIODS = load_periods(CFG, "live")
NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def manifest(**over):
    args = dict(sport="mlb", dataset="live", periods=PERIODS, holdout=holdout_period(PERIODS),
                model_type="logistic_residual", model_config_hash="m1", model_artifact_hash="sha256:a1",
                quote_quality="candle_degraded", feature_set_hash="f1", data_version="d1", thresholds_hash="t1",
                fee_schedule_ids=["quadratic-taker-v1"], holdout_registration=locked_registration(PERIODS[-1]),
                git_commit="abc123", now=NOW)
    return H.build_manifest(**(args | over))


def test_build_manifest_records_what_26_3_requires():
    m = manifest(git_commit=None)
    assert m["status"] == "LOCKED"
    assert m["git_commit"] == "unavailable: not a git repository"
    assert m["freeze_timestamp"] == "2026-09-24T12:00:00+00:00"
    assert m["period_definition_hash"] == definition_hash(PERIODS)
    assert m["holdout"] == {"period_id": "P3", "start": "2024-03-01", "end": "2024-06-30"}
    assert (m["model_config_hash"], m["feature_set_hash"], m["data_version"], m["thresholds_hash"]) == (
        "m1", "f1", "d1", "t1")
    assert m["holdout_evaluated"] is False
    assert m["manifest_hash"] == H.manifest_hash(m)


def test_holdout_must_be_the_final_holdout_period():
    with pytest.raises(ValueError, match="not the final holdout"):
        manifest(holdout=PERIODS[1])


def test_manifest_hash_ignores_the_timestamp_but_not_the_config():
    assert H.manifest_hash(manifest()) == H.manifest_hash(manifest(now=datetime(2027, 1, 1, tzinfo=timezone.utc)))
    assert H.manifest_hash(manifest()) != H.manifest_hash(manifest(thresholds_hash="t2"))


def test_thresholds_hash_tracks_decision_settings():
    cfg = {"decision": {"minimum_edge": 0.04}, "model": {"l2": 1.0}}
    assert H.thresholds_hash(cfg) != H.thresholds_hash({**cfg, "decision": {"minimum_edge": 0.03}})
    assert H.thresholds_hash(cfg) == H.thresholds_hash({**cfg, "model": {"l2": 2.0}})  # model has its own hash


def test_freeze_writes_frozen_manifest_and_history(tmp_path):
    path = tmp_path / "holdout_freeze.json"
    frozen = H.freeze(path, manifest(), now=NOW)
    assert frozen["status"] == "FROZEN"
    assert json.loads(path.read_text()) == frozen
    events = [json.loads(line) for line in H.history_path(path).read_text().splitlines()]
    assert [e["event"] for e in events] == ["freeze"]
    assert events[0]["manifest"]["manifest_hash"] == frozen["manifest_hash"]
    assert H.history_path(path).name == "holdout_freeze.history.jsonl"


def test_a_frozen_manifest_is_never_overwritten(tmp_path):
    path = tmp_path / "holdout_freeze.json"
    H.freeze(path, manifest())
    before = path.read_text()
    with pytest.raises(H.AlreadyFrozen):
        H.freeze(path, manifest(feature_set_hash="f2"))  # e.g. retrying after a failed holdout
    assert path.read_text() == before
    events = [json.loads(line) for line in H.history_path(path).read_text().splitlines()]
    assert [e["event"] for e in events] == ["freeze", "freeze_refused"]


def test_check_unlock_passes_only_on_an_exact_match(tmp_path):
    frozen = H.freeze(tmp_path / "m.json", manifest())
    H.check_unlock(frozen, manifest(now=datetime(2027, 1, 1, tzinfo=timezone.utc)))  # no raise
    with pytest.raises(HoldoutLocked) as exc:
        H.check_unlock(frozen, manifest(feature_set_hash="f2", data_version="d2"))
    assert exc.value.mismatches == ["feature_set_hash", "data_version"]


def test_an_unfrozen_manifest_cannot_unlock():
    with pytest.raises(HoldoutLocked, match="LOCKED") as exc:
        H.check_unlock(manifest(), manifest())
    assert exc.value.mismatches == ["status"]


def hand_frozen(m):
    """A FROZEN manifest that freeze() would refuse, written by hand or by an older version."""
    return {**m, "status": "FROZEN", "manifest_hash": H.manifest_hash(m)}


@pytest.mark.parametrize(("commit", "reason"), [
    (None, "git_commit (unavailable: code version not pinned)"),
    ("abc123-dirty", "git_commit (abc123-dirty: uncommitted code)"),
], ids=["no-commit", "dirty"])
def test_code_that_is_not_a_clean_commit_cannot_unlock(commit, reason):
    with pytest.raises(HoldoutLocked) as exc:
        H.check_unlock(hand_frozen(manifest(git_commit=commit)), manifest(git_commit=commit))
    assert exc.value.mismatches == [reason]


@pytest.mark.parametrize("commit", ["abc123-dirty", None], ids=["dirty", "no-commit"])
def test_a_freeze_requires_a_clean_tree(tmp_path, commit):
    """A manifest with no commit records GIT_UNAVAILABLE, which is not clean either."""
    path = tmp_path / "m.json"
    with pytest.raises(RuntimeError, match="clean tree") as exc:
        H.freeze(path, manifest(git_commit=commit))
    assert exc.type.__name__ == "DirtyTree" and "clean tree" in str(exc.value)
    assert not path.exists() and not H.history_path(path).exists()


def test_a_frozen_manifest_edited_by_hand_cannot_unlock(tmp_path):
    frozen = H.freeze(tmp_path / "m.json", manifest())
    tampered = {**frozen, "thresholds_hash": "t2"}
    with pytest.raises(HoldoutLocked) as exc:
        H.check_unlock(tampered, manifest(thresholds_hash="t2"))
    assert exc.value.mismatches == ["manifest_hash (frozen manifest edited after freeze)"]


def test_status_without_a_freeze_is_locked(tmp_path):
    status = H.manifest_status(tmp_path / "missing.json", manifest())
    assert status["status"] == "LOCKED"
    assert status["note"] == "holdout not evaluated in this engagement"
    assert status["holdout_evaluated"] is False
    assert status["frozen"] is None
    assert status["current"]["feature_set_hash"] == "f1"
    assert status["current"]["period_definition_hash"] == definition_hash(PERIODS)


def test_status_after_freeze(tmp_path):
    path = tmp_path / "m.json"
    H.freeze(path, manifest())
    match = H.manifest_status(path, manifest())
    assert (match["status"], match["mismatches"], match["holdout_evaluated"]) == ("UNLOCKED", [], False)
    drift = H.manifest_status(path, manifest(model_type="lgbm_residual"))
    assert (drift["status"], drift["mismatches"]) == ("FROZEN", ["model_type"])
    json.dumps(drift)  # the review artifact must be JSON-serializable


def test_synthetic_dataset_is_never_frozen(tmp_path):
    path = tmp_path / "m.json"
    with pytest.raises(H.SyntheticFreeze):
        H.freeze(path, manifest(dataset="synthetic"))
    assert not path.exists() and not H.history_path(path).exists()
