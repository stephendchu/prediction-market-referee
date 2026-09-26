"""Which paper decisions qualify for 26.18.

A decision carries the frozen manifest's hash only under a FROZEN manifest whose fields all match
the current configuration except `data_version`, and a model trained no later than the freeze.
The manifests here are hand-built `live` ones frozen into tmp_path; no real manifest is touched.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import holdout as H
from src.backtest.execution import fee_schedule_rows
from src.backtest.walk_forward import holdout_period, load_periods
from src.config import load_sport_config
from src.portfolio import paper as P
from tests.factories import locked_registration

FREEZE_AT = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
DB = SimpleNamespace(sport="mlb")
SELECTION = {"selected": "logistic_residual"}


def manifest(cfg, **over):
    periods = load_periods(cfg, "live")
    fields = dict(sport="mlb", dataset="live", periods=periods, holdout=holdout_period(periods),
                  model_type="logistic_residual", model_config_hash="cfg-1", model_artifact_hash="sha256:a1",
                  quote_quality="candle_degraded", feature_set_hash="fs-1", data_version="dv-1",
                  thresholds_hash=H.thresholds_hash(cfg),
                  fee_schedule_ids=fee_schedule_rows()["fee_schedule_id"].tolist(),
                  holdout_registration=locked_registration(holdout_period(periods)), git_commit="abc1234",
                  now=FREEZE_AT)
    return H.build_manifest(**(fields | over))


def slate_model(created_at: datetime) -> P.SlateModel:
    return P.SlateModel({"model_run_id": "mr-test", "created_at": pd.Timestamp(created_at)}, None)


@pytest.fixture
def qualify(tmp_path, monkeypatch):
    """qualification() against a manifest frozen in tmp_path and a given current manifest."""
    cfg = load_sport_config("mlb")
    path = tmp_path / "mlb_holdout_freeze.json"
    monkeypatch.setattr(P, "manifest_path", lambda sport: path)

    def run(current: dict, *, frozen: dict | None = None, hand_frozen: dict | None = None,
            trained_at: datetime = FREEZE_AT - timedelta(days=1)):
        if frozen is not None:
            H.freeze(path, frozen, now=FREEZE_AT)
        if hand_frozen is not None:  # one freeze() refuses, written by hand or by an older version
            m = hand_frozen | {"status": "FROZEN", "manifest_hash": H.manifest_hash(hand_frozen)}
            path.write_text(json.dumps(m))
        monkeypatch.setattr(P, "current_manifest", lambda db, cfg, selection, now: current)
        return P.qualification(DB, cfg, SELECTION, slate_model(trained_at), FREEZE_AT + timedelta(days=10))

    run.cfg = cfg
    return run


def test_exact_match_qualifies(qualify):
    frozen = manifest(qualify.cfg)
    q = qualify(manifest(qualify.cfg), frozen=frozen)
    assert q.manifest_hash == H.manifest_hash(frozen) and q.reasons == []


def test_new_data_alone_still_qualifies(qualify):
    """Paper trading adds data by design: data_version is the one field allowed to move."""
    frozen = manifest(qualify.cfg)
    q = qualify(manifest(qualify.cfg, data_version="dv-2"), frozen=frozen)
    assert q.status == "FROZEN" and q.manifest_hash == H.manifest_hash(frozen)


@pytest.mark.parametrize("field, value", [
    ("thresholds_hash", "changed"), ("model_config_hash", "cfg-2"), ("feature_set_hash", "fs-2"),
    ("model_type", "lightgbm_residual"), ("git_commit", "def5678"), ("fee_schedule_ids", ["other"]),
    ("model_artifact_hash", "sha256:a2"), ("quote_quality", "executable"),
])
def test_any_other_change_disqualifies(qualify, field, value):
    q = qualify(manifest(qualify.cfg, **{field: value}), frozen=manifest(qualify.cfg))
    assert q.manifest_hash is None and field in q.reasons


def test_model_trained_after_the_freeze_disqualifies(qualify):
    q = qualify(manifest(qualify.cfg), frozen=manifest(qualify.cfg), trained_at=FREEZE_AT + timedelta(seconds=1))
    assert q.manifest_hash is None and q.reasons == ["model mr-test was trained after the freeze"]


def test_no_manifest_or_no_commit_disqualifies(qualify):
    q = qualify(manifest(qualify.cfg))
    assert (q.status, q.manifest_hash) == ("LOCKED", None)
    unpinned = manifest(qualify.cfg, git_commit=None)
    q = qualify(unpinned, hand_frozen=unpinned)
    assert q.manifest_hash is None and any(r.startswith("git_commit") for r in q.reasons)
