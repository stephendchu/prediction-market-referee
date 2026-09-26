"""Hypothesis registry (26.4) and experiment registry (Sec 20, 26.5)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import yaml

from src.backtest.registry import (
    HypothesisChanged,
    current_statuses,
    definition_hash,
    is_active,
    experiment_id,
    hypothesis_rows,
    load_hypotheses,
    record_experiments,
    register_hypotheses,
    validate_hypothesis,
)
from src.models.train import feature_sets
from src.storage.schemas import HYPOTHESES

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def hyp(hid="H_MLB_TEST_1", **over) -> dict:
    row = {
        "hypothesis_id": hid,
        "family": "starter",
        "description": "Starter K-BB% gap is underpriced",
        "mechanism": "Casual flow anchors on W-L record rather than strikeout/walk skill",
        "features": [{"name": "starter_k_minus_bb_diff", "definition": "home - away K-BB% over last 5 starts"}],
        "expected_direction": "positive",
        "minimum_sample": 500,
        "evaluation_periods": ["2023H2", "2024H1"],
        "primary_metric": "delta_brier",
        "label": "CONFIRMATORY",
    }
    return row | over


def write_yaml(tmp_path, hyps) -> str:
    path = tmp_path / "hyps.yaml"
    path.write_text(yaml.safe_dump({"hypotheses": hyps}))
    return path


def experiment(**over) -> dict:
    metrics = {"reference": "market", "candidate": "logistic_residual", "n_obs": 1200,
               "brier_ref": 0.2400, "brier_cand": 0.2390, "delta_brier": 0.2400 - 0.2390}
    row = {"hypothesis_id": "H_MLB_TEST_1", "experiment_family": "starter", "label": "CONFIRMATORY",
           "feature_set_id": "fs-starter", "feature_set_hash": "fsh1", "model_type": "logistic_residual",
           "config_hash": "cfg1", "train_period": "2023H1", "test_period": "2023H2", "n_games": 600,
           "metrics_json": json.dumps(metrics), "p_value": 0.21, "quote_quality": "executable", "created_at": NOW}
    return row | over


# -- hypotheses -------------------------------------------------------------------

def test_load_hypotheses_normalizes_features(tmp_path):
    mapping_features = hyp("H2", features={"park_run_factor": "runs per game at venue, prior seasons"})
    hyps = load_hypotheses(write_yaml(tmp_path, [hyp(), mapping_features]))
    assert [h["hypothesis_id"] for h in hyps] == ["H_MLB_TEST_1", "H2"]
    assert hyps[1]["features"] == [{"name": "park_run_factor", "definition": "runs per game at venue, prior seasons"}]


@pytest.mark.parametrize("bad,match", [
    (hyp(mechanism=""), "missing"),
    ({k: v for k, v in hyp().items() if k != "primary_metric"}, "missing"),
    (hyp(label="CONFIRMED"), "label"),
    (hyp(minimum_sample="lots"), "minimum_sample"),
    (hyp(features=[{"name": "x"}]), "exact definition"),
])
def test_load_hypotheses_rejects_incomplete_definitions(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        load_hypotheses(write_yaml(tmp_path, [bad]))


def test_load_hypotheses_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_hypotheses(write_yaml(tmp_path, [hyp(), hyp(description="other")]))


def test_hypothesis_rows_match_the_table():
    rows = hypothesis_rows([hyp()], dataset="synthetic", is_synthetic=True, git_commit=None, now=NOW)
    assert list(rows.columns) == HYPOTHESES.column_names
    assert json.loads(rows.iloc[0]["feature_definitions_json"])[0]["name"] == "starter_k_minus_bb_diff"
    later = hypothesis_rows([hyp()], dataset="synthetic", is_synthetic=True, git_commit="abc", now=datetime.now(timezone.utc))
    assert rows.iloc[0]["definition_hash"] == later.iloc[0]["definition_hash"]  # not timestamps or commits


def test_definition_hash_covers_every_definition_field():
    base = definition_hash(hyp())
    for change in ({"mechanism": "other"}, {"label": "EXPLORATORY"}, {"minimum_sample": 200},
                   {"evaluation_periods": ["2023H2"]},
                   {"features": [{"name": "starter_k_minus_bb_diff", "definition": "last 10 starts"}]}):
        assert definition_hash(hyp(**change)) != base, change


def test_register_hypotheses_is_idempotent(syn_db):
    assert register_hypotheses(syn_db, [hyp(), hyp("H2")], now=NOW) == 2
    assert register_hypotheses(syn_db, [hyp(), hyp("H2")], now=NOW) == 0
    assert register_hypotheses(syn_db, [hyp(), hyp("H3")], git_commit="abc", now=NOW) == 1
    stored = syn_db.df("SELECT hypothesis_id, git_commit, is_synthetic FROM hypotheses ORDER BY hypothesis_id")
    assert list(stored["hypothesis_id"]) == ["H2", "H3", "H_MLB_TEST_1"]
    assert stored["is_synthetic"].all()


def test_a_registered_hypothesis_cannot_be_redefined(syn_db):
    register_hypotheses(syn_db, [hyp()], now=NOW)
    with pytest.raises(HypothesisChanged, match="new id"):
        register_hypotheses(syn_db, [hyp("H_NEW"), hyp(label="EXPLORATORY")], now=NOW)
    assert syn_db.count("hypotheses") == 1  # all-or-nothing: H_NEW was not written either


DISABLE = {"status": "DISABLED", "status_reason": "built ahead of scope",
           "status_decision": "docs/decisions/README.md"}


def test_status_is_not_part_of_the_definition():
    assert definition_hash(hyp(**DISABLE)) == definition_hash(hyp())


@pytest.mark.parametrize("missing", ["status_reason", "status_decision"])
def test_a_non_active_status_needs_a_reason_and_a_decision(tmp_path, missing):
    bad = {k: v for k, v in DISABLE.items() if k != missing}
    with pytest.raises(ValueError, match="needs status_reason and status_decision"):
        load_hypotheses(write_yaml(tmp_path, [hyp(**bad)]))
    with pytest.raises(ValueError, match="status must be one of"):
        load_hypotheses(write_yaml(tmp_path, [hyp(status="RETIRED")]))


def test_disabling_appends_a_status_row_and_keeps_the_hypothesis(syn_db):
    later = NOW.replace(hour=13)
    register_hypotheses(syn_db, [hyp(), hyp("H2")], now=NOW)
    assert syn_db.count("hypothesis_status") == 0          # ACTIVE since registration, no row needed
    assert register_hypotheses(syn_db, [hyp(**DISABLE), hyp("H2")], now=later) == 0
    assert syn_db.count("hypotheses") == 2                 # still registered, definition unchanged
    st = current_statuses(syn_db)
    assert st["H_MLB_TEST_1"]["status"] == "DISABLED" and st["H_MLB_TEST_1"]["reason"] == "built ahead of scope"
    assert st["H2"]["status"] == "ACTIVE"
    register_hypotheses(syn_db, [hyp(**DISABLE), hyp("H2")], now=later.replace(hour=14))
    assert syn_db.count("hypothesis_status") == 1          # unchanged status: nothing appended
    register_hypotheses(syn_db, [hyp(), hyp("H2")], now=later.replace(hour=15))
    assert current_statuses(syn_db)["H_MLB_TEST_1"]["status"] == "ACTIVE"
    assert syn_db.count("hypothesis_status") == 2          # the history is kept


def test_a_disabled_hypothesis_feeds_no_model_or_experiment():
    families = {"starter_k_minus_bb_diff": "starter", "gap": "market", "nc_random": "negative_control"}
    expl = hyp("H_EXPL", family="market", label="EXPLORATORY", features=[{"name": "gap", "definition": "d"}])
    on = feature_sets(families, [validate_hypothesis(hyp()), validate_hypothesis(expl)])
    assert on.model == ["starter_k_minus_bb_diff"] and on.exploratory == {"H_EXPL": ["gap"]}
    off = feature_sets(families, [validate_hypothesis(hyp(**DISABLE)), validate_hypothesis(expl | DISABLE)])
    assert off.model == [] and off.exploratory == {} and off.negative_controls == []
    assert off.disabled == {"H_MLB_TEST_1": ["starter_k_minus_bb_diff"], "H_EXPL": ["gap"]}
    assert not is_active(validate_hypothesis(hyp(**DISABLE))) and is_active(validate_hypothesis(hyp()))


def test_the_mlb_registry_keeps_h3_registered_but_disabled():
    hyps = {h["hypothesis_id"]: h for h in load_hypotheses("config/mlb_hypotheses.yaml")}
    h3 = hyps["H_MLB_003"]
    assert h3["label"] == "EXPLORATORY" and h3["status"] == "DISABLED"
    assert "out of scope" in h3["status_reason"] and h3["status_decision"] == "docs/decisions/README.md"
    assert [h for h, v in hyps.items() if not is_active(v)] == ["H_MLB_003"]



def test_the_log_odds_gap_is_in_the_market_only_model_and_nothing_else_changed():
    """cross_market_gap in log-odds enters the residual model's X under a new
    CONFIRMATORY id, since a registered definition is never edited; the probability form stays in the
    DISABLED H_MLB_003, and H_MLB_L1_MARKET keeps its two features."""
    from src.features.mlb import feature_families
    hyps = {h["hypothesis_id"]: h for h in load_hypotheses("config/mlb_hypotheses.yaml")}
    g = hyps["H_MLB_L1_GAP"]
    assert (g["label"], g["family"], is_active(g)) == ("CONFIRMATORY", "market", True)
    assert [f["name"] for f in g["features"]] == ["cross_market_gap_log_odds"]
    assert [f["name"] for f in hyps["H_MLB_L1_MARKET"]["features"]] == ["market_logit", "market_move_60m"]
    fs = feature_sets(feature_families(), list(hyps.values()), ["market"])
    assert "cross_market_gap_log_odds" in fs.model and "cross_market_gap" not in fs.model
    assert fs.model == sorted(["book_move_since_news", "is_afternoon_game", "market_move_since_news", "latency_gap",
                               "log_open_interest", "log_volume", "market_logit", "market_move_60m", "market_spread",
                               "minutes_since_last_news", "cross_market_gap_log_odds"])

# -- experiments ------------------------------------------------------------------

def test_experiment_id_is_deterministic():
    fields = dict(experiment_family="starter", hypothesis_id="H1", feature_set_hash="fsh1",
                  model_type="logistic_residual", config_hash="cfg1", train_period="2023H1",
                  test_period="2023H2", dataset="synthetic", quote_quality="executable", created_at=NOW)
    eid = experiment_id(**fields)
    assert eid == experiment_id(**fields, metrics_json="{}", p_value=0.5)  # non-key fields ignored
    assert eid == experiment_id(**{**fields, "family": "starter"})
    assert eid != experiment_id(**{**fields, "test_period": "2024H1"})
    assert eid != experiment_id(**{**fields, "quote_quality": "candle_degraded"})
    assert eid.startswith("exp-")
    with pytest.raises(ValueError, match="config_hash"):
        experiment_id(**{k: v for k, v in fields.items() if k != "config_hash"})



# An experiment id takes the commit and the invocation time, as a model run id does
# (ADR 0002 item 4), so a clean retrain never reuses the rows a dirty train wrote.

ID_FIELDS = dict(experiment_family="starter", hypothesis_id="H1", feature_set_hash="fsh1", model_type="logistic_residual",
                 config_hash="cfg1", train_period="2023H1", test_period="2023H2", dataset="synthetic",
                 quote_quality="executable", git_commit="a" * 40, created_at=NOW)


@pytest.mark.parametrize(("field", "other"), [
    ("git_commit", "b" * 40), ("git_commit", "a" * 40 + "-dirty"), ("git_commit", None),
    ("created_at", NOW + timedelta(microseconds=1)),
])
def test_an_experiment_id_changes_with_the_commit_and_the_invocation_time(field, other):
    assert experiment_id(**ID_FIELDS | {field: other}) != experiment_id(**ID_FIELDS)


def test_an_experiment_id_reads_the_instant_not_its_spelling():
    same = [NOW, pd.Timestamp(NOW), pd.Timestamp(NOW).tz_convert("America/New_York"), NOW.isoformat()]
    assert {experiment_id(**ID_FIELDS | {"created_at": t}) for t in same} == {experiment_id(**ID_FIELDS)}


@pytest.mark.parametrize("missing", ["absent", None, float("nan"), pd.NaT], ids=["absent", "none", "nan", "nat"])
def test_an_experiment_id_needs_the_invocation_time(missing):
    fields = {k: v for k, v in ID_FIELDS.items() if k != "created_at"}
    if missing != "absent":
        fields["created_at"] = missing
    with pytest.raises(ValueError, match="created_at"):
        experiment_id(**fields)


def test_a_clean_rerun_logs_new_rows_beside_the_dirty_ones(syn_db):
    dirty = experiment(git_commit="a" * 40 + "-dirty")
    clean = experiment(git_commit="a" * 40, created_at=NOW + timedelta(hours=1))
    assert record_experiments(syn_db, [dirty]) == 1
    assert record_experiments(syn_db, [clean]) == 1
    assert syn_db.df("SELECT git_commit FROM experiments ORDER BY created_at")["git_commit"].tolist() == [
        "a" * 40 + "-dirty", "a" * 40]


def test_rows_without_a_time_take_one_time_per_call(syn_db):
    """record_experiments fills a missing created_at with one time per call, so a batch still dedupes."""
    row = {k: v for k, v in experiment().items() if k != "created_at"}
    assert record_experiments(syn_db, [row, row | {"p_value": 0.5}]) == 1

def test_record_experiments_keeps_successes_and_failures(syn_db):
    win = experiment()
    fail_metrics = {"reference": "market", "candidate": "logistic_residual", "n_obs": 1100,
                    "brier_ref": 0.2400, "brier_cand": 0.2450, "delta_brier": 0.2400 - 0.2450}
    loss = experiment(test_period="2024H1", metrics_json=json.dumps(fail_metrics), p_value=0.97)
    assert record_experiments(syn_db, [win, loss]) == 2
    assert record_experiments(syn_db, [win, loss]) == 0  # already logged
    stored = syn_db.df("SELECT test_period, metrics_json, dataset FROM experiments ORDER BY test_period")
    deltas = [json.loads(m)["delta_brier"] for m in stored["metrics_json"]]
    assert deltas[0] == pytest.approx(0.001) and deltas[1] == pytest.approx(-0.005)  # the failure is kept
    assert set(stored["dataset"]) == {"synthetic"}


def test_record_experiments_dedupes_within_a_batch(syn_db):
    assert record_experiments(syn_db, [experiment(), experiment(p_value=0.5)]) == 1
    # the same experiment on candle quotes is a different experiment, not a duplicate
    assert record_experiments(syn_db, [experiment(quote_quality="candle_degraded")]) == 1


@pytest.mark.parametrize("metrics,match", [
    ("not json", "not JSON"),
    (json.dumps({"reference": "market", "candidate": "x", "n_obs": 1, "brier_ref": 0.24}), "missing"),
    (json.dumps({"reference": "market", "candidate": "x", "n_obs": 1, "brier_ref": 0.24, "brier_cand": 0.23,
                 "delta_brier": -0.01}), "delta_brier"),  # sign flipped: must be ref - cand
])
def test_record_experiments_validates_metrics(syn_db, metrics, match):
    with pytest.raises(ValueError, match=match):
        record_experiments(syn_db, [experiment(metrics_json=metrics)])
    assert syn_db.count("experiments") == 0


def test_record_experiments_rejects_foreign_ids_and_fields(syn_db):
    with pytest.raises(ValueError, match="deterministic"):
        record_experiments(syn_db, [experiment(experiment_id="exp-mine")])
    with pytest.raises(ValueError, match="unknown"):
        record_experiments(syn_db, [experiment(brier=0.2)])
    with pytest.raises(ValueError, match="dataset"):
        record_experiments(syn_db, [experiment(dataset="live")])
    good = experiment()
    good["experiment_id"] = experiment_id(**good, dataset="synthetic")
    assert record_experiments(syn_db, [good]) == 1


# -- registration as its own step (26.4) ------------------------------------------

def test_register_now_uses_the_current_time_and_keeps_the_first(syn_db, tmp_path):
    from src.backtest.registry import register_now
    path = write_yaml(tmp_path, [hyp()])
    before = datetime.now(timezone.utc)
    n, reg = register_now(syn_db, path)
    after = datetime.now(timezone.utc)
    first = pd.Timestamp(reg["registered_at"].iloc[0])
    first = first.tz_localize("UTC") if first.tzinfo is None else first
    assert n == 1 and list(reg["hypothesis_id"]) == ["H_MLB_TEST_1"]
    assert before <= first <= after                              # now, not a passed-in (backdatable) time
    n2, reg2 = register_now(syn_db, path)                        # re-registration is a no-op
    assert n2 == 0 and pd.Timestamp(reg2["registered_at"].iloc[0]) == pd.Timestamp(reg["registered_at"].iloc[0])
    with pytest.raises(HypothesisChanged):
        register_now(syn_db, write_yaml(tmp_path, [hyp(label="EXPLORATORY")]))


def test_register_now_has_no_time_parameter():
    import inspect
    from src.backtest.registry import register_now
    assert set(inspect.signature(register_now).parameters) == {"db", "path"}


def test_registry_main_registers_the_sport_file(tmp_path, monkeypatch, capsys):
    from src.backtest.registry import USAGE, main
    from src.config import CONFIG_DIR
    from src.storage.database import connect
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    assert main(["register", "mlb"]) == 2 and USAGE in capsys.readouterr().err
    assert main(["register", "mlb", "synthetic"]) == 0
    out = capsys.readouterr().out
    n = len(load_hypotheses(CONFIG_DIR / "mlb_hypotheses.yaml"))
    assert out.splitlines()[0] == f"mlb_synthetic: {n} hypotheses newly registered"
    assert "H_MLB_L1_MARKET" in out and "H_MLB_NC" in out
    assert (tmp_path / "normalized" / "mlb_synthetic.duckdb").exists()
    assert main(["register", "mlb", "synthetic"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "mlb_synthetic: 0 hypotheses newly registered"
    db = connect("mlb", "synthetic", read_only=True)
    try:
        assert db.count("hypotheses") == n
    finally:
        db.close()
