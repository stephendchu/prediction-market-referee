"""Strategy family and feature presets are separate settings; the live baseball gate."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.registry import load_hypotheses
from src.config import BASEBALL_FAMILIES, BaseballGateClosed, feature_preset, load_sport_config, strategy_family
from src.features.mlb import feature_families
from src.models.train import feature_sets, run_training
from src.reports.diagnostics import ReviewInputs, build_review


def cfg(**over) -> dict:
    c = copy.deepcopy(load_sport_config("mlb"))
    for dotted, v in over.items():
        sec, key = dotted.split("__")
        c.setdefault(sec, {})[key] = v
    return c


def test_outcome_residual_is_the_default_and_market_behavior_is_refused():
    assert strategy_family(cfg()) == "outcome_residual"
    assert strategy_family({}) == "outcome_residual"
    with pytest.raises(NotImplementedError, match="26.20 evaluator"):
        strategy_family(cfg(strategy__family="market_behavior"))
    with pytest.raises(ValueError, match="strategy.family must be one of"):
        strategy_family(cfg(strategy__family="momentum"))


def test_market_behavior_is_refused_by_train_backtest_and_review(syn_db):
    mb = cfg(strategy__family="market_behavior")
    with pytest.raises(NotImplementedError):
        run_training(syn_db, mb)
    with pytest.raises(NotImplementedError):
        run_backtest(syn_db, mb, {"selected": "logistic_residual"})
    with pytest.raises(NotImplementedError):
        build_review(ReviewInputs(sport="mlb", dataset="synthetic", is_synthetic=True, periods=[],
                                  predictions=pd.DataFrame(), decisions=pd.DataFrame()), mb)
    assert syn_db.count("hypotheses") == 0     # refused before anything was registered or fitted


def test_market_only_is_the_default_preset_and_has_no_baseball():
    name, fams = feature_preset(cfg(), "live")
    assert name == "market_only" and fams == ["market"]
    assert not set(fams) & set(BASEBALL_FAMILIES)
    assert feature_preset(cfg(), "synthetic", "mlb_level1")[1] == ["market", *BASEBALL_FAMILIES]
    with pytest.raises(ValueError, match="not one of features.presets"):
        feature_preset(cfg(), "synthetic", "everything")


def test_baseball_families_are_refused_on_live_until_the_gate_decision_exists():
    with pytest.raises(BaseballGateClosed, match="recorded decision"):
        feature_preset(cfg(), "live", "mlb_level1")
    for bad in ("docs/decisions/9999-not-written.md", "README.md", "../docs/decisions/0001-dirty-trees-and-append-only-backtest-runs.md",
                "docs/decisions/../../README.md", "docs/decisions"):
        with pytest.raises(BaseballGateClosed):
            feature_preset(cfg(features__baseball_gate_decision=bad), "live", "mlb_level1")
    ok = cfg(features__baseball_gate_decision="docs/decisions/0001-dirty-trees-and-append-only-backtest-runs.md")
    assert feature_preset(ok, "live", "mlb_level1")[1][0] == "market"   # gate opens only via a real record
    assert feature_preset(cfg(), "synthetic", "mlb_level1")              # synthetic: mechanism test, allowed


def test_live_training_with_a_baseball_preset_is_refused_before_anything_runs(live_db):
    with pytest.raises(BaseballGateClosed):
        run_training(live_db, cfg(), preset="mlb_level1")
    assert live_db.count("hypotheses") == 0


def test_the_preset_limits_model_and_exploratory_features_but_not_negative_controls():
    hyps = load_hypotheses("config/mlb_hypotheses.yaml")
    fams = feature_families()
    only = feature_sets(fams, hyps, ["market"])
    assert only.model and {fams[f] for f in only.model} == {"market"}
    assert set(only.excluded_by_preset) == {"H_MLB_L1_STARTER", "H_MLB_L1_OFFENSE", "H_MLB_L1_BULLPEN", "H_MLB_L1_ENV"}
    assert only.negative_controls == feature_sets(fams, hyps).negative_controls != []
    full = feature_sets(fams, hyps, ["market", *BASEBALL_FAMILIES])
    assert {fams[f] for f in full.model} == {"market", *BASEBALL_FAMILIES} and full.excluded_by_preset == {}
