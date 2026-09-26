"""The planted-alpha synthetic world v2 and the null world beside it.

The null world (`null_v1`, generator synthetic-mlb-0.3.0) stays the default and byte-identical: the
golden fingerprints below were taken from synthetic-mlb-0.3.0, the per-game, per-component random
streams of ADR 0003. (The 0.1.0 fingerprints are recorded in ADR 0003.) The
planted world (`planted_v2`, synthetic-mlb-0.3.0-planted) uses the same seed and the same random
streams, so games, outcomes and p_true are the null world's; only the market moves:

- the market's home logit path is divided by `market_slope` (< 1: overconfident), exchange and books;
- the exchange's home logit, and only the exchange's, is shifted by `planted_shift_logit` in games hosted by
  one of `planted_home_teams`.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import config
from src.config import load_sport_config
from src.reports.diagnostics import referee_config_hash
from src.sports import synthetic
from src.sports.synthetic import (DEFAULT_WORLD, GENERATOR_VERSION, PLANTED_GENERATOR_VERSION, PLANTED_HOME_TEAMS,
                                  WORLD_ENV, WORLD_MARKER, WORLDS, SyntheticSource, WorldParams, resolve_world)
from src.storage.ingest import ingest

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
SMALL = {"seasons": [2023], "through_date": "2023-04-20", "max_games_per_day": 4}
START, END = date(2023, 1, 1), date(2023, 4, 20)

# Taken from synthetic-mlb-0.3.0 as published here: the default world, SMALL, NOW. Four differ from the
# 0.3.0 column of ADR 0003's table (contracts, games, market_quotes, truth): this repository names the
# series SYN-MLBGAME and two columns market_event_id and market_noise_logit. Every value is otherwise the same.
GOLDEN_NULL = {
    "consensus_odds": "ed2ab6e443ac6a87", "contracts": "d2e732e143f85c61", "games": "362ce898835f4a20",
    "info_events": "739e27947264bbbd", "market_quotes": "f740b5da50f7d11b", "mlb_batting_lines": "67bd3ebda9433047",
    "mlb_lineups": "a5def83273d527a8", "mlb_pitching_lines": "1ad777fa87eb3377", "mlb_players": "18ba6196bc720870",
    "mlb_probable_starters": "fdda7626ceb4fec7", "mlb_weather": "cfb986b6c07fd6a6", "venues": "b11ecb77cb71d604",
    "truth": "bde4dc819db841db"}

DELTA = 0.4                  # test-only sizes, large enough to see; the registered world is in WORLDS
SLOPE = 0.5
SHIFT_ONLY = WorldParams(planted_shift_logit=DELTA, planted_home_teams=PLANTED_HOME_TEAMS)
SLOPE_ONLY = WorldParams(market_slope=SLOPE, planted_home_teams=PLANTED_HOME_TEAMS)
BOTH = WorldParams(market_slope=SLOPE, planted_shift_logit=DELTA, planted_home_teams=PLANTED_HOME_TEAMS)
PLANTED_FIELDS = ("market_slope", "planted_shift_logit", "planted_home_teams")
PLANTED_COLS = ["in_planted_subset", "planted_shift_logit", "market_slope", "market_home_logit_at_first_pitch"]


def _cfg() -> dict:
    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= SMALL
    return cfg


def _fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([[c, str(t)] for c, t in df.dtypes.items()]).encode())
    h.update(str(len(df)).encode())
    if len(df):
        h.update(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=None)
def _world(params: WorldParams) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, SyntheticSource]:
    src = SyntheticSource(_cfg(), params, now=NOW)
    frames = src.load(START, END)
    return frames, src.truth, src


def _world_copy(params: WorldParams) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames, truth, _ = _world(params)
    return {k: v.copy() for k, v in frames.items()}, truth.copy()


def _logit(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


def _home_of(frames: dict[str, pd.DataFrame]) -> pd.Series:
    return frames["games"].set_index("game_id")["home_team"]


def _is_home(frames: dict[str, pd.DataFrame], quotes: pd.DataFrame) -> pd.Series:
    return quotes["contract_id"].map(frames["contracts"].set_index("contract_id")["yes_side"]) == "home"


def _mids(quotes: pd.DataFrame) -> pd.Series:
    return (quotes["yes_bid"] + quotes["yes_ask"]) / 2


@pytest.fixture(autouse=True)
def _no_world_env(monkeypatch):
    monkeypatch.delenv(WORLD_ENV, raising=False)


# -- the null world ------------------------------------------------------------------------------

def test_null_world_is_byte_identical_to_the_0_3_0_generator():
    src = SyntheticSource(_cfg(), now=NOW)                  # the default path: no params, no env
    frames = src.load(START, END)
    got = {k: _fingerprint(v) for k, v in frames.items()} | {"truth": _fingerprint(src.truth)}
    assert got == GOLDEN_NULL
    assert src.world == DEFAULT_WORLD == "null_v1"
    assert GENERATOR_VERSION == "synthetic-mlb-0.3.0"
    assert {v for df in frames.values() for v in df["collector_version"].unique()} == {GENERATOR_VERSION}


# -- the registry and world selection ------------------------------------------------------------

def test_registry_holds_the_null_world_and_a_modest_planted_world():
    assert set(WORLDS) == {"null_v1", "planted_v2"}
    assert WORLDS["null_v1"] == WorldParams()
    assert not WorldParams().planted and WorldParams().generator_version == GENERATOR_VERSION
    p = WORLDS["planted_v2"]
    assert p.planted and p.generator_version == PLANTED_GENERATOR_VERSION == "synthetic-mlb-0.3.0-planted"
    assert 0.5 < p.market_slope < 1.0                      # overconfident, not absurdly so
    assert 0.0 < p.planted_shift_logit <= 0.5               # a modest shift
    assert p.planted_home_teams == PLANTED_HOME_TEAMS == tuple(sorted(synthetic._POPULAR))
    v1 = [f.name for f in dataclasses.fields(WorldParams) if f.name not in PLANTED_FIELDS]
    assert all(getattr(p, k) == getattr(WorldParams(), k) for k in v1)      # the v1 positive controls are kept


def test_world_resolution_default_config_env_and_unknown(monkeypatch):
    cfg = _cfg()
    assert resolve_world(cfg) == "null_v1"
    assert resolve_world(cfg | {"synthetic": cfg["synthetic"] | {"world": "planted_v2"}}) == "planted_v2"
    monkeypatch.setenv(WORLD_ENV, "planted_v2")
    assert resolve_world(cfg) == "planted_v2"
    monkeypatch.setenv(WORLD_ENV, "null_v1")                # the environment wins over the config
    assert resolve_world(cfg | {"synthetic": cfg["synthetic"] | {"world": "planted_v2"}}) == "null_v1"
    monkeypatch.setenv(WORLD_ENV, "planted_v3")
    with pytest.raises(ValueError, match="planted_v3"):
        resolve_world(cfg)


def test_environment_selects_the_planted_parameters(monkeypatch):
    monkeypatch.setenv(WORLD_ENV, "planted_v2")
    src = SyntheticSource(_cfg(), now=NOW)
    assert src.world == "planted_v2" and src.params == WORLDS["planted_v2"]
    assert src.description["generator"] == PLANTED_GENERATOR_VERSION and src.description["world"] == "planted_v2"


# -- the planted effects -------------------------------------------------------------------------

def _drop_version(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["collector_version"])


def test_shift_moves_the_market_only_and_only_in_home_games_of_the_subset():
    null, _ = _world_copy(WorldParams())
    planted, _ = _world_copy(SHIFT_ONLY)
    # The books never see the shift.
    pd.testing.assert_frame_equal(_drop_version(planted["consensus_odds"]), _drop_version(null["consensus_odds"]))
    q0, q1 = null["market_quotes"], planted["market_quotes"]
    assert q0[["contract_id", "quote_ts"]].equals(q1[["contract_id", "quote_ts"]])     # the same draws, row for row
    home = q0["game_id"].map(_home_of(null))
    subset = home.isin(PLANTED_HOME_TEAMS)
    assert subset.sum() > 500 and (~subset).sum() > 500
    # Outside the subset the exchange is untouched.
    pd.testing.assert_frame_equal(_drop_version(q1[~subset]), _drop_version(q0[~subset]))
    # Inside it, the home contract is dearer and the away contract cheaper by about DELTA in logit.
    is_home = _is_home(null, q0)
    d = pd.Series(_logit(_mids(q1)) - _logit(_mids(q0)), index=q0.index).dropna()
    assert 0.8 * DELTA < d[subset & is_home].mean() < 1.2 * DELTA
    assert -1.2 * DELTA < d[subset & ~is_home].mean() < -0.8 * DELTA


def _slope(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def test_slope_below_one_stretches_the_market_logit_on_both_venues():
    null, _ = _world_copy(WorldParams())
    planted, _ = _world_copy(SLOPE_ONLY)
    k = 1 / SLOPE
    o0, o1 = null["consensus_odds"], planted["consensus_odds"]
    assert o0[["game_id", "book", "odds_ts"]].equals(o1[["game_id", "book", "odds_ts"]])
    assert 0.95 * k < _slope(_logit(o0["p_home_devig"]), _logit(o1["p_home_devig"])) < 1.05 * k
    q0, q1 = null["market_quotes"], planted["market_quotes"]
    is_home = _is_home(null, q0)
    assert 0.85 * k < _slope(_logit(_mids(q0[is_home])), _logit(_mids(q1[is_home]))) < 1.05 * k
    # Nothing but the market moves: schedule, outcomes, lineups, weather are the null world's.
    for name in set(null) - {"market_quotes", "consensus_odds"}:
        pd.testing.assert_frame_equal(_drop_version(planted[name]), _drop_version(null[name]), obj=name)


def test_truth_keeps_p_true_and_records_the_planted_columns():
    frames, truth = _world_copy(BOTH)
    _, null_truth = _world_copy(WorldParams())
    _, shift_truth = _world_copy(SHIFT_ONLY)
    assert set(PLANTED_COLS).isdisjoint(null_truth.columns)
    pd.testing.assert_frame_equal(truth[list(null_truth.columns)], null_truth)          # p_true unchanged
    home = truth["game_id"].map(_home_of(frames))
    in_subset = home.isin(PLANTED_HOME_TEAMS)
    assert truth["in_planted_subset"].equals(in_subset.rename("in_planted_subset"))
    assert np.allclose(truth["planted_shift_logit"], DELTA * in_subset)
    assert (truth["market_slope"] == SLOPE).all()
    # The exchange's home logit at first pitch: path / slope + shift + its game noise.
    unshifted = truth["market_home_logit_at_first_pitch"] - truth["planted_shift_logit"] - truth["market_noise_logit"]
    base = (shift_truth["market_home_logit_at_first_pitch"] - shift_truth["planted_shift_logit"]
            - shift_truth["market_noise_logit"])
    assert np.allclose(unshifted, base / SLOPE)
    # ... and it is where the last home quote before first pitch sits (up to the quote wander and cents).
    q = frames["market_quotes"]
    q = q[_is_home(frames, q)]
    last = q.sort_values("quote_ts").groupby("game_id").tail(1).set_index("game_id")
    last = last[last["quote_ts"] > last.index.map(frames["games"].set_index("game_id")["scheduled_start"])
                - pd.Timedelta(minutes=10)]
    k_fp = truth.drop_duplicates("game_id").set_index("game_id")["market_home_logit_at_first_pitch"]
    gap = (pd.Series(_logit(_mids(last)), index=last.index) - k_fp.reindex(last.index)).dropna()
    assert len(gap) > 40 and gap.abs().median() < 0.06


def test_planted_rows_stay_synthetic_with_the_planted_version(syn_db):
    frames, truth, src = _world(BOTH)
    for name, df in frames.items():
        assert set(df["collector_version"]) == {PLANTED_GENERATOR_VERSION}, name
        assert set(df["source"]) == {"synthetic"} and df["is_synthetic"].all(), name
        assert set(df["availability_basis"]) == {"synthetic"}, name
    assert truth["SYNTHETIC_TRUTH"].notna().all()
    assert src.description["generator"] == PLANTED_GENERATOR_VERSION
    planted = ingest(syn_db, SyntheticSource(_cfg(), BOTH, now=NOW), START, END)
    assert syn_db.df("SELECT DISTINCT collector_version FROM market_quotes")["collector_version"].tolist() == \
        [PLANTED_GENERATOR_VERSION]
    null = ingest(syn_db, SyntheticSource(_cfg(), now=NOW), START, END)
    assert planted.data_version != null.data_version


def test_selecting_the_planted_world_leaves_the_config_and_referee_hash_alone(monkeypatch):
    cfg = load_sport_config("mlb")
    before, h0 = copy.deepcopy(cfg), referee_config_hash(cfg)
    monkeypatch.setenv(WORLD_ENV, "planted_v2")
    SyntheticSource(cfg, now=NOW)
    assert cfg == before and referee_config_hash(cfg) == h0


# -- one world per data root ----------------------------------------------------------------------

def _source(world: str, monkeypatch) -> SyntheticSource:
    monkeypatch.setenv(WORLD_ENV, world)
    return SyntheticSource(_cfg(), now=NOW)


def test_data_root_guard(tmp_path, monkeypatch, syn_db):
    default = tmp_path / "default"
    # The planted world never lands in the default data root.
    with pytest.raises(ValueError, match="QUANT_DATA_DIR"):
        _source("planted_v2", monkeypatch).claim_data_root(default, syn_db, default_root=default)
    assert not (default / WORLD_MARKER).exists()
    # The null world may, and the root is then marked.
    _source("null_v1", monkeypatch).claim_data_root(default, syn_db, default_root=default)
    assert json.loads((default / WORLD_MARKER).read_text())["world"] == "null_v1"
    # A marked root refuses the other world, both ways.
    own = tmp_path / "planted"
    _source("planted_v2", monkeypatch).claim_data_root(own, syn_db, default_root=default)
    marker = json.loads((own / WORLD_MARKER).read_text())
    assert marker["world"] == "planted_v2" and marker["generator"] == PLANTED_GENERATOR_VERSION
    with pytest.raises(ValueError, match="planted_v2"):
        _source("null_v1", monkeypatch).claim_data_root(own, syn_db, default_root=default)
    other = tmp_path / "other_null"
    _source("null_v1", monkeypatch).claim_data_root(other, syn_db, default_root=default)
    with pytest.raises(ValueError, match="null_v1"):
        _source("planted_v2", monkeypatch).claim_data_root(other, syn_db, default_root=default)
    # An unmarked root that already holds ingests is of unknown world: the planted world refuses it.
    ingest(syn_db, SyntheticSource(_cfg(), now=NOW), START, END)
    unmarked = tmp_path / "unmarked"
    with pytest.raises(ValueError, match="ingest"):
        _source("planted_v2", monkeypatch).claim_data_root(unmarked, syn_db, default_root=default)
    assert not (unmarked / WORLD_MARKER).exists()


def test_cli_ingest_refuses_the_planted_world_in_the_default_data_root(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from src.cli import app

    def unreachable(self, start, end):
        raise RuntimeError("load reached: the guard did not run")

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)       # the "default" data root is tmp_path/data
    monkeypatch.setattr(SyntheticSource, "load", unreachable)
    monkeypatch.delenv("QUANT_DATA_DIR", raising=False)
    monkeypatch.setenv("QUANT_DATASET", "synthetic")
    monkeypatch.setenv(WORLD_ENV, "planted_v2")
    res = CliRunner().invoke(app, ["ingest", "--sport", "mlb", "--start", "2023-01-01"])
    assert res.exit_code != 0
    assert isinstance(res.exception, ValueError) and "QUANT_DATA_DIR" in str(res.exception)
    assert not (tmp_path / "data" / WORLD_MARKER).exists()
