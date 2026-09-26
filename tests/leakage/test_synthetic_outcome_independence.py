"""The synthetic generator's pregame and market values do not depend on outcomes (ADR 0003 item 2).

If a game's pre-outcome state is held fixed, flipping or redrawing its outcome must leave these
bit-identical:
- every pregame value the generator writes;
- every exchange quote;
- every sportsbook price.

Before synthetic-mlb-0.3.0 one random stream fed everything, and the market frames were drawn
after every outcome. So an outcome changed how many draws the final-score loop used, and with it
every later draw. The leakage audit before ADR 0003 found no dependence of values on outcomes,
but nothing in the generator ruled one out.

Two interventions, each run on both registered worlds, change half of one day's games:
- flip: `_World._home_wins` returns the other winner, and the box score follows it from the same
  stream;
- redraw: `_World._outcome_stream` hands those games an unrelated stream, so the winner and the
  whole box score are drawn again.

Interpretation (labelled in ADR 0003): "pregame X input" means every row the generator writes that
is available before first pitch. Features are built from those rows only. What must stay equal:
1. **every game up to and including the intervention date:**
   - its schedule row;
   - its contracts, without result and settlement;
   - its starters, lineups, news events and weather;
   - its truth row (p_true and the generator quantities);
   - its quotes and odds;
   - for every game except the changed ones, its outcome and box score.
2. **every later game whose pre-outcome state is unchanged:** its quotes and odds. A later game's
   pre-outcome state may change legitimately, because bullpen fatigue carries the last three days'
   relief work. That is the only path in the generator from one game's outcome to another game's
   pregame values.

The interventions cannot see two components that share one sequence, because that is a correlation,
not a dependence. Two tests check the streams themselves:
- `test_streams_are_keyed_by_seed_world_game_and_component`: every key has its own sequence, and
  `_World._stream` hands out the one asked for;
- `test_each_game_asks_for_its_own_stream_of_each_component`: every game asks for each of its
  streams once, under its own id.
"""

from __future__ import annotations

import functools
import hashlib
from collections import Counter
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.config import load_sport_config
from src.sports import synthetic
from src.sports.synthetic import WORLD_ENV, WORLDS, SyntheticSource

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
SMALL = {"seasons": [2023], "through_date": "2023-04-20", "max_games_per_day": 4}
START, END = date(2023, 1, 1), date(2023, 4, 20)
DAY = date(2023, 4, 10)        # the intervention date: games before, on and after it

OUTCOME_COLS = {"games": ["actual_start", "status", "home_score", "away_score", "home_win", "outcome_available_at"],
                "contracts": ["result", "settled_at"]}
PREGAME = ("games", "contracts", "mlb_probable_starters", "mlb_lineups", "info_events", "mlb_weather", "truth")
MARKET = ("market_quotes", "consensus_odds")
RESULT = ("games", "contracts", "mlb_pitching_lines", "mlb_batting_lines")   # with the outcome columns


@pytest.fixture(autouse=True)
def _no_world_env(monkeypatch):
    monkeypatch.delenv(WORLD_ENV, raising=False)


def _load(world: str) -> dict[str, pd.DataFrame]:
    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= SMALL
    src = SyntheticSource(cfg, WORLDS[world], now=NOW)
    frames = src.load(START, END)
    return frames | {"truth": src.truth}


@functools.lru_cache(maxsize=None)
def _base(world: str) -> dict[str, pd.DataFrame]:
    return _load(world)


def _changed_games(world: str) -> list[str]:
    games = _base(world)["games"]
    on_day = sorted(games.loc[games["official_date"] == DAY, "game_id"])
    assert len(on_day) >= 2
    return on_day[::2]             # half of the day's games: the other half must not move either


def _flip(monkeypatch, chosen: set[str]) -> None:
    orig = synthetic._World._home_wins

    def flipped(self, r, gid, p_true):
        won = orig(self, r, gid, p_true)
        return (not won) if gid in chosen else won
    monkeypatch.setattr(synthetic._World, "_home_wins", flipped)


def _redraw(monkeypatch, chosen: set[str]) -> None:
    orig = synthetic._World._outcome_stream

    def unrelated(self, gid):
        return np.random.default_rng([20260926, int(gid.rsplit("-", 1)[1])]) if gid in chosen else orig(self, gid)
    monkeypatch.setattr(synthetic._World, "_outcome_stream", unrelated)


def _per_game(frames: dict[str, pd.DataFrame], table: str, *, with_outcome: bool) -> pd.Series:
    """One hash per game over all of its rows in `table`, in order: equal hashes = bit-identical rows."""
    df = frames[table]
    if not with_outcome:
        df = df.drop(columns=OUTCOME_COLS.get(table, []))
    rows = pd.Series(pd.util.hash_pandas_object(df, index=False).to_numpy())
    return rows.groupby(df["game_id"].to_numpy()).agg(lambda h: hashlib.sha256(h.to_numpy().tobytes()).hexdigest())


def _same(base, var, tables, ids, *, with_outcome: bool = False) -> pd.Series:
    eq = pd.Series(True, index=ids)
    for t in tables:
        a = _per_game(base, t, with_outcome=with_outcome).reindex(ids)
        b = _per_game(var, t, with_outcome=with_outcome).reindex(ids)
        eq &= (a == b) | (a.isna() & b.isna())
    return eq


@pytest.mark.parametrize("intervention", ["flip", "redraw"])
@pytest.mark.parametrize("world", ["null_v1", "planted_v2"])
def test_outcomes_do_not_reach_pregame_or_market_values(world, intervention, monkeypatch):
    base = _base(world)
    chosen = set(_changed_games(world))
    (_flip if intervention == "flip" else _redraw)(monkeypatch, chosen)
    var = _load(world)

    games = base["games"].set_index("game_id")
    ids, day_of = games.index, games["official_date"]
    is_chosen = pd.Series(ids.isin(sorted(chosen)), index=ids)
    upto = day_of <= DAY
    pre = _same(base, var, PREGAME, ids)
    market = _same(base, var, MARKET, ids)
    result = _same(base, var, RESULT, ids, with_outcome=True)

    # 1. Up to the intervention date, nothing before first pitch moves: pregame rows and the market.
    assert pre[upto].all(), sorted(ids[upto & ~pre])
    assert market[upto].all(), f"market values moved in {int((upto & ~market).sum())} games on or before {DAY}"

    # The intervention took effect on every chosen game, and on no other result up to that day.
    if intervention == "flip":
        won = var["games"].set_index("game_id")["home_win"]
        assert (won[sorted(chosen)] != games.loc[sorted(chosen), "home_win"]).all()
    assert not result[is_chosen].any()
    assert result[upto & ~is_chosen].all(), sorted(ids[upto & ~is_chosen & ~result])

    # 2. A later game whose pre-outcome state is unchanged has an unchanged market.
    later_same = ~upto & pre
    assert later_same.sum() >= 5, "no later game kept its pre-outcome state: case 2 would test nothing"
    assert market[later_same].all(), f"market values moved in {int((later_same & ~market).sum())} later games"


def test_streams_are_keyed_by_seed_world_game_and_component():
    seed = int(load_sport_config("mlb")["synthetic"]["seed"])
    components = synthetic.STREAM_COMPONENTS
    assert {"pregame", "outcome", "kalshi", "books"} <= set(components)
    keys = [(seed, "null_v1", gid, c) for gid in ("syn-mlb-000001", "syn-mlb-000002") for c in components]
    keys += [(seed + 1, "null_v1", "syn-mlb-000001", "outcome"), (seed, "other_world", "syn-mlb-000001", "outcome")]
    seeds = [synthetic._stream_seed(*k) for k in keys]
    assert len(set(seeds)) == len(keys)                 # every (seed, world, game, component) has its own stream
    firsts = [np.random.default_rng(s).random(4).tobytes() for s in seeds]
    assert len(set(firsts)) == len(keys)                # ... and its own sequence

    # The two registered worlds draw from the same streams: one world key, the shared game state.
    assert {w.rng_world for w in WORLDS.values()} == {"null_v1"}
    null = synthetic._World(seed, WORLDS["null_v1"], NOW)
    planted = synthetic._World(seed, WORLDS["planted_v2"], NOW)
    for c in components:
        assert null._stream(c, "syn-mlb-000001").random(4).tobytes() == \
            planted._stream(c, "syn-mlb-000001").random(4).tobytes(), c

    # _World._stream hands out the sequence of exactly the (seed, world, key, component) asked for.
    for c in components:
        for key in ("", "2023", "2023-04-10", "syn-mlb-000001", "syn-mlb-000002"):
            want = np.random.default_rng(synthetic._stream_seed(seed, "null_v1", key, c)).random(4).tobytes()
            assert null._stream(c, key).random(4).tobytes() == want, (c, key)


def test_each_game_asks_for_its_own_stream_of_each_component(monkeypatch):
    """Every game gets one pregame, one exchange and one sportsbook stream, keyed by its id, and one
    outcome stream if it has been played. A component that reused another's stream, or a stream keyed
    by anything else, would be a correlation the interventions above cannot see."""
    calls: list[tuple[str, str]] = []
    orig = synthetic._World._stream

    def recording(self, component, key=""):
        calls.append((component, key))
        return orig(self, component, key)
    monkeypatch.setattr(synthetic._World, "_stream", recording)
    games = _load("null_v1")["games"]

    per_game = ("pregame", "outcome", "kalshi", "books")
    got = Counter(c for c in calls if c[0] in per_game)
    played = games.loc[games["home_win"].notna(), "game_id"]
    want = Counter({(c, g): 1 for g in games["game_id"] for c in ("pregame", "kalshi", "books")}
                   | {("outcome", g): 1 for g in played})
    assert 0 < len(played) < len(games)                 # today's games have no outcome stream
    assert got == want, sorted((got - want) + (want - got))
