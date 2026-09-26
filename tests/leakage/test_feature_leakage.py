"""Feature-builder leakage tests (spec 26.11: feature_available_at <= snapshot_time).

The observation's cutoff is its quote time, three minutes before the snapshot.
Each case injects rows that became available just after the cutoff, and:

1. asserts the whole feature output (value, available_at, source_ref) is
   unchanged, so the injected rows are ignored;
2. repeats the build with the point-in-time filters broken by monkeypatch
   (``common.asof`` / ``common.available`` look past the cutoff) and asserts
   the named features DO change, so test 1 would fail if the filters were
   removed. A leakage test that cannot fail is not a test;
3. with the filters broken but the output guard intact, asserts the build
   raises FeatureLeakError, so a broken filter cannot pass silently.

The world is self-contained (no shared helpers with the unit tests) and built
so every feature a case targets has a real value in the base build.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta as td
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from factories import GAME_DATE, START, frame, observation, provenance

from src.features import common as pit
from src.features import mlb

SNAP = START - td(minutes=30)
CUT = SNAP - td(minutes=3)            # quote_ts: the information cutoff
AFTER = CUT + td(minutes=1)           # after the cutoff, before the snapshot
SOON = CUT + td(seconds=30)
EARLY = START - td(days=1)
SCHEDULED = START - td(days=30)
YES, OPP = "SYN-NYY@T-30", "SYN-TB@T-30"
FOREVER = 100 * 365 * pit.DAY         # "filter removed"
FIVE_MIN = 5 * pit.MIN                # "filter looks 5 minutes ahead"


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def rows(*dicts: dict, at: pd.Timestamp = EARLY) -> pd.DataFrame:
    out = []
    for d in dicts:
        d = dict(d)
        out.append(provenance(d.pop("available_at", at)) | d)
    return pd.DataFrame(out)


def final(d: str) -> pd.Timestamp:
    """When a box score dated `d` became final."""
    return pd.Timestamp(d, tz="UTC") + td(days=1, hours=3)


def pline(pid: int, team: str, d: str, *, starter: bool, gid: str | None = None, **stats) -> dict:
    base = {"game_id": gid or f"g-{team}-{d}", "official_date": date.fromisoformat(d), "side": "home",
            "team": team, "player_id": pid, "is_starter": starter, "outs": 3, "batters_faced": 4, "pitches": 15,
            "strikeouts": 1, "walks": 0, "hit_by_pitch": 0, "home_runs": 0, "ground_balls": 1, "fly_balls": 1,
            "saves": 0, "holds": 0, "available_at": final(d)}
    return base | stats


def bline(pid: int, team: str, slot: int, gid: str = "g-prev", d: str = "2025-05-31", **stats) -> dict:
    base = {"game_id": gid, "official_date": date.fromisoformat(d), "side": "home" if team == "NYY" else "away",
            "team": team, "player_id": pid, "batting_order": slot, "plate_appearances": 4, "at_bats": 4, "hits": 1,
            "doubles": 0, "triples": 0, "home_runs": 0, "walks": 0, "hit_by_pitch": 0, "strikeouts": 1,
            "available_at": final(d)}
    return base | stats


def lineup(team: str, players: list[int], at: pd.Timestamp) -> list[dict]:
    side = "home" if team == "NYY" else "away"
    return [{"game_id": "syn-mlb-1", "side": side, "team": team, "batting_order": i + 1, "player_id": p,
             "available_at": at} for i, p in enumerate(players)]


def game(gid: str, d: date, venue: int, score: tuple[int, int] | None = None, home: str = "NYY",
         away: str = "TB", **over) -> dict:
    out_at = pd.Timestamp(d, tz="UTC") + td(days=1, hours=3) if score else None
    row = {"game_id": gid, "sport": "mlb", "season": d.year, "official_date": d, "game_type": "R",
           "scheduled_start": pd.Timestamp(d, tz="UTC") + td(hours=23), "home_team": home, "away_team": away,
           "venue_id": venue, "home_score": score and score[0], "away_score": score and score[1],
           "home_win": (score[0] > score[1]) if score else None, "outcome_available_at": out_at,
           "available_at": SCHEDULED}
    return row | over


def venue(vid: int, roof: str, az: float, at: pd.Timestamp = pd.Timestamp("2023-01-01", tz="UTC")) -> dict:
    return {"venue_id": vid, "name": f"V{vid}", "roof_type": roof, "cf_azimuth_deg": az, "available_at": at}


def forecast(gid: str, at: pd.Timestamp, **over) -> dict:
    base = {"game_id": gid, "kind": "forecast", "forecast_for": START, "temperature_f": 70.0,
            "wind_speed_mph": 10.0, "wind_direction_deg": 180.0, "humidity_pct": 50.0, "precip_prob": 0.1,
            "roof_status": "unknown", "available_at": at}
    return base | over


def quote(cid: str, t: pd.Timestamp, bid: float) -> dict:
    return {"contract_id": cid, "game_id": "syn-mlb-1", "quote_ts": t, "quote_source": "synthetic", "yes_bid": bid,
            "yes_ask": bid + 0.02, "no_bid": 0.98 - bid, "no_ask": 1 - bid, "available_at": t + td(seconds=1)}


def odds(book: str, t: pd.Timestamp, p_home: float) -> dict:
    return {"game_id": "syn-mlb-1", "book": book, "market_type": "ml", "odds_ts": t - td(seconds=5),
            "p_home_devig": p_home, "p_away_devig": 1 - p_home, "devig_method": "proportional", "available_at": t}


def news(t: pd.Timestamp, kind: str = "injury", gid: str | None = "syn-mlb-1", team: str = "NYY") -> dict:
    return {"event_id": f"e-{kind}-{t.value}-{team}", "game_id": gid, "team": team, "type": kind,
            "first_seen": False, "available_at": t}


# ---------------------------------------------------------------------------
# The base world: two games on the date, every feature populated where it can be
# ---------------------------------------------------------------------------

def observations() -> pd.DataFrame:
    common = {"quote_ts": CUT, "quote_available_at": CUT, "quote_age_s": 180.0, "volume": 1000.0,
              "open_interest": 500.0}
    home = observation(30, p_consensus=0.58, **common)
    away = observation(30, observation_id=OPP, contract_id="SYN-TB", yes_team="TB", yes_side="away",
                       market_yes_bid=0.43, market_yes_ask=0.45, p_mid=0.44, p_consensus=0.42, **common)
    return frame(home, away)


NYY_NINE = list(range(101, 110))
TB_NINE = list(range(201, 210))


def world() -> dict[str, pd.DataFrame]:
    starts = [pline(1, "NYY", "2025-05-22", starter=True, outs=18, batters_faced=24, strikeouts=6, pitches=92),
              pline(1, "NYY", "2025-05-27", starter=True, outs=21, batters_faced=26, strikeouts=8, pitches=98),
              pline(2, "TB", "2025-05-21", starter=True, outs=15, batters_faced=22, strikeouts=4, walks=3,
                    pitches=88),
              pline(2, "TB", "2025-05-26", starter=True, outs=18, batters_faced=25, strikeouts=5, home_runs=2,
                    pitches=95),
              pline(3, "NYY", "2025-05-25", starter=True, outs=12, batters_faced=22, strikeouts=2, walks=4,
                    home_runs=3, pitches=85)]
    relief = [pline(301, "NYY", "2025-05-29", starter=False, pitches=12, saves=1),
              pline(301, "NYY", "2025-05-30", starter=False, pitches=15, saves=1),
              pline(302, "NYY", "2025-05-29", starter=False, pitches=14, holds=1),
              pline(302, "NYY", "2025-05-31", starter=False, pitches=18, holds=1),
              pline(303, "NYY", "2025-05-28", starter=False, pitches=16, holds=1, walks=1),
              pline(401, "TB", "2025-05-25", starter=False, pitches=14, saves=1),
              pline(401, "TB", "2025-05-28", starter=False, pitches=13, saves=1),
              pline(402, "TB", "2025-05-27", starter=False, pitches=11, holds=1),
              pline(402, "TB", "2025-05-30", starter=False, pitches=17, holds=1, home_runs=1),
              pline(403, "TB", "2025-05-29", starter=False, pitches=20, holds=1)]
    batting = [bline(p, "NYY", i + 1, hits=1 + i % 2, doubles=i % 3 == 0) for i, p in enumerate(NYY_NINE)]
    batting += [bline(p, "TB", i + 1, walks=i % 2, hits=i % 3 == 0) for i, p in enumerate(TB_NINE)]
    batting += [bline(210, "TB", 9, gid="g-old", d="2025-05-20", home_runs=1)]
    bats = {**dict.fromkeys((101, 102, 103, 150), "L"), **dict.fromkeys((104, 105, 106, 108, 109), "R"), 107: "S",
            **dict.fromkeys(TB_NINE, "R"), 210: "L"}
    players = [{"player_id": 1, "throws": "L"}, {"player_id": 2, "throws": "R"}, {"player_id": 3, "throws": "R"},
               *({"player_id": p, "bats": b} for p, b in bats.items())]
    past = [game("p1", date(2025, 5, 28), 1, (6, 4)), game("p2", date(2025, 5, 29), 1, (7, 5)),
            game("p3", date(2025, 5, 30), 2, (4, 2), home="BOS", away="SEA"),
            game("p4", date(2025, 5, 30), 2, (5, 1), home="BOS", away="SEA")]
    return {
        "games": rows(game("syn-mlb-1", GAME_DATE, 1),
                      game("syn-mlb-2", GAME_DATE, 2, home="BOS", away="SEA"), *past),
        "venues": rows(venue(1, "retractable", 0.0), venue(2, "open", 90.0)),
        "mlb_weather": rows(forecast("syn-mlb-1", START - td(hours=6)),
                            forecast("syn-mlb-2", START - td(hours=6), temperature_f=80.0,
                                     wind_direction_deg=270.0, wind_speed_mph=8.0)),
        "mlb_probable_starters": rows({"game_id": "syn-mlb-1", "side": "home", "team": "NYY", "player_id": 1},
                                      {"game_id": "syn-mlb-1", "side": "away", "team": "TB", "player_id": 2}),
        "mlb_players": rows(*players),
        "mlb_pitching_lines": rows(*starts, *relief),
        "mlb_batting_lines": rows(*batting),
        "mlb_lineups": rows(*lineup("TB", [*TB_NINE[:8], 210], START - td(hours=2))),   # NYY: projected
        "market_quotes": rows(*(quote("SYN-NYY", CUT - td(minutes=m), b) for m, b in ((70, 0.51), (30, 0.49), (0, 0.55))),
                              *(quote("SYN-TB", CUT - td(minutes=m), b) for m, b in ((70, 0.47), (30, 0.49), (0, 0.43)))),
        "consensus_odds": rows(odds("A", CUT - td(minutes=40), 0.55), odds("B", CUT - td(minutes=35), 0.57),
                               odds("A", CUT - td(minutes=10), 0.60)),
        "info_events": rows(news(CUT - td(minutes=20))),
    }


def with_rows(base: dict[str, pd.DataFrame], extra: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    out = dict(base)
    for name, new in extra.items():
        out[name] = pd.concat([out[name], rows(*new)], ignore_index=True)
    return out


def build(tables: dict[str, pd.DataFrame]) -> mlb.FeatureBuild:
    return mlb.build_features(observations(), tables, build_run_id="leak-test", is_synthetic=True)


# ---------------------------------------------------------------------------
# Injections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    name: str
    extra: Callable[[], dict[str, list[dict]]]
    affected: tuple[str, ...]        # features that change once the filters leak
    lookahead: int = FOREVER


def _pitching_same_date():
    d = GAME_DATE.isoformat()   # this game's own box score
    return {"mlb_pitching_lines": [
        pline(1, "NYY", d, starter=True, gid="syn-mlb-1", outs=3, batters_faced=12, strikeouts=0, walks=5,
              home_runs=4, pitches=70, available_at=AFTER),
        pline(301, "NYY", d, starter=False, gid="syn-mlb-1", walks=3, home_runs=2, pitches=30, saves=1,
              available_at=AFTER)]}


def _pitching_prev_day():
    return {"mlb_pitching_lines": [   # a late-final box score from the previous date
        pline(401, "TB", "2025-05-31", starter=False, pitches=30, saves=1, walks=3, home_runs=2, available_at=AFTER),
        pline(402, "TB", "2025-05-31", starter=False, pitches=26, holds=1, available_at=AFTER)]}


def _starter_change():
    assert CUT < AFTER < SNAP
    return {"mlb_probable_starters": [{"game_id": "syn-mlb-1", "side": "home", "team": "NYY", "player_id": 3,
                                       "available_at": AFTER}]}


def _lineup_after_cutoff():
    return {"mlb_lineups": [*lineup("NYY", [*NYY_NINE[:6], 150, 108, 109], AFTER),
                            *lineup("TB", [*TB_NINE[:7], 210, 209], AFTER)]}


def _forecast_after_cutoff():
    fc = dict(temperature_f=45.0, wind_speed_mph=25.0, wind_direction_deg=0.0, humidity_pct=90.0, precip_prob=0.9,
              roof_status="closed")
    return {"mlb_weather": [forecast("syn-mlb-1", AFTER, **fc), forecast("syn-mlb-2", AFTER, **fc)]}


def _outcome_after_cutoff():
    # The schedule row is old; only the result is new. And one whole games row that is new.
    return {"games": [game("p5", date(2025, 5, 31), 1, (15, 12), outcome_available_at=AFTER),
                      game("p6", date(2025, 5, 31), 1, (11, 10), outcome_available_at=CUT, available_at=AFTER)]}


def _odds_after_quote():
    return {"consensus_odds": [odds("A", SOON, 0.70), odds("B", SOON, 0.68), odds("C", SOON, 0.69)]}


def _odds_flip_gap_history():
    # Home at 0.40 would put every earlier NYY quote's gap below zero (and TB's above): each one a break.
    return {"consensus_odds": [odds("A", AFTER, 0.40), odds("B", AFTER, 0.40)]}


def _later_quote():
    return {"market_quotes": [quote("SYN-NYY", AFTER, 0.80), quote("SYN-TB", AFTER, 0.18)]}


def _news_after_cutoff():
    return {"info_events": [news(AFTER, "starter"), news(AFTER, "lineup", gid=None, team="TB")]}


def _batting_after_cutoff():
    own = [bline(p, "NYY", i + 1, gid="syn-mlb-1", d=GAME_DATE.isoformat(), hits=3, home_runs=2, strikeouts=0,
                 available_at=AFTER) for i, p in enumerate(NYY_NINE)]
    late = bline(150, "NYY", 1, plate_appearances=6, at_bats=6, hits=0, strikeouts=4, available_at=AFTER)
    return {"mlb_batting_lines": [*own, late]}


def _player_after_cutoff():
    return {"mlb_players": [{"player_id": 1, "throws": "R", "available_at": AFTER}]}


def _venue_after_cutoff():
    return {"venues": [venue(1, "dome", 90.0, at=AFTER)]}


def _schedule_after_cutoff():
    ids = [f"syn-mlb-{k}" for k in range(3, 9)]
    return {"games": [game(g, GAME_DATE, 2, home="BOS", away="SEA", available_at=AFTER) for g in ids],
            "mlb_weather": [forecast(g, START - td(hours=6), temperature_f=90.0 + k) for k, g in enumerate(ids)]}


CASES = [
    Case("pitching_line_same_date", _pitching_same_date,
         ("sp_fip_diff", "sp_k_pct_diff", "sp_bb_pct_diff", "sp_hr9_diff", "sp_exp_ip_diff", "sp_last_pitches_diff",
          "sp_days_rest_diff", "bp_quality_diff")),
    Case("pitching_line_previous_day", _pitching_prev_day,
         ("bp_pitches_prev1_diff", "bp_pitches_prev3_diff", "bp_quality_diff", "bp_closer_avail_diff",
          "bp_hi_lev_avail_diff")),
    Case("starter_change_between_quote_and_snapshot", _starter_change,
         ("sp_fip_diff", "sp_k_pct_diff", "sp_exp_ip_diff", "sp_yes_throws_left", "sp_opp_throws_left",
          "off_platoon_adv_diff")),
    Case("lineup_posted_after_cutoff", _lineup_after_cutoff,
         ("off_lineup_posted_yes", "off_lineup_posted_opp", "off_quality_diff", "off_platoon_adv_diff",
          "off_lineup_changes_diff")),
    Case("forecast_after_cutoff", _forecast_after_cutoff,
         ("temp_f", "wind_out_mph", "humidity_pct", "roof_closed", "nc_permuted_temp", "nc_permuted_wind")),
    Case("game_outcome_after_cutoff", _outcome_after_cutoff, ("park_run_factor",)),
    Case("odds_after_quote", _odds_after_quote, ("book_move_since_news", "latency_gap", "gap_persistence_minutes"),
         lookahead=FIVE_MIN),
    Case("odds_after_cutoff_flip_gap_history", _odds_flip_gap_history, ("gap_persistence_minutes",)),
    Case("later_quote", _later_quote,
         ("market_move_60m", "market_move_since_news", "latency_gap", "gap_persistence_minutes")),
    Case("news_after_cutoff", _news_after_cutoff,
         ("minutes_since_last_news", "book_move_since_news", "market_move_since_news", "latency_gap")),
    Case("batting_line_after_cutoff", _batting_after_cutoff, ("off_quality_diff", "off_iso_diff", "off_k_pct_diff")),
    Case("player_row_after_cutoff", _player_after_cutoff,
         ("sp_yes_throws_left", "sp_opp_throws_left", "off_platoon_adv_diff")),
    Case("venue_row_after_cutoff", _venue_after_cutoff, ("roof_closed", "wind_out_mph")),
    Case("schedule_row_after_cutoff", _schedule_after_cutoff, ("nc_permuted_temp",)),
]
IDS = [c.name for c in CASES]


def leak(monkeypatch, lookahead: int, *, guard: bool) -> None:
    """Break the point-in-time filters: every lookup sees rows up to `lookahead` past its cutoff."""
    def asof(left, right, *, on, by=(), cols=(), order=()):
        lon, ron = on
        shifted = left.assign(**{lon: pit.shift(left[lon].to_numpy(), lookahead)})
        return pit._asof_raw(shifted, right, on=on, by=by, cols=cols, order=order)

    def available(t, cutoff):
        t = np.asarray(t, dtype=np.int64)
        return (t != pit.NAT) & (t <= pit.shift(cutoff, lookahead))

    monkeypatch.setattr(pit, "asof", asof)
    monkeypatch.setattr(pit, "available", available)
    if not guard:
        monkeypatch.setattr(mlb, "_check_invariants", lambda *a, **k: None)


def changed(a: mlb.FeatureBuild, b: mlb.FeatureBuild) -> set[str]:
    """Features whose value or available_at differs for at least one observation."""
    key = ["observation_id", "feature_name"]
    x, y = a.values.set_index(key), b.values.set_index(key).reindex(a.values.set_index(key).index)
    diff = pd.Series(False, index=x.index)
    for col in ("value", "available_at"):
        both_null = x[col].isna() & y[col].isna()
        diff |= ~both_null & ((x[col] != y[col]) | x[col].isna() | y[col].isna())
    return set(diff[diff].index.get_level_values("feature_name"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_base_world_populates_the_targeted_features():
    fb = build(world())
    v = fb.values.set_index(["observation_id", "feature_name"])["value"]
    targeted = {f for c in CASES for f in c.affected}
    null = {f for f in targeted if np.isnan(v[(YES, f)]) or np.isnan(v[(OPP, f)])}
    # NYY's lineup is projected at the cutoff, so the changes diff is NULL until a lineup (leaked) is posted.
    assert null == {"off_lineup_changes_diff"}, null
    assert v[(YES, "off_lineup_posted_yes")] == 0 and v[(YES, "off_lineup_posted_opp")] == 1
    assert (fb.values["cutoff_time"] == CUT).all() and (fb.values["snapshot_time"] == SNAP).all()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_rows_available_after_the_cutoff_are_ignored(case):
    base = build(world())
    hit = build(with_rows(world(), case.extra()))
    pd.testing.assert_frame_equal(base.values, hit.values)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_same_rows_change_features_when_the_filter_leaks(case, monkeypatch):
    """Proof the test above can fail: break the filters and the injected rows get through."""
    leak(monkeypatch, case.lookahead, guard=False)
    base = build(world())
    hit = build(with_rows(world(), case.extra()))
    missed = set(case.affected) - changed(base, hit)
    assert not missed, f"{case.name}: leaked rows did not reach {sorted(missed)}"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_output_guard_refuses_what_a_broken_filter_lets_through(case, monkeypatch):
    leak(monkeypatch, case.lookahead, guard=True)
    with pytest.raises(mlb.FeatureLeakError, match="available_at > cutoff_time"):
        build(with_rows(world(), case.extra()))



def test_log_odds_gap_reads_the_observations_consensus_not_later_odds(monkeypatch):
    """cross_market_gap_log_odds uses the observation's own p_consensus, whose odds the
    observations CHECK holds to consensus_available_at <= quote_ts (test_db_invariants). Odds that
    arrive after the quote must not reach it, even with the point-in-time filters broken."""
    leak(monkeypatch, FIVE_MIN, guard=False)
    base = build(world())
    hit = build(with_rows(world(), _odds_after_quote()))
    assert "book_move_since_news" in changed(base, hit)          # the leak is live
    assert "cross_market_gap_log_odds" not in changed(base, hit)
    v = base.values
    r = v[(v["observation_id"] == YES) & (v["feature_name"] == "cross_market_gap_log_odds")].iloc[0]
    assert np.isfinite(r["value"]) and r["available_at"] <= r["cutoff_time"]

# --- The boundary is exact: available at the cutoff is used, one nanosecond later is not -------------------

ONE_NS = pd.Timedelta(1, "ns")

BOUNDARY = {
    "forecast": (lambda t: {"mlb_weather": [forecast("syn-mlb-1", t, temperature_f=45.0)]}, "temp_f"),
    "starter": (lambda t: {"mlb_probable_starters": [{"game_id": "syn-mlb-1", "side": "home", "team": "NYY",
                                                      "player_id": 3, "available_at": t}]}, "sp_yes_throws_left"),
    "lineup": (lambda t: {"mlb_lineups": lineup("NYY", NYY_NINE, t)}, "off_lineup_posted_yes"),
    "odds": (lambda t: {"consensus_odds": [odds("C", t, 0.75)]}, "book_move_since_news"),
    "relief_line": (lambda t: {"mlb_pitching_lines": [pline(401, "TB", "2025-05-31", starter=False, pitches=30,
                                                            saves=1, available_at=t)]}, "bp_pitches_prev1_diff"),
    "news": (lambda t: {"info_events": [news(t, "starter")]}, "minutes_since_last_news"),
}


@pytest.mark.parametrize("kind", list(BOUNDARY))
def test_cutoff_boundary_is_inclusive_and_exact(kind):
    extra, feature = BOUNDARY[kind]
    base = build(world())
    at_cut = build(with_rows(world(), extra(CUT)))
    assert feature in changed(base, at_cut)
    row = at_cut.values[(at_cut.values["observation_id"] == YES) & (at_cut.values["feature_name"] == feature)]
    assert row["available_at"].iloc[0] == CUT
    just_after = build(with_rows(world(), extra(CUT + ONE_NS)))
    pd.testing.assert_frame_equal(base.values, just_after.values)


# --- gap_persistence_minutes: each earlier quote sees only the odds known at its own quote_ts --------------

def gap_persistence(fb: mlb.FeatureBuild, oid: str = YES) -> float:
    v = fb.values
    return float(v.loc[(v["observation_id"] == oid) & (v["feature_name"] == "gap_persistence_minutes"), "value"].iloc[0])


def test_gap_history_ignores_odds_that_arrived_after_the_quote(monkeypatch):
    """Book A's 0.40 at cut - 12 was known at the cutoff but not at the cut - 30 quote.

    At the cutoff it is superseded by A's cut - 10 price and it follows the news, so the clean build is
    unchanged: both gaps have held since the cut - 70 quote (no consensus there = the break). A lookup that
    peeks 19 minutes ahead gives the cut - 30 quote a consensus of (0.40 + 0.57) / 2 = 0.485, flipping both
    gaps there, so persistence drops to 30. The output guard cannot see this one (every input precedes the
    cutoff), so the asof filter is what holds.
    """
    extra = {"consensus_odds": [odds("A", CUT - td(minutes=12), 0.40)]}
    base, hit = build(world()), build(with_rows(world(), extra))
    pd.testing.assert_frame_equal(base.values, hit.values)
    assert gap_persistence(hit) == 70 and gap_persistence(hit, OPP) == 70
    leak(monkeypatch, 19 * pit.MIN, guard=True)
    leaky = build(with_rows(world(), extra))
    assert gap_persistence(leaky) == 30 and gap_persistence(leaky, OPP) == 30


def test_guard_attributes_leaked_gap_history_to_gap_persistence(monkeypatch):
    """With later odds leaking into every earlier quote, the first refused row is gap_persistence_minutes."""
    leak(monkeypatch, FOREVER, guard=True)
    with pytest.raises(mlb.FeatureLeakError, match="available_at > cutoff_time: gap_persistence_minutes"):
        build(with_rows(world(), _odds_flip_gap_history()))


def test_snapshot_time_is_not_the_cutoff():
    """Rows between quote_ts and snapshot_time are after the cutoff (26.11), even though they precede the snapshot."""
    assert CUT < AFTER < SNAP
    starter = build(with_rows(world(), _starter_change()))
    ref = starter.values.loc[starter.values["feature_name"] == "sp_fip_diff", "source_ref"]
    assert set(ref) == {"sp yes=1 opp=2", "sp yes=2 opp=1"}


def test_leak_free_output_is_accepted_by_the_database(syn_db):
    everything = world()
    for case in CASES:
        everything = with_rows(everything, case.extra())
    fb = build(everything)
    pd.testing.assert_frame_equal(fb.values, build(world()).values)
    assert syn_db.insert("feature_definitions", fb.definitions) == len(mlb.REGISTRY)
    assert syn_db.insert("feature_values", fb.values) == 2 * len(mlb.REGISTRY)
