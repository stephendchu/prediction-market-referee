"""MLB Level 1 feature builder: exact values on hand-built inputs (spec Sec 8, 26.11, 26.16).

Every expected number below is worked out by hand from the formulas in
src/features/mlb.py. The observation factory puts quote_ts (the cutoff) one
minute before the snapshot.
"""

from __future__ import annotations

import math
from datetime import date, timedelta as td

import numpy as np
import pandas as pd
import pytest
from factories import GAME_DATE, START, frame, observation, provenance

from src.features import common as pit
from src.features import mlb
from src.storage.schemas import FEATURE_DEFINITIONS, FEATURE_FAMILIES, FEATURE_VALUES

CUT = START - td(minutes=31)      # quote_ts of the T-30 observation
EARLY = START - td(days=1)
YES, OPP = "SYN-NYY@T-30", "SYN-TB@T-30"
DIFFS = [n for n in mlb.feature_families() if n.endswith("_diff")]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def rows(*dicts: dict, at: pd.Timestamp = EARLY) -> pd.DataFrame:
    out = []
    for d in dicts:
        d = dict(d)
        out.append(provenance(d.pop("available_at", at)) | d)
    return pd.DataFrame(out)


def obs_pair(offset: int = 30) -> pd.DataFrame:
    common = {"volume": 1000.0, "open_interest": 500.0}
    home = observation(offset, p_consensus=0.58, **common)
    away = observation(offset, observation_id=f"SYN-TB@T-{offset}", contract_id="SYN-TB", yes_team="TB",
                       yes_side="away", market_yes_bid=0.43, market_yes_ask=0.45, p_mid=0.44, p_consensus=0.42,
                       **common)
    return frame(home, away)


def build(obs: pd.DataFrame, tables: dict, seed: int = 0) -> mlb.FeatureBuild:
    return mlb.build_features(obs, tables, build_run_id="test-build", is_synthetic=True, seed=seed)


def row(fb: mlb.FeatureBuild, name: str, oid: str = YES) -> pd.Series:
    v = fb.values
    hit = v[(v["observation_id"] == oid) & (v["feature_name"] == name)]
    assert len(hit) == 1, (name, oid)
    return hit.iloc[0]


def val(fb: mlb.FeatureBuild, name: str, oid: str = YES) -> float:
    return float(row(fb, name, oid)["value"])


def day_after(d: str, hours: int = 3) -> pd.Timestamp:
    return pd.Timestamp(d, tz="UTC") + td(days=1, hours=hours)


def pline(pid: int, team: str, d: str, *, starter: bool, **stats) -> dict:
    base = {"game_id": f"g-{team}-{d}", "official_date": date.fromisoformat(d), "side": "home", "team": team,
            "player_id": pid, "is_starter": starter, "outs": 0, "batters_faced": 0, "pitches": 0, "strikeouts": 0,
            "walks": 0, "hit_by_pitch": 0, "home_runs": 0, "saves": 0, "holds": 0, "available_at": day_after(d)}
    return base | stats


def bline(pid: int, team: str, slot: int, **stats) -> dict:
    base = {"game_id": "g-prev", "official_date": date(2025, 5, 31), "side": "home" if team == "NYY" else "away",
            "team": team, "player_id": pid, "batting_order": slot, "plate_appearances": 4, "at_bats": 4, "hits": 0,
            "doubles": 0, "triples": 0, "home_runs": 0, "walks": 0, "hit_by_pitch": 0, "strikeouts": 0,
            "available_at": day_after("2025-05-31")}
    return base | stats


def starter(side: str, team: str, pid: int, at: pd.Timestamp = EARLY) -> dict:
    return {"game_id": "syn-mlb-1", "side": side, "team": team, "player_id": pid, "available_at": at}


def game(gid: str, d: date, venue: int, home: int | None = None, away: int | None = None, **over) -> dict:
    final = home is not None
    out_at = pd.Timestamp(d, tz="UTC") + td(days=1, hours=3) if final else None
    row_ = {"game_id": gid, "sport": "mlb", "season": d.year, "official_date": d, "game_type": "R",
            "scheduled_start": pd.Timestamp(d, tz="UTC") + td(hours=23), "home_team": "NYY", "away_team": "TB",
            "venue_id": venue, "home_score": home, "away_score": away,
            "home_win": (home > away) if final else None, "outcome_available_at": out_at,
            "available_at": out_at if final else START - td(days=3)}
    return row_ | over


def venue(vid: int, roof: str | None = "open", az: float | None = 0.0) -> dict:
    return {"venue_id": vid, "name": f"V{vid}", "roof_type": roof, "cf_azimuth_deg": az,
            "available_at": pd.Timestamp("2023-01-01", tz="UTC")}


def forecast(gid: str = "syn-mlb-1", at: pd.Timestamp = START - td(hours=6), **over) -> dict:
    base = {"game_id": gid, "kind": "forecast", "forecast_for": START, "temperature_f": 70.0,
            "wind_speed_mph": 10.0, "wind_direction_deg": 180.0, "humidity_pct": 50.0, "precip_prob": 0.1,
            "roof_status": "unknown", "available_at": at}
    return base | over


def quote(cid: str, t: pd.Timestamp, bid: float, ask: float) -> dict:
    return {"contract_id": cid, "game_id": "syn-mlb-1", "quote_ts": t, "quote_source": "synthetic", "yes_bid": bid,
            "yes_ask": ask, "no_bid": 1 - ask, "no_ask": 1 - bid, "available_at": t + td(seconds=1)}


def odds(book: str, t: pd.Timestamp, p_home: float) -> dict:
    return {"game_id": "syn-mlb-1", "book": book, "market_type": "ml", "odds_ts": t - td(seconds=5),
            "p_home_devig": p_home, "p_away_devig": 1 - p_home, "devig_method": "proportional", "available_at": t}


def event(t: pd.Timestamp, kind: str = "injury", gid: str | None = "syn-mlb-1", team: str | None = "NYY",
          first_seen: bool = False) -> dict:
    return {"event_id": f"e-{kind}-{t.value}-{team}", "game_id": gid, "team": team, "type": kind,
            "first_seen": first_seen, "available_at": t}


# --- Worlds -----------------------------------------------------------------------------------------

LINE1 = dict(outs=30, batters_faced=40, strikeouts=10, walks=4, home_runs=1, pitches=95, ground_balls=10,
             fly_balls=5)


def starter_world() -> dict[str, pd.DataFrame]:
    return {"mlb_probable_starters": rows(starter("home", "NYY", 1), starter("away", "TB", 2)),
            "mlb_players": rows({"player_id": 1, "throws": "L"}, {"player_id": 2, "throws": "R"}),
            "mlb_pitching_lines": rows(pline(1, "NYY", "2025-05-27", starter=True, **LINE1))}


def offense_world() -> dict[str, pd.DataFrame]:
    """NYY: no lineup posted at T-30 (projected from g-prev, 100 * slot order codes); TB: posted."""
    nyy = [bline(100 + i, "NYY", 100 * i, hits=1, strikeouts=1) for i in range(1, 10)]
    nyy.append(bline(120, "NYY", 301, plate_appearances=1, at_bats=1))          # substitute: not a starter
    tb = [bline(200 + i, "TB", i, at_bats=3, walks=1) for i in range(1, 10)]
    tb_lineup = [{"game_id": "syn-mlb-1", "side": "away", "team": "TB", "batting_order": i + 1, "player_id": p,
                  "available_at": START - td(hours=2)} for i, p in enumerate([*range(201, 209), 210])]
    nyy_lineup = [{"game_id": "syn-mlb-1", "side": "home", "team": "NYY", "batting_order": i + 1, "player_id": p,
                   "available_at": START - td(minutes=20)} for i, p in enumerate([*range(101, 108), 120, 121])]
    bats = {**{p: "L" for p in (101, 102, 103)}, **{p: "R" for p in (104, 105, 106)}, 107: "S",
            **{p: "R" for p in [*range(201, 209), 210]}}
    players = [{"player_id": 1, "throws": "L"}, {"player_id": 2, "throws": "R"},
               *({"player_id": p, "bats": b} for p, b in bats.items())]
    return {"mlb_batting_lines": rows(*nyy, *tb), "mlb_lineups": rows(*tb_lineup, *nyy_lineup),
            "mlb_players": rows(*players),
            "mlb_probable_starters": rows(starter("home", "NYY", 1), starter("away", "TB", 2))}


def bullpen_world() -> dict[str, pd.DataFrame]:
    rel = dict(starter=False, outs=3, batters_faced=4, strikeouts=1)
    nyy = [(301, d, p, 1, 0) for d, p in (("05-20", 15), ("05-22", 12), ("05-25", 14), ("05-30", 15), ("05-31", 20))]
    nyy += [(302, d, p, 0, 1) for d, p in (("05-21", 14), ("05-23", 16), ("05-26", 13), ("05-31", 30))]
    nyy += [(303, "05-24", 10, 0, 1), (303, "05-29", 18, 0, 1), (304, "05-15", 12, 0, 1),
            (304, "04-20", 12, 5, 0),    # 42 days back: outside the 30-day leverage window
            (305, "05-31", 40, 0, 0)]
    tb = [(401, "05-25", 14, 1, 0), (401, "05-28", 15, 1, 0), (402, "05-27", 11, 0, 1)]
    lines = [pline(p, team, f"2025-{d}", pitches=n, saves=sv, holds=h, **rel)
             for team, group in (("NYY", nyy), ("TB", tb)) for p, d, n, sv, h in group]
    lines.append(pline(1, "NYY", "2025-05-31", starter=True, outs=18, pitches=100, strikeouts=5))  # not bullpen
    return {"mlb_pitching_lines": rows(*lines)}


def env_world(roof: str | None = "open", az: float | None = 0.0, **fc) -> dict[str, pd.DataFrame]:
    return {"games": rows(game("syn-mlb-1", GAME_DATE, 1)), "venues": rows(venue(1, roof, az)),
            "mlb_weather": rows(forecast(**fc))}


def latency_world() -> dict[str, pd.DataFrame]:
    q = [quote("SYN-NYY", CUT - td(minutes=70), 0.51, 0.53), quote("SYN-NYY", CUT - td(minutes=30), 0.49, 0.51),
         quote("SYN-NYY", CUT, 0.55, 0.57), quote("SYN-TB", CUT - td(minutes=30), 0.47, 0.49),
         quote("SYN-TB", CUT, 0.43, 0.45)]
    o = [odds("A", CUT - td(minutes=40), 0.55), odds("B", CUT - td(minutes=35), 0.57),
         odds("A", CUT - td(minutes=10), 0.60)]
    e = [event(CUT - td(minutes=20)), event(CUT - td(minutes=5), first_seen=True),
         event(CUT - td(minutes=3), kind="weather"), event(CUT - td(minutes=25), gid=None, team="TB")]
    return {"market_quotes": rows(*q), "consensus_odds": rows(*o), "info_events": rows(*e)}


def merged(*worlds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for w in worlds:
        for k, df in w.items():
            out[k] = pd.concat([out[k], df], ignore_index=True) if k in out else df
    out["mlb_players"] = out["mlb_players"].groupby("player_id", as_index=False).last()
    out["mlb_probable_starters"] = out["mlb_probable_starters"].drop_duplicates(["game_id", "side", "available_at"])
    return out


def full_world() -> dict[str, pd.DataFrame]:
    return merged(starter_world(), offense_world(), bullpen_world(), env_world(), latency_world())


# ---------------------------------------------------------------------------
# Registry and output contract
# ---------------------------------------------------------------------------

def test_definitions_match_schema_and_registry():
    d = mlb.feature_definitions()
    assert list(d.columns) == FEATURE_DEFINITIONS.column_names
    assert d["feature_name"].is_unique and len(d) == 47
    assert set(d["family"]) <= set(FEATURE_FAMILIES)
    assert mlb.feature_families() == dict(zip(d["feature_name"], d["family"]))
    assert (d["version"] == mlb.FEATURE_VERSION).all() and d["hypothesis_id"].notna().all()
    assert d["availability_rule"].str.len().gt(10).all()
    nc = d[d["family"] == "negative_control"]
    assert set(nc["feature_name"]) == {"nc_random", "nc_team_hash", "nc_permuted_temp", "nc_permuted_wind"}
    assert (nc["hypothesis_id"] == "H_MLB_NC").all()
    h3 = d[d["hypothesis_id"] == "H_MLB_003"]
    assert set(h3["feature_name"]) == {"cross_market_gap", "gap_inside_fee_band", "gap_persistence_minutes"}
    assert (h3["family"] == "market").all()


def test_output_contract_on_full_world():
    obs = obs_pair()
    fb = build(obs, full_world())
    v = fb.values
    assert list(v.columns) == FEATURE_VALUES.column_names
    assert len(v) == 2 * 47 and not v.duplicated(["observation_id", "feature_name"]).any()
    assert (v["cutoff_time"] == pd.Timestamp(CUT)).all()
    assert (v["snapshot_time"] == pd.Timestamp(START - td(minutes=30))).all()
    has = v["available_at"].notna()
    assert (v.loc[has, "available_at"] <= v.loc[has, "cutoff_time"]).all()
    assert (has == v["value"].notna()).all()
    assert (v["build_run_id"] == "test-build").all() and v["is_synthetic"].all()
    # Everything is populated except where an input is genuinely missing: TB has no quote 60 min back, TB's
    # starter has no previous start, NYY's lineup is not posted, and the date has a single game.
    null = set(map(tuple, v.loc[v["value"].isna(), ["observation_id", "feature_name"]].to_numpy()))
    both = {"sp_last_pitches_diff", "sp_days_rest_diff", "off_lineup_changes_diff", "nc_permuted_temp",
            "nc_permuted_wind"}
    assert null == {(OPP, "market_move_60m")} | {(o, n) for o in (YES, OPP) for n in both}


def test_empty_inputs_give_nulls_and_empty_obs_gives_empty_frame():
    fb = build(obs_pair(), {})
    v = fb.values.set_index(["observation_id", "feature_name"])["value"]
    for name in ("sp_fip_diff", "off_quality_diff", "bp_quality_diff", "bp_pitches_prev1_diff", "temp_f",
                 "park_run_factor", "minutes_since_last_news", "nc_permuted_temp", "market_move_60m",
                 "gap_persistence_minutes"):
        assert math.isnan(v[(YES, name)]), name
    assert v[(YES, "market_logit")] == pytest.approx(math.log(0.56 / 0.44))
    empty = build(obs_pair().iloc[:0], full_world())
    assert empty.values.empty and list(empty.values.columns) == FEATURE_VALUES.column_names
    with pytest.raises(KeyError):
        build(obs_pair().drop(columns=["quote_ts"]), {})


# ---------------------------------------------------------------------------
# Starter
# ---------------------------------------------------------------------------

def test_starter_shrinkage_on_fallback_league():
    fb = build(obs_pair(), starter_world())
    # YES starter: 10 IP, FIP numerator 13*1 + 3*4 - 2*10 = 5; league < 500 IP -> fallback raw 1.07.
    assert val(fb, "sp_fip_diff") == pytest.approx((5 + 40 * 1.07) / 50 - 1.07)          # -0.114
    assert val(fb, "sp_k_pct_diff") == pytest.approx((10 + 150 * 0.227) / 190 - 0.227)
    assert val(fb, "sp_bb_pct_diff") == pytest.approx((4 + 150 * 0.086) / 190 - 0.086)
    assert val(fb, "sp_gb_pct_diff") == pytest.approx((10 + 100 * 0.52) / 115 - 0.52)
    assert val(fb, "sp_hr9_diff") == pytest.approx(9 * (1 + 40 * 1.22 / 9) / 50 - 1.22)  # -0.064
    assert val(fb, "sp_exp_ip_diff") == pytest.approx((10 + 5.3 * 2) / 3 - 5.3)
    assert math.isnan(val(fb, "sp_last_pitches_diff"))       # opponent has no previous start
    assert val(fb, "sp_yes_throws_left") == 1 and val(fb, "sp_opp_throws_left") == 0
    r = row(fb, "sp_fip_diff")
    assert r["source_ref"] == "sp yes=1 opp=2" and r["available_at"] == EARLY
    assert val(fb, "sp_fip_diff", OPP) == pytest.approx(-val(fb, "sp_fip_diff"))


def test_starter_league_window_and_threshold():
    world = starter_world()
    filler = pline(99, "BOS", "2025-05-01", starter=False, outs=1500, batters_faced=6000, home_runs=50, walks=150,
                   hit_by_pitch=10, strikeouts=400)
    stale = pline(98, "BOS", "2024-04-01", starter=False, outs=1500, batters_faced=6000, home_runs=500)  # > 365 d
    world["mlb_pitching_lines"] = pd.concat([world["mlb_pitching_lines"], rows(filler, stale)], ignore_index=True)
    fb = build(obs_pair(), world)
    r = (13 * 51 + 3 * (154 + 10) - 2 * 410) / 510                  # league = filler + YES starter's line
    assert val(fb, "sp_fip_diff") == pytest.approx((5 + 40 * r) / 50 - r)
    k = 410 / 6040
    assert val(fb, "sp_k_pct_diff") == pytest.approx((10 + 150 * k) / 190 - k)
    assert row(fb, "sp_fip_diff")["available_at"] == EARLY          # announcement is the latest input


def test_expected_innings_uses_last_five_starts():
    outs = [18, 15, 21, 12, 18, 24]
    days_ = ["2025-05-01", "2025-05-06", "2025-05-11", "2025-05-16", "2025-05-21", "2025-05-26"]
    starts = [pline(1, "NYY", d, starter=True, outs=o, pitches=90 + i) for i, (d, o) in enumerate(zip(days_, outs))]
    relief = pline(1, "NYY", "2025-05-29", starter=False, outs=6, pitches=30)   # not a start
    world = starter_world() | {"mlb_pitching_lines": rows(*starts, relief)}
    fb = build(obs_pair(), world)
    assert val(fb, "sp_exp_ip_diff") == pytest.approx((90 / 3 + 5.3 * 2) / 7 - 5.3)     # 5.8 - 5.3
    assert val(fb, "sp_exp_ip_diff", OPP) == pytest.approx(-0.5)
    world["mlb_pitching_lines"] = pd.concat([world["mlb_pitching_lines"],
                                             rows(pline(2, "TB", "2025-05-28", starter=True, outs=15, pitches=80))])
    fb = build(obs_pair(), world)
    assert val(fb, "sp_last_pitches_diff") == 95 - 80
    assert val(fb, "sp_days_rest_diff") == 6 - 4


# ---------------------------------------------------------------------------
# Offense
# ---------------------------------------------------------------------------

W_NYY = (0.89 + 200 * 0.315) / 204                        # 4 PA, one single, league fallback
W_TB = (0.69 + 200 * 0.315) / 204                         # 4 PA, one walk


def test_projected_versus_posted_lineup_and_platoon_share():
    fb = build(obs_pair(), offense_world())
    assert val(fb, "off_quality_diff") == pytest.approx(W_NYY - (8 * W_TB + 0.315) / 9)
    assert val(fb, "off_k_pct_diff") == pytest.approx((1 + 200 * 0.227) / 204 - (8 * 45.4 / 204 + 0.227) / 9)
    assert val(fb, "off_bb_pct_diff") == pytest.approx(17.2 / 204 - (8 * 18.2 / 204 + 0.086) / 9)
    assert val(fb, "off_iso_diff") == pytest.approx(33 / 204 - (8 * 33 / 203 + 0.165) / 9)
    # NYY vs a RHP: L, L, L, S have the edge of 7 known hands; TB (all R) vs a LHP: 9 of 9.
    assert val(fb, "off_platoon_adv_diff") == pytest.approx(4 / 7 - 1)
    assert val(fb, "off_lineup_posted_yes") == 0 and val(fb, "off_lineup_posted_opp") == 1
    assert row(fb, "off_lineup_posted_opp")["available_at"] == START - td(hours=2)
    assert math.isnan(val(fb, "off_lineup_changes_diff"))    # NYY not posted
    assert row(fb, "off_quality_diff")["source_ref"] == "lineup yes=projected opp=posted"
    assert row(fb, "off_quality_diff", OPP)["source_ref"] == "lineup yes=posted opp=projected"


def test_posted_lineup_changes_and_later_version():
    fb = build(obs_pair(10), offense_world())          # T-10: NYY posted at START - 20 min
    oid = "SYN-NYY@T-10"
    assert val(fb, "off_lineup_posted_yes", oid) == 1
    assert val(fb, "off_lineup_changes_diff", oid) == 2 - 1      # NYY: 120 (was a sub) and 121; TB: 210
    nyy = (7 * W_NYY + 63 / 201 + 0.315) / 9
    assert val(fb, "off_quality_diff", oid) == pytest.approx(nyy - (8 * W_TB + 0.315) / 9)
    assert row(fb, "off_lineup_changes_diff", oid)["available_at"] == START - td(minutes=20)


def test_platoon_null_when_opposing_starter_hand_unknown():
    world = offense_world()
    world["mlb_players"] = world["mlb_players"][world["mlb_players"]["player_id"] != 2]
    fb = build(obs_pair(), world)
    assert math.isnan(val(fb, "off_platoon_adv_diff"))
    assert not math.isnan(val(fb, "off_quality_diff"))


# ---------------------------------------------------------------------------
# Bullpen
# ---------------------------------------------------------------------------

def test_bullpen_prev_day_pitches_and_availability():
    fb = build(obs_pair(), bullpen_world())
    assert val(fb, "bp_pitches_prev1_diff") == 90 - 0           # 20 + 30 + 40 on 5/31; the starter excluded
    assert val(fb, "bp_pitches_prev3_diff") == 123 - 0          # + 15 (5/30) + 18 (5/29)
    # NYY top 3 (301 sv5, 302 hld4, 303 hld2): 301 pitched both days, 302 threw 30 -> only 303. TB: 401, 402.
    assert val(fb, "bp_hi_lev_avail_diff") == 1 - 2
    assert val(fb, "bp_closer_avail_diff") == 0 - 1             # 304's 5 saves on 4/20 are outside 30 days
    assert val(fb, "bp_quality_diff") == pytest.approx((-28 + 150 * 1.07) / 164 - (-6 + 150 * 1.07) / 153)
    # TB's zero is a statement about absence as of the cutoff, so the diff is available at the cutoff.
    assert row(fb, "bp_pitches_prev1_diff")["available_at"] == CUT
    assert row(fb, "bp_quality_diff")["available_at"] == day_after("2025-05-31")
    assert val(fb, "bp_closer_avail_diff", OPP) == 1


def test_bullpen_null_without_team_history():
    w = bullpen_world()
    w["mlb_pitching_lines"] = w["mlb_pitching_lines"][w["mlb_pitching_lines"]["team"] != "TB"]
    fb = build(obs_pair(), w)
    for name in ("bp_quality_diff", "bp_pitches_prev1_diff", "bp_pitches_prev3_diff", "bp_hi_lev_avail_diff",
                 "bp_closer_avail_diff"):
        assert math.isnan(val(fb, name)), name


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("roof, az, frm, expected", [
    ("open", 0.0, 180.0, 10.0),      # wind from the south blows north = straight out to a north-facing CF
    ("open", 90.0, 180.0, 0.0),      # crosswind
    ("open", 0.0, 0.0, -10.0),       # blowing in
    ("open", 45.0, 225.0, 10.0),
    ("dome", 0.0, 180.0, 0.0),
    ("open", None, 180.0, float("nan")),
])
def test_wind_geometry(roof, az, frm, expected):
    fb = build(obs_pair(), env_world(roof, az, wind_direction_deg=frm))
    got = val(fb, "wind_out_mph")
    assert (math.isnan(got) and math.isnan(expected)) or got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("roof, fc, expected", [
    ("dome", {}, 1.0), ("open", {"precip_prob": 0.9}, 0.0),
    ("retractable", {"roof_status": "closed"}, 1.0), ("retractable", {"roof_status": "open", "precip_prob": 0.9}, 0.0),
    ("retractable", {"precip_prob": 0.5}, 1.0), ("retractable", {"temperature_f": 50.0}, 1.0),
    ("retractable", {}, 0.0), ("retractable", {"precip_prob": None}, float("nan")), (None, {}, float("nan")),
])
def test_roof_closed(roof, fc, expected):
    fb = build(obs_pair(), env_world(roof, 0.0, **fc))
    got = val(fb, "roof_closed")
    assert (math.isnan(got) and math.isnan(expected)) or got == expected


def test_forecast_values_and_availability():
    fb = build(obs_pair(), env_world(temperature_f=81.5, humidity_pct=40.0))
    assert val(fb, "temp_f") == 81.5 and val(fb, "humidity_pct") == 40.0
    assert row(fb, "temp_f")["available_at"] == START - td(hours=6)
    assert val(fb, "yes_is_home") == 1 and val(fb, "yes_is_home", OPP) == 0
    assert val(fb, "temp_f", OPP) == 81.5


def test_park_factor_shrinkage():
    past = [game("p1", date(2025, 5, 1), 1, 6, 4), game("p2", date(2025, 5, 2), 1, 7, 5),
            game("p3", date(2025, 5, 3), 1, 5, 3), game("p4", date(2025, 5, 4), 2, 4, 2),
            game("p5", date(2025, 5, 5), 2, 5, 1),
            game("old", date(2021, 6, 1), 1, 20, 10),                               # > 3 years back
            game("nores", date(2025, 5, 6), 1, 9, 9, home_win=None, outcome_available_at=None)]
    this = game("syn-mlb-1", GAME_DATE, 1, available_at=pd.Timestamp("2025-04-01", tz="UTC"))
    world = env_world() | {"games": rows(this, *past)}
    fb = build(obs_pair(), world)
    ratio = (30 / 3) / (42 / 5)
    assert val(fb, "park_run_factor") == pytest.approx((3 * ratio + 150) / 153)           # 1.003735
    assert row(fb, "park_run_factor")["available_at"] == pd.Timestamp("2025-05-06T03:00Z")   # p5's result


# ---------------------------------------------------------------------------
# Market and latency
# ---------------------------------------------------------------------------

def test_market_features():
    fb = build(obs_pair(), latency_world())
    assert val(fb, "market_logit") == pytest.approx(math.log(0.56 / 0.44))
    assert val(fb, "market_logit", OPP) == pytest.approx(-val(fb, "market_logit"))
    assert val(fb, "market_spread") == pytest.approx(0.02)
    assert val(fb, "log_volume") == pytest.approx(math.log1p(1000)) and val(fb, "log_open_interest") == pytest.approx(
        math.log1p(500))
    assert val(fb, "is_afternoon_game") == 0                      # 19:05 New York
    assert val(fb, "market_move_60m") == pytest.approx(0.56 - 0.52)
    assert math.isnan(val(fb, "market_move_60m", OPP))            # TB has no quote 60 min back
    assert val(fb, "cross_market_gap") == pytest.approx(0.02) and val(fb, "cross_market_gap", OPP) == pytest.approx(
        -0.02)
    # The model's gap is in log-odds, logit(p_consensus) - logit(p_mid), not 0.02 in probability
    gap_lo = math.log(0.58 / 0.42) - math.log(0.56 / 0.44)
    assert val(fb, "cross_market_gap_log_odds") == pytest.approx(gap_lo)
    assert val(fb, "cross_market_gap_log_odds", OPP) == pytest.approx(-gap_lo)
    day = obs_pair()
    day["event_start_time"] = pd.Timestamp("2025-06-01T17:05Z")   # 13:05 EDT
    assert val(build(day, {}), "is_afternoon_game") == 1



def test_log_odds_gap_definition_and_null_without_consensus():
    """H_MLB_L1_GAP's feature, family market, available at the cutoff; NULL without a consensus."""
    d = mlb.feature_definitions().set_index("feature_name")
    assert d.loc["cross_market_gap_log_odds", "family"] == "market"
    assert d.loc["cross_market_gap_log_odds", "hypothesis_id"] == "H_MLB_L1_GAP"
    assert d.loc["cross_market_gap", "hypothesis_id"] == "H_MLB_003"          # the probability form is unchanged
    obs = obs_pair()
    obs["p_consensus"] = [np.nan, 0.42]
    fb = build(obs, latency_world())
    assert math.isnan(val(fb, "cross_market_gap_log_odds"))
    assert val(fb, "cross_market_gap_log_odds", OPP) == pytest.approx(math.log(0.42 / 0.58) - math.log(0.44 / 0.56))
    r = row(fb, "cross_market_gap_log_odds", OPP)
    assert r["available_at"] == r["cutoff_time"]

def test_latency_features():
    fb = build(obs_pair(), latency_world())
    assert val(fb, "minutes_since_last_news") == pytest.approx(20)   # first_seen / weather events ignored
    assert val(fb, "market_move_since_news") == pytest.approx(0.56 - 0.50)
    assert val(fb, "book_move_since_news") == pytest.approx((0.60 + 0.57) / 2 - (0.55 + 0.57) / 2)
    assert val(fb, "latency_gap") == pytest.approx(0.025 - 0.06)
    assert val(fb, "book_move_since_news", OPP) == pytest.approx(-0.025)
    assert row(fb, "book_move_since_news")["available_at"] == CUT - td(minutes=10)


def test_latency_team_level_news_and_window():
    w = latency_world()
    ev = w["info_events"]
    w["info_events"] = ev[ev["game_id"].isna()]                   # only the TB team-level event at cut - 25
    fb = build(obs_pair(), w)
    assert val(fb, "minutes_since_last_news") == pytest.approx(25)
    w["info_events"] = rows(event(START - td(minutes=200)))       # before start - 180 min
    fb = build(obs_pair(), w)
    for name in ("minutes_since_last_news", "book_move_since_news", "market_move_since_news", "latency_gap"):
        assert math.isnan(val(fb, name)), name


# --- H_MLB_003: fee band and persistence ------------------------------------------------------------
# Taker fee, rate 0.07, $100 stake. YES at 0.57: Q = 175, fee = ceil(0.07*175*0.57*0.43 = $3.0025) = $3.01,
# 0.0172 per contract, band 0.01 + 0.0172 = 0.0272. NO at 1 - 0.55 = 0.45: Q = 222,
# fee = ceil(0.07*222*0.45*0.55 = $3.846) = $3.85, 0.017342 per contract, band 0.027342.

def with_consensus(p: float, oid_row: int = 0) -> pd.DataFrame:
    obs = obs_pair()
    obs.loc[oid_row, "p_consensus"] = p
    return obs


def test_gap_inside_fee_band_at_the_favored_ask():
    fb = build(obs_pair(), {})
    assert val(fb, "gap_inside_fee_band") == 1 and val(fb, "gap_inside_fee_band", OPP) == 1   # |gap| 0.02
    assert row(fb, "gap_inside_fee_band")["available_at"] == CUT
    assert row(fb, "gap_inside_fee_band")["source_ref"] == "fee_schedule=quadratic-taker-v1"
    assert val(build(with_consensus(0.56 + 0.0272), {}), "gap_inside_fee_band") == 1   # boundary is inclusive
    assert val(build(with_consensus(0.56 + 0.0273), {}), "gap_inside_fee_band") == 0
    # gap -0.0273 favors NO: inside the NO band (0.027342) though outside the YES band (0.0272).
    assert val(build(with_consensus(0.56 - 0.0273), {}), "gap_inside_fee_band") == 1
    assert val(build(with_consensus(0.56 - 0.0274), {}), "gap_inside_fee_band") == 0
    assert math.isnan(val(build(with_consensus(np.nan), {}), "gap_inside_fee_band"))


def test_gap_inside_fee_band_contract_count_uses_the_stake():
    # $0.50 buys no YES contract at 0.57 (NULL) but one NO at 0.45: fee ceil(0.07*0.45*0.55) = $0.02, band 0.03.
    small = dict(stake_dollars=0.5)
    fb = mlb.build_features(obs_pair(), {}, build_run_id="t", is_synthetic=True, **small)
    assert math.isnan(val(fb, "gap_inside_fee_band"))
    for p, inside in ((0.56 - 0.03, 1), (0.56 - 0.031, 0)):
        fb = mlb.build_features(with_consensus(p), {}, build_run_id="t", is_synthetic=True, **small)
        assert val(fb, "gap_inside_fee_band") == inside
    with pytest.raises(ValueError):
        mlb.build_features(obs_pair(), {}, build_run_id="t", is_synthetic=True, stake_dollars=0.0)


def test_gap_inside_fee_band_uses_the_schedule_in_force_on_the_quote_date(monkeypatch):
    from src.backtest import execution as ex
    old = ex.QuadraticFeeSchedule(fee_schedule_id="t-old", maker_or_taker="taker", effective_from=date(2021, 1, 1),
                                  effective_to=date(2025, 5, 31), params={"rate": 0.07})
    new = ex.QuadraticFeeSchedule(fee_schedule_id="t-new", maker_or_taker="taker", effective_from=date(2025, 6, 1),
                                  params={"rate": 0.35})
    registry = [(old, new)]
    monkeypatch.setattr(mlb, "fee_schedule_for", lambda when, role="taker": ex.fee_schedule_for(when, role, registry[0]))
    obs = with_consensus(0.60)       # gap 0.04: outside the 0.07 band, inside the 0.35 one (0.01 + 15.02 / 175)
    fb = build(obs, {})
    assert val(fb, "gap_inside_fee_band") == 1 and row(fb, "gap_inside_fee_band")["source_ref"] == "fee_schedule=t-new"
    obs["quote_ts"] = pd.Timestamp("2025-05-31T23:59:59Z")                  # previous UTC date
    fb = build(obs, {})
    assert val(fb, "gap_inside_fee_band") == 0 and row(fb, "gap_inside_fee_band")["source_ref"] == "fee_schedule=t-old"
    registry[0] = (new,)                                                    # no schedule covers 2025-05-31
    assert math.isnan(val(build(obs, {}), "gap_inside_fee_band"))


def gap_world(mids: dict[int, float] | None = None, odds_at: tuple[int, ...] = (210, 180, 150, 120, 90, 60, 30),
              extra_odds: tuple[dict, ...] = ()) -> dict[str, pd.DataFrame]:
    """NYY quotes at cut - m minutes (midpoint, +-0.01 spread); book A at 0.60 home at cut - `odds_at` minutes.

    Gaps vs the consensus known at each quote: m=200 -0.02 (breaks; outside the window), 150 +0.05, 100 -0.03
    (breaks), 40 +0.04, 10 +0.03. The observation's own gap is +0.02.
    """
    mids = {200: 0.62, 150: 0.55, 100: 0.63, 40: 0.56, 10: 0.57} if mids is None else mids
    q = [quote("SYN-NYY", CUT - td(minutes=m), mid - 0.01, mid + 0.01) for m, mid in mids.items()]
    o = [odds("A", CUT - td(minutes=m), 0.60) for m in odds_at]
    return {"market_quotes": rows(*q), "consensus_odds": rows(*o, *extra_odds)}


def persistence(obs: pd.DataFrame, tables: dict, oid: str = YES, **kw) -> float:
    fb = mlb.build_features(obs, tables, build_run_id="t", is_synthetic=True, **kw)
    return val(fb, "gap_persistence_minutes", oid)


def test_gap_persistence_minutes():
    base = {200: 0.62, 150: 0.55, 100: 0.63, 40: 0.56, 10: 0.57}
    fb = build(obs_pair(), gap_world())
    assert val(fb, "gap_persistence_minutes") == pytest.approx(100)     # last broke at the cut - 100 quote
    assert row(fb, "gap_persistence_minutes")["available_at"] == CUT
    assert math.isnan(val(fb, "gap_persistence_minutes", OPP))         # TB: no earlier quote at all
    assert persistence(obs_pair(), gap_world(base | {10: 0.595})) == pytest.approx(10)   # |gap| 0.005 < 0.01
    no_100 = {m: p for m, p in base.items() if m != 100}
    assert persistence(obs_pair(), gap_world(no_100)) == 180            # the cut - 200 break is capped
    assert persistence(obs_pair(), gap_world({180: 0.62, 40: 0.56})) == 180   # exactly 180 back
    two = pd.concat([obs_pair(), obs_pair(10)], ignore_index=True)            # T-10 cutoff = T-30 cutoff + 20 min
    assert persistence(two, gap_world({170: 0.63, 40: 0.56})) == pytest.approx(170)
    assert persistence(two, gap_world({170: 0.63, 40: 0.56}), "SYN-NYY@T-10") == 180     # 190 min: capped
    assert persistence(with_consensus(0.54), gap_world()) == pytest.approx(10)   # own gap -0.02: +0.03 breaks
    assert persistence(with_consensus(0.565), gap_world()) == 0          # own |gap| < 0.01
    assert math.isnan(persistence(with_consensus(np.nan), gap_world()))
    assert math.isnan(persistence(obs_pair(), {}))                        # no earlier quote: unknown, not 180


def test_gap_persistence_needs_evaluated_history():
    """No break: minutes since the first quote with a consensus (a lower bound), NULL if none is in the window."""
    assert persistence(obs_pair(), gap_world({150: 0.55, 40: 0.56, 10: 0.57})) == pytest.approx(150)
    assert persistence(obs_pair(), gap_world({40: 0.56})) == pytest.approx(40)
    assert persistence(obs_pair(), gap_world({200: 0.55, 40: 0.56})) == 180       # held since 200 back: capped
    assert persistence(obs_pair(), gap_world({180: 0.55})) == 180                 # exactly 180 back, in the window
    assert math.isnan(persistence(obs_pair(), gap_world({200: 0.55})))            # evaluated only before the window
    assert math.isnan(persistence(obs_pair(), gap_world({181: 0.62})))            # a break, but outside the window
    # Quotes without a consensus (book A 170+ min old) are breaks, but they are not history: unknown.
    no_consensus = gap_world({40: 0.56, 10: 0.57}, odds_at=(210,))
    assert math.isnan(persistence(obs_pair(), no_consensus))
    assert persistence(obs_pair(), no_consensus, consensus_max_age_min=240.0) == pytest.approx(40)
    # Own |gap| < 0.01 is 0 whatever the history; the observation's own quote at the cutoff is not history.
    assert persistence(with_consensus(0.565), {}) == 0
    assert math.isnan(persistence(obs_pair(), gap_world({0: 0.56})))
    # Per observation: the T-10 cutoff (cut + 20 min) sees the cut + 10 quote; the T-30 one does not.
    two = pd.concat([obs_pair(), obs_pair(10)], ignore_index=True)
    assert math.isnan(persistence(two, gap_world({-10: 0.56})))
    assert persistence(two, gap_world({-10: 0.56}), "SYN-NYY@T-10") == pytest.approx(10)


def test_gap_persistence_consensus_age_limit():
    # Without the cut - 60 and cut - 30 prices, book A is 80 min old at the cut - 10 quote: no consensus = a break.
    stale = gap_world(odds_at=(210, 180, 150, 120, 90))
    assert persistence(obs_pair(), stale) == pytest.approx(10)
    assert persistence(obs_pair(), stale, consensus_max_age_min=90.0) == pytest.approx(100)


def test_gap_persistence_uses_only_what_was_known_at_each_quote():
    # Book B at 0.70 exactly at the cut - 100 quote makes that gap (0.65 - 0.63) +0.02: no break there.
    at_quote = gap_world(extra_odds=(odds("B", CUT - td(minutes=100), 0.70),))
    assert persistence(obs_pair(), at_quote) == 180
    one_ns_later = gap_world(extra_odds=(odds("B", CUT - td(minutes=100) + pd.Timedelta(1, "ns"), 0.70),))
    assert persistence(obs_pair(), one_ns_later) == pytest.approx(100)
    # The table's quote at the cutoff is the observation's own: its gap is cross_market_gap, not recomputed.
    at_cut = gap_world({200: 0.62, 150: 0.55, 100: 0.63, 40: 0.56, 10: 0.57, 0: 0.60})   # gap 0 at the cut
    assert persistence(obs_pair(), at_cut) == pytest.approx(100)


def test_two_host_quotes_at_one_quote_ts_pick_the_same_row_in_any_order():
    """Two quote sources can observe one quote_ts. Same source and available_at: a fixed
    host order (alphabetical) decides, never the row order."""
    ref = CUT - td(minutes=70)                                    # market_move_60m's reference quote
    tie = [quote("SYN-NYY", ref, 0.51, 0.53) | {"host": "host_a"}, quote("SYN-NYY", ref, 0.47, 0.49) | {"host": "host_b"}]
    for order in (tie, tie[::-1]):
        assert val(build(obs_pair(), {"market_quotes": rows(*order)}), "market_move_60m") == pytest.approx(0.56 - 0.52)
    # Gap history: host_b's 0.55 at cut - 100 would not break the run; host_a's 0.63 does and is the pick.
    g = gap_world()
    host_a = g["market_quotes"].assign(host="host_a")
    win = rows(quote("SYN-NYY", CUT - td(minutes=100), 0.54, 0.56) | {"host": "host_b"})
    for q in (pd.concat([host_a, win], ignore_index=True), pd.concat([win, host_a], ignore_index=True)):
        assert persistence(obs_pair(), g | {"market_quotes": q}) == pytest.approx(100)


def test_quote_ts_tie_prefers_the_earliest_available_quote():
    """At one quote_ts the pick is the quote known first, whatever its host: never the later-known one."""
    ref = CUT - td(minutes=70)
    early = quote("SYN-NYY", ref, 0.51, 0.53) | {"host": "host_b", "available_at": ref + td(seconds=1)}
    late = quote("SYN-NYY", ref, 0.47, 0.49) | {"host": "host_a", "available_at": ref + td(seconds=30)}
    for order in ((early, late), (late, early)):
        assert val(build(obs_pair(), {"market_quotes": rows(*order)}), "market_move_60m") == pytest.approx(0.56 - 0.52)


# ---------------------------------------------------------------------------
# Negative controls and orientation
# ---------------------------------------------------------------------------

def test_negative_controls_are_deterministic():
    obs = pd.concat([obs_pair(), obs_pair(10)], ignore_index=True)
    a, b, c = build(obs, {}), build(obs, {}), build(obs, {}, seed=1)
    pd.testing.assert_frame_equal(a.values, b.values)
    assert val(a, "nc_random") == pytest.approx(float(pit.hash_normal([YES], ["nc_random", 0])[0]))
    assert val(a, "nc_random") != val(c, "nc_random")
    assert val(a, "nc_team_hash") == val(c, "nc_team_hash") == val(a, "nc_team_hash", "SYN-NYY@T-10")
    assert val(a, "nc_team_hash") != val(a, "nc_team_hash", OPP)
    assert 0 <= val(a, "nc_team_hash") < 1


def test_permuted_weather_uses_another_game_on_the_same_date():
    g2 = observation(30, observation_id="SYN-BOS@T-30", game_id="syn-mlb-2", contract_id="SYN-BOS",
                     home_team="BOS", away_team="SEA", yes_team="BOS")
    g3 = observation(30, observation_id="SYN-SEA@T-30", game_id="syn-mlb-3", contract_id="SYN-SEA",
                     home_team="SEA", away_team="BOS", yes_team="SEA", official_date=date(2025, 6, 2))
    world = {"games": rows(game("syn-mlb-1", GAME_DATE, 1), game("syn-mlb-2", GAME_DATE, 2, home_team="BOS"),
                           game("syn-mlb-3", date(2025, 6, 2), 2)),
             "venues": rows(venue(1), venue(2)),
             "mlb_weather": rows(forecast(temperature_f=70.0), forecast("syn-mlb-2", temperature_f=80.0),
                                 forecast("syn-mlb-3", temperature_f=90.0))}
    fb = build(frame(observation(30), g2, g3), world)
    assert val(fb, "nc_permuted_temp") == 80.0 and val(fb, "nc_permuted_temp", "SYN-BOS@T-30") == 70.0
    assert row(fb, "nc_permuted_temp")["source_ref"] == "donor=syn-mlb-2"
    assert math.isnan(val(fb, "nc_permuted_temp", "SYN-SEA@T-30"))        # alone on its date
    assert val(fb, "nc_permuted_wind") == pytest.approx(10.0)             # donor at venue 2 (az 0, from 180)


def test_yes_orientation_mirrors_the_two_contracts():
    fb = build(obs_pair(), full_world())
    v = fb.values.pivot(index="feature_name", columns="observation_id", values="value")
    finite = v.loc[DIFFS].dropna()
    assert len(finite) == 16    # all diffs but last pitches / days rest (no TB start) and lineup changes
    np.testing.assert_allclose(finite[YES], -finite[OPP], atol=1e-12)
    for yes_name, opp_name in (("sp_yes_throws_left", "sp_opp_throws_left"),
                               ("off_lineup_posted_yes", "off_lineup_posted_opp")):
        assert v.loc[yes_name, YES] == v.loc[opp_name, OPP] and v.loc[opp_name, YES] == v.loc[yes_name, OPP]
    for name in ("book_move_since_news", "market_logit", "cross_market_gap", "cross_market_gap_log_odds"):
        assert v.loc[name, YES] == pytest.approx(-v.loc[name, OPP])
    for name in ("temp_f", "wind_out_mph", "humidity_pct", "roof_closed", "park_run_factor"):
        assert v.loc[name, YES] == v.loc[name, OPP]


# ---------------------------------------------------------------------------
# Guard and database
# ---------------------------------------------------------------------------

def test_invariant_guard_refuses_leaky_rows():
    names = np.array(["temp_f", "temp_f"], dtype=object)
    ids = np.array(["a", "b"], dtype=object)
    cut = np.array([100, 100], dtype=np.int64)
    ok = dict(names=names, obs_ids=ids, value=np.array([1.0, np.nan]), at=np.array([100, pit.NAT]), cutoff=cut,
              snapshot=cut + 5)
    mlb._check_invariants(**ok)
    for bad in ({"at": np.array([101, pit.NAT])}, {"at": np.array([pit.NAT, pit.NAT])},
                {"snapshot": cut - 1}, {"names": np.array(["temp_f", "nope"], dtype=object)},
                {"obs_ids": np.array(["a", "a"], dtype=object)}):
        with pytest.raises(mlb.FeatureLeakError):
            mlb._check_invariants(**(ok | bad))


def test_insert_into_database(syn_db):
    fb = build(pd.concat([obs_pair(), obs_pair(10)], ignore_index=True), full_world())
    assert syn_db.insert("feature_definitions", fb.definitions) == 47
    assert syn_db.insert("feature_values", fb.values) == 4 * 47
    back = syn_db.df("SELECT count(*) AS n, count(value) AS v FROM feature_values WHERE available_at <= cutoff_time")
    assert int(back["v"][0]) == int(fb.values["value"].notna().sum())


# ---------------------------------------------------------------------------
# write_features: rebuilding over a populated database
# ---------------------------------------------------------------------------

def test_rebuild_keeps_unchanged_definitions_and_replaces_changed_ones(syn_db, monkeypatch):
    """A rerun with the same definitions rewrites only the values; a changed definition replaces the
    definitions table (values emptied first, the DuckDB foreign-key limitation). Either way the database
    ends with exactly the definitions and values just built."""
    from src.config import load_sport_config
    from src.sports.synthetic import SyntheticSource
    from src.storage.ingest import ingest

    cfg = load_sport_config("mlb")
    cfg["synthetic"] |= {"seasons": [2026], "through_date": "2026-04-03"}
    ingest(syn_db, SyntheticSource(cfg), date(2026, 1, 1), date(2026, 4, 3))
    first = mlb.write_features(syn_db, cfg, date(2026, 4, 3))
    assert first.n_values > 0

    calls: list[list[str]] = []
    real_replace = syn_db.replace_tables
    monkeypatch.setattr(syn_db, "replace_tables", lambda frames: calls.append(list(frames)) or real_replace(frames))
    count = lambda t: int(syn_db.df(f"SELECT count(*) AS n FROM {t}")["n"].iloc[0])
    feature_calls = lambda: [c for c in calls if any(t.startswith("feature_") for t in c)]   # observations too

    again = mlb.write_features(syn_db, cfg, date(2026, 4, 3))
    assert feature_calls() == [["feature_values"]]
    assert count("feature_values") == again.n_values == first.n_values

    built = mlb.build_features
    def renamed_description(*a, **kw):
        fb = built(*a, **kw)
        fb.definitions.loc[fb.definitions.index[0], "description"] = "changed in code"
        return fb
    monkeypatch.setattr(mlb, "build_features", renamed_description)
    calls.clear()
    changed = mlb.write_features(syn_db, cfg, date(2026, 4, 3))
    assert feature_calls() == [["feature_values"], ["feature_definitions", "feature_values"]]
    assert "changed in code" in set(syn_db.df("SELECT description FROM feature_definitions")["description"])
    assert count("feature_values") == changed.n_values == first.n_values
