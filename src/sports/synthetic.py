"""Synthetic MLB world: generated data with known answers, never real market data.

EVERY ROW THIS MODULE PRODUCES IS SYNTHETIC: source = "synthetic",
availability_basis = "synthetic", is_synthetic = true, ids prefixed syn- / SYN-,
player ids from 9,000,001, and the synthetic database refuses real rows (and
vice versa). It exists so the whole pipeline (ingest -> features -> train ->
backtest -> review) runs end to end with no API keys, against known answers:

  positive controls   the synthetic exchange is inefficient in four ways a
                      model could learn, all fixed in WorldParams before any
                      result was seen:
                        - it underweights bullpen fatigue (0.3 of the true weight)
                        - it underweights forecast wind x lineup power (0.5)
                        - it absorbs lineup/starter news with a 12-minute lag
                          (sportsbooks: 1.5 minutes)                 H_MLB_001
                        - thin markets carry extra pricing noise that the
                          sportsbook consensus does not               H_MLB_002
  negative controls   nothing in the world depends on team-id hashes, random
                      numbers or another game's weather (26.16).

The market is otherwise sharp: it knows every player's latent skill exactly,
which the stat-based features only estimate. So most features should add
nothing beyond the market, and the planted effects are the only real residual.

The latent win probability at first pitch (`p_true`) goes to a separate,
clearly labelled truth frame (`SyntheticSource.truth`, written to
data/features/mlb_synthetic_truth.parquet by `quant ingest`). It is used only to
validate interval coverage on synthetic data (26.12). No pipeline table holds it
and no model sees it.

Two named worlds (WORLDS). `null_v1` (synthetic-mlb-0.3.0) is the
default and the null sample. `planted_v2` (synthetic-mlb-0.3.0-planted) has the
same seed and the same random streams, so the same games, outcomes and p_true, plus
a known, modest edge in the market: the home logit path is divided by
`market_slope` (< 1: overconfident) on the exchange and the books, and the exchange
alone is shifted by `planted_shift_logit` in games hosted by a popular team. Its truth
frame also records the subset, the shift, the slope and the exchange's home logit
at first pitch. QUANT_SYNTHETIC_WORLD or config `synthetic.world` selects the world,
and each world needs its own data root.

Random streams (ADR 0003): every draw comes from a stream seeded from (master seed,
world key, game id or other key, component), so no draw's value or position depends
on another component's draws. In particular an outcome never moves a pregame value
or a quote (tests/leakage/test_synthetic_outcome_independence.py). Before 0.3.0 one
stream fed everything, and the market was drawn after every outcome.

Simplifications (a demonstration world, not a baseball simulator): regular
season only, no doubleheaders, random daily pairings instead of series, box-score
lines consistent with player skill but not with the final score. Quotes exist
every 5 minutes from T-200 to T-5; some are missing, stale, one-sided, zero-size
or too wide, so the 26.10 exclusions are exercised.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import config
from ..pricing import expit
from ..storage.schemas import ALL_TABLES, TS

ET = ZoneInfo("America/New_York")
# The 30 club codes the synthetic league uses. TEAMS sorts them, so the set fixes the draw order.
MLB_TEAM_CODES = {
    "AZ", "ATH", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS",
    "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
}

GENERATOR_VERSION = "synthetic-mlb-0.3.0"
PLANTED_GENERATOR_VERSION = "synthetic-mlb-0.3.0-planted"
# Versions before ADR 0003 (one random stream for everything): 0.1.0 (null_v1), 0.2.0-planted (planted_v2).
WORLD_ENV = "QUANT_SYNTHETIC_WORLD"
WORLD_MARKER = "synthetic_world.json"
SOURCE = "synthetic"
SERIES = "SYN-MLBGAME"
TRUTH_FILE = "mlb_synthetic_truth.parquet"
TEAMS = tuple(sorted(MLB_TEAM_CODES))

QUOTE_GRID_MIN = np.arange(-200, 0, 5, dtype=float)   # exchange quotes, minutes from first pitch
ODDS_GRID_MIN = np.arange(-240, 0, 10, dtype=float)   # sportsbook updates
BOOKS = (("syn_book_a", 0.040), ("syn_book_b", 0.045), ("syn_book_c", 0.050))

SEASON_DATES = {2023: ("2023-03-30", "2023-10-01"), 2024: ("2024-03-28", "2024-09-29"),
                2025: ("2025-03-27", "2025-09-28"), 2026: ("2026-03-26", "2026-09-27")}
ALL_STAR_BREAK = ((7, 14), (7, 17))  # (month, day) inclusive, every season

_UTC_OFFSET = {**{t: -4 for t in ("ATL", "BAL", "BOS", "CIN", "CLE", "DET", "MIA", "NYM", "NYY",
                                 "PHI", "PIT", "TB", "TOR", "WSH")},
               **{t: -5 for t in ("CHC", "CWS", "HOU", "KC", "MIL", "MIN", "STL", "TEX")},
               "COL": -6, **{t: -7 for t in ("AZ", "ATH", "LAA", "LAD", "SD", "SEA", "SF")}}
_ROOF = {"TB": "dome", **{t: "retractable" for t in ("AZ", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR")}}
_REGION = {-4: (39.0, -77.0), -5: (37.0, -92.0), -6: (39.7, -105.0), -7: (35.0, -118.0)}
_POPULAR = {"NYY": 1.3, "LAD": 1.3, "BOS": 1.0, "CHC": 1.0, "NYM": 0.8, "PHI": 0.7, "ATL": 0.7, "SF": 0.5}
_MONTH_TEMP = {3: 55, 4: 60, 5: 67, 6: 76, 7: 81, 8: 80, 9: 73, 10: 62}

_WEEKDAY_STARTS = ((19, 5, 0.55), (18, 40, 0.15), (20, 10, 0.10), (13, 10, 0.12), (16, 10, 0.08))
_WEEKEND_STARTS = ((13, 5, 0.45), (16, 10, 0.15), (19, 10, 0.40))

# Random streams (ADR 0003). Each is seeded from (master seed, world, key, component) and used for one
# thing only, so no draw's value or position depends on another component's draws:
#   league    players, teams and venues                 key: none
#   season    skill drift into a season                 key: the season
#   schedule  the day's pairings                        key: the date
#   pregame   start time, starters, lineups, news, weather, hidden truth    key: the game id
#   outcome   the winner and the box score              key: the game id
#   kalshi    liquidity, exchange noise, every quote draw  key: the game id
#   books     sportsbook noise, every odds draw         key: the game id
# The exchange component's key string is "kalshi", a legacy venue name. It is kept because it is
# hashed into every stream seed (_stream_seed): renaming it would redraw every quote and change the
# recorded results. It names no data source; every row here is synthetic.
STREAM_COMPONENTS = ("league", "season", "schedule", "pregame", "outcome", "kalshi", "books")


def _stream_seed(seed: int, world: str, key: str, component: str) -> int:
    h = hashlib.sha256(f"{seed}|{world}|{key}|{component}".encode()).digest()
    return int.from_bytes(h[:16], "big")


@dataclass(frozen=True)
class WorldParams:
    """The planted structure. Fixed before any results were seen; never tuned against them."""
    hfa: float = 0.08                    # home-field logit
    w_starter: float = 0.14              # per unit of starter-skill difference
    w_offense: float = 0.20              # per unit of lineup-quality difference
    w_bullpen: float = 0.07
    w_platoon: float = 0.15              # per unit of platoon-advantage share
    w_fatigue: float = 0.12              # per unit of standardized bullpen-fatigue difference
    w_wind: float = 0.20                 # per (10 mph blowing out x unit lineup-power difference)
    hidden_sd: float = 0.10              # true information nobody observes
    market_fatigue_weight: float = 0.3   # PLANTED: the market underweights fatigue
    market_wind_weight: float = 0.5      # PLANTED: ... and forecast wind x power
    market_lag_min: float = 12.0         # PLANTED: the exchange absorbs news slowly (H_MLB_001)
    book_lag_min: float = 1.5
    market_noise_sd: float = 0.03
    market_thin_noise_sd: float = 0.08   # PLANTED: extra mispricing in thin markets (H_MLB_002)
    book_noise_sd: float = 0.025
    market_ou_sd: float = 0.015          # quote wander around fair value
    fatigue_mean: float = 90.0           # standardizes relievers' P1 + (P2 + P3)/2 pitches
    fatigue_sd: float = 45.0
    p_scratch: float = 0.03              # announced starter replaced on game day
    p_rest: float = 0.08                 # a regular sits
    p_revision: float = 0.05             # a posted lineup is revised
    # Planted-alpha world v2 (values in WORLDS).
    # The defaults are the null world: they change nothing and draw no extra random numbers.
    market_slope: float = 1.0            # market home logit = path / market_slope, exchange AND books (< 1: overconfident)
    planted_shift_logit: float = 0.0     # added to the exchange's home logit only (books unshifted) ...
    planted_home_teams: tuple[str, ...] = ()   # ... in games these teams host
    # The world key of the random streams (ADR 0003): worlds with the same key share every draw, so the
    # same games, outcomes and p_true. The planted mechanism draws nothing, so planted_v2 keeps null_v1's.
    rng_world: str = "null_v1"

    @property
    def planted(self) -> bool:
        return (self.market_slope, self.planted_shift_logit, self.planted_home_teams) != (1.0, 0.0, ())

    @property
    def generator_version(self) -> str:
        return PLANTED_GENERATOR_VERSION if self.planted else GENERATOR_VERSION


# Named worlds. `null_v1` is the default and the null sample (26.16: the referee must stay
# INCONCLUSIVE on it); `planted_v2` carries a known, modest edge on top of the v1 controls. Both use
# the config's seed and the stream key null_v1, so they share games, outcomes and p_true; only the
# market differs. Each world
# lives in its own data root (SyntheticSource.claim_data_root). Once a model has been trained on
# planted_v2, its values do not change: a new edge is a new world name.
PLANTED_HOME_TEAMS = tuple(sorted(_POPULAR))   # the exchange overprices the popular teams at home
WORLDS = {
    "null_v1": WorldParams(),
    "planted_v2": WorldParams(market_slope=0.65, planted_shift_logit=0.4, planted_home_teams=PLANTED_HOME_TEAMS),
}
DEFAULT_WORLD = "null_v1"


def resolve_world(cfg: dict[str, Any]) -> str:
    """The synthetic world to generate: QUANT_SYNTHETIC_WORLD, else config `synthetic.world`, else null_v1."""
    name = os.environ.get(WORLD_ENV) or (cfg.get("synthetic") or {}).get("world") or DEFAULT_WORLD
    if name not in WORLDS:
        raise ValueError(f"unknown synthetic world {name!r}; known: {sorted(WORLDS)}")
    return name


# ---------------------------------------------------------------------------
# Players and teams
# ---------------------------------------------------------------------------

@dataclass
class _Pitcher:
    pid: int
    throws: str
    skill: float
    gb: float


@dataclass
class _Hitter:
    pid: int
    bats: str
    skill: float
    power: float


@dataclass
class _Team:
    code: str
    venue_id: int
    pop: float
    off: float
    bp: float
    rotation: list[_Pitcher]
    spot: _Pitcher
    relievers: list[_Pitcher]            # [0] closer, [1:3] setup
    hitters: list[_Hitter]               # [:9] regulars in batting order, [9:] bench
    rot_idx: int = 0
    rp_log: dict[date, np.ndarray] = field(default_factory=dict)   # date -> pitches per reliever

    def relief_pitches(self, d: date, days_back: int) -> float:
        return float(self.rp_log[d - timedelta(days=days_back)].sum()) if d - timedelta(days=days_back) in self.rp_log else 0.0

    def reliever_available(self, i: int, d: date) -> bool:
        y1, y2 = self.rp_log.get(d - timedelta(days=1)), self.rp_log.get(d - timedelta(days=2))
        if y1 is not None and y1[i] > 30:
            return False
        return not (y1 is not None and y2 is not None and y1[i] > 0 and y2[i] > 0)


def _stable_uniform(*parts: Any) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

class _World:
    def __init__(self, seed: int, params: WorldParams, now: datetime) -> None:
        self.seed = seed
        self.p = params
        self.now = now
        self.next_pid = 9_000_001
        self.players: list[dict] = []
        self.venues: list[dict] = []
        self.venue_attr: dict[int, dict] = {}
        league = self._stream("league")
        self.teams = {code: self._team(league, code, i) for i, code in enumerate(TEAMS)}
        self.n_games = 0
        rows = ("games", "contracts", "starters", "lineups", "weather", "pitching", "batting", "info", "truth")
        self.rows: dict[str, list[dict]] = {k: [] for k in rows}
        self.market: list[dict] = []       # per game: arrays driving quotes and odds

    def _stream(self, component: str, key: str = "") -> np.random.Generator:
        """A fresh random stream for one component (STREAM_COMPONENTS) of one key."""
        assert component in STREAM_COMPONENTS, component
        return np.random.default_rng(_stream_seed(self.seed, self.p.rng_world, key, component))

    # -- construction --------------------------------------------------------
    def _pid(self) -> int:
        self.next_pid += 1
        return self.next_pid - 1

    def _pitcher(self, r: np.random.Generator, mean: float, role: str) -> _Pitcher:
        p = _Pitcher(self._pid(), "L" if r.random() < 0.28 else "R", float(r.normal(mean, 1.0)),
                     float(np.clip(r.normal(0.43, 0.05), 0.3, 0.6)))
        self.players.append({"player_id": p.pid, "name": f"Syn Pitcher {p.pid}", "throws": p.throws,
                             "bats": "R" if r.random() < 0.8 else "L", "primary_position": role})
        return p

    def _hitter(self, r: np.random.Generator, team_off: float, mean_shift: float = 0.0) -> _Hitter:
        u = r.random()
        h = _Hitter(self._pid(), "L" if u < 0.35 else ("S" if u < 0.45 else "R"),
                    float(team_off + mean_shift + r.normal(0, 0.7)), 0.0)
        h.power = float(0.5 * h.skill + r.normal(0, 0.8))
        pos = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH")[int(r.integers(9))]
        self.players.append({"player_id": h.pid, "name": f"Syn Hitter {h.pid}", "throws": "R",
                             "bats": h.bats, "primary_position": pos})
        return h

    def _team(self, r: np.random.Generator, code: str, i: int) -> _Team:
        venue_id = 900_001 + i
        tz = _UTC_OFFSET[code]
        lat, lon = _REGION[tz]
        roof = _ROOF.get(code, "open")
        attr = {"run": 1.15 if code == "COL" else float(r.normal(1.0, 0.04)),
                "hr": 1.2 if code == "COL" else float(r.normal(1.0, 0.08)),
                "az": float(r.uniform(0, 360)), "roof": roof, "tz": tz, "lat": lat + float(r.normal(0, 3)),
                "lon": lon + float(r.normal(0, 4))}
        self.venue_attr[venue_id] = attr
        self.venues.append({"venue_id": venue_id, "name": f"SYN {code} Park", "home_team": code,
                            "lat": attr["lat"], "lon": attr["lon"],
                            "elevation_ft": 5200.0 if code == "COL" else float(r.uniform(0, 1100)),
                            "roof_type": roof, "cf_azimuth_deg": attr["az"]})
        off, bp = float(r.normal(0, 0.6)), float(r.normal(0, 1.0))
        rotation = sorted((self._pitcher(r, 0.0, "SP") for _ in range(5)), key=lambda p: -p.skill)
        spot = self._pitcher(r, -0.8, "SP")
        relievers = sorted((self._pitcher(r, bp, "RP") for _ in range(8)), key=lambda p: -p.skill)
        hitters = sorted((self._hitter(r, off) for _ in range(9)), key=lambda h: -h.skill)
        hitters += [self._hitter(r, off, -0.6) for _ in range(4)]
        return _Team(code, venue_id, _POPULAR.get(code, float(r.normal(-0.2, 0.4))), off, bp,
                     rotation, spot, relievers, hitters)

    def _new_season(self, season: int) -> None:
        """Skills drift between seasons (so career stats are informative but imperfect)."""
        r = self._stream("season", str(season))
        for t in self.teams.values():
            t.off = 0.8 * t.off + float(r.normal(0, 0.35))
            for p in [*t.rotation, t.spot, *t.relievers]:
                p.skill = 0.85 * p.skill + float(r.normal(0, 0.3))
            t.bp = float(np.mean([p.skill for p in t.relievers]))
            for h in t.hitters:
                h.skill = 0.85 * h.skill + 0.15 * t.off + float(r.normal(0, 0.3))
                h.power = 0.85 * h.power + float(r.normal(0, 0.3))
            t.relievers.sort(key=lambda p: -p.skill)
            regulars = sorted(t.hitters[:9], key=lambda h: -h.skill)
            t.hitters = regulars + t.hitters[9:]
            t.rot_idx = 0

    # -- simulation ----------------------------------------------------------
    def simulate(self, seasons: list[int], through: date, max_games_per_day: int | None = None) -> None:
        self.max_games_per_day = max_games_per_day
        for season in seasons:
            first, last = (date.fromisoformat(x) for x in SEASON_DATES[season])
            if first > through:
                break
            if season != seasons[0]:
                self._new_season(season)
            d = first
            while d <= min(last, through):
                if not (ALL_STAR_BREAK[0] <= (d.month, d.day) <= ALL_STAR_BREAK[1]):
                    self._day(season, d, is_today=(d == through))
                d += timedelta(days=1)

    def _day(self, season: int, d: date, is_today: bool) -> None:
        n = 20 if d.weekday() in (0, 3) else 30
        playing = self._stream("schedule", d.isoformat()).permutation(TEAMS)[:n]
        for home, away in list(zip(playing[0::2], playing[1::2]))[:self.max_games_per_day]:  # cap: test-sized data
            self._game(season, d, self.teams[str(home)], self.teams[str(away)], is_today)

    def _start_time(self, r: np.random.Generator, d: date, home: _Team) -> tuple[datetime, bool]:
        options = _WEEKEND_STARTS if d.weekday() >= 5 else _WEEKDAY_STARTS
        probs = np.array([o[2] for o in options])
        hh, mm, _ = options[int(r.choice(len(options), p=probs / probs.sum()))]
        local = datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone(timedelta(hours=self.venue_attr[home.venue_id]["tz"])))
        return local.astimezone(timezone.utc), hh >= 17

    def _lineup_quality(self, lineup: list[_Hitter], opp_throws: str) -> tuple[float, float, float]:
        skill = float(np.mean([h.skill for h in lineup]))
        power = float(np.mean([h.power for h in lineup]))
        platoon = float(np.mean([h.bats == "S" or h.bats != opp_throws for h in lineup]))
        return skill, power, platoon

    def _fair(self, sp: dict, lu: dict, wind_out: float, fatigue: dict, *, market: bool) -> float:
        """Home-perspective logit. market=True applies the market's (planted) underweights."""
        p = self.p
        q = {s: self._lineup_quality(lu[s], sp["away" if s == "home" else "home"].throws) for s in ("home", "away")}
        fw = p.market_fatigue_weight if market else 1.0
        ww = p.market_wind_weight if market else 1.0
        home, away = self.teams_of_game
        return (p.hfa
                + p.w_starter * (sp["home"].skill - sp["away"].skill)
                + p.w_offense * (q["home"][0] - q["away"][0])
                + p.w_platoon * (q["home"][2] - q["away"][2])
                + p.w_bullpen * (home.bp - away.bp)
                - fw * p.w_fatigue * (fatigue["home"] - fatigue["away"])
                + ww * p.w_wind * wind_out / 10.0 * (q["home"][1] - q["away"][1]))

    def _weather(self, r: np.random.Generator, d: date, venue_id: int) -> dict:
        a = self.venue_attr[venue_id]
        temp = float(r.normal(_MONTH_TEMP.get(d.month, 70) + (4 if a["tz"] in (-5, -7) else 0), 7))
        rain = bool(r.random() < 0.12)
        closed = a["roof"] == "dome" or (a["roof"] == "retractable" and (rain or temp < 55))
        speed, frm = float(r.gamma(2.0, 4.0)), float(r.uniform(0, 360))
        return {"temp": temp, "rain": rain, "closed": closed, "speed": speed, "from": frm,
                "humidity": float(np.clip(r.normal(55, 15), 10, 100))}

    def _wind_out(self, venue_id: int, speed: float, frm: float, closed: bool) -> float:
        if closed:
            return 0.0
        toward = (frm + 180.0) % 360.0
        return speed * math.cos(math.radians(toward - self.venue_attr[venue_id]["az"]))

    def _forecasts(self, r: np.random.Generator, actual: dict, venue_id: int, start: datetime) -> list[dict]:
        roof = self.venue_attr[venue_id]["roof"]
        out = []
        for lead_h, sd in ((24, 1.0), (6, 0.6), (2, 0.35)):
            temp = actual["temp"] + float(r.normal(0, 4 * sd))
            speed = max(0.0, actual["speed"] + float(r.normal(0, 4 * sd)))
            frm = (actual["from"] + float(r.normal(0, 30 * sd))) % 360
            precip = float(np.clip((0.7 if actual["rain"] else 0.12) + r.normal(0, 0.2 * sd), 0, 1))
            exp_closed = roof == "dome" or (roof == "retractable" and (precip > 0.4 or temp < 55))
            out.append({"issued": start - timedelta(hours=lead_h), "temp": temp, "speed": speed, "from": frm,
                        "humidity": float(np.clip(actual["humidity"] + r.normal(0, 8 * sd), 5, 100)),
                        "precip": precip, "exp_closed": exp_closed,
                        "roof_status": {"dome": "closed", "open": "open"}.get(roof, "unknown"),
                        "wind_out": self._wind_out(venue_id, speed, frm, exp_closed)})
        return out

    def _lineup(self, r: np.random.Generator, team: _Team) -> list[_Hitter]:
        regulars, bench = team.hitters[:9], list(team.hitters[9:])
        lineup = []
        for h in regulars:
            lineup.append(bench.pop(0) if bench and r.random() < self.p.p_rest else h)
        return lineup

    def _game(self, season: int, d: date, home: _Team, away: _Team, is_today: bool) -> None:
        p = self.p
        self.teams_of_game = (home, away)
        self.n_games += 1
        gid = f"syn-mlb-{self.n_games:06d}"
        r = self._stream("pregame", gid)
        start, night = self._start_time(r, d, home)
        teams = {"home": home, "away": away}
        events: list[tuple[datetime, str, str, dict]] = []   # (time, kind, side, payload)

        # Starters: announced a day or more ahead; sometimes scratched on game day.
        announced, actual_sp = {}, {}
        for side, t in teams.items():
            announced[side] = t.rotation[t.rot_idx % 5]
            t.rot_idx += 1
            actual_sp[side] = announced[side]
            t_ann = start - timedelta(hours=float(r.uniform(26, 70)))
            self._starter_row(gid, side, t, announced[side], t_ann)
            self._info(gid, t, side, "starter", t_ann, {"player_id": announced[side].pid, "kind": "announced"})
            if r.random() < p.p_scratch:
                actual_sp[side] = t.spot
                events.append((start - timedelta(minutes=float(r.uniform(40, 200))), "scratch", side,
                               {"player_id": t.spot.pid, "replaces": announced[side].pid}))

        # Lineups: the market projects the regulars until a lineup is posted.
        projected = {s: t.hitters[:9] for s, t in teams.items()}
        posted, final_lu = {}, {}
        for side, t in teams.items():
            posted[side] = self._lineup(r, t)
            t_post = start - timedelta(minutes=float(r.uniform(45, 240)))
            events.append((t_post, "post", side, {"lineup": posted[side]}))
            final_lu[side] = posted[side]
            if r.random() < p.p_revision:
                unused = [h for h in t.hitters if h not in posted[side]]
                slot = int(r.integers(4))
                revised = list(posted[side])
                revised[slot] = unused[0]
                t_rev = min(t_post + timedelta(minutes=float(r.uniform(10, 60))), start - timedelta(minutes=12))
                events.append((t_rev, "revise", side, {"lineup": revised}))
                final_lu[side] = revised

        # Weather and bullpen fatigue.
        wx = self._weather(r, d, home.venue_id)
        fcs = self._forecasts(r, wx, home.venue_id, start)
        wind_actual = self._wind_out(home.venue_id, wx["speed"], wx["from"], wx["closed"])
        fatigue = {s: (t.relief_pitches(d, 1) + 0.5 * (t.relief_pitches(d, 2) + t.relief_pitches(d, 3))
                       - p.fatigue_mean) / p.fatigue_sd for s, t in teams.items()}
        events.append((start - timedelta(minutes=120), "forecast", "home", {}))
        events.sort(key=lambda e: e[0])

        # Market fair value: state before the quote window, then a jump per event.
        sp_state, lu_state, fc_idx = dict(announced), dict(projected), 1
        f0 = self._fair(sp_state, lu_state, fcs[fc_idx]["wind_out"], fatigue, market=True)
        prev, ev_t, ev_d, ev_news = f0, [], [], []
        for t_ev, kind, side, payload in events:
            if kind == "scratch":
                sp_state[side] = teams[side].spot
            elif kind in ("post", "revise"):
                lu_state[side] = payload["lineup"]
            else:
                fc_idx = 2
            f = self._fair(sp_state, lu_state, fcs[fc_idx]["wind_out"], fatigue, market=True)
            ev_t.append((t_ev - start).total_seconds() / 60)
            ev_d.append(f - prev)
            ev_news.append(kind != "forecast")
            prev = f
            if kind == "scratch":
                self._starter_row(gid, side, teams[side], teams[side].spot, t_ev)
                self._info(gid, teams[side], side, "starter", t_ev, {**payload, "kind": "scratch"})
            elif kind in ("post", "revise"):
                self._lineup_rows(gid, side, teams[side], payload["lineup"], t_ev)
                self._info(gid, teams[side], side, "lineup", t_ev,
                           {"kind": "posted" if kind == "post" else "revised",
                            "player_ids": [h.pid for h in payload["lineup"]]})

        # Truth: actual starters and lineups, actual weather, full weights, plus hidden information.
        truth_base = self._fair(actual_sp, final_lu, wind_actual, fatigue, market=False)
        hidden = float(r.normal(0, p.hidden_sd))
        l_true = truth_base + hidden
        p_true = float(expit(l_true))

        # Liquidity and the market's own noise, from the game's exchange and sportsbook streams. The
        # market frames continue the same two streams (market_frames).
        mkt, books = self._stream("kalshi", gid), self._stream("books", gid)
        weekend = d.weekday() >= 5
        z = 0.6 * (home.pop + away.pop) + 0.6 * night + 0.4 * weekend + float(mkt.normal(0, 0.6))
        thin = 0.5 * math.erfc(z / 1.3 / math.sqrt(2))
        eps_k = float(mkt.normal(0, p.market_noise_sd + p.market_thin_noise_sd * thin))
        eps_b = books.normal(0, p.book_noise_sd, size=len(BOOKS))
        self.market.append({"game_id": gid, "start": start, "f0": f0, "eps_k": eps_k, "eps_b": eps_b,
                            "ev_t": np.array(ev_t), "ev_d": np.array(ev_d), "ev_news": np.array(ev_news),
                            "thin": thin, "home": home.code, "away": away.code, "date": d,
                            "market_rng": mkt, "books_rng": books})

        # Outcome, box score (not for today's games: they have not been played).
        outcome = None
        if not is_today:
            outcome = self._play(gid, d, start, teams, actual_sp, final_lu, p_true, wind_actual)
        self._game_rows(gid, season, d, start, home, away, outcome)
        self._weather_rows(gid, start, fcs, wx, home.venue_id)
        comps = {"comp_fatigue_home_minus_away": fatigue["home"] - fatigue["away"],
                 "wind_out_actual_mph": wind_actual, "wind_out_forecast_2h_mph": fcs[2]["wind_out"],
                 "hidden_logit": hidden, "market_fair_logit_at_first_pitch": prev,
                 "market_noise_logit": eps_k, "thinness": thin}
        for side in ("home", "away"):
            self.rows["truth"].append({"game_id": gid, "contract_id": self._ticker(start, home, away, teams[side]),
                                       "official_date": d, "yes_side": side,
                                       "p_true": p_true if side == "home" else 1 - p_true,
                                       "true_logit_home": l_true, **comps})

    # -- box score -----------------------------------------------------------
    def _outcome_stream(self, gid: str) -> np.random.Generator:
        return self._stream("outcome", gid)

    def _home_wins(self, r: np.random.Generator, gid: str, p_true: float) -> bool:
        """The draw that decides the game (tests/leakage/test_synthetic_outcome_independence.py flips it)."""
        return bool(r.random() < p_true)

    def _play(self, gid: str, d: date, start: datetime, teams: dict, sp: dict, lu: dict,
              p_true: float, wind_out: float) -> dict:
        r = self._outcome_stream(gid)
        home_win = self._home_wins(r, gid, p_true)
        park = self.venue_attr[teams["home"].venue_id]
        lam = 4.4 * park["run"]
        for _ in range(100):
            hs, as_ = int(r.poisson(lam)), int(r.poisson(lam))
            if hs != as_ and (hs > as_) == home_win:
                break
        else:
            hs, as_ = (as_ + 1, as_) if home_win else (hs, hs + 1)
        extra = int(r.random() < 0.08) * int(r.integers(1, 4))
        duration = 170 + float(r.normal(0, 15)) + 18 * extra
        actual_start = start + timedelta(minutes=float(r.uniform(0, 8)))
        final_at = actual_start + timedelta(minutes=duration + 5)
        lines_at = final_at + timedelta(minutes=25)
        scores = {"home": hs, "away": as_}
        for side, t in teams.items():
            opp = "away" if side == "home" else "home"
            won = scores[side] > scores[opp]
            sp_outs = self._pitching_line(r, gid, d, side, t, sp[side], True, None, lines_at, final_at, lam)
            self._relief(r, gid, d, side, t, sp_outs, won, abs(hs - as_), 3 * extra, lines_at, final_at)
            self._batting(r, gid, d, side, t, lu[side], sp[opp], park, wind_out, lines_at, final_at)
        return {"home_score": hs, "away_score": as_, "home_win": home_win, "actual_start": actual_start,
                "final_at": final_at}

    def _pitching_line(self, r, gid, d, side, team, pitcher: _Pitcher, starter: bool, outs: int | None,
                       lines_at, final_at, lam, save=0, hold=0) -> int:
        s = pitcher.skill
        if outs is None:
            mean = 13.0 if pitcher is team.spot else 16.5 + 1.2 * s
            outs = int(np.clip(round(r.normal(mean, 3.2)), 1, 27))
        hits = int(r.poisson(outs * 0.30 * (1 - 0.08 * s)))
        walks = int(r.poisson(outs * 0.11 * (1 - 0.10 * s)))
        hbp = int(r.poisson(outs * 0.012))
        bf = outs + hits + walks + hbp
        k = int(r.binomial(bf, float(np.clip(0.22 + 0.025 * s, 0.08, 0.4))))
        hr = int(min(hits, r.binomial(bf, float(np.clip(0.030 - 0.004 * s, 0.005, 0.08)))))
        er = int(r.poisson(max(0.05, lam * outs / 27 * (1 - 0.12 * s))))
        bip = max(0, bf - k - walks - hbp - hr)
        gb = int(r.binomial(bip, pitcher.gb))
        pitches = int(np.clip(round(bf * 3.85 + r.normal(0, 5 if starter else 3)), 5, 125))
        self.rows["pitching"].append({
            "game_id": gid, "official_date": d, "side": side, "team": team.code, "player_id": pitcher.pid,
            "is_starter": starter, "outs": outs, "batters_faced": bf, "pitches": pitches, "strikeouts": k,
            "walks": walks, "hit_by_pitch": hbp, "home_runs": hr, "hits": hits, "earned_runs": er,
            "ground_balls": gb, "fly_balls": bip - gb, "saves": save, "holds": hold,
            "_src": final_at, "_avail": lines_at})
        return outs if starter else pitches

    def _relief(self, r, gid, d, side, team: _Team, sp_outs: int, won: bool, margin: int, extra_outs: int,
                lines_at, final_at) -> None:
        need = max(0, 27 - sp_outs) + extra_outs
        used = np.zeros(len(team.relievers))
        if need > 0:
            avail = [i for i in range(len(team.relievers)) if team.reliever_available(i, d)] or list(range(len(team.relievers)))
            save_sit = won and margin <= 3 and need >= 3
            plan = ((1, 0) if need >= 6 else (0,)) if save_sit else ()
            tail = [i for i in plan if i in avail]   # setup then closer
            others = [i for i in r.permutation(avail) if i not in tail]
            n_total = int(np.clip(math.ceil(need / 3.5), 1, 5))
            order = [int(i) for i in others[:max(0, n_total - len(tail))]] + tail
            outs = self._split_outs(need, order, tail)
            for i, o in zip(order, outs):
                save = int(save_sit and i == 0 and order[-1] == 0)
                hold = int(save_sit and i == 1 and 0 in order)
                used[i] = self._pitching_line(r, gid, d, side, team, team.relievers[i], False, o, lines_at,
                                              final_at, 4.4, save, hold)
        team.rp_log[d] = used
        for old in [k for k in team.rp_log if (d - k).days > 4]:
            del team.rp_log[old]

    @staticmethod
    def _split_outs(need: int, order: list[int], tail: list[int]) -> list[int]:
        fixed = {i: 3 for i in tail}
        rest = need - 3 * len(tail)
        if rest < 0:   # fewer outs than the closer/setup plan: the last pitcher gets them all
            return [0] * (len(order) - 1) + [need] if len(order) > 1 else [need]
        free = [i for i in order if i not in fixed]
        if not free:
            fixed[order[0]] += rest
            return [fixed[i] for i in order]
        base, extra = divmod(rest, len(free))
        split = {i: base + (1 if j < extra else 0) for j, i in enumerate(free)}
        outs = [fixed.get(i, split.get(i, 0)) for i in order]
        return outs

    def _batting(self, r, gid, d, side, team: _Team, lineup: list[_Hitter], opp_sp: _Pitcher, park: dict,
                 wind_out: float, lines_at, final_at) -> None:
        skill = np.array([h.skill for h in lineup])
        power = np.array([h.power for h in lineup])
        hr_mult = park["hr"] * (1 + 0.025 * wind_out)
        k_adj = 0.025 * opp_sp.skill
        rates = np.column_stack([
            0.080 + 0.012 * skill,                           # BB
            np.full(9, 0.010),                               # HBP
            0.220 - 0.020 * skill + 0.015 * power + k_adj,   # K
            0.145 + 0.012 * skill,                           # 1B
            0.045 + 0.006 * power,                           # 2B
            np.full(9, 0.004),                               # 3B
            (0.030 + 0.009 * power) * hr_mult,               # HR
            np.full(9, 0.006),                               # SF
        ])
        rates = np.clip(rates, 0.002, None)
        rates = np.column_stack([rates, np.clip(1 - rates.sum(axis=1), 0.2, None)])
        rates /= rates.sum(axis=1, keepdims=True)
        pa = np.clip(np.round(4.65 - 0.1 * np.arange(9) + r.normal(0, 0.7, 9)), 2, 6).astype(int)
        counts = r.multinomial(pa, rates)
        for slot, (h, c) in enumerate(zip(lineup, counts)):
            bb, hbp, k, b1, b2, b3, hr, sf, _ = (int(x) for x in c)
            self.rows["batting"].append({
                "game_id": gid, "official_date": d, "side": side, "team": team.code, "player_id": h.pid,
                "batting_order": slot + 1, "plate_appearances": int(pa[slot]), "at_bats": int(pa[slot]) - bb - hbp - sf,
                "hits": b1 + b2 + b3 + hr, "doubles": b2, "triples": b3, "home_runs": hr, "walks": bb,
                "hit_by_pitch": hbp, "strikeouts": k, "sac_flies": sf, "_src": final_at, "_avail": lines_at})

    # -- row builders ----------------------------------------------------------
    @staticmethod
    def _ticker(start: datetime, home: _Team, away: _Team, yes: _Team | None = None) -> str:
        et = start.astimezone(ET)
        event = f"{SERIES}-{et:%y}{et.strftime('%b').upper()}{et:%d%H%M}{away.code}{home.code}"
        return event if yes is None else f"{event}-{yes.code}"

    def _game_rows(self, gid, season, d, start, home, away, outcome) -> None:
        event = self._ticker(start, home, away)
        schedule_known = datetime(season, 1, 15, tzinfo=timezone.utc)
        self.rows["games"].append({
            "game_id": gid, "sport": "mlb", "season": season, "official_date": d, "game_type": "R",
            "scheduled_start": start, "actual_start": outcome and outcome["actual_start"],
            "home_team": home.code, "away_team": away.code, "venue_id": home.venue_id, "double_header": "N",
            "game_number": 1, "status": "Final" if outcome else "Scheduled", "market_event_id": event,
            "book_event_id": f"syn-book-{gid}",
            "home_score": outcome and outcome["home_score"], "away_score": outcome and outcome["away_score"],
            "home_win": outcome and outcome["home_win"], "outcome_available_at": outcome and outcome["final_at"],
            "_src": schedule_known, "_avail": schedule_known})
        for side, t in (("home", home), ("away", away)):
            result = None if outcome is None else ("yes" if outcome["home_win"] == (side == "home") else "no")
            self.rows["contracts"].append({
                "contract_id": f"{event}-{t.code}", "game_id": gid, "sport": "mlb", "market_type": "ml",
                "yes_team": t.code, "yes_side": side, "line": None, "series_ticker": SERIES, "event_ticker": event,
                "open_time": start - timedelta(days=2), "close_time": start + timedelta(hours=5),
                "result": result, "settled_at": outcome and outcome["final_at"] + timedelta(minutes=10),
                "_src": start - timedelta(days=2), "_avail": start - timedelta(days=2)})

    def _starter_row(self, gid, side, team, pitcher, at) -> None:
        self.rows["starters"].append({"game_id": gid, "side": side, "team": team.code, "player_id": pitcher.pid,
                                      "_src": at, "_avail": at})

    def _lineup_rows(self, gid, side, team, lineup, at) -> None:
        for slot, h in enumerate(lineup):
            self.rows["lineups"].append({"game_id": gid, "side": side, "team": team.code, "batting_order": slot + 1,
                                         "player_id": h.pid, "position": None, "_src": at, "_avail": at})

    def _info(self, gid, team, side, kind, at, payload) -> None:
        body = json.dumps(payload, sort_keys=True, default=str)
        eid = hashlib.sha256(f"{gid}|{kind}|{side}|{body}|{at.isoformat()}".encode()).hexdigest()[:16]
        self.rows["info"].append({"event_id": eid, "game_id": gid, "team": team.code, "side": side, "type": kind,
                                  "event_ts": at, "payload_json": body, "first_seen": False,
                                  "_src": at, "_avail": at})

    def _weather_rows(self, gid, start, fcs, wx, venue_id) -> None:
        for fc in fcs:
            self.rows["weather"].append({
                "game_id": gid, "kind": "forecast", "forecast_for": start, "temperature_f": fc["temp"],
                "wind_speed_mph": fc["speed"], "wind_direction_deg": fc["from"], "humidity_pct": fc["humidity"],
                "precip_prob": fc["precip"], "roof_status": fc["roof_status"],
                "_src": fc["issued"], "_avail": fc["issued"]})
        self.rows["weather"].append({
            "game_id": gid, "kind": "reported", "forecast_for": start, "temperature_f": wx["temp"],
            "wind_speed_mph": wx["speed"], "wind_direction_deg": wx["from"], "humidity_pct": wx["humidity"],
            "precip_prob": 1.0 if wx["rain"] else 0.0, "roof_status": "closed" if wx["closed"] else "open",
            "_src": start, "_avail": start + timedelta(minutes=10)})

    # -- market paths (vectorized over games; each game's draws from its own streams) ------------
    def market_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Quotes and odds. Call once: it continues each game's exchange and sportsbook streams."""
        p = self.p
        m = self.market
        g = len(m)
        n_ev = max(len(x["ev_t"]) for x in m)
        ev_t = np.full((g, n_ev), np.inf)
        ev_d = np.zeros((g, n_ev))
        ev_news = np.zeros((g, n_ev), dtype=bool)
        for i, x in enumerate(m):
            k = len(x["ev_t"])
            ev_t[i, :k], ev_d[i, :k], ev_news[i, :k] = x["ev_t"], x["ev_d"], x["ev_news"]
        f0 = np.array([x["f0"] for x in m])
        thin = np.array([x["thin"] for x in m])
        eps_k = np.array([x["eps_k"] for x in m])
        k = 1 / p.market_slope                                          # 1 in the null world
        in_subset = np.isin([x["home"] for x in m], p.planted_home_teams)
        shift = p.planted_shift_logit * in_subset                        # exchange only; 0 in the null world

        def path(grid: np.ndarray, lag: float) -> np.ndarray:
            dt = grid[None, None, :] - ev_t[:, :, None]                 # games x events x grid
            tau = np.where(ev_news, lag, 1e-6)[:, :, None]
            absorbed = np.where(dt >= 0, 1 - np.exp(-np.maximum(dt, 0) / tau), 0.0)
            return f0[:, None] + (ev_d[:, :, None] * absorbed).sum(axis=1)

        # Exchange: lagged path (scaled, shifted) + game noise + a small mean-reverting wander.
        k_logit = k * path(QUOTE_GRID_MIN, p.market_lag_min) + shift[:, None] + eps_k[:, None]
        self.planted_by_game = pd.DataFrame({
            "game_id": [x["game_id"] for x in m], "in_planted_subset": in_subset,
            "planted_shift_logit": shift, "market_slope": p.market_slope,
            "market_home_logit_at_first_pitch": k * path(np.zeros(1), p.market_lag_min)[:, 0] + shift + eps_k})
        ou = np.zeros_like(k_logit)
        rho = 0.8
        innov = np.stack([x["market_rng"].standard_normal(ou.shape[1]) for x in m])
        ou[:, 0] = p.market_ou_sd * innov[:, 0]
        for j in range(1, ou.shape[1]):
            ou[:, j] = rho * ou[:, j - 1] + p.market_ou_sd * math.sqrt(1 - rho**2) * innov[:, j]
        quotes = self._quote_frame(expit(k_logit + ou), thin)

        b_logit = k * path(ODDS_GRID_MIN, p.book_lag_min)
        odds = self._odds_frame(b_logit, np.stack([x["eps_b"] for x in m]))
        return quotes, odds

    def _quote_frame(self, mid_home: np.ndarray, thin: np.ndarray) -> pd.DataFrame:
        m = self.market
        g, n = mid_home.shape
        starts = pd.to_datetime([x["start"] for x in m], utc=True)
        frames = []
        for side in ("home", "away"):
            per_game = [_quote_draws(x["market_rng"], n, float(th), side) for x, th in zip(m, thin)]
            dr = {k: np.stack([x[k] for x in per_game]) for k in per_game[0]}     # games x grid, or games
            mid = mid_home if side == "home" else 1 - mid_home + dr["mid_noise"]
            base = 1 + dr["base"][:, None]
            spread_c = np.maximum(1, base + np.where(dr["spread_move"], dr["spread_step"], 0))
            bid_c = np.clip(np.floor(mid * 100 - spread_c / 2 + 0.5), 1, 99 - spread_c)
            ask_c = bid_c + spread_c
            median = 30 + 500 * (1 - thin)[:, None]
            bid_sz = np.maximum(1, np.round(median * dr["bid_size_ln"]))
            ask_sz = np.maximum(1, np.round(median * dr["ask_size_ln"]))
            zero = dr["zero"]
            bid_sz = np.where(zero & dr["zero_bid"], 0, bid_sz)
            ask_sz = np.where(zero & ~(bid_sz == 0), 0, ask_sz)
            yes_bid, yes_ask = bid_c / 100, ask_c / 100
            one_sided, which = dr["one_sided"], dr["one_sided_bid"]
            yes_bid = np.where(one_sided & which, np.nan, yes_bid)
            yes_ask = np.where(one_sided & ~which, np.nan, yes_ask)
            dark, dark_end, dark_len = dr["dark"], dr["dark_end"], dr["dark_len"]
            in_dark = dark[:, None] & (QUOTE_GRID_MIN[None, :] <= dark_end[:, None]) & \
                (QUOTE_GRID_MIN[None, :] > (dark_end - dark_len)[:, None])
            keep = dr["keep"] & ~in_dark
            oi_final = np.round((100 + 8000 * (1 - thin)) * dr["oi_ln"])
            ramp = 0.3 + 0.7 * (QUOTE_GRID_MIN - QUOTE_GRID_MIN[0]) / (QUOTE_GRID_MIN[-1] - QUOTE_GRID_MIN[0])
            oi = np.round(oi_final[:, None] * ramp[None, :])
            gi, qi = np.nonzero(keep)
            jitter = dr["jitter"][gi, qi]
            quote_ts = starts[gi] + pd.to_timedelta(QUOTE_GRID_MIN[qi] * 60 - jitter, unit="s")
            yb, ya = yes_bid[gi, qi], yes_ask[gi, qi]
            bs, as_ = bid_sz[gi, qi], ask_sz[gi, qi]
            depth = [None if (np.isnan(b) or np.isnan(a)) else
                     f'{{"yes_bids": [[{b:.2f}, {s1:.0f}], [{b - .01:.2f}, {s1 * 1.8:.0f}], [{b - .02:.2f}, {s1 * 3:.0f}]], '
                     f'"yes_asks": [[{a:.2f}, {s2:.0f}], [{a + .01:.2f}, {s2 * 1.8:.0f}], [{a + .02:.2f}, {s2 * 3:.0f}]]}}'
                     for b, a, s1, s2 in zip(yb, ya, bs, as_)]
            events = [m[i]["game_id"] for i in gi]
            tickers = [self.contract_of[(m[i]["game_id"], side)] for i in gi]
            frames.append(pd.DataFrame({
                "contract_id": tickers, "game_id": events, "quote_ts": quote_ts, "quote_source": "synthetic",
                "yes_bid": yb, "yes_ask": ya, "no_bid": 1 - ya, "no_ask": 1 - yb,
                "last_price": np.round(np.clip(mid[gi, qi] + dr["last_step"][gi, qi], 0.01, 0.99), 2),
                "bid_size": bs, "ask_size": as_, "depth_json": depth,
                "volume": np.round(oi[gi, qi] * 0.8), "open_interest": oi[gi, qi], "http_age_s": np.nan}))
        out = pd.concat(frames, ignore_index=True)
        out["source_timestamp"] = out["quote_ts"]
        out["available_at"] = out["quote_ts"]
        return out

    def _odds_frame(self, b_logit: np.ndarray, eps_b: np.ndarray) -> pd.DataFrame:
        m = self.market
        g, n = b_logit.shape
        starts = pd.to_datetime([x["start"] for x in m], utc=True)
        # per game, per book in order: which updates exist, then their timing jitter
        per_game = [[(x["books_rng"].random(n) >= 0.03, x["books_rng"].uniform(0, 60, n)) for _ in BOOKS] for x in m]
        frames = []
        for b, (book, vig) in enumerate(BOOKS):
            p = expit(b_logit + eps_b[:, b:b + 1])
            price_h, price_a = _american(p * (1 + vig)), _american((1 - p) * (1 + vig))
            imp_h, imp_a = _implied(price_h), _implied(price_a)
            keep = np.stack([x[b][0] for x in per_game])
            gi, qi = np.nonzero(keep)
            jitter = np.stack([x[b][1] for x in per_game])[gi, qi]
            odds_ts = starts[gi] + pd.to_timedelta(ODDS_GRID_MIN[qi] * 60 - jitter, unit="s")
            ph = imp_h[gi, qi] / (imp_h[gi, qi] + imp_a[gi, qi])
            frames.append(pd.DataFrame({
                "game_id": [m[i]["game_id"] for i in gi], "book": book, "market_type": "ml", "line": np.nan,
                "odds_ts": odds_ts, "price_home": price_h[gi, qi], "price_away": price_a[gi, qi],
                "price_over": np.nan, "price_under": np.nan, "p_home_devig": ph, "p_away_devig": 1 - ph,
                "devig_method": "proportional"}))
        out = pd.concat(frames, ignore_index=True)
        out["source_timestamp"] = out["odds_ts"]
        out["available_at"] = out["odds_ts"] + pd.Timedelta(seconds=5)
        return out

    @property
    def contract_of(self) -> dict[tuple[str, str], str]:
        if not hasattr(self, "_contract_of"):
            self._contract_of = {(c["game_id"], c["yes_side"]): c["contract_id"] for c in self.rows["contracts"]}
        return self._contract_of


def _quote_draws(r: np.random.Generator, n: int, thin: float, side: str) -> dict[str, Any]:
    """One game's random draws for one side's quotes, from its exchange stream, always in this order."""
    d: dict[str, Any] = {"mid_noise": r.normal(0, 0.004, n)} if side == "away" else {}
    d["base"] = r.poisson(0.3 + 7 * thin**2)
    d["spread_move"] = r.random(n) < 0.3
    d["spread_step"] = r.integers(-1, 2, n)
    d["bid_size_ln"] = r.lognormal(0, 0.7, n)
    d["ask_size_ln"] = r.lognormal(0, 0.7, n)
    d["zero"] = r.random(n) < 0.006
    d["zero_bid"] = r.random(n) < 0.5
    d["one_sided"] = r.random(n) < 0.008
    d["one_sided_bid"] = r.random(n) < 0.5
    d["keep"] = r.random(n) >= 0.04
    d["dark"] = r.random() < 0.03                     # a quiet stretch with no quotes
    d["dark_end"] = r.uniform(-60, -5)
    d["dark_len"] = r.uniform(10, 45)
    d["oi_ln"] = r.lognormal(0, 0.5)
    d["jitter"] = r.uniform(0, 45, n)
    d["last_step"] = r.choice([-0.01, 0, 0.01], n)
    return d


def _american(implied: np.ndarray) -> np.ndarray:
    implied = np.clip(implied, 0.01, 0.99)
    return np.round(np.where(implied >= 0.5, -100 * implied / (1 - implied), 100 * (1 - implied) / implied))


def _implied(price: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(price > 0, 100 / (price + 100), -price / (-price + 100))


# ---------------------------------------------------------------------------
# IngestSource
# ---------------------------------------------------------------------------

_TABLE_ROWS = {"games": "games", "contracts": "contracts", "mlb_probable_starters": "starters",
               "mlb_lineups": "lineups", "mlb_weather": "weather", "mlb_pitching_lines": "pitching",
               "mlb_batting_lines": "batting", "info_events": "info"}


class SyntheticSource:
    """`quant ingest` source for the synthetic dataset (implements sports.base.IngestSource)."""

    name, kind = "synthetic", "synthetic"

    def __init__(self, cfg: dict[str, Any], params: WorldParams | None = None, *, now: datetime | None = None) -> None:
        s = cfg["synthetic"]
        self.seed = int(s["seed"])
        self.seasons = [int(x) for x in s["seasons"]]
        self.through = date.fromisoformat(str(s["through_date"]))
        self.max_games_per_day = s.get("max_games_per_day")   # optional; unset = full slates
        if params is None:
            self.world = resolve_world(cfg)
            self.params = WORLDS[self.world]
        else:
            self.params = params
            self.world = next((n for n, w in WORLDS.items() if w == params), "custom")
        self.now = now or datetime.now(timezone.utc)
        self.truth: pd.DataFrame | None = None

    @property
    def description(self) -> dict[str, Any]:
        d = {"generator": self.params.generator_version, "world": self.world, "seed": self.seed,
             "seasons": self.seasons, "through_date": str(self.through), "world_params": asdict(self.params)}
        if self.max_games_per_day is not None:
            d["max_games_per_day"] = self.max_games_per_day
        return d

    def claim_data_root(self, root: Path, db: Any, *, default_root: Path | None = None) -> None:
        """One world per data root; `quant ingest` calls this before generating anything.

        The default root (data/) holds the null world only, so a planted world needs QUANT_DATA_DIR.
        A root is marked with WORLD_MARKER on first ingest and then refuses every other world. A
        planted world also refuses an unmarked root whose database already holds ingests (their
        world is unknown). Raises ValueError; nothing is written on refusal.
        """
        default_root = default_root or config.REPO_ROOT / "data"
        marker = root / WORLD_MARKER
        if self.world != DEFAULT_WORLD and root.resolve() == default_root.resolve():
            raise ValueError(f"synthetic world {self.world} needs its own data root: set QUANT_DATA_DIR "
                             f"({default_root} holds {DEFAULT_WORLD})")
        if marker.exists():
            held = json.loads(marker.read_text())["world"]
            if held != self.world:
                raise ValueError(f"{root} holds synthetic world {held}; refusing to ingest {self.world} into it")
        elif self.world != DEFAULT_WORLD and int(db.df("SELECT count(*) AS n FROM ingest_runs")["n"].iloc[0]):
            raise ValueError(f"{root} already holds ingests but no {WORLD_MARKER}: ingest {self.world} "
                             "into an empty data root")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(self.description, indent=1, sort_keys=True) + "\n")

    def load(self, start: date, end: date) -> dict[str, pd.DataFrame]:
        world = _World(self.seed, self.params, self.now)
        world.simulate(self.seasons, self.through, self.max_games_per_day)
        quotes, odds = world.market_frames()
        frames = {t: pd.DataFrame(world.rows[k]) for t, k in _TABLE_ROWS.items()}
        frames["venues"] = pd.DataFrame(world.venues)
        frames["mlb_players"] = pd.DataFrame(world.players)
        frames["market_quotes"] = quotes
        frames["consensus_odds"] = odds
        static = datetime(2023, 1, 1, tzinfo=timezone.utc)
        for name in ("venues", "mlb_players"):
            frames[name]["_src"] = static
            frames[name]["_avail"] = static

        games = frames["games"]
        in_range = set(games.loc[(games["official_date"] >= start) & (games["official_date"] <= end), "game_id"])
        out = {}
        for name, df in frames.items():
            if "game_id" in df.columns:
                df = df[df["game_id"].isin(in_range)]
            out[name] = _provenance(name, df.reset_index(drop=True), self.now, self.params.generator_version)
        truth = pd.DataFrame(world.rows["truth"])
        if self.params.planted:                   # home-oriented, on both contracts' rows
            truth = truth.merge(world.planted_by_game, on="game_id", how="left", validate="many_to_one")
        truth = truth[truth["game_id"].isin(in_range)].reset_index(drop=True)
        truth.insert(0, "SYNTHETIC_TRUTH", "latent p_true at first pitch; coverage validation only (26.12)")
        self.truth = truth
        return out


def _provenance(table: str, df: pd.DataFrame, now: datetime, version: str) -> pd.DataFrame:
    df = df.copy()
    if "_src" in df.columns:
        df["source_timestamp"] = df.pop("_src")
        df["available_at"] = df.pop("_avail")
    for col in ALL_TABLES[table].columns:
        if col.type == TS and col.name in df.columns:
            df[col.name] = pd.to_datetime(df[col.name], utc=True)
    df["source"] = SOURCE
    df["ingest_timestamp"] = pd.Timestamp(now)
    df["availability_basis"] = "synthetic"
    df["collector_version"] = version
    df["is_synthetic"] = True
    return df
