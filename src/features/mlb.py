"""MLB Level 1 point-in-time features (spec Sec 8 Level 1, 26.11, 26.16).

The information cutoff of an observation is its quote time (26.11):
``cutoff_time = obs.quote_ts``. A row of any input table may feed a feature
only if it was available at or before the cutoff:

    market_quotes   quote_ts <= cutoff (same contract; the observation's own quote is at cutoff)
    everything else available_at <= cutoff; stat lines and game outcomes become available when
                    the box score / result was final (never official_date)

``available_at`` of a feature value is the latest availability time of every
input it used. Values computed only from the observation's own quote and fields,
or from constants (priors, hashes), get ``available_at = cutoff_time``. Missing
inputs give NULL values; nothing is imputed here. Every lookup goes through
``common.asof`` / ``common.asof_sums`` / ``common.available``, and
``_check_invariants`` refuses to return a row with available_at > cutoff_time.

Orientation: every feature is from the YES team's point of view; ``*_diff`` is
YES team minus opponent, so the two contracts of a game give mirrored rows.
``expected_direction`` in the registry is the sign of the association with the
YES team winning before the market prior; the model estimates the residual.

Negative controls (26.16): a seeded random number, a team-ID hash, and weather
permuted across same-day games. The randomized-umpire control is omitted because
umpire features are excluded from this build; there are no Level 2 interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

from ..backtest.execution import NoFeeSchedule, fee_schedule_for
from ..pricing import logit
from ..storage.schemas import ALL_TABLES, FEATURE_DEFINITIONS, FEATURE_VALUES, TS
from . import common as pit
from .common import DAY, MIN, NAT

if TYPE_CHECKING:
    from ..storage.database import Database
    from .observations import BuildResult

FEATURE_VERSION = "mlb-l1-0.1.2"  # 0.1.2: cross_market_gap_log_odds; 0.1.1: same-quote_ts tie-break (_PICK)

# --- Priors and fallbacks (documented constants; not fitted) ------------------------------------------
FIP_CONST = 3.10
PRIOR_FIP_IP, PRIOR_RATE_BF, PRIOR_GB_BIP, PRIOR_HR_IP = 40.0, 150.0, 100.0, 40.0
PRIOR_EXP_IP, PRIOR_EXP_IP_STARTS, EXP_IP_LAST_STARTS = 5.3, 2.0, 5
PRIOR_BAT_PA, PRIOR_BULLPEN_IP, PRIOR_PARK_GAMES = 200.0, 150.0, 150.0
LEAGUE_WINDOW, PARK_WINDOW = 365 * DAY, 3 * 365 * DAY
MIN_LEAGUE_OUTS, MIN_LEAGUE_PA = 1500, 2000     # below this the trailing league sample falls back to:
PITCH_FALLBACK = {"fip_raw": 1.07, "k": 0.227, "bb": 0.086, "gb": 0.52, "hr_ip": 1.22 / 9}  # FIP ~4.17
BAT_FALLBACK = {"woba": 0.315, "k": 0.227, "bb": 0.086, "iso": 0.165}
NEWS_TYPES = ("lineup", "starter", "injury", "transaction")
NEWS_WINDOW = 180 * MIN
HI_LEV_TOP, HI_LEV_DAYS, TIRED_PITCHES = 3, 30, 25
RETRACTABLE_PRECIP, RETRACTABLE_TEMP_F = 0.4, 55.0   # precip_prob is a 0..1 fraction
AFTERNOON_BEFORE_MIN = 17 * 60                       # America/New_York
GAP_MIN, GAP_WINDOW = 0.01, 180 * MIN                # gap_persistence_minutes: |gap| floor and cap
CONSENSUS_MAX_AGE_MIN = 60.0                         # default = config snapshot_policy.consensus_max_age_min
STAKE_DOLLARS = 100.0                                # default = config backtest.stake_dollars
_EPS = 1e-9                                          # float noise guard for floor and <= (as backtest.execution)
_QUOTE_PRIORITY = {"orderbook": 2, "synthetic": 2, "listing": 1, "candle": 0}

_OBS_COLS = ("observation_id", "game_id", "contract_id", "official_date", "home_team", "away_team", "yes_team",
             "yes_side", "event_start_time", "snapshot_time", "quote_ts", "p_mid")
_OBS_OPTIONAL = ("spread", "volume", "open_interest", "p_consensus",   # nullable in OBSERVATIONS; bid/ask are
                 "market_yes_bid", "market_yes_ask")                   # not, but a frame without them gets NULLs

_TABLE_COLS = {
    "market_quotes": ("contract_id", "quote_ts", "quote_source", "yes_bid", "yes_ask", "host"),
    "consensus_odds": ("game_id", "book", "market_type", "odds_ts", "p_home_devig"),
    "info_events": ("game_id", "team", "type", "first_seen"),
    "mlb_players": ("player_id", "throws", "bats"),
    "mlb_probable_starters": ("game_id", "side", "player_id"),
    "mlb_lineups": ("game_id", "side", "batting_order", "player_id"),
    "mlb_weather": ("game_id", "kind", "source_timestamp", "temperature_f", "wind_speed_mph", "wind_direction_deg",
                    "humidity_pct", "precip_prob", "roof_status"),
    "mlb_pitching_lines": ("game_id", "official_date", "team", "player_id", "is_starter", "outs", "batters_faced",
                           "pitches", "strikeouts", "walks", "hit_by_pitch", "home_runs", "ground_balls",
                           "fly_balls", "saves", "holds"),
    "mlb_batting_lines": ("game_id", "official_date", "team", "player_id", "batting_order", "plate_appearances",
                          "at_bats", "hits", "doubles", "triples", "home_runs", "walks", "hit_by_pitch",
                          "strikeouts"),
    "games": ("game_id", "official_date", "venue_id", "home_score", "away_score", "outcome_available_at"),
    "venues": ("venue_id", "roof_type", "cf_azimuth_deg"),
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    hypothesis_id: str
    description: str
    availability_rule: str
    inputs: str
    expected_direction: str = "unknown"


_AT_CUTOFF = "cutoff_time (only the observation's own quote/fields at quote_ts)"
_SP_RULE = ("max(starter announcement available_at, latest career pitching line available_at, latest league "
            "line available_at in the trailing 365 d), each <= cutoff; both sides")
_OFF_RULE = ("max(lineup version available_at, or if projected the previous game's latest batting line "
             "available_at; latest career batting line of each hitter; latest league line in 365 d), each <= cutoff")
_BP_DAYS_RULE = "max(available_at of the relief lines summed, each <= cutoff); cutoff_time when none"
_BP_AVAIL_RULE = ("max(available_at of the team's relief lines in the 30-day window, each <= cutoff)")
_ENV_W_RULE = "available_at of the latest kind='forecast' mlb_weather row <= cutoff"
_NEWS = ("news = info_events of type lineup/starter/injury/transaction for this game (or game_id NULL and team "
         "= either team), first_seen not true, available_at in [start - 180 min, cutoff]; NULL without news")


def _specs() -> tuple[FeatureSpec, ...]:
    S = FeatureSpec
    m, L1M = "market", "H_MLB_L1_MARKET"
    return (
        S("market_logit", m, L1M, "logit(p_mid); p_mid = YES bid/ask midpoint, never last trade",
          _AT_CUTOFF, "observations.p_mid", "+"),
        S("market_spread", m, "H_MLB_002", "YES ask - YES bid at the quote", _AT_CUTOFF, "observations.spread"),
        S("log_open_interest", m, "H_MLB_002", "log1p(open interest at the quote); NULL if unknown",
          _AT_CUTOFF, "observations.open_interest"),
        S("log_volume", m, "H_MLB_002", "log1p(contract volume at the quote); NULL if unknown",
          _AT_CUTOFF, "observations.volume"),
        S("is_afternoon_game", m, "H_MLB_002", "1 if the scheduled start is before 17:00 America/New_York",
          "cutoff_time (schedule field; no time-varying input)", "observations.event_start_time"),
        S("market_move_60m", m, L1M, "p_mid - midpoint of the same contract's latest two-sided quote with "
          "quote_ts <= quote_ts - 60 min; NULL if none",
          "cutoff_time; the reference quote has quote_ts <= cutoff - 60 min", "observations.p_mid; market_quotes"),
        S("cross_market_gap", m, "H_MLB_003", "EXPLORATORY: p_consensus (YES-oriented de-vigged book mean) - "
          "p_mid; NULL without consensus", "cutoff_time (observation consensus_available_at <= quote_ts)",
          "observations.p_consensus, observations.p_mid", "+"),
        S("cross_market_gap_log_odds", m, "H_MLB_L1_GAP", "logit(p_consensus) - logit(p_mid): the spec Sec 6 "
          "gap in the residual model's log-odds scale; NULL without consensus",
          "cutoff_time (observation consensus_available_at <= quote_ts)",
          "observations.p_consensus, observations.p_mid", "+"),
        S("gap_inside_fee_band", m, "H_MLB_003", "EXPLORATORY: 1 if |cross_market_gap| <= (yes_ask - yes_bid) / 2 "
          "+ the taker fee per contract of buying the side the gap favors (gap > 0: YES at yes_ask; else NO at "
          "no_ask = 1 - yes_bid), else 0. Fee per contract = fee(Q, price) / Q, Q = floor(stake_dollars / price), "
          "from the versioned taker FeeSchedule in force on the quote's UTC date (backtest.execution."
          "fee_schedule_for). NULL without cross_market_gap, without a valid two-sided quote, when Q = 0 or when "
          "no schedule covers the date", "cutoff_time (the observation's own quote and consensus; the fee "
          "schedule is fixed by date)",
          "observations.p_consensus, p_mid, market_yes_bid, market_yes_ask; fee schedule registry"),
        S("gap_persistence_minutes", m, "H_MLB_003", "EXPLORATORY: minutes the current gap (cross_market_gap) "
          "has held over the contract's earlier quotes (quote_ts < cutoff), capped at 180. An earlier quote breaks "
          "it if its gap has the opposite sign, |gap| < 0.01, or it has no consensus. 0 if |cross_market_gap| < "
          "0.01; NULL without cross_market_gap, or when no earlier quote with quote_ts in [cutoff - 180 min, "
          "cutoff) has a consensus (no evaluated history: persistence unknown); else minutes from the latest break "
          "to the cutoff or, if none broke, from the earliest quote with a consensus (it held at least that long; "
          "180 only when that quote is >= 180 min old). Gap at a quote = YES consensus known at its quote_ts (per "
          "book the latest ml de-vig with available_at <= quote_ts and odds_ts at most consensus_max_age_min "
          f"(default {int(CONSENSUS_MAX_AGE_MIN)}) old, mean over books) - the quote's midpoint",
          "cutoff_time (quotes with quote_ts < cutoff; each quote's odds available_at <= its quote_ts); NULL "
          "unless a quote in [cutoff - 180 min, cutoff) has such a consensus",
          "observations.p_consensus, p_mid; market_quotes; consensus_odds"),
        S("minutes_since_last_news", m, "H_MLB_001", "minutes from the latest news event to the cutoff. " + _NEWS,
          "cutoff_time (event available_at <= cutoff)", "info_events"),
        S("book_move_since_news", m, "H_MLB_001", "YES consensus at cutoff - YES consensus from odds available "
          "<= the news time (per book latest de-vigged ml price, mean over books). " + _NEWS,
          "max(news available_at, latest consensus_odds available_at <= cutoff)", "consensus_odds; info_events",
          "+"),
        S("market_move_since_news", m, "H_MLB_001", "p_mid - midpoint of the contract's latest two-sided quote "
          "with quote_ts <= the news time. " + _NEWS, "cutoff_time (reference quote_ts <= news time <= cutoff)",
          "market_quotes; info_events", "-"),
        S("latency_gap", m, "H_MLB_001", "book_move_since_news - market_move_since_news (positive = the exchange "
          "has not caught up with the books)", "cutoff_time", "consensus_odds; market_quotes; info_events", "+"),
        *_starter_specs(),
        *_offense_specs(),
        S("bp_quality_diff", "bullpen", "H_MLB_L1_BULLPEN", "team relievers' FIP-like over the trailing 365 d, "
          f"{PRIOR_BULLPEN_IP:.0f} IP prior toward the league reliever rate; diff",
          "max(latest team relief line, latest league relief line in 365 d; each <= cutoff); cutoff_time if only "
          "the constant fallback was used", "mlb_pitching_lines (is_starter = false)", "-"),
        S("bp_pitches_prev1_diff", "bullpen", "H_MLB_L1_BULLPEN", "relievers' total pitches on the previous "
          "official date (no line = 0); diff", _BP_DAYS_RULE, "mlb_pitching_lines", "-"),
        S("bp_pitches_prev3_diff", "bullpen", "H_MLB_L1_BULLPEN", "relievers' total pitches over the previous 3 "
          "official dates (no line = 0); diff", _BP_DAYS_RULE, "mlb_pitching_lines", "-"),
        S("bp_hi_lev_avail_diff", "bullpen", "H_MLB_L1_BULLPEN", f"available relievers among the top "
          f"{HI_LEV_TOP} by saves+holds over the previous {HI_LEV_DAYS} dates (available = <= {TIRED_PITCHES} "
          "pitches on the previous date and not pitched on both previous two dates); NULL if no reliever has a "
          "save or hold; diff", _BP_AVAIL_RULE, "mlb_pitching_lines", "+"),
        S("bp_closer_avail_diff", "bullpen", "H_MLB_L1_BULLPEN", f"1/0 availability (same rule) of the reliever "
          f"with the most saves over the previous {HI_LEV_DAYS} dates; NULL if none has a save; diff",
          _BP_AVAIL_RULE, "mlb_pitching_lines", "+"),
        S("yes_is_home", "environment", "H_MLB_L1_ENV", "1 if the YES team is the home team",
          "cutoff_time (contract field)", "observations.yes_side", "+"),
        S("park_run_factor", "environment", "H_MLB_L1_ENV", "venue runs/game over league runs/game, games with "
          f"results known <= cutoff in the trailing 3 years, {PRIOR_PARK_GAMES:.0f}-game prior toward 1.0",
          "max(games row of this game, latest counted result: max(outcome_available_at, available_at)) <= cutoff",
          "games (venue_id, scores, outcome_available_at)", "none"),
        S("temp_f", "environment", "H_MLB_L1_ENV", "forecast game-time temperature (F)", _ENV_W_RULE,
          "mlb_weather", "none"),
        S("wind_out_mph", "environment", "H_MLB_L1_ENV", "forecast wind component blowing out to center: speed * "
          "cos((from + 180) - cf_azimuth); 0 for domes; NULL if the azimuth is unknown",
          "max(forecast available_at, games and venues rows) <= cutoff", "mlb_weather; venues; games", "none"),
        S("humidity_pct", "environment", "H_MLB_L1_ENV", "forecast relative humidity (%)", _ENV_W_RULE,
          "mlb_weather", "none"),
        S("roof_closed", "environment", "H_MLB_L1_ENV", "dome 1; open 0; retractable: forecast roof_status if "
          f"known, else 1 if precip_prob > {RETRACTABLE_PRECIP} or temp < {RETRACTABLE_TEMP_F:.0f}F, else 0; "
          "NULL if the roof type is unknown", "max(venues/games rows, forecast if used) <= cutoff",
          "venues.roof_type; mlb_weather", "none"),
        S("nc_random", "negative_control", "H_MLB_NC", "NEGATIVE CONTROL: N(0,1) from a stable hash of "
          "(observation_id, seed)", "cutoff_time (no data input)", "observation_id", "none"),
        S("nc_team_hash", "negative_control", "H_MLB_NC", "NEGATIVE CONTROL: uniform [0,1) from a stable hash of "
          "yes_team", "cutoff_time (no data input)", "observations.yes_team", "none"),
        S("nc_permuted_temp", "negative_control", "H_MLB_NC", "NEGATIVE CONTROL: temp_f of a different game on "
          "the same official date (seeded cyclic order over the games whose schedule row is available <= cutoff); "
          "NULL if that game's forecast is not available <= this cutoff or there is no other game. "
          "The randomized-umpire control is omitted (umpires excluded).",
          "max(donor forecast, donor games row) available_at <= cutoff (donor chosen from games rows <= cutoff)",
          "mlb_weather; games", "none"),
        S("nc_permuted_wind", "negative_control", "H_MLB_NC", "NEGATIVE CONTROL: wind_out_mph of the same donor "
          "game as nc_permuted_temp", "max(donor forecast, donor games and venue rows) <= cutoff",
          "mlb_weather; venues; games", "none"),
    )


def _starter_specs() -> tuple[FeatureSpec, ...]:
    h, fam = "H_MLB_L1_STARTER", "starter"
    base = ("probable starter = latest mlb_probable_starters row for (game, side) <= cutoff; career-to-date "
            "lines <= cutoff shrunk toward the league (trailing 365 d; fallback constants under 500 IP); diff")
    wl = ("max(starter announcement, the starter's latest start line) available_at, each <= cutoff")
    hand = "max(starter announcement, mlb_players row) available_at, each <= cutoff"
    return (
        FeatureSpec("sp_fip_diff", fam, h, f"FIP-like (13HR + 3(BB+HBP) - 2K)/IP + {FIP_CONST}, "
                    f"{PRIOR_FIP_IP:.0f} IP prior. " + base, _SP_RULE, "mlb_pitching_lines", "-"),
        FeatureSpec("sp_k_pct_diff", fam, h, f"K per batter faced, {PRIOR_RATE_BF:.0f} BF prior. " + base,
                    _SP_RULE, "mlb_pitching_lines", "+"),
        FeatureSpec("sp_bb_pct_diff", fam, h, f"BB per batter faced, {PRIOR_RATE_BF:.0f} BF prior. " + base,
                    _SP_RULE, "mlb_pitching_lines", "-"),
        FeatureSpec("sp_gb_pct_diff", fam, h, f"GB/(GB+FB), {PRIOR_GB_BIP:.0f} BIP prior; lines missing GB or FB "
                    "are skipped. " + base, _SP_RULE, "mlb_pitching_lines", "+"),
        FeatureSpec("sp_hr9_diff", fam, h, f"HR per 9 IP, {PRIOR_HR_IP:.0f} IP prior. " + base, _SP_RULE,
                    "mlb_pitching_lines", "-"),
        FeatureSpec("sp_exp_ip_diff", fam, h, f"expected innings: mean outs/3 over the last {EXP_IP_LAST_STARTS} "
                    f"starts with a {PRIOR_EXP_IP} IP prior weighted {PRIOR_EXP_IP_STARTS:.0f} starts; diff", wl,
                    "mlb_pitching_lines (is_starter)", "+"),
        FeatureSpec("sp_last_pitches_diff", fam, h, "pitches in the starter's last start; diff", wl,
                    "mlb_pitching_lines (is_starter)"),
        FeatureSpec("sp_days_rest_diff", fam, h, "days from the starter's last start to this official date; diff",
                    wl, "mlb_pitching_lines (is_starter)"),
        FeatureSpec("sp_yes_throws_left", fam, h, "1 if the YES team's probable starter throws left", hand,
                    "mlb_probable_starters; mlb_players"),
        FeatureSpec("sp_opp_throws_left", fam, h, "1 if the opponent's probable starter throws left", hand,
                    "mlb_probable_starters; mlb_players"),
    )


def _offense_specs() -> tuple[FeatureSpec, ...]:
    h, fam = "H_MLB_L1_OFFENSE", "offense"
    base = ("lineup = latest mlb_lineups version <= cutoff, else projected = the team's batting order in its most "
            "recent game (latest official date) with batting lines available <= cutoff, built only from those lines "
            "(source_ref says which); hitters' career-to-date "
            f"lines <= cutoff, {PRIOR_BAT_PA:.0f} PA prior toward league rates (trailing 365 d); lineup mean; diff")
    posted = "version available_at if posted, else cutoff_time"
    return (
        FeatureSpec("off_quality_diff", fam, h, "wOBA-like (0.69BB + 0.72HBP + 0.89 1B + 1.27 2B + 1.62 3B + "
                    "2.10HR)/PA. " + base, _OFF_RULE, "mlb_lineups; mlb_batting_lines", "+"),
        FeatureSpec("off_k_pct_diff", fam, h, "K/PA. " + base, _OFF_RULE, "mlb_lineups; mlb_batting_lines", "-"),
        FeatureSpec("off_bb_pct_diff", fam, h, "BB/PA. " + base, _OFF_RULE, "mlb_lineups; mlb_batting_lines", "+"),
        FeatureSpec("off_iso_diff", fam, h, "ISO (2B + 2*3B + 3*HR)/AB (prior in AB). " + base, _OFF_RULE,
                    "mlb_lineups; mlb_batting_lines", "+"),
        FeatureSpec("off_platoon_adv_diff", fam, h, "share of lineup hitters (known hand) with the platoon "
                    "advantage vs the opposing probable starter (L vs R, R vs L, S always); NULL if the opposing "
                    "starter's hand is unknown; diff",
                    "max(lineup, mlb_players rows, opposing starter announcement) available_at, each <= cutoff",
                    "mlb_lineups; mlb_batting_lines; mlb_players; mlb_probable_starters", "+"),
        FeatureSpec("off_lineup_posted_yes", fam, h, "1 if the YES team's lineup is posted at the cutoff", posted,
                    "mlb_lineups", "none"),
        FeatureSpec("off_lineup_posted_opp", fam, h, "1 if the opponent's lineup is posted at the cutoff", posted,
                    "mlb_lineups", "none"),
        FeatureSpec("off_lineup_changes_diff", fam, h, "posted hitters not in the team's previous game's starting "
                    "nine; NULL unless posted and a previous game exists; diff",
                    "max(lineup version, previous game's last batting line) available_at, each <= cutoff",
                    "mlb_lineups; mlb_batting_lines"),
    )


REGISTRY: tuple[FeatureSpec, ...] = _specs()
_NAMES = tuple(f.name for f in REGISTRY)


@dataclass
class FeatureBuild:
    values: pd.DataFrame        # exactly the FEATURE_VALUES columns
    definitions: pd.DataFrame   # exactly the FEATURE_DEFINITIONS columns (one row per feature)


class FeatureLeakError(RuntimeError):
    """A feature row would break available_at <= cutoff_time <= snapshot_time."""


def feature_definitions() -> pd.DataFrame:
    rows = [{"feature_name": f.name, "family": f.family, "dtype": "float", "description": f.description,
             "availability_rule": f.availability_rule, "inputs": f.inputs, "hypothesis_id": f.hypothesis_id,
             "expected_direction": f.expected_direction, "version": FEATURE_VERSION} for f in REGISTRY]
    return pd.DataFrame(rows, columns=FEATURE_DEFINITIONS.column_names)


def feature_families() -> dict[str, str]:
    return {f.name: f.family for f in REGISTRY}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_features(obs: pd.DataFrame, tables: Mapping[str, pd.DataFrame], *, build_run_id: str,
                   is_synthetic: bool, seed: int = 0, stake_dollars: float = STAKE_DOLLARS,
                   consensus_max_age_min: float = CONSENSUS_MAX_AGE_MIN) -> FeatureBuild:
    """Long-format feature values for `obs` (OBSERVATIONS rows) from the normalized `tables`.

    stake_dollars sizes the order whose taker fee defines gap_inside_fee_band (config backtest.stake_dollars);
    consensus_max_age_min is the book-price age limit of gap_persistence_minutes' gap history (config
    snapshot_policy.consensus_max_age_min, the rule behind observations.p_consensus).
    """
    missing = [c for c in _OBS_COLS if c not in obs.columns]
    if missing:
        raise KeyError(f"observations frame lacks {missing}")
    if not stake_dollars > 0 or not consensus_max_age_min >= 0:
        raise ValueError("stake_dollars must be > 0 and consensus_max_age_min >= 0")
    obs = obs.assign(**{col: np.nan for col in _OBS_OPTIONAL if col not in obs.columns})
    defs = feature_definitions()
    if obs.empty:
        return FeatureBuild(pd.DataFrame(columns=FEATURE_VALUES.column_names), defs)
    with np.errstate(all="ignore"):
        c = _Ctx(obs, tables, seed)
        out = _Out(c.n)
        _market(c, out)
        _gaps(c, out, float(stake_dollars), int(round(consensus_max_age_min * MIN)))
        _latency(c, out)
        sp = _starter(c, out)
        _offense(c, out, sp)
        _bullpen(c, out)
        env = _environment(c, out)
        _controls(c, out, env)
    return FeatureBuild(_assemble(c, out, build_run_id, is_synthetic), defs)


def _check_invariants(names: np.ndarray, obs_ids: np.ndarray, value: np.ndarray, at: np.ndarray,
                      cutoff: np.ndarray, snapshot: np.ndarray) -> None:
    """Refuse (never clip) rows that could leak: every used input must precede the quote."""
    unknown = set(np.unique(names)) - set(_NAMES)
    if unknown:
        raise FeatureLeakError(f"unregistered features {sorted(unknown)}")
    if len(set(obs_ids)) != len(obs_ids):
        raise FeatureLeakError("duplicate observation_id: (observation, feature) would not be unique")
    n = len(obs_ids)
    checks = (("available_at > cutoff_time", (at != NAT) & (at > cutoff)),
              ("value without available_at", np.isfinite(value) & (at == NAT)),
              ("cutoff_time > snapshot_time", cutoff > snapshot))
    for what, bad in checks:
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise FeatureLeakError(f"{what}: {names[i]} for {obs_ids[i % n]} ({int(bad.sum())} rows)")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _num(s) -> np.ndarray:
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _obj(s) -> np.ndarray:
    s = pd.Series(s)
    return s.astype(object).where(s.notna(), None).to_numpy(dtype=object)


def _table(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    """Needed columns only: times -> int64 ns, dates -> int64 days, numbers/bools -> float, text -> object."""
    spec = {c.name: c.type for c in ALL_TABLES[name].columns}
    raw = tables.get(name)
    raw = pd.DataFrame() if raw is None else raw.reset_index(drop=True)
    out = {}
    for col in (*_TABLE_COLS[name], "available_at"):
        s = raw[col] if col in raw.columns else pd.Series([None] * len(raw), dtype=object)
        kind = spec[col]
        if kind == TS:
            out[col] = pit.ns(s)
        elif kind == "DATE":
            out[col] = pit.days(s)
        elif kind == "BOOLEAN":
            out[col] = s.astype("boolean").to_numpy(dtype=float, na_value=np.nan)
        elif kind in ("DOUBLE", "INTEGER", "BIGINT"):
            out[col] = _num(s)
        else:
            out[col] = _obj(s)
    df = pd.DataFrame(out)
    return df[df["available_at"].to_numpy() != NAT].reset_index(drop=True)


def _z(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.nan_to_num(df[col].to_numpy(dtype=float))


def _rate(num, den, ok, fallback) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    good = np.asarray(ok) & (den > 0)
    return np.where(good, num / np.where(good, den, 1.0), fallback)


def _ids(x) -> pd.Series:
    return pd.Series(x).astype("Int64").astype(str)


class _Ctx:
    """Observations as arrays plus a 2N-row `sides` frame: rows [0, N) home, [N, 2N) away."""

    def __init__(self, obs: pd.DataFrame, tables: Mapping[str, pd.DataFrame], seed: int) -> None:
        self.obs = obs = obs.reset_index(drop=True)
        self.seed = seed
        self.n = n = len(obs)
        self.t = {name: _table(tables, name) for name in _TABLE_COLS}
        self.pitching = _pitching(self.t["mlb_pitching_lines"])
        self.batting = _batting(self.t["mlb_batting_lines"])
        self.cut = pit.ns(obs["quote_ts"])
        self.snap = pit.ns(obs["snapshot_time"])
        self.start = pit.ns(obs["event_start_time"])
        self.day = pit.days(obs["official_date"])
        self.yes_home = obs["yes_side"].to_numpy() == "home"
        self.gid = _obj(obs["game_id"])
        self.home, self.away = _obj(obs["home_team"]), _obj(obs["away_team"])
        self.p = _num(obs["p_mid"])
        cut2 = np.r_[self.cut, self.cut]
        self.sides = pd.DataFrame({
            "game_id": np.r_[self.gid, self.gid],
            "side": np.array(["home"] * n + ["away"] * n, dtype=object),
            "team": np.r_[self.home, self.away],
            "cut": cut2, "lo": pit.shift(cut2, -LEAGUE_WINDOW), "day": np.r_[self.day, self.day]})

    def orient(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Per-side array (2N) -> (YES team, opponent) arrays (N)."""
        x = np.asarray(x)
        h, a = x[: self.n], x[self.n:]
        return np.where(self.yes_home, h, a), np.where(self.yes_home, a, h)

    def swap(self, x) -> np.ndarray:
        """Per-side array -> the same array for the opposing side."""
        x = np.asarray(x)
        return np.r_[x[self.n:], x[: self.n]]


class _Out:
    def __init__(self, n: int) -> None:
        self.n = n
        self.cols: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def add(self, name: str, value, at, ref) -> None:
        if name not in _NAMES:
            raise KeyError(f"feature {name} is not registered")
        v = np.broadcast_to(np.asarray(value, dtype=float), (self.n,))
        t = np.broadcast_to(np.asarray(at, dtype=np.int64), (self.n,))
        r = np.broadcast_to(np.asarray(ref, dtype=object), (self.n,))
        self.cols[name] = (v, t, r)

    def diff(self, c: _Ctx, name: str, x, at, ref) -> None:
        """YES-minus-opponent of a per-side value; available_at = the later side."""
        (yv, ov), (ya, oa) = c.orient(x), c.orient(at)
        self.add(name, yv - ov, pit.tmax(ya, oa), ref)


def _assemble(c: _Ctx, out: _Out, build_run_id: str, is_synthetic: bool) -> pd.DataFrame:
    absent = [nm for nm in _NAMES if nm not in out.cols]
    if absent:
        raise RuntimeError(f"features not built: {absent}")
    n, k = c.n, len(_NAMES)
    value = np.concatenate([out.cols[nm][0] for nm in _NAMES]).astype(float)
    value[~np.isfinite(value)] = np.nan
    at = np.concatenate([out.cols[nm][1] for nm in _NAMES]).astype(np.int64)
    at[np.isnan(value)] = NAT
    ref = np.concatenate([out.cols[nm][2] for nm in _NAMES])
    names = np.repeat(np.array(_NAMES, dtype=object), n)
    obs_ids = c.obs["observation_id"].to_numpy(dtype=object)
    cut, snap = np.tile(c.cut, k), np.tile(c.snap, k)
    _check_invariants(names, obs_ids, value, at, cut, snap)
    return pd.DataFrame({
        "observation_id": np.tile(obs_ids, k), "feature_name": names, "value": value,
        "available_at": pit.to_timestamps(at), "snapshot_time": pit.to_timestamps(snap),
        "cutoff_time": pit.to_timestamps(cut), "source_ref": ref, "build_run_id": build_run_id,
        "is_synthetic": bool(is_synthetic)}, columns=FEATURE_VALUES.column_names)


# ---------------------------------------------------------------------------
# Market and latency (H_MLB_L1_MARKET, H_MLB_L1_GAP, H_MLB_001-003)
# ---------------------------------------------------------------------------

_PICK = ["prio", "early", "host_rank"]  # same-quote_ts tie-break keys: the last row after sorting wins


def _quotes(c: _Ctx) -> pd.DataFrame:
    """Two-sided quotes only (0 < bid < ask < 1), with the midpoint and a same-quote_ts preference (_PICK).

    Of the contract's quotes at one quote_ts, the pick is the highest source priority, then the
    EARLIEST available_at (never the one that became known later), then a fixed host order:
    alphabetical, NULL (candle or synthetic) last. Two recording hosts can observe the same quote in
    the same second, so the pick never depends on row order.
    """
    q = c.t["market_quotes"]
    bid, ask = q["yes_bid"].to_numpy(), q["yes_ask"].to_numpy()
    q = q[(bid > 0) & (ask > bid) & (ask < 1)]
    prio = pd.Series(q["quote_source"].to_numpy()).map(_QUOTE_PRIORITY).fillna(0).to_numpy()
    host = pd.Series(q["host"].to_numpy(), dtype=object)
    order = {h: i for i, h in enumerate(sorted(set(host.dropna())))}
    host_rank = -host.map(order).fillna(len(order)).to_numpy(dtype=np.int64)
    return q.assign(mid=(q["yes_bid"] + q["yes_ask"]) / 2, prio=prio,
                    early=-q["available_at"].to_numpy(dtype=np.int64), host_rank=host_rank)


def _ref_mid(c: _Ctx, quotes: pd.DataFrame, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint and quote_ts of the contract's latest quote with quote_ts <= t (NaN / NAT if none)."""
    left = pd.DataFrame({"contract_id": _obj(c.obs["contract_id"]), "t": t})
    r = pit.asof(left, quotes, on=("t", "quote_ts"), by=["contract_id"], cols=["mid"], order=_PICK)
    return r["mid"].to_numpy(), r["at"].to_numpy()


def _market(c: _Ctx, out: _Out) -> None:
    o, p, cut = c.obs, c.p, c.cut
    out.add("market_logit", np.where(np.isfinite(p), logit(p), np.nan), cut, "observations.p_mid")
    out.add("market_spread", _num(o["spread"]), cut, "observations.spread")
    out.add("log_open_interest", np.log1p(_num(o["open_interest"])), cut, "observations.open_interest")
    out.add("log_volume", np.log1p(_num(o["volume"])), cut, "observations.volume")
    local = pd.to_datetime(o["event_start_time"], utc=True).dt.tz_convert("America/New_York")
    minute = (local.dt.hour * 60 + local.dt.minute).to_numpy(dtype=float)
    out.add("is_afternoon_game", np.where(np.isfinite(minute), minute < AFTERNOON_BEFORE_MIN, np.nan), cut,
            "observations.event_start_time")
    c.quotes = _quotes(c)
    # available_at: the reference quote precedes the cutoff, so this is the cutoff; taking the max anyway
    # means a broken filter trips _check_invariants instead of passing silently.
    ref, ref_at = _ref_mid(c, c.quotes, pit.shift(cut, -60 * MIN))
    out.add("market_move_60m", p - ref, pit.tmax(cut, ref_at), "market_quotes (T-60m reference)")
    out.add("cross_market_gap", _num(o["p_consensus"]) - p, cut, "observations.p_consensus")
    pc = _num(o["p_consensus"])
    out.add("cross_market_gap_log_odds", np.where(np.isfinite(pc) & np.isfinite(p), logit(pc) - logit(p), np.nan),
            cut, "observations.p_consensus")


def _gaps(c: _Ctx, out: _Out, stake: float, max_age: int) -> None:
    """H_MLB_003: is the gap smaller than the cost of trading it, and how long has it lasted."""
    gap = _num(c.obs["p_consensus"]) - c.p
    band, schedule = _fee_band(c, gap, stake)
    out.add("gap_inside_fee_band", band, c.cut, schedule)
    minutes, at = _gap_persistence(c, gap, max_age)
    out.add("gap_persistence_minutes", minutes, at, "market_quotes; consensus_odds")


def _fee_band(c: _Ctx, gap: np.ndarray, stake: float) -> tuple[np.ndarray, np.ndarray]:
    bid, ask = _num(c.obs["market_yes_bid"]), _num(c.obs["market_yes_ask"])
    price = np.where(gap > 0, ask, 1.0 - bid)          # the ask of the side the gap favors
    ok = np.isfinite(gap) & (bid > 0) & (ask > bid) & (ask < 1) & (c.cut != NAT)
    qty = np.floor(stake / np.where(ok, price, 1.0) + _EPS)
    ok &= qty >= 1
    per_contract = np.full(c.n, np.nan)
    ref = np.full(c.n, "fee_schedule=none", dtype=object)
    if ok.any():
        keys = pd.DataFrame({"day": c.cut[ok] // DAY, "price": price[ok], "qty": qty[ok]})
        uniq = keys.drop_duplicates().reset_index(drop=True)
        schedules = {d: _taker_schedule(int(d)) for d in uniq["day"].unique()}
        s = [schedules[d] for d in uniq["day"]]
        uniq["fee"] = [np.nan if sc is None else sc.fee(q, p) / q for sc, p, q in zip(s, uniq["price"], uniq["qty"])]
        uniq["ref"] = ["fee_schedule=none" if sc is None else f"fee_schedule={sc.fee_schedule_id}" for sc in s]
        hit = keys.merge(uniq, on=["day", "price", "qty"], how="left")
        per_contract[ok], ref[ok] = hit["fee"].to_numpy(), hit["ref"].to_numpy(dtype=object)
    band = np.abs(gap) <= (ask - bid) / 2 + per_contract + _EPS
    return np.where(np.isfinite(per_contract), band.astype(float), np.nan), ref


def _taker_schedule(day: int):
    try:
        return fee_schedule_for(date(1970, 1, 1) + timedelta(days=day), "taker")
    except NoFeeSchedule:
        return None


def _gap_persistence(c: _Ctx, gap: np.ndarray, max_age: int) -> tuple[np.ndarray, np.ndarray]:
    """Minutes the current gap has held: since the latest break, else since the first evaluated quote; capped.

    NULL (unknown) unless an earlier quote in [cutoff - GAP_WINDOW, cutoff) was evaluated (had a consensus).
    """
    hist = _gap_history(c, max_age)
    # The table's quotes strictly before the cutoff; the observation's own quote at the cutoff is the current gap.
    left = pd.DataFrame({"contract_id": _obj(c.obs["contract_id"]), "t": pit.shift(c.cut, -1)})
    on, by = ("t", "quote_ts"), ["contract_id"]
    broke_pos = pit.asof(left, hist[hist["broke_pos"].to_numpy()], on=on, by=by)["at"].to_numpy()
    broke_neg = pit.asof(left, hist[hist["broke_neg"].to_numpy()], on=on, by=by)["at"].to_numpy()
    last_eval = pit.asof(left, hist[hist["evaluated"].to_numpy()], on=on, by=by)["at"].to_numpy()
    # As of the latest quote before the cutoff: the first evaluated quote, and every quote's time and odds.
    run = pit.asof(left, hist, on=on, by=by, cols=["first_eval", "run_at"])
    first_eval, used_at = run["first_eval"].to_numpy(), run["run_at"].to_numpy()
    t_break = np.where(gap > 0, broke_pos, broke_neg)
    since = np.where(t_break != NAT, t_break, first_eval)
    known = (last_eval != NAT) & (last_eval >= pit.shift(c.cut, -GAP_WINDOW)) & (since != NAT) & (c.cut != NAT)
    minutes = np.minimum((c.cut - np.where(known, since, c.cut)) / MIN, GAP_WINDOW / MIN)
    minutes = np.where(known, minutes, np.nan)
    minutes = np.where(np.abs(gap) < GAP_MIN, 0.0, minutes)
    return np.where(np.isfinite(gap), minutes, np.nan), pit.tmax(c.cut, t_break, last_eval, used_at)


def _gap_history(c: _Ctx, max_age: int) -> pd.DataFrame:
    """Per quote of an observed contract: gap = YES consensus known at its quote_ts - its midpoint.

    The consensus follows observations.p_consensus: per book the latest ml de-vig row with
    available_at <= quote_ts, kept if quote_ts - odds_ts <= max_age, averaged over books.
    """
    con = pd.DataFrame({"contract_id": _obj(c.obs["contract_id"]), "game_id": c.gid, "yes_home": c.yes_home,
                        "lo": pit.shift(c.cut, -GAP_WINDOW)})
    con = con.groupby("contract_id", sort=False).agg(game_id=("game_id", "first"), yes_home=("yes_home", "first"),
                                                     lo=("lo", "min")).reset_index()
    q = c.quotes.merge(con, on="contract_id")
    # Quotes older than every window of the contract matter only as "the window is covered" (value capped at
    # GAP_WINDOW whether they broke or not): the latest of them stands in for all.
    qts, lo = q["quote_ts"].to_numpy(dtype=np.int64), q["lo"].to_numpy(dtype=np.int64)
    old = (qts < lo) & (qts != NAT)
    last_old = pd.Series(np.where(old, qts, NAT)).groupby(q["contract_id"].to_numpy()).transform("max").to_numpy()
    q = q[(qts >= lo) | (old & (qts == last_old))]
    q = (q.sort_values(["contract_id", "quote_ts", *_PICK], kind="stable")
         .drop_duplicates(["contract_id", "quote_ts"], keep="last").reset_index(drop=True))
    odds = c.t["consensus_odds"]
    odds = odds[(odds["market_type"].to_numpy() == "ml") & np.isfinite(odds["p_home_devig"].to_numpy())]
    books = odds[["game_id", "book"]].drop_duplicates()
    pairs = pd.DataFrame({"qi": np.arange(len(q)), "game_id": q["game_id"].to_numpy(),
                          "quote_ts": q["quote_ts"].to_numpy()}).merge(books, on="game_id")
    r = pit.asof(pairs, odds, on=("quote_ts", "available_at"), by=["game_id", "book"],
                 cols=["odds_ts", "p_home_devig"], order=["odds_ts"])
    odds_ts, qts = r["odds_ts"].to_numpy(), pairs["quote_ts"].to_numpy()
    fresh = (r["at"].to_numpy() != NAT) & (odds_ts != NAT) & (qts - np.where(odds_ts == NAT, qts, odds_ts) <= max_age)
    qi = pairs["qi"].to_numpy()[fresh]
    books_n = np.bincount(qi, minlength=len(q))
    p_home = np.bincount(qi, weights=r["p_home_devig"].to_numpy()[fresh], minlength=len(q)) / np.where(
        books_n > 0, books_n, 1)
    p_home = np.where(books_n > 0, p_home, np.nan)
    g = np.where(q["yes_home"].to_numpy(dtype=bool), p_home, 1.0 - p_home) - q["mid"].to_numpy()
    evaluated = np.isfinite(g)
    cid, quote_ts = q["contract_id"].to_numpy(dtype=object), q["quote_ts"].to_numpy(dtype=np.int64)
    top = np.iinfo(np.int64).max
    first = pd.Series(np.where(evaluated, quote_ts, top)).groupby(cid).cummin().to_numpy(dtype=np.int64)
    odds_at = _group_tmax(len(q), qi, r["at"].to_numpy()[fresh])
    used = pd.Series(pit.tmax(quote_ts, odds_at))
    return pd.DataFrame({"contract_id": cid, "quote_ts": quote_ts, "evaluated": evaluated,
                         "broke_pos": ~(g >= GAP_MIN), "broke_neg": ~(g <= -GAP_MIN),
                         "first_eval": np.where(first == top, NAT, first),
                         "run_at": used.groupby(cid).cummax().to_numpy(dtype=np.int64)})


def _consensus_timeline(odds: pd.DataFrame) -> pd.DataFrame:
    """Per game, the mean over books of each book's latest de-vigged home probability, at every change."""
    o = odds[(odds["market_type"].to_numpy() == "ml") & np.isfinite(odds["p_home_devig"].to_numpy())]
    if o.empty:
        return pd.DataFrame({"game_id": np.array([], dtype=object), "available_at": np.array([], dtype=np.int64),
                             "c": np.array([], dtype=float)})
    o = (o.sort_values(["game_id", "book", "available_at", "odds_ts"], kind="stable")
         .drop_duplicates(["game_id", "book", "available_at"], keep="last"))
    wide = o.set_index(["game_id", "available_at", "book"])["p_home_devig"].unstack("book")
    wide = wide.groupby(level="game_id", sort=False).ffill()
    return pd.DataFrame({"game_id": wide.index.get_level_values(0).to_numpy(dtype=object),
                         "available_at": wide.index.get_level_values(1).to_numpy(dtype=np.int64),
                         "c": wide.mean(axis=1).to_numpy()})


def _latency(c: _Ctx, out: _Out) -> None:
    ev = c.t["info_events"]
    ev = ev[pd.Series(ev["type"].to_numpy()).isin(NEWS_TYPES).to_numpy() & (ev["first_seen"].to_numpy() != 1.0)]
    game_ev = ev[pd.notna(ev["game_id"].to_numpy())]
    team_ev = ev[pd.isna(ev["game_id"].to_numpy()) & pd.notna(ev["team"].to_numpy())]
    t_news = pit.asof(pd.DataFrame({"game_id": c.gid, "cut": c.cut}), game_ev, on=("cut", "available_at"),
                      by=["game_id"])["at"].to_numpy()
    for team in (c.home, c.away):
        t_team = pit.asof(pd.DataFrame({"team": team, "cut": c.cut}), team_ev, on=("cut", "available_at"),
                          by=["team"])["at"].to_numpy()
        t_news = pit.tmax(t_news, t_team)
    has = (t_news != NAT) & (t_news >= pit.shift(c.start, -NEWS_WINDOW))
    t_news = np.where(has, t_news, NAT)

    minutes = np.where(has, (c.cut - np.where(has, t_news, c.cut)) / MIN, np.nan)
    tl = _consensus_timeline(c.t["consensus_odds"])
    left = pd.DataFrame({"game_id": c.gid, "cut": c.cut, "tn": t_news})
    now = pit.asof(left, tl, on=("cut", "available_at"), by=["game_id"], cols=["c"])
    then = pit.asof(left, tl, on=("tn", "available_at"), by=["game_id"], cols=["c"])

    def yes(x):
        x = np.asarray(x, dtype=float)
        return np.where(c.yes_home, x, 1 - x)

    book = np.where(has, yes(now["c"]) - yes(then["c"]), np.nan)
    book_at = pit.tmax(now["at"].to_numpy(), then["at"].to_numpy(), t_news)
    ref, ref_at = _ref_mid(c, c.quotes, t_news)
    mkt, mkt_at = c.p - ref, pit.tmax(c.cut, t_news, ref_at)      # = cutoff unless a filter is broken
    out.add("minutes_since_last_news", minutes, pit.tmax(c.cut, t_news), "info_events")
    out.add("book_move_since_news", book, book_at, "consensus_odds; info_events")
    out.add("market_move_since_news", mkt, mkt_at, "market_quotes; info_events")
    out.add("latency_gap", book - mkt, pit.tmax(book_at, mkt_at), "consensus_odds; market_quotes; info_events")


# ---------------------------------------------------------------------------
# Starting pitcher (H_MLB_L1_STARTER)
# ---------------------------------------------------------------------------

_PSUMS = ("outs0", "bf", "k", "bb", "hr", "fip_num", "gb", "bip")


def _pitching(pl: pd.DataFrame) -> pd.DataFrame:
    ok_bip = np.isfinite(pl["ground_balls"].to_numpy()) & np.isfinite(pl["fly_balls"].to_numpy())
    gb, fb = _z(pl, "ground_balls"), _z(pl, "fly_balls")
    return pl.assign(
        outs0=_z(pl, "outs"), bf=_z(pl, "batters_faced"), k=_z(pl, "strikeouts"), bb=_z(pl, "walks"),
        hr=_z(pl, "home_runs"), pitches0=_z(pl, "pitches"), sv=_z(pl, "saves"), hld=_z(pl, "holds"),
        fip_num=13 * _z(pl, "home_runs") + 3 * (_z(pl, "walks") + _z(pl, "hit_by_pitch")) - 2 * _z(pl, "strikeouts"),
        gb=np.where(ok_bip, gb, 0.0), bip=np.where(ok_bip, gb + fb, 0.0),
        ip_known=_z(pl, "outs") / 3, has_outs=np.isfinite(pl["outs"].to_numpy()).astype(float))


def _league_pitching(queries: pd.DataFrame, lines: pd.DataFrame) -> dict[str, np.ndarray]:
    """Trailing-365-day league rates at each query cutoff; constants when under 500 IP."""
    lg = pit.asof_sums(queries, lines, by=[], time="available_at", cols=_PSUMS, upper="cut", lower="lo")
    ok = lg["outs0"].to_numpy() >= MIN_LEAGUE_OUTS
    ip = lg["outs0"] / 3
    return {"fip_raw": _rate(lg["fip_num"], ip, ok, PITCH_FALLBACK["fip_raw"]),
            "k": _rate(lg["k"], lg["bf"], ok, PITCH_FALLBACK["k"]),
            "bb": _rate(lg["bb"], lg["bf"], ok, PITCH_FALLBACK["bb"]),
            "gb": _rate(lg["gb"], lg["bip"], ok, PITCH_FALLBACK["gb"]),
            "hr_ip": _rate(lg["hr"], ip, ok, PITCH_FALLBACK["hr_ip"]),
            "at": np.where(ok, lg["at"].to_numpy(), NAT)}


def _starter(c: _Ctx, out: _Out) -> dict[str, np.ndarray]:
    S = c.sides
    sp = pit.asof(S, c.t["mlb_probable_starters"], on=("cut", "available_at"), by=["game_id", "side"],
                  cols=["player_id"])
    pid, sp_at = sp["player_id"].to_numpy(), sp["at"].to_numpy()
    known = np.isfinite(pid)
    q = pd.DataFrame({"player_id": pid, "cut": S["cut"].to_numpy(), "lo": S["lo"].to_numpy()})
    hand = pit.asof(q, c.t["mlb_players"], on=("cut", "available_at"), by=["player_id"], cols=["throws"])
    throws = hand["throws"].to_numpy(dtype=object)
    throws_at = np.where(known, pit.tmax(sp_at, hand["at"].to_numpy()), NAT)
    left = np.where(throws == "L", 1.0, np.where(throws == "R", 0.0, np.nan))

    pl = c.pitching
    car = pit.asof_sums(q, pl, by=["player_id"], time="available_at", cols=_PSUMS, upper="cut")
    lg = _league_pitching(q, pl)
    ip = car["outs0"].to_numpy() / 3
    stats = {
        "sp_fip_diff": pit.shrink(car["fip_num"], ip, lg["fip_raw"], PRIOR_FIP_IP) + FIP_CONST,
        "sp_k_pct_diff": pit.shrink(car["k"], car["bf"], lg["k"], PRIOR_RATE_BF),
        "sp_bb_pct_diff": pit.shrink(car["bb"], car["bf"], lg["bb"], PRIOR_RATE_BF),
        "sp_gb_pct_diff": pit.shrink(car["gb"], car["bip"], lg["gb"], PRIOR_GB_BIP),
        "sp_hr9_diff": 9 * pit.shrink(car["hr"], ip, lg["hr_ip"], PRIOR_HR_IP),
    }
    stat_at = pit.tmax(sp_at, car["at"].to_numpy(), lg["at"])

    starts = pl[pl["is_starter"].to_numpy() == 1.0]
    roll = pit.last_n_sums(starts, by="player_id", order=["available_at", "official_date"],
                           cols=["ip_known", "has_outs"], n=EXP_IP_LAST_STARTS)
    starts = starts.assign(r_ip=roll["ip_known"].to_numpy(), r_n=roll["has_outs"].to_numpy())
    last = pit.asof(q, starts, on=("cut", "available_at"), by=["player_id"],
                    cols=["r_ip", "r_n", "pitches", "official_date"], order=["official_date"])
    exp_ip = pit.shrink(np.nan_to_num(last["r_ip"].to_numpy()), np.nan_to_num(last["r_n"].to_numpy()),
                        PRIOR_EXP_IP, PRIOR_EXP_IP_STARTS)
    last_day = last["official_date"].to_numpy()
    rest = np.where(last_day != NAT, (S["day"].to_numpy() - last_day).astype(float), np.nan)
    work_at = pit.tmax(sp_at, last["at"].to_numpy())

    (y_id, o_id) = c.orient(pid)
    ref = ("sp yes=" + _ids(y_id) + " opp=" + _ids(o_id)).to_numpy(dtype=object)
    for name, x in stats.items():
        out.diff(c, name, np.where(known, x, np.nan), stat_at, ref)
    out.diff(c, "sp_exp_ip_diff", np.where(known, exp_ip, np.nan), work_at, ref)
    out.diff(c, "sp_last_pitches_diff", np.where(known, last["pitches"].to_numpy(), np.nan), work_at, ref)
    out.diff(c, "sp_days_rest_diff", np.where(known, rest, np.nan), work_at, ref)
    (yl, ol), (yat, oat) = c.orient(left), c.orient(throws_at)
    out.add("sp_yes_throws_left", yl, yat, ref)
    out.add("sp_opp_throws_left", ol, oat, ref)
    return {"throws": np.where(known, throws, None), "throws_at": throws_at}


# ---------------------------------------------------------------------------
# Offense (H_MLB_L1_OFFENSE)
# ---------------------------------------------------------------------------

_BSUMS = ("pa", "ab", "woba_num", "bk", "bbb", "iso_num")


def _batting(bl: pd.DataFrame) -> pd.DataFrame:
    h, d, t, hr = _z(bl, "hits"), _z(bl, "doubles"), _z(bl, "triples"), _z(bl, "home_runs")
    bb, hbp = _z(bl, "walks"), _z(bl, "hit_by_pitch")
    code = bl["batting_order"].to_numpy(dtype=float)
    # Slots 1-9; codes of the form 100 * slot + substitution index are accepted too.
    slot = np.where((code >= 1) & (code <= 9), code, np.where(code >= 100, np.floor(code / 100), np.nan))
    starter = np.where(code >= 100, np.mod(code, 100) == 0, np.isfinite(slot))
    return bl.assign(pa=_z(bl, "plate_appearances"), ab=_z(bl, "at_bats"), bk=_z(bl, "strikeouts"), bbb=bb,
                     woba_num=0.69 * bb + 0.72 * hbp + 0.89 * (h - d - t - hr) + 1.27 * d + 1.62 * t + 2.10 * hr,
                     iso_num=d + 2 * t + 3 * hr, slot=slot, starter_slot=starter.astype(float))


def _per_side(n2: int, pos: np.ndarray, x: np.ndarray, how: str) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float)).groupby(pos)
    return (s.mean() if how == "mean" else s.sum()).reindex(np.arange(n2)).to_numpy()


def _group_tmax(size: int, idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Latest time per group, exact in int64 (a float round trip can move a time past the cutoff)."""
    out = np.full(size, NAT, dtype=np.int64)
    np.maximum.at(out, np.asarray(idx, dtype=np.int64), np.asarray(t, dtype=np.int64))
    return out


def _offense(c: _Ctx, out: _Out, sp: dict[str, np.ndarray]) -> None:
    S, n2 = c.sides, 2 * c.n
    cut2 = S["cut"].to_numpy()
    lu = c.t["mlb_lineups"]
    versions = lu[["game_id", "side", "available_at"]].drop_duplicates()
    v_at = pit.asof(S, versions, on=("cut", "available_at"), by=["game_id", "side"])["at"].to_numpy()
    posted = v_at != NAT

    bl = c.batting
    prev_at, kid, nine = _previous_nine(S, bl)
    projected = ~posted & (prev_at != NAT)

    pos = np.arange(n2)
    shown = pd.DataFrame({"pos": pos[posted], "game_id": S["game_id"].to_numpy()[posted],
                          "side": S["side"].to_numpy()[posted], "available_at": v_at[posted]}).merge(
        lu[["game_id", "side", "available_at", "player_id"]], on=["game_id", "side", "available_at"])
    guess = pd.DataFrame({"pos": pos[projected], "kid": kid[projected]}).merge(nine, on="kid")
    players = pd.concat([shown[["pos", "player_id"]], guess[["pos", "player_id"]]], ignore_index=True)
    ppos = players["pos"].to_numpy()
    lineup_at = np.where(posted, v_at, np.where(projected, prev_at, NAT))

    q = pd.DataFrame({"player_id": players["player_id"].to_numpy(dtype=float), "cut": cut2[ppos]})
    car = pit.asof_sums(q, bl, by=["player_id"], time="available_at", cols=_BSUMS, upper="cut")
    lg = pit.asof_sums(S, bl, by=[], time="available_at", cols=_BSUMS, upper="cut", lower="lo")
    ok = lg["pa"].to_numpy() >= MIN_LEAGUE_PA
    rates = {"off_quality_diff": ("woba_num", "pa", _rate(lg["woba_num"], lg["pa"], ok, BAT_FALLBACK["woba"])),
             "off_k_pct_diff": ("bk", "pa", _rate(lg["bk"], lg["pa"], ok, BAT_FALLBACK["k"])),
             "off_bb_pct_diff": ("bbb", "pa", _rate(lg["bbb"], lg["pa"], ok, BAT_FALLBACK["bb"])),
             "off_iso_diff": ("iso_num", "ab", _rate(lg["iso_num"], lg["ab"], ok, BAT_FALLBACK["iso"]))}
    hitters_at = _group_tmax(n2, ppos, car["at"].to_numpy())
    side_at = pit.tmax(lineup_at, hitters_at, np.where(ok, lg["at"].to_numpy(), NAT))

    y_src, o_src = c.orient(np.where(posted, "posted", np.where(projected, "projected", "none")))
    ref = ("lineup yes=" + pd.Series(y_src) + " opp=" + pd.Series(o_src)).to_numpy(dtype=object)
    for name, (num, den, lg_rate) in rates.items():
        x = pit.shrink(car[num], car[den], lg_rate[ppos], PRIOR_BAT_PA)
        out.diff(c, name, _per_side(n2, ppos, x, "mean"), side_at, ref)

    bats = pit.asof(q, c.t["mlb_players"], on=("cut", "available_at"), by=["player_id"], cols=["bats"])
    b = bats["bats"].to_numpy(dtype=object)
    opp_throws = c.swap(sp["throws"])[ppos]
    hand_known = (b == "L") | (b == "R") | (b == "S")
    adv = (b == "S") | ((b == "L") & (opp_throws == "R")) | ((b == "R") & (opp_throws == "L"))
    n_known = _per_side(n2, ppos, hand_known.astype(float), "sum")
    share = _per_side(n2, ppos, (adv & hand_known).astype(float), "sum") / n_known
    opp_hand_known = np.isin(c.swap(sp["throws"]).astype(str), ["L", "R"])
    share = np.where((n_known > 0) & opp_hand_known, share, np.nan)
    bats_at = _group_tmax(n2, ppos, np.where(hand_known, bats["at"].to_numpy(), NAT))
    out.diff(c, "off_platoon_adv_diff", share, pit.tmax(lineup_at, bats_at, c.swap(sp["throws_at"])), ref)

    flag_at = np.where(posted, v_at, cut2)
    (yp, op), (ya, oa) = c.orient(posted.astype(float)), c.orient(flag_at)
    out.add("off_lineup_posted_yes", yp, ya, ref)
    out.add("off_lineup_posted_opp", op, oa, ref)

    chg = shown.assign(kid=kid[shown["pos"].to_numpy()])
    kept = nine.drop_duplicates(["kid", "player_id"]).assign(kept=1.0)
    chg = chg.merge(kept, on=["kid", "player_id"], how="left")
    changes = _per_side(n2, chg["pos"].to_numpy(), np.isnan(chg["kept"].to_numpy()).astype(float), "sum")
    changes = np.where(posted & (prev_at != NAT), np.nan_to_num(changes), np.nan)
    out.diff(c, "off_lineup_changes_diff", changes, pit.tmax(v_at, prev_at), ref)


def _previous_nine(S: pd.DataFrame, bl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per side, the starting nine of the team's previous game, from batting lines available <= cutoff.

    The previous game is the latest (official_date, first box-score availability) among the team's
    games with a line available <= cutoff; its nine is built only from that game's lines available
    <= cutoff. Returns (prev_at, kid, nine): prev_at = latest availability of that game's lines
    <= cutoff (NAT if none), kid = key into nine (-1 if none), nine = (kid, player_id) rows.
    """
    n2 = len(S)
    ver = bl.groupby(["team", "game_id", "available_at"], as_index=False, sort=False).agg(
        official_date=("official_date", "max"))
    ver["first"] = ver.groupby(["team", "game_id"])["available_at"].transform("min").to_numpy(dtype=np.int64)
    ver["gk"] = ver.groupby(["official_date", "first", "game_id"], sort=True).ngroup().to_numpy(dtype=float)
    ver = ver.sort_values(["team", "available_at", "gk"], kind="stable")
    ver["best"] = ver.groupby("team")["gk"].cummax().to_numpy()     # latest game seen so far, per team
    best = pit.asof(S, ver, on=("cut", "available_at"), by=["team"], cols=["best"], order=["best"])["best"]
    gid_of = ver.drop_duplicates("gk").set_index("gk")["game_id"]
    prev_gid = gid_of.reindex(best.to_numpy()).to_numpy(dtype=object)
    left = pd.DataFrame({"team": S["team"].to_numpy(), "game_id": prev_gid, "cut": S["cut"].to_numpy()})
    prev_at = pit.asof(left, ver, on=("cut", "available_at"), by=["team", "game_id"])["at"].to_numpy()

    keys = pd.DataFrame({"team": left["team"], "game_id": prev_gid, "vt": prev_at})
    has = prev_at != NAT
    kid = np.full(n2, -1, dtype=np.int64)
    kid[has] = keys[has].groupby(["team", "game_id", "vt"], sort=False).ngroup().to_numpy()
    K = keys[has].assign(kid=kid[has]).drop_duplicates("kid")
    lines = bl.loc[np.isfinite(bl["slot"].to_numpy()),
                   ["team", "game_id", "slot", "starter_slot", "pa", "player_id", "available_at"]]
    m = K.merge(lines, on=["team", "game_id"])
    m = m[pit.available(m["available_at"].to_numpy(), m["vt"].to_numpy())]
    nine = (m.sort_values(["kid", "slot", "starter_slot", "pa"], ascending=[True, True, False, False])
            .drop_duplicates(["kid", "slot"])[["kid", "player_id"]])
    return prev_at, kid, nine


# ---------------------------------------------------------------------------
# Bullpen (H_MLB_L1_BULLPEN)
# ---------------------------------------------------------------------------

def _bullpen(c: _Ctx, out: _Out) -> None:
    S, n2 = c.sides, 2 * c.n
    cut2, day2 = S["cut"].to_numpy(), S["day"].to_numpy()
    rl = c.pitching[c.pitching["is_starter"].to_numpy() == 0.0]
    ref = "mlb_pitching_lines (relievers)"

    hi, closer, at, state = _leverage(S, rl)
    hist = state != NAT          # no relief line at all before the cutoff: every bullpen feature is NULL

    team = pit.asof_sums(S, rl, by=["team"], time="available_at", cols=["fip_num", "outs0"], upper="cut", lower="lo")
    lg = pit.asof_sums(S, rl, by=[], time="available_at", cols=["fip_num", "outs0"], upper="cut", lower="lo")
    ok = lg["outs0"].to_numpy() >= MIN_LEAGUE_OUTS
    lg_fip = _rate(lg["fip_num"], lg["outs0"] / 3, ok, PITCH_FALLBACK["fip_raw"])
    quality = pit.shrink(team["fip_num"], team["outs0"] / 3, lg_fip, PRIOR_BULLPEN_IP) + FIP_CONST
    q_at = pit.tmax(team["at"].to_numpy(), np.where(ok, lg["at"].to_numpy(), NAT))
    out.diff(c, "bp_quality_diff", np.where(hist, quality, np.nan), np.where(q_at == NAT, cut2, q_at), ref)

    by_day = rl.groupby(["team", "official_date", "available_at"], as_index=False)["pitches0"].sum()
    lags = np.repeat([1, 2, 3], n2)
    m = pd.DataFrame({"pos": np.tile(np.arange(n2), 3), "k": lags, "team": np.tile(S["team"].to_numpy(), 3),
                      "official_date": np.tile(day2, 3) - lags}).merge(by_day, on=["team", "official_date"])
    m = m[pit.available(m["available_at"].to_numpy(), cut2[m["pos"].to_numpy()])]
    for name, used in (("bp_pitches_prev1_diff", m[m["k"].to_numpy() == 1]), ("bp_pitches_prev3_diff", m)):
        pos = used["pos"].to_numpy()
        total = np.bincount(pos, weights=used["pitches0"].to_numpy(), minlength=n2)
        t = _group_tmax(n2, pos, used["available_at"].to_numpy())
        t = np.where(t == NAT, cut2, t)
        out.diff(c, name, np.where(hist, total, np.nan), t, ref)

    out.diff(c, "bp_hi_lev_avail_diff", hi, at, ref)
    out.diff(c, "bp_closer_avail_diff", closer, at, ref)


def _leverage(S: pd.DataFrame, rl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """High-leverage reliever and closer availability per side.

    Sides with the same (team, date, latest relief line available <= cutoff) see
    exactly the same lines, so they are computed once.
    """
    state = pit.asof(S, rl, on=("cut", "available_at"), by=["team"])["at"].to_numpy()
    key = pd.DataFrame({"team": S["team"].to_numpy(), "day": S["day"].to_numpy(), "state": state})
    sid = key.groupby(["team", "day", "state"], sort=False).ngroup().to_numpy()
    first = np.unique(sid, return_index=True)[1]
    U = key.iloc[first].assign(sid=sid[first])
    lag = np.tile(np.arange(1, HI_LEV_DAYS + 1), len(U))
    E = pd.DataFrame({"sid": np.repeat(U["sid"].to_numpy(), HI_LEV_DAYS),
                      "team": np.repeat(U["team"].to_numpy(), HI_LEV_DAYS),
                      "state": np.repeat(U["state"].to_numpy(), HI_LEV_DAYS),
                      "official_date": np.repeat(U["day"].to_numpy(), HI_LEV_DAYS) - lag, "k": lag})
    lines = rl[["team", "official_date", "player_id", "sv", "hld", "pitches0", "available_at"]]
    m = E.merge(lines, on=["team", "official_date"])
    m = m[pit.available(m["available_at"].to_numpy(), m["state"].to_numpy())]
    k = m["k"].to_numpy()
    m = m.assign(svh=m["sv"] + m["hld"], p1=np.where(k == 1, m["pitches0"], 0.0),
                 app1=(k == 1).astype(float), app2=(k == 2).astype(float))
    g = m.groupby(["sid", "player_id"], as_index=False).agg(
        svh=("svh", "sum"), sv=("sv", "sum"), hld=("hld", "sum"), p1=("p1", "sum"), app1=("app1", "max"),
        app2=("app2", "max"))
    g["avail"] = ((g["p1"] <= TIRED_PITCHES) & ~((g["app1"] > 0) & (g["app2"] > 0))).astype(float)
    n_states = len(U)
    top = (g[g["svh"] >= 1].sort_values(["sid", "svh", "sv", "player_id"], ascending=[True, False, False, True]))
    top = top[top.groupby("sid").cumcount().to_numpy() < HI_LEV_TOP]
    hi = top.groupby("sid")["avail"].sum().reindex(np.arange(n_states)).to_numpy()
    close = (g[g["sv"] >= 1].sort_values(["sid", "sv", "hld", "player_id"], ascending=[True, False, False, True])
             .drop_duplicates("sid").set_index("sid")["avail"].reindex(np.arange(n_states)).to_numpy())
    at = _group_tmax(n_states, m["sid"].to_numpy(), m["available_at"].to_numpy())
    return hi[sid], close[sid], at[sid], state


# ---------------------------------------------------------------------------
# Environment (H_MLB_L1_ENV)
# ---------------------------------------------------------------------------

def _game_env(c: _Ctx, game_ids: np.ndarray, cut: np.ndarray) -> dict[str, np.ndarray]:
    """Venue and latest-forecast weather of `game_ids`, as known at `cut`."""
    left = pd.DataFrame({"game_id": game_ids, "cut": cut})
    g = pit.asof(left, c.t["games"], on=("cut", "available_at"), by=["game_id"], cols=["venue_id"])
    v = pit.asof(pd.DataFrame({"venue_id": g["venue_id"].to_numpy(), "cut": cut}), c.t["venues"],
                 on=("cut", "available_at"), by=["venue_id"], cols=["roof_type", "cf_azimuth_deg"])
    venue_at = np.where(v["at"].to_numpy() != NAT, pit.tmax(g["at"].to_numpy(), v["at"].to_numpy()), NAT)
    fc = c.t["mlb_weather"]
    fc = fc[fc["kind"].to_numpy() == "forecast"]
    w = pit.asof(left, fc, on=("cut", "available_at"), by=["game_id"], order=["source_timestamp"],
                 cols=["temperature_f", "wind_speed_mph", "wind_direction_deg", "humidity_pct", "precip_prob",
                       "roof_status"])
    w_at = w["at"].to_numpy()
    roof = v["roof_type"].to_numpy(dtype=object)
    az = v["cf_azimuth_deg"].to_numpy()
    speed, frm = w["wind_speed_mph"].to_numpy(), w["wind_direction_deg"].to_numpy()
    temp, precip = w["temperature_f"].to_numpy(), w["precip_prob"].to_numpy()
    dome = roof == "dome"

    toward = np.mod(frm + 180.0, 360.0)
    wind = np.where(speed == 0, 0.0, speed * np.cos(np.deg2rad(toward - az)))
    wind = np.where(np.isfinite(az), wind, np.nan)
    wind = np.where(dome, 0.0, wind)
    wind_at = np.where(dome, venue_at, pit.tmax(w_at, venue_at))

    status = w["roof_status"].to_numpy(dtype=object)
    wet_or_cold = (precip > RETRACTABLE_PRECIP) | (temp < RETRACTABLE_TEMP_F)
    guess = np.where(wet_or_cold, 1.0, np.where(np.isfinite(precip) & np.isfinite(temp), 0.0, np.nan))
    retract = np.where(status == "closed", 1.0, np.where(status == "open", 0.0, guess))
    roof_closed = np.select([dome, roof == "open", roof == "retractable"], [1.0, 0.0, retract], np.nan)
    roof_at = np.where(roof == "retractable", pit.tmax(venue_at, w_at), venue_at)
    return {"venue_id": g["venue_id"].to_numpy(), "game_at": g["at"].to_numpy(), "w_at": w_at,
            "temp": temp, "humidity": w["humidity_pct"].to_numpy(), "wind": wind, "wind_at": wind_at,
            "roof_closed": roof_closed, "roof_at": roof_at}


def _environment(c: _Ctx, out: _Out) -> dict[str, np.ndarray]:
    cut = c.cut
    out.add("yes_is_home", c.yes_home.astype(float), cut, "observations.yes_side")
    env = _game_env(c, c.gid, cut)
    gm = c.t["games"]
    t_out = gm["outcome_available_at"].to_numpy()
    scored = np.isfinite(gm["home_score"].to_numpy()) & np.isfinite(gm["away_score"].to_numpy()) & (t_out != NAT)
    past = gm[scored].assign(t=pit.tmax(t_out[scored], gm["available_at"].to_numpy()[scored]),
                             runs=gm["home_score"].to_numpy()[scored] + gm["away_score"].to_numpy()[scored])
    q = pd.DataFrame({"venue_id": env["venue_id"], "cut": cut, "lo": pit.shift(cut, -PARK_WINDOW)})
    venue = pit.asof_sums(q, past, by=["venue_id"], time="t", cols=["runs"], upper="cut", lower="lo")
    league = pit.asof_sums(q, past, by=[], time="t", cols=["runs"], upper="cut", lower="lo")
    vn, ln = venue["n"].to_numpy(), league["n"].to_numpy()
    lg_rpg = _rate(league["runs"], ln, True, np.nan)
    ratio = np.where((vn > 0) & (lg_rpg > 0), _rate(venue["runs"], vn, True, np.nan) / lg_rpg, 1.0)
    park = pit.shrink(vn * ratio, vn, 1.0, PRIOR_PARK_GAMES)
    park = np.where(np.isfinite(env["venue_id"]), park, np.nan)
    park_at = pit.tmax(env["game_at"], venue["at"].to_numpy(), league["at"].to_numpy())
    out.add("park_run_factor", park, park_at, "games (venue results, 3 y)")
    out.add("temp_f", env["temp"], env["w_at"], "mlb_weather forecast")
    out.add("wind_out_mph", env["wind"], env["wind_at"], "mlb_weather forecast; venues")
    out.add("humidity_pct", env["humidity"], env["w_at"], "mlb_weather forecast")
    out.add("roof_closed", env["roof_closed"], env["roof_at"], "venues; mlb_weather forecast")
    return env


# ---------------------------------------------------------------------------
# Negative controls (26.16)
# ---------------------------------------------------------------------------

def _controls(c: _Ctx, out: _Out, env: dict[str, np.ndarray]) -> None:
    ids = c.obs["observation_id"].to_numpy(dtype=object)
    out.add("nc_random", pit.hash_normal(ids, ["nc_random", c.seed]), c.cut, "hash(observation_id, seed)")
    out.add("nc_team_hash", pit.hash_uniform(_obj(c.obs["yes_team"]), "nc_team_hash"), c.cut, "hash(yes_team)")

    donor_gid = _donors(c)
    d = _game_env(c, donor_gid, c.cut)
    has = d["w_at"] != NAT
    d_at = pit.tmax(d["w_at"], d["game_at"])       # the donor's games row (its date) is an input too
    ref = ("donor=" + pd.Series(donor_gid).fillna("none").astype(str)).to_numpy(dtype=object)
    out.add("nc_permuted_temp", np.where(has, d["temp"], np.nan), d_at, ref)
    out.add("nc_permuted_wind", np.where(has, d["wind"], np.nan), pit.tmax(d_at, d["wind_at"]), ref)


def _donors(c: _Ctx) -> np.ndarray:
    """Permuted-weather donor game per observation (None if there is none).

    Candidates are the other games whose latest games row available <= cutoff has the same official
    date. The donor is the next candidate after the observation's own game in a seeded cyclic order,
    so with every game of the date known this is a derangement of the date's games (26.16).
    """
    gm = c.t["games"]
    q = pd.DataFrame({"own": c.gid, "day": c.day, "cut": c.cut})
    qid = q.groupby(["own", "day", "cut"], sort=False).ngroup().to_numpy()
    Q = q.assign(qid=qid).drop_duplicates("qid")
    cand = pd.DataFrame({"game_id": gm["game_id"].to_numpy(dtype=object),
                         "day": gm["official_date"].to_numpy()}).drop_duplicates()
    m = Q.merge(cand, on="day")
    m = m[m["game_id"].to_numpy() != m["own"].to_numpy()].reset_index(drop=True)
    known = pit.asof(m, gm, on=("cut", "available_at"), by=["game_id"], cols=["official_date"])
    m = m[known["official_date"].to_numpy() == m["day"].to_numpy()]
    ids = pd.unique(np.r_[c.gid, cand["game_id"].to_numpy(dtype=object)])
    rank = pd.Series(pit.hash_uniform(ids, ["nc_permuted_weather", c.seed]), index=ids)
    r, own = rank.reindex(m["game_id"]).to_numpy(), rank.reindex(m["own"]).to_numpy()
    pick = (m.assign(key=np.where(r > own, r, r + 1.0)).sort_values(["qid", "key"], kind="stable")
            .drop_duplicates("qid"))
    donor = np.full(len(Q), None, dtype=object)
    donor[pick["qid"].to_numpy()] = pick["game_id"].to_numpy(dtype=object)
    return donor[qid]


# ---------------------------------------------------------------------------
# Database build (`quant features`)
# ---------------------------------------------------------------------------

@dataclass
class FeatureRun:
    observations: BuildResult
    n_values: int
    n_definitions: int
    build_run_id: str


def write_features(db: Database, cfg: Mapping, through: date | None = None) -> FeatureRun:
    """Rebuild observations, snapshot_exclusions, feature_definitions and feature_values through `through`.

    The observations come first (26.10 snapshot rules), then `build_features` reads them with the
    normalized tables. The database CHECKs reject any feature value with available_at > cutoff_time
    again on insert.
    """
    from .observations import build_observations

    built = build_observations(db, dict(cfg), through)
    obs = db.df("SELECT * FROM observations")
    tables = {name: db.df(f"SELECT * FROM {name}") for name in _TABLE_COLS}
    fb = build_features(obs, tables, build_run_id=built.run_id, is_synthetic=db.is_synthetic,
                        seed=int(cfg["model"]["seed"]), stake_dollars=float(cfg["backtest"]["stake_dollars"]),
                        consensus_max_age_min=float(cfg["snapshot_policy"]["consensus_max_age_min"]))
    old_defs = db.df("SELECT * FROM feature_definitions")
    if _same_rows(old_defs, fb.definitions):
        db.replace_tables({"feature_values": fb.values})
    else:
        db.replace_tables({"feature_values": fb.values.iloc[:0]})     # DuckDB FK limitation; see replace_tables
        db.replace_tables({"feature_definitions": fb.definitions, "feature_values": fb.values})
    return FeatureRun(built, int(len(fb.values)), int(len(fb.definitions)), built.run_id)


def _same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cols = sorted(set(a.columns) & set(b.columns))
    key = lambda d: sorted(map(tuple, d[cols].astype(str).to_numpy().tolist()))
    return len(a) == len(b) and key(a) == key(b)
