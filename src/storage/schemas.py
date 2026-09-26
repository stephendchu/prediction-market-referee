"""DuckDB schemas for the research database (spec Sec 25, "DuckDB schemas").

Two kinds of storage:

  derived tables    Normalized games/contracts/quotes/odds/info events/MLB stat
                    lines, observations (Sec 5), point-in-time feature values,
                    predictions, backtest decisions. Built by `quant ingest`,
                    `quant features`, `quant train`, `quant backtest`. The
                    DuckDB file is a rebuildable cache, so a schema change means
                    "rebuild", never "migrate".
  persistent logs   Records that can't be regenerated: paper trades, their
                    settlements, the hypothesis registry, the experiment
                    registry. Append-only Parquet exposed as
                    DuckDB views, so a rebuild never loses them and nothing is
                    edited after the fact (Sec 21, 26.4, 26.5).

One DuckDB file per dataset (`data/normalized/<sport>_<dataset>.duckdb`, dataset
= synthetic | live). The dataset is baked into CHECK constraints, so synthetic
and real rows cannot end up in the same table.

Invariants the database enforces on every insert (not only pipeline code):
  - feature_values.available_at <= cutoff_time = the observation's quote
    time, which is itself <= snapshot_time                     (Sec 5, 26.11)
  - observations: the quote was available at or before the snapshot and
    the consensus at or before the quote (26.11); the quote is two-sided with bid < ask; p_mid lies inside the
    executable bid/ask (so it can't be a last trade outside the quote)  (Sec 6)
  - observations: snapshot_time = event_start_time - offset; a canonical
    observation is the T-30 snapshot; displayed size is nonzero, except
    for historical candles, which carry no size and are flagged
    size_verified = false instead                                    (26.10)
  - feature_values reference a registered feature definition            (26.4)
  - model_runs: training data ends before the first predicted snapshot  (Sec 16)
  - predictions: p_conservative_yes = lower_bound, p_conservative_no =
    1 - upper_bound, lower_bound <= p_model <= upper_bound              (26.12)
  - backtest trades carry a fee schedule and never fill better than the
    quoted executable price                                  (26.8, 26.9)
  - normalized rows: available_at >= source_timestamp; data observed live
    can't be available before we received it
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pyarrow as pa

SCHEMA_VERSION = "db-0.5.0"
DATASETS = ("synthetic", "live")

# An executable quote (order-book or snapshot quote with displayed size) and a
# historical candle's bid/ask close (no size, no depth) never share one result.
EXECUTABLE, CANDLE_DEGRADED = "executable", "candle_degraded"
QUOTE_QUALITIES = (EXECUTABLE, CANDLE_DEGRADED)
DEGRADED_EXECUTION = "DEGRADED EXECUTION: candle quotes, no displayed size; P&L is not an executable-price result"

# The one canonical decision snapshot (26.10). A spec rule, not a tunable threshold.
CANONICAL_OFFSET_MIN = 30


@dataclass(frozen=True)
class Col:
    name: str
    type: str
    nullable: bool = True
    doc: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    doc: str
    columns: tuple[Col, ...]
    primary_key: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    foreign_keys: tuple[tuple[str, str], ...] = ()  # (column, "table(column)")
    kind: str = "derived"  # derived | persistent
    append_only: bool = False  # derived, but rows are only ever added: never replaced or deleted (ADR 0001 N1)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def required(self) -> list[str]:
        return [c.name for c in self.columns if not c.nullable]

    def dataset_check(self, is_synthetic: bool) -> str | None:
        if "is_synthetic" not in self.column_names:
            return None
        return f"is_synthetic = {'true' if is_synthetic else 'false'}"

    def all_checks(self, is_synthetic: bool) -> list[str]:
        extra = self.dataset_check(is_synthetic)
        return [*self.checks, *([extra] if extra else [])]

    def ddl(self, is_synthetic: bool) -> str:
        parts = [f"  {c.name} {c.type}{'' if c.nullable else ' NOT NULL'}" for c in self.columns]
        if self.primary_key:
            parts.append(f"  PRIMARY KEY ({', '.join(self.primary_key)})")
        for col, ref in self.foreign_keys:
            parts.append(f"  FOREIGN KEY ({col}) REFERENCES {ref}")
        for chk in self.all_checks(is_synthetic):
            parts.append(f"  CHECK ({chk})")
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n" + ",\n".join(parts) + "\n)"


def _prob(col: str) -> str:
    return f"{col} IS NULL OR ({col} >= 0 AND {col} <= 1)"


def _in(col: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{col} IS NULL OR {col} IN ({quoted})"


TS = "TIMESTAMPTZ"

# ---------------------------------------------------------------------------
# Provenance carried by every normalized row.
# ---------------------------------------------------------------------------

AVAILABILITY_BASES = ("observed", "source_declared", "conservative", "synthetic")

PROVENANCE = (
    Col("source", "VARCHAR", False, "provider; every row in this repository is synthetic"),
    Col("source_timestamp", TS, False, "when the provider says the data was true"),
    Col("ingest_timestamp", TS, False, "when we received it"),
    Col("available_at", TS, False, "earliest time a point-in-time feature may use it"),
    Col("availability_basis", "VARCHAR", False,
        "observed (seen live when recorded) | source_declared (recorded later: provider "
        "timestamp + publication lag) | conservative (a later time proving publicity, "
        "e.g. game final) | synthetic"),
    Col("collector_version", "VARCHAR", False, "version of the recorder or generator that produced the row"),
    Col("ingest_run_id", "VARCHAR", False),
    Col("is_synthetic", "BOOLEAN", False),
)

PROVENANCE_CHECKS = (
    "available_at >= source_timestamp",
    _in("availability_basis", AVAILABILITY_BASES),
    "availability_basis <> 'observed' OR available_at >= ingest_timestamp",
    "(availability_basis = 'synthetic') = is_synthetic",
)


def _normalized(name: str, doc: str, cols: tuple[Col, ...], *, primary_key: tuple[str, ...] = (),
                checks: tuple[str, ...] = (), foreign_keys: tuple[tuple[str, str], ...] = ()) -> Table:
    return Table(name, doc, cols + PROVENANCE, primary_key, checks + PROVENANCE_CHECKS, foreign_keys)


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

DB_META = Table(
    "db_meta", "Key/value facts about this database file (schema_version, dataset, ...).",
    (Col("key", "VARCHAR", False), Col("value", "VARCHAR", False)),
    primary_key=("key",),
)

INGEST_RUNS = Table(
    "ingest_runs", "One row per `quant ingest` run: what was loaded, from where (26.3 data version).",
    (
        Col("run_id", "VARCHAR", False),
        Col("sport", "VARCHAR", False),
        Col("dataset", "VARCHAR", False),
        Col("kind", "VARCHAR", False, "synthetic | snapshots | historical"),
        Col("started_at", TS, False),
        Col("finished_at", TS),
        Col("params_json", "VARCHAR"),
        Col("row_counts_json", "VARCHAR"),
        Col("data_version", "VARCHAR", doc="hash of the inputs (seed + params, or snapshot file list)"),
        Col("code_version", "VARCHAR"),
        Col("git_commit", "VARCHAR"),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("run_id",),
    checks=(_in("kind", ("synthetic", "snapshots", "historical")),),
)

# ---------------------------------------------------------------------------
# Normalized sports and market data
# ---------------------------------------------------------------------------

VENUES = _normalized(
    "venues", "Ballparks. Static attributes only; park factors are computed point-in-time from results.",
    (
        Col("venue_id", "BIGINT", False),
        Col("name", "VARCHAR"),
        Col("home_team", "VARCHAR"),
        Col("lat", "DOUBLE"),
        Col("lon", "DOUBLE"),
        Col("elevation_ft", "DOUBLE"),
        Col("roof_type", "VARCHAR", doc="open | retractable | dome"),
        Col("cf_azimuth_deg", "DOUBLE", doc="compass bearing home plate -> center field (wind in/out)"),
    ),
    primary_key=("venue_id",),
    checks=(_in("roof_type", ("open", "retractable", "dome")),),
)

GAMES = _normalized(
    "games", "One row per game (latest known state). Outcome fields fill in after the game.",
    (
        Col("game_id", "VARCHAR", False, "mlb-<gamePk>, or syn-mlb-<n> for synthetic"),
        Col("sport", "VARCHAR", False),
        Col("season", "INTEGER", False),
        Col("official_date", "DATE", False),
        Col("game_type", "VARCHAR", doc="R regular | F/D/L/W postseason"),
        Col("scheduled_start", TS, False),
        Col("actual_start", TS),
        Col("home_team", "VARCHAR", False),
        Col("away_team", "VARCHAR", False),
        Col("venue_id", "BIGINT"),
        Col("double_header", "VARCHAR"),
        Col("game_number", "INTEGER"),
        Col("status", "VARCHAR"),
        Col("market_event_id", "VARCHAR", doc="moneyline event ticker"),
        Col("book_event_id", "VARCHAR"),
        Col("home_score", "INTEGER"),
        Col("away_score", "INTEGER"),
        Col("home_win", "BOOLEAN"),
        Col("outcome_available_at", TS, doc="when the result became known (label availability)"),
    ),
    primary_key=("game_id",),
    checks=(
        "home_team <> away_team",
        "home_win IS NULL OR outcome_available_at IS NOT NULL",
        "home_win IS NULL OR home_score IS NULL OR away_score IS NULL OR home_win = (home_score > away_score)",
    ),
)

CONTRACTS = _normalized(
    "contracts", "Binary game-winner contracts on the exchange.",
    (
        Col("contract_id", "VARCHAR", False, "contract id"),
        Col("game_id", "VARCHAR", False),
        Col("sport", "VARCHAR", False),
        Col("market_type", "VARCHAR", False, "ml | rl | total"),
        Col("yes_team", "VARCHAR", doc="ml/rl: team the YES side refers to"),
        Col("yes_side", "VARCHAR", doc="ml/rl: home | away; total: over"),
        Col("line", "DOUBLE"),
        Col("series_ticker", "VARCHAR"),
        Col("event_ticker", "VARCHAR"),
        Col("open_time", TS),
        Col("close_time", TS),
        Col("result", "VARCHAR", doc="yes | no | void, once settled"),
        Col("settled_at", TS),
    ),
    primary_key=("contract_id",),
    checks=(
        _in("market_type", ("ml", "rl", "total")),
        _in("yes_side", ("home", "away", "over")),
        "market_type <> 'ml' OR yes_side IN ('home', 'away')",
        _in("result", ("yes", "no", "void")),
    ),
)

MARKET_QUOTES = _normalized(
    "market_quotes", "Executable exchange quotes over time (order book, listing, candle, or synthetic).",
    (
        Col("contract_id", "VARCHAR", False),
        Col("game_id", "VARCHAR"),
        Col("quote_ts", TS, False, "when the quote was true per the source"),
        Col("quote_source", "VARCHAR", False, "listing | orderbook | candle | synthetic"),
        Col("yes_bid", "DOUBLE"),
        Col("yes_ask", "DOUBLE", doc="executable YES price"),
        Col("no_bid", "DOUBLE"),
        Col("no_ask", "DOUBLE", doc="executable NO price = 1 - yes_bid"),
        Col("last_price", "DOUBLE", doc="stored for completeness; never used as a probability"),
        Col("bid_size", "DOUBLE"),
        Col("ask_size", "DOUBLE"),
        Col("depth_json", "VARCHAR", doc='{"yes_bids": [[p, q], ..], "yes_asks": [[p, q], ..]}'),
        Col("volume", "DOUBLE"),
        Col("open_interest", "DOUBLE"),
        Col("http_age_s", "DOUBLE", doc="CDN Age header, if the quote came from a cache"),
        Col("host", "VARCHAR", doc="recording host that observed the quote; "
                                   "NULL for candle and synthetic rows"),
    ),
    checks=(
        _in("quote_source", ("listing", "orderbook", "candle", "synthetic")),
        _prob("yes_bid"), _prob("yes_ask"), _prob("no_bid"), _prob("no_ask"), _prob("last_price"),
        "bid_size IS NULL OR bid_size >= 0",
        "ask_size IS NULL OR ask_size >= 0",
        "quote_ts <= available_at",
    ),
)

CONSENSUS_ODDS = _normalized(
    "consensus_odds", "Sportsbook prices per book over time, with proportional de-vig per book.",
    (
        Col("game_id", "VARCHAR", False),
        Col("book", "VARCHAR", False),
        Col("market_type", "VARCHAR", False, "ml | rl | total"),
        Col("line", "DOUBLE"),
        Col("odds_ts", TS, False, "book-reported last update"),
        Col("price_home", "DOUBLE", doc="American odds"),
        Col("price_away", "DOUBLE"),
        Col("price_over", "DOUBLE"),
        Col("price_under", "DOUBLE"),
        Col("p_home_devig", "DOUBLE"),
        Col("p_away_devig", "DOUBLE"),
        Col("devig_method", "VARCHAR"),
    ),
    checks=(
        _in("market_type", ("ml", "rl", "total")),
        _prob("p_home_devig"), _prob("p_away_devig"),
        "p_home_devig IS NULL OR p_away_devig IS NULL OR abs(p_home_devig + p_away_devig - 1) < 1e-6",
        "odds_ts <= available_at",
    ),
)

INFO_EVENT_TYPES = ("lineup", "starter", "injury", "transaction", "weather", "official", "status", "postponed")

INFO_EVENTS = _normalized(
    "info_events", "Timestamped information changes (news). available_at = first time the state was public.",
    (
        Col("event_id", "VARCHAR", False, "hash of game, type, side, payload, available_at"),
        Col("game_id", "VARCHAR"),
        Col("team", "VARCHAR"),
        Col("side", "VARCHAR", doc="home | away"),
        Col("type", "VARCHAR", False),
        Col("event_ts", TS, doc="provider-reported event time, if any"),
        Col("payload_json", "VARCHAR"),
        Col("first_seen", "BOOLEAN", doc="state existed before collection started (timestamp is an upper bound only)"),
    ),
    primary_key=("event_id",),
    checks=(_in("type", INFO_EVENT_TYPES), _in("side", ("home", "away"))),
)

MLB_PLAYERS = _normalized(
    "mlb_players", "Players and handedness.",
    (
        Col("player_id", "BIGINT", False),
        Col("name", "VARCHAR"),
        Col("throws", "VARCHAR"),
        Col("bats", "VARCHAR"),
        Col("primary_position", "VARCHAR"),
    ),
    primary_key=("player_id",),
    checks=(_in("throws", ("L", "R")), _in("bats", ("L", "R", "S"))),
)

MLB_PROBABLE_STARTERS = _normalized(
    "mlb_probable_starters", "Probable-starter announcements; one row per (game, side, announcement).",
    (
        Col("game_id", "VARCHAR", False),
        Col("side", "VARCHAR", False),
        Col("team", "VARCHAR", False),
        Col("player_id", "BIGINT", False),
    ),
    primary_key=("game_id", "side", "available_at"),
    checks=(_in("side", ("home", "away")),),
)

MLB_LINEUPS = _normalized(
    "mlb_lineups", "Posted batting orders; each posting (or change) is a new version keyed by available_at.",
    (
        Col("game_id", "VARCHAR", False),
        Col("side", "VARCHAR", False),
        Col("team", "VARCHAR", False),
        Col("batting_order", "INTEGER", False),
        Col("player_id", "BIGINT", False),
        Col("position", "VARCHAR"),
    ),
    primary_key=("game_id", "side", "available_at", "batting_order"),
    checks=(_in("side", ("home", "away")), "batting_order BETWEEN 1 AND 9"),
)

MLB_WEATHER = _normalized(
    "mlb_weather", "Game-time weather. Only forecasts issued before a snapshot may feed a feature.",
    (
        Col("game_id", "VARCHAR", False),
        Col("kind", "VARCHAR", False, "forecast | reported (reported = at game time; not usable pregame)"),
        Col("forecast_for", TS),
        Col("temperature_f", "DOUBLE"),
        Col("wind_speed_mph", "DOUBLE"),
        Col("wind_direction_deg", "DOUBLE", doc="meteorological: direction the wind comes FROM"),
        Col("humidity_pct", "DOUBLE"),
        Col("precip_prob", "DOUBLE"),
        Col("roof_status", "VARCHAR", doc="open | closed | unknown"),
    ),
    checks=(_in("kind", ("forecast", "reported")), _in("roof_status", ("open", "closed", "unknown"))),
)

_LINE_CHECKS = (_in("side", ("home", "away")),)

MLB_PITCHING_LINES = _normalized(
    "mlb_pitching_lines", "Per-game pitching lines. available_at = when the box score was final.",
    (
        Col("game_id", "VARCHAR", False),
        Col("official_date", "DATE", False),
        Col("side", "VARCHAR", False),
        Col("team", "VARCHAR", False),
        Col("player_id", "BIGINT", False),
        Col("is_starter", "BOOLEAN", False),
        Col("outs", "INTEGER"),
        Col("batters_faced", "INTEGER"),
        Col("pitches", "INTEGER"),
        Col("strikeouts", "INTEGER"),
        Col("walks", "INTEGER"),
        Col("hit_by_pitch", "INTEGER"),
        Col("home_runs", "INTEGER"),
        Col("hits", "INTEGER"),
        Col("earned_runs", "INTEGER"),
        Col("ground_balls", "INTEGER"),
        Col("fly_balls", "INTEGER"),
        Col("saves", "INTEGER"),
        Col("holds", "INTEGER"),
    ),
    primary_key=("game_id", "player_id"),
    checks=_LINE_CHECKS,
)

MLB_BATTING_LINES = _normalized(
    "mlb_batting_lines", "Per-game batting lines. available_at = when the box score was final.",
    (
        Col("game_id", "VARCHAR", False),
        Col("official_date", "DATE", False),
        Col("side", "VARCHAR", False),
        Col("team", "VARCHAR", False),
        Col("player_id", "BIGINT", False),
        Col("batting_order", "INTEGER"),
        Col("plate_appearances", "INTEGER"),
        Col("at_bats", "INTEGER"),
        Col("hits", "INTEGER"),
        Col("doubles", "INTEGER"),
        Col("triples", "INTEGER"),
        Col("home_runs", "INTEGER"),
        Col("walks", "INTEGER"),
        Col("hit_by_pitch", "INTEGER"),
        Col("strikeouts", "INTEGER"),
        Col("sac_flies", "INTEGER"),
    ),
    primary_key=("game_id", "player_id"),
    checks=_LINE_CHECKS,
)

# ---------------------------------------------------------------------------
# Research tables
# ---------------------------------------------------------------------------

FEATURE_FAMILIES = ("market", "starter", "offense", "bullpen", "environment", "negative_control")

FEATURE_DEFINITIONS = Table(
    "feature_definitions",
    "Point-in-time feature registry. A value can't be stored for an unregistered feature.",
    (
        Col("feature_name", "VARCHAR", False),
        Col("family", "VARCHAR", False, "Sec 8 Level 1 family, or negative_control (26.16)"),
        Col("dtype", "VARCHAR", False),
        Col("description", "VARCHAR", False),
        Col("availability_rule", "VARCHAR", False, "how available_at is derived from the inputs"),
        Col("inputs", "VARCHAR", doc="source tables/columns"),
        Col("hypothesis_id", "VARCHAR"),
        Col("expected_direction", "VARCHAR"),
        Col("version", "VARCHAR", False),
    ),
    primary_key=("feature_name",),
    checks=(_in("family", FEATURE_FAMILIES),),
)

OBSERVATIONS = Table(
    "observations",
    "Sec 5 observation: one per contract per snapshot offset. Only valid quotes become observations; "
    "everything else is recorded in snapshot_exclusions.",
    (
        Col("observation_id", "VARCHAR", False, "<contract_id>@T-<offset>"),
        Col("game_id", "VARCHAR", False, "event_id in Sec 5"),
        Col("contract_id", "VARCHAR", False),
        Col("sport", "VARCHAR", False),
        Col("season", "INTEGER", False),
        Col("official_date", "DATE", False),
        Col("home_team", "VARCHAR", False),
        Col("away_team", "VARCHAR", False),
        Col("yes_team", "VARCHAR", False),
        Col("yes_side", "VARCHAR", False),
        Col("event_start_time", TS, False),
        Col("snapshot_offset_min", "INTEGER", False),
        Col("snapshot_time", TS, False),
        Col("is_canonical", "BOOLEAN", False, "the one T-30 decision snapshot (26.10)"),
        Col("quote_ts", TS, False),
        Col("quote_available_at", TS, False),
        Col("quote_age_s", "DOUBLE", False),
        Col("quote_source", "VARCHAR", False),
        Col("market_yes_bid", "DOUBLE", False),
        Col("market_yes_ask", "DOUBLE", False),
        Col("market_no_bid", "DOUBLE"),
        Col("market_no_ask", "DOUBLE"),
        Col("market_last", "DOUBLE", doc="kept because Sec 5 lists it; never a probability or a feature"),
        Col("spread", "DOUBLE", False),
        Col("bid_size", "DOUBLE"),
        Col("ask_size", "DOUBLE"),
        Col("depth_json", "VARCHAR"),
        Col("size_verified", "BOOLEAN", False,
            "displayed size seen nonzero on both sides. False only for historical candles, which "
            "report bid/ask but no size: 26.10's size rule is unverified for those rows"),
        Col("quote_quality", "VARCHAR", False,
            "executable | candle_degraded: candle rows are never an executable-price result"),
        Col("volume", "DOUBLE"),
        Col("open_interest", "DOUBLE"),
        Col("p_mid", "DOUBLE", False, "market probability = midpoint of the executable bid/ask"),
        Col("p_consensus", "DOUBLE", doc="external_market_probability: de-vigged book consensus"),
        Col("consensus_books", "INTEGER"),
        Col("consensus_available_at", TS),
        Col("cross_market_gap", "DOUBLE", doc="p_consensus - p_mid"),
        Col("liquidity_tercile", "INTEGER", doc="1 = least liquid; cut points from earlier data only"),
        Col("news_event_count_t180_t30", "INTEGER"),
        Col("outcome", "INTEGER", doc="1 if the contract resolved YES"),
        Col("outcome_available_at", TS),
        Col("build_run_id", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("observation_id",),
    checks=(
        "quote_available_at <= snapshot_time",
        "quote_ts <= quote_available_at",
        "consensus_available_at IS NULL OR consensus_available_at <= quote_ts",
        "p_consensus IS NULL OR consensus_available_at IS NOT NULL",
        "snapshot_offset_min > 0",
        "snapshot_time = event_start_time - to_minutes(snapshot_offset_min)",
        f"is_canonical = (snapshot_offset_min = {CANONICAL_OFFSET_MIN})",
        "quote_age_s >= 0",
        "market_yes_bid > 0 AND market_yes_bid < market_yes_ask AND market_yes_ask < 1",
        "abs(spread - (market_yes_ask - market_yes_bid)) < 1e-9",
        "p_mid >= market_yes_bid AND p_mid <= market_yes_ask",
        "NOT size_verified OR (bid_size > 0 AND ask_size > 0)",
        "size_verified OR quote_source = 'candle'",
        _in("quote_quality", QUOTE_QUALITIES),
        f"(quote_quality = '{CANDLE_DEGRADED}') = (quote_source = 'candle')",
        _prob("p_consensus"),
        _in("yes_side", ("home", "away")),
        "outcome IS NULL OR outcome IN (0, 1)",
        "liquidity_tercile IS NULL OR liquidity_tercile BETWEEN 1 AND 3",
    ),
)

EXCLUSION_REASONS = ("no_quote", "stale", "one_sided", "crossed", "no_size", "wide_spread")

SNAPSHOT_EXCLUSIONS = Table(
    "snapshot_exclusions", "Snapshots that produced NO OBSERVATION, and why (26.10).",
    (
        Col("game_id", "VARCHAR", False),
        Col("contract_id", "VARCHAR", False),
        Col("official_date", "DATE", False),
        Col("snapshot_offset_min", "INTEGER", False),
        Col("snapshot_time", TS, False),
        Col("reason", "VARCHAR", False),
        Col("detail", "VARCHAR"),
        Col("build_run_id", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("contract_id", "snapshot_offset_min"),
    checks=(_in("reason", EXCLUSION_REASONS),),
)

FEATURE_VALUES = Table(
    "feature_values",
    "Point-in-time feature values, long format: one row per observation x feature.",
    (
        Col("observation_id", "VARCHAR", False),
        Col("feature_name", "VARCHAR", False),
        Col("value", "DOUBLE", doc="NULL = missing (not imputed here)"),
        Col("available_at", TS, doc="latest availability time of every input used"),
        Col("snapshot_time", TS, False, "copied from the observation"),
        Col("cutoff_time", TS, False,
            "information cutoff = the observation's quote_ts (26.11: features must precede the quote, "
            "not merely the snapshot or the game start)"),
        Col("source_ref", "VARCHAR", doc="what the value was computed from"),
        Col("build_run_id", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("observation_id", "feature_name"),
    foreign_keys=(("feature_name", "feature_definitions(feature_name)"),),
    checks=(
        "available_at <= cutoff_time",
        "cutoff_time <= snapshot_time",
        "value IS NULL OR available_at IS NOT NULL",
    ),
)

WALK_FORWARD_PERIODS = Table(
    "walk_forward_periods", "Chronological periods, fixed in config before any results (26.2, 26.3). Append-only: a "
    "period_id, once written, keeps its role and window; a train appends only new ids.",
    (
        Col("period_id", "VARCHAR", False),
        Col("ordinal", "INTEGER", False),
        Col("role", "VARCHAR", False, "train_only | evaluation | holdout"),
        Col("start_date", "DATE", False),
        Col("end_date", "DATE", False),
        Col("definition_hash", "VARCHAR", False),
    ),
    primary_key=("period_id",),
    checks=(_in("role", ("train_only", "evaluation", "holdout")), "start_date <= end_date"),
    append_only=True,
)

MODEL_TYPES = ("market", "logistic_residual", "lgbm_residual")

MODEL_RUNS = Table(
    "model_runs", "One fitted model for one prediction period.",
    (
        Col("model_run_id", "VARCHAR", False),
        Col("model_type", "VARCHAR", False),
        Col("period_id", "VARCHAR", False, "the period this model predicts"),
        Col("feature_set_id", "VARCHAR", False),
        Col("feature_set_hash", "VARCHAR", False),
        Col("config_hash", "VARCHAR", False),
        Col("params_json", "VARCHAR"),
        Col("calibration_method", "VARCHAR", doc="none | platt | isotonic, chosen on earlier data only"),
        Col("n_train_obs", "INTEGER"),
        Col("n_train_games", "INTEGER"),
        Col("train_data_through", TS, doc="latest outcome_available_at among training rows"),
        Col("predict_from", TS, False, "earliest snapshot_time this model predicts"),
        Col("artifact_path", "VARCHAR"),
        Col("dataset", "VARCHAR", False),
        Col("git_commit", "VARCHAR"),
        Col("created_at", TS, False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("model_run_id",),
    checks=(
        _in("model_type", MODEL_TYPES),
        _in("calibration_method", ("none", "platt", "isotonic")),
        "train_data_through IS NULL OR train_data_through < predict_from",
    ),
    append_only=True,          # one row set per `quant train`, never replaced (ADR 0002 item 4)
)

PREDICTIONS = Table(
    "predictions", "Out-of-sample predictions with uncertainty bounds (26.12).",
    (
        Col("model_run_id", "VARCHAR", False),
        Col("observation_id", "VARCHAR", False),
        Col("period_id", "VARCHAR", False),
        Col("p_market", "DOUBLE", False),
        Col("p_model_raw", "DOUBLE"),
        Col("p_model", "DOUBLE", False, "after calibration"),
        Col("lower_bound", "DOUBLE", False),
        Col("upper_bound", "DOUBLE", False),
        Col("p_conservative_yes", "DOUBLE", False),
        Col("p_conservative_no", "DOUBLE", False),
        Col("uncertainty_json", "VARCHAR", doc="components of the interval"),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("model_run_id", "observation_id"),
    checks=(
        _prob("p_market"), _prob("p_model_raw"), _prob("p_model"),
        "lower_bound >= 0 AND upper_bound <= 1",
        "lower_bound <= p_model AND p_model <= upper_bound",
        "p_conservative_yes = lower_bound",
        "abs(p_conservative_no - (1 - upper_bound)) < 1e-12",
    ),
    append_only=True,          # ADR 0002 item 4
)

LINEAGE_KINDS = ("model_run", "experiment", "hypothesis")

TRAIN_LINEAGE = Table(
    "train_lineage", "What each `quant train` invocation produced and read: its model runs, the experiments it "
    "logged and the hypotheses it loaded. Written in the transaction that writes the model runs. "
    "The review checks a run whose feeding model runs all have lineage against these rows only.",
    (
        Col("invocation_id", "VARCHAR", False, "ti-<hash of dataset, commit, time, run config, quote quality>"),
        Col("kind", "VARCHAR", False, "model_run | experiment | hypothesis"),
        Col("row_id", "VARCHAR", False, "model_run_id, experiment_id or hypothesis_id"),
        Col("dataset", "VARCHAR", False),
        Col("git_commit", "VARCHAR", doc="the invocation's commit"),
        Col("created_at", TS, False, "the invocation's time"),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("invocation_id", "kind", "row_id"),
    checks=(_in("kind", LINEAGE_KINDS),),
    append_only=True,
)

FEE_SCHEDULES = Table(
    "fee_schedules", "Mirror of the versioned FeeSchedule registry in code (26.8).",
    (
        Col("fee_schedule_id", "VARCHAR", False),
        Col("venue", "VARCHAR", False),
        Col("maker_or_taker", "VARCHAR", False),
        Col("effective_from", "DATE", False),
        Col("effective_to", "DATE"),
        Col("formula", "VARCHAR", False, "id of the implementation in code"),
        Col("params_json", "VARCHAR"),
        Col("verified", "BOOLEAN", False, "false until checked against the venue's published schedule"),
        Col("source_note", "VARCHAR"),
    ),
    primary_key=("fee_schedule_id",),
    checks=(_in("maker_or_taker", ("maker", "taker")),
            "effective_to IS NULL OR effective_from <= effective_to"),
)

BACKTEST_RUNS = Table(
    "backtest_runs", "One `quant backtest` run.",
    (
        Col("backtest_run_id", "VARCHAR", False),
        Col("model_type", "VARCHAR", False),
        Col("snapshot_offset_min", "INTEGER", False),
        Col("config_json", "VARCHAR", False),
        Col("config_hash", "VARCHAR", False),
        Col("quote_quality", "VARCHAR", False, "the one quote class this run used"),
        Col("dataset", "VARCHAR", False),
        Col("git_commit", "VARCHAR"),
        Col("created_at", TS, False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("backtest_run_id",),
    checks=(_in("quote_quality", QUOTE_QUALITIES),),
    append_only=True,
)

DECISIONS = ("BUY_YES", "BUY_NO", "PASS")

PASS_REASONS = ("below_min_edge", "wide_spread", "insufficient_depth", "no_price", "missing_prediction",
                "same_game_position")  # the game's mirror contract (same exposure) was the better trade

# Exit leg. hold_to_settlement never sells; exit_at_offset sells at a later pre-game snapshot.
HOLD_TO_SETTLEMENT, EXIT_AT_OFFSET = "hold_to_settlement", "exit_at_offset"
EXIT_RULES = (HOLD_TO_SETTLEMENT, EXIT_AT_OFFSET)
EXIT_STATUSES = ("held", "exited", "partial")  # partial: part sold, the rest held to settlement
EXIT_FALLBACK_REASONS = (
    "exit_not_after_entry",     # the exit snapshot is not later than the entry (e.g. a T-10 entry, T-10 exit)
    "no_valid_exit_quote",      # no 26.10-valid observation at the exit snapshot
    "no_exit_price",            # the side's bid, less the penalty, is at or below 1 - NO_PRICE_AT
    "insufficient_exit_depth",  # displayed bid-side size could not take the whole position
)

BACKTEST_DECISIONS = Table(
    "backtest_decisions",
    "Every observation's decision (trades and passes) per slippage scenario (26.6, 26.8, 26.9).\n"
    "Semantics (all money in dollars, one contract pays $1):\n"
    "  side            the side considered: the one with the larger net_edge (set on PASS rows too)\n"
    "  quoted_price    executable ask of that side: yes_ask, or no_ask = 1 - yes_bid\n"
    "  execution_price quoted_price + spread_penalty_frac * spread (never the midpoint)\n"
    "  predicted_edge  p_model(side) - quoted_price; edge buckets (26.6) use this\n"
    "  net_edge        p_conservative(side) - execution_price - fee per contract\n"
    "  stake           quantity * execution_price\n"
    "  slippage_cost   quantity * (execution_price - quoted_price)\n"
    "  gross_pnl       quantity * (side_won - quoted_price)   (before fees and slippage)\n"
    "  net_pnl         gross_pnl - slippage_cost - calculated_fee\n"
    "Exit leg (exit_rule exit_at_offset). The sale crosses the bid of the side held:\n"
    "  exit_quoted_price  yes_bid, or no_bid = 1 - yes_ask, at exit_snapshot_time\n"
    "  exit_price         exit_quoted_price - spread_penalty_frac * spread (never above the bid)\n"
    "  exit_proceeds      exit_quantity * exit_price; exit_fee from the schedule in force at exit\n"
    "  exit_slippage_cost exit_quantity * (exit_quoted_price - exit_price)\n"
    "  held_quantity      quantity - exit_quantity, held to settlement\n"
    "With an exit, gross_pnl and net_pnl are round-trip:\n"
    "  gross_pnl  exit_quantity * (exit_quoted_price - quoted_price) + held_quantity * (side_won - quoted_price)\n"
    "  net_pnl    exit_proceeds + held_quantity * side_won - stake - calculated_fee - exit_fee\n"
    "             = gross_pnl - slippage_cost - exit_slippage_cost - calculated_fee - exit_fee\n"
    "A planned exit that cannot happen is held (or partly held) with exit_reason set, never hidden.\n"
    "The trade/pass decision is made once at the baseline penalty; the other scenarios re-price the\n"
    "same trades, so they isolate execution sensitivity (26.9).",
    (
        Col("backtest_run_id", "VARCHAR", False),
        Col("scenario", "VARCHAR", False, "spread penalty: pen_0 | pen_25 | pen_50 | pen_100"),
        Col("model_run_id", "VARCHAR", doc="model that produced p_model"),
        Col("observation_id", "VARCHAR", False),
        Col("game_id", "VARCHAR", False),
        Col("contract_id", "VARCHAR", False),
        Col("period_id", "VARCHAR", False),
        Col("snapshot_time", TS, False),
        Col("decision", "VARCHAR", False),
        Col("pass_reason", "VARCHAR"),
        Col("side", "VARCHAR", doc="yes | no"),
        Col("p_market", "DOUBLE", False),
        Col("p_model", "DOUBLE", False),
        Col("p_conservative", "DOUBLE", doc="for the side considered"),
        Col("predicted_edge", "DOUBLE", doc="p_model - executable price, side considered"),
        Col("net_edge", "DOUBLE", doc="p_conservative - price - fee - slippage"),
        Col("quoted_price", "DOUBLE", doc="executable ask of the side bought"),
        Col("spread_penalty_frac", "DOUBLE", False),
        Col("execution_price", "DOUBLE"),
        Col("quantity", "DOUBLE"),
        Col("stake", "DOUBLE"),
        Col("fee_schedule_id", "VARCHAR"),
        Col("fee_schedule_effective_date", "DATE"),
        Col("maker_or_taker", "VARCHAR"),
        Col("calculated_fee", "DOUBLE"),
        Col("slippage_cost", "DOUBLE"),
        Col("depth_verified", "BOOLEAN", doc="quantity capped by displayed top-of-book size (false: size unknown)"),
        Col("outcome", "INTEGER", doc="1 if the contract resolved YES"),
        Col("side_won", "INTEGER", doc="outcome for the side considered"),
        Col("gross_pnl", "DOUBLE", doc="round trip before fees and slippage"),
        Col("net_pnl", "DOUBLE", doc="round trip after both fees and both slippages"),
        Col("exit_rule", "VARCHAR", False, "hold_to_settlement | exit_at_offset"),
        Col("exit_status", "VARCHAR", doc="trades only: held | exited | partial"),
        Col("exit_reason", "VARCHAR", doc="why a planned exit was held or partial"),
        Col("exit_observation_id", "VARCHAR", doc="observation that priced the exit"),
        Col("exit_snapshot_time", TS, doc="planned exit time (set on fallbacks too)"),
        Col("exit_quoted_price", "DOUBLE", doc="executable bid of the side sold"),
        Col("exit_price", "DOUBLE"),
        Col("exit_quantity", "DOUBLE"),
        Col("held_quantity", "DOUBLE", doc="contracts held to settlement"),
        Col("exit_proceeds", "DOUBLE"),
        Col("exit_fee_schedule_id", "VARCHAR"),
        Col("exit_fee_schedule_effective_date", "DATE"),
        Col("exit_fee", "DOUBLE"),
        Col("exit_slippage_cost", "DOUBLE"),
        Col("exit_depth_verified", "BOOLEAN"),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    primary_key=("backtest_run_id", "scenario", "observation_id"),
    checks=(
        _in("decision", DECISIONS),
        _in("pass_reason", PASS_REASONS),
        "decision <> 'PASS' OR pass_reason IS NOT NULL",
        "side_won IS NULL OR side_won IN (0, 1)",
        _in("side", ("yes", "no")),
        "decision = 'PASS' OR (side IS NOT NULL AND quoted_price IS NOT NULL AND execution_price IS NOT NULL "
        "AND quantity > 0 AND fee_schedule_id IS NOT NULL AND calculated_fee IS NOT NULL "
        "AND maker_or_taker IS NOT NULL AND fee_schedule_effective_date IS NOT NULL)",
        "decision = 'PASS' OR execution_price >= quoted_price",
        "spread_penalty_frac >= 0",
        "outcome IS NULL OR outcome IN (0, 1)",
        _in("exit_rule", EXIT_RULES),
        _in("exit_status", EXIT_STATUSES),
        _in("exit_reason", EXIT_FALLBACK_REASONS),
        "(decision = 'PASS') = (exit_status IS NULL)",
        # hold_to_settlement never sells and has nothing to explain
        f"exit_rule <> '{HOLD_TO_SETTLEMENT}' OR ((exit_status IS NULL OR exit_status = 'held') "
        "AND exit_reason IS NULL AND exit_quantity IS NULL AND exit_snapshot_time IS NULL)",
        # a planned exit that did not fully happen carries its reason (never hidden)
        f"exit_rule <> '{EXIT_AT_OFFSET}' OR exit_status IS NULL OR exit_status = 'exited' "
        "OR exit_reason IS NOT NULL",
        "exit_status IS NULL OR exit_status = 'held' OR (exit_quoted_price IS NOT NULL AND exit_price IS NOT NULL "
        "AND exit_quantity > 0 AND exit_proceeds IS NOT NULL AND exit_fee IS NOT NULL "
        "AND exit_fee_schedule_id IS NOT NULL AND exit_fee_schedule_effective_date IS NOT NULL "
        "AND exit_snapshot_time IS NOT NULL AND exit_snapshot_time > snapshot_time "
        "AND exit_price <= exit_quoted_price)",
        "exit_status IS NULL OR exit_status <> 'held' OR (exit_quantity IS NULL AND exit_price IS NULL "
        "AND exit_fee IS NULL AND exit_proceeds IS NULL)",
        # NULL-safe: a check that evaluates to NULL passes, so held_quantity must be present
        "exit_status IS NULL OR (held_quantity IS NOT NULL "
        "AND abs(held_quantity + coalesce(exit_quantity, 0) - quantity) < 1e-9)",
        "exit_status IS NULL OR (exit_status = 'exited') = (held_quantity = 0)",
    ),
    append_only=True,
)

DERIVED_TABLES: tuple[Table, ...] = (
    DB_META, INGEST_RUNS,
    VENUES, GAMES, CONTRACTS, MARKET_QUOTES, CONSENSUS_ODDS, INFO_EVENTS,
    MLB_PLAYERS, MLB_PROBABLE_STARTERS, MLB_LINEUPS, MLB_WEATHER, MLB_PITCHING_LINES, MLB_BATTING_LINES,
    FEATURE_DEFINITIONS, OBSERVATIONS, SNAPSHOT_EXCLUSIONS, FEATURE_VALUES,
    WALK_FORWARD_PERIODS, MODEL_RUNS, PREDICTIONS, TRAIN_LINEAGE, FEE_SCHEDULES, BACKTEST_RUNS,
    BACKTEST_DECISIONS,
)

# ---------------------------------------------------------------------------
# Persistent logs: append-only Parquet under data/<log>/<sport>_<dataset>/,
# exposed as DuckDB views. Checks are validated before each write.
# ---------------------------------------------------------------------------

# Paper trading (Sec 15, 21). Paper passes add two sizing reasons to the backtest's.
PAPER_PASS_REASONS = PASS_REASONS + (
    "exposure_cap",               # the per-game or per-day cap left too little to buy the minimum fill
    "kelly_size_below_minimum",   # the Kelly stake, uncapped, buys fewer than the minimum fill
)
PAPER_CAPS = ("position", "game", "daily")
MANIFEST_STATUSES = ("LOCKED", "FROZEN", "UNLOCKED")

PAPER_TRADES = Table(
    "paper_trades",
    "Paper decisions, logged before the game starts (Sec 21). Never edited; exits and settlement are\n"
    "separate records. Every decision is logged, passes included, so paper results cover all decisions.\n"
    "  decision_ts     the canonical snapshot the decision was made at (only data available then)\n"
    "  created_at      when the row was logged: live rows are logged after decision_ts and before the game\n"
    "  bid, ask        the YES quote at decision_ts, never a later or closing price\n"
    "  p_market        midpoint of that quote; model_probability = the model's YES probability\n"
    "  quoted_price    executable ask of the side considered: ask (yes) or 1 - bid (no)\n"
    "  entry_price     quoted_price + spread_penalty_frac * spread, or the depth VWAP if higher\n"
    "  hypothetical_size contracts (0 on a pass); stake = hypothetical_size * entry_price\n"
    "  requested_stake fractional-Kelly dollars; allowed_stake after the position, game and daily caps\n"
    "  freeze_manifest_hash set only when the trade qualifies for 26.18 (frozen configuration in force)",
    (
        Col("paper_trade_id", "VARCHAR", False, "pt-<hash of dataset, observation_id>: one decision per snapshot"),
        Col("created_at", TS, False),
        Col("sport", "VARCHAR", False),
        Col("dataset", "VARCHAR", False),
        Col("game_id", "VARCHAR", False),
        Col("contract_id", "VARCHAR", False),
        Col("observation_id", "VARCHAR", False),
        Col("official_date", "DATE", False),
        Col("event_start_time", TS, False),
        Col("snapshot_offset_min", "INTEGER", False),
        Col("quote_quality", "VARCHAR", False),
        Col("period_id", "VARCHAR", False, "the model run's walk-forward period"),
        Col("model_run_id", "VARCHAR", False),
        Col("model_type", "VARCHAR", False),
        Col("train_data_through", TS, False, "latest outcome the model was trained on"),
        Col("manifest_status", "VARCHAR", False, "LOCKED | FROZEN | UNLOCKED at decision time"),
        Col("freeze_manifest_hash", "VARCHAR", doc="frozen configuration this trade qualifies under (26.18)"),
        Col("config_hash", "VARCHAR", False, "decision, fees, execution, backtest and paper settings"),
        Col("decision_ts", TS, False),
        Col("quote_ts", TS, False),
        Col("quote_available_at", TS, False),
        Col("bid", "DOUBLE", False, "YES bid at decision_ts"),
        Col("ask", "DOUBLE", False, "YES ask at decision_ts"),
        Col("bid_size", "DOUBLE"),
        Col("ask_size", "DOUBLE"),
        Col("depth_json", "VARCHAR"),
        Col("p_market", "DOUBLE", False),
        Col("model_probability", "DOUBLE", False),
        Col("p_conservative_yes", "DOUBLE", False),
        Col("p_conservative_no", "DOUBLE", False),
        Col("conservative_probability", "DOUBLE", doc="p_conservative of the side considered"),
        Col("predicted_edge", "DOUBLE", doc="p_model(side) - quoted_price"),
        Col("net_edge", "DOUBLE", doc="p_conservative(side) - entry_price - fee per contract"),
        Col("decision", "VARCHAR", False),
        Col("pass_reason", "VARCHAR"),
        Col("side", "VARCHAR", doc="the side considered (set on passes when one was priced)"),
        Col("hypothetical_size", "DOUBLE", False),
        Col("quoted_price", "DOUBLE"),
        Col("spread_penalty_frac", "DOUBLE", False),
        Col("entry_price", "DOUBLE"),
        Col("stake", "DOUBLE"),
        Col("fee_schedule_id", "VARCHAR"),
        Col("fee_schedule_effective_date", "DATE"),
        Col("maker_or_taker", "VARCHAR"),
        Col("calculated_fee", "DOUBLE"),
        Col("slippage_cost", "DOUBLE"),
        Col("depth_verified", "BOOLEAN"),
        Col("requested_stake", "DOUBLE"),
        Col("allowed_stake", "DOUBLE"),
        Col("capped_by", "VARCHAR", doc="position | game | daily: the cap that bound"),
        Col("exit_rule", "VARCHAR", False),
        Col("exit_offset_min", "INTEGER"),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(
        "quote_ts <= quote_available_at",
        "quote_available_at <= decision_ts",
        "decision_ts < event_start_time",
        "train_data_through < decision_ts",
        # a live decision is logged prospectively: after its snapshot, before the game starts
        "is_synthetic OR (created_at >= decision_ts AND created_at < event_start_time)",
        _in("decision", DECISIONS),
        _in("pass_reason", PAPER_PASS_REASONS),
        "(decision = 'PASS') = (pass_reason IS NOT NULL)",
        _in("side", ("yes", "no")),
        "decision <> 'BUY_YES' OR side = 'yes'",
        "decision <> 'BUY_NO' OR side = 'no'",
        "decision = 'PASS' OR (side IS NOT NULL AND quoted_price IS NOT NULL AND entry_price IS NOT NULL "
        "AND stake IS NOT NULL AND hypothetical_size > 0 AND fee_schedule_id IS NOT NULL "
        "AND fee_schedule_effective_date IS NOT NULL AND maker_or_taker IS NOT NULL AND calculated_fee IS NOT NULL "
        "AND slippage_cost IS NOT NULL AND depth_verified IS NOT NULL AND net_edge IS NOT NULL "
        "AND conservative_probability IS NOT NULL AND predicted_edge IS NOT NULL "
        "AND requested_stake IS NOT NULL AND allowed_stake IS NOT NULL)",
        "decision <> 'PASS' OR (hypothetical_size = 0 AND entry_price IS NULL AND stake IS NULL "
        "AND calculated_fee IS NULL)",
        "hypothetical_size >= 0",
        "bid > 0 AND bid < ask AND ask < 1",
        # the market probability is the midpoint of the logged executable quote, never the last trade
        "abs(p_market - (bid + ask) / 2) < 1e-9",
        # the price is the decision-time ask of the side, never a later or closing price
        "quoted_price IS NULL OR coalesce((side = 'yes' AND abs(quoted_price - ask) < 1e-9) "
        "OR (side = 'no' AND abs(quoted_price - (1 - bid)) < 1e-9), false)",
        "entry_price IS NULL OR entry_price >= quoted_price",
        "conservative_probability IS NULL OR coalesce((side = 'yes' AND conservative_probability = p_conservative_yes) "
        "OR (side = 'no' AND conservative_probability = p_conservative_no), false)",
        "spread_penalty_frac >= 0",
        "stake IS NULL OR (stake <= allowed_stake + 1e-9 AND allowed_stake <= requested_stake + 1e-9)",
        _in("capped_by", PAPER_CAPS),
        _in("exit_rule", EXIT_RULES),
        f"(exit_rule = '{EXIT_AT_OFFSET}') = (exit_offset_min IS NOT NULL)",
        _in("manifest_status", MANIFEST_STATUSES),
        "freeze_manifest_hash IS NULL OR manifest_status <> 'LOCKED'",
        "NOT is_synthetic OR freeze_manifest_hash IS NULL",   # 26.18: only live data qualifies
        _in("quote_quality", QUOTE_QUALITIES),
        _prob("p_market"), _prob("model_probability"), _prob("p_conservative_yes"), _prob("p_conservative_no"),
    ),
    kind="persistent",
)

PAPER_EXITS = Table(
    "paper_exits",
    "The planned sale of an exit_at_offset paper position, priced by the same exit leg as the\n"
    "backtest: the bid of the side held at the exit snapshot, less the spread penalty. A planned exit that\n"
    "cannot happen is recorded as held (or partial) with its reason. Hold-to-settlement trades have none.",
    (
        Col("paper_trade_id", "VARCHAR", False),
        Col("created_at", TS, False),
        Col("side", "VARCHAR", False, "the side held and sold"),
        Col("exit_rule", "VARCHAR", False),
        Col("exit_status", "VARCHAR", False, "held | exited | partial"),
        Col("exit_reason", "VARCHAR"),
        Col("exit_observation_id", "VARCHAR"),
        Col("exit_snapshot_time", TS, False, "planned exit time (set on fallbacks too)"),
        Col("exit_quote_ts", TS),
        Col("exit_quote_available_at", TS),
        Col("exit_bid", "DOUBLE", doc="YES bid at the exit snapshot"),
        Col("exit_ask", "DOUBLE", doc="YES ask at the exit snapshot"),
        Col("exit_quoted_price", "DOUBLE", doc="bid of the side sold: exit_bid (yes) or 1 - exit_ask (no)"),
        Col("exit_price", "DOUBLE"),
        Col("exit_quantity", "DOUBLE"),
        Col("held_quantity", "DOUBLE", False),
        Col("exit_proceeds", "DOUBLE"),
        Col("exit_fee_schedule_id", "VARCHAR"),
        Col("exit_fee_schedule_effective_date", "DATE"),
        Col("exit_fee", "DOUBLE"),
        Col("exit_slippage_cost", "DOUBLE"),
        Col("exit_depth_verified", "BOOLEAN"),
        Col("spread_penalty_frac", "DOUBLE", False),
        Col("dataset", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(
        f"exit_rule = '{EXIT_AT_OFFSET}'",
        _in("side", ("yes", "no")),
        _in("exit_status", EXIT_STATUSES),
        _in("exit_reason", EXIT_FALLBACK_REASONS),
        "exit_status = 'exited' OR exit_reason IS NOT NULL",
        "exit_status = 'held' OR (exit_observation_id IS NOT NULL AND exit_bid IS NOT NULL AND exit_ask IS NOT NULL "
        "AND exit_quoted_price IS NOT NULL AND exit_price IS NOT NULL AND exit_quantity > 0 "
        "AND exit_proceeds IS NOT NULL AND exit_fee IS NOT NULL AND exit_fee_schedule_id IS NOT NULL "
        "AND exit_fee_schedule_effective_date IS NOT NULL AND exit_price <= exit_quoted_price)",
        "exit_status <> 'held' OR (exit_quantity IS NULL AND exit_price IS NULL AND exit_fee IS NULL "
        "AND exit_proceeds IS NULL)",
        "held_quantity >= 0",
        "(exit_status = 'exited') = (held_quantity = 0)",
        # the sale crosses the exit-time bid of the side held, never a later or closing price
        "exit_quoted_price IS NULL OR coalesce((side = 'yes' AND abs(exit_quoted_price - exit_bid) < 1e-9) "
        "OR (side = 'no' AND abs(exit_quoted_price - (1 - exit_ask)) < 1e-9), false)",
        "exit_observation_id IS NULL OR (exit_quote_ts <= exit_quote_available_at "
        "AND exit_quote_available_at <= exit_snapshot_time)",
        "spread_penalty_frac >= 0",
        # a live exit is logged only once its snapshot has happened
        "is_synthetic OR created_at >= exit_snapshot_time",
    ),
    kind="persistent",
)

PAPER_SETTLEMENTS = Table(
    "paper_settlements",
    "Resolution of a paper decision (Sec 21), passes included (outcome only), so paper Brier scores cover\n"
    "every decision. P&L comes from the position rebuilt from paper_trades and paper_exits:\n"
    "  pnl      gross round trip (before fees and slippage); fees = entry fee + exit fee\n"
    "  net_pnl  exit proceeds + held_quantity * side_won - stake - fees\n"
    "Void settlement is not implemented: no settlement feed is recorded.",
    (
        Col("paper_trade_id", "VARCHAR", False),
        Col("settled_at", TS, False),
        Col("outcome", "INTEGER"),
        Col("outcome_available_at", TS),
        Col("result", "VARCHAR", False, "yes | no | void"),
        Col("decision", "VARCHAR", False),
        Col("side", "VARCHAR"),
        Col("side_won", "INTEGER"),
        Col("pnl", "DOUBLE"),
        Col("fees", "DOUBLE"),
        Col("net_pnl", "DOUBLE"),
        Col("held_quantity", "DOUBLE"),
        Col("exit_proceeds", "DOUBLE"),
        Col("dataset", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(
        _in("result", ("yes", "no", "void")),
        "outcome IS NULL OR outcome IN (0, 1)",
        "result = 'void' OR (outcome = 1 AND result = 'yes') OR (outcome = 0 AND result = 'no')",
        "result <> 'void' OR outcome IS NULL",
        "result = 'void' OR (outcome_available_at IS NOT NULL AND settled_at >= outcome_available_at)",
        _in("decision", DECISIONS),
        _in("side", ("yes", "no")),
        "side_won IS NULL OR (side = 'yes' AND side_won = outcome) OR (side = 'no' AND side_won = 1 - outcome)",
        "decision = 'PASS' OR result = 'void' OR (side IS NOT NULL AND side_won IS NOT NULL AND pnl IS NOT NULL "
        "AND net_pnl IS NOT NULL AND fees >= 0 AND held_quantity >= 0)",
        "decision <> 'PASS' OR (pnl IS NULL AND net_pnl IS NULL AND fees IS NULL)",
    ),
    kind="persistent",
)

HYPOTHESES = Table(
    "hypotheses", "Hypothesis registry (26.4). Registered before results are examined; never deleted.",
    (
        Col("hypothesis_id", "VARCHAR", False),
        Col("family", "VARCHAR", False),
        Col("description", "VARCHAR", False),
        Col("mechanism", "VARCHAR", False),
        Col("feature_definitions_json", "VARCHAR", False),
        Col("expected_direction", "VARCHAR", False),
        Col("minimum_sample", "INTEGER", False),
        Col("evaluation_periods_json", "VARCHAR", False),
        Col("primary_metric", "VARCHAR", False),
        Col("label", "VARCHAR", False, "CONFIRMATORY | EXPLORATORY"),
        Col("definition_hash", "VARCHAR", False),
        Col("git_commit", "VARCHAR"),
        Col("registered_at", TS, False),
        Col("dataset", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(_in("label", ("CONFIRMATORY", "EXPLORATORY")),),
    kind="persistent",
)

EXPERIMENTS = Table(
    "experiments", "Experiment registry (Sec 20, 26.5): successes and failures alike.",
    (
        Col("experiment_id", "VARCHAR", False),
        Col("hypothesis_id", "VARCHAR"),
        Col("experiment_family", "VARCHAR", False, "BH-FDR is applied within a family"),
        Col("label", "VARCHAR", False),
        Col("feature_set_id", "VARCHAR", False),
        Col("feature_set_hash", "VARCHAR", False),
        Col("model_type", "VARCHAR", False),
        Col("config_hash", "VARCHAR", False),
        Col("train_period", "VARCHAR", False),
        Col("test_period", "VARCHAR", False),
        Col("n_games", "INTEGER"),
        Col("metrics_json", "VARCHAR", False,
            'candidate vs reference on the same games: {"reference": .., "candidate": .., "n_obs": .., '
            '"brier_ref": .., "brier_cand": .., "delta_brier": ref - cand (positive = candidate better), '
            '"logloss_ref": .., "logloss_cand": .., "ece_ref": .., "ece_cand": .., "roi_ref": .., "roi_cand": ..}'),
        Col("p_value", "DOUBLE", doc="one-sided paired game-bootstrap p-value for delta_brier > 0"),
        Col("quote_quality", "VARCHAR", False, "quote class of the observations tested"),
        Col("git_commit", "VARCHAR"),
        Col("created_at", TS, False),
        Col("dataset", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(_in("label", ("CONFIRMATORY", "EXPLORATORY")), _prob("p_value"), _in("quote_quality", QUOTE_QUALITIES)),
    kind="persistent",
)

HYPOTHESIS_STATUSES = ("ACTIVE", "PAUSED", "DISABLED")

HYPOTHESIS_STATUS = Table(
    "hypothesis_status",
    "Status history of registered hypotheses (26.5: paused and disabled experiments stay visible). "
    "Append-only; the latest row per hypothesis is its current status; no row = ACTIVE since registration.",
    (
        Col("hypothesis_id", "VARCHAR", False),
        Col("status", "VARCHAR", False, " | ".join(HYPOTHESIS_STATUSES)),
        Col("reason", "VARCHAR"),
        Col("decision_ref", "VARCHAR", doc="decision record, e.g. docs/decisions/README.md"),
        Col("definition_hash", "VARCHAR", False, "the registered definition this status applies to"),
        Col("recorded_at", TS, False),
        Col("dataset", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(_in("status", HYPOTHESIS_STATUSES), "status = 'ACTIVE' OR reason IS NOT NULL"),
    kind="persistent",
)

# holdout.is_clean_commit in SQL: a commit that does not pin the code. Blank or padded is not clean;
# the trim set is every character Python's str.strip() removes, so the two agree on every string.
_WHITESPACE = "".join(chr(i) for i in range(sys.maxunicode + 1) if chr(i).isspace())
_NOT_CLEAN = ("(git_commit IS NULL OR git_commit = '' OR git_commit <> trim(git_commit, '" + _WHITESPACE + "') "
              "OR git_commit LIKE '%-dirty' OR git_commit LIKE 'unavailable%')")

REVIEW_FLAGS = Table(
    "review_flags",
    "ADR 0002 blocking flags. A `raise` row: a review found 26.16 FAIL (a negative control works) on a run "
    "without a clean commit. A `clear` row: a review of a later run from a clean commit found 26.16 PASS. "
    "A flag is open while its flag_id has no clear row; `quant freeze` refuses while any flag is open, and "
    "hypothesis registration runs once, for the clearing run. Append-only; nothing is deleted.",
    (
        Col("flag_id", "VARCHAR", False, "rf- + hash of (dataset, flagged backtest_run_id)"),
        Col("event", "VARCHAR", False, "raise | clear"),
        Col("criterion", "VARCHAR", False, "26.16"),
        Col("backtest_run_id", "VARCHAR", False, "raise: the flagged run; clear: the clean run that clears it"),
        Col("run_created_at", TS, False, "created_at of backtest_run_id"),
        Col("git_commit", "VARCHAR", doc="commit of backtest_run_id: not clean on a raise, clean on a clear"),
        Col("config_hash", "VARCHAR"),
        Col("detail", "VARCHAR", False),
        Col("recorded_at", TS, False),
        Col("referee_version", "VARCHAR", False),
        Col("dataset", "VARCHAR", False),
        Col("is_synthetic", "BOOLEAN", False),
    ),
    checks=(_in("event", ("raise", "clear")), _in("criterion", ("26.16",)),
            f"event <> 'raise' OR {_NOT_CLEAN}", f"event <> 'clear' OR NOT {_NOT_CLEAN}"),
    kind="persistent",
)

PERSISTENT_TABLES: tuple[Table, ...] = (PAPER_TRADES, PAPER_EXITS, PAPER_SETTLEMENTS, HYPOTHESES, EXPERIMENTS,
                                        HYPOTHESIS_STATUS, REVIEW_FLAGS)

ALL_TABLES: dict[str, Table] = {t.name: t for t in (*DERIVED_TABLES, *PERSISTENT_TABLES)}

_ARROW_TYPES = {
    "VARCHAR": pa.string(), "DOUBLE": pa.float64(), "BIGINT": pa.int64(), "INTEGER": pa.int32(),
    "BOOLEAN": pa.bool_(), TS: pa.timestamp("us", tz="UTC"), "DATE": pa.date32(),
}


def arrow_schema(table: Table) -> pa.Schema:
    """Arrow schema for a persistent (Parquet-backed) table."""
    return pa.schema([pa.field(c.name, _ARROW_TYPES[c.type], nullable=c.nullable) for c in table.columns])
