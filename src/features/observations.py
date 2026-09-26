"""Sec 5 observations: one per contract per pre-game snapshot offset (26.10, 26.11).

For every moneyline contract and every configured offset (T-180, -120, -60,
-30, -10), the snapshot's quote is the latest quote whose `available_at` is at
or before `snapshot_time = scheduled start - offset`. It becomes an
observation only if it passes the 26.10 rules, checked in this order (the
first failure is the recorded exclusion reason):

  no_quote     nothing available at the snapshot
  stale        quote older than max_quote_age_s (measured from quote_ts)
  one_sided    a side is missing (an empty side is stored as bid 0 / ask 1)
  crossed      bid >= ask
  no_size      zero or missing displayed size (not checkable on historical candles,
               which carry no size: those pass with size_verified = false)
  wide_spread  ask - bid > max_spread

Market probability is the bid/ask midpoint, never the last trade (Sec 6).
The sportsbook consensus is the mean over books of each book's latest de-vigged
price available at or before the QUOTE time and at most consensus_max_age_min
old, so the gap between them compares two prices that were both knowable at
the same instant (26.11).

Liquidity terciles use cut points from observations on earlier dates only
(same offset), so a tercile never depends on the day it labels. The news count
covers events that became public in (start - 180 min, min(start - 30 min, quote_ts)].
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..storage.database import Database
from ..storage.schemas import CANDLE_DEGRADED, CANONICAL_OFFSET_MIN, EXCLUSION_REASONS, EXECUTABLE

NEWS_TYPES = ("lineup", "starter", "injury", "transaction", "status", "postponed")

# The canonical snapshot rules (26.10) may be tightened in config, never relaxed.
_LIMITS = {"max_quote_age_s": 300, "max_spread": 0.10}

# One quote per (contract, available_at): the order book beats the listing beats a candle, then the
# latest quote_ts, then a fixed host order (two recording hosts can observe the same
# quote at the same instant; NULL host = candle or synthetic rows, last).
_SOURCE_PRIORITY = "CASE quote_source WHEN 'orderbook' THEN 0 WHEN 'synthetic' THEN 0 WHEN 'listing' THEN 1 ELSE 2 END"


@dataclass(frozen=True)
class SnapshotPolicy:
    offsets_min: tuple[int, ...]
    max_quote_age_s: float
    max_spread: float
    consensus_max_age_min: float
    liquidity_min_prior_obs: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SnapshotPolicy":
        sp = cfg["snapshot_policy"]
        if int(sp["canonical_offset_min"]) != CANONICAL_OFFSET_MIN:
            raise ValueError(f"the canonical snapshot is T-{CANONICAL_OFFSET_MIN} (26.10); config says "
                             f"T-{sp['canonical_offset_min']}")
        for key, limit in _LIMITS.items():
            if float(sp[key]) > limit:
                raise ValueError(f"snapshot_policy.{key} = {sp[key]} relaxes the 26.10 rule ({limit})")
        if not sp.get("require_displayed_size", True):
            raise ValueError("snapshot_policy.require_displayed_size cannot be turned off (26.10)")
        offsets = sorted({CANONICAL_OFFSET_MIN, *(int(x) for x in sp["sensitivity_offsets_min"])}, reverse=True)
        return cls(tuple(offsets), float(sp["max_quote_age_s"]), float(sp["max_spread"]),
                   float(sp["consensus_max_age_min"]), int(sp["liquidity_min_prior_obs"]))


@dataclass
class BuildResult:
    run_id: str
    n_observations: int
    exclusions: dict[str, int]
    n_canonical: int


def build_observations(db: Database, cfg: dict[str, Any], through: date | None = None) -> BuildResult:
    """Rebuild `observations` and `snapshot_exclusions` for games on or before `through`."""
    policy = SnapshotPolicy.from_config(cfg)
    run_id = f"obs-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    snaps = _snapshots(db, policy, through or date.max)
    reason = classify(snaps, policy)
    valid = snaps[reason.isna()].copy()
    obs = _observations(db, valid, policy, run_id)
    excl = snaps[reason.notna()].assign(reason=reason[reason.notna()])
    excl = pd.DataFrame({
        "game_id": excl["game_id"], "contract_id": excl["contract_id"], "official_date": excl["official_date"],
        "snapshot_offset_min": excl["snapshot_offset_min"], "snapshot_time": excl["snapshot_time"],
        "reason": excl["reason"], "detail": _detail(excl), "build_run_id": run_id, "is_synthetic": db.is_synthetic})
    db.replace_tables({"observations": obs, "snapshot_exclusions": excl})
    counts = {r: int((excl["reason"] == r).sum()) for r in EXCLUSION_REASONS}
    return BuildResult(run_id, len(obs), counts, int(obs["is_canonical"].sum()) if len(obs) else 0)


def _snapshots(db: Database, policy: SnapshotPolicy, through: date) -> pd.DataFrame:
    offsets = ", ".join(f"({int(o)})" for o in policy.offsets_min)
    return db.df(f"""
        WITH snaps AS (
            SELECT c.contract_id, c.game_id, c.yes_team, c.yes_side, c.result, c.settled_at,
                   g.season, g.official_date, g.home_team, g.away_team, g.outcome_available_at,
                   g.scheduled_start AS event_start_time, o.off AS snapshot_offset_min,
                   g.scheduled_start - to_minutes(o.off) AS snapshot_time
            FROM contracts c JOIN games g USING (game_id) CROSS JOIN (VALUES {offsets}) o(off)
            WHERE c.market_type = 'ml' AND g.official_date <= ?
        ),
        q AS (
            SELECT * FROM market_quotes
            QUALIFY row_number() OVER (PARTITION BY contract_id, available_at
                                       ORDER BY {_SOURCE_PRIORITY}, quote_ts DESC, host NULLS LAST) = 1
        )
        SELECT s.*, q.quote_ts, q.available_at AS quote_available_at, q.quote_source,
               q.yes_bid, q.yes_ask, q.no_bid, q.no_ask, q.last_price, q.bid_size, q.ask_size,
               q.depth_json, q.volume, q.open_interest
        FROM snaps s ASOF LEFT JOIN q
          ON s.contract_id = q.contract_id AND q.available_at <= s.snapshot_time
        ORDER BY s.official_date, s.contract_id, s.snapshot_offset_min DESC
    """, [through])


def classify(snaps: pd.DataFrame, policy: SnapshotPolicy) -> pd.Series:
    """First failing 26.10 rule per snapshot, or NaN if the quote is a valid observation."""
    age = (snaps["snapshot_time"] - snaps["quote_ts"]).dt.total_seconds()
    bid, ask = snaps["yes_bid"], snaps["yes_ask"]
    candle = snaps["quote_source"] == "candle"
    size_ok = (snaps["bid_size"] > 0) & (snaps["ask_size"] > 0)
    rules = [
        ("no_quote", snaps["quote_ts"].isna()),
        ("stale", age > policy.max_quote_age_s),
        ("one_sided", bid.isna() | ask.isna() | (bid <= 0) | (ask >= 1)),
        ("crossed", bid >= ask),
        ("no_size", ~candle & ~size_ok),
        ("wide_spread", (ask - bid) > policy.max_spread + 1e-9),
    ]
    reason = pd.Series(np.nan, index=snaps.index, dtype=object)
    for name, failed in rules:
        reason = reason.where(reason.notna() | ~failed.fillna(False), name)
    return reason


def _detail(excl: pd.DataFrame) -> pd.Series:
    age = (excl["snapshot_time"] - excl["quote_ts"]).dt.total_seconds().round()
    text = ("age_s=" + age.astype("Int64").astype(str) + " bid=" + excl["yes_bid"].round(2).astype(str)
            + " ask=" + excl["yes_ask"].round(2).astype(str) + " source=" + excl["quote_source"].astype(str))
    return text.where(excl["reason"] != "no_quote", None)


def _observations(db: Database, v: pd.DataFrame, policy: SnapshotPolicy, run_id: str) -> pd.DataFrame:
    v = v.reset_index(drop=True)
    v["observation_id"] = v["contract_id"] + "@T-" + v["snapshot_offset_min"].astype(str)
    v["p_mid"] = (v["yes_bid"] + v["yes_ask"]) / 2
    v = v.merge(_consensus(db, v, policy), on="observation_id", how="left")
    v = v.merge(_news_counts(db, v), on="observation_id", how="left")
    v["liquidity_tercile"] = liquidity_terciles(v, policy.liquidity_min_prior_obs)
    outcome = v["result"].map({"yes": 1, "no": 0})
    label_at = v["outcome_available_at"].fillna(v["settled_at"])
    return pd.DataFrame({
        "observation_id": v["observation_id"], "game_id": v["game_id"], "contract_id": v["contract_id"],
        "sport": db.sport, "season": v["season"], "official_date": v["official_date"],
        "home_team": v["home_team"], "away_team": v["away_team"], "yes_team": v["yes_team"],
        "yes_side": v["yes_side"], "event_start_time": v["event_start_time"],
        "snapshot_offset_min": v["snapshot_offset_min"], "snapshot_time": v["snapshot_time"],
        "is_canonical": v["snapshot_offset_min"] == CANONICAL_OFFSET_MIN,
        "quote_ts": v["quote_ts"], "quote_available_at": v["quote_available_at"],
        "quote_age_s": (v["snapshot_time"] - v["quote_ts"]).dt.total_seconds(),
        "quote_source": v["quote_source"], "market_yes_bid": v["yes_bid"], "market_yes_ask": v["yes_ask"],
        "market_no_bid": 1 - v["yes_ask"], "market_no_ask": 1 - v["yes_bid"], "market_last": v["last_price"],
        "spread": v["yes_ask"] - v["yes_bid"], "bid_size": v["bid_size"], "ask_size": v["ask_size"],
        "depth_json": v["depth_json"],
        "size_verified": (v["quote_source"] != "candle") & (v["bid_size"] > 0) & (v["ask_size"] > 0),
        "quote_quality": np.where(v["quote_source"] == "candle", CANDLE_DEGRADED, EXECUTABLE),
        "volume": v["volume"], "open_interest": v["open_interest"], "p_mid": v["p_mid"],
        "p_consensus": v["p_consensus"], "consensus_books": v["consensus_books"].astype("Int64"),
        "consensus_available_at": v["consensus_available_at"],
        "cross_market_gap": v["p_consensus"] - v["p_mid"],
        "liquidity_tercile": v["liquidity_tercile"],
        "news_event_count_t180_t30": v["news_event_count_t180_t30"].fillna(0).astype(int),
        "outcome": outcome.astype("Int64"), "outcome_available_at": label_at.where(outcome.notna()),
        "build_run_id": run_id, "is_synthetic": db.is_synthetic})


def _consensus(db: Database, v: pd.DataFrame, policy: SnapshotPolicy) -> pd.DataFrame:
    """Per observation: mean YES-oriented de-vigged book probability known at the quote time."""
    keys = v[["observation_id", "game_id", "yes_side", "quote_ts"]]
    db.con.register("_obs_keys", keys)
    try:
        per_book = db.df("""
            WITH odds AS (SELECT * FROM consensus_odds WHERE market_type = 'ml' AND p_home_devig IS NOT NULL),
                 pairs AS (SELECT k.*, b.book FROM _obs_keys k
                           JOIN (SELECT DISTINCT game_id, book FROM odds) b USING (game_id))
            SELECT p.observation_id, p.yes_side, p.quote_ts, o.odds_ts, o.available_at, o.p_home_devig
            FROM pairs p ASOF JOIN odds o
              ON p.game_id = o.game_id AND p.book = o.book AND o.available_at <= p.quote_ts
        """)
    finally:
        db.con.unregister("_obs_keys")
    cols = ["observation_id", "p_consensus", "consensus_books", "consensus_available_at"]
    if per_book.empty:
        return pd.DataFrame(columns=cols)
    fresh = per_book[(per_book["quote_ts"] - per_book["odds_ts"]).dt.total_seconds()
                     <= policy.consensus_max_age_min * 60]
    fresh = fresh.assign(p_yes=np.where(fresh["yes_side"] == "home", fresh["p_home_devig"], 1 - fresh["p_home_devig"]))
    out = fresh.groupby("observation_id").agg(p_consensus=("p_yes", "mean"), consensus_books=("p_yes", "size"),
                                              consensus_available_at=("available_at", "max")).reset_index()
    return out[cols]


def _news_counts(db: Database, v: pd.DataFrame) -> pd.DataFrame:
    keys = v[["observation_id", "game_id", "event_start_time", "quote_ts"]]
    db.con.register("_obs_keys", keys)
    types = ", ".join(f"'{t}'" for t in NEWS_TYPES)
    try:
        return db.df(f"""
            SELECT k.observation_id, count(e.event_id) AS news_event_count_t180_t30
            FROM _obs_keys k LEFT JOIN info_events e
              ON e.game_id = k.game_id AND e.type IN ({types}) AND NOT coalesce(e.first_seen, false)
             AND e.available_at > k.event_start_time - INTERVAL 180 MINUTE
             AND e.available_at <= least(k.event_start_time - INTERVAL 30 MINUTE, k.quote_ts)
            GROUP BY 1
        """)
    finally:
        db.con.unregister("_obs_keys")


def liquidity_measure(v: pd.DataFrame) -> pd.Series:
    """Open interest at the quote (volume where open interest is missing)."""
    return v["open_interest"].fillna(v["volume"])


def liquidity_terciles(v: pd.DataFrame, min_prior: int) -> pd.Series:
    """1 = least liquid. Cut points per offset from strictly earlier official dates only."""
    out = pd.Series(pd.NA, index=v.index, dtype="Int64")
    measure = liquidity_measure(v)
    for _, grp in v.groupby("snapshot_offset_min"):
        dates = grp["official_date"]
        seen = np.array([], dtype=float)
        for d in sorted(dates.unique()):
            idx = grp.index[dates == d]
            if len(seen) >= min_prior:
                lo, hi = np.quantile(seen, [1 / 3, 2 / 3])
                vals = measure.loc[idx]
                t = np.where(vals <= lo, 1, np.where(vals <= hi, 2, 3))
                out.loc[idx] = pd.array(np.where(vals.isna(), pd.NA, t), dtype="Int64")
            new = measure.loc[idx].dropna().to_numpy(dtype=float)
            seen = np.concatenate([seen, new])
    return out
