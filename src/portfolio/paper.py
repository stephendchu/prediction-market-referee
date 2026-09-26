"""Paper trading, scan and explain (spec Sec 4, 15, 21, 22). Research only.

Nothing here places, routes, or signs an order, and nothing here prices a trade: entry decisions
come from the backtest's rule (`engine.decide`), fills, exits and P&L from `backtest.execution`,
stakes from `portfolio.sizing` and `portfolio.exposure`.

The slate for a date is the canonical (T-30) observation of every contract still unresolved,
priced by the final-period ("holdout") model of the selected type: trained only on outcomes
known before the holdout period began, and refused unless its training data ends before every
slate snapshot.

A paper run, in order:
  1. exits   exit_at_offset positions are sold into their exit snapshot (paper_exits)
  2. settle  decisions whose outcome is known are settled (paper_settlements), passes included
  3. decide  the date's slate is decided and every decision is logged (paper_trades):
             a. the backtest decision at the flat stake and baseline penalty, one position per game
             b. fractional Kelly on p_conservative against the all-in price, capped per position,
                per game and per day (the book is seeded from earlier paper trades)
             c. the backtest decision again at the capped stake; a BUY commits its stake
Decisions are as of the snapshot (decision_ts = snapshot_time); `created_at` records when a row
was logged. A live row must be logged before the game starts (a paper_trades CHECK).

A trade carries `freeze_manifest_hash` only when it qualifies for 26.18: a FROZEN manifest whose
fields all match the current configuration except `data_version` (paper trading adds data by
design), and a model trained no later than the freeze. P&L of other trades is withheld.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..backtest.engine import Policy, decide, one_position_per_game
from ..backtest.execution import ExitFill, ExitRule, Fill, Position, Quote, side_won
from ..backtest.holdout import current_manifest, manifest_path, manifest_status, mismatches
from ..backtest.walk_forward import HoldoutLocked, latest_model_runs, load_periods, period_of
from ..config import stable_hash, strategy_family
from ..features.mlb import FeatureLeakError
from ..models.residual import FittedModel
from ..storage.database import Database
from ..storage.schemas import (BACKTEST_DECISIONS, CANDLE_DEGRADED, DEGRADED_EXECUTION, EXECUTABLE,
                               PAPER_EXITS, PAPER_SETTLEMENTS, PAPER_TRADES, QUOTE_QUALITIES)
from .exposure import ExposureBook
from .sizing import kelly_fraction as full_kelly

ET = ZoneInfo("America/New_York")
PAPER_SECTIONS = ("decision", "fees", "execution", "backtest", "paper")
WITHHELD = "P&L withheld: not made under a frozen configuration (26.3, 26.18)"

# Fill.record() key -> paper_trades column (the other keys share their name)
_FILL_TO_PAPER = {"execution_price": "entry_price", "quantity": "hypothetical_size"}
_FILL_KEYS = ("side", "quoted_price", "spread_penalty_frac", "execution_price", "quantity", "stake",
              "fee_schedule_id", "fee_schedule_effective_date", "maker_or_taker", "calculated_fee",
              "slippage_cost", "depth_verified")


class LookAheadModel(RuntimeError):
    """The model's training data does not end before the snapshot it would price."""


def today_et(now: datetime | None = None) -> date:
    """The MLB official date that is 'today' (US Eastern)."""
    return (now or datetime.now(timezone.utc)).astimezone(ET).date()


def paper_trade_id(dataset: str, observation_id: str) -> str:
    """One paper decision per snapshot: rerunning a date never logs a contract twice."""
    return "pt-" + stable_hash({"dataset": dataset, "observation_id": observation_id})


# ---------------------------------------------------------------------------
# The slate
# ---------------------------------------------------------------------------

_OBS_COLS = """o.observation_id, o.game_id, o.contract_id, o.official_date, o.home_team, o.away_team, o.yes_team,
       o.event_start_time, o.snapshot_offset_min, o.snapshot_time, o.quote_ts, o.quote_available_at, o.quote_age_s,
       o.market_yes_bid, o.market_yes_ask, o.bid_size, o.ask_size, o.depth_json, o.spread, o.p_mid,
       o.p_consensus, o.quote_quality, o.outcome, o.outcome_available_at"""

_SLATE_SQL = f"""
SELECT {_OBS_COLS} FROM observations o
WHERE o.official_date = ? AND o.snapshot_offset_min = ? AND o.quote_quality = ? AND o.outcome IS NULL
ORDER BY o.event_start_time, o.game_id, o.contract_id
"""

_TS_COLS = ("event_start_time", "snapshot_time", "quote_ts", "quote_available_at", "outcome_available_at")


@dataclass
class SlateModel:
    run: dict[str, Any]       # its model_runs row
    model: FittedModel


@dataclass
class Slate:
    day: date
    rows: pd.DataFrame        # observations + predictions: engine.decide input rows
    X: pd.DataFrame | None    # the model's features, same index (None: market model)
    model: SlateModel
    quote_quality: str
    notes: list[str] = field(default_factory=list)


def _selection_guard(cfg: dict[str, Any], selection: Mapping[str, Any]) -> str:
    strategy = strategy_family(cfg)                    # market_behavior is refused
    if selection.get("strategy_family", strategy) != strategy:
        raise ValueError(f"model selection was trained as {selection['strategy_family']!r}; config says "
                         f"{strategy!r}. Retrain with `quant train`.")
    quality = selection.get("quote_quality", EXECUTABLE)  # one quote class, never mixed
    if quality not in QUOTE_QUALITIES:
        raise ValueError(f"model selection has unknown quote_quality {quality!r}")
    return quality


def slate_model(db: Database, selection: Mapping[str, Any]) -> SlateModel:
    """The final-period model of the selected type, loaded from its artifact."""
    ids = list(selection.get("holdout_models") or [])
    runs = db.df("SELECT * FROM model_runs WHERE model_run_id IN (SELECT unnest(?::VARCHAR[])) AND model_type = ?",
                 [ids, selection["selected"]])
    if len(runs) != 1:
        raise RuntimeError(f"no final-period {selection['selected']} model to price the slate "
                           f"({len(runs)} found); run `quant train`")
    run = runs.iloc[0].to_dict()
    model = FittedModel.load(run["artifact_path"])
    if model.model_type != run["model_type"]:
        raise RuntimeError(f"{run['artifact_path']} holds a {model.model_type} model, not {run['model_type']}")
    run["train_data_through"] = pd.Timestamp(run["train_data_through"])
    run["created_at"] = pd.Timestamp(run["created_at"])
    return SlateModel(run, model)


def model_features(db: Database, obs: pd.DataFrame, model: FittedModel) -> pd.DataFrame | None:
    """The model's features for `obs`, as training assembles them (NaN = missing), point-in-time checked."""
    if model.model_type == "market":
        return None
    defined = set(db.df("SELECT feature_name FROM feature_definitions")["feature_name"])
    gone = sorted(set(model.features) - defined)
    if gone:
        raise RuntimeError(f"the model's features {gone} are no longer defined; run `quant train`")
    fv = db.df("SELECT observation_id, feature_name, value, available_at FROM feature_values "
               "WHERE observation_id IN (SELECT unnest(?::VARCHAR[])) AND feature_name IN (SELECT unnest(?::VARCHAR[]))",
               [obs["observation_id"].tolist(), list(model.features)])
    snap = obs.set_index("observation_id")["snapshot_time"]
    avail = pd.to_datetime(fv["available_at"], utc=True)
    late = avail.notna() & (avail > pd.to_datetime(fv["observation_id"].map(snap), utc=True))
    if late.any():
        bad = fv[late].iloc[0]
        raise FeatureLeakError(f"{bad['feature_name']} for {bad['observation_id']} available at "
                               f"{bad['available_at']}, after the snapshot")
    wide = fv.pivot(index="observation_id", columns="feature_name", values="value") if len(fv) else pd.DataFrame()
    X = wide.reindex(index=obs["observation_id"], columns=list(model.features)).astype(float)
    X.index = obs.index
    return X


def price_rows(obs: pd.DataFrame, X: pd.DataFrame | None, sm: SlateModel) -> pd.DataFrame:
    """engine.decide input rows: `obs` plus the model's prediction, refusing any look-ahead model."""
    through = sm.run["train_data_through"]
    if len(obs) and not through < obs["snapshot_time"].min():
        raise LookAheadModel(f"model {sm.run['model_run_id']} was trained on outcomes through {through}, not "
                             f"before the snapshot at {obs['snapshot_time'].min()}")
    pred = sm.model.predict(X, obs["p_mid"].to_numpy()) if len(obs) else pd.DataFrame()
    rows = obs.reset_index(drop=True).copy()
    for c in ("p_model_raw", "p_model", "lower_bound", "upper_bound", "p_conservative_yes", "p_conservative_no",
              "uncertainty_json"):
        rows[c] = pred[c].to_numpy() if len(pred) else []
    rows["p_market"] = rows["p_mid"]
    rows["model_run_id"] = sm.run["model_run_id"]
    rows["period_id"] = sm.run["period_id"]
    return rows


def _typed(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    for c in _TS_COLS:
        if c in obs:
            obs[c] = pd.to_datetime(obs[c], utc=True)
    obs["official_date"] = pd.to_datetime(obs["official_date"]).dt.date
    return obs


def load_slate(db: Database, cfg: dict[str, Any], selection: Mapping[str, Any], day: date, *,
               now: datetime | None = None) -> Slate:
    """The date's unresolved canonical observations, priced by the final-period model."""
    now = now or datetime.now(timezone.utc)
    quality = _selection_guard(cfg, selection)
    notes = [DEGRADED_EXECUTION] if quality == CANDLE_DEGRADED else []
    offset = int(cfg["snapshot_policy"]["canonical_offset_min"])
    obs = _typed(db.df(_SLATE_SQL, [day, offset, quality]))
    if not db.is_synthetic and len(obs):
        # live: a decision is logged before its game starts, from a snapshot already taken
        started = obs["event_start_time"] <= pd.Timestamp(now)
        ahead = obs["snapshot_time"] > pd.Timestamp(now)
        if started.any():
            notes.append(f"{int(started.sum())} contracts not decided: the game had started "
                         f"(run `quant features` then `quant paper` before first pitch)")
        obs = obs[~started & ~ahead]
    if db.is_synthetic:
        notes.append("SYNTHETIC replay: decisions are made as of each snapshot; created_at is the run time")
    sm = slate_model(db, selection)
    X = model_features(db, obs, sm.model)
    rows = price_rows(obs, X, sm)
    if X is not None:
        X = X.reset_index(drop=True)
    return Slate(day, rows, X, sm, quality, notes)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperPolicy:
    bankroll: float
    kelly_fraction: float
    max_position_frac: float
    max_game_frac: float
    max_daily_new_frac: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PaperPolicy:
        p = cfg["paper"]
        return cls(float(p["bankroll"]), float(p["kelly_fraction"]), float(p["max_position_frac"]),
                   float(p["max_game_frac"]), float(p["max_daily_new_frac"]))

    def book(self) -> ExposureBook:
        return ExposureBook(self.bankroll, self.max_game_frac, self.max_daily_new_frac)


def entry_policy(cfg: dict[str, Any]) -> Policy:
    """The backtest's decision rule at the baseline penalty. The entry is decided alone: an
    exit_at_offset sale is priced later, from the exit snapshot (paper_exits)."""
    pol = Policy.from_config(cfg)
    return replace(pol, scenarios=(pol.baseline,), exit_rule=ExitRule())


def flat_decisions(rows: pd.DataFrame, pol: Policy) -> pd.DataFrame:
    """The backtest decision for every row (flat stake), one position per game; same order as `rows`."""
    if rows.empty:
        return pd.DataFrame(columns=BACKTEST_DECISIONS.column_names)
    out = [decide(r, pol)[0] for r in rows.to_dict(orient="records")]
    return one_position_per_game(pd.DataFrame(out).reindex(columns=BACKTEST_DECISIONS.column_names), pol)


def _seed_book(book: ExposureBook, logged: pd.DataFrame) -> None:
    """Commit the stakes of earlier paper trades (same caps across runs of a day)."""
    buys = logged[logged["decision"] != "PASS"] if len(logged) else logged
    for r in buys.itertuples(index=False):
        day = pd.Timestamp(r.official_date).date()
        book.by_game[r.game_id] = book.by_game.get(r.game_id, 0.0) + float(r.stake)
        book.by_day[day] = book.by_day.get(day, 0.0) + float(r.stake)


def _size(flat: Mapping[str, Any], pp: PaperPolicy, book: ExposureBook, day: date) -> dict[str, Any]:
    """Fractional Kelly on the side's conservative probability against its all-in price, then the caps."""
    side = flat["side"]
    p = float(flat["p_conservative"])
    q = float(flat["execution_price"]) + float(flat["calculated_fee"]) / float(flat["quantity"])  # all-in
    uncapped = pp.kelly_fraction * full_kelly(p, q) * pp.bankroll
    requested = min(uncapped, pp.max_position_frac * pp.bankroll)
    capped_by = "position" if uncapped > requested + 1e-12 else None
    game_left, day_left = book.headroom(flat["game_id"], day)
    allowed = min(requested, game_left, day_left)
    if allowed < requested - 1e-12:
        capped_by = "game" if game_left <= day_left else "daily"
    return {"side": side, "requested_stake": requested, "allowed_stake": max(allowed, 0.0), "capped_by": capped_by}


def decide_slate(rows: pd.DataFrame, cfg: dict[str, Any], logged: pd.DataFrame) -> list[dict[str, Any]]:
    """(slate row, decision, sizing) for every row: the three-step paper decision (module docstring)."""
    pol, pp = entry_policy(cfg), PaperPolicy.from_config(cfg)
    flat = flat_decisions(rows, pol)
    book = pp.book()
    _seed_book(book, logged)
    held_games = set(logged.loc[logged["decision"] != "PASS", "game_id"]) if len(logged) else set()
    by_obs = {r["observation_id"]: r for r in rows.to_dict(orient="records")}
    ranked = flat.assign(_buy=flat["decision"] != "PASS").sort_values(
        ["_buy", "net_edge", "contract_id"], ascending=[False, False, True], na_position="last")
    out = []
    for f in ranked.drop(columns="_buy").to_dict(orient="records"):
        row = by_obs[f["observation_id"]]
        f = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in f.items()}
        sizing: dict[str, Any] = {"requested_stake": None, "allowed_stake": None, "capped_by": None}
        if f["decision"] == "PASS":
            out.append((row, f, sizing))
            continue
        if f["game_id"] in held_games:          # a position in this game was logged by an earlier run
            out.append((row, _as_pass(f, "same_game_position"), sizing))
            continue
        day = row["official_date"]
        sizing = _size(f, pp, book, day)
        if sizing["allowed_stake"] <= 0:
            out.append((row, _as_pass(f, "exposure_cap"), sizing))
            continue
        d = decide(row, replace(pol, stake=sizing["allowed_stake"]))[0]
        if d["decision"] != "PASS" and d["side"] == f["side"]:
            book.allocate(f["game_id"], day, float(d["stake"]))
            out.append((row, d, sizing))
        elif d["decision"] == "PASS" and d["pass_reason"] not in ("insufficient_depth", None):
            out.append((row, d, sizing))
        else:                                    # too small to fill at this size (or no longer this side)
            capped = sizing["capped_by"] in ("game", "daily")
            out.append((row, _as_pass(f, "exposure_cap" if capped else "kelly_size_below_minimum"), sizing))
    order = {oid: i for i, oid in enumerate(rows["observation_id"])}
    return sorted(out, key=lambda t: order[t[0]["observation_id"]])


def _as_pass(d: Mapping[str, Any], reason: str) -> dict[str, Any]:
    keep = ("model_run_id", "observation_id", "game_id", "contract_id", "period_id", "snapshot_time", "p_market",
            "p_model", "side", "p_conservative", "quoted_price", "predicted_edge", "net_edge", "spread_penalty_frac",
            "outcome", "side_won", "exit_rule", "scenario")
    return {k: d.get(k) for k in keep} | {"decision": "PASS", "pass_reason": reason}


def paper_record(row: Mapping[str, Any], d: Mapping[str, Any], sizing: Mapping[str, Any], *, db: Database,
                 model_type: str, train_data_through: Any, manifest_status: str, freeze_hash: str | None,
                 config_hash: str, exit_rule: ExitRule, created_at: datetime) -> dict[str, Any]:
    """One paper_trades row from a slate row and its decision."""
    buy = d["decision"] != "PASS"
    rec = {
        "paper_trade_id": paper_trade_id(db.dataset, row["observation_id"]), "created_at": created_at,
        "sport": db.sport, "dataset": db.dataset, "game_id": row["game_id"], "contract_id": row["contract_id"],
        "observation_id": row["observation_id"], "official_date": row["official_date"],
        "event_start_time": row["event_start_time"], "snapshot_offset_min": int(row["snapshot_offset_min"]),
        "quote_quality": row["quote_quality"], "period_id": row["period_id"], "model_run_id": row["model_run_id"],
        "model_type": model_type, "train_data_through": train_data_through, "manifest_status": manifest_status,
        "freeze_manifest_hash": freeze_hash, "config_hash": config_hash,
        "decision_ts": row["snapshot_time"], "quote_ts": row["quote_ts"], "quote_available_at": row["quote_available_at"],
        "bid": row["market_yes_bid"], "ask": row["market_yes_ask"], "bid_size": row["bid_size"],
        "ask_size": row["ask_size"], "depth_json": row["depth_json"],
        "p_market": row["p_market"], "model_probability": row["p_model"],
        "p_conservative_yes": row["p_conservative_yes"], "p_conservative_no": row["p_conservative_no"],
        "conservative_probability": d.get("p_conservative") if d.get("side") else None,
        "predicted_edge": d.get("predicted_edge"), "net_edge": d.get("net_edge"),
        "decision": d["decision"], "pass_reason": d.get("pass_reason"), "side": d.get("side"),
        "quoted_price": d.get("quoted_price"), "spread_penalty_frac": d["spread_penalty_frac"],
        "requested_stake": sizing["requested_stake"], "allowed_stake": sizing["allowed_stake"],
        "capped_by": sizing["capped_by"], "exit_rule": exit_rule.rule, "exit_offset_min": exit_rule.offset_min,
        "is_synthetic": db.is_synthetic,
    }
    fill = {k: d.get(k) if buy else None for k in _FILL_KEYS if k not in ("side", "quoted_price", "spread_penalty_frac")}
    rec |= {_FILL_TO_PAPER.get(k, k): v for k, v in fill.items()}
    rec["hypothetical_size"] = float(d["quantity"]) if buy else 0.0
    return rec


def entry_fill(trade: Mapping[str, Any]) -> Fill:
    """A logged BUY's fill, rebuilt and checked against its fee schedule."""
    rec = {k: trade[_FILL_TO_PAPER.get(k, k)] for k in _FILL_KEYS}
    return Fill.from_record(rec, when=pd.Timestamp(trade["decision_ts"]).to_pydatetime())


def trade_exit_rule(trade: Mapping[str, Any]) -> ExitRule:
    off = trade.get("exit_offset_min")
    return ExitRule(trade["exit_rule"], None if off is None or pd.isna(off) else int(off))


# ---------------------------------------------------------------------------
# Qualifying (26.18)
# ---------------------------------------------------------------------------

@dataclass
class Qualification:
    status: str                    # LOCKED | FROZEN | UNLOCKED
    manifest_hash: str | None      # set only when trades made now qualify
    reasons: list[str]             # why they don't


def qualification(db: Database, cfg: dict[str, Any], selection: Mapping[str, Any], sm: SlateModel,
                  now: datetime) -> Qualification:
    current = current_manifest(db, cfg, dict(selection), now)
    report = manifest_status(manifest_path(db.sport), current)
    frozen = report["frozen"]
    if not frozen or frozen.get("status") != "FROZEN":
        return Qualification(report["status"], None, ["no frozen manifest (26.3)"])
    bad = [m for m in mismatches(frozen, current) if m != "data_version"]  # paper trading adds data by design
    if frozen.get("dataset") != "live":        # holdout.freeze refuses others; a hand-written file must not pass
        bad.append(f"frozen manifest is for dataset {frozen.get('dataset')!r}; only live data qualifies (26.18)")
    if not bad and sm.run["created_at"] > pd.Timestamp(frozen["freeze_timestamp"]):
        bad = [f"model {sm.run['model_run_id']} was trained after the freeze"]
    return Qualification(report["status"], None if bad else frozen["manifest_hash"], bad)


# ---------------------------------------------------------------------------
# Exits and settlement
# ---------------------------------------------------------------------------

_EXIT_OBS_SQL = f"""
SELECT {_OBS_COLS} FROM observations o
WHERE o.contract_id IN (SELECT unnest(?::VARCHAR[])) AND o.snapshot_offset_min = ?
"""


def _logged(db: Database, table: str) -> pd.DataFrame:
    return db.df(f"SELECT * FROM {table}")


def run_exits(db: Database, cfg: dict[str, Any], *, now: datetime) -> pd.DataFrame:
    """Sell every open exit_at_offset position whose exit snapshot is in (paper_exits)."""
    trades, done = _logged(db, "paper_trades"), _logged(db, "paper_exits")
    todo = trades[(trades["decision"] != "PASS") & (trades["exit_rule"] == "exit_at_offset")
                  & ~trades["paper_trade_id"].isin(done["paper_trade_id"])] if len(trades) else trades
    if todo.empty:
        return pd.DataFrame(columns=PAPER_EXITS.column_names)
    pol = Policy.from_config(cfg)
    outcomes = _outcomes(db, todo["observation_id"].tolist())
    rows = []
    for t in todo.to_dict(orient="records"):
        rule = trade_exit_rule(t)
        obs = _typed(db.df(_EXIT_OBS_SQL, [[t["contract_id"]], rule.offset_min]))
        obs = obs[obs["quote_quality"] == t["quote_quality"]]
        x = obs.iloc[0].to_dict() if len(obs) else None
        if x is None and pd.isna(outcomes.get(t["observation_id"], {}).get("outcome")):
            continue                            # the exit snapshot may still arrive; decide when the game is over
        pos = Position(entry_fill(t), pd.Timestamp(t["decision_ts"]).to_pydatetime(), rule)
        if x is None:
            pos = pos.close(None, rule.exit_time(pd.Timestamp(t["event_start_time"]).to_pydatetime()),
                            penalty_frac=float(t["spread_penalty_frac"]), max_fraction_top=pol.max_fraction_top,
                            maker_or_taker=t["maker_or_taker"])
        else:
            pos = pos.close(Quote.from_row(x), x["snapshot_time"].to_pydatetime(), ref=x["observation_id"],
                            penalty_frac=float(t["spread_penalty_frac"]), max_fraction_top=pol.max_fraction_top,
                            maker_or_taker=t["maker_or_taker"])
        rec = {k: v for k, v in pos.record(None).items() if k not in ("gross_pnl", "net_pnl")}
        rec |= {"paper_trade_id": t["paper_trade_id"], "created_at": now, "side": t["side"],
                "spread_penalty_frac": float(t["spread_penalty_frac"]), "dataset": db.dataset,
                "is_synthetic": db.is_synthetic}
        rec |= dict.fromkeys(("exit_quote_ts", "exit_quote_available_at", "exit_bid", "exit_ask"))  # typed NULLs
        if x is not None:
            rec |= {"exit_quote_ts": x["quote_ts"], "exit_quote_available_at": x["quote_available_at"],
                    "exit_bid": x["market_yes_bid"], "exit_ask": x["market_yes_ask"]}
        rows.append(rec)
    df = pd.DataFrame(rows).reindex(columns=PAPER_EXITS.column_names)
    db.append_log("paper_exits", df, ts=now)
    return df


def _outcomes(db: Database, observation_ids: list[str]) -> dict[str, dict[str, Any]]:
    o = db.df("SELECT observation_id, outcome, outcome_available_at FROM observations "
              "WHERE observation_id IN (SELECT unnest(?::VARCHAR[]))", [observation_ids])
    o["outcome_available_at"] = pd.to_datetime(o["outcome_available_at"], utc=True)
    return {r["observation_id"]: r for r in o.to_dict(orient="records")}


def position_from_logs(trade: Mapping[str, Any], exit_row: Mapping[str, Any] | None) -> Position:
    """A logged BUY's position, rebuilt from paper_trades and paper_exits through the shared exit leg types."""
    rule = trade_exit_rule(trade)
    pos = Position(entry_fill(trade), pd.Timestamp(trade["decision_ts"]).to_pydatetime(), rule)
    if not rule.exits:
        return pos
    if exit_row is None:
        raise ValueError(f"{trade['paper_trade_id']}: exit_at_offset position has no paper_exits row")
    x = None
    if exit_row["exit_status"] != "held":
        x = ExitFill.from_record(exit_row, side=trade["side"], penalty_frac=float(exit_row["spread_penalty_frac"]),
                                 maker_or_taker=trade["maker_or_taker"],
                                 when=pd.Timestamp(exit_row["exit_snapshot_time"]).to_pydatetime())
    reason = exit_row.get("exit_reason")
    pos = replace(pos, exit=x, exit_time=pd.Timestamp(exit_row["exit_snapshot_time"]).to_pydatetime(),
                  exit_reason=None if reason is None or pd.isna(reason) else reason,
                  exit_ref=exit_row.get("exit_observation_id"))
    if abs(pos.held_quantity - float(exit_row["held_quantity"])) > 1e-9:
        raise ValueError(f"{trade['paper_trade_id']}: logged held_quantity {exit_row['held_quantity']} "
                         f"!= {pos.held_quantity}")
    return pos


def run_settlement(db: Database, *, now: datetime) -> pd.DataFrame:
    """Settle every logged decision whose outcome is known (passes: outcome only)."""
    trades, done = _logged(db, "paper_trades"), _logged(db, "paper_settlements")
    todo = trades[~trades["paper_trade_id"].isin(done["paper_trade_id"])] if len(trades) else trades
    if todo.empty:
        return pd.DataFrame(columns=PAPER_SETTLEMENTS.column_names)
    exits = _logged(db, "paper_exits")
    exit_by = {r["paper_trade_id"]: r for r in exits.to_dict(orient="records")}
    outcomes = _outcomes(db, todo["observation_id"].tolist())
    rows = []
    for t in todo.to_dict(orient="records"):
        o = outcomes.get(t["observation_id"])
        if o is None or pd.isna(o["outcome"]) or pd.isna(o["outcome_available_at"]) \
                or o["outcome_available_at"] > pd.Timestamp(now):
            continue
        outcome = int(o["outcome"])
        side = t["side"] if isinstance(t["side"], str) else None
        rec = {"paper_trade_id": t["paper_trade_id"], "settled_at": now, "outcome": outcome,
               "outcome_available_at": o["outcome_available_at"], "result": "yes" if outcome == 1 else "no",
               "decision": t["decision"], "side": side, "side_won": side_won(side, outcome) if side else None,
               "dataset": db.dataset, "is_synthetic": db.is_synthetic}
        if t["decision"] != "PASS":
            if t["exit_rule"] == "exit_at_offset" and t["paper_trade_id"] not in exit_by:
                continue                        # the exit leg is recorded first
            pos = position_from_logs(t, exit_by.get(t["paper_trade_id"]))
            pnl = pos.pnl(rec["side_won"])
            rec |= {"pnl": pnl["gross_pnl"], "net_pnl": pnl["net_pnl"],
                    "fees": pos.entry.fee + (pos.exit.fee if pos.exit else 0.0),
                    "held_quantity": pos.held_quantity, "exit_proceeds": pos.exit.proceeds if pos.exit else None}
        rows.append(rec)
    df = pd.DataFrame(rows).reindex(columns=PAPER_SETTLEMENTS.column_names)
    db.append_log("paper_settlements", df, ts=now)
    return df


# ---------------------------------------------------------------------------
# The paper run
# ---------------------------------------------------------------------------

@dataclass
class PaperResult:
    day: date
    decisions: pd.DataFrame          # rows logged by this run
    already_logged: int              # slate contracts decided by an earlier run
    exits: pd.DataFrame
    settlements: pd.DataFrame
    qualification: Qualification
    slate: Slate
    ledger: pd.DataFrame             # every logged decision with its settlement (for the summary)
    notes: list[str]


def config_hash(cfg: dict[str, Any]) -> str:
    return stable_hash({s: cfg.get(s) for s in PAPER_SECTIONS})


def run_paper(db: Database, cfg: dict[str, Any], selection: Mapping[str, Any], day: date, *,
              now: datetime | None = None) -> PaperResult:
    now = now or datetime.now(timezone.utc)
    exits = run_exits(db, cfg, now=now)
    settled = run_settlement(db, now=now)
    slate = load_slate(db, cfg, selection, day, now=now)
    qual = qualification(db, cfg, selection, slate.model, now)
    logged = _logged(db, "paper_trades")
    ids = slate.rows["observation_id"].map(lambda o: paper_trade_id(db.dataset, o))
    fresh = slate.rows[~ids.isin(set(logged["paper_trade_id"]))].reset_index(drop=True)
    rule = ExitRule.from_config(cfg)
    run = slate.model.run
    recs = [paper_record(row, d, sizing, db=db, model_type=run["model_type"],
                         train_data_through=run["train_data_through"], manifest_status=qual.status,
                         freeze_hash=qual.manifest_hash, config_hash=config_hash(cfg), exit_rule=rule,
                         created_at=now)
            for row, d, sizing in decide_slate(fresh, cfg, logged)]
    df = pd.DataFrame(recs).reindex(columns=PAPER_TRADES.column_names)
    db.append_log("paper_trades", df, ts=now)
    notes = list(slate.notes)
    if not qual.manifest_hash:
        notes.append(f"{WITHHELD}: {'; '.join(qual.reasons)}")
    return PaperResult(day, df, int(len(slate.rows) - len(fresh)), exits, settled, qual, slate, ledger(db), notes)


def ledger(db: Database) -> pd.DataFrame:
    """Every logged decision joined to its settlement (latest), for summaries."""
    t, s = _logged(db, "paper_trades"), _logged(db, "paper_settlements")
    if t.empty:
        return t
    s = s.drop_duplicates("paper_trade_id", keep="last")
    cols = ["paper_trade_id", "settled_at", "outcome", "result", "side_won", "pnl", "fees", "net_pnl"]
    return t.merge(s[cols], on="paper_trade_id", how="left")


# ---------------------------------------------------------------------------
# Scan (Sec 4, 22) and explain (Sec 4)
# ---------------------------------------------------------------------------

def scan(db: Database, cfg: dict[str, Any], selection: Mapping[str, Any], day: date, *,
         now: datetime | None = None) -> tuple[pd.DataFrame, Slate]:
    """The date's slate under the backtest decision rule (flat stake), ranked by |net edge|. Logs nothing."""
    slate = load_slate(db, cfg, selection, day, now=now)
    flat = flat_decisions(slate.rows, entry_policy(cfg))
    extra = slate.rows[["observation_id", "home_team", "away_team", "yes_team", "event_start_time", "p_consensus",
                        "market_yes_bid", "market_yes_ask", "lower_bound", "upper_bound"]]
    out = flat.drop(columns=["event_start_time"], errors="ignore").merge(extra, on="observation_id", how="left")
    out["abs_net_edge"] = out["net_edge"].abs()
    out = out.sort_values(["abs_net_edge", "contract_id"], ascending=[False, True], na_position="last")
    return out.reset_index(drop=True), slate


@dataclass
class Explanation:
    contract_id: str
    row: dict[str, Any]              # observation + prediction
    decision: dict[str, Any]         # engine.decide at the baseline penalty, flat stake
    contributions: pd.Series         # logit contributions of the model, largest |value| first
    source: str                      # where the prediction came from
    notes: list[str]


_EXPLAIN_SQL = f"""
SELECT {_OBS_COLS} FROM observations o
WHERE o.contract_id = ? AND o.snapshot_offset_min = ? AND o.quote_quality = ?
"""


def explain(db: Database, cfg: dict[str, Any], selection: Mapping[str, Any], contract_id: str, *,
            now: datetime | None = None) -> Explanation:
    """Why the model priced one contract as it did, at its canonical snapshot (Sec 4 `quant explain`)."""
    now = now or datetime.now(timezone.utc)
    quality = _selection_guard(cfg, selection)
    offset = int(cfg["snapshot_policy"]["canonical_offset_min"])
    obs = _typed(db.df(_EXPLAIN_SQL, [contract_id, offset, quality]))
    if obs.empty:
        raise LookupError(f"no {quality} T-{offset} observation for {contract_id}; run `quant features`")
    notes = [DEGRADED_EXECUTION] if quality == CANDLE_DEGRADED else []
    periods = load_periods(cfg, db.dataset)
    period = period_of(obs["official_date"].iloc[0], periods)
    settled = obs["outcome"].notna().iloc[0]
    if settled and period is not None and period.is_holdout:
        raise HoldoutLocked(f"{contract_id} is a settled {period.period_id} contract: the final holdout is not "
                            f"evaluated before a freeze (26.3)")
    if not settled:
        sm = slate_model(db, selection)
        source = f"final-period model {sm.run['model_run_id']} (unsettled contract)"
    else:
        if period is None or period.role != "evaluation":
            raise LookupError(f"{contract_id} is settled outside the evaluation periods; no prediction to explain")
        run = latest_model_runs(db)                 # the latest `quant train` only (ADR 0002 item 4)
        run = run[(run["model_type"] == selection["selected"]) & (run["period_id"] == period.period_id)]
        if run.empty:
            raise LookupError(f"no {selection['selected']} model for {period.period_id}; run `quant train`")
        r = run.iloc[0].to_dict()
        r["train_data_through"], r["created_at"] = pd.Timestamp(r["train_data_through"]), pd.Timestamp(r["created_at"])
        sm = SlateModel(r, FittedModel.load(r["artifact_path"]))
        source = f"{period.period_id} walk-forward model {r['model_run_id']} (out-of-sample)"
    X = model_features(db, obs, sm.model)
    rows = price_rows(obs, X, sm)
    row = rows.iloc[0].to_dict()
    d = flat_decisions(rows, entry_policy(cfg)).iloc[0].to_dict()
    contrib = sm.model.contributions(X, rows["p_market"].to_numpy()).iloc[0]
    contrib = contrib.drop(labels=[c for c in ("market", "intercept") if c in contrib.index])
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    return Explanation(contract_id, row, d, contrib, source, notes)


def uncertainty_components(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("uncertainty_json")
    return json.loads(raw) if isinstance(raw, str) and raw else {}
