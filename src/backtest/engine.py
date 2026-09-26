"""Walk-forward backtest (spec Sec 14, 16, 26.6-26.10).

Replays the selected model's out-of-sample predictions (model_runs /
predictions written by `quant train`) against the executable quotes stored on
each observation. Nothing here places an order.

Per observation, at the baseline spread penalty (execution.baseline_penalty):
  1. PASS no_price     the quote is not two-sided (bid < ask inside (0, 1))
  2. PASS wide_spread  spread > decision.maximum_spread
  3. for YES and NO: screening net edge = p_conservative(side) - (ask(side) +
     penalty x spread) - fee per contract (versioned FeeSchedule in force at the
     snapshot time); the side with the larger net edge is the side considered
     (YES-oriented p_market / p_model are stored as is)
  4. PASS below_min_edge  best screening net edge < decision.minimum_edge
  5. simulate the fill (26.9 fill model); NoFill -> PASS with its reason; the
     realised net edge (VWAP / size cap / fee on the filled quantity) must still
     clear minimum_edge, else PASS below_min_edge
  6. BUY_YES / BUY_NO
  7. one position per game: a game's two contracts mirror each other (YES home =
     NO away), so of its trades only the one with the larger baseline net edge is
     kept; the other becomes PASS same_game_position
The decision is taken once, at the baseline; every scenario in
execution.spread_penalty_scenarios re-prices the same trades (a trade that
cannot be re-priced in a scenario becomes a PASS there with the fill's reason).
Thresholds come from config and are never fitted.

Exit leg, execution.exit_rule:
  hold_to_settlement  every trade is held to settlement (the default)
  exit_at_offset      every trade is sold at the same contract's T-exit_offset_min
                      observation (same quote class), at that scenario's penalty;
                      contracts that cannot be sold are held, with exit_reason
The entry decision never sees the exit quote: steps 1-7 are unchanged, and the
entry screen is the hold-to-settlement edge (exit costs are not anticipated).

Runs: one at the canonical offset (T-30) and one at each sensitivity offset
(26.10), each using the same period models. Only evaluation-period
predictions are read; a holdout row raises HoldoutLocked.

H_MLB_003's primary metric - realised edge per contract on trades whose
|cross_market_gap| is inside the fee band vs outside it - is recorded here as
an EXPLORATORY experiment, per evaluation period and pooled, with a two-sided
game-bootstrap p-value (brier fields are null: it is not a Brier comparison).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..config import stable_hash, strategy_family
from ..storage.database import Database
from ..storage.schemas import BACKTEST_DECISIONS, CANDLE_DEGRADED, DEGRADED_EXECUTION, EXECUTABLE, QUOTE_QUALITIES
from . import holdout
from . import metrics as M
from .execution import (
    ExitRule,
    Fill,
    Position,
    Quote,
    estimated_net_edge,
    fee_schedule_rows,
    net_edge,
    scenario_name,
    side_won,
    simulate_fill,
)
from .registry import ACTIVE, current_statuses, record_experiments
from .walk_forward import assert_not_holdout, evaluation_periods, latest_model_runs, load_periods

H3 = "H_MLB_003"
H3_FEATURE = "gap_inside_fee_band"
THRESHOLD_SECTIONS = ("decision", "fees", "execution", "backtest")

_ROWS_SQL = """
SELECT p.model_run_id, p.observation_id, p.period_id, p.p_market, p.p_model, p.p_conservative_yes,
       p.p_conservative_no, o.game_id, o.contract_id, o.snapshot_time, o.snapshot_offset_min, o.event_start_time,
       o.market_yes_bid, o.market_yes_ask, o.bid_size, o.ask_size, o.depth_json, o.spread, o.outcome
FROM predictions p JOIN observations o USING (observation_id)
WHERE p.model_run_id IN (SELECT unnest(?::VARCHAR[])) AND o.snapshot_offset_min = ? AND o.quote_quality = ?
ORDER BY o.snapshot_time, o.game_id, o.contract_id
"""

# the quote an exit_at_offset position is sold into: same contract, exit offset, same quote class
_EXIT_SQL = """
SELECT game_id, contract_id, observation_id, snapshot_time, market_yes_bid, market_yes_ask, bid_size, ask_size,
       depth_json
FROM observations WHERE snapshot_offset_min = ? AND quote_quality = ?
"""


@dataclass(frozen=True)
class Policy:
    minimum_edge: float
    maximum_spread: float
    min_contracts: int
    stake: float
    max_fraction_top: float
    maker_or_taker: str
    baseline: float
    scenarios: tuple[float, ...]
    exit_rule: ExitRule = ExitRule()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Policy:
        d, e = cfg["decision"], cfg["execution"]
        scen = tuple(sorted({float(x) for x in e["spread_penalty_scenarios"]} | {float(e["baseline_penalty"])}))
        return cls(float(d["minimum_edge"]), float(d["maximum_spread"]), int(d["minimum_fill_contracts"]),
                   float(cfg["backtest"]["stake_dollars"]), float(e["max_fraction_of_top_of_book"]),
                   str((cfg.get("fees") or {}).get("maker_or_taker", "taker")), float(e["baseline_penalty"]), scen,
                   ExitRule.from_config(cfg))


def _fill(side: str, q: Quote, k: float, pol: Policy, when: Any) -> Any:
    return simulate_fill(side, q, penalty_frac=k, stake_dollars=pol.stake, max_fraction_top=pol.max_fraction_top,
                         min_contracts=pol.min_contracts, when=when, maker_or_taker=pol.maker_or_taker)


def _close(pos: Position, row: dict[str, Any], exit_row: dict[str, Any] | None, k: float, pol: Policy) -> Position:
    if exit_row is None:
        return pos.close(None, pol.exit_rule.exit_time(row["event_start_time"]), penalty_frac=k,
                         max_fraction_top=pol.max_fraction_top, maker_or_taker=pol.maker_or_taker)
    return pos.close(Quote.from_row(exit_row), exit_row["snapshot_time"], ref=exit_row["observation_id"],
                     penalty_frac=k, max_fraction_top=pol.max_fraction_top, maker_or_taker=pol.maker_or_taker)


def decide(row: dict[str, Any], pol: Policy, exit_row: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """backtest_decisions rows (one per scenario) for one prediction row.

    `exit_row` is the observation an exit_at_offset position is sold into (None: none valid). It is read
    only after the entry is decided.
    """
    q = Quote.from_row(row)
    when = row["snapshot_time"]
    p_model = float(row["p_model"])
    cons = {"yes": float(row["p_conservative_yes"]), "no": float(row["p_conservative_no"])}
    base = {"model_run_id": row["model_run_id"], "observation_id": row["observation_id"], "game_id": row["game_id"],
            "contract_id": row["contract_id"], "period_id": row["period_id"], "snapshot_time": when,
            "p_market": float(row["p_market"]), "p_model": p_model}
    outcome = None if pd.isna(row["outcome"]) else int(row["outcome"])

    side, reason, screen = None, None, {}
    if not q.is_valid:
        reason = "no_price"
    elif q.yes_ask - q.yes_bid > pol.maximum_spread + 1e-12:
        reason = "wide_spread"
    else:
        screen = {s: estimated_net_edge(cons[s], q, s, pol.baseline, pol.stake, when, pol.maker_or_taker)
                  for s in ("yes", "no")}
        priced = {s: v for s, v in screen.items() if v is not None}
        if not priced:
            reason = "no_price"
        else:
            side = max(priced, key=lambda s: (priced[s], s == "yes"))
            if priced[side] < pol.minimum_edge:
                reason = "below_min_edge"
    fill0 = None
    if reason is None:
        fill0 = _fill(side, q, pol.baseline, pol, when)
        if not isinstance(fill0, Fill):
            reason = fill0.reason
        elif net_edge(cons[side], fill0.execution_price, fill0.fee, fill0.quantity) < pol.minimum_edge:
            reason = "below_min_edge"

    common = dict(base, side=side, outcome=outcome)
    if side is not None:
        quoted = q.yes_ask if side == "yes" else 1.0 - q.yes_bid
        common |= {"p_conservative": cons[side], "quoted_price": quoted,
                   "predicted_edge": (p_model if side == "yes" else 1.0 - p_model) - quoted,
                   "side_won": None if outcome is None else side_won(side, outcome)}
    rows = []
    for k in pol.scenarios:
        r = dict(common, scenario=scenario_name(k), spread_penalty_frac=k, exit_rule=pol.exit_rule.rule)
        f = None
        if reason is None:
            f = fill0 if k == pol.baseline else _fill(side, q, k, pol, when)
        if isinstance(f, Fill):
            r |= f.record()
            r |= {"decision": "BUY_YES" if side == "yes" else "BUY_NO", "pass_reason": None,
                  "net_edge": net_edge(cons[side], f.execution_price, f.fee, f.quantity)}
            pos = Position(f, when, pol.exit_rule)
            if pol.exit_rule.exits:
                pos = _close(pos, row, exit_row, k, pol)
            r |= pos.record(r.get("side_won"))
        else:
            r |= {"decision": "PASS", "pass_reason": reason or f.reason}
            if side is not None and screen.get(side) is not None:
                r["net_edge"] = screen[side] if k == pol.baseline else estimated_net_edge(
                    cons[side], q, side, k, pol.stake, when, pol.maker_or_taker)
        rows.append(r)
    return rows


FILL_COLUMNS = ("execution_price", "quantity", "stake", "fee_schedule_id", "fee_schedule_effective_date",
                "maker_or_taker", "calculated_fee", "slippage_cost", "depth_verified", "gross_pnl", "net_pnl",
                "exit_status", "exit_reason", "exit_observation_id", "exit_snapshot_time", "exit_quoted_price",
                "exit_price", "exit_quantity", "held_quantity", "exit_proceeds", "exit_fee_schedule_id",
                "exit_fee_schedule_effective_date", "exit_fee", "exit_slippage_cost", "exit_depth_verified")


def one_position_per_game(df: pd.DataFrame, pol: Policy) -> pd.DataFrame:
    """Keep one trade per game (the larger baseline net edge; ties -> first contract_id)."""
    base = df[(df["scenario"] == scenario_name(pol.baseline)) & (df["decision"] != "PASS")]
    ranked = base.sort_values(["game_id", "net_edge", "contract_id"], ascending=[True, False, True])
    drop = set(ranked[ranked.duplicated("game_id")]["observation_id"])
    if not drop:
        return df
    m = df["observation_id"].isin(drop)
    df = df.copy()
    df.loc[m, "decision"], df.loc[m, "pass_reason"] = "PASS", "same_game_position"
    for c in FILL_COLUMNS:
        col = df[c].astype("boolean") if df[c].dtype == bool else df[c]
        df[c] = col.astype(object).where(~m, None) if col.dtype == object else col.mask(m)
    return df


@dataclass
class BacktestResult:
    runs: pd.DataFrame
    decisions: int
    summary: dict[int, dict[str, Any]]
    experiments_logged: int
    notes: list[str]


def run_backtest(db: Database, cfg: dict[str, Any], selection: dict[str, Any], *,
                 now: datetime | None = None) -> BacktestResult:
    strategy = strategy_family(cfg)                          # market_behavior (26.20) is refused
    if selection.get("strategy_family", strategy) != strategy:
        raise ValueError(f"model selection was trained as {selection['strategy_family']!r}; config says "
                         f"{strategy!r}. Retrain with `quant train`.")
    quality = selection.get("quote_quality", EXECUTABLE)    # one quote class per run, never mixed
    if quality not in QUOTE_QUALITIES:
        raise ValueError(f"model selection has unknown quote_quality {quality!r}")
    now = now or datetime.now(timezone.utc)
    commit = holdout.git_commit()                           # spec Sec 20; None without git
    periods = load_periods(cfg, db.dataset)
    eval_ids = [p.period_id for p in evaluation_periods(periods)]
    model_type = selection["selected"]
    runs = latest_model_runs(db)                            # the latest `quant train` only (ADR 0002 item 4)
    runs = runs[(runs["model_type"] == model_type) & runs["period_id"].isin(eval_ids)]
    if runs.empty:
        raise RuntimeError(f"no {model_type} model runs for the evaluation periods; run `quant train` first")
    notes = [f"{p}: no {model_type} model; not backtested" for p in eval_ids if p not in set(runs["period_id"])]
    if quality == CANDLE_DEGRADED:
        notes.insert(0, DEGRADED_EXECUTION)
    sp = cfg["snapshot_policy"]
    offsets = [int(sp["canonical_offset_min"])] + [int(x) for x in sp.get("sensitivity_offsets_min") or []]
    pol = Policy.from_config(cfg)
    thresholds = {s: cfg.get(s) for s in THRESHOLD_SECTIONS}
    config_hash = stable_hash(thresholds)
    run_ids = sorted(runs["model_run_id"])

    run_rows, decision_frames, summary = [], [], {}
    exits: dict[tuple[str, str], dict[str, Any]] = {}
    if pol.exit_rule.exits:
        ex = db.df(_EXIT_SQL, [pol.exit_rule.offset_min, quality])
        exits = {(r["game_id"], r["contract_id"]): r for r in ex.to_dict(orient="records")}
    for off in offsets:
        rows = db.df(_ROWS_SQL, [run_ids, off, quality])
        assert_not_holdout(rows["period_id"].unique(), periods, purpose="backtest")
        out = [r for rec in rows.to_dict(orient="records")
               for r in decide(rec, pol, exits.get((rec["game_id"], rec["contract_id"])))]
        df = one_position_per_game(pd.DataFrame(out).reindex(columns=BACKTEST_DECISIONS.column_names), pol)
        # one id per invocation: the commit and the time make a rerun a new run, never an overwrite (ADR 0001 N1)
        bt_id = "bt-" + stable_hash({"model_runs": run_ids, "offset": off, "config": config_hash,
                                     "model_type": model_type, "quote_quality": quality,
                                     "git_commit": commit, "created_at": now.isoformat()})
        df["backtest_run_id"] = bt_id
        df["is_synthetic"] = db.is_synthetic
        decision_frames.append(df)
        run_rows.append({
            "backtest_run_id": bt_id, "model_type": model_type, "snapshot_offset_min": off,
            "config_json": json.dumps(thresholds | {"model_run_ids": run_ids, "snapshot_offset_min": off,
                                                    "selection_reason": selection.get("reason")},
                                      sort_keys=True, default=str),
            "config_hash": config_hash, "quote_quality": quality, "dataset": db.dataset, "git_commit": commit,
            "created_at": now, "is_synthetic": db.is_synthetic})
        base = df[df["scenario"] == scenario_name(pol.baseline)]
        summary[off] = {"observations": int(len(base)), "trades": int((base["decision"] != "PASS").sum()),
                        "pass_reasons": base["pass_reason"].value_counts().to_dict(),
                        "exit_rule": pol.exit_rule.rule, "exit_statuses": base["exit_status"].value_counts().to_dict(),
                        "exit_reasons": base["exit_reason"].value_counts().to_dict(),
                        "by_scenario": {s: float(g["net_pnl"].sum()) for s, g in df.groupby("scenario")}}
    decisions = pd.concat(decision_frames, ignore_index=True)
    db.replace_tables({"fee_schedules": fee_schedule_rows()})             # static reference rows
    # append-only: earlier runs stay; a repeated id raises ConstraintViolation and writes nothing (ADR 0001 N1)
    db.append_tables({"backtest_runs": pd.DataFrame(run_rows), "backtest_decisions": decisions})

    canon_id = run_rows[0]["backtest_run_id"]
    # model_run_ids carry the data fingerprint, so new data is a new experiment
    logged = _record_h3(db, cfg, decisions[decisions["backtest_run_id"] == canon_id], pol, model_type,
                        stable_hash({"thresholds": config_hash, "model_runs": run_ids}), notes, now, quality)
    return BacktestResult(pd.DataFrame(run_rows), int(len(decisions)), summary, logged, notes)


# ---------------------------------------------------------------------------
# H_MLB_003 primary metric
# ---------------------------------------------------------------------------

def _diff_bootstrap(t: pd.DataFrame, *, n: int, seed: int, ci: float) -> dict[str, Any]:
    """mean(edge | inside) - mean(edge | outside), resampling games; two-sided p."""
    inside = t["inside"].to_numpy(bool)
    e = t["edge"].to_numpy(float)
    sums = M._game_sums(t["game_id"], np.where(inside, e, 0.0), inside.astype(float),
                        np.where(~inside, e, 0.0), (~inside).astype(float))
    point = e[inside].mean() - e[~inside].mean()
    boot = M._bootstrap_sums(sums, n, np.random.default_rng(seed))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = boot[:, 0] / boot[:, 1] - boot[:, 2] / boot[:, 3]
    d = d[np.isfinite(d)]
    lo, hi = M._ci(d, ci)
    p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))) if len(d) else None
    return {"diff": float(point), "ci_low": lo, "ci_high": hi, "p_value": p, "n_games": int(sums.shape[0])}


def _record_h3(db: Database, cfg: dict[str, Any], dec: pd.DataFrame, pol: Policy, model_type: str,
               config_hash: str, notes: list[str], now: datetime, quote_quality: str = EXECUTABLE) -> int:
    hyp = db.df("SELECT minimum_sample, label FROM hypotheses WHERE hypothesis_id = ? AND dataset = ?",
                [H3, db.dataset])
    if hyp.empty:
        notes.append(f"{H3} not registered; primary metric not recorded")
        return 0
    st = current_statuses(db).get(H3, {})
    if st.get("status", ACTIVE) != ACTIVE:
        notes.append(f"{H3} is {st['status']} ({st.get('decision_ref')}); primary metric not recorded")
        return 0
    band = db.df("SELECT observation_id, value FROM feature_values WHERE feature_name = ?", [H3_FEATURE])
    if band.empty:
        notes.append(f"{H3}: feature {H3_FEATURE} not built; primary metric not recorded")
        return 0
    trades = dec[(dec["scenario"] == scenario_name(pol.baseline)) & (dec["decision"] != "PASS")
                 & dec["net_pnl"].notna()]
    t = trades.merge(band, on="observation_id", how="inner").dropna(subset=["value"])
    t = t.assign(inside=t["value"] > 0.5, edge=t["net_pnl"] / t["quantity"])
    rc = cfg.get("review") or {}
    n_boot, ci, seed = int(rc.get("bootstrap_resamples", 10_000)), float(rc.get("ci_level", 0.95)), int(rc.get("seed", 7))
    minimum = int(hyp["minimum_sample"].iloc[0])
    commit = holdout.git_commit()
    rows = []
    for pid, g in [*t.groupby("period_id", sort=True), ("pooled", t)]:
        if g["inside"].all() or not g["inside"].any():
            notes.append(f"{H3} {pid}: trades only on one side of the fee band ({len(g)} trades); not tested")
            continue
        bs = _diff_bootstrap(g, n=n_boot, seed=seed, ci=ci)
        metrics = {"reference": "trades outside the fee band", "candidate": "trades inside the fee band",
                   "n_obs": int(len(g)), "n_games": bs["n_games"], "brier_ref": None, "brier_cand": None,
                   "delta_brier": None, "metric": "realised net edge per contract (net_pnl / quantity)",
                   "realized_edge_inside": float(g.loc[g["inside"], "edge"].mean()),
                   "realized_edge_outside": float(g.loc[~g["inside"], "edge"].mean()),
                   "n_trades_inside": int(g["inside"].sum()), "n_trades_outside": int((~g["inside"]).sum()),
                   "diff_inside_minus_outside": bs["diff"], "diff_ci": [bs["ci_low"], bs["ci_high"]],
                   "p_value_sidedness": "two-sided", "scenario": scenario_name(pol.baseline),
                   "minimum_sample": minimum, "sufficient_sample": bool(len(g) >= minimum)}
        rows.append({"hypothesis_id": H3, "experiment_family": "exploratory", "label": hyp["label"].iloc[0],
                     "feature_set_id": "fee_band_split", "feature_set_hash": stable_hash([H3_FEATURE]),
                     "model_type": model_type, "config_hash": config_hash,
                     "train_period": "walk_forward" if pid == "pooled" else f"before:{pid}", "test_period": pid,
                     "n_games": bs["n_games"], "metrics_json": json.dumps(metrics, sort_keys=True),
                     "p_value": bs["p_value"], "quote_quality": quote_quality, "git_commit": commit,
                     "created_at": now, "dataset": db.dataset})
    return record_experiments(db, rows)
