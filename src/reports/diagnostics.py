"""Evaluation report (spec Sec 17) and the `quant review` artifact (Sec 26, 27).

`build_review` applies the fixed Sec 26 acceptance rules mechanically. Every
criterion is {id, title, state, value, threshold, detail} with state PASS |
FAIL | INSUFFICIENT SAMPLE, and the overall status (26.19) follows from the
states alone (`decide_status`). No subjective assessment is produced.

FINAL_HOLDOUT is never evaluated (26.3): every frame is checked by
`assert_not_holdout` and against the holdout date window before any metric is
computed, so holdout rows raise HoldoutLocked. Only evaluation-role periods are
scored. Every threshold comes from the sport config; none is tuned here.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..backtest import metrics as M
from ..backtest.holdout import DIRTY, GIT_UNAVAILABLE, is_clean_commit, tree_state
from ..backtest.registry import hypotheses_with_status
from ..backtest.walk_forward import HoldoutLocked, Period, assert_not_holdout, evaluation_periods, \
    holdout_period, load_periods
from ..config import stable_hash, strategy_family
from ..storage.database import Database
from ..storage.schemas import CANDLE_DEGRADED, DEGRADED_EXECUTION, EXECUTABLE, LINEAGE_KINDS, QUOTE_QUALITIES

PASS, FAIL, INSUFFICIENT = "PASS", "FAIL", M.INSUFFICIENT
STATES = (PASS, FAIL, INSUFFICIENT)
STATUSES = ("REJECTED", "INCONCLUSIVE", "RESEARCH-PROMISING", "PROSPECTIVELY-VALIDATED")
BANNER = "SYNTHETIC DATA — pipeline demonstration, not evidence about any real market"
REVIEW_VERSION = "review-v1"
# The version of the criterion logic in this file.
#   referee-v1  the first logic (no 26.2 floor; 26.12 read the binned-frequency proxy; 26.9 FAILED on
#               zero trades).
#   referee-v2  the 26.2 period floor, the 26.12 outcome upper bound, 26.9 INSUFFICIENT on zero trades.
# Later changes also run as referee-v2: the stricter 26.5; the 26.13 slope CI; negative controls outside the
# BH-FDR set; the 26.4 feature-family and 26.5 FDR-family labels; the exploratory family outside the BH-FDR
# set (ADR 0001 Q1); -dirty provenance (ADR 0001 Q4: 26.5 on the BH set, a cap on the run under review); a
# row feeding the run under review without a clean commit caps the run, as the run's own commit does; and
# run lineage: for a run with lineage, that cap and 26.16 read only the rows of the run's own train
# invocations, 26.5 stays global, and a run without lineage is labelled LEGACY_LINEAGE_FALLBACK. The config
# values they add change the config hash.
REFEREE_VERSION = "referee-v2"
CRITERIA_IDS = tuple(f"26.{i}" for i in range(1, 19))
PROMISING_REQUIRES = ("26.1", "26.2", "26.7", "26.9", "26.12", "26.14")
POOLED_TEST_PERIODS = ("pooled", "all", "aggregate")
HOLDOUT_NOT_EVALUATED = "FINAL_HOLDOUT locked; not evaluated in this engagement (26.3)"
NEGATIVE_CONTROL = "negative_control"
CONTROLS_LABEL = "Negative controls (sanity checks, not hypotheses)"
CONTROLS_NOTE = ("negative-control experiments are sanity checks, not hypotheses: they are "
                 "outside the BH-FDR set (review.fdr_families) and have no q_value or success. control_q is "
                 "Benjamini-Hochberg within the controls not tested on the holdout: the statistic 26.16 reads")
EXPLORATORY = "exploratory"
EXPLORATORY_LABEL = "EXPLORATORY (not in the BH-FDR set; never counted toward any acceptance criterion)"
EXPLORATORY_NOTE = ("exploratory experiments are outside the BH-FDR set (ADR 0001 Q1): raw p only, no q and no "
                    "success, and no acceptance criterion counts them. A result here is a lead for a new "
                    "preregistered hypothesis, not evidence")
LINEAGE = "LINEAGE"
LEGACY_LINEAGE_FALLBACK = "LEGACY_LINEAGE_FALLBACK"
LINEAGE_NOTE = ("the feeding-rows cap and 26.16 read only the experiments and hypotheses of the train invocations "
                "that wrote the feeding model runs; 26.5 reads every row")
FALLBACK_NOTE = ("the feeding-rows cap and 26.16 read the whole experiments and hypotheses logs, as for a run "
                 "trained before lineage was recorded; 26.5 reads every row")


def referee_config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable hash of the referee's configuration: the config's review section and the criteria set
    (the criterion ids and the ones RESEARCH-PROMISING requires). Key order in the YAML does not matter."""
    return stable_hash({"review": dict(cfg.get("review") or {}), "criteria": list(CRITERIA_IDS),
                        "promising_requires": list(PROMISING_REQUIRES)})


class MixedQuoteQuality(ValueError):
    """Executable and candle_degraded rows never share one result."""

# 26.11: every one of these counts must be present and 0.
REQUIRED_LEAKAGE_CHECKS = ("feature_available_after_cutoff", "feature_cutoff_after_quote",
                           "quote_available_after_snapshot")
LEAKAGE_SQL = {
    "feature_available_after_cutoff": "SELECT count(*) FROM feature_values WHERE available_at > cutoff_time",
    "feature_cutoff_after_quote": "SELECT count(*) FROM feature_values f JOIN observations o "
                                  "ON o.observation_id = f.observation_id WHERE f.cutoff_time > o.quote_ts",
    "quote_available_after_snapshot": "SELECT count(*) FROM observations WHERE quote_available_at > snapshot_time",
    "feature_available_after_snapshot": "SELECT count(*) FROM feature_values f JOIN observations o "
                                        "ON o.observation_id = f.observation_id WHERE f.available_at > o.snapshot_time",
    "feature_snapshot_mismatch": "SELECT count(*) FROM feature_values f JOIN observations o "
                                 "ON o.observation_id = f.observation_id WHERE f.snapshot_time <> o.snapshot_time",
    "feature_value_without_available_at": "SELECT count(*) FROM feature_values "
                                          "WHERE value IS NOT NULL AND available_at IS NULL",
    "consensus_available_after_quote": "SELECT count(*) FROM observations WHERE consensus_available_at > quote_ts",
}

PRED_NUM = ("p_market", "p_model", "lower_bound", "upper_bound", "p_conservative_yes", "p_conservative_no",
            "outcome", "quote_age_s", "spread", "p_true", "snapshot_offset_min")
DEC_NUM = ("p_market", "p_model", "p_conservative", "predicted_edge", "net_edge", "quoted_price",
           "spread_penalty_frac", "execution_price", "quantity", "stake", "calculated_fee", "slippage_cost",
           "outcome", "side_won", "gross_pnl", "net_pnl", "quote_age_s", "spread", "snapshot_offset_min",
           "exit_quoted_price", "exit_price", "exit_quantity", "held_quantity", "exit_proceeds", "exit_fee",
           "exit_slippage_cost")
BUCKET_COLUMNS = ("bucket", "rank", "lower", "upper", "observations", "n_games", "trades", "mean_p_market",
                  "mean_p_model", "mean_p_conservative", "empirical_frequency", "mean_predicted_edge",
                  "realized_edge", "realized_edge_ci_low", "realized_edge_ci_high", "gross_roi", "net_roi",
                  "sufficient", "label")


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass
class ReviewInputs:
    """Everything `build_review` needs; no database access after loading.

    predictions: one row per evaluated canonical observation (see module
    docstring of load_review_inputs for columns). decisions: backtest_decisions
    rows of the T-30 run (all scenarios) plus observation metadata.
    sensitivity: {snapshot offset: that offset's decisions}.
    feeding_runs: the model runs the run under review and its sensitivity runs decided with, plus any the
    run's config names (columns table, row_id, git_commit; a named run missing from model_runs has no
    commit). Read by `feeding_rows`.
    lineage: the train_lineage rows (invocation_id, kind, row_id) of the train invocations that wrote the
    feeding model runs. Read by `lineage_scope`.
    """

    sport: str
    dataset: str
    is_synthetic: bool
    periods: list[Period]
    predictions: pd.DataFrame
    decisions: pd.DataFrame
    sensitivity: dict[int, pd.DataFrame] = field(default_factory=dict)
    experiments: pd.DataFrame = field(default_factory=_empty)
    hypotheses: pd.DataFrame = field(default_factory=_empty)
    paper_trades: pd.DataFrame = field(default_factory=_empty)
    paper_settlements: pd.DataFrame = field(default_factory=_empty)
    holdout_manifest: dict[str, Any] = field(default_factory=dict)
    leakage_audit: dict[str, Any] = field(default_factory=dict)
    model_selection: dict[str, Any] = field(default_factory=dict)
    exclusions: pd.DataFrame = field(default_factory=_empty)
    baseline_scenario: str = "pen_25"
    fee_schedules: pd.DataFrame = field(default_factory=_empty)
    backtest_run: dict[str, Any] = field(default_factory=dict)
    observation_counts: dict[str, Any] = field(default_factory=dict)
    quote_quality: str = EXECUTABLE
    feeding_runs: pd.DataFrame = field(default_factory=_empty)
    lineage: pd.DataFrame = field(default_factory=_empty)


# ---------------------------------------------------------------------------
# Loading (the only database access in this module)
# ---------------------------------------------------------------------------

INVOCATION = ("created_at", "git_commit", "config_hash", "model_type", "quote_quality")


def _same_invocation(runs: pd.DataFrame, run: pd.Series) -> pd.Series:
    """Rows of `runs` written by the same `quant backtest` invocation as `run` (ADR 0001 N1).
    A missing git_commit matches a missing one."""
    mask = pd.Series(True, index=runs.index)
    for col in INVOCATION:
        v = run[col]
        mask &= runs[col].isna() if pd.isna(v) else (runs[col] == v).fillna(False).astype(bool)
    return mask


def load_review_inputs(db: Database, cfg: dict[str, Any], *, backtest_run_id: str | None = None,
                       holdout_manifest: dict[str, Any] | None = None, leakage_audit: dict[str, Any] | None = None,
                       model_selection: dict[str, Any] | None = None,
                       truth: pd.DataFrame | Mapping[str, float] | None = None,
                       quote_quality: str | None = None) -> ReviewInputs:
    """Assemble ReviewInputs from the database.

    Uses the latest backtest_runs row at the canonical offset (T-30) unless
    `backtest_run_id` is given. predictions = the (model_run_id,
    observation_id) pairs that run decided on, joined to observations.
    sensitivity = the run at each `snapshot_policy.sensitivity_offsets_min` offset from
    the same `quant backtest` invocation as the run under review: the same created_at,
    git_commit, config_hash, model_type and quote_quality. backtest_runs is append-only,
    so an older or other invocation is never mixed in; an offset it lacks is left out
    (ADR 0001 N1). Only evaluation-role periods are kept;
    holdout-dated rows are dropped here and rejected again by build_review.
    `truth` (synthetic only) maps observation_id -> p_true; it is never a model input.
    `quote_quality` restricts the run choice to one class; everything else loaded
    (sensitivity runs, experiments, observation counts) is then that run's class
    only.
    """
    periods = load_periods(cfg, db.dataset)
    sp = cfg.get("snapshot_policy") or {}
    canon = int(sp.get("canonical_offset_min", 30))
    runs = db.df("SELECT * FROM backtest_runs ORDER BY created_at DESC, backtest_run_id DESC")
    if quote_quality is not None:
        if quote_quality not in QUOTE_QUALITIES:
            raise ValueError(f"quote_quality must be one of {QUOTE_QUALITIES}, got {quote_quality!r}")
        runs = runs[runs["quote_quality"] == quote_quality]
    if backtest_run_id is not None:
        sel = runs[runs["backtest_run_id"] == backtest_run_id]
        if sel.empty:
            raise ValueError(f"backtest run {backtest_run_id!r} not found")
    else:
        sel = runs[M.num(runs["snapshot_offset_min"]) == canon]
        if sel.empty:
            raise ValueError(f"no backtest run at T-{canon}"
                             f"{f' with quote_quality {quote_quality}' if quote_quality else ''}; "
                             f"run `quant backtest` first")
    run = _run_dict(sel.iloc[0])
    rid = run["backtest_run_id"]
    quality = run.get("quote_quality") or EXECUTABLE
    same_invocation = _same_invocation(runs, sel.iloc[0])          # also one quote class

    decisions = _keep_evaluation(_load_decisions(db, rid), periods)
    predictions = _keep_evaluation(db.df(_PREDICTIONS_SQL, [rid]), periods)
    avail = _availability(db, rid)
    for name in ("decisions", "predictions"):
        frame = decisions if name == "decisions" else predictions
        frame = frame.merge(avail, on="observation_id", how="left")
        frame["availability_state"] = frame["availability_state"].fillna("no_features")
        if name == "decisions":
            decisions = frame
        else:
            predictions = frame
    if truth is not None:
        t = truth if isinstance(truth, pd.DataFrame) else pd.DataFrame(
            {"observation_id": list(truth), "p_true": list(truth.values())})
        predictions = predictions.merge(t[["observation_id", "p_true"]], on="observation_id", how="left")

    sensitivity: dict[int, pd.DataFrame] = {}
    run_ids = [rid]
    for off in sp.get("sensitivity_offsets_min") or []:
        cand = runs[same_invocation & (M.num(runs["snapshot_offset_min"]) == int(off))]
        if len(cand):
            run_ids.append(str(cand["backtest_run_id"].iloc[0]))
            sensitivity[int(off)] = _keep_evaluation(_load_decisions(db, run_ids[-1]), periods)
    named = [str(m) for m in (run["config"].get("model_run_ids") or [])]
    feeding_runs = db.df(_FEEDING_RUNS_SQL, [named, run_ids])
    lineage = db.df(_LINEAGE_SQL, [feeding_runs["row_id"].astype(str).tolist()])

    exclusions = _keep_evaluation(db.df("SELECT * FROM snapshot_exclusions"), periods)
    counts = _keep_evaluation(db.df(
        "SELECT official_date, snapshot_offset_min, count(*) AS n, "
        "count(*) FILTER (WHERE NOT size_verified) AS n_size_unverified FROM observations "
        "WHERE quote_quality = ? GROUP BY ALL", [quality]), periods)
    by_off = counts.groupby("snapshot_offset_min")[["n", "n_size_unverified"]].sum() if len(counts) else None
    observation_counts = {
        "by_offset": {int(k): int(v) for k, v in by_off["n"].items()} if by_off is not None else {},
        "canonical": int(by_off["n"].get(canon, 0)) if by_off is not None else 0,
        "canonical_size_unverified": int(by_off["n_size_unverified"].get(canon, 0)) if by_off is not None else 0,
        "scope": f"evaluation periods only; quote_quality = {quality}",
    }
    ms = dict(model_selection or {})
    if "feature_families" not in ms:
        fams = db.df(
            "SELECT DISTINCT d.family FROM feature_values f JOIN feature_definitions d USING (feature_name) "
            "WHERE f.observation_id IN (SELECT observation_id FROM backtest_decisions WHERE backtest_run_id = ?)",
            [rid])["family"].tolist()
        ms["feature_families"] = sorted(fams)
        ms["feature_families_source"] = "families present in feature_values for the evaluated observations"
    return ReviewInputs(
        sport=db.sport, dataset=db.dataset, is_synthetic=db.is_synthetic, periods=periods,
        predictions=predictions, decisions=decisions, sensitivity=sensitivity,
        experiments=db.df("SELECT * FROM experiments WHERE quote_quality = ?", [quality]),
        hypotheses=hypotheses_with_status(db),
        paper_trades=db.df("SELECT * FROM paper_trades"), paper_settlements=db.df("SELECT * FROM paper_settlements"),
        holdout_manifest=dict(holdout_manifest or {}), leakage_audit=dict(leakage_audit or {}),
        model_selection=ms, exclusions=exclusions,
        baseline_scenario=_scenario_name((cfg.get("execution") or {}).get("baseline_penalty", 0.25)),
        fee_schedules=db.df("SELECT * FROM fee_schedules"), backtest_run=run, observation_counts=observation_counts,
        quote_quality=quality, feeding_runs=feeding_runs, lineage=lineage)


_OBS_META = ("o.official_date, o.season, o.home_team, o.away_team, o.yes_team, o.yes_side, o.liquidity_tercile, "
             "o.snapshot_offset_min, o.is_canonical, o.quote_age_s, o.spread, o.size_verified, o.quote_quality")

_PREDICTIONS_SQL = f"""
SELECT p.observation_id, o.game_id, o.contract_id, p.model_run_id, p.period_id, p.p_market, p.p_model,
       p.lower_bound, p.upper_bound, p.p_conservative_yes, p.p_conservative_no, o.outcome, o.quote_source,
       {_OBS_META}
FROM (SELECT DISTINCT model_run_id, observation_id FROM backtest_decisions
      WHERE backtest_run_id = ? AND model_run_id IS NOT NULL) k
JOIN predictions p ON p.model_run_id = k.model_run_id AND p.observation_id = k.observation_id
JOIN observations o ON o.observation_id = p.observation_id
ORDER BY o.snapshot_time, p.observation_id
"""


# The model runs named in the run's config (first parameter) and those that decided in the
# run or its sensitivity runs (second). A run missing from model_runs keeps a NULL commit: not clean.
_FEEDING_RUNS_SQL = """
SELECT 'model_runs' AS "table", k.model_run_id AS row_id, m.git_commit
FROM (SELECT unnest(?::VARCHAR[]) AS model_run_id
      UNION
      SELECT DISTINCT model_run_id FROM backtest_decisions
      WHERE backtest_run_id IN (SELECT unnest(?::VARCHAR[])) AND model_run_id IS NOT NULL) k
LEFT JOIN model_runs m USING (model_run_id)
ORDER BY k.model_run_id
"""

# Every lineage row of the train invocations that wrote any of the feeding model runs.
_LINEAGE_SQL = """
SELECT invocation_id, kind, row_id FROM train_lineage
WHERE invocation_id IN (SELECT invocation_id FROM train_lineage
                        WHERE kind = 'model_run' AND row_id IN (SELECT unnest(?::VARCHAR[])))
ORDER BY invocation_id, kind, row_id
"""


def _load_decisions(db: Database, run_id: str) -> pd.DataFrame:
    return db.df(f"SELECT d.*, {_OBS_META} FROM backtest_decisions d LEFT JOIN observations o "
                 f"ON o.observation_id = d.observation_id WHERE d.backtest_run_id = ? "
                 f"ORDER BY d.scenario, d.snapshot_time, d.observation_id", [run_id])


def _availability(db: Database, run_id: str) -> pd.DataFrame:
    """availability_state per observation: complete | missing:<families> | no_features."""
    df = db.df(
        "SELECT f.observation_id, string_agg(DISTINCT d.family, '+' ORDER BY d.family) "
        "FILTER (WHERE f.value IS NULL) AS missing FROM feature_values f JOIN feature_definitions d "
        "USING (feature_name) WHERE f.observation_id IN "
        "(SELECT observation_id FROM backtest_decisions WHERE backtest_run_id = ?) GROUP BY 1", [run_id])
    miss = df["missing"].astype(object)
    df["availability_state"] = np.where(miss.isna(), "complete", "missing:" + miss.fillna("").astype(str))
    return df[["observation_id", "availability_state"]]


def _run_dict(row: pd.Series) -> dict[str, Any]:
    run = {k: _jsonable(v) for k, v in row.items() if k != "config_json"}
    try:
        conf = json.loads(row.get("config_json") or "{}")
    except (TypeError, ValueError):
        conf = {}
    run["model_type"] = run.get("model_type") or conf.get("model_type")
    run["config"] = conf
    return run


def _scenario_name(penalty: float) -> str:
    return f"pen_{int(round(100 * float(penalty)))}"


def run_leakage_audit(db: Database) -> dict[str, int]:
    """26.11 counts over the whole database. Every count must be 0."""
    return {name: int(db.scalar(sql) or 0) for name, sql in LEAKAGE_SQL.items()}


# ---------------------------------------------------------------------------
# Holdout guard and preparation
# ---------------------------------------------------------------------------

def _dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    return d.dt.normalize()


def _in_window(s: pd.Series, start: date, end: date) -> np.ndarray:
    d = _dates(s)
    return ((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))).to_numpy(dtype=bool)


def _keep_evaluation(df: pd.DataFrame, periods: list[Period]) -> pd.DataFrame:
    """Rows in evaluation-role periods only; never a holdout-dated row."""
    if df.empty:
        return df
    keep = np.ones(len(df), dtype=bool)
    if "period_id" in df:
        keep &= df["period_id"].isin([p.period_id for p in evaluation_periods(periods)]).to_numpy(dtype=bool)
    if "official_date" in df:
        in_eval = np.zeros(len(df), dtype=bool)
        for p in evaluation_periods(periods):
            in_eval |= _in_window(df["official_date"], p.start, p.end)
        keep &= in_eval
    return df[keep].reset_index(drop=True)


def _guard_holdout(inputs: ReviewInputs) -> None:
    """Raise HoldoutLocked if anything handed to the review lies in FINAL_HOLDOUT (26.3)."""
    hold = holdout_period(inputs.periods)
    frames = {"predictions": inputs.predictions, "decisions": inputs.decisions,
              **{f"sensitivity T-{k}": v for k, v in inputs.sensitivity.items()}}
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        if "period_id" in df:
            assert_not_holdout(pd.unique(df["period_id"].dropna()), inputs.periods, purpose=f"review ({name})")
        if "official_date" in df:
            n = int(_in_window(df["official_date"], hold.start, hold.end).sum())
            if n:
                raise HoldoutLocked(f"review ({name}): {n} rows dated inside holdout {hold.period_id} "
                                    f"({hold.start}..{hold.end}); refusing to evaluate (26.3)")


def _numeric(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = M.num(out[c])
    return out


def _prep_predictions(df: pd.DataFrame, eval_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if df is None or df.empty:
        return pd.DataFrame(columns=["observation_id", "game_id", "period_id", "y", "p_model", "p_market"]), \
            ["no predictions"]
    p = _numeric(df, PRED_NUM)
    n0 = len(p)
    p = p[p["period_id"].isin(eval_ids)]
    if len(p) < n0:
        notes.append(f"{n0 - len(p)} prediction rows outside evaluation periods ignored")
    settled = np.isfinite(p["outcome"].to_numpy())
    if (~settled).any():
        notes.append(f"{int((~settled).sum())} unsettled prediction rows ignored")
    p = p[settled].reset_index(drop=True)
    p["y"] = p["outcome"]
    p["size_verified"] = M.flag(p["size_verified"]) if "size_verified" in p else np.zeros(len(p), dtype=bool)
    return p, notes


def _prep_decisions(df: pd.DataFrame, eval_ids: list[str], label: str = "decision") -> tuple[pd.DataFrame, list[str]]:
    if df is None or df.empty:
        return pd.DataFrame(), [f"no {label} rows"]
    d = _numeric(df, DEC_NUM)
    notes = []
    n0 = len(d)
    d = d[d["period_id"].isin(eval_ids)]
    if len(d) < n0:
        notes.append(f"{n0 - len(d)} {label} rows outside evaluation periods ignored")
    settled = np.isfinite(_yes_outcome(d))
    if (~settled).any():
        notes.append(f"{int((~settled).sum())} unsettled {label} rows ignored")
    return d[settled].reset_index(drop=True), notes


def _guard_quote_quality(inputs: ReviewInputs) -> None:
    """One quote-quality class per result, never mixed."""
    q = inputs.quote_quality
    if q not in QUOTE_QUALITIES:
        raise ValueError(f"quote_quality must be one of {QUOTE_QUALITIES}, got {q!r}")
    frames = {"predictions": inputs.predictions, "decisions": inputs.decisions,
              **{f"T-{k} sensitivity decisions": v for k, v in inputs.sensitivity.items()}}
    for name, df in frames.items():
        if df is not None and "quote_quality" in df:
            other = sorted(set(df["quote_quality"].dropna()) - {q})
            if other:
                raise MixedQuoteQuality(f"review is {q} but its {name} include {other} rows")


def _prepare(inputs: ReviewInputs) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    _guard_holdout(inputs)
    _guard_quote_quality(inputs)
    eval_ids = [p.period_id for p in evaluation_periods(inputs.periods)]
    preds, pn = _prep_predictions(inputs.predictions, eval_ids)
    dec, dn = _prep_decisions(inputs.decisions, eval_ids)
    return preds, dec, eval_ids, pn + dn


# ---------------------------------------------------------------------------
# Sec 17 evaluation report (`quant report`, `quant backtest`)
# ---------------------------------------------------------------------------

def _quality(g: pd.DataFrame, bins: int) -> dict[str, Any]:
    y, pm, pk = g["y"], g["p_model"], g["p_market"]
    cm, ck = M.calibration_table(y, pm, bins), M.calibration_table(y, pk, bins)
    bm, bk = M.brier(y, pm), M.brier(y, pk)
    return {"n_obs": int(len(g)), "n_games": int(g["game_id"].nunique()), "brier_market": bk, "brier_model": bm,
            "delta_brier": bk - bm, "bss": M.bss(bm, bk), "log_loss_market": M.log_loss(y, pk),
            "log_loss_model": M.log_loss(y, pm), "ece_market": ck["ece"], "ece_model": cm["ece"],
            "calibration_slope_model": cm["slope"], "calibration_slope_market": ck["slope"]}


def _scenario_order(dec: pd.DataFrame) -> list[str]:
    if dec.empty:
        return []
    pen = dec.groupby("scenario")["spread_penalty_frac"].first() if "spread_penalty_frac" in dec else \
        pd.Series(0.0, index=pd.unique(dec["scenario"]))
    return [str(s) for s in pen.sort_values(kind="stable").index]


def _evaluation(inputs: ReviewInputs, cfg: dict[str, Any], preds: pd.DataFrame, dec: pd.DataFrame,
                eval_ids: list[str], notes: list[str]) -> dict[str, Any]:
    rc = cfg["review"]
    bins = int(rc["calibration_bins"])
    base = dec[dec["scenario"] == inputs.baseline_scenario] if len(dec) else dec
    pq: dict[str, Any] = {"pooled": None, "per_period": pd.DataFrame(), "calibration": {}, "bootstrap": None}
    if len(preds):
        pq["pooled"] = _quality(preds, bins)
        pq["per_period"] = pd.DataFrame([{"period_id": pid, **_quality(g, bins)}
                                         for pid in eval_ids if len(g := preds[preds["period_id"] == pid])])
        pq["calibration"] = {k: M.calibration_table(preds["y"], preds[c], bins)
                             for k, c in (("model", "p_model"), ("market", "p_market"))}
        pq["bootstrap"] = M.paired_game_bootstrap(preds, n=int(rc["bootstrap_resamples"]), seed=int(rc["seed"]),
                                                  ci=float(rc["ci_level"]))
    edges = list(rc["bucket_edges"])
    buckets = M.edge_buckets(base, edges, int(rc["bucket_min_games"]), n_boot=int(rc["bootstrap_resamples"]),
                             seed=int(rc["seed"]), ci=float(rc["ci_level"])) if len(base) else \
        pd.DataFrame([{"bucket": b, "rank": i, "observations": 0, "n_games": 0, "trades": 0, "sufficient": False,
                       "label": INSUFFICIENT} for i, b in enumerate(M.bucket_labels(edges))],
                     columns=list(BUCKET_COLUMNS))
    return {
        "sport": inputs.sport, "dataset": inputs.dataset, "is_synthetic": inputs.is_synthetic,
        "banner": BANNER if inputs.is_synthetic else None, "backtest_run": inputs.backtest_run,
        "quote_quality": inputs.quote_quality,
        "execution_banner": DEGRADED_EXECUTION if inputs.quote_quality == CANDLE_DEGRADED else None,
        "baseline_scenario": inputs.baseline_scenario, "evaluation_periods": eval_ids,
        "holdout": HOLDOUT_NOT_EVALUATED, "data_notes": notes, "prediction_quality": pq,
        "trading": {s: M.trading_summary(dec[dec["scenario"] == s]) for s in _scenario_order(dec)},
        "edge_buckets": buckets,
    }


def evaluation_report(inputs: ReviewInputs, cfg: dict[str, Any]) -> dict[str, Any]:
    """Sec 17: prediction quality (market vs model, per period and pooled, with calibration
    tables and the paired game bootstrap), trading per scenario, and the edge-bucket table."""
    preds, dec, eval_ids, notes = _prepare(inputs)
    return _evaluation(inputs, cfg, preds, dec, eval_ids, notes)


# ---------------------------------------------------------------------------
# Sec 26 review
# ---------------------------------------------------------------------------

def _crit(cid: str, title: str, state: str, value: Any = None, threshold: Any = None, detail: str = "",
          **extra: Any) -> dict[str, Any]:
    if state not in STATES:
        raise ValueError(f"{cid}: invalid state {state!r}")
    return {"id": cid, "title": title, "state": state, "value": value, "threshold": threshold, "detail": detail,
            **extra}


def _ceil(x: float) -> int:
    return int(math.ceil(round(x, 9)))


def _parse_json(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return s
    try:
        out = json.loads(s) if isinstance(s, str) else {}
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def experiment_registry(experiments: pd.DataFrame, *, fdr_q: float, fdr_families: Any,
                        holdout_id: str | None = None) -> pd.DataFrame:
    """Experiments with raw p, BH q within each experiment_family, and success (q <= fdr_q and
    delta_brier > 0), for the FDR families only (`fdr_families`, review.fdr_families).

    fdr_role is "fdr" for an FDR family, "negative_control" for the controls (sanity checks, not
    hypotheses: no q here; see negative_control_rows), "exploratory" for the exploratory family (ADR 0001
    Q1; see exploratory_rows) and "unassigned" for any other family. Controls, exploratory and unassigned
    rows get no q and no success, even when review.fdr_families names them; 26.5 fails closed on unassigned
    rows. Experiments tested on the holdout get no q and are flagged."""
    cols = ["experiment_id", "experiment_family", "fdr_role", "hypothesis_id", "label", "feature_set_id",
            "model_type", "train_period", "test_period", "n_games", "delta_brier", "p_value", "q_value", "success",
            "tested_on_holdout", "git_commit", "created_at"]
    if experiments is None or experiments.empty:
        return pd.DataFrame(columns=cols)
    e = experiments.copy()
    metrics = e["metrics_json"].map(_parse_json) if "metrics_json" in e else pd.Series([{}] * len(e))
    e["delta_brier"] = M.num(pd.Series([m.get("delta_brier") for m in metrics], index=e.index, dtype=object))
    e["p_value"] = M.num(e["p_value"]) if "p_value" in e else np.nan
    e["tested_on_holdout"] = (e["test_period"] == holdout_id).to_numpy(dtype=bool) if holdout_id else False
    fam = e["experiment_family"].astype("string")
    is_nc = (fam == NEGATIVE_CONTROL).fillna(False).to_numpy(dtype=bool)
    is_x = (fam == EXPLORATORY).fillna(False).to_numpy(dtype=bool)
    in_set = fam.isin([str(f) for f in fdr_families]).fillna(False).to_numpy(dtype=bool)
    e["fdr_role"] = np.where(is_nc, NEGATIVE_CONTROL, np.where(is_x, EXPLORATORY,
                                                               np.where(in_set, "fdr", "unassigned")))
    q = np.full(len(e), np.nan)
    ok = ~e["tested_on_holdout"].to_numpy(dtype=bool) & (e["fdr_role"] == "fdr").to_numpy()
    for _, idx in e[ok].groupby("experiment_family").indices.items():
        pos = np.flatnonzero(ok)[idx]
        q[pos] = M.benjamini_hochberg(e["p_value"].to_numpy()[pos])
    e["q_value"] = q
    e["success"] = (q <= fdr_q) & (e["delta_brier"].to_numpy() > 0)
    return e.reindex(columns=cols).reset_index(drop=True)


def negative_control_rows(reg: pd.DataFrame) -> pd.DataFrame:
    """The negative-control rows of `experiment_registry`, reported apart from the FDR families (CONTROLS_NOTE).
    control_q is BH within the controls not tested on the holdout, the q 26.16 compares with fdr_q."""
    cols = ["control_q" if c == "q_value" else c for c in reg.columns if c not in ("fdr_role", "success")]
    nc = reg[reg["fdr_role"] == NEGATIVE_CONTROL]
    if nc.empty:
        return pd.DataFrame(columns=cols)
    q = np.full(len(nc), np.nan)
    live = ~nc["tested_on_holdout"].astype(bool).to_numpy()
    q[live] = M.benjamini_hochberg(nc["p_value"].to_numpy()[live])
    return nc.assign(control_q=q).reindex(columns=cols).reset_index(drop=True)


def exploratory_rows(reg: pd.DataFrame) -> pd.DataFrame:
    """The exploratory rows of `experiment_registry`, reported apart (EXPLORATORY_NOTE): raw p, no q, no success."""
    cols = [c for c in reg.columns if c not in ("fdr_role", "q_value", "success")]
    x = reg[reg["fdr_role"] == EXPLORATORY] if len(reg) else reg
    return x.reindex(columns=cols).reset_index(drop=True)


def _fdr_table(reg: pd.DataFrame) -> pd.DataFrame:
    """The 26.5 listing: every row but the negative controls and the exploratory rows, which have their own
    sections. Successes, failures and -dirty rows all stay listed (ADR 0001 Q4)."""
    return reg[~reg["fdr_role"].isin([NEGATIVE_CONTROL, EXPLORATORY])].reset_index(drop=True) if len(reg) else reg


def _unassigned_families(reg: pd.DataFrame) -> list[str]:
    """Experiment families in neither review.fdr_families nor negative_control (26.5 fails closed on them)."""
    fams = reg.loc[reg["fdr_role"] == "unassigned", "experiment_family"] if len(reg) else pd.Series(dtype=object)
    return sorted(fams.astype("string").fillna("<null>").unique().tolist())


def _c261(ev: dict[str, Any], rc: dict[str, Any]) -> tuple[dict, dict]:
    bs = ev["prediction_quality"]["bootstrap"]
    thr = {"bss_min": rc["bss_min"], "ci_level": rc["ci_level"], "resamples": rc["bootstrap_resamples"],
           "unit": "game"}
    holdout = _crit("26.1", "Final-holdout BSS vs market with game-level CI", INSUFFICIENT, None, thr,
                    HOLDOUT_NOT_EVALUATED, evaluated=False)
    title = "Walk-forward pooled BSS vs market, paired game bootstrap (informational)"
    if not bs or not bs["n_games"]:
        return holdout, _crit("26.1-wf", title, INSUFFICIENT, None, thr, "no evaluated predictions",
                              informational=True)
    ok = bs["bss"] >= rc["bss_min"] and bs["mean_delta"] > 0 and bs["ci_low"] > 0
    val = {k: bs[k] for k in ("bss", "delta_brier", "mean_delta", "ci_low", "ci_high", "p_value", "brier_model",
                              "brier_market", "n_games", "n_obs")}
    return holdout, _crit("26.1-wf", title, PASS if ok else FAIL, val, thr,
                          "walk-forward evaluation periods pooled; not a substitute for the holdout test",
                          informational=True)


def _c262(preds: pd.DataFrame, eval_ids: list[str], rc: dict[str, Any]) -> tuple[dict, pd.DataFrame]:
    title = "Beats market in most walk-forward periods; no period dominates"
    floor = float(rc["period_bss_floor"])
    thr = {"min_periods": rc["min_periods"], "min_period_win_fraction": rc["min_period_win_fraction"],
           "max_period_share": rc["max_period_share"], "period_bss_floor": floor}
    if preds.empty:
        return _crit("26.2", title, INSUFFICIENT, None, thr, "no evaluated predictions"), pd.DataFrame()
    y = preds["y"].to_numpy()
    d = (preds["p_market"].to_numpy() - y) ** 2 - (preds["p_model"].to_numpy() - y) ** 2
    game_delta = pd.DataFrame({"period_id": preds["period_id"], "game_id": preds["game_id"], "d": d}) \
        .groupby(["period_id", "game_id"], sort=False)["d"].mean()
    rows = []
    for pid in eval_ids:
        g = preds[preds["period_id"] == pid]
        if not len(g):
            continue
        bm, bk = M.brier(g["y"], g["p_model"]), M.brier(g["y"], g["p_market"])
        rows.append({"period_id": pid, "n_obs": int(len(g)), "n_games": int(g["game_id"].nunique()),
                     "brier_model": bm, "brier_market": bk, "delta_brier": bk - bm, "bss": M.bss(bm, bk),
                     # the 26.2 floor: a period counts as improved only if its BSS >= 0.001 (sign-only
                     # counting passes a null world about 11% of the time)
                     "improved": bool(M.bss(bm, bk) >= floor), "improvement": float(game_delta.loc[pid].sum())})
    tbl = pd.DataFrame(rows)
    n = len(tbl)
    total = float(tbl["improvement"].sum())
    tbl["share_of_improvement"] = tbl["improvement"] / total if total > 0 else np.nan
    pooled = M.bss(M.brier(y, preds["p_model"]), M.brier(y, preds["p_market"]))
    required = _ceil(rc["min_period_win_fraction"] * n)
    wins = int(tbl["improved"].sum())
    max_share = float(tbl["share_of_improvement"].max()) if total > 0 else None
    val = {"periods_with_data": n, "periods_improved": wins, "required": required, "pooled_bss": pooled,
           "total_improvement": total, "max_period_share": max_share}
    if n < rc["min_periods"]:
        return _crit("26.2", title, INSUFFICIENT, val, thr,
                     f"only {n} evaluation period(s) with data; at least {rc['min_periods']} are needed, so "
                     f"consistency across periods cannot be assessed"), tbl
    if total <= 0:
        return _crit("26.2", title, FAIL, val, thr, "total improvement over the market <= 0"), tbl
    fails = [m for m, bad in ((f"improved (BSS >= {floor:g}) in {wins} of {n} periods (< {required})",
                               wins < required),
                              (f"pooled BSS {pooled:.4g} <= 0", not pooled > 0),
                              (f"one period holds {max_share:.1%} of total improvement", max_share > rc["max_period_share"]))
             if bad]
    detail = "; ".join(fails) or (f"improved (BSS >= {floor:g}) in {wins} of {n} periods (>= {required}); "
                                  f"pooled BSS {pooled:.4g}; "
                                  f"largest period share {max_share:.1%}")
    return _crit("26.2", title, FAIL if fails else PASS, val, thr, detail), tbl


def _c263(inputs: ReviewInputs, reg: pd.DataFrame) -> tuple[dict, dict]:
    hold = holdout_period(inputs.periods)
    manifest = inputs.holdout_manifest or {}
    status = str(manifest.get("status") or "").upper()
    exp_hold = int(reg["tested_on_holdout"].sum()) if len(reg) else 0
    val = {"holdout_period": hold.period_id, "evaluated_holdout_rows": 0, "experiments_on_holdout": exp_hold,
           "manifest_status": status or None, "holdout_evaluated": bool(manifest.get("holdout_evaluated", False))}
    problems = []
    if status not in ("LOCKED", "FROZEN"):
        problems.append("no holdout manifest provided" if not manifest else f"manifest status is {status or 'missing'}")
    if exp_hold:
        problems.append(f"{exp_hold} experiment(s) record the holdout as test_period (excluded from this review)")
    if val["holdout_evaluated"]:
        problems.append("manifest records the holdout as evaluated")
    detail = "; ".join(problems) or (f"no evaluated row lies in {hold.period_id} ({hold.start}..{hold.end}); "
                                     f"manifest {status}")
    return _crit("26.3", "Final holdout untouched; freeze manifest in place", FAIL if problems else PASS, val,
                 {"manifest_status_in": ["LOCKED", "FROZEN"]}, detail), {"manifest": manifest}


def _utc(s: Any) -> pd.Series:
    return pd.to_datetime(pd.Series(s), utc=True, errors="coerce")


def _c264(inputs: ReviewInputs) -> tuple[dict, pd.DataFrame]:
    title = "Every feature family has a hypothesis registered before testing"
    hyps, exps = inputs.hypotheses, inputs.experiments
    if hyps is None or hyps.empty:
        return _crit("26.4", title, INSUFFICIENT, None, None, "no hypotheses registered"), pd.DataFrame()
    fam_of = dict(zip(hyps["hypothesis_id"].astype(str), hyps["family"].astype(str)))
    used = {str(f) for f in inputs.model_selection.get("feature_families") or []}
    unknown_ids: list[str] = []
    if exps is not None and len(exps):
        for hid in exps.get("hypothesis_id", pd.Series(dtype=object)).dropna().astype(str).unique():
            (used.add(fam_of[hid]) if hid in fam_of else unknown_ids.append(hid))
        for m in exps["metrics_json"].map(_parse_json) if "metrics_json" in exps else []:
            used.update(str(f) for f in m.get("feature_families") or [])
    if exps is not None and len(exps) and "created_at" in exps:
        ref, basis = _utc(exps["created_at"]).min(), "earliest experiment created_at"
    elif inputs.backtest_run.get("created_at"):
        ref, basis = _utc([inputs.backtest_run["created_at"]]).iloc[0], "backtest run created_at"
    else:
        ref, basis = pd.NaT, "no experiment or run timestamp: registration existence only"
    if not used:
        return _crit("26.4", title, INSUFFICIENT, None, None,
                     "no feature families recorded (model_selection.feature_families or experiment hypotheses)"), \
            pd.DataFrame()
    reg_at = _utc(hyps["registered_at"]) if "registered_at" in hyps else _utc([pd.NaT] * len(hyps))
    rows = []
    for fam in sorted(used):
        m = (hyps["family"].astype(str) == fam).to_numpy()
        first = reg_at[m].min() if m.any() else pd.NaT
        ok = bool(m.any() and (pd.isna(ref) or (pd.notna(first) and first < ref)))
        rows.append({"feature_family": fam, "hypotheses": int(m.sum()), "first_registered_at": first,
                     "reference_time": ref, "registered_before_testing": ok})
    tbl = pd.DataFrame(rows)
    bad = tbl.loc[~tbl["registered_before_testing"], "feature_family"].tolist()
    # Registered, but at the reference instant itself (a `quant train` run registers its hypotheses with the
    # timestamp it stamps on its experiments): not "before testing" under the strict order 26.4 checks.
    tie = tbl.loc[~tbl["registered_before_testing"] & (tbl["first_registered_at"] == tbl["reference_time"]),
                  "feature_family"].tolist()
    detail = "; ".join(filter(None, [
        f"feature families without a prior hypothesis: {bad}" if bad else "",
        f"{tie} first registered at the same instant as the reference ({basis}), not before it; register with "
        f"`python -m src.backtest.registry register` before `quant train`" if tie else "",
        f"experiments cite unregistered hypotheses: {unknown_ids}" if unknown_ids else ""]))
    val = {"feature_families_used": sorted(used), "reference_time_basis": basis, "missing": bad,
           "registered_at_reference_instant": tie, "unregistered_hypothesis_ids": unknown_ids}
    return _crit("26.4", title, FAIL if (bad or unknown_ids) else PASS, val, None,
                 detail or f"all {len(used)} feature families registered before {basis}"), tbl


def _c265(reg: pd.DataFrame, rc: dict[str, Any]) -> tuple[dict, pd.DataFrame]:
    """The FDR families' registry. The negative controls and the exploratory rows are reported in their
    own sections; exploratory rows are not counted here at all (ADR 0001 Q1). A family in none of those
    sets gets no q and fails 26.5 (fail-closed). A -dirty row in the BH set fails 26.5 and stays listed
    (ADR 0001 Q4)."""
    title = "Experiment registry with BH-FDR q-values per FDR family"
    fdr_set = [str(f) for f in rc["fdr_families"]]
    thr = {"fdr_q": rc["fdr_q"], "fdr_families": fdr_set}
    table = _fdr_table(reg)
    n_x = int((reg["fdr_role"] == EXPLORATORY).sum()) if len(reg) else 0
    counted = reg[reg["fdr_role"] != EXPLORATORY] if len(reg) else reg
    live = counted[~counted["tested_on_holdout"].astype(bool)] if len(counted) else counted
    if live.empty:
        return _crit("26.5", title, FAIL, {"experiments": 0, "exploratory_experiments": n_x}, thr,
                     "experiment registry is empty"), table
    missing = int(live["p_value"].isna().sum())
    # Spec Sec 20: every experiment records its git commit. Counted over every row, holdout ones included.
    commit = counted["git_commit"].astype("string").fillna("").str.strip()
    no_commit = int(((commit == "") | (commit == GIT_UNAVAILABLE)).sum())
    # ADR 0001 Q4: a -dirty row has no provenance for acceptance. Counted over the BH set, holdout rows included.
    dirty = int((commit.str.endswith(DIRTY) & (counted["fdr_role"] == "fdr")).sum())
    fdr = live[live["fdr_role"] == "fdr"]
    n_fams = int(fdr["experiment_family"].nunique())
    unassigned, unassigned_fams = reg[reg["fdr_role"] == "unassigned"], _unassigned_families(reg)
    n_controls = int((live["fdr_role"] == NEGATIVE_CONTROL).sum())
    val = {"experiments": int(len(live)), "fdr_families": n_fams, "fdr_experiments": int(len(fdr)),
           "negative_controls": n_controls, "successes": int(fdr["success"].sum()),
           "failures": int((~fdr["success"].astype(bool)).sum()), "missing_p_values": missing,
           "missing_git_commit": no_commit, "dirty_git_commit": dirty, "unassigned_families": unassigned_fams,
           "exploratory_experiments": n_x}
    problems = [f"{missing} experiment(s) lack a p-value; q cannot be computed"] if missing else []
    if no_commit:
        problems.append(f"{no_commit} experiment(s) have no git commit (spec Sec 20); they cannot be traced to the "
                        f"code that produced them")
    if dirty:
        problems.append(f"{dirty} experiment(s) in the BH-FDR set ran from uncommitted code (-dirty) and have no "
                        f"provenance for acceptance; they stay listed: rerun them from a clean commit (ADR 0001 Q4)")
    if unassigned_fams:
        problems.append(f"experiment_family {unassigned_fams} is in neither review.fdr_families nor "
                        f"{NEGATIVE_CONTROL}; fail-closed: no q computed for its {len(unassigned)} experiment(s)")
    if NEGATIVE_CONTROL in fdr_set:
        problems.append(f"review.fdr_families names {NEGATIVE_CONTROL}: controls are sanity checks, not "
                        f"hypotheses, and stay out of the BH-FDR set")
    if EXPLORATORY in fdr_set:
        problems.append(f"review.fdr_families names {EXPLORATORY}: exploratory rows are never in the BH-FDR set "
                        f"(ADR 0001 Q1)")
    if fdr.empty:
        problems.append("no experiment in an FDR family (review.fdr_families); no q computed")
    if problems:
        return _crit("26.5", title, FAIL, val, thr, "; ".join(problems)), table
    return _crit("26.5", title, PASS, val, thr,
                 f"q computed within {n_fams} FDR famil{'y' if n_fams == 1 else 'ies'}; {n_controls} negative "
                 f"control(s) reported separately as sanity checks, not hypotheses; successes and failures "
                 f"both listed"), table


def _c266_267(buckets: pd.DataFrame, base: pd.DataFrame, rc: dict[str, Any],
              min_edge: float) -> tuple[dict, dict, dict]:
    suff = buckets[buckets["sufficient"].astype(bool)]
    n_suff = int(len(suff))
    thr6 = {"bucket_min_games": rc["bucket_min_games"], "minimum_edge": min_edge, "min_sufficient_buckets": 3}
    title6 = "Realized edge > 0 in every sufficiently-sampled bucket at or above minimum edge"
    lower = M.num(suff["lower"].astype(object).where(suff["lower"].notna(), -np.inf)) if n_suff else np.array([])
    above = suff[lower >= min_edge - 1e-12] if n_suff else suff
    if n_suff < 3:
        c6 = _crit("26.6", title6, INSUFFICIENT, {"sufficient_buckets": n_suff}, thr6,
                   f"{n_suff} bucket(s) with >= {rc['bucket_min_games']} games (3 needed)")
    elif above.empty:
        c6 = _crit("26.6", title6, INSUFFICIENT, {"sufficient_buckets": n_suff, "sufficient_at_or_above": 0}, thr6,
                   f"no bucket at or above minimum edge {min_edge:.0%} has >= {rc['bucket_min_games']} games")
    else:
        neg = above.loc[~(M.num(above["realized_edge"]) > 0), "bucket"].tolist()
        c6 = _crit("26.6", title6, FAIL if neg else PASS,
                   {"sufficient_buckets": n_suff, "sufficient_at_or_above": int(len(above)),
                    "min_realized_edge_at_or_above": float(np.nanmin(M.num(above["realized_edge"])))}, thr6,
                   f"realized edge <= 0 in {neg}" if neg else "realized edge > 0 in every qualifying bucket")
    mono = M.edge_monotonicity(buckets, base) if len(base) else {"n_sufficient_buckets": n_suff}
    thr7 = {"monotonic_spearman_min": rc["monotonic_spearman_min"], "edge_coefficient": "> 0"}
    title7 = "Edge buckets broadly monotonic (Spearman) and predicted edge adds to market probability"
    if n_suff < 3:
        c7 = _crit("26.7", title7, INSUFFICIENT, {"sufficient_buckets": n_suff}, thr7,
                   f"{n_suff} sufficiently-sampled bucket(s) (3 needed)")
    else:
        rho, coef = mono.get("spearman", np.nan), mono.get("coef_edge", np.nan)
        ok = bool(np.isfinite(rho) and rho >= rc["monotonic_spearman_min"] and np.isfinite(coef) and coef > 0)
        c7 = _crit("26.7", title7, PASS if ok else FAIL,
                   {"spearman": rho, "coef_edge": coef, "se_edge": mono.get("se_edge"), "p_edge": mono.get("p_edge"),
                    "sufficient_buckets": n_suff}, thr7,
                   f"Spearman {rho:.3g} over {n_suff} buckets; logistic side_won ~ p_market_side + predicted_edge "
                   f"edge coefficient {coef:.3g} ({mono.get('se_method', '')} SE)")
    return c6, c7, mono


def _c268_269(ev: dict[str, Any], base: pd.DataFrame, inputs: ReviewInputs) -> tuple[dict, dict, dict]:
    b = ev["trading"].get(inputs.baseline_scenario)
    trades = base[base["decision"] != "PASS"] if len(base) else base
    fee_ids = sorted(trades["fee_schedule_id"].dropna().astype(str).unique()) if len(trades) else []
    fs = inputs.fee_schedules
    verified = dict(zip(fs["fee_schedule_id"].astype(str), fs["verified"].astype("boolean"))) \
        if fs is not None and len(fs) and "verified" in fs else {}
    fee_info = [{"fee_schedule_id": f, "verified": (None if pd.isna(verified.get(f, pd.NA)) else bool(verified[f]))}
                for f in fee_ids]
    unverified = [f["fee_schedule_id"] for f in fee_info if f["verified"] is not True]
    title8 = "Profitable after fees and slippage (baseline scenario)"
    thr8 = {"net_pnl_after_fees_slippage": "> 0", "scenario": inputs.baseline_scenario}
    if not b or b["trades"] == 0:
        c8 = _crit("26.8", title8, INSUFFICIENT, {"trades": 0, "fee_schedules": fee_info}, thr8,
                   "no baseline trades")
    else:
        val = {"trades": b["trades"], "pnl_before_fees": b["pnl_before_fees"], "pnl_after_fees": b["pnl_after_fees"],
               "pnl_after_fees_slippage": b["pnl_after_fees_slippage"], "roi": b["roi"], "fee_schedules": fee_info}
        note = f"; fee schedule(s) not verified against a venue's published schedule: {unverified}" if unverified else ""
        c8 = _crit("26.8", title8, PASS if b["net_pnl"] > 0 else FAIL, val, thr8,
                   f"net P&L after fees and slippage {b['net_pnl']:.2f} over {b['trades']} trades{note}")
    rows = [{"scenario": s, "trades": t["trades"], "gross_pnl": t["gross_pnl"], "fees": t["fees"],
             "slippage": t["slippage"], "net_pnl": t["net_pnl"], "roi": t["roi"]} for s, t in ev["trading"].items()]
    tbl = pd.DataFrame(rows)
    depth = float(np.mean(M.flag(trades["depth_verified"]))) if len(trades) and "depth_verified" in trades else None
    net = b["net_pnl"] if b else None
    pen0 = ev["trading"].get("pen_0")
    sensitive = bool(pen0 and pen0["net_pnl"] > 0 and (net is None or net <= 0))
    val9 = {"baseline_net_pnl": net, "baseline_trades": b["trades"] if b else 0, "execution_sensitive": sensitive,
            "depth_verified_share": depth,
            **{f"{r['scenario']}_net_pnl": r["net_pnl"] for r in rows}, **{f"{r['scenario']}_roi": r["roi"] for r in rows}}
    detail = "; ".join(filter(None, [
        f"{inputs.baseline_scenario} net P&L {net:.2f}" if net is not None else f"{inputs.baseline_scenario} missing",
        "execution-sensitive: profitable only at pen_0" if sensitive else "",
        f"depth verified for {depth:.1%} of baseline trades" if depth is not None else ""]))
    title9 = "Survives the spread-penalty scenarios (baseline pen_25 net > 0)"
    thr9 = {"baseline_scenario": inputs.baseline_scenario, "net_pnl": "> 0"}
    if not b or b["trades"] == 0:
        # zero trades give INSUFFICIENT SAMPLE in both 26.8 and 26.9, not FAIL in one
        c9 = _crit("26.9", title9, INSUFFICIENT, val9, thr9, f"no baseline trades; {detail}")
    else:
        c9 = _crit("26.9", title9, PASS if net > 0 else FAIL, val9, thr9, detail)
    return c8, c9, {"scenarios": tbl, "fee_schedules": fee_info}


def _c2610(inputs: ReviewInputs, preds: pd.DataFrame, base: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict, dict]:
    sp = cfg["snapshot_policy"]
    canon, max_age, max_spread = int(sp["canonical_offset_min"]), float(sp["max_quote_age_s"]), float(sp["max_spread"])
    title = "Canonical T-30 snapshot only; quote age and spread within policy"
    thr = {"canonical_offset_min": canon, "max_quote_age_s": max_age, "max_spread": max_spread}
    viol: dict[str, int] = {}
    for name, df in (("predictions", preds), ("baseline decisions", base)):
        if df.empty:
            continue
        if "snapshot_offset_min" in df:
            viol[f"{name}: offset != T-{canon}"] = int((~(M.num(df["snapshot_offset_min"]) == canon)).sum())
        if "is_canonical" in df:
            viol[f"{name}: not canonical"] = int((~M.flag(df["is_canonical"])).sum())
        for col, lim in (("quote_age_s", max_age), ("spread", max_spread)):
            if col in df:
                viol[f"{name}: {col} > {lim:g} or missing"] = int((~(M.num(df[col]) <= lim + 1e-12)).sum())
            elif name == "predictions":
                viol[f"predictions: {col} column missing"] = int(len(df))
    run_off = inputs.backtest_run.get("snapshot_offset_min")
    if run_off is not None and int(run_off) != canon:
        viol[f"backtest run offset T-{run_off}"] = 1
    ex = inputs.exclusions
    excl = {}
    if ex is not None and len(ex):
        at_canon = ex[M.num(ex["snapshot_offset_min"]) == canon] if "snapshot_offset_min" in ex else ex
        excl = {"canonical_by_reason": {str(k): int(v) for k, v in at_canon["reason"].value_counts().items()},
                "all_offsets": ex.groupby(["snapshot_offset_min", "reason"]).size().rename("n").reset_index()}
    n_unver = int((~preds["size_verified"].astype(bool)).sum()) if len(preds) else 0
    if n_unver and inputs.quote_quality == EXECUTABLE:
        viol["predictions: displayed size unverified in an executable run"] = n_unver
    val = {"evaluated_observations": int(len(preds)),
           "canonical_observations": inputs.observation_counts.get("canonical", int(len(preds))),
           "exclusions_at_canonical": sum(excl.get("canonical_by_reason", {}).values()),
           "violations": int(sum(viol.values())),
           "size_verified_share": float(1 - n_unver / len(preds)) if len(preds) else None,
           "size_unverified": n_unver}
    sup: dict[str, Any] = {"violations": viol, "exclusions": excl, "observation_counts": inputs.observation_counts}
    if preds.empty:
        return _crit("26.10", title, INSUFFICIENT, val, thr, "no evaluated observations"), sup
    if inputs.quote_quality == CANDLE_DEGRADED:
        return _crit("26.10", title, INSUFFICIENT, val, thr,
                     "not evaluable: candle_degraded quotes carry no displayed size or depth, so the 26.10 size "
                     "rule cannot be checked"), sup
    bad = {k: v for k, v in viol.items() if v}
    detail = f"violations: {bad}" if bad else (f"all {len(preds)} evaluated observations are canonical T-{canon} "
                                               f"within quote-age and spread limits, with displayed size")
    return _crit("26.10", title, FAIL if bad else PASS, val, thr, detail), sup


def _c2611(audit: dict[str, Any]) -> dict:
    title = "No look-ahead: features precede the quote; quotes precede the snapshot"
    thr = {c: 0 for c in REQUIRED_LEAKAGE_CHECKS}
    if not audit:
        return _crit("26.11", title, INSUFFICIENT, None, thr, "leakage audit not provided")
    counts = {k: (int(v) if v is not None and np.isfinite(float(v)) else None) for k, v in audit.items()}
    missing = [c for c in REQUIRED_LEAKAGE_CHECKS if counts.get(c) is None]
    bad = {k: v for k, v in counts.items() if v}
    if bad:
        return _crit("26.11", title, FAIL, counts, thr, f"leakage found: {bad}")
    if missing:
        return _crit("26.11", title, INSUFFICIENT, counts, thr, f"audit lacks required checks {missing}")
    return _crit("26.11", title, PASS, counts, thr, f"all {len(counts)} leakage counts are 0")


COVERAGE_NOTE = (
    "p_true (synthetic generator truth) governs when available for >= coverage_bin_min_obs rows. Otherwise the "
    "outcome upper bound governs, and it can only FAIL: one outcome per observation identifies the mean of p_true "
    "within a p_model bin, not its spread around that mean, so per-observation coverage is not identified from "
    "outcomes. The bound is the largest coverage consistent with the bin outcome frequencies (Wilson intervals at "
    "1 - coverage_bound_alpha / bins, game-clustered effective size). Below the band it shows under-coverage; at or "
    "above the band low it shows nothing, and over-coverage is never detectable. The retired binned-frequency proxy "
    "measured coverage of the bin mean, not of p_true, and is not reported.")


def _c2612(preds: pd.DataFrame, inputs: ReviewInputs, cfg: dict[str, Any]) -> tuple[dict, dict]:
    unc, rc = cfg["uncertainty"], cfg["review"]
    band, min_obs = [float(b) for b in unc["coverage_band"]], int(rc["coverage_bin_min_obs"])
    alpha = float(rc["coverage_bound_alpha"])
    title = "Uncertainty interval coverage within band"
    thr = {"nominal_coverage": unc["nominal_coverage"], "coverage_band": band, "coverage_bin_min_obs": min_obs,
           "coverage_bound_alpha": alpha}
    if preds.empty:
        return _crit("26.12", title, INSUFFICIENT, None, thr, "no evaluated predictions"), {}

    def bound(g: pd.DataFrame) -> dict[str, Any] | None:
        return M.coverage_upper_bound(g["y"], g["p_model"], g["lower_bound"], g["upper_bound"], g["game_id"],
                                      min_obs_per_bin=min_obs, alpha=alpha)

    lo, hi = preds["lower_bound"], preds["upper_bound"]
    n_true = int(np.isfinite(M.num(preds["p_true"])).sum()) if "p_true" in preds else 0
    cov_true = M.interval_coverage(lo, hi, preds["p_true"]) if n_true else None
    b = bound(preds)
    ub = b["bound"] if b else None
    governs = "p_true" if inputs.is_synthetic and n_true >= min_obs else "outcome_upper_bound"
    width = float(np.nanmean(M.num(hi) - M.num(lo)))
    val = {"coverage_true": cov_true, "coverage_upper_bound": ub, "governing_target": governs,
           "coverage": cov_true if governs == "p_true" else None,
           "bound_holds": (bool(cov_true <= ub + 1e-12) if cov_true is not None and ub is not None else None),
           "n_obs": int(len(preds)), "n_true": n_true, "bound_bins": b["n_bins"] if b else None,
           "bound_bins_binding": b["bins_binding"] if b else None, "mean_width": width}
    per = []
    for pid, g in preds.groupby("period_id", sort=False):
        gb = bound(g)
        per.append({"period_id": pid, "n_obs": int(len(g)),
                    "coverage_true": M.interval_coverage(g["lower_bound"], g["upper_bound"], g["p_true"])
                    if n_true else None,
                    "coverage_upper_bound": gb["bound"] if gb else None})
    sup = {"per_period": pd.DataFrame(per), "bound_bins": b["table"] if b else pd.DataFrame(), "note": COVERAGE_NOTE}
    ub_txt = f"outcome upper bound {ub:.3f}" if ub is not None else "outcome upper bound n/a"
    if governs == "p_true":
        if cov_true is None or not np.isfinite(cov_true):
            return _crit("26.12", title, INSUFFICIENT, val, thr, "p_true coverage cannot be formed"), sup
        ok = band[0] <= cov_true <= band[1]
        return _crit("26.12", title, PASS if ok else FAIL, val, thr,
                     f"coverage {cov_true:.3f} against p_true (governs); band {band[0]:.2f}-{band[1]:.2f}; "
                     f"{ub_txt} (can only fail)"), sup
    if ub is None:
        return _crit("26.12", title, INSUFFICIENT, val, thr,
                     f"outcome upper bound cannot be formed (fewer than {min_obs} observations)"), sup
    if ub < band[0]:
        return _crit("26.12", title, FAIL, val, thr,
                     f"under-coverage shown by outcomes: coverage is at most {ub:.3f} (upper bound, "
                     f"alpha {alpha:g}) < band low {band[0]:.2f}"), sup
    return _crit("26.12", title, INSUFFICIENT, val, thr,
                 f"{ub_txt} >= band low {band[0]:.2f}: outcomes cannot establish coverage (per-observation "
                 f"coverage is not identified from one outcome per observation; the bound can only fail)"), sup


def _c2613(preds: pd.DataFrame, rc: dict[str, Any]) -> tuple[dict, dict]:
    title = "No severe calibration failure (model ECE vs market, calibration slope)"
    band = [float(b) for b in rc["calibration_slope_band"]]
    thr = {"severe_calibration_ece_excess": rc["severe_calibration_ece_excess"], "calibration_slope_band": band,
           "bins": rc["calibration_bins"], "ci_level": rc["ci_level"], "resamples": rc["bootstrap_resamples"],
           "unit": "game", "slope_rule": "FAIL only if the whole slope CI lies outside calibration_slope_band"}
    if preds.empty:
        return _crit("26.13", title, INSUFFICIENT, None, thr, "no evaluated predictions"), {}
    bins, y = int(rc["calibration_bins"]), preds["y"]
    tabs = {"model": M.calibration_table(y, preds["p_model"], bins),
            "market": M.calibration_table(y, preds["p_market"], bins),
            "conservative_yes": M.calibration_table(y, preds["p_conservative_yes"], bins),
            "conservative_no": M.calibration_table(1 - y, preds["p_conservative_no"], bins)}
    cm, ck = tabs["model"], tabs["market"]
    # The slope with a game-bootstrap CI; it fails only when the whole CI lies outside the band.
    sb = M.calibration_slope_bootstrap(y, preds["p_model"], preds["game_id"], n=int(rc["bootstrap_resamples"]),
                                       seed=int(rc["seed"]), ci=float(rc["ci_level"]))
    excess, slope, lo, hi = cm["ece"] - ck["ece"], sb["slope"], sb["ci_low"], sb["ci_high"]
    # ADR 0001 Q2: one diagnostic line, no state change, when the finite CI excludes 1.0 (touching 1.0 does not).
    finite_ci = bool(np.isfinite(lo) and np.isfinite(hi))
    excludes_1 = bool(hi < 1.0 or lo > 1.0) if finite_ci else None
    val = {"ece_model": cm["ece"], "ece_market": ck["ece"], "ece_excess": excess, "mce_model": cm["mce"],
           "slope_model": slope, "slope_ci_low": lo, "slope_ci_high": hi, "slope_ci_excludes_1": excludes_1,
           "slope_ci_level": sb["ci_level"],
           "slope_resamples_valid": sb["n_valid"], "intercept_model": cm["intercept"], "slope_market": ck["slope"],
           "ece_conservative_yes": tabs["conservative_yes"]["ece"], "ece_conservative_no": tabs["conservative_no"]["ece"]}
    sup = {"calibration": tabs, "slope_bootstrap": sb}
    ci_txt = f"slope {slope:.3f} ({sb['ci_level']:.0%} CI {lo:.3f}–{hi:.3f})"
    diag = (f"; DIAGNOSTIC (no state change, ADR 0001 Q2): slope CI excludes 1.0, "
            f"{'below 1: model overconfident' if hi < 1.0 else 'above 1: model underconfident'}") if excludes_1 else ""
    fails = []
    if excess > rc["severe_calibration_ece_excess"]:
        fails.append(f"model ECE exceeds market ECE by {excess:.4f}")
    if np.isfinite(lo) and np.isfinite(hi) and (hi < band[0] or lo > band[1]):
        fails.append(f"calibration {ci_txt} wholly outside {band}")
    if fails:
        return _crit("26.13", title, FAIL, val, thr, "severe calibration failure: " + "; ".join(fails) + diag), sup
    if not (np.isfinite(slope) and np.isfinite(lo) and np.isfinite(hi)):
        return _crit("26.13", title, INSUFFICIENT, val, thr,
                     f"calibration slope could not be estimated with a CI ({ci_txt})"), sup
    outside = f"; point estimate outside {band} but the CI overlaps the band" if not band[0] <= slope <= band[1] else ""
    return _crit("26.13", title, PASS, val, thr,
                 f"ECE excess {excess:.4f}; {ci_txt}{outside} (p_conservative tables reported for yes and no "
                 f"sides){diag}"), sup


def _c2614(base: pd.DataFrame, rc: dict[str, Any]) -> tuple[dict, dict]:
    title = "P&L not concentrated in a few trades or one team"
    thr = {"concentration_top10_max": rc["concentration_top10_max"], "concentration_team_max": rc["concentration_team_max"]}
    trades = base[base["decision"] != "PASS"] if len(base) else base
    if trades.empty:
        return _crit("26.14", title, INSUFFICIENT, {"trades": 0}, thr, "no baseline trades"), {}
    c = M.concentration(trades, top10_max=rc["concentration_top10_max"], team_max=rc["concentration_team_max"],
                        edges=list(rc["bucket_edges"]))
    val = {k: c[k] for k in ("trades", "total_net_pnl", "positive_pnl", "largest_trade", "largest_trade_share",
                             "top5_share", "top10_share", "best_team", "best_team_share", "best_month",
                             "best_month_share")} | {"net_pnl_ex_top10": c["ex_top10"]["net_pnl"]}
    detail = "; ".join(c["flags"]) or ("no positive P&L to concentrate" if c["positive_pnl"] <= 0 else
                                       f"top-10 share {c['top10_share']:.1%}; best team share {c['best_team_share']:.1%}")
    detail += f"; net P&L excluding the top ten trades {c['ex_top10']['net_pnl']:.2f}"
    return _crit("26.14", title, FAIL if c["flags"] else PASS, val, thr, detail), c


def _c2615(base: pd.DataFrame, eval_ids: list[str], rc: dict[str, Any]) -> tuple[dict, dict]:
    title = "No edge decay across periods"
    thr = {"slope_ci_high": "< 0 warns", "recent_two_periods": "both below aggregate warns", "ci_level": rc["ci_level"]}
    trades = base[base["decision"] != "PASS"] if len(base) else base
    if trades.empty:
        return _crit("26.15", title, INSUFFICIENT, {"periods_with_trades": 0}, thr, "no baseline trades"), {}
    dec = M.edge_decay(trades, eval_ids, ci=float(rc["ci_level"]))
    val = {k: dec[k] for k in ("n_periods", "aggregate_realized_edge", "slope", "slope_ci_low", "slope_ci_high",
                               "recent_two_underperform", "decay_warning")}
    if dec["n_periods"] < 3:
        return _crit("26.15", title, INSUFFICIENT, val, thr, f"{dec['n_periods']} period(s) with trades (3 needed)"), dec
    detail = ("DECAY WARNING: " + "; ".join(filter(None, [
        "slope CI upper bound < 0" if np.isfinite(dec["slope_ci_high"]) and dec["slope_ci_high"] < 0 else "",
        "the two most recent periods both underperform the aggregate" if dec["recent_two_underperform"] else ""]))) \
        if dec["decay_warning"] else f"slope {dec['slope']:.4g} (CI {dec['slope_ci_low']:.4g}, {dec['slope_ci_high']:.4g})"
    return _crit("26.15", title, FAIL if dec["decay_warning"] else PASS, val, thr, detail), dec


def _c2616(controls: pd.DataFrame, rc: dict[str, Any]) -> tuple[dict, pd.DataFrame]:
    """`controls` is negative_control_rows: q is control_q, BH within the controls (not an FDR family)."""
    title = "Negative controls do not 'work'"
    thr = {"min_period_win_fraction": rc["min_period_win_fraction"], "fdr_q": rc["fdr_q"],
           "q": "control_q: BH within the negative controls not tested on the holdout"}
    nc = controls[~controls["tested_on_holdout"].astype(bool)] if len(controls) else controls
    if nc.empty:
        return _crit("26.16", title, INSUFFICIENT, {"candidates": 0}, thr, "no negative-control experiments"), pd.DataFrame()
    rows = []
    for cand, g in nc.groupby("feature_set_id", sort=True):
        pooled = g[g["test_period"].astype(str).str.lower().isin(POOLED_TEST_PERIODS)]
        per = g.drop(pooled.index)
        n = int(per["test_period"].nunique())
        wins = int((per.groupby("test_period")["delta_brier"].mean() > 0).sum())
        required = _ceil(rc["min_period_win_fraction"] * n)
        if len(pooled):
            q, q_src = float(np.nanmin(pooled["control_q"])), "pooled row"
        else:
            q, q_src = (float(np.nanmin(per["control_q"])) if per["control_q"].notna().any() else np.nan), \
                "min over period rows (no pooled row recorded)"
        win_rule = wins >= required if n else bool((M.num(pooled["delta_brier"]) > 0).any())
        rows.append({"candidate": cand, "periods": n, "periods_improved": wins, "required": required,
                     "pooled_q": q, "q_source": q_src, "works": bool(win_rule and np.isfinite(q) and q < rc["fdr_q"])})
    tbl = pd.DataFrame(rows)
    works = tbl.loc[tbl["works"], "candidate"].tolist()
    return _crit("26.16", title, FAIL if works else PASS, {"candidates": int(len(tbl)), "working": works}, thr,
                 f"negative control(s) beat the reference: {works}" if works
                 else f"none of {len(tbl)} negative control(s) passes both the period and FDR rules"), tbl


def _c2617(preds: pd.DataFrame, rc: dict[str, Any]) -> tuple[dict, dict]:
    title = "Real result lies in the favorable tail of the outcome-shuffle distribution"
    thr = {"shuffle_p_max": rc["shuffle_p_max"], "permutations": rc["shuffle_permutations"]}
    if preds.empty:
        return _crit("26.17", title, INSUFFICIENT, None, thr, "no evaluated predictions"), {}
    s = M.shuffle_test(preds, n_perm=int(rc["shuffle_permutations"]), seed=int(rc["seed"]))
    val = {k: s[k] for k in ("observed_delta_brier", "p_value", "perm_mean", "perm_sd", "n_perm", "n_groups")}
    return _crit("26.17", title, PASS if s["p_value"] <= rc["shuffle_p_max"] else FAIL, val, thr,
                 f"p = {s['p_value']:.4g}; {s['method']}"), s


def _c2618(inputs: ReviewInputs, rc: dict[str, Any], ev: dict[str, Any], base: pd.DataFrame) -> tuple[dict, dict]:
    title = "Prospective paper trading consistent with the backtest"
    thr = {"paper_min_trades": rc["paper_min_trades"], "paper_min_days": rc["paper_min_days"],
           "bss_ratio_min": 0.5, "net_edge_ratio_min": 0.5, "trade_frequency_ratio": [0.5, 1.5]}
    pt, ps = inputs.paper_trades, inputs.paper_settlements
    if pt is None or pt.empty or ps is None or ps.empty:
        return _crit("26.18", title, INSUFFICIENT, {"settled_trades": 0, "days": 0}, thr,
                     "no settled paper trades"), {}
    hold = holdout_period(inputs.periods)
    frozen = str((inputs.holdout_manifest or {}).get("status") or "").upper() == "FROZEN"
    # Qualifying decisions, passes included: every one settles with its outcome, so the paper Brier covers
    # all of them, like the backtest's out-of-sample BSS. Trade count, edge and frequency use the BUYs.
    qd = pt[pt["freeze_manifest_hash"].notna()]
    excluded = 0
    if not frozen and len(qd):
        in_hold = _in_window(qd["decision_ts"], hold.start, hold.end)
        excluded, qd = int(in_hold.sum()), qd[~in_hold]
    s = ps[ps["result"].isin(["yes", "no"])].drop_duplicates("paper_trade_id", keep="last")
    qd = qd.merge(s[["paper_trade_id", "outcome", "net_pnl"]], on="paper_trade_id", how="inner")
    q = qd[qd["decision"] != "PASS"]
    days_s = _dates(q["decision_ts"]) if len(q) else pd.Series(dtype="datetime64[ns]")
    days = int((days_s.max() - days_s.min()).days + 1) if len(q) else 0
    val: dict[str, Any] = {"settled_trades": int(len(q)), "settled_decisions": int(len(qd)), "days": days,
                           "excluded_holdout_window_without_freeze": excluded}
    if len(q) < rc["paper_min_trades"] or days < rc["paper_min_days"]:
        return _crit("26.18", title, INSUFFICIENT, val, thr,
                     f"{len(q)} settled qualifying trades over {days} days "
                     f"(need {rc['paper_min_trades']} and {rc['paper_min_days']})"), val
    y = M.num(qd["outcome"])
    mid = (M.num(qd["bid"]) + M.num(qd["ask"])) / 2
    paper_bss = M.bss(M.brier(y, qd["model_probability"]), M.brier(y, mid))
    bt = ev["prediction_quality"]["bootstrap"]
    bt_bss = bt["bss"] if bt else np.nan
    realized = float(np.nanmean(M.num(q["net_pnl"]) / M.num(q["hypothetical_size"]).clip(min=1e-12)))
    bt_trades = base[base["decision"] != "PASS"] if len(base) else base
    expected = float(np.nanmean(M.num(bt_trades["net_pnl"]) / M.num(bt_trades["quantity"]))) if len(bt_trades) else np.nan
    bt_days = _dates(base["official_date"]).nunique() if len(base) and "official_date" in base else 0
    freq_bt = len(bt_trades) / bt_days if bt_days else np.nan
    qual_days = _dates(qd["decision_ts"]).nunique()   # days with a qualifying slate
    freq_paper = len(q) / qual_days if qual_days else np.nan
    ratio = freq_paper / freq_bt if freq_bt else np.nan
    val |= {"paper_bss": paper_bss, "backtest_bss": bt_bss, "realized_net_edge": realized,
            "expected_net_edge": expected, "trades_per_day_paper": freq_paper, "trades_per_day_backtest": freq_bt,
            "trade_frequency_ratio": ratio}
    checks = {"paper BSS >= 50% of backtest BSS and > 0":
                  bool(np.isfinite(paper_bss) and paper_bss > 0 and np.isfinite(bt_bss) and paper_bss >= 0.5 * bt_bss),
              "realized net edge >= 50% of expected":
                  bool(np.isfinite(realized) and np.isfinite(expected) and realized >= 0.5 * expected),
              "trade frequency within 50-150% of expected": bool(np.isfinite(ratio) and 0.5 <= ratio <= 1.5)}
    failed = [k for k, ok in checks.items() if not ok]
    return _crit("26.18", title, FAIL if failed else PASS, val, thr,
                 f"failed: {failed}" if failed else "all three paper checks hold"), val | {"checks": checks}


# ADR 0002: a run without a clean commit keeps every rejection reason, prefixed, and names the flag.
HELD_BACK = "held back by the commit cap (REJECTED on a clean commit): "
BLOCKING_FLAG = "blocking flag raised: "
FLAG_RAISE, FLAG_CLEAR = "raise", "clear"


def negative_control_flag(criteria: Mapping[str, dict[str, Any]], run_commit: str | None) -> str | None:
    """ADR 0002: FLAG_RAISE when 26.16 FAILs on a run without a clean commit; FLAG_CLEAR when 26.16
    PASSes on a clean one (it clears flags raised on earlier runs, src/backtest/review_flags.py);
    otherwise None. 26.16 INSUFFICIENT (no controls) neither raises nor clears."""
    state = (criteria.get("26.16") or {}).get("state")
    clean = is_clean_commit(run_commit)
    if state == FAIL and not clean:
        return FLAG_RAISE
    if state == PASS and clean:
        return FLAG_CLEAR
    return None


FEEDING_TABLES = ("experiments", "hypotheses", "model_runs")


def lineage_scope(lineage: pd.DataFrame | None, feeding_runs: pd.DataFrame | None,
                  experiment_ids: Sequence[str]) -> dict[str, Any]:
    """Which rows the feeding-rows cap and 26.16 read for the run under review.

    LINEAGE when every model run feeding the run has train_lineage rows and every experiment those invocations
    logged is among `experiment_ids` (the experiments read): the scope is then the experiments and hypotheses of
    the invocations that wrote the feeding model runs. Otherwise LEGACY_LINEAGE_FALLBACK, with the reason: the
    whole logs, as before (fail-closed). A legacy run and a new run that lost its lineage look the same here;
    both fall back. `experiment_ids` and `hypothesis_ids` are None in the fallback."""
    cols = ["invocation_id", "kind", "row_id"]
    lin = lineage.reindex(columns=cols) if lineage is not None and len(lineage) else pd.DataFrame(columns=cols)
    lin = lin.astype(str)
    runs = set() if feeding_runs is None or feeding_runs.empty else set(feeding_runs["row_id"].astype(str))
    invs = sorted(set(lin.loc[(lin["kind"] == "model_run") & lin["row_id"].isin(runs), "invocation_id"]))
    mine = lin[lin["invocation_id"].isin(invs)]
    ids = {k: sorted(set(mine.loc[mine["kind"] == k, "row_id"])) for k in LINEAGE_KINDS}
    missing = runs - set(ids["model_run"])
    unread = set(ids["experiment"]) - {str(e) for e in experiment_ids}
    if not runs:
        reason = "no model run feeds the run under review"
    elif missing == runs:
        reason = f"none of the {len(runs)} feeding model runs has lineage (trained before lineage was recorded)"
    elif missing:
        reason = f"{len(missing)} of the {len(runs)} feeding model runs have no lineage"
    elif unread:
        reason = f"{len(unread)} experiments in the lineage are not in the experiments read"
    else:
        return {"mode": LINEAGE, "reason": None, "note": LINEAGE_NOTE, "invocations": invs,
                "rows": {k: len(ids[k]) for k in LINEAGE_KINDS},
                "experiment_ids": ids["experiment"], "hypothesis_ids": ids["hypothesis"]}
    return {"mode": LEGACY_LINEAGE_FALLBACK, "reason": reason, "note": FALLBACK_NOTE, "invocations": invs,
            "rows": {k: len(ids[k]) for k in LINEAGE_KINDS}, "experiment_ids": None, "hypothesis_ids": None}


def _in_lineage(df: pd.DataFrame, id_col: str, ids: Sequence[str] | None) -> pd.DataFrame:
    """`df` restricted to the lineage's ids; all of it in the fallback (ids None)."""
    if ids is None or df is None or df.empty or id_col not in df:
        return df
    return df[df[id_col].astype(str).isin(set(ids))].reset_index(drop=True)


def feeding_rows(reg: pd.DataFrame, hypotheses: pd.DataFrame, feeding_runs: pd.DataFrame) -> pd.DataFrame:
    """The rows feeding the run under review, with
    each row's commit and whether it is clean. They are the experiment rows (`experiment_registry`) in every
    fdr_role but exploratory, which no criterion counts (ADR 0001 Q1); every hypotheses row; and the model
    runs of `feeding_runs`. A commit that cannot be read (no git_commit column, NULL, NaN, blank, padded) is
    not clean (fail-closed)."""
    cols = ["table", "role", "row_id", "git_commit", "clean"]
    exps = reg[reg["fdr_role"] != EXPLORATORY] if len(reg) else reg
    parts = []
    for table, df, id_col, role in (("experiments", exps, "experiment_id", "fdr_role"),
                                    ("hypotheses", hypotheses, "hypothesis_id", None),
                                    ("model_runs", feeding_runs, "row_id", None)):
        if df is None or df.empty:
            continue
        commit = df["git_commit"] if "git_commit" in df else pd.Series(None, index=df.index, dtype=object)
        parts.append(pd.DataFrame({"table": table, "role": df[role].astype(object) if role else None,
                                   "row_id": df[id_col].astype(object),
                                   "git_commit": commit.astype(object)}, index=df.index))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=bool if c == "clean" else object) for c in cols})
    out = pd.concat(parts, ignore_index=True)
    out["clean"] = [is_clean_commit(c) for c in out["git_commit"]]     # the raw value
    out["git_commit"] = pd.Series([None if pd.isna(c) else str(c) for c in out["git_commit"]], dtype=object)
    return out[cols]


def dirty_input_lines(rows: pd.DataFrame) -> list[str]:
    """One line per FEEDING_TABLES table with a row that is not clean: how many of its rows, and for the
    experiments, how many per fdr_role."""
    lines = []
    for table in FEEDING_TABLES:
        mine = rows[rows["table"] == table]
        bad = mine[~mine["clean"].astype(bool)]
        if bad.empty:
            continue
        line = f"{table}: {len(bad)} of {len(mine)} rows"
        if table == "experiments":
            line += " (" + ", ".join(f"{r} {n}" for r, n in sorted(bad["role"].value_counts().items())) + ")"
        lines.append(line)
    return lines


def decide_status(criteria: Mapping[str, dict[str, Any]], *, baseline_trades: int,
                  bucket_min_games: int, run_commit: str | None, dirty_inputs: Sequence[str],
                  quote_quality: str = EXECUTABLE) -> tuple[str, list[str]]:
    """26.19 status, mechanically, plus the reasons that decided it.

    Nothing above INCONCLUSIVE may rest on non-executable quotes. The quote cap never hides
    a rejection. A run whose commit does not pin the code (`run_commit` -dirty, missing or unavailable)
    gets no terminal status at all (ADR 0002, superseding ADR 0001 Q4's "neither cap hides a rejection"):
    it is INCONCLUSIVE, and each reason that would reject it on a clean commit is kept, prefixed
    HELD_BACK. A 26.16 FAIL on such a run also names the blocking flag it raises. `dirty_inputs`
    (`dirty_input_lines`) names the rows feeding the run that have no clean commit, negative controls
    included; any one caps a clean run the same way. A clean retrain logs new rows under
    their own lineage, so a run reviewed in LINEAGE mode is checked against those alone. The
    flag reads the run's own commit
    only. Neither `run_commit` nor `dirty_inputs` has a default: every caller states the provenance of the
    run it reviews and of what fed it."""
    st = {k: v["state"] for k, v in criteria.items()}
    rejected = []
    c1 = criteria.get("26.1", {})
    if c1.get("evaluated"):
        b = (c1.get("value") or {}).get("bss")
        if b is None or not np.isfinite(b) or b <= 0:
            rejected.append("26.1: holdout evaluated with BSS <= 0")
    for cid, why in (("26.13", "severe calibration failure"), ("26.11", "leakage"),
                     ("26.16", "a negative control works")):
        if st.get(cid) == FAIL:
            rejected.append(f"{cid} FAIL ({why})")
    if st.get("26.9") == FAIL and baseline_trades >= bucket_min_games:
        rejected.append(f"26.9 FAIL with {baseline_trades} baseline trades (>= {bucket_min_games})")
    run_clean = is_clean_commit(run_commit)
    clean = run_clean and not dirty_inputs
    if rejected and clean:
        return "REJECTED", rejected
    caps = [f"capped at INCONCLUSIVE: quote_quality is {quote_quality}, not executable"
            ] if quote_quality != EXECUTABLE else []
    if not run_clean:
        caps.append(f"capped at INCONCLUSIVE: the run under review has no clean commit ({tree_state(run_commit)}); "
                    f"rerun it from a clean tree (ADR 0001 Q4, ADR 0002)")
    if dirty_inputs:
        caps.append(f"capped at INCONCLUSIVE: rows feeding the run under review have no clean commit "
                    f"({'; '.join(dirty_inputs)}); retrain from a clean tree")
    if not clean:
        caps += [HELD_BACK + r for r in rejected]
        if negative_control_flag(criteria, run_commit) == FLAG_RAISE:
            caps.append(BLOCKING_FLAG + "26.16 FAIL on a run without a clean commit; `quant freeze` refuses, and "
                        "hypothesis registration runs once, for the clearing run, until a review of a later run "
                        "from a clean commit has 26.16 PASS (ADR 0002)")
    if caps:
        return "INCONCLUSIVE", caps
    leak = [f"leakage flag: {cid} {st.get(cid, 'missing')}" for cid, bad in
            (("26.11", st.get("26.11") != PASS), ("26.16", st.get("26.16") == FAIL), ("26.3", st.get("26.3") != PASS))
            if bad]
    unmet = [f"{cid} {st.get(cid, 'missing')}" for cid in PROMISING_REQUIRES if st.get(cid) != PASS]
    if unmet or leak:
        return "INCONCLUSIVE", [f"not RESEARCH-PROMISING: {r}" for r in unmet + leak]
    if st.get("26.18") == PASS:
        return "PROSPECTIVELY-VALIDATED", [f"{', '.join(PROMISING_REQUIRES)} and 26.18 PASS; no leakage flags"]
    return "RESEARCH-PROMISING", [f"{', '.join(PROMISING_REQUIRES)} PASS; no leakage flags",
                                  f"not PROSPECTIVELY-VALIDATED: 26.18 {st.get('26.18', 'missing')}"]


def _ablation(reg_src: pd.DataFrame) -> pd.DataFrame:
    """Sec 19: per ablation candidate and period, deltas as ref - cand (positive = candidate
    better) for Brier, log loss and ECE; ROI as cand - ref. Pooled = the recorded pooled
    row, else the n_obs-weighted mean of period rows."""
    if reg_src is None or reg_src.empty or "experiment_family" not in reg_src:
        return pd.DataFrame()
    ab = reg_src[reg_src["experiment_family"] == "ablation"]
    rows = []
    for _, r in ab.iterrows():
        m = _parse_json(r.get("metrics_json"))
        f = lambda a, b: (float(m[a]) - float(m[b])) if m.get(a) is not None and m.get(b) is not None else np.nan
        rows.append({"candidate": r.get("feature_set_id"), "reference": m.get("reference"),
                     "test_period": r.get("test_period"), "n_obs": m.get("n_obs"),
                     "delta_brier": f("brier_ref", "brier_cand"), "delta_log_loss": f("logloss_ref", "logloss_cand"),
                     "delta_ece": f("ece_ref", "ece_cand"), "delta_roi": f("roi_cand", "roi_ref"),
                     "p_value": r.get("p_value"), "source": "recorded"})
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    out = []
    for cand, g in tbl.groupby("candidate", sort=True):
        pooled = g[g["test_period"].astype(str).str.lower().isin(POOLED_TEST_PERIODS)]
        out.append(g.drop(pooled.index))
        if len(pooled):
            out.append(pooled)
        else:
            w = M.num(g["n_obs"])
            w = np.where(np.isfinite(w), w, 0)
            agg = {c: float(np.average(M.num(g[c]), weights=w)) if w.sum() > 0 else float(np.nanmean(M.num(g[c])))
                   for c in ("delta_brier", "delta_log_loss", "delta_ece", "delta_roi")}
            out.append(pd.DataFrame([{"candidate": cand, "reference": g["reference"].iloc[0], "test_period": "pooled",
                                      "n_obs": int(w.sum()), **agg, "p_value": np.nan,
                                      "source": "n_obs-weighted mean of period rows"}]))
    return pd.concat(out, ignore_index=True)


def _yes_outcome(d: pd.DataFrame) -> np.ndarray:
    """YES outcome per decision row: `outcome`, else derived from side_won and side."""
    won = M.num(d["side_won"]) if "side_won" in d else np.full(len(d), np.nan)
    is_no = d["side"].eq("no").fillna(False).to_numpy(dtype=bool) if "side" in d else np.zeros(len(d), bool)
    derived = np.where(is_no, 1 - won, won)
    return np.where(np.isfinite(M.num(d["outcome"])), M.num(d["outcome"]), derived) if "outcome" in d else derived


def _sensitivity(inputs: ReviewInputs, eval_ids: list[str], base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frames = {**{int(k): v for k, v in inputs.sensitivity.items()}}
    canon_off = int(inputs.backtest_run.get("snapshot_offset_min") or 30)
    for off in sorted(set(frames) | {canon_off}, reverse=True):
        if off == canon_off:
            d, label = base, "canonical (decision snapshot)"
        else:
            d, _ = _prep_decisions(frames[off], eval_ids, label=f"T-{off}")
            d = d[d["scenario"] == inputs.baseline_scenario] if len(d) else d
            label = "sensitivity only"
        if d.empty:
            rows.append({"offset_min": off, "role": label, "observations": 0})
            continue
        t = M.trading_summary(d)
        y = _yes_outcome(d)
        ok = np.isfinite(y)
        bm, bk = M.brier(y[ok], M.num(d["p_model"])[ok]), M.brier(y[ok], M.num(d["p_market"])[ok])
        rows.append({"offset_min": off, "role": label, "observations": t["observations"], "trades": t["trades"],
                     "brier_model": bm, "brier_market": bk, "bss": M.bss(bm, bk), "net_pnl": t["net_pnl"],
                     "roi": t["roi"], "avg_realized_edge": t["avg_realized_edge"]})
    return pd.DataFrame(rows)


def _inactive_hypotheses(hyps: pd.DataFrame) -> pd.DataFrame:
    """26.5: paused and disabled hypotheses stay registered and are listed with their reason."""
    cols = ["hypothesis_id", "label", "family", "status", "status_reason", "status_decision"]
    if hyps is None or hyps.empty or "status" not in hyps:
        return pd.DataFrame(columns=cols)
    return hyps.loc[hyps["status"].fillna("ACTIVE") != "ACTIVE", cols].reset_index(drop=True)


def build_review(inputs: ReviewInputs, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the Sec 26 rules and return the review (see module docstring)."""
    strategy_family(cfg)      # Sec 26 reviews outcome_residual only; market_behavior (26.20) is refused
    rc = cfg["review"]
    preds, dec, eval_ids, notes = _prepare(inputs)
    ev = _evaluation(inputs, cfg, preds, dec, eval_ids, notes)
    base = dec[dec["scenario"] == inputs.baseline_scenario] if len(dec) else dec
    hold = holdout_period(inputs.periods)
    reg = experiment_registry(inputs.experiments, fdr_q=float(rc["fdr_q"]), fdr_families=rc["fdr_families"],
                              holdout_id=hold.period_id)
    scope = lineage_scope(inputs.lineage, inputs.feeding_runs, reg["experiment_id"].astype(str).tolist())
    lin_reg = _in_lineage(reg, "experiment_id", scope["experiment_ids"])
    lin_hyps = _in_lineage(inputs.hypotheses, "hypothesis_id", scope["hypothesis_ids"])
    controls, exploratory = negative_control_rows(lin_reg), exploratory_rows(reg)
    min_edge = float((cfg.get("decision") or {}).get("minimum_edge", 0.0))

    c1, c1wf = _c261(ev, rc)
    c2, t2 = _c262(preds, eval_ids, rc)
    c3, s3 = _c263(inputs, reg)
    c4, t4 = _c264(inputs)
    c5, t5 = _c265(reg, rc)
    c6, c7, mono = _c266_267(ev["edge_buckets"], base, rc, min_edge)
    c8, c9, s9 = _c268_269(ev, base, inputs)
    c10, s10 = _c2610(inputs, preds, base, cfg)
    c11 = _c2611(inputs.leakage_audit)
    c12, s12 = _c2612(preds, inputs, cfg)
    c13, s13 = _c2613(preds, rc)
    c14, s14 = _c2614(base, rc)
    c15, s15 = _c2615(base, eval_ids, rc)
    c16, t16 = _c2616(controls, rc)
    c17, s17 = _c2617(preds, rc)
    c18, s18 = _c2618(inputs, rc, ev, base)
    criteria = [c1, c1wf, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18]
    by_id = {c["id"]: c for c in criteria}
    baseline_trades = int(ev["trading"].get(inputs.baseline_scenario, {}).get("trades", 0))
    feeding = feeding_rows(lin_reg, lin_hyps, inputs.feeding_runs)
    status, reasons = decide_status(by_id, baseline_trades=baseline_trades,
                                    bucket_min_games=int(rc["bucket_min_games"]),
                                    run_commit=inputs.backtest_run.get("git_commit"),
                                    dirty_inputs=dirty_input_lines(feeding), quote_quality=inputs.quote_quality)
    return {
        "artifact": "quant review", "spec": "docs/PROTOCOL.md (Sec 26) and Sec 27", "review_version": REVIEW_VERSION,
        "referee_version": REFEREE_VERSION, "referee_config_hash": referee_config_hash(cfg),
        "sport": inputs.sport, "dataset": inputs.dataset, "is_synthetic": inputs.is_synthetic,
        "banner": BANNER if inputs.is_synthetic else None,
        "quote_quality": inputs.quote_quality,
        "execution_banner": DEGRADED_EXECUTION if inputs.quote_quality == CANDLE_DEGRADED else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backtest_run": inputs.backtest_run, "model_type": inputs.backtest_run.get("model_type"),
        "baseline_scenario": inputs.baseline_scenario,
        "review_config": {k: cfg.get(k) for k in ("review", "snapshot_policy", "uncertainty", "decision", "execution")},
        "periods": [asdict(p) for p in inputs.periods],
        "holdout_period": {"period_id": hold.period_id, "start": hold.start, "end": hold.end,
                           "evaluated": False, "note": HOLDOUT_NOT_EVALUATED},
        "data_notes": notes,
        "lineage": {k: scope[k] for k in ("mode", "reason", "note", "invocations", "rows")},
        "criteria": criteria,
        "negative_controls": {"label": CONTROLS_LABEL, "note": CONTROLS_NOTE, "experiments": controls},
        "exploratory": {"label": EXPLORATORY_LABEL, "note": EXPLORATORY_NOTE, "experiments": exploratory},
        "supporting": {
            "26.2": {"per_period": t2}, "26.3": s3, "26.4": {"feature_families": t4},
            "26.5": {"registry": t5, "hypotheses_not_active": _inactive_hypotheses(inputs.hypotheses)},
            "26.6": {"edge_buckets": ev["edge_buckets"]}, "26.7": {"monotonicity": mono}, "26.9": s9,
            "26.10": s10, "26.12": s12, "26.13": s13, "26.14": {"concentration": s14},
            "26.15": {"edge_decay": s15}, "26.16": {"negative_controls": t16}, "26.17": {"shuffle": s17},
            "26.18": {"paper": s18},
        },
        "informational": {
            "prediction_quality": ev["prediction_quality"], "trading": ev["trading"],
            "edge_buckets": ev["edge_buckets"], "ablation": _ablation(inputs.experiments),
            "model_selection": inputs.model_selection, "sensitivity": _sensitivity(inputs, eval_ids, base),
        },
        "feeding_rows": {"checked": {t: int((feeding["table"] == t).sum()) for t in FEEDING_TABLES},
                         "not_clean": feeding.loc[~feeding["clean"].astype(bool),
                                                  ["table", "role", "row_id", "git_commit"]].to_dict(orient="records")},
        "status": status, "status_reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Serialization and rendering
# ---------------------------------------------------------------------------

def _jsonable(o: Any) -> Any:
    """Strict-JSON value: NaN/inf -> None, numpy -> Python, DataFrame -> records, dates -> ISO."""
    if isinstance(o, pd.DataFrame):
        return [_jsonable(r) for r in o.to_dict(orient="records")]
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Period):
        return _jsonable(asdict(o))
    if o is None or o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, np.datetime64):
        return None if np.isnat(o) else pd.Timestamp(o).isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, str):
        return o
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def to_json(obj: Any) -> str:
    return json.dumps(_jsonable(obj), indent=2, allow_nan=False) + "\n"


def _fmt(v: Any) -> str:
    v = _jsonable(v)
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1000 else f"{v:.4g}"
    if isinstance(v, (list, dict)):
        s = json.dumps(v)
        return s if len(s) <= 80 else s[:77] + "..."
    return str(v)


def _short(v: Any, limit: int = 4) -> str:
    if isinstance(v, dict):
        items = [(k, x) for k, x in v.items() if not isinstance(x, (dict, list, pd.DataFrame))][:limit]
        return ", ".join(f"{k}={_fmt(x)}" for k, x in items)
    return _fmt(v)


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _md_table(df: pd.DataFrame, max_rows: int = 60) -> list[str]:
    if df is None or df.empty:
        return ["_(empty)_", ""]
    cols = list(df.columns)
    lines = ["| " + " | ".join(_md_escape(str(c)) for c in cols) + " |", "|" + "---|" * len(cols)]
    for rec in df.head(max_rows).to_dict(orient="records"):
        lines.append("| " + " | ".join(_md_escape(_fmt(rec[c])) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"_({len(df) - max_rows} more rows in the JSON artifact)_")
    return lines + [""]


def _md_obj(name: str, obj: Any, level: int) -> list[str]:
    head = "#" * min(level, 6) + " " + name
    if isinstance(obj, pd.DataFrame):
        return [head, "", *_md_table(obj)]
    if isinstance(obj, dict):
        scalars = {k: v for k, v in obj.items() if not isinstance(v, (dict, pd.DataFrame, list)) or
                   (isinstance(v, list) and not any(isinstance(x, dict) for x in v))}
        lines = [head, ""]
        if scalars:
            lines += _md_table(pd.DataFrame({"field": list(scalars), "value": [_fmt(v) for v in scalars.values()]}))
        for k, v in obj.items():
            if k in scalars or v is None:
                continue
            if isinstance(v, list):
                v = pd.DataFrame([x for x in v if isinstance(x, dict)])
            lines += _md_obj(f"{name}: {k}", v, level + 1)
        return lines
    return [head, "", _fmt(obj), ""]


def _flag_lines(review: dict[str, Any]) -> list[str]:
    """ADR 0002 flags as text lines; [] when the review carries no review_flags (not recorded)."""
    rf = review.get("review_flags")
    if rf is None:
        return []
    did = {FLAG_RAISE: "raised", FLAG_CLEAR: "cleared"}.get(rf.get("event"), "")
    lines = [f"this review {did} {r['flag_id']} ({r['backtest_run_id']})" for r in rf.get("recorded") or []]
    if not lines:
        lines = ["this review raised or cleared no flag"]
    opened = rf.get("open") or []
    lines.append(f"open flags: {len(opened)}" + ("; `quant freeze` refuses, and hypothesis registration runs once, "
                                                 "for the clearing run, until a review of a later clean run has "
                                                 "26.16 PASS" if opened else ""))
    lines += [f"open: {o['flag_id']}: 26.16 FAIL on {o['backtest_run_id']} ({o.get('git_commit')!r}, run created "
              f"{o.get('run_created_at')})" for o in opened]
    return lines


def _lineage_text(review: dict[str, Any]) -> str:
    lin = review.get("lineage") or {}
    mode = lin.get("mode") or "n/a"
    if mode == LINEAGE:
        return f"{mode} ({', '.join(lin.get('invocations') or [])}): {lin.get('note')}"
    return f"{mode} ({lin.get('reason')}): {lin.get('note')}" if lin else mode


def render_markdown(review: dict[str, Any]) -> str:
    """Markdown review. The final line is exactly `STATUS: <STATE>`."""
    lines = [f"# Review: {review['sport'].upper()} ({review['dataset']})", ""]
    banner = [line for key in ("execution_banner", "banner") if review.get(key)
              for line in (f"> **{review[key]}**", "")]
    lines += banner
    run = review.get("backtest_run") or {}
    hold = review["holdout_period"]
    lines += [f"- Generated: {review['generated_at']}",
              f"- Referee: {review.get('referee_version') or 'n/a'}, config hash "
              f"{review.get('referee_config_hash') or 'n/a'}",
              f"- Backtest run: {run.get('backtest_run_id', 'n/a')}",
              f"- Model type: {review.get('model_type') or 'n/a'}",
              f"- Quote quality: {review.get('quote_quality') or EXECUTABLE}",
              f"- Baseline scenario: {review['baseline_scenario']}",
              f"- Lineage: {_lineage_text(review)}",
              f"- Holdout: {hold['period_id']} ({_fmt(hold['start'])} to {_fmt(hold['end'])}): {hold['note']}", ""]
    for n in review.get("data_notes") or []:
        lines.append(f"- Note: {n}")
    lines += ["", "## Criteria (Sec 26)", ""]
    crit = pd.DataFrame([{"id": c["id"], "criterion": c["title"], "state": c["state"], "value": _short(c["value"]),
                          "detail": c["detail"]} for c in review["criteria"]])
    lines += _md_table(crit, max_rows=100)
    lines += ["## Status rules applied (26.19)", "", *[f"- {r}" for r in review["status_reasons"]], ""]
    flags = _flag_lines(review)
    if flags:
        lines += ["## Review flags (ADR 0002)", "", *[f"- {f}" for f in flags], ""]
    nc = review.get("negative_controls")
    if nc:
        lines += [f"## {nc['label']}", "", nc["note"], "", *_md_table(pd.DataFrame(nc["experiments"]))]
    ex = review.get("exploratory")
    if ex:
        lines += [f"## {ex['label']}", "", ex["note"], "", *_md_table(pd.DataFrame(ex["experiments"]))]
    lines += ["## Supporting tables", ""]
    for cid, sup in review["supporting"].items():
        lines += _md_obj(cid, sup, 3)
    lines += ["## Informational", ""]
    for key, obj in review["informational"].items():
        lines += _md_obj(key, obj, 3)
    lines += banner
    lines.append(f"STATUS: {review['status']}")
    return "\n".join(lines) + "\n"


def write_review_artifacts(review: dict[str, Any], out_dir: Path | str, *, experiments: pd.DataFrame,
                           hypotheses: pd.DataFrame, holdout_manifest: dict[str, Any]) -> dict[str, Path]:
    """Write <sport>_review.json, <sport>_review.md, experiment_registry.json, holdout_manifest.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rv = review["review_config"]["review"]
    fdr_q, fdr_set = float(rv["fdr_q"]), [str(f) for f in rv["fdr_families"]]
    reg = experiment_registry(experiments, fdr_q=fdr_q, fdr_families=fdr_set,
                              holdout_id=review["holdout_period"]["period_id"])
    table = _fdr_table(reg)
    fdr = table[table["fdr_role"] == "fdr"] if len(table) else table
    registry = {
        "generated_at": review["generated_at"], "sport": review["sport"], "dataset": review["dataset"],
        "is_synthetic": review["is_synthetic"], "fdr_q": fdr_q, "fdr_family_set": fdr_set,
        "note": "every experiment is listed, successes and failures; q = Benjamini-Hochberg within each "
                "experiment_family in review.fdr_families (the FDR families); a family in neither that set nor "
                "negative_control gets no q (unassigned_families); the negative controls are listed apart, as "
                "sanity checks, not hypotheses, and the exploratory rows apart, outside the BH-FDR set and never "
                "counted toward any acceptance criterion (ADR 0001 Q1)",
        "hypotheses": hypotheses if hypotheses is not None else [],
        "experiments": table,
        "fdr_families": [{"experiment_family": f, "experiments": int(len(g)), "successes": int(g["success"].sum()),
                          "failures": int(len(g) - g["success"].sum())}
                         for f, g in fdr.groupby("experiment_family", sort=True)] if len(fdr) else [],
        "unassigned_families": _unassigned_families(reg),
        "negative_controls": {"label": CONTROLS_LABEL, "note": CONTROLS_NOTE,
                              "experiments": negative_control_rows(reg)},
        "exploratory": {"label": EXPLORATORY_LABEL, "note": EXPLORATORY_NOTE, "experiments": exploratory_rows(reg)},
    }
    paths = {"review_json": out / f"{review['sport']}_review.json", "review_md": out / f"{review['sport']}_review.md",
             "experiment_registry": out / "experiment_registry.json", "holdout_manifest": out / "holdout_manifest.json"}
    paths["review_json"].write_text(to_json(review))
    paths["review_md"].write_text(render_markdown(review))
    paths["experiment_registry"].write_text(to_json(registry))
    paths["holdout_manifest"].write_text(to_json(holdout_manifest or {}))
    return paths


_STYLE = {PASS: "green", FAIL: "red", INSUFFICIENT: "yellow"}


def _rich_table(title: str, df: pd.DataFrame, max_rows: int = 30) -> Table:
    t = Table(title=title, show_lines=False)
    for c in df.columns:
        t.add_column(str(c), overflow="fold")
    for rec in df.head(max_rows).to_dict(orient="records"):
        t.add_row(*[_fmt(rec[c]) for c in df.columns])
    return t


def _banner(console: Console, report: Mapping[str, Any]) -> None:
    for key in ("execution_banner", "banner"):
        if report.get(key):
            console.print(Panel(report[key], style="bold white on red"), markup=False, highlight=False)


def print_review(review: dict[str, Any], console: Console) -> None:
    """Terminal review. The final printed line is exactly `STATUS: <STATE>`."""
    _banner(console, review)
    run = review.get("backtest_run") or {}
    console.print(f"Review {review['sport']} / {review['dataset']}  run {run.get('backtest_run_id', 'n/a')}  "
                  f"model {review.get('model_type') or 'n/a'}  baseline {review['baseline_scenario']}",
                  markup=False, highlight=False)
    console.print(f"Referee {review.get('referee_version') or 'n/a'}  config hash "
                  f"{review.get('referee_config_hash') or 'n/a'}", markup=False, highlight=False)
    console.print(f"Holdout {review['holdout_period']['period_id']}: {review['holdout_period']['note']}",
                  markup=False, highlight=False)
    console.print(f"Lineage {_lineage_text(review)}", markup=False, highlight=False)
    for n in review.get("data_notes") or []:
        console.print(f"note: {n}", markup=False, highlight=False)
    t = Table(title="Sec 26 criteria")
    for c in ("id", "criterion", "state", "value", "detail"):
        t.add_column(c, overflow="fold")
    for c in review["criteria"]:
        t.add_row(c["id"], c["title"], f"[{_STYLE[c['state']]}]{c['state']}[/]", _short(c["value"], 3), c["detail"])
    console.print(t)
    nc = review.get("negative_controls")
    if nc:
        rows = pd.DataFrame(nc["experiments"])
        cols = [c for c in ("feature_set_id", "test_period", "n_games", "delta_brier", "p_value", "control_q")
                if c in rows]
        console.print(_rich_table(nc["label"], rows[cols], max_rows=60))
    ex = review.get("exploratory")
    if ex and len(ex["experiments"]):
        rows = pd.DataFrame(ex["experiments"])
        cols = [c for c in ("experiment_id", "feature_set_id", "test_period", "n_games", "delta_brier", "p_value",
                            "git_commit") if c in rows]
        console.print(_rich_table(ex["label"], rows[cols], max_rows=60))
    elif ex:
        console.print(f"{ex['label']}: none", markup=False, highlight=False)
    sup, inf = review["supporting"], review["informational"]
    buckets = inf["edge_buckets"]
    if isinstance(buckets, pd.DataFrame) and len(buckets):
        cols = [c for c in ("bucket", "observations", "n_games", "trades", "mean_predicted_edge", "realized_edge",
                            "realized_edge_ci_low", "realized_edge_ci_high", "net_roi", "label") if c in buckets]
        console.print(_rich_table(f"Edge buckets ({review['baseline_scenario']})", buckets[cols]))
    per = sup["26.2"]["per_period"]
    if isinstance(per, pd.DataFrame) and len(per):
        console.print(_rich_table("Per-period Brier vs market (26.2)", per))
    scen = sup["26.9"]["scenarios"]
    if isinstance(scen, pd.DataFrame) and len(scen):
        console.print(_rich_table("Spread-penalty scenarios (26.9)", scen))
    sens = inf.get("sensitivity")
    if isinstance(sens, pd.DataFrame) and len(sens):
        console.print(_rich_table("Snapshot-offset sensitivity (informational)", sens))
    console.print("Status rules applied (26.19):", markup=False, highlight=False)
    for r in review["status_reasons"]:
        console.print(f"  - {r}", markup=False, highlight=False)
    flags = _flag_lines(review)
    if flags:
        console.print("Review flags (ADR 0002):", markup=False, highlight=False)
        for f in flags:
            console.print(f"  - {f}", markup=False, highlight=False)
    _banner(console, review)
    console.print(f"STATUS: {review['status']}", markup=False, highlight=False, soft_wrap=True)


def print_evaluation(report: dict[str, Any], console: Console) -> None:
    """Terminal Sec 17 evaluation (`quant report` / `quant backtest`)."""
    _banner(console, report)
    run = report.get("backtest_run") or {}
    console.print(f"Evaluation {report['sport']} / {report['dataset']}  run {run.get('backtest_run_id', 'n/a')}  "
                  f"model {run.get('model_type') or 'n/a'}  ({report['holdout']})", markup=False, highlight=False)
    for n in report.get("data_notes") or []:
        console.print(f"note: {n}", markup=False, highlight=False)
    pq = report["prediction_quality"]
    if pq["pooled"]:
        pooled = pd.DataFrame([{"scope": "pooled", **pq["pooled"]}])
        per = pq["per_period"].rename(columns={"period_id": "scope"})
        console.print(_rich_table("Prediction quality: market vs model (Sec 17)", pd.concat([per, pooled])))
        bs = pq["bootstrap"]
        console.print(f"Paired game bootstrap ({bs['n_resamples']} resamples, {bs['n_games']} games): "
                      f"delta Brier {bs['mean_delta']:.5f} [{bs['ci_low']:.5f}, {bs['ci_high']:.5f}], "
                      f"one-sided p {bs['p_value']:.4f}, BSS {bs['bss']:.4f}", markup=False, highlight=False)
        for k, tab in pq["calibration"].items():
            console.print(_rich_table(f"Calibration: {k} (ECE {tab['ece']:.4f}, slope {tab['slope']:.3f})",
                                      tab["table"][["bin", "n", "mean_predicted", "empirical_frequency",
                                                    "ci_low", "ci_high"]]))
    else:
        console.print("no evaluated predictions", markup=False, highlight=False)
    if report["trading"]:
        keys = ("trades", "win_rate", "avg_entry_probability", "realized_outcome_probability", "pnl_before_fees",
                "pnl_after_fees", "pnl_after_fees_slippage", "roi", "max_drawdown", "longest_losing_streak")
        console.print(_rich_table("Trading by spread-penalty scenario",
                                  pd.DataFrame([{"scenario": s, **{k: t[k] for k in keys}}
                                                for s, t in report["trading"].items()])))
    b = report["edge_buckets"]
    cols = [c for c in ("bucket", "observations", "n_games", "trades", "mean_p_market", "mean_p_model",
                        "empirical_frequency", "mean_predicted_edge", "realized_edge", "realized_edge_ci_low",
                        "realized_edge_ci_high", "gross_roi", "net_roi", "label") if c in b]
    console.print(_rich_table(f"Model-edge buckets ({report['baseline_scenario']})", b[cols]))
    _banner(console, report)
