"""Walk-forward training, LightGBM-vs-logistic selection, and the experiment log
(spec Sec 16, 19, 26.2, 26.4, 26.5, 26.16).

For each model period P (every evaluation period, then the holdout):

  training rows  canonical observations (model.train_offset_min) with a known
                 outcome, official_date < P.start and outcome_available_at <
                 P.start 00:00 UTC. Nothing dated in or after P, and no outcome
                 published after P began, reaches P's model.
  predictions    every observation (all snapshot offsets) dated in P. For the
                 holdout, only observations whose outcome is not yet known:
                 the holdout model exists for `scan` / `paper`, and no holdout
                 row with an outcome is ever predicted, so none can be scored
                 (26.3).
  test rows      canonical observations in P with an outcome (evaluation
                 periods only); every comparison below is on these.

Models: the market baseline, the logistic residual and the LightGBM residual
(src/models/residual.py). Selection follows config model.selection: LightGBM
only if its Brier beats logistic's in at least `min_periods_won` evaluation
periods AND its pooled Brier is lower. Otherwise logistic.

Feature set: registered features (feature_definitions) that a CONFIRMATORY,
non-negative-control hypothesis names. Negative controls and EXPLORATORY
features are only ever added inside experiments, never to the model that
backtest / scan / paper use.

Experiments (append-only log, deterministic ids) - per evaluation period and
pooled, candidate vs reference on the same test rows, one-sided paired game
bootstrap p-value for delta_brier > 0:

  model             logistic / LightGBM residual vs the market
  ablation          full model vs full minus one family (Sec 19)
  negative_control  full + one control vs full (26.16)
  hypothesis        H_MLB_001 (rows with |latency_gap| > 0.01), H_MLB_002 (per
                    liquidity tercile): selected model vs the market
  exploratory       H_MLB_003 features added to the full model (secondary
                    metric; its primary metric is recorded by `quant backtest`)

Experiment config_hash covers the model / uncertainty / walk-forward config and
a content fingerprint of the data, so re-running on the same data adds nothing
and re-running on changed data logs new experiments.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..backtest import holdout
from ..backtest import metrics as M
from ..backtest.registry import (experiment_id, is_active, load_hypotheses, pending_registration, record_experiments,
                                 register_hypotheses)
from ..backtest.walk_forward import Period, evaluation_periods, holdout_period, load_periods, new_period_rows
from ..config import CONFIG_DIR, data_dir, feature_preset, stable_hash, strategy_family
from ..storage.database import ConstraintViolation, Database
from ..storage.schemas import CANDLE_DEGRADED, DEGRADED_EXECUTION, EXECUTABLE, QUOTE_QUALITIES
from .residual import MODEL_TYPES, FittedModel, fit_model

MARKET, LOGISTIC, LGBM = MODEL_TYPES
POOLED = "pooled"
POOLED_TRAIN = "walk_forward"
LATENCY_GAP_MIN = 0.01          # fixed at registration (H_MLB_001 primary_metric)
EXPERIMENT_ENSEMBLE = 2         # experiments compare p_model only; the interval is unused there


def model_dir(db: Database) -> Path:
    return data_dir() / "models" / f"{db.sport}_{db.dataset}"


def hypotheses_path(sport: str) -> Path:
    return CONFIG_DIR / f"{sport}_hypotheses.yaml"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_OBS_SQL = """
SELECT observation_id, game_id, contract_id, official_date, snapshot_offset_min, snapshot_time, quote_ts,
       p_mid, liquidity_tercile, cross_market_gap, news_event_count_t180_t30, outcome, outcome_available_at
FROM observations WHERE quote_quality = ? ORDER BY snapshot_time, game_id, contract_id
"""


@dataclass
class TrainData:
    obs: pd.DataFrame            # time-ordered observations, RangeIndex
    X: pd.DataFrame              # wide feature values, same index as obs; NaN = missing
    families: dict[str, str]     # feature -> family (feature_definitions)
    fingerprint: str
    quote_quality: str = EXECUTABLE


def model_run_id(model_type: str, period_id: str, feature_hash: str, config_hash: str, data_fingerprint: str,
                 git_commit: str | None, created_at: datetime) -> str:
    """One id per `quant train` invocation: the commit and the time make a rerun a new run, never an
    overwrite of an earlier one (ADR 0002 item 4, the ADR 0001 N1 pattern)."""
    return "mr-" + stable_hash({"model_type": model_type, "period": period_id, "features": feature_hash,
                                "config": config_hash, "data": data_fingerprint, "git_commit": git_commit,
                                "created_at": created_at.isoformat()})


def load_train_data(db: Database, quote_quality: str = EXECUTABLE) -> TrainData:
    """Observations of ONE quote class (executable and candle_degraded never share a result)."""
    if quote_quality not in QUOTE_QUALITIES:
        raise ValueError(f"quote_quality must be one of {QUOTE_QUALITIES}, got {quote_quality!r}")
    obs = db.df(_OBS_SQL, [quote_quality])
    if obs.empty:
        n = db.scalar("SELECT count(*) FROM observations")
        hint = ("run `quant features` first" if not n else
                f"the {n} observations are all of another quote class; candle-only data can be run as "
                f"--quote-quality {CANDLE_DEGRADED} (DEGRADED EXECUTION, descriptive use only)")
        raise RuntimeError(f"no {quote_quality} observations; {hint}")
    obs["official_date"] = pd.to_datetime(obs["official_date"])
    obs["outcome"] = pd.to_numeric(obs["outcome"], errors="coerce").astype(float)
    for c in ("snapshot_time", "quote_ts", "outcome_available_at"):
        obs[c] = pd.to_datetime(obs[c], utc=True)
    defs = db.df("SELECT feature_name, family FROM feature_definitions ORDER BY feature_name")
    families = dict(zip(defs["feature_name"], defs["family"]))
    if families and db.count("feature_values"):
        wide = db.df("PIVOT (SELECT observation_id, feature_name, value FROM feature_values) "
                     "ON feature_name USING first(value) GROUP BY observation_id")
        X = wide.set_index("observation_id").reindex(obs["observation_id"])
    else:
        X = pd.DataFrame(index=obs["observation_id"])
    X = X.reindex(columns=sorted(families)).astype(float)
    X.index = obs.index
    o = db.con.execute("SELECT count(*), round(sum(p_mid), 6), sum(coalesce(outcome, 0)), min(snapshot_time), "
                       "max(snapshot_time) FROM observations WHERE quote_quality = ?", [quote_quality]).fetchone()
    f = db.con.execute("SELECT count(*), count(value), round(sum(coalesce(value, 0)), 4) FROM feature_values").fetchone()
    return TrainData(obs, X, families, stable_hash({"observations": list(o), "feature_values": list(f),
                                                    "quote_quality": quote_quality}), quote_quality)


@dataclass
class FeatureSets:
    model: list[str]                          # the full model's features
    by_family: dict[str, list[str]]           # model features per family (ablation units)
    negative_controls: list[str]
    exploratory: dict[str, list[str]]         # hypothesis_id -> features added in its experiment
    unused: list[str]                         # registered, but named by no hypothesis: never used
    missing: dict[str, list[str]]             # hypothesis_id -> named features that were not built
    disabled: dict[str, list[str]] = field(default_factory=dict)  # PAUSED / DISABLED hypothesis -> its features
    excluded_by_preset: dict[str, list[str]] = field(default_factory=dict)  # family outside the feature preset


def feature_sets(families: dict[str, str], hyps: list[dict[str, Any]],
                 allowed_families: list[str] | None = None) -> FeatureSets:
    """Split hypothesis features into model / negative-control / exploratory sets.

    allowed_families (the feature preset) limits the model and the exploratory experiments;
    negative controls run under every preset. None allows every family."""
    registered = set(families)
    allowed = None if allowed_families is None else set(allowed_families)
    excluded: dict[str, list[str]] = {}
    model: set[str] = set()
    nc: list[str] = []
    expl: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    named: set[str] = set()
    disabled: dict[str, list[str]] = {}
    for h in hyps:
        names = [f["name"] for f in h["features"]]
        named |= set(names)
        built = [n for n in names if n in registered]
        if len(built) < len(names):
            missing[h["hypothesis_id"]] = [n for n in names if n not in registered]
        if not is_active(h):
            disabled[h["hypothesis_id"]] = built    # registered and visible, used by no model or experiment
        elif h["family"] == "negative_control":
            nc += [n for n in built if n not in nc]
        elif allowed is not None and h["family"] not in allowed:
            excluded[h["hypothesis_id"]] = built
        elif h["label"] == "EXPLORATORY":
            expl[h["hypothesis_id"]] = built
        else:
            model |= set(built)
    expl = {k: [n for n in v if n not in model] for k, v in expl.items()}
    ordered = sorted(model)
    by_family: dict[str, list[str]] = {}
    for n in ordered:
        by_family.setdefault(families[n], []).append(n)
    return FeatureSets(ordered, by_family, nc, expl, sorted(registered - named), missing, disabled, excluded)


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

@dataclass
class PeriodSplit:
    period: Period
    cutoff: pd.Timestamp          # training uses outcomes published strictly before this
    train: np.ndarray             # positions in obs
    predict: np.ndarray
    test: np.ndarray


def split_period(obs: pd.DataFrame, period: Period, train_offset: int) -> PeriodSplit:
    start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
    cutoff = pd.Timestamp(period.start, tz="UTC")
    known = obs["outcome"].notna().to_numpy()
    canon = (obs["snapshot_offset_min"] == train_offset).to_numpy()
    in_p = obs["official_date"].between(start, end).to_numpy()
    avail = obs["outcome_available_at"]
    train = canon & known & (obs["official_date"] < start).to_numpy() & (avail.notna() & (avail < cutoff)).to_numpy()
    predict = in_p & (~known if period.is_holdout else True)
    test = in_p & canon & known & (not period.is_holdout)
    return PeriodSplit(period, cutoff, np.flatnonzero(train), np.flatnonzero(predict), np.flatnonzero(test))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit(model_type: str, data: TrainData, rows: np.ndarray, feats: list[str], cfg: dict[str, Any], *,
         seed: int, threads: int) -> FittedModel:
    tr = data.obs.iloc[rows]
    if model_type == MARKET:
        X, fam = None, None
    else:
        X, fam = data.X.iloc[rows][feats], {f: data.families[f] for f in feats}
    return fit_model(model_type, X, tr["p_mid"].to_numpy(), tr["outcome"].to_numpy(), tr["game_id"].to_numpy(),
                     fam, cfg, seed=seed, num_threads=threads)


def _predict(model: FittedModel, data: TrainData, rows: np.ndarray) -> pd.DataFrame:
    p = data.obs["p_mid"].to_numpy()[rows]
    X = None if model.model_type == MARKET else data.X.iloc[rows][model.features]
    out = model.predict(X, p).reset_index(drop=True)
    out.insert(0, "observation_id", data.obs["observation_id"].to_numpy()[rows])
    return out


def _test_frame(data: TrainData, rows: np.ndarray) -> pd.DataFrame:
    o = data.obs.iloc[rows]
    return pd.DataFrame({"observation_id": o["observation_id"].to_numpy(), "game_id": o["game_id"].to_numpy(),
                         "y": o["outcome"].to_numpy(), "p_market": o["p_mid"].to_numpy()})


@dataclass
class Comparison:
    """One candidate-vs-reference test on one period's test rows."""
    family: str
    hypothesis_id: str | None
    label: str
    feature_set_id: str
    features: list[str]
    model_type: str
    reference: str
    candidate: str
    period_id: str
    frame: pd.DataFrame           # game_id, y, p_ref, p_cand
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.family, self.hypothesis_id, self.feature_set_id, self.model_type)


def _frame(test: pd.DataFrame, p_ref: Any, p_cand: Any, mask: np.ndarray | None = None) -> pd.DataFrame:
    f = pd.DataFrame({"game_id": test["game_id"].to_numpy(), "y": test["y"].to_numpy(),
                      "p_ref": np.asarray(p_ref, float), "p_cand": np.asarray(p_cand, float)})
    return f[mask].reset_index(drop=True) if mask is not None else f


def compare_metrics(frame: pd.DataFrame, *, reference: str, candidate: str, n_boot: int, seed: int,
                    ci: float) -> tuple[dict[str, Any], float | None]:
    bs = M.paired_game_bootstrap(frame, n=n_boot, seed=seed, ci=ci, p_model="p_cand", p_market="p_ref")
    y = frame["y"]
    metrics = {
        "reference": reference, "candidate": candidate, "n_obs": bs["n_obs"], "n_games": bs["n_games"],
        "brier_ref": bs["brier_market"], "brier_cand": bs["brier_model"],
        "delta_brier": bs["brier_market"] - bs["brier_model"],
        "delta_brier_ci": [bs["ci_low"], bs["ci_high"]], "ci_level": ci, "bootstrap_resamples": n_boot,
        "logloss_ref": bs["log_loss_market"], "logloss_cand": bs["log_loss_model"],
        "ece_ref": M.calibration_table(y, frame["p_ref"])["ece"] if len(frame) else None,
        "ece_cand": M.calibration_table(y, frame["p_cand"])["ece"] if len(frame) else None,
        "roi_ref": None, "roi_cand": None,
    }
    p = bs["p_value"]
    return metrics, (float(p) if p is not None and np.isfinite(p) else None)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    model_runs: pd.DataFrame
    n_predictions: int
    selection: dict[str, Any]
    per_period: pd.DataFrame
    experiments_logged: int
    experiments_total: int
    hypotheses_registered: int
    notes: list[str]


def _selection(per: pd.DataFrame, pooled: dict[str, float], rule: dict[str, Any]) -> dict[str, Any]:
    need = int(rule.get("min_periods_won", 4))
    require_pooled = bool(rule.get("require_lower_aggregate_brier", True))
    won = int((per["brier_lgbm_residual"] < per["brier_logistic_residual"]).sum()) if len(per) else 0
    pooled_lower = pooled.get(LGBM, np.inf) < pooled.get(LOGISTIC, -np.inf)
    use_lgbm = won >= need and (pooled_lower or not require_pooled)
    reason = (f"LightGBM beat logistic in {won} of {len(per)} evaluation periods (need {need}); pooled Brier "
              f"LightGBM {pooled.get(LGBM, float('nan')):.5f} vs logistic {pooled.get(LOGISTIC, float('nan')):.5f} "
              f"({'lower' if pooled_lower else 'not lower'})")
    return {"selected": LGBM if use_lgbm else LOGISTIC, "reason": reason, "lgbm_periods_won": won,
            "evaluation_periods": int(len(per)), "min_periods_won": need,
            "require_lower_aggregate_brier": require_pooled, "pooled_brier": pooled,
            "caveat": "selection uses the walk-forward evaluation periods that are also reported; the final "
                      "holdout is untouched and is the unbiased test (26.3)"}


def run_training(db: Database, cfg: dict[str, Any], *, hyps_path: Path | None = None, now: datetime | None = None,
                 threads: int | None = None, preset: str | None = None, quote_quality: str = EXECUTABLE,
                 log: Callable[[str], None] = lambda s: None) -> TrainResult:
    strategy = strategy_family(cfg)                                  # market_behavior is refused
    preset_name, preset_families = feature_preset(cfg, db.dataset, preset)   # live baseball gate
    now = now or datetime.now(timezone.utc)
    commit = holdout.git_commit()                                   # spec Sec 20; None without git
    threads = threads or max(1, min(8, os.cpu_count() or 1))
    mcfg, rc = cfg.get("model") or {}, cfg.get("review") or {}
    seed = int(mcfg.get("seed", 7))
    train_offset = int(mcfg.get("train_offset_min", 30))
    min_games = int(mcfg.get("min_train_games", 300))
    n_boot, ci, boot_seed = int(rc.get("bootstrap_resamples", 10_000)), float(rc.get("ci_level", 0.95)), int(rc.get("seed", 7))
    periods = load_periods(cfg, db.dataset)
    # a written period_id keeps its window; refused here, before anything is written
    new_periods = new_period_rows(db, periods)
    notes: list[str] = []

    # 26.4: hypotheses are registered before any model is fitted or scored. Only from a clean tree
    # (ADR 0001 Q3): a dirty tree may train, tagged -dirty, but the registration step is refused, out loud.
    hyps = load_hypotheses(hyps_path or hypotheses_path(db.sport))
    if holdout.is_clean_commit(commit):
        n_registered = register_hypotheses(db, hyps, git_commit=commit, now=now)
    else:
        n_registered = 0
        new_ids, status_ids = pending_registration(db, hyps)     # validates; an edited definition still stops
        refusal = (f"hypothesis registration refused (ADR 0001 Q3): {holdout.tree_state(commit)}. Training "
                   f"continues, tagged {commit or 'without a commit'}. Not registered: {new_ids or 'none pending'}; "
                   f"status changes not recorded: {status_ids or 'none pending'}. Commit, then register from a "
                   f"clean tree (`quant register`).")
        log(refusal)
        notes.append(refusal)
    by_id = {h["hypothesis_id"]: h for h in hyps}

    data = load_train_data(db, quote_quality)
    if quote_quality == CANDLE_DEGRADED:
        notes.append(DEGRADED_EXECUTION)
    fs = feature_sets(data.families, hyps, preset_families)
    if not fs.model:
        raise RuntimeError(f"no registered model features in preset {preset_name!r} {preset_families}; "
                           f"run `quant features` first")
    log(f"strategy {strategy}; feature preset {preset_name}: {preset_families}")
    for hid, names in fs.excluded_by_preset.items():
        notes.append(f"{hid}: family outside feature preset {preset_name!r}, not used: {names}")
    for hid, names in fs.missing.items():
        notes.append(f"{hid}: features not built, so not used: {names}")
    if fs.unused:
        notes.append(f"registered features named by no hypothesis, not used: {fs.unused}")

    exp_cfg = copy.deepcopy(cfg)
    exp_cfg.setdefault("uncertainty", {})["ensemble_size"] = EXPERIMENT_ENSEMBLE
    run_hash = stable_hash({"model": mcfg, "uncertainty": cfg.get("uncertainty"), "walk_forward": cfg.get("walk_forward"),
                            "data": data.fingerprint, "strategy": strategy, "preset_families": preset_families})
    feature_set_hash = stable_hash(fs.model)
    out_dir = model_dir(db)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    preds: list[pd.DataFrame] = []
    per_rows: list[dict[str, Any]] = []
    tests: dict[str, pd.DataFrame] = {}
    test_pred: dict[tuple[str, str], np.ndarray] = {}    # (period, model_type) -> p_model on test rows
    splits: dict[str, PeriodSplit] = {}

    def fit_and_store(model_type: str, split: PeriodSplit) -> FittedModel:
        feats = [] if model_type == MARKET else fs.model
        model = _fit(model_type, data, split.train, feats, cfg, seed=seed, threads=threads)
        info = model.summary()
        tr = data.obs.iloc[split.train]
        p_rows = data.obs.iloc[split.predict]
        predict_from = p_rows["snapshot_time"].min() if len(p_rows) else split.cutoff
        through = tr["outcome_available_at"].max()
        if not through < predict_from or not through < split.cutoff:
            raise AssertionError(f"{split.period.period_id}: training outcome at {through} is not before "
                                 f"{min(predict_from, split.cutoff)}")
        fsid = "market" if model_type == MARKET else "full"
        fhash = stable_hash(feats)
        run_id = model_run_id(model_type, split.period.period_id, fhash, info["config_hash"], data.fingerprint,
                              commit, now)
        path = out_dir / f"{run_id}.pkl"
        if path.exists():               # the same invocation again: refuse before rewriting an earlier model
            raise ConstraintViolation("model_runs", "model_run_id is unique (ADR 0002 item 4)",
                                      detail=f"{run_id} already exists ({path}); a rerun needs a new commit or time")
        model.save(path)
        runs.append({
            "model_run_id": run_id, "model_type": model_type, "period_id": split.period.period_id,
            "feature_set_id": fsid, "feature_set_hash": fhash, "config_hash": info["config_hash"],
            "params_json": json.dumps(info, sort_keys=True, default=_json_default),
            "calibration_method": info["calibration_method"],
            "n_train_obs": int(info["n_fit_obs"] + info["n_cal_obs"]), "n_train_games": int(info["n_games"]),
            "train_data_through": through, "predict_from": predict_from, "artifact_path": str(path),
            "dataset": db.dataset, "git_commit": commit, "created_at": now, "is_synthetic": db.is_synthetic,
        })
        if len(split.predict):
            p = _predict(model, data, split.predict)
            p.insert(0, "model_run_id", run_id)
            p.insert(2, "period_id", split.period.period_id)
            p["is_synthetic"] = db.is_synthetic
            preds.append(p)
        if len(split.test):
            test_pred[(split.period.period_id, model_type)] = _predict(model, data, split.test)["p_model"].to_numpy()
        return model

    # 1. Evaluation periods: all three model types.
    for period in evaluation_periods(periods):
        split = split_period(data.obs, period, train_offset)
        n_games = data.obs["game_id"].iloc[split.train].nunique()
        if n_games < min_games:
            notes.append(f"{period.period_id}: {n_games} earlier games (< {min_games}); no model")
            continue
        if not len(split.test):
            notes.append(f"{period.period_id}: no scored observations; no model")
            continue
        splits[period.period_id] = split
        tests[period.period_id] = _test_frame(data, split.test)
        log(f"{period.period_id}: training on {n_games} games")
        row: dict[str, Any] = {"period_id": period.period_id, "train_games": int(n_games),
                               "test_obs": int(len(split.test)),
                               "test_games": int(tests[period.period_id]["game_id"].nunique())}
        for mt in MODEL_TYPES:
            m = fit_and_store(mt, split)
            t = tests[period.period_id]
            row[f"brier_{mt}"] = M.brier(t["y"], test_pred[(period.period_id, mt)])
            if mt != MARKET:
                row[f"calibration_{mt}"] = m.info["calibration_method"]
                row[f"interval_scale_{mt}"] = m.info["uncertainty"]["scale"]
        per_rows.append(row)
    per = pd.DataFrame(per_rows)
    if per.empty:
        raise RuntimeError("no evaluation period could be trained; " + "; ".join(notes))

    # 2. LightGBM vs logistic (config model.selection).
    all_tests = pd.concat([tests[p] for p in per["period_id"]], ignore_index=True)
    pooled = {mt: M.brier(all_tests["y"], np.concatenate([test_pred[(p, mt)] for p in per["period_id"]]))
              for mt in MODEL_TYPES}
    selection = _selection(per, pooled, mcfg.get("selection") or {})
    selected = selection["selected"]
    selection |= {"features": fs.model, "feature_families": sorted(fs.by_family), "feature_set_hash": feature_set_hash,
                  "negative_controls": fs.negative_controls, "exploratory": fs.exploratory,
                  "unused_features": fs.unused, "disabled_hypotheses": fs.disabled,
                  "strategy_family": strategy, "feature_preset": preset_name, "preset_families": preset_families,
                  "excluded_by_preset": fs.excluded_by_preset, "quote_quality": quote_quality,
                  "per_period": per.to_dict(orient="records"),
                  "train_offset_min": train_offset, "seed": seed, "dataset": db.dataset,
                  "is_synthetic": bool(db.is_synthetic),
                  "status": ("MECHANISM_TEST: chosen on synthetic data; not a model decision and never frozen"
                             if db.is_synthetic else "CANDIDATE: not frozen until `quant freeze` (26.3)")}
    log(f"selected {selected}: {selection['reason']}")

    # 3. Holdout: market + the selected model, trained only on pre-holdout outcomes; predicts unsettled rows only.
    hold = holdout_period(periods)
    hsplit = split_period(data.obs, hold, train_offset)
    if data.obs["game_id"].iloc[hsplit.train].nunique() >= min_games:
        for mt in (MARKET, selected):
            fit_and_store(mt, hsplit)
        selection["holdout_models"] = [r["model_run_id"] for r in runs if r["period_id"] == hold.period_id]
        selection["holdout_predictions"] = int(len(hsplit.predict))
    else:
        notes.append(f"{hold.period_id}: too few earlier games for a holdout model")

    # 4. Experiments on evaluation test rows.
    comps: list[Comparison] = []
    for pid in per["period_id"]:
        t, split = tests[pid], splits[pid]
        market = t["p_market"].to_numpy()
        full = test_pred[(pid, selected)]
        for mt in (LOGISTIC, LGBM):
            comps.append(Comparison("model", None, "CONFIRMATORY", "full", fs.model, mt, MARKET, mt, pid,
                                    _frame(t, market, test_pred[(pid, mt)])))

        def refit(feats: list[str]) -> np.ndarray:
            m = _fit(selected, data, split.train, feats, exp_cfg, seed=seed, threads=threads)
            return _predict(m, data, split.test)["p_model"].to_numpy()

        for fam, names in fs.by_family.items():
            rest = [f for f in fs.model if f not in names]
            hid = f"H_MLB_L1_{fam.upper()}"
            p_ref = refit(rest) if rest else market
            comps.append(Comparison("ablation", hid if hid in by_id else None, by_id.get(hid, {}).get("label", "CONFIRMATORY"),
                                    f"ablate_{fam}", fs.model, selected, f"full minus {fam}", "full", pid,
                                    _frame(t, p_ref, full), {"family": fam, "family_features": names}))
        for c in fs.negative_controls:
            comps.append(Comparison("negative_control", "H_MLB_NC", "CONFIRMATORY", f"full+{c}", fs.model + [c],
                                    selected, "full", f"full + {c}", pid, _frame(t, full, refit(fs.model + [c]))))
        for hid, names in fs.exploratory.items():
            if names:
                comps.append(Comparison("exploratory", hid, by_id[hid]["label"], f"full+{hid}", fs.model + names,
                                        selected, "full", f"full + {hid} features", pid,
                                        _frame(t, full, refit(fs.model + names)),
                                        {"metric_role": "secondary; primary metric recorded by quant backtest"}))
        obs_t = data.obs.iloc[split.test]
        if "H_MLB_001" in by_id and "latency_gap" in data.X:
            gap = data.X["latency_gap"].to_numpy()[split.test]
            mask = np.abs(np.nan_to_num(gap, nan=0.0)) > LATENCY_GAP_MIN
            news_games = obs_t.loc[obs_t["news_event_count_t180_t30"].fillna(0).to_numpy() > 0, "game_id"].nunique()
            comps.append(Comparison("hypothesis", "H_MLB_001", by_id["H_MLB_001"]["label"], "latency_gap_gt_0.01",
                                    fs.model, selected, MARKET, selected, pid, _frame(t, market, full, mask),
                                    {"subset": f"|latency_gap| > {LATENCY_GAP_MIN}", "games_with_news": int(news_games)}))
        if "H_MLB_002" in by_id:
            terc = pd.to_numeric(obs_t["liquidity_tercile"], errors="coerce").to_numpy()
            gap = np.abs(pd.to_numeric(obs_t["cross_market_gap"], errors="coerce").to_numpy())
            for k in (1, 2, 3):
                mask = terc == k
                g = gap[mask]
                comps.append(Comparison("hypothesis", "H_MLB_002", by_id["H_MLB_002"]["label"], f"liquidity_tercile_{k}",
                                        fs.model, selected, MARKET, selected, pid, _frame(t, market, full, mask),
                                        {"tercile": k, "mean_abs_cross_market_gap":
                                            float(np.nanmean(g)) if np.isfinite(g).any() else None,
                                         "n_with_gap": int(np.isfinite(g).sum())}))
        log(f"{pid}: experiments fitted")

    rows, total = _experiment_rows(comps, by_id, run_hash=run_hash, dataset=db.dataset, now=now, git_commit=commit,
                                   n_boot=n_boot, seed=boot_seed, ci=ci, notes=notes, quote_quality=quote_quality)
    logged = record_experiments(db, rows)

    run_df = pd.DataFrame(runs)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    lineage = _lineage_rows(db, run_df, rows, by_id, commit=commit, now=now, run_hash=run_hash,
                            quote_quality=quote_quality)
    # model_runs and predictions only grow (ADR 0002 item 4); readers take the latest invocation
    # (walk_forward.latest_model_runs). walk_forward_periods only grows too: its new
    # period ids go in the same transaction, so a period id written in between rolls the whole train back.
    # So does the lineage: a model run is never written without it.
    db.append_tables({"walk_forward_periods": new_periods, "model_runs": run_df, "predictions": pred_df,
                      "train_lineage": lineage})
    selection["experiments"] = {"total": total, "newly_logged": logged}
    (out_dir / "model_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True, default=_json_default))
    return TrainResult(run_df, int(len(pred_df)), selection, per, logged, total, n_registered, notes)


def _lineage_rows(db: Database, run_df: pd.DataFrame, exp_rows: list[dict[str, Any]], by_id: dict[str, Any], *,
                  commit: str | None, now: datetime, run_hash: str, quote_quality: str) -> pd.DataFrame:
    """One invocation's lineage. Its model runs, the experiments it logged (their ids, as
    record_experiments computes them) and every hypothesis it loaded."""
    invocation = "ti-" + stable_hash({"dataset": db.dataset, "git_commit": commit, "created_at": now.isoformat(),
                                      "config": run_hash, "quote_quality": quote_quality})
    kinds = (("model_run", run_df["model_run_id"] if len(run_df) else []),
             ("experiment", [experiment_id(**r) for r in exp_rows]), ("hypothesis", by_id))
    return pd.DataFrame([{"invocation_id": invocation, "kind": kind, "row_id": rid, "dataset": db.dataset,
                          "git_commit": commit, "created_at": now, "is_synthetic": db.is_synthetic}
                         for kind, ids in kinds for rid in dict.fromkeys(ids)])


def _experiment_rows(comps: list[Comparison], by_id: dict[str, dict[str, Any]], *, run_hash: str, dataset: str,
                     now: datetime, git_commit: str | None, n_boot: int, seed: int, ci: float, notes: list[str],
                     quote_quality: str) -> tuple[list[dict], int]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple, list[Comparison]] = {}
    for c in comps:
        groups.setdefault(c.key, []).append(c)
    for key, cs in groups.items():
        first = cs[0]
        pooled = Comparison(first.family, first.hypothesis_id, first.label, first.feature_set_id, first.features,
                            first.model_type, first.reference, first.candidate, POOLED,
                            pd.concat([c.frame for c in cs], ignore_index=True), {**first.extra, "pooled_over":
                                                                                 [c.period_id for c in cs]})
        for c in (*cs, pooled):
            if not len(c.frame):
                notes.append(f"{c.family}/{c.feature_set_id} {c.period_id}: no rows; not tested")
                continue
            metrics, p = compare_metrics(c.frame, reference=c.reference, candidate=c.candidate, n_boot=n_boot,
                                         seed=seed, ci=ci)
            h = by_id.get(c.hypothesis_id or "")
            if h is not None:
                metrics |= {"minimum_sample": h["minimum_sample"],
                            "sufficient_sample": bool(metrics["n_games"] >= h["minimum_sample"])}
            metrics |= c.extra
            rows.append({
                "hypothesis_id": c.hypothesis_id, "experiment_family": c.family, "label": c.label,
                "feature_set_id": c.feature_set_id, "feature_set_hash": stable_hash(sorted(c.features)),
                "model_type": c.model_type, "config_hash": run_hash,
                "train_period": POOLED_TRAIN if c.period_id == POOLED else f"before:{c.period_id}",
                "test_period": c.period_id, "n_games": int(metrics["n_games"]),
                "metrics_json": json.dumps(metrics, sort_keys=True, default=_json_default), "p_value": p,
                "quote_quality": quote_quality, "git_commit": git_commit, "created_at": now, "dataset": dataset,
            })
    return rows, len(rows)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def load_selection(db: Database) -> dict[str, Any]:
    path = model_dir(db) / "model_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `quant train` first")
    return json.loads(path.read_text())
