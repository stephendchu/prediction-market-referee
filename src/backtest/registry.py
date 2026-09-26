"""Hypothesis and experiment registries (spec Sec 20, 26.4, 26.5).

Hypotheses are registered before results are examined. A registered hypothesis
is never redefined: re-registering the same definition is a no-op, and the same
id with a different definition raises `HypothesisChanged` (a changed idea needs
a new id). Relabeling EXPLORATORY -> CONFIRMATORY is a definition change too.

A hypothesis's status (ACTIVE | PAUSED | DISABLED, with a reason and the
decision record) is not part of its definition. Each status change is appended
to the `hypothesis_status` log, so a disabled hypothesis stays visible (26.5)
instead of disappearing from the registry.

Experiments are logged with deterministic ids, successes and failures alike.
Nothing here filters by result; BH-FDR (26.5) needs every test in a family.
Both registries are append-only persistent logs (src/storage/schemas.py).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import stable_hash
from ..storage.database import Database
from ..storage.schemas import EXPERIMENTS, HYPOTHESES, HYPOTHESIS_STATUS, HYPOTHESIS_STATUSES

LABELS = ("CONFIRMATORY", "EXPLORATORY")
ACTIVE = "ACTIVE"

# Fields that define a hypothesis; definition_hash covers exactly these.
DEFINITION_FIELDS = ("hypothesis_id", "family", "description", "mechanism", "features", "expected_direction",
                     "minimum_sample", "evaluation_periods", "primary_metric", "label")

# Fields that identify an experiment (experiment_id).
# quote_quality is a key field: an executable and a candle_degraded experiment are never the same
# experiment, so neither can be dropped as a duplicate of the other.
# git_commit and created_at (the invocation's time) are key fields too, as they are for a
# model run (ADR 0002 item 4): each invocation logs its own rows, so a clean retrain never reuses rows that
# dirty code wrote.
EXPERIMENT_KEY_FIELDS = ("experiment_family", "hypothesis_id", "feature_set_hash", "model_type", "config_hash",
                         "train_period", "test_period", "dataset", "quote_quality", "git_commit", "created_at")

REQUIRED_METRICS = ("reference", "candidate", "n_obs", "brier_ref", "brier_cand", "delta_brier")


class HypothesisChanged(ValueError):
    """A registered hypothesis_id was given a different definition (26.4)."""


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

def load_hypotheses(path: Path | str) -> list[dict[str, Any]]:
    """Validated hypotheses from a YAML file with a top-level `hypotheses:` list."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    hyps = [validate_hypothesis(h) for h in raw.get("hypotheses") or []]
    if not hyps:
        raise ValueError(f"{path}: no hypotheses")
    _check_unique(hyps)
    return hyps


def validate_hypothesis(h: dict[str, Any]) -> dict[str, Any]:
    """Check one hypothesis and normalize `features` to [{name, definition, ...}, ...]."""
    hid = h.get("hypothesis_id", "?")
    missing = [k for k in DEFINITION_FIELDS if h.get(k) in (None, "", [])]
    if missing:
        raise ValueError(f"hypothesis {hid}: missing {missing}")
    if h["label"] not in LABELS:
        raise ValueError(f"hypothesis {hid}: label must be one of {LABELS}, got {h['label']!r}")
    if isinstance(h["minimum_sample"], bool) or not isinstance(h["minimum_sample"], int) or h["minimum_sample"] < 1:
        raise ValueError(f"hypothesis {hid}: minimum_sample must be a positive integer")
    if not isinstance(h["evaluation_periods"], list):
        raise ValueError(f"hypothesis {hid}: evaluation_periods must be a list")
    status = h.get("status") or ACTIVE
    if status not in HYPOTHESIS_STATUSES:
        raise ValueError(f"hypothesis {hid}: status must be one of {HYPOTHESIS_STATUSES}, got {status!r}")
    if status != ACTIVE and not (h.get("status_reason") and h.get("status_decision")):
        raise ValueError(f"hypothesis {hid}: status {status} needs status_reason and status_decision")
    features = h["features"]
    if isinstance(features, dict):
        features = [{"name": k, "definition": v} for k, v in features.items()]
    if not isinstance(features, list) or not all(
            isinstance(f, dict) and f.get("name") and f.get("definition") for f in features):
        raise ValueError(f"hypothesis {hid}: every feature needs a name and an exact definition")
    return {**h, "hypothesis_id": str(h["hypothesis_id"]), "features": features, "status": status}


def is_active(h: dict[str, Any]) -> bool:
    return (h.get("status") or ACTIVE) == ACTIVE


def definition_hash(h: dict[str, Any]) -> str:
    return stable_hash({k: h.get(k) for k in DEFINITION_FIELDS})


def hypothesis_rows(hyps: list[dict[str, Any]], *, dataset: str, is_synthetic: bool, git_commit: str | None,
                    now: datetime) -> pd.DataFrame:
    """Rows for the hypotheses log (columns in table order)."""
    registered_at = _utc(now)
    rows = [{
        "hypothesis_id": h["hypothesis_id"],
        "family": h["family"],
        "description": h["description"],
        "mechanism": h["mechanism"],
        "feature_definitions_json": json.dumps(h["features"], sort_keys=True),
        "expected_direction": str(h["expected_direction"]),
        "minimum_sample": int(h["minimum_sample"]),
        "evaluation_periods_json": json.dumps(h["evaluation_periods"]),
        "primary_metric": h["primary_metric"],
        "label": h["label"],
        "definition_hash": definition_hash(h),
        "git_commit": git_commit,
        "registered_at": registered_at,
        "dataset": dataset,
        "is_synthetic": is_synthetic,
    } for h in hyps]
    return pd.DataFrame(rows, columns=HYPOTHESES.column_names)


def _unregistered(db: Database, hyps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(validated hyps, those not yet registered). HypothesisChanged on an edited definition. Reads only."""
    hyps = [validate_hypothesis(h) for h in hyps]
    _check_unique(hyps)
    registered = dict(db.con.execute(
        "SELECT hypothesis_id, definition_hash FROM hypotheses WHERE dataset = ?", [db.dataset]).fetchall())
    new = []
    for h in hyps:
        known = registered.get(h["hypothesis_id"])
        if known is None:
            new.append(h)
        elif known != definition_hash(h):
            raise HypothesisChanged(f"{h['hypothesis_id']} is registered with definition {known}; the given "
                                    f"definition hashes to {definition_hash(h)}. Register it under a new id.")
    return hyps, new


def register_hypotheses(db: Database, hyps: list[dict[str, Any]], *, git_commit: str | None = None,
                        now: datetime | None = None) -> int:
    """Append hypotheses not yet registered; returns how many were new. All-or-nothing."""
    hyps, new = _unregistered(db, hyps)
    now = now or datetime.now(timezone.utc)
    if new:
        db.append_log("hypotheses", hypothesis_rows(new, dataset=db.dataset, is_synthetic=db.is_synthetic,
                                                    git_commit=git_commit, now=now))
    _record_status_changes(db, hyps, now)
    return len(new)


def pending_registration(db: Database, hyps: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """What register_hypotheses would write, without writing it: (ids not yet registered, ids whose
    status would change). It validates as register_hypotheses does, so an edited definition still
    raises HypothesisChanged. For `quant train` on a dirty tree (ADR 0001 Q3)."""
    hyps, new = _unregistered(db, hyps)
    changes = _status_change_rows(db, hyps, datetime.now(timezone.utc))
    return [h["hypothesis_id"] for h in new], [r["hypothesis_id"] for r in changes]


def current_statuses(db: Database) -> dict[str, dict[str, Any]]:
    """hypothesis_id -> latest status row (status, reason, decision_ref, recorded_at) for db.dataset.

    A registered hypothesis with no status row has been ACTIVE since registration."""
    reg = db.df("SELECT hypothesis_id, registered_at FROM hypotheses WHERE dataset = ?", [db.dataset])
    out = {h: {"status": ACTIVE, "reason": None, "decision_ref": None, "recorded_at": t}
           for h, t in zip(reg["hypothesis_id"], reg["registered_at"])}
    rows = db.df("SELECT hypothesis_id, status, reason, decision_ref, recorded_at FROM hypothesis_status "
                 "WHERE dataset = ? ORDER BY recorded_at", [db.dataset])
    for r in rows.to_dict(orient="records"):     # ascending: the last row per id wins
        out[r.pop("hypothesis_id")] = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
    return out


def hypotheses_with_status(db: Database) -> pd.DataFrame:
    """The registered hypotheses of db.dataset with their current status, reason and decision record."""
    hyps = db.df("SELECT * FROM hypotheses WHERE dataset = ? ORDER BY hypothesis_id", [db.dataset])
    st = current_statuses(db)
    for col, key in (("status", "status"), ("status_reason", "reason"), ("status_decision", "decision_ref"),
                     ("status_recorded_at", "recorded_at")):
        hyps[col] = [st.get(h, {}).get(key) for h in hyps["hypothesis_id"]]
    return hyps


def _status_change_rows(db: Database, hyps: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    current = current_statuses(db)
    return [{"hypothesis_id": h["hypothesis_id"], "status": h["status"], "reason": h.get("status_reason"),
             "decision_ref": h.get("status_decision"), "definition_hash": definition_hash(h),
             "recorded_at": _utc(now), "dataset": db.dataset, "is_synthetic": db.is_synthetic}
            for h in hyps if _status_key(current.get(h["hypothesis_id"], {"status": ACTIVE}))
            != (h["status"], h.get("status_reason"), h.get("status_decision"))]


def _record_status_changes(db: Database, hyps: list[dict[str, Any]], now: datetime) -> int:
    rows = _status_change_rows(db, hyps, now)
    if rows:
        db.append_log("hypothesis_status", pd.DataFrame(rows, columns=HYPOTHESIS_STATUS.column_names))
    return len(rows)


def _status_key(row: dict[str, Any]) -> tuple:
    return row["status"], row.get("reason"), row.get("decision_ref")


def _check_unique(hyps: list[dict[str, Any]]) -> None:
    ids = [h["hypothesis_id"] for h in hyps]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate hypothesis_id {dupes}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_id(**fields: Any) -> str:
    """Deterministic id from EXPERIMENT_KEY_FIELDS (`family` is accepted for experiment_family;
    other keys are ignored, so `experiment_id(**row)` works). A missing git_commit is None (no git);
    created_at has no default and is read as a UTC instant."""
    if "experiment_family" not in fields and "family" in fields:
        fields = {**fields, "experiment_family": fields["family"]}
    fields = {"hypothesis_id": None, "git_commit": None, **fields}
    missing = [k for k in EXPERIMENT_KEY_FIELDS if k not in fields or (k == "created_at" and pd.isna(fields[k]))]
    if missing:
        raise ValueError(f"experiment_id needs {missing}")
    key = {k: fields[k] for k in EXPERIMENT_KEY_FIELDS} | {"created_at": _utc(fields["created_at"]).isoformat()}
    return "exp-" + stable_hash(key)


def validate_metrics(metrics_json: str | dict[str, Any]) -> str:
    """Canonical metrics JSON; requires REQUIRED_METRICS and delta_brier = brier_ref - brier_cand."""
    try:
        metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else dict(metrics_json)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metrics_json is not JSON: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ValueError("metrics_json must be a JSON object")
    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        raise ValueError(f"metrics_json missing {missing}")
    ref, cand, delta = metrics["brier_ref"], metrics["brier_cand"], metrics["delta_brier"]
    if None not in (ref, cand, delta) and not math.isclose(delta, ref - cand, abs_tol=1e-9):
        raise ValueError(f"delta_brier {delta} != brier_ref - brier_cand ({ref} - {cand})")
    return json.dumps(metrics, sort_keys=True)


def record_experiments(db: Database, rows: list[dict[str, Any]]) -> int:
    """Append experiments whose ids are not yet logged; returns how many were new.

    Missing experiment_id, dataset, is_synthetic and created_at are filled in; a
    supplied experiment_id must equal the deterministic one.
    """
    seen = set(db.df("SELECT experiment_id FROM experiments")["experiment_id"])
    columns = set(EXPERIMENTS.column_names)
    now = datetime.now(timezone.utc)                  # one time per call: created_at is an id field
    new = []
    for row in rows:
        if "family" in row:
            row = {"experiment_family": row["family"], **{k: v for k, v in row.items() if k != "family"}}
        unknown = sorted(set(row) - columns)
        if unknown:
            raise ValueError(f"unknown experiment fields {unknown}")
        row = {"dataset": db.dataset, "is_synthetic": db.is_synthetic, "created_at": now, **row}
        if row["dataset"] != db.dataset:
            raise ValueError(f"experiment for dataset {row['dataset']!r} written to {db.dataset!r}")
        eid = experiment_id(**row)
        if row.setdefault("experiment_id", eid) != eid:
            raise ValueError(f"experiment_id {row['experiment_id']!r} is not the deterministic id {eid!r}")
        row["metrics_json"] = validate_metrics(row.get("metrics_json"))
        row["created_at"] = _utc(row["created_at"])
        if eid not in seen:
            seen.add(eid)
            new.append(row)
    if new:
        db.append_log("experiments", pd.DataFrame(new).reindex(columns=EXPERIMENTS.column_names))
    return len(new)


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Registration as its own step (26.4)
# ---------------------------------------------------------------------------

USAGE = "usage: python -m src.backtest.registry register <sport> <dataset>"


def register_now(db: Database, path: Path | str) -> tuple[int, pd.DataFrame]:
    """Register the hypotheses in `path` at the current UTC time; returns (new, registry of db.dataset).

    26.4 needs every hypothesis registered strictly before the earliest experiment. `quant train`
    registers its hypotheses file too, but with the same timestamp it stamps on the experiments it
    logs, so a hypothesis first registered by a train run is never "before testing". Registering
    here first, then running `quant train`, gives the order 26.4 checks: train's own registration is
    a no-op for an unchanged definition, and the registry keeps each id's first registration time.
    Nothing is backdated: the time is taken now, never passed in. It runs only from a clean tree:
    holdout.DirtyTree otherwise. While an ADR 0002 blocking flag is open, it runs once, for the
    clearing run: review_flags.ReviewBlocked otherwise. Either refusal comes before anything is
    written.
    """
    from . import holdout
    from .review_flags import require_registration_allowed  # review_flags imports this module
    commit = holdout.require_clean_tree("register hypotheses")
    hyps = load_hypotheses(path)
    require_registration_allowed(db, hyps)
    n = register_hypotheses(db, hyps, git_commit=commit, now=datetime.now(timezone.utc))
    reg = db.df("SELECT hypothesis_id, family, label, registered_at FROM hypotheses WHERE dataset = ? "
                "ORDER BY registered_at, hypothesis_id", [db.dataset])
    return n, reg


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "register":
        print(USAGE, file=sys.stderr)
        return 2
    from ..config import CONFIG_DIR
    from ..storage.database import connect

    sport, dataset = argv[1], argv[2]
    db = connect(sport, dataset)
    try:
        n, reg = register_now(db, CONFIG_DIR / f"{sport}_hypotheses.yaml")
        first = db.df("SELECT min(created_at) AS t, count(*) AS n FROM experiments")
    finally:
        db.close()
    print(f"{sport}_{dataset}: {n} hypothes{'is' if n == 1 else 'es'} newly registered")
    for r in reg.itertuples():
        print(f"  {r.hypothesis_id:<20} {r.family:<18} {r.label:<13} registered_at {_utc(r.registered_at).isoformat()}")
    if int(first["n"].iloc[0]):
        print(f"note: {int(first['n'].iloc[0])} experiment(s) already logged in {sport}_{dataset}, earliest "
              f"{_utc(first['t'].iloc[0]).isoformat()}; 26.4 compares every family's first registration with "
              f"that earliest experiment")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
