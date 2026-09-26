"""Walk-forward periods and the holdout lock (spec Sec 16, 26.2, 26.3).

Periods are fixed in `config/<sport>.yaml` before any results are seen, one list
per dataset: `walk_forward.periods` for synthetic data (mechanism tests) and
`walk_forward.live_periods` for a real dataset (none ships here; the refusal
tests use it). Roles:

  train_only  earliest data; never scored, only trained on
  evaluation  scored out of sample by a model trained on everything before it
  holdout     FINAL_HOLDOUT (26.3): the most recent period. A model is trained
              for it (for `scan` / `paper`), but nothing in this engagement may
              evaluate it. Any evaluation helper that is handed holdout rows
              raises HoldoutLocked.

Training for period P uses only observations whose outcome was known before
P's first snapshot (`outcome_available_at < predict_from`), which the database
also enforces on model_runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable

import pandas as pd

ROLES = ("train_only", "evaluation", "holdout")
PERIOD_KEYS = {"synthetic": "periods", "live": "live_periods"}   # dataset -> walk_forward key


class HoldoutLocked(RuntimeError):
    """Raised when code tries to evaluate FINAL_HOLDOUT data (26.3)."""


@dataclass(frozen=True)
class Period:
    period_id: str
    ordinal: int
    role: str
    start: date
    end: date

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    @property
    def is_holdout(self) -> bool:
        return self.role == "holdout"


def load_periods(cfg: dict[str, Any], dataset: str) -> list[Period]:
    """Parse and validate the dataset's walk-forward periods from a sport config."""
    if dataset not in PERIOD_KEYS:
        raise ValueError(f"dataset must be one of {sorted(PERIOD_KEYS)}, got {dataset!r}")
    key = PERIOD_KEYS[dataset]
    raw = (cfg.get("walk_forward") or {}).get(key) or []
    periods = [Period(str(p["id"]), i, p["role"], _date(p["start"]), _date(p["end"])) for i, p in enumerate(raw)]
    if not periods:
        raise ValueError(f"walk_forward.{key} is empty")
    for p in periods:
        if p.role not in ROLES:
            raise ValueError(f"{p.period_id}: role must be one of {ROLES}")
        if p.start > p.end:
            raise ValueError(f"{p.period_id}: start after end")
    for a, b in zip(periods, periods[1:]):
        if b.start <= a.end:
            raise ValueError(f"periods overlap or are out of order: {a.period_id}, {b.period_id}")
    holdouts = [p for p in periods if p.is_holdout]
    if len(holdouts) != 1 or holdouts[0] is not periods[-1]:
        raise ValueError("exactly one holdout period, and it must be the most recent (26.3)")
    if not any(p.role == "evaluation" for p in periods):
        raise ValueError("no evaluation periods")
    return periods


def definition_hash(periods: Iterable[Period]) -> str:
    body = json.dumps([asdict(p) for p in periods], default=str, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def period_rows(periods: list[Period]) -> pd.DataFrame:
    """Rows for the walk_forward_periods table."""
    h = definition_hash(periods)
    return pd.DataFrame([{"period_id": p.period_id, "ordinal": p.ordinal, "role": p.role,
                          "start_date": p.start, "end_date": p.end, "definition_hash": h} for p in periods])


# walk_forward_periods is append-only. A period_id, once written, names its role and
# window for good; ordinal and definition_hash are those of the definition it was first written under.
WRITTEN_PERIODS_SQL = "SELECT period_id, role, start_date, end_date FROM walk_forward_periods"


class PeriodChanged(ValueError):
    """The config gives a period_id already in walk_forward_periods another role or window."""


def new_period_rows(db: Any, periods: list[Period]) -> pd.DataFrame:
    """The rows of `periods` not yet in walk_forward_periods, to append with the train's own rows. Reads only.

    PeriodChanged, naming every such period, if a period_id already written has another role, start or
    end in `periods`: the model runs and predictions stored under that id would otherwise point at a
    window they were never trained or scored on. A written period missing from `periods` stays.
    """
    rows = period_rows(periods)
    written = {r.period_id: (r.role, _date(r.start_date), _date(r.end_date))
               for r in db.df(WRITTEN_PERIODS_SQL).itertuples()}
    changed = [f"{p.period_id}: written as {_show(written[p.period_id])}, configured as "
               f"{_show((p.role, p.start, p.end))}"
               for p in periods if p.period_id in written and written[p.period_id] != (p.role, p.start, p.end)]
    if changed:
        raise PeriodChanged(f"walk_forward_periods is append-only: {'; '.join(changed)}. A "
                            f"period_id keeps its role and window for good; give the new window a new period_id.")
    return rows[~rows["period_id"].isin(set(written))].reset_index(drop=True)


def _show(v: tuple[str, date, date]) -> str:
    return f"{v[0]} {v[1].isoformat()}..{v[2].isoformat()}"


def period_of(d: date, periods: list[Period]) -> Period | None:
    """The period containing date `d`, or None (off-season gaps belong to no period)."""
    for p in periods:
        if p.contains(d):
            return p
    return None


def assign_periods(dates: pd.Series, periods: list[Period]) -> pd.Series:
    """period_id for each date (None outside every period)."""
    return pd.Series([getattr(period_of(_date(d), periods), "period_id", None) for d in dates],
                     index=dates.index, dtype="object")


def evaluation_periods(periods: list[Period]) -> list[Period]:
    return [p for p in periods if p.role == "evaluation"]


def holdout_period(periods: list[Period]) -> Period:
    return next(p for p in periods if p.is_holdout)


# The latest `quant train` invocation: the rows sharing the newest row's created_at and git_commit (a
# missing commit matches a missing one). A tie on time goes to the larger model_run_id, and only rows
# with its commit are taken, so two invocations are never mixed.
LATEST_MODEL_RUNS_SQL = """
WITH last AS (SELECT created_at, git_commit FROM model_runs ORDER BY created_at DESC, model_run_id DESC LIMIT 1)
SELECT m.* FROM model_runs m JOIN last ON m.created_at = last.created_at
 AND m.git_commit IS NOT DISTINCT FROM last.git_commit
ORDER BY m.period_id, m.model_type
"""


def latest_model_runs(db: Any) -> pd.DataFrame:
    """model_runs of the latest `quant train` only (ADR 0002 item 4: the table is append-only)."""
    return db.df(LATEST_MODEL_RUNS_SQL)


def model_periods(periods: list[Period]) -> list[Period]:
    """Periods a model is fitted for: every evaluation period, plus the holdout (for scan/paper only)."""
    return [p for p in periods if p.role in ("evaluation", "holdout")]


def assert_not_holdout(period_ids: Iterable[str | None], periods: list[Period], *, purpose: str) -> None:
    """Refuse to evaluate anything in FINAL_HOLDOUT. Call before every metric/backtest/review."""
    hold = holdout_period(periods).period_id
    if any(pid == hold for pid in period_ids):
        raise HoldoutLocked(f"{purpose}: refusing to evaluate holdout period {hold} (26.3). "
                            f"The holdout is evaluated only after a freeze manifest is committed.")


def _date(v: Any) -> date:
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))
