"""ADR 0002 blocking flags: a negative control that works on a run without a clean commit.

A `quant review` whose run has no clean commit and whose 26.16 is FAIL raises a flag. While any flag
is open for a dataset, `quant freeze` refuses with ReviewBlocked, before anything is written. A later
review clears every flag raised on an earlier run when its own run has a clean commit and
26.16 PASS; 26.16 INSUFFICIENT clears nothing.

Hypothesis registration (`registry.register_now`) is allowed once while a flag is open, for the
clearing run: the first registration that writes a row after the latest open flag
was raised, and only when its hypotheses include an ACTIVE negative control, without which the run's
26.16 cannot PASS. Any other registration refuses with ReviewBlocked, before anything is written.

Flags live in the persistent append-only log `review_flags` (a raise row, then a clear row), so a
re-ingest that rebuilds the derived tables does not remove one.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..reports.diagnostics import FLAG_CLEAR, FLAG_RAISE, NEGATIVE_CONTROL, negative_control_flag
from .registry import is_active
from ..storage.database import Database

TABLE = "review_flags"
CRITERION = "26.16"

OPEN_SQL = ("SELECT * FROM review_flags r WHERE r.event = 'raise' AND r.dataset = ? AND NOT EXISTS "
            "(SELECT 1 FROM review_flags c WHERE c.event = 'clear' AND c.flag_id = r.flag_id) "
            "ORDER BY r.run_created_at, r.backtest_run_id")
RAISED_SQL = "SELECT count(*) FROM review_flags WHERE flag_id = ? AND event = 'raise'"
# Rows a registration wrote at or after a time: hypotheses and status changes.
REGISTERED_SINCE_SQL = ("SELECT count(*) AS n, min(t) AS first_at FROM ("
                        "SELECT registered_at AS t FROM hypotheses WHERE dataset = ? AND registered_at >= ? "
                        "UNION ALL "
                        "SELECT recorded_at AS t FROM hypothesis_status WHERE dataset = ? AND recorded_at >= ?)")


class ReviewBlocked(RuntimeError):
    """An open ADR 0002 flag: a clean rerun must clear it first."""


def flag_id(dataset: str, backtest_run_id: str) -> str:
    return "rf-" + hashlib.sha256(f"{dataset}|{backtest_run_id}".encode()).hexdigest()[:16]


def open_flags(db: Database) -> pd.DataFrame:
    """Raise rows of db.dataset whose flag has no clear row, oldest run first."""
    return db.df(OPEN_SQL, [db.dataset])


def _rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: (v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def record_review(db: Database, review: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Raise or clear flags for one review (build_review's dict); returns what it recorded and what is open.

    A raise is written once per flagged run: reviewing the same run again adds nothing.
    """
    run = review.get("backtest_run") or {}
    commit = run.get("git_commit")
    event = negative_control_flag({c["id"]: c for c in review["criteria"]}, commit)
    recorded: list[dict[str, Any]] = []
    if event is not None:
        rid, created = run.get("backtest_run_id"), run.get("created_at")
        if not rid or created is None:
            raise ValueError("a review flag needs the run's backtest_run_id and created_at")
        created = pd.Timestamp(created)
        created = created.tz_localize("UTC") if created.tzinfo is None else created.tz_convert("UTC")
        base = {"criterion": CRITERION, "backtest_run_id": rid, "run_created_at": created, "git_commit": commit,
                "config_hash": run.get("config_hash"), "recorded_at": pd.Timestamp(now or datetime.now(timezone.utc)),
                "referee_version": review.get("referee_version"), "dataset": review["dataset"],
                "is_synthetic": bool(review["is_synthetic"])}
        if event == FLAG_RAISE:
            fid = flag_id(review["dataset"], rid)
            new = [] if db.scalar(RAISED_SQL, [fid]) else [
                base | {"flag_id": fid, "event": FLAG_RAISE,
                        "detail": f"26.16 FAIL on {rid} ({commit!r}): a negative control works on a run "
                                  f"without a clean commit"}]
        else:
            flagged = open_flags(db)
            earlier = flagged[pd.to_datetime(flagged["run_created_at"], utc=True) < created]
            new = [base | {"flag_id": f.flag_id, "event": FLAG_CLEAR,
                           "detail": f"clears the flag on {f.backtest_run_id}: 26.16 PASS on the later clean "
                                     f"run {rid}"} for f in earlier.itertuples()]
        if new:
            df = pd.DataFrame(new)
            db.append_log(TABLE, df)
            recorded = _rows(df)
    return {"event": event, "recorded": recorded, "open": _rows(open_flags(db))}


def _blocked(db: Database, action: str, flagged: pd.DataFrame) -> str:
    runs = ", ".join(f"{f.backtest_run_id} ({f.git_commit!r}, {f.flag_id})" for f in flagged.itertuples())
    return (f"{action} refused: {len(flagged)} open blocking flag(s) in {db.sport}/{db.dataset}, 26.16 FAIL "
            f"(a negative control works) on a run without a clean commit: {runs}. A `quant review` of a later "
            f"run from a clean commit with 26.16 PASS clears it (ADR 0002).")


def require_unblocked(db: Database, action: str) -> None:
    """ReviewBlocked if any flag is open for db.dataset (ADR 0002)."""
    flagged = open_flags(db)
    if len(flagged):
        raise ReviewBlocked(_blocked(db, action, flagged))


def require_registration_allowed(db: Database, hyps: list[dict[str, Any]]) -> None:
    """ReviewBlocked unless registering `hyps` now is allowed. Reads only.

    With no flag open, any registration is. While a flag is open, only the clearing run's: no
    hypotheses or hypothesis_status row may have been written since the latest open flag was raised
    (that was the one registration), and `hyps` must hold an ACTIVE negative_control hypothesis, the
    rule `quant train` uses to run the controls, since without one 26.16 is INSUFFICIENT and the run
    cannot clear the flag. A new flag, raised after that registration, allows one more.
    """
    flagged = open_flags(db)
    if not len(flagged):
        return
    latest = pd.to_datetime(flagged["recorded_at"], utc=True).max()
    used = db.df(REGISTERED_SINCE_SQL, [db.dataset, latest, db.dataset, latest])
    n = int(used["n"].iloc[0])
    if n:
        why = (f"the one registration allowed while a flag is open, for the clearing run, was used at "
               f"{pd.Timestamp(used['first_at'].iloc[0]).isoformat()} ({n} row(s) since the latest flag was "
               f"raised at {latest.isoformat()})")
    elif not any(is_active(h) and h["family"] == NEGATIVE_CONTROL for h in hyps):
        why = (f"the hypotheses have no ACTIVE {NEGATIVE_CONTROL} hypothesis, so their run's 26.16 cannot PASS "
               f"and the run cannot clear the flag")
    else:
        return
    raise ReviewBlocked(f"{_blocked(db, 'register hypotheses', flagged)} Registration: {why}.")
