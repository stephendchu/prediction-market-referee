"""`quant ingest`: load one IngestSource into the normalized tables.

Ingest is a rebuild, not an append: every normalized table the source covers is
emptied and reloaded in one transaction, and one `ingest_runs` row records what
was loaded (26.3 data version). Downstream tables (observations, features,
predictions, backtests) are rebuilt by the later commands.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from ..sports.base import IngestSource
from .database import Database
from .schemas import ALL_TABLES

# Tables `quant ingest` owns, in load order.
NORMALIZED_TABLES = (
    "venues", "games", "contracts", "market_quotes", "consensus_odds", "info_events",
    "mlb_players", "mlb_probable_starters", "mlb_lineups", "mlb_weather",
    "mlb_pitching_lines", "mlb_batting_lines",
)

INGEST_CODE_VERSION = "ingest-0.1.0"


@dataclass
class IngestResult:
    run_id: str
    counts: dict[str, int]
    data_version: str


def ingest(db: Database, source: IngestSource, start: date, end: date, params: dict | None = None) -> IngestResult:
    started = datetime.now(timezone.utc)
    run_id = f"ingest-{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    frames = source.load(start, end)
    unknown = set(frames) - set(NORMALIZED_TABLES)
    if unknown:
        raise ValueError(f"{source.name} returned tables ingest does not own: {sorted(unknown)}")
    full = {}
    for table in NORMALIZED_TABLES:  # every owned table is rebuilt, even if the source has no rows for it
        df = frames.get(table)
        df = pd.DataFrame(columns=ALL_TABLES[table].column_names) if df is None else df.copy()
        if not df.empty:
            df["ingest_run_id"] = run_id
        full[table] = df
    counts = db.replace_tables(full)
    version = data_version(source, start, end, full)
    from ..backtest import holdout                  # local: storage does not import backtest at module level
    run = {"run_id": run_id, "sport": db.sport, "dataset": db.dataset, "kind": source.kind,
           "started_at": started, "finished_at": datetime.now(timezone.utc),
           "params_json": json.dumps({"source": source.name, "start": str(start), "end": str(end)} | (params or {}),
                                     default=str, sort_keys=True),
           "row_counts_json": json.dumps(counts, sort_keys=True), "data_version": version,
           "code_version": INGEST_CODE_VERSION, "git_commit": holdout.git_commit(), "is_synthetic": db.is_synthetic}
    db.insert("ingest_runs", pd.DataFrame([run]))
    return IngestResult(run_id, counts, version)


def data_version(source: IngestSource, start: date, end: date, frames: dict[str, pd.DataFrame]) -> str:
    """Content hash of what was loaded (independent of run id and ingest time)."""
    h = hashlib.sha256(f"{source.name}|{start}|{end}".encode())
    for table in sorted(frames):
        df = frames[table].drop(columns=["ingest_run_id", "ingest_timestamp"], errors="ignore")
        h.update(table.encode())
        h.update(str(len(df)).encode())
        if not df.empty:
            h.update(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes())
    return h.hexdigest()[:16]
