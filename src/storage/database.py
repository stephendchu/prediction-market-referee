"""DuckDB access for the research database.

    db = connect("mlb")                 # dataset from config/env (default: synthetic)
    db.insert("games", games_df)        # checked insert; constraint failures name the rows
    db.replace("observations", df, "official_date = ?", [d])
    db.append_log("paper_trades", df)   # persistent, append-only Parquet
    db.df("select * from observations where is_canonical")

One file per dataset: data/normalized/<sport>_<dataset>.duckdb. The file is a
derived cache — on a schema-version change delete it and rerun `quant ingest`.
Persistent-log views are TEMP views created per connection, so the file never
stores machine-specific paths and new Parquet files appear automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd
import pyarrow as pa

from ..config import data_dir, load_sport_config
from .schemas import (
    ALL_TABLES,
    DATASETS,
    DERIVED_TABLES,
    PERSISTENT_TABLES,
    SCHEMA_VERSION,
    Table,
    arrow_schema,
)
from .writer import AppendOnlyWriter


class SchemaMismatch(RuntimeError):
    pass


class DatasetMismatch(RuntimeError):
    pass


class ConstraintViolation(ValueError):
    """A write broke a schema rule. `check` is the rule; `rows` a sample of offenders."""

    def __init__(self, table: str, check: str, rows: pd.DataFrame | None = None, detail: str = "") -> None:
        self.table, self.check, self.rows = table, check, rows
        sample = "" if rows is None or rows.empty else f"\n{rows.head(5).to_string(index=False)}"
        super().__init__(f"{table}: violates {check}{(' — ' + detail) if detail else ''}{sample}")


def default_dataset(sport: str) -> str:
    dataset = os.environ.get("QUANT_DATASET") or load_sport_config(sport).get("dataset", "synthetic")
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    return dataset


def db_path(sport: str, dataset: str) -> Path:
    return data_dir() / "normalized" / f"{sport}_{dataset}.duckdb"


def logs_root(sport: str, dataset: str) -> Path:
    return data_dir() / "logs" / f"{sport}_{dataset}"


@dataclass
class Database:
    con: duckdb.DuckDBPyConnection
    sport: str
    dataset: str
    path: Path | None
    logs: Path
    read_only: bool = False
    _log_writer: AppendOnlyWriter | None = field(default=None, repr=False)

    @property
    def is_synthetic(self) -> bool:
        return self.dataset == "synthetic"

    # -- reads ---------------------------------------------------------------
    def df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        row = self.con.execute(sql, params or []).fetchone()
        return row[0] if row else None

    def count(self, table: str) -> int:
        return int(self.scalar(f"SELECT count(*) FROM {table}"))

    # -- writes to derived tables ----------------------------------------------
    def insert(self, table: str, df: pd.DataFrame) -> int:
        spec = _derived(table)
        try:
            return self._insert(spec, df)
        except duckdb.ConstraintException as exc:
            raise self._explain(spec, df, exc) from exc

    def replace(self, table: str, df: pd.DataFrame, where: str, params: Sequence[Any] | None = None) -> int:
        """Delete rows matching `where`, then insert `df`, atomically. Refuses an append-only table."""
        spec = _replaceable(table)
        self.con.begin()
        try:
            self.con.execute(f"DELETE FROM {table} WHERE {where}", params or [])
            n = self._insert(spec, df)
            self.con.commit()
        except duckdb.ConstraintException as exc:
            self.con.rollback()  # the failed statement aborted the transaction; explain after rollback
            raise self._explain(spec, df, exc) from exc
        except BaseException:
            self.con.rollback()
            raise
        return n

    def replace_tables(self, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
        """Empty each named table and load its frame, all in one transaction.

        Frames are loaded in the order given and emptied in reverse, so a referenced table comes before
        the tables whose foreign keys point at it (feature_definitions before feature_values). DuckDB
        checks a foreign key against child rows deleted earlier in the same transaction, so a caller
        replacing referenced keys must empty the child table in its own call first (see
        `features.mlb.write_features`). Refuses the whole call, before emptying anything, if it names
        an append-only table.
        """
        specs = [_replaceable(t) for t in frames]
        self.con.begin()
        try:
            for spec in reversed(specs):
                self.con.execute(f"DELETE FROM {spec.name}")
            counts = {}
            for spec in specs:
                try:
                    counts[spec.name] = self._insert(spec, frames[spec.name])
                except duckdb.ConstraintException as exc:
                    self.con.rollback()
                    raise self._explain(spec, frames[spec.name], exc) from exc
            self.con.commit()
        except ConstraintViolation:
            raise
        except BaseException:
            self.con.rollback()
            raise
        return counts

    def append_tables(self, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
        """Insert each frame, all in one transaction; nothing is deleted. A key that is already there
        (or any other broken rule) rolls every table back and raises ConstraintViolation."""
        specs = [_derived(t) for t in frames]
        self.con.begin()
        try:
            counts = {}
            for spec in specs:
                try:
                    counts[spec.name] = self._insert(spec, frames[spec.name])
                except duckdb.ConstraintException as exc:
                    self.con.rollback()
                    raise self._explain(spec, frames[spec.name], exc) from exc
            self.con.commit()
        except ConstraintViolation:
            raise
        except BaseException:
            self.con.rollback()
            raise
        return counts

    # -- persistent append-only logs ------------------------------------------
    def append_log(self, table: str, df: pd.DataFrame, ts: datetime | None = None) -> Path | None:
        spec = ALL_TABLES[table]
        if spec.kind != "persistent":
            raise ValueError(f"{table} is not a persistent log")
        if df.empty:
            return None
        _conform(spec, df)
        violation = find_violation(self.con, spec, df, self.is_synthetic)
        if violation is not None:
            raise violation
        if self._log_writer is None:
            schemas = {t.name: arrow_schema(t) for t in PERSISTENT_TABLES}
            self._log_writer = AppendOnlyWriter(self.logs, schemas=schemas)
        rows = pa.Table.from_pandas(df.reindex(columns=spec.column_names), schema=arrow_schema(spec),
                                    preserve_index=False).to_pylist()
        path = self._log_writer.write(table, rows, ts or datetime.now(timezone.utc))
        self._create_log_view(spec)  # an empty typed view becomes a Parquet view
        return path

    def close(self) -> None:
        self.con.close()

    # -- internals ------------------------------------------------------------
    def _insert(self, spec: Table, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        _conform(spec, df)
        self.con.register("_insert_df", df)
        try:
            self.con.execute(f"INSERT INTO {spec.name} BY NAME SELECT * FROM _insert_df")
        finally:
            self.con.unregister("_insert_df")
        return len(df)

    def _explain(self, spec: Table, df: pd.DataFrame, exc: duckdb.ConstraintException) -> ConstraintViolation:
        found = find_violation(self.con, spec, df, self.is_synthetic)
        return found or ConstraintViolation(spec.name, "key/foreign-key constraint", detail=str(exc))

    def _create_log_view(self, spec: Table) -> None:
        _create_parquet_view(self.con, spec.name, self.logs / spec.name, arrow_schema(spec))


def connect(sport: str = "mlb", dataset: str | None = None, *, path: Path | str | None = None,
            read_only: bool = False, logs: Path | None = None) -> Database:
    """Open (creating if needed) the database for one sport and dataset.

    `path=":memory:"` gives a throwaway database (tests).
    """
    dataset = dataset or default_dataset(sport)
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    file = None if str(path) == ":memory:" else Path(path) if path else db_path(sport, dataset)
    if file is not None and not read_only:
        file.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(file) if file else ":memory:", read_only=read_only)
    con.execute("SET TimeZone = 'UTC'")
    db = Database(con, sport, dataset, file, logs or logs_root(sport, dataset), read_only)
    _check_meta(db)
    if not read_only:
        init_schema(db)
    for spec in PERSISTENT_TABLES:
        db._create_log_view(spec)
    return db


def init_schema(db: Database) -> None:
    for spec in DERIVED_TABLES:
        db.con.execute(spec.ddl(db.is_synthetic))
    if db.scalar("SELECT count(*) FROM db_meta") == 0:
        meta = {"schema_version": SCHEMA_VERSION, "sport": db.sport, "dataset": db.dataset,
                "is_synthetic": str(db.is_synthetic).lower(),
                "created_at": datetime.now(timezone.utc).isoformat()}
        db.con.executemany("INSERT INTO db_meta VALUES (?, ?)", list(meta.items()))


def _check_meta(db: Database) -> None:
    exists = db.scalar("SELECT count(*) FROM information_schema.tables "
                       "WHERE table_name = 'db_meta' AND table_schema = 'main'")
    if not exists:
        return
    meta = dict(db.con.execute("SELECT key, value FROM db_meta").fetchall())
    if meta and meta.get("schema_version") != SCHEMA_VERSION:
        raise SchemaMismatch(
            f"{db.path} has schema {meta.get('schema_version')}, code expects {SCHEMA_VERSION}. "
            f"The database is a rebuildable cache: delete it and rerun `quant ingest --sport {db.sport}`.")
    if meta and (meta.get("dataset") != db.dataset or meta.get("sport") != db.sport):
        raise DatasetMismatch(f"{db.path} holds {meta.get('sport')}/{meta.get('dataset')}, "
                              f"not {db.sport}/{db.dataset}")


def _derived(table: str) -> Table:
    spec = ALL_TABLES.get(table)
    if spec is None or spec.kind != "derived":
        raise ValueError(f"{table} is not a derived table")
    return spec


def _replaceable(table: str) -> Table:
    spec = _derived(table)
    if spec.append_only:
        raise ValueError(f"{table} is append-only: rows are added, never replaced or deleted, so a rerun "
                         f"cannot overwrite an earlier run (ADR 0001 N1). Use append_tables.")
    return spec


def _conform(spec: Table, df: pd.DataFrame) -> None:
    extra = sorted(set(df.columns) - set(spec.column_names))
    if extra:
        raise ConstraintViolation(spec.name, "columns must exist in the schema", detail=f"unknown {extra}")
    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        raise ConstraintViolation(spec.name, "required columns present", detail=f"missing {missing}")
    nulls = [c for c in spec.required if df[c].isna().any()]
    if nulls:
        rows = df[df[nulls].isna().any(axis=1)]
        raise ConstraintViolation(spec.name, f"NOT NULL {nulls}", rows)


def find_violation(con: duckdb.DuckDBPyConnection, spec: Table, df: pd.DataFrame,
                   is_synthetic: bool) -> ConstraintViolation | None:
    """Evaluate the table's CHECK rules against `df`; return the first one broken."""
    cols = ", ".join(c.name for c in spec.columns)
    select = ", ".join(f"CAST(NULL AS {c.type}) AS {c.name}" if c.name not in df.columns
                       else f"CAST({c.name} AS {c.type}) AS {c.name}" for c in spec.columns)
    con.register("_check_df", df)
    try:
        con.execute(f"CREATE OR REPLACE TEMP VIEW _check_rows AS SELECT {select} FROM _check_df")
        for check in spec.all_checks(is_synthetic):
            bad = con.execute(f"SELECT {cols} FROM _check_rows WHERE NOT ({check}) LIMIT 5").df()
            if not bad.empty:
                return ConstraintViolation(spec.name, check, bad)
    finally:
        con.execute("DROP VIEW IF EXISTS _check_rows")
        con.unregister("_check_df")
    return None


_DUCK_TYPES = {pa.string(): "VARCHAR", pa.float64(): "DOUBLE", pa.int64(): "BIGINT", pa.int32(): "INTEGER",
               pa.bool_(): "BOOLEAN", pa.date32(): "DATE"}


def _duck_type(t: pa.DataType) -> str:
    if pa.types.is_timestamp(t):
        return "TIMESTAMPTZ"
    return _DUCK_TYPES[t]


def _create_parquet_view(con: duckdb.DuckDBPyConnection, view: str, root: Path | None, schema: pa.Schema) -> None:
    """TEMP view over root/**/*.parquet, or an empty view with the same columns if none exist yet."""
    if root is not None and any(root.glob("**/*.parquet")):
        glob = str(root / "**" / "*.parquet").replace("'", "''")
        body = f"SELECT * FROM read_parquet('{glob}', union_by_name = true, hive_partitioning = false)"
    else:
        cols = ", ".join(f"CAST(NULL AS {_duck_type(f.type)}) AS {f.name}" for f in schema)
        body = f"SELECT {cols} WHERE false"
    con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS {body}")
