"""Append-only Parquet writer for the persistent log tables. Never overwrites."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class AppendOnlyWriter:
    def __init__(self, root: Path, schemas: dict[str, pa.Schema]) -> None:
        self.root = root
        self.schemas = schemas
        self.written: dict[str, int] = {}

    def write(self, table: str, rows: list[dict[str, Any]], cycle_ts: datetime) -> Path | None:
        if not rows:
            return None
        schema = self.schemas[table]
        arrow = pa.Table.from_pylist([_conform(r, schema) for r in rows], schema=schema)
        part = self.root / table / f"date={cycle_ts:%Y-%m-%d}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / f"{table}-{cycle_ts:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        if path.exists():  # uuid collision: refuse rather than overwrite
            raise FileExistsError(path)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(arrow, tmp, compression="zstd")
        tmp.rename(path)  # atomic within a filesystem; readers never see partial files
        self.written[table] = self.written.get(table, 0) + len(rows)
        return path


def _conform(row: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    missing = [f.name for f in schema if not f.nullable and row.get(f.name) is None]
    if missing:
        raise ValueError(f"row missing required fields {missing}")
    extra = set(row) - set(schema.names)
    if extra:
        raise ValueError(f"row has fields not in schema: {sorted(extra)}")
    return {name: row.get(name) for name in schema.names}
