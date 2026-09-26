"""Sport-agnostic data-source interface.

The interface is a Protocol so a real-data adapter and the synthetic generator are
interchangeable. This repository ships only the synthetic generator. An adapter
with no real implementation raises NotAvailable with the reason; it never returns
fabricated data.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

# ---------------------------------------------------------------------------
# Data-source interface used by `quant ingest`.
#
# A source returns normalized rows, keyed by table name (src/storage/schemas.py),
# for games whose official_date falls in [start, end]. Every row carries the
# PROVENANCE columns; `ingest_run_id` is filled in by the caller. Everything
# downstream is identical across datasets.
#
# A source must not claim more than it proves: availability_basis = source_declared
# or conservative, and a pregame fact whose publication time is unknown (e.g. who
# actually started, the posted lineup) gets available_at = scheduled start, which
# keeps it out of every pregame snapshot (26.11).
# ---------------------------------------------------------------------------

class IngestSource(Protocol):
    name: str                 # e.g. synthetic
    kind: str                 # ingest_runs.kind: synthetic | snapshots | historical

    def load(self, start: date, end: date) -> dict[str, pd.DataFrame]: ...


class NotAvailable(NotImplementedError):
    """A source with no real implementation. Never replaced by fabricated data."""
