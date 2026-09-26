"""Configuration loading: config/<sport>.yaml, plus QUANT_* environment overrides."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def data_dir() -> Path:
    override = os.environ.get("QUANT_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data"


@lru_cache(maxsize=8)
def _parse_sport_config(sport: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{sport}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config for sport '{sport}' at {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_sport_config(sport: str) -> dict[str, Any]:
    """A fresh copy of the parsed config: a caller that edits it cannot change what the next caller sees."""
    return copy.deepcopy(_parse_sport_config(sport))


def stable_hash(obj: Any, n: int = 16) -> str:
    """Order-independent content hash of JSON-able config (26.3 config/feature-set hashes)."""
    body = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Strategy family and feature presets
# ---------------------------------------------------------------------------

STRATEGY_FAMILIES = ("outcome_residual", "market_behavior")
BASEBALL_FAMILIES = ("starter", "offense", "bullpen", "environment")


class BaseballGateClosed(RuntimeError):
    """Baseball feature families on live data before a recorded decision opens them."""


def strategy_family(cfg: dict[str, Any]) -> str:
    """The configured strategy family. Only outcome_residual is implemented; market_behavior (26.20) is a stub."""
    fam = (cfg.get("strategy") or {}).get("family", "outcome_residual")
    if fam not in STRATEGY_FAMILIES:
        raise ValueError(f"strategy.family must be one of {STRATEGY_FAMILIES}, got {fam!r}")
    if fam == "market_behavior":
        raise NotImplementedError("strategy.family market_behavior is scored by the 26.20 evaluator, which is "
                                  "not implemented; no market-behavior result may be produced")
    return fam


def feature_preset(cfg: dict[str, Any], dataset: str, preset: str | None = None) -> tuple[str, list[str]]:
    """(preset name, feature families) for a run, after the live-data baseball gate."""
    fc = cfg.get("features") or {}
    presets = fc.get("presets") or {}
    name = preset or fc.get("preset")
    if name not in presets:
        raise ValueError(f"feature preset {name!r} is not one of features.presets {sorted(presets)}")
    families = [str(f) for f in presets[name]]
    baseball = [f for f in families if f in BASEBALL_FAMILIES]
    if baseball and dataset == "live":
        ref = fc.get("baseball_gate_decision")
        decisions = (REPO_ROOT / "docs" / "decisions").resolve()
        target = (REPO_ROOT / str(ref)).resolve() if ref else None
        ok = target is not None and target.parent == decisions and target.suffix == ".md" and target.is_file()
        if not ok:
            raise BaseballGateClosed(
                f"preset {name!r} uses baseball families {baseball}; on live data they wait for a recorded "
                f"decision: a docs/decisions/ record set as features.baseball_gate_decision (got {ref!r})")
    return name, families
