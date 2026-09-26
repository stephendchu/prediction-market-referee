"""FINAL_HOLDOUT freeze manifest (spec Sec 26.3).

This module builds the mechanism only. Nothing here reads, scores, or
evaluates holdout data, and no function in this engagement evaluates the
holdout.

Status lifecycle:
  LOCKED    no freeze manifest committed; nothing may evaluate the holdout
  FROZEN    manifest committed: git commit, model config hash, feature-set
            hash, data version, thresholds hash, fee schedules, period
            definition and freeze timestamp are recorded
  UNLOCKED  evaluation permitted, because the current hashes match the frozen
            ones and the code is pinned to a real git commit

A frozen manifest is never overwritten (`AlreadyFrozen`): a failed holdout
stays a failed experiment. Only the live dataset can be frozen
(`SyntheticFreeze`): what synthetic data selects is a mechanism test. Every freeze, and every refused re-freeze, is
appended to `<manifest>.history.jsonl` next to the manifest.

Holdout registration. A live freeze also pins the registration entry of the
dataset's FINAL_HOLDOUT: `freeze` refuses unless that entry is LOCKED with the
same dates as the holdout period and still matches its hash, and unless the
fitted model artifact exists (its SHA-256 is pinned too). This repository ships
no real dataset and no registry, so a live freeze always refuses here; the
synthetic dataset is never frozen at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import stable_hash
from .walk_forward import HoldoutLocked, Period, definition_hash, latest_model_runs

MANIFEST_VERSION = "holdout-manifest-v2"
GIT_UNAVAILABLE = "unavailable: not a git repository"
DIRTY = "-dirty"  # git_commit() suffix: tracked files differ from HEAD
NOT_EVALUATED = "holdout not evaluated in this engagement"
NOT_APPLICABLE = "not applicable: synthetic dataset"
NOT_REGISTERED = "unavailable: holdout not registered"
UNAVAILABLE = "unavailable"
STATUSES = ("LOCKED", "FROZEN", "UNLOCKED")

# What a freeze pins. Any difference between frozen and current keeps the holdout locked.
FROZEN_FIELDS = ("sport", "dataset", "period_definition_hash", "holdout", "model_type", "model_config_hash",
                 "model_artifact_hash", "quote_quality", "feature_set_hash", "data_version", "thresholds_hash",
                 "fee_schedule_ids", "holdout_registration_hash", "git_commit")

# Config sections whose values must be fixed before the holdout is seen (26.3; Sec 14; 26.8-26.12).
THRESHOLD_SECTIONS = ("snapshot_policy", "decision", "fees", "execution", "backtest", "uncertainty",
                      "paper", "review")


class AlreadyFrozen(RuntimeError):
    """A FROZEN manifest already exists; it is never replaced (26.3)."""


class SyntheticFreeze(ValueError):
    """Only the live dataset can be frozen: a model chosen on synthetic data is a mechanism test."""


class HoldoutNotLocked(ValueError):
    """A live freeze needs its dataset's holdout registered and LOCKED."""


class RegistryTampered(ValueError):
    """A registry entry no longer matches its recorded hashes."""


class DirtyTree(RuntimeError):
    """A freeze or a registration from code that is not a clean commit."""


def thresholds_hash(cfg: dict[str, Any], sections: Iterable[str] = THRESHOLD_SECTIONS) -> str:
    """Hash of the decision, cost, uncertainty, paper and acceptance settings in a sport config."""
    return stable_hash({s: cfg.get(s) for s in sections})


def build_manifest(*, sport: str, dataset: str, periods: list[Period], holdout: Period, model_type: str,
                   model_config_hash: str, model_artifact_hash: str, quote_quality: str, feature_set_hash: str,
                   data_version: str, thresholds_hash: str, fee_schedule_ids: Iterable[str],
                   holdout_registration: dict[str, Any] | None, git_commit: str | None,
                   now: datetime | None = None) -> dict[str, Any]:
    """The manifest for the current configuration (status LOCKED until `freeze` commits it).

    `holdout_registration` is the registration entry of this dataset's holdout, or None where
    none applies (synthetic data) or none is registered.
    """
    if not holdout.is_holdout or holdout not in periods or holdout != periods[-1]:
        raise ValueError(f"{holdout.period_id} is not the final holdout period of `periods` (26.3)")
    if holdout_registration is not None:
        reg_hash = entry_hash(holdout_registration)
    else:
        reg_hash = NOT_REGISTERED if dataset == "live" else NOT_APPLICABLE
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "LOCKED",
        "sport": sport,
        "dataset": dataset,
        "freeze_timestamp": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "git_commit": git_commit or GIT_UNAVAILABLE,
        "model_type": model_type,
        "model_config_hash": model_config_hash,
        "model_artifact_hash": model_artifact_hash,
        "quote_quality": quote_quality,
        "feature_set_hash": feature_set_hash,
        "data_version": data_version,
        "thresholds_hash": thresholds_hash,
        "fee_schedule_ids": sorted(fee_schedule_ids),
        "period_definition_hash": definition_hash(periods),
        "periods": [{"period_id": p.period_id, "role": p.role, "start": p.start.isoformat(),
                     "end": p.end.isoformat()} for p in periods],
        "holdout": {"period_id": holdout.period_id, "start": holdout.start.isoformat(),
                    "end": holdout.end.isoformat()},
        "holdout_registration": holdout_registration,
        "holdout_registration_hash": reg_hash,
        "holdout_evaluated": False,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash of the frozen fields (paper_trades.freeze_manifest_hash, 26.18)."""
    return stable_hash({k: manifest.get(k) for k in FROZEN_FIELDS})


def history_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.history.jsonl")


def load_manifest(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def freeze(path: Path | str, manifest: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Commit `manifest` as FROZEN at `path`. Refuses to replace an existing frozen manifest."""
    path = Path(path)
    at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    missing = [k for k in FROZEN_FIELDS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing {missing}")
    if manifest.get("dataset") != "live":
        raise SyntheticFreeze(f"refusing to freeze dataset {manifest.get('dataset')!r}: model selection on "
                              f"synthetic data tests the mechanism and is never the frozen model choice")
    if not is_clean_commit(manifest.get("git_commit")):
        raise DirtyTree(f"refusing to freeze: {tree_state(manifest.get('git_commit'))}; a freeze requires a "
                        f"clean tree")
    if str(manifest["model_artifact_hash"]).startswith(UNAVAILABLE):
        raise ValueError(f"refusing to freeze: {manifest['model_artifact_hash']} (the fitted model is pinned)")
    _check_registration_for_freeze(manifest)
    existing = load_manifest(path)
    if existing is not None and existing.get("status") == "FROZEN":
        _append_history(path, {"event": "freeze_refused", "at": at,
                               "existing_manifest_hash": existing.get("manifest_hash"),
                               "attempted_manifest_hash": manifest_hash(manifest)})
        raise AlreadyFrozen(f"{path} is already FROZEN (manifest {existing.get('manifest_hash')}); "
                            f"a frozen holdout is never re-frozen (26.3)")
    frozen = {**manifest, "status": "FROZEN", "manifest_hash": manifest_hash(manifest)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    _append_history(path, {"event": "freeze", "at": at, "manifest": frozen})
    return frozen


def _check_registration_for_freeze(manifest: dict[str, Any]) -> None:
    """The holdout a live freeze pins must be registered, LOCKED, and the period the manifest holds out."""
    entry = manifest.get("holdout_registration")
    if entry is None:
        raise HoldoutNotLocked("refusing to freeze: no holdout is registered for this dataset")
    if manifest.get("holdout_registration_hash") != entry_hash(entry) or entry.get("entry_hash") != entry_hash(entry):
        raise RegistryTampered("refusing to freeze: the holdout registration does not match its hash")
    if entry.get("status") != "LOCKED":
        raise HoldoutNotLocked(f"refusing to freeze: the {entry.get('name')} holdout is {entry.get('status')}, "
                               f"not LOCKED")
    lock, held = entry.get("lock") or {}, manifest.get("holdout") or {}
    if (held.get("period_id"), held.get("start"), held.get("end")) != (
            entry.get("period_id"), lock.get("start"), lock.get("end")):
        raise HoldoutNotLocked(f"refusing to freeze: the manifest holds out {held}, but the {entry.get('name')} "
                               f"holdout is locked as {entry.get('period_id')} {lock.get('start')}..{lock.get('end')}")


def _append_history(path: Path, event: dict[str, Any]) -> None:
    hist = history_path(path)
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def mismatches(frozen: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Frozen fields whose current value differs, plus reasons the freeze can't be trusted."""
    bad = [k for k in FROZEN_FIELDS if frozen.get(k) != current.get(k)]
    if frozen.get("manifest_hash") != manifest_hash(frozen):
        bad.append("manifest_hash (frozen manifest edited after freeze)")
    commit = frozen.get("git_commit")
    if "git_commit" not in bad and commit in (None, GIT_UNAVAILABLE):
        bad.append("git_commit (unavailable: code version not pinned)")
    elif "git_commit" not in bad and not is_clean_commit(commit):
        bad.append(f"git_commit ({commit}: uncommitted code)")
    return bad


def check_unlock(frozen: dict[str, Any], current: dict[str, Any]) -> None:
    """Raise HoldoutLocked unless `frozen` is a FROZEN manifest matching `current` exactly."""
    if frozen.get("status") != "FROZEN":
        exc = HoldoutLocked(f"holdout is {frozen.get('status', 'LOCKED')}: no freeze manifest committed (26.3)")
        exc.mismatches = ["status"]
        raise exc
    bad = mismatches(frozen, current)
    if bad:
        exc = HoldoutLocked(f"holdout stays locked: current configuration differs from the frozen "
                            f"manifest in {bad} (26.3)")
        exc.mismatches = bad
        raise exc


def manifest_status(path: Path | str, current: dict[str, Any]) -> dict[str, Any]:
    """The holdout status report `quant review` writes to artifacts/holdout_manifest.json."""
    frozen = load_manifest(path)
    report = {
        "status": "LOCKED",
        "holdout": current.get("holdout"),
        "holdout_evaluated": False,
        "note": NOT_EVALUATED,
        "manifest_path": str(path),
        "current": {k: current.get(k) for k in FROZEN_FIELDS} | {"manifest_hash": manifest_hash(current)},
        "frozen": frozen,
        "mismatches": [],
    }
    if frozen is None:
        report["detail"] = "no freeze manifest committed; nothing may evaluate the holdout"
        return report
    try:
        check_unlock(frozen, current)
    except HoldoutLocked as exc:
        report["status"] = "FROZEN" if frozen.get("status") == "FROZEN" else "LOCKED"
        report["mismatches"] = exc.mismatches
        report["detail"] = str(exc)
        return report
    report["status"] = "UNLOCKED"
    report["detail"] = "current hashes match the frozen manifest; evaluation would be permitted"
    return report


# ---------------------------------------------------------------------------
# The current configuration as a manifest (review, paper, freeze)
# ---------------------------------------------------------------------------

def manifest_path(sport: str) -> Path:
    """Where the sport's freeze manifest is committed: config/<sport>_holdout_freeze.json."""
    from ..config import CONFIG_DIR
    return CONFIG_DIR / f"{sport}_holdout_freeze.json"


def git_commit() -> str | None:
    """HEAD of the repository, `<HEAD>-dirty` if tracked files differ from it, or None (no git, or no
    commits yet). A run from uncommitted code is recorded, and tagged.

    Untracked files do not count, as in `git describe --dirty`: a new file changes a run only through
    a tracked file that uses it, and data/ is ignored. If the state of the tree cannot be read, it
    counts as dirty: unknown is not clean.
    """
    import subprocess

    from ..config import REPO_ROOT
    try:
        out = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    head = out.stdout.strip() if out.returncode == 0 else ""
    if not head:
        return None
    try:  # `git status` refreshes the index first, so a new mtime alone is not a change
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return head + DIRTY
    return head if st.returncode == 0 and not st.stdout.strip() else head + DIRTY


def is_clean_commit(commit: Any) -> bool:
    """A commit that pins the code: a non-empty string with no surrounding whitespace,
    not `unavailable...` and not `-dirty`. Anything else is not clean, fail-closed:
    None, NaN, pd.NA, blank, padded, or not a string."""
    if not isinstance(commit, str) or not commit or commit != commit.strip():
        return False
    return not commit.startswith(UNAVAILABLE) and not commit.endswith(DIRTY)


def tree_state(commit: Any) -> str:
    """Why `commit` does not pin the code, in words (for refusals)."""
    if not isinstance(commit, str) or not commit.strip() or commit.startswith(UNAVAILABLE):
        return "no git commit (missing or blank, not a git repository, or no commits yet)"
    if commit != commit.strip():
        return f"the commit {commit!r} is not a bare commit id (surrounding whitespace)"
    return f"tracked files differ from the commit ({commit})"


def require_clean_tree(action: str, ref: str = "ADR 0001 Q3") -> str:
    """HEAD, if the tracked tree matches it; else DirtyTree. For the steps that run only from
    committed code, such as hypothesis registration (the freeze checks its manifest)."""
    commit = git_commit()
    if not is_clean_commit(commit):
        raise DirtyTree(f"refusing to {action}: {tree_state(commit)}; commit first, and run it from a clean "
                        f"tree ({ref})")
    return commit


def current_manifest(db: Any, cfg: dict[str, Any], selection: dict[str, Any],
                     now: datetime | None = None) -> dict[str, Any]:
    """build_manifest for what `quant train` / `quant ingest` last produced in `db`.

    model_config_hash is the config_hash of the selected model type's final-period
    run (the model a freeze would pin) and model_artifact_hash the SHA-256 of its
    pickle; data_version is the latest ingest run's. No holdout registry ships with
    this repository, so the registration is None (a live freeze refuses).
    """
    from .execution import fee_schedule_rows
    from .walk_forward import holdout_period, load_periods

    periods = load_periods(cfg, db.dataset)
    hold = holdout_period(periods)
    model_type = selection["selected"]
    quote_quality = str(selection.get("quote_quality") or "executable")
    run = latest_model_runs(db)                     # the latest `quant train` only (ADR 0002 item 4)
    run = run[(run["model_type"] == model_type) & (run["period_id"] == hold.period_id)]
    ingest = db.df("SELECT data_version FROM ingest_runs ORDER BY started_at DESC LIMIT 1")
    return build_manifest(
        sport=db.sport, dataset=db.dataset, periods=periods, holdout=hold, model_type=model_type,
        model_config_hash=str(run["config_hash"].iloc[0]) if len(run) else "unavailable: no final-period model",
        model_artifact_hash=artifact_hash(run["artifact_path"].iloc[0] if len(run) else None),
        quote_quality=quote_quality, feature_set_hash=str(selection.get("feature_set_hash")),
        data_version=str(ingest["data_version"].iloc[0]) if len(ingest) else "unavailable: no ingest run",
        thresholds_hash=thresholds_hash(cfg), fee_schedule_ids=fee_schedule_rows()["fee_schedule_id"].tolist(),
        holdout_registration=None, git_commit=git_commit(), now=now)


def artifact_hash(path: Path | str | None) -> str:
    """SHA-256 of a fitted model's pickle, or why there is none."""
    if not path:
        return "unavailable: no final-period model"
    path = Path(path)
    if not path.is_file():
        return f"unavailable: model artifact missing ({path})"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def entry_hash(entry: dict[str, Any]) -> str:
    """Hash of a registration entry, all of it but the hash itself (pinned by a freeze manifest)."""
    return stable_hash({k: v for k, v in entry.items() if k != "entry_hash"})
