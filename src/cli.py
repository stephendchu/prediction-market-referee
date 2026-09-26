"""`quant` command-line interface: the synthetic pipeline and its review.

    quant ingest --sport mlb --start 2023-01-01
    quant features --sport mlb --date 2026-09-24
    quant train --sport mlb
    quant backtest --sport mlb
    quant review --sport mlb
    quant freeze --sport mlb       # 26.3; refuses on synthetic data by design
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Prediction-market referee: a pre-registered evaluation harness run on synthetic worlds.")
console = Console()


@app.callback()
def main() -> None:
    """Prediction-market referee. Research only: no command places orders."""


# ---------------------------------------------------------------------------

SPORT_HELP = "Sport (mlb only)."


def _sport(sport: str) -> str:
    if sport == "all":
        console.print("note: only MLB is implemented; `--sport all` runs mlb", highlight=False)
        return "mlb"
    if sport != "mlb":
        console.print(f"[red]error:[/] only mlb is implemented, not {sport!r}")
        raise typer.Exit(code=2)
    return sport


def _open(sport: str):
    from .config import load_sport_config
    from .storage.database import connect

    sport = _sport(sport)
    return load_sport_config(sport), connect(sport)


def _selection(db):
    from .models.train import load_selection

    try:
        return load_selection(db)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=1)


def _parse_date(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def _fail(exc: Exception) -> None:
    console.print(f"[red]error:[/] {type(exc).__name__}: {exc}", highlight=False, soft_wrap=True)
    raise typer.Exit(code=1)


@app.command()
def ingest(
    sport: str = typer.Option("mlb", help=SPORT_HELP),
    start: str = typer.Option(..., help="First date (YYYY-MM-DD)."),
    end: str = typer.Option(None, help="Last date (default: the synthetic through_date)."),
) -> None:
    """Load the normalized tables from the synthetic generator."""
    from .config import data_dir

    cfg, db = _open(sport)
    s, e = date.fromisoformat(start), _parse_date(end)
    if db.is_synthetic:
        from .reports.terminal import banners
        from .sports.synthetic import TRUTH_FILE, SyntheticSource
        from .storage.ingest import ingest as run_ingest

        banners(console, synthetic=True)
        src = SyntheticSource(cfg)
        src.claim_data_root(data_dir(), db)          # one synthetic world per data root
        console.print(f"synthetic world {src.world} ({src.params.generator_version})", highlight=False)
        res = run_ingest(db, src, s, e or src.through)
        (data_dir() / "features").mkdir(parents=True, exist_ok=True)
        src.truth.to_parquet(data_dir() / "features" / TRUTH_FILE)   # generator truth: review only, never a model input
        counts, version = res.counts, res.data_version
    else:
        from .sports.base import NotAvailable

        _fail(NotAvailable(f"dataset {db.dataset!r}: this repository ships no real-data adapter; "
                           f"only the synthetic dataset can be ingested"))
    console.print(f"ingest {db.dataset}: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                  highlight=False, soft_wrap=True)
    console.print(f"data_version {version}", highlight=False)


@app.command()
def features(
    sport: str = typer.Option("mlb", help=SPORT_HELP),
    date_: str = typer.Option(None, "--date", help="Build observations and features through this date (default: all)."),
) -> None:
    """Build observations (snapshot rules) and point-in-time features through --date."""
    from .features.mlb import write_features
    from .reports.terminal import banners

    cfg, db = _open(sport)
    banners(console, synthetic=db.is_synthetic)
    fr = write_features(db, cfg, _parse_date(date_))
    b = fr.observations
    console.print(f"observations {b.n_observations} (canonical {b.n_canonical}); feature values {fr.n_values}; "
                  f"definitions {fr.n_definitions}; build {fr.build_run_id}", highlight=False, soft_wrap=True)
    for reason, n in sorted((b.exclusions or {}).items()):
        console.print(f"  excluded {reason}: {n}", highlight=False)


@app.command()
def train(
    sport: str = typer.Option("mlb", help=SPORT_HELP),
    preset: str = typer.Option(None, help="Feature preset from the config."),
    quote_quality: str = typer.Option("executable", help="executable | candle_degraded (never mixed)."),
) -> None:
    """Walk-forward training: market baseline, logistic and LightGBM residual models; selection rule."""
    from .models.train import run_training
    from .reports.terminal import print_train

    cfg, db = _open(sport)
    try:
        res = run_training(db, cfg, preset=preset, quote_quality=quote_quality,
                           log=lambda m: console.print(f"  {m}", highlight=False, markup=False))
    except (ValueError, RuntimeError) as exc:
        _fail(exc)
    print_train(console, res, synthetic=db.is_synthetic)


def _review_inputs(db, cfg, *, audit: bool = True):
    from .backtest.holdout import current_manifest, manifest_path, manifest_status
    from .config import data_dir
    from .reports import diagnostics as D
    from .sports.synthetic import TRUTH_FILE

    sel = _selection(db)
    status = manifest_status(manifest_path(db.sport), current_manifest(db, cfg, sel))
    truth = None
    tf = data_dir() / "features" / TRUTH_FILE
    if db.is_synthetic and tf.exists():   # generator truth (review only): p_true per contract
        import pandas as pd

        t = pd.read_parquet(tf, columns=["contract_id", "p_true"])
        ids = db.df("SELECT observation_id, contract_id FROM observations")
        truth = ids.merge(t, on="contract_id")[["observation_id", "p_true"]]
    return D.load_review_inputs(db, cfg, holdout_manifest=status,
                                leakage_audit=D.run_leakage_audit(db) if audit else None,
                                model_selection=sel, truth=truth, quote_quality=sel.get("quote_quality"))


@app.command()
def backtest(sport: str = typer.Option("mlb", help=SPORT_HELP)) -> None:
    """Walk-forward backtest of the selected model on the evaluation periods (never the holdout)."""
    from .backtest.engine import run_backtest
    from .reports import diagnostics as D
    from .reports.terminal import print_backtest

    cfg, db = _open(sport)
    try:
        res = run_backtest(db, cfg, _selection(db))
    except (ValueError, RuntimeError) as exc:
        _fail(exc)
    print_backtest(console, res, cfg, synthetic=db.is_synthetic)
    D.print_evaluation(D.evaluation_report(_review_inputs(db, cfg, audit=False), cfg), console)


@app.command()
def review(
    sport: str = typer.Option("mlb", help=SPORT_HELP),
    out: Path = typer.Option(None, help="Artifact directory (default: artifacts/ next to data/)."),
) -> None:
    """Sec 26 review: every criterion, the status rules applied, and the Sec 27 artifacts."""
    from .backtest import review_flags
    from .config import data_dir
    from .reports import diagnostics as D

    cfg, db = _open(sport)
    try:
        inputs = _review_inputs(db, cfg)
        rev = D.build_review(inputs, cfg)
        rev["review_flags"] = review_flags.record_review(db, rev)  # ADR 0002
    except ValueError as exc:
        _fail(exc)
    out = out or data_dir().parent / "artifacts"
    D.write_review_artifacts(rev, out, experiments=inputs.experiments, hypotheses=inputs.hypotheses,
                             holdout_manifest=inputs.holdout_manifest)
    console.print(f"artifacts written to {out}/", highlight=False)
    D.print_review(rev, console)


@app.command("freeze")
def freeze_cmd(sport: str = typer.Option("mlb", help=SPORT_HELP)) -> None:
    """Freeze the configuration before the holdout is evaluated (26.3). Live dataset only; one-way.
    Refused while an ADR 0002 blocking flag is open."""
    from .backtest import holdout, review_flags

    cfg, db = _open(sport)
    try:
        review_flags.require_unblocked(db, "freeze")
        m = holdout.freeze(holdout.manifest_path(db.sport), holdout.current_manifest(db, cfg, _selection(db)))
    except (ValueError, RuntimeError) as exc:
        _fail(exc)
    console.print(f"FROZEN {m['manifest_hash']} at {m.get('freeze_timestamp')}", highlight=False)


if __name__ == "__main__":
    app()
