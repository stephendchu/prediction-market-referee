"""Terminal output for train, backtest, scan, explain and paper (spec Sec 4, 21, 22).

The review and the Sec 17 evaluation print from `diagnostics` (print_review, print_evaluation).
Every command on synthetic data opens and closes with the SYNTHETIC banner; candle-quote runs
also carry DEGRADED EXECUTION.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..storage.schemas import DEGRADED_EXECUTION
from .diagnostics import BANNER

SYNTHETIC_BANNER = BANNER  # the review's wording, so every command says the same thing


def say(console: Console, text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


def banners(console: Console, *, synthetic: bool, notes: Iterable[str] = ()) -> None:
    if synthetic:
        console.print(Panel(SYNTHETIC_BANNER, style="bold black on yellow"), markup=False, highlight=False)
    if DEGRADED_EXECUTION in set(notes):
        console.print(Panel(DEGRADED_EXECUTION, style="bold white on red"), markup=False, highlight=False)


def _num(v: Any, fmt: str = "{:.3f}") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return fmt.format(v)


def _pct(v: Any) -> str:
    return _num(None if v is None else 100 * v, "{:+.1f}%")


def _table(title: str, df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 40) -> Table:
    df = df[cols] if cols else df
    t = Table(title=title)
    for c in df.columns:
        t.add_column(str(c), overflow="fold")
    for rec in df.head(max_rows).to_dict(orient="records"):
        t.add_row(*[_cell(rec[c]) for c in df.columns])
    return t


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.5g}"
    return "n/a" if v is None else str(v)


# ---------------------------------------------------------------------------

def print_train(console: Console, res: Any, *, synthetic: bool) -> None:
    banners(console, synthetic=synthetic)
    sel = res.selection
    say(console, f"model runs {len(res.model_runs)}, predictions {res.n_predictions}, "
                 f"hypotheses registered {res.hypotheses_registered}, experiments logged {res.experiments_logged} "
                 f"(of {res.experiments_total})")
    if len(res.per_period):
        console.print(_table("Per evaluation period (out-of-sample)", res.per_period))
    say(console, f"selected: {sel.get('selected')}  quote_quality: {sel.get('quote_quality')}")
    say(console, f"reason: {sel.get('reason')}")
    if sel.get("status"):
        say(console, f"status: {sel['status']}")
    for n in res.notes:
        say(console, f"note: {n}")
    banners(console, synthetic=synthetic)


def print_backtest(console: Console, res: Any, cfg: Mapping[str, Any], *, synthetic: bool) -> None:
    banners(console, synthetic=synthetic, notes=res.notes)
    ex = cfg["execution"]
    rule = ex.get("exit_rule", "hold_to_settlement")
    say(console, f"exit rule: {rule}" + (f" (T-{ex.get('exit_offset_min')})" if rule == "exit_at_offset" else ""))
    rows = []
    for off, s in res.summary.items():
        rows.append({"offset": f"T-{off}", "observations": s["observations"], "trades (baseline)": s["trades"],
                     **{f"net P&L {k}": round(v, 2) for k, v in s["by_scenario"].items()}})
    console.print(_table("Backtest by snapshot offset (T-30 canonical; others are sensitivity)", pd.DataFrame(rows)))
    for off, s in res.summary.items():
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s["pass_reasons"].items()))
        say(console, f"T-{off} pass reasons (baseline): {reasons or 'none'}")
    say(console, f"decisions stored {res.decisions}; H_MLB_003 experiments logged {res.experiments_logged}")
    for n in res.notes:
        if n != DEGRADED_EXECUTION:
            say(console, f"note: {n}")


# ---------------------------------------------------------------------------
# Scan (Sec 4 example, Sec 22 ranking)
# ---------------------------------------------------------------------------

def _team_line(r: Mapping[str, Any]) -> str:
    return f"{r.get('away_team')} @ {r.get('home_team')}: {r.get('yes_team')} win  [{r['contract_id']}]"


def print_scan(console: Console, scan: pd.DataFrame, slate: Any, *, synthetic: bool, top: int = 10) -> None:
    banners(console, synthetic=synthetic, notes=slate.notes)
    run = slate.model.run
    offs = sorted({int(x) for x in slate.rows.get("snapshot_offset_min", [])})
    say(console, f"MLB MARKET SCAN  {slate.day}  (T-{offs[0] if offs else 'n/a'} snapshots, {slate.quote_quality} "
                 f"quotes; model {run['model_type']} {run['model_run_id']}, "
                 f"trained through {run['train_data_through']})")
    say(console, "-" * 64)
    candidates = int((scan["decision"] != "PASS").sum()) if len(scan) else 0
    for r in scan.head(top).to_dict(orient="records"):
        side = r.get("side")
        lo, hi = r.get("lower_bound"), r.get("upper_bound")
        cost = None
        if r.get("execution_price") is not None and not pd.isna(r.get("execution_price")) and r.get("quantity"):
            cost = (r["execution_price"] - r["quoted_price"]) + r["calculated_fee"] / r["quantity"]
        say(console, _team_line(r))
        say(console, f"  Market YES bid/ask:  {_num(r.get('market_yes_bid'))} / {_num(r.get('market_yes_ask'))}"
                     f"  (mid {_num(r.get('p_market'))})")
        say(console, f"  Consensus market:    {_num(r.get('p_consensus'))}")
        say(console, f"  Model fair value:    {_num(r.get('p_model'))}  (YES)")
        say(console, f"  Conservative fair:   {_num(r.get('p_conservative'))}  ({side or 'n/a'} side)")
        say(console, f"  Gross edge:          {_pct(r.get('predicted_edge'))}  vs {side or 'n/a'} price "
                     f"{_num(r.get('quoted_price'))}")
        say(console, f"  Costs:               {_pct(None if cost is None else -cost)}  (penalty + fee per contract)"
                     if cost is not None else "  Costs:               n/a (no fill at the flat stake)")
        say(console, f"  Uncertainty:         {_num(None if lo is None or hi is None else hi - lo)}  "
                     f"(interval {_num(lo)} to {_num(hi)})")
        say(console, f"  Tradable edge:       {_pct(r.get('net_edge'))}")
        say(console, f"  Decision:            {r['decision']}" + (f" ({r['pass_reason']})" if r.get("pass_reason") else ""))
    say(console, "-" * 64)
    say(console, f"TRADE CANDIDATES: {candidates}   PASSES: {len(scan) - candidates}   (of {len(scan)} contracts)")
    say(console, "Ranked by |net edge| (Sec 22). `quant explain --contract ID` shows the largest adjustments.")
    for n in slate.notes:
        if n != DEGRADED_EXECUTION:
            say(console, f"note: {n}")
    banners(console, synthetic=synthetic, notes=slate.notes)


def print_explain(console: Console, ex: Any, *, synthetic: bool, top: int = 8) -> None:
    banners(console, synthetic=synthetic, notes=ex.notes)
    r, d = ex.row, ex.decision
    say(console, _team_line(r))
    say(console, f"snapshot {r['snapshot_time']} (T-{int(r['snapshot_offset_min'])}); quote {r['quote_ts']}; "
                 f"prediction: {ex.source}")
    say(console, f"market YES bid/ask {_num(r['market_yes_bid'])} / {_num(r['market_yes_ask'])}; "
                 f"p_market (mid) {_num(r['p_market'])}; consensus {_num(r.get('p_consensus'))}")
    say(console, f"model (raw) {_num(r['p_model_raw'])}; calibrated {_num(r['p_model'])}; "
                 f"interval {_num(r['lower_bound'])} to {_num(r['upper_bound'])}")
    say(console, f"conservative YES {_num(r['p_conservative_yes'])}; conservative NO {_num(r['p_conservative_no'])}")
    say(console, f"decision (baseline penalty, flat stake): {d['decision']}"
                 + (f" ({d['pass_reason']})" if isinstance(d.get("pass_reason"), str) else "")
                 + f"; side {d.get('side')}; gross edge {_pct(d.get('predicted_edge'))}; "
                   f"net edge {_pct(d.get('net_edge'))}")
    if len(ex.contributions):
        say(console, "Largest adjustments (logit contribution to the model; market term and intercept excluded):")
        for name, v in ex.contributions.head(top).items():
            say(console, f"  {v:+.4f}  {name}")
    else:
        say(console, "market model: no feature adjustments")
    if "outcome" in r and not pd.isna(r["outcome"]):
        say(console, f"outcome: {'YES' if int(r['outcome']) == 1 else 'NO'}")
    banners(console, synthetic=synthetic, notes=ex.notes)


# ---------------------------------------------------------------------------
# Paper (Sec 21)
# ---------------------------------------------------------------------------

def print_paper(console: Console, res: Any, *, synthetic: bool) -> None:
    from ..portfolio.paper import WITHHELD

    banners(console, synthetic=synthetic, notes=res.notes)
    q = res.qualification
    say(console, f"PAPER TRADING {res.day} (no orders are placed). manifest {q.status}; "
                 f"{'trades qualify for 26.18' if q.manifest_hash else 'trades do not qualify for 26.18'}")
    say(console, f"exits recorded {len(res.exits)}; decisions settled {len(res.settlements)}; "
                 f"decisions logged {len(res.decisions)}; already logged earlier {res.already_logged}")
    d = res.decisions
    if len(d):
        cols = ["contract_id", "decision", "pass_reason", "side", "bid", "ask", "model_probability",
                "conservative_probability", "net_edge", "requested_stake", "allowed_stake", "capped_by",
                "hypothetical_size", "entry_price", "stake"]
        console.print(_table("Decisions logged this run", d, cols))
        buys = int((d["decision"] != "PASS").sum())
        say(console, f"TRADE CANDIDATES: {buys}   PASSES: {len(d) - buys}")
    led = res.ledger
    if len(led):
        buys = led[led["decision"] != "PASS"]
        settled = buys[buys["net_pnl"].notna()]
        say(console, f"ledger: {len(led)} decisions, {len(buys)} paper trades, {len(settled)} settled")
        qual = settled[settled["freeze_manifest_hash"].notna()]
        if len(qual):
            say(console, f"qualifying settled trades {len(qual)}: net P&L {qual['net_pnl'].sum():+.2f} "
                         f"on stake {qual['stake'].sum():.2f}")
        if len(settled) - len(qual):
            say(console, f"{len(settled) - len(qual)} settled trades: {WITHHELD}")
    for n in res.notes:
        if n != DEGRADED_EXECUTION:
            say(console, f"note: {n}")
    banners(console, synthetic=synthetic, notes=res.notes)
