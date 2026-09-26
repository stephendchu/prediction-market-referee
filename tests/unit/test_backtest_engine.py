"""Backtest decision rule (Sec 14, 26.6-26.9): thresholds, sides, scenarios, one position per game."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.backtest.engine import Policy, decide, one_position_per_game
from src.backtest.execution import estimated_net_edge, Quote
from src.storage.schemas import BACKTEST_DECISIONS, PASS_REASONS

WHEN = datetime(2025, 6, 1, 22, 35, tzinfo=timezone.utc)
POL = Policy(minimum_edge=0.04, maximum_spread=0.10, min_contracts=10, stake=100.0, max_fraction_top=0.10,
             maker_or_taker="taker", baseline=0.25, scenarios=(0.0, 0.25, 0.5, 1.0))


def row(p_yes=0.50, bid=0.45, ask=0.47, outcome=1, *, cons_yes=None, cons_no=None, contract="C-H", game="G1",
        bid_size=5000.0, ask_size=5000.0, **over) -> dict:
    r = {"model_run_id": "mr-1", "observation_id": f"o-{contract}", "period_id": "2025H1", "game_id": game,
         "contract_id": contract, "snapshot_time": WHEN, "p_market": (bid + ask) / 2, "p_model": p_yes,
         "p_conservative_yes": p_yes if cons_yes is None else cons_yes,
         "p_conservative_no": 1 - p_yes if cons_no is None else cons_no,
         "market_yes_bid": bid, "market_yes_ask": ask, "bid_size": bid_size, "ask_size": ask_size,
         "depth_json": None, "spread": ask - bid, "outcome": outcome}
    return r | over


def by_scenario(rows: list[dict]) -> dict[str, dict]:
    return {r["scenario"]: r for r in rows}


def test_one_row_per_scenario_and_schema_columns():
    rows = decide(row(), POL)
    assert [r["scenario"] for r in rows] == ["pen_0", "pen_25", "pen_50", "pen_100"]
    assert set().union(*rows) <= set(BACKTEST_DECISIONS.column_names)


def test_buys_yes_when_conservative_edge_clears_threshold_and_settles():
    r = by_scenario(decide(row(p_yes=0.60, bid=0.45, ask=0.47, outcome=1), POL))["pen_25"]
    assert r["decision"] == "BUY_YES" and r["side"] == "yes" and r["pass_reason"] is None
    assert r["quoted_price"] == pytest.approx(0.47)
    assert r["execution_price"] == pytest.approx(0.47 + 0.25 * 0.02)
    assert r["predicted_edge"] == pytest.approx(0.60 - 0.47)
    assert r["net_edge"] >= POL.minimum_edge
    q = r["quantity"]
    assert r["gross_pnl"] == pytest.approx(q * (1 - 0.47))
    assert r["net_pnl"] == pytest.approx(r["gross_pnl"] - r["slippage_cost"] - r["calculated_fee"])
    assert r["side_won"] == 1


def test_buys_no_at_one_minus_bid():
    r = by_scenario(decide(row(p_yes=0.35, bid=0.45, ask=0.47, outcome=1), POL))["pen_25"]
    assert r["decision"] == "BUY_NO" and r["side"] == "no"
    assert r["quoted_price"] == pytest.approx(0.55)
    assert r["side_won"] == 0 and r["net_pnl"] < 0
    assert r["p_model"] == pytest.approx(0.35)  # stored YES-oriented


def test_uses_conservative_probability_not_point_estimate():
    # point estimate says +13 points, the conservative bound says +1: no trade
    rows = decide(row(p_yes=0.60, cons_yes=0.48, bid=0.45, ask=0.47), POL)
    assert {r["decision"] for r in rows} == {"PASS"}
    assert {r["pass_reason"] for r in rows} == {"below_min_edge"}


def test_threshold_is_net_of_fee_and_penalty():
    q = Quote(0.45, 0.47, 5000.0, 5000.0)
    screen = estimated_net_edge(0.515, q, "yes", 0.25, 100.0, WHEN)
    assert screen < 0.04 < 0.515 - 0.47  # raw edge 4.5 points, net of fee + slippage below 4
    assert by_scenario(decide(row(p_yes=0.515), POL))["pen_25"]["pass_reason"] == "below_min_edge"


def test_wide_spread_and_no_price():
    wide = decide(row(p_yes=0.9, bid=0.30, ask=0.45), POL)
    assert {r["pass_reason"] for r in wide} == {"wide_spread"}
    one_sided = decide(row(p_yes=0.9, bid=float("nan"), ask=0.45), POL)
    assert {r["pass_reason"] for r in one_sided} == {"no_price"}
    crossed = decide(row(p_yes=0.9, bid=0.47, ask=0.45), POL)
    assert {r["pass_reason"] for r in crossed} == {"no_price"}


def test_insufficient_depth_is_a_pass_with_side_recorded():
    rows = decide(row(p_yes=0.60, ask_size=50.0), POL)  # 10% of 50 = 5 < 10 contracts
    r = by_scenario(rows)["pen_25"]
    assert r["decision"] == "PASS" and r["pass_reason"] == "insufficient_depth" and r["side"] == "yes"
    assert pd.isna(r.get("quantity"))


def test_decision_is_made_once_at_baseline_scenarios_reprice():
    # clears 4 points at the 25% penalty; the 100% penalty re-prices the same trade (lower net edge, still traded)
    rows = by_scenario(decide(row(p_yes=0.545, bid=0.45, ask=0.47), POL))
    assert rows["pen_25"]["decision"] == "BUY_YES"
    assert {r["decision"] for r in rows.values()} == {"BUY_YES"}
    assert rows["pen_100"]["net_edge"] < rows["pen_25"]["net_edge"] < rows["pen_0"]["net_edge"]
    assert rows["pen_100"]["net_pnl"] < rows["pen_0"]["net_pnl"]
    # a trade rejected at baseline is not taken in the cheaper pen_0 scenario either
    rej = by_scenario(decide(row(p_yes=0.528, bid=0.45, ask=0.47), POL))
    assert rej["pen_25"]["decision"] == "PASS"
    assert rej["pen_0"]["net_edge"] > rej["pen_25"]["net_edge"]
    assert rej["pen_0"]["decision"] == "PASS"


def test_unsettled_trade_has_no_pnl():
    r = by_scenario(decide(row(p_yes=0.60, outcome=None), POL))["pen_25"]
    assert r["decision"] == "BUY_YES" and r["outcome"] is None and r["side_won"] is None
    assert r.get("net_pnl") is None and r.get("gross_pnl") is None and r["exit_status"] == "held"


def test_thresholds_come_from_config():
    cfg = {"decision": {"minimum_edge": 0.02, "maximum_spread": 0.05, "minimum_fill_contracts": 3},
           "execution": {"spread_penalty_scenarios": [0.0, 1.0], "baseline_penalty": 0.5,
                         "max_fraction_of_top_of_book": 0.2},
           "backtest": {"stake_dollars": 50}, "fees": {"maker_or_taker": "taker"}}
    p = Policy.from_config(cfg)
    assert (p.minimum_edge, p.maximum_spread, p.min_contracts, p.stake, p.baseline) == (0.02, 0.05, 3, 50.0, 0.5)
    assert p.scenarios == (0.0, 0.5, 1.0)  # baseline always included


def test_one_position_per_game_keeps_better_mirror():
    home = decide(row(p_yes=0.60, bid=0.45, ask=0.47, contract="C-H"), POL)
    away = decide(row(p_yes=0.42, bid=0.52, ask=0.54, contract="C-A", outcome=0), POL)  # BUY_NO on away
    other = decide(row(p_yes=0.60, contract="C-X", game="G2"), POL)
    df = pd.DataFrame(home + away + other).reindex(columns=BACKTEST_DECISIONS.column_names)
    base = df[df["scenario"] == "pen_25"].set_index("contract_id")
    assert (base["decision"] != "PASS").all()
    out = one_position_per_game(df, POL)
    kept = out[out["decision"] != "PASS"]
    keep_home = base.loc["C-H", "net_edge"] >= base.loc["C-A", "net_edge"]
    dropped = "C-A" if keep_home else "C-H"
    assert set(kept["contract_id"]) == {"C-X", "C-H" if keep_home else "C-A"}
    d = out[out["contract_id"] == dropped]
    assert (d["pass_reason"] == "same_game_position").all() and len(d) == 4
    assert d["net_pnl"].isna().all() and d["quantity"].isna().all()
    assert "same_game_position" in PASS_REASONS


def test_a_disabled_h3_records_no_primary_metric(syn_db):
    """H_MLB_003 stays registered but its fee-band metric is not computed."""
    from src.backtest import engine
    from src.backtest.registry import load_hypotheses, register_hypotheses
    from src.config import load_sport_config

    hyps = load_hypotheses("config/mlb_hypotheses.yaml")
    register_hypotheses(syn_db, hyps, now=WHEN)
    cfg, notes = load_sport_config("mlb"), []
    dec = pd.DataFrame(columns=BACKTEST_DECISIONS.column_names)
    assert engine._record_h3(syn_db, cfg, dec, POL, "logistic_residual", "cfg", notes, WHEN) == 0
    assert notes == ["H_MLB_003 is DISABLED (docs/decisions/README.md); "
                     "primary metric not recorded"]
    assert syn_db.count("experiments") == 0 and syn_db.count("hypotheses") == len(hyps)
