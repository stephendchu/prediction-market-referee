"""Fractional-Kelly position sizing for PAPER TRADING ONLY (spec Sec 15).

These are research parameters, not claims of optimal sizing. The backtest does
not use them: it stakes a flat `backtest.stake_dollars` per trade so results are
not driven by sizing. Nothing here places an order.

For win probability p and all-in price q per $1 contract (pass the execution
price plus fee per contract when known): b = (1 - q) / q,
f* = (b * p - (1 - p)) / b = (p - q) / (1 - q), floored at 0.
"""

from __future__ import annotations


def _f_star(p: float, q: float) -> float:
    if not 0 < q < 1:
        raise ValueError(f"price q must be in (0, 1), got {q}")
    if not 0 <= p <= 1:
        raise ValueError(f"probability p must be in [0, 1], got {p}")
    b = (1.0 - q) / q
    return max(0.0, (b * p - (1.0 - p)) / b)


def kelly_fraction(p: float, q: float) -> float:
    """Full-Kelly bankroll fraction f* for a $1 contract bought at q with win probability p."""
    return _f_star(p, q)


def kelly_stake(p: float, q: float, bankroll: float, kelly_fraction: float = 0.25,
                max_position_frac: float = 0.01) -> float:
    """Dollars to stake: kelly_fraction * f* * bankroll, capped at max_position_frac * bankroll."""
    if bankroll < 0:
        raise ValueError("bankroll must be >= 0")
    return min(kelly_fraction * _f_star(p, q) * bankroll, max_position_frac * bankroll)
