"""Exposure caps for PAPER TRADING ONLY (spec Sec 15).

Hard caps as fractions of bankroll: per position (applied in sizing), per game
(both contracts of a game are one correlated exposure), and new exposure per
trading day. These are research parameters, not claims of optimal sizing, and
the flat-stake backtest does not use them. Nothing here places an order.

Orders are allocated first come, first served in the order the caller ranks
them; a capped order gets whatever capacity remains (possibly 0).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Hashable, Mapping


def _day(day: date | str) -> date | str:
    return day.date() if isinstance(day, datetime) else day


class ExposureBook:
    """Committed dollars per game_id and per trading day."""

    def __init__(self, bankroll: float, max_game_frac: float = 0.03, max_daily_new_frac: float = 0.05) -> None:
        if bankroll < 0:
            raise ValueError("bankroll must be >= 0")
        self.bankroll = bankroll
        self.game_cap = max_game_frac * bankroll
        self.daily_cap = max_daily_new_frac * bankroll
        self.by_game: dict[Hashable, float] = {}
        self.by_day: dict[Hashable, float] = {}

    def headroom(self, game_id: Hashable, day: date | str) -> tuple[float, float]:
        """(remaining game capacity, remaining daily capacity), each >= 0."""
        return (max(0.0, self.game_cap - self.by_game.get(game_id, 0.0)),
                max(0.0, self.daily_cap - self.by_day.get(_day(day), 0.0)))

    def remaining(self, game_id: Hashable, day: date | str) -> float:
        return min(self.headroom(game_id, day))

    def allocate(self, game_id: Hashable, day: date | str, requested_dollars: float) -> float:
        """Commit and return min(requested, remaining game capacity, remaining daily capacity)."""
        if requested_dollars < 0:
            raise ValueError("requested_dollars must be >= 0")
        allowed = min(requested_dollars, self.remaining(game_id, day))
        if allowed > 0:
            self.by_game[game_id] = self.by_game.get(game_id, 0.0) + allowed
            self.by_day[_day(day)] = self.by_day.get(_day(day), 0.0) + allowed
        return allowed


def apply_caps(orders: list[dict[str, Any]], bankroll: float, caps: Mapping[str, float]) -> list[dict[str, Any]]:
    """Cap ranked orders ({game_id, day, requested_dollars, ...}); first come, first served.

    `caps` uses the `paper` config keys: max_game_frac, max_daily_new_frac and
    optionally max_position_frac. Returns copies of every order (none dropped)
    with `allowed_dollars` and `capped_by` (None | position | game | daily; the
    tightest binding cap).
    """
    book = ExposureBook(bankroll, caps.get("max_game_frac", 0.03), caps.get("max_daily_new_frac", 0.05))
    position_cap = caps.get("max_position_frac")
    out = []
    for order in orders:
        requested = float(order["requested_dollars"])
        capped_by = None
        if position_cap is not None and requested > position_cap * bankroll:
            requested, capped_by = position_cap * bankroll, "position"
        game_left, day_left = book.headroom(order["game_id"], order["day"])
        allowed = book.allocate(order["game_id"], order["day"], requested)
        if allowed < requested:
            capped_by = "game" if game_left <= day_left else "daily"
        out.append({**order, "allowed_dollars": allowed, "capped_by": capped_by})
    return out
