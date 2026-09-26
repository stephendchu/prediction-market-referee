"""Transaction costs and simulated fills (spec Sec 14, 16, 26.8, 26.9).

Simulation only: nothing here places, routes, or signs an order.

Fees (26.8). `FeeSchedule` is a versioned interface. Callers never choose a
formula: they ask `fee_schedule_for(when)` for the schedule in force at the
historical timestamp, and every simulated trade records the schedule it was
charged under (fee_schedule_id, effective date, maker_or_taker, fee).

Prices. A YES price p lies in (0, 1); one contract pays $1. Buying YES pays
yes_ask and consumes the YES asks (ask_size); buying NO pays no_ask = 1 - yes_bid
and consumes the YES bids (bid_size), since a YES bid at p is a NO ask at 1 - p.
Never the midpoint, never the last trade.

Fill model (26.9), stake S, spread penalty k:
  base = quoted_price + k * spread
  depth known        Q = floor(S / base) capped at displayed depth; price = max(VWAP(Q), base)
  top size known     Q = floor(min(S / base, max_fraction_top * top_size)); price = base
  no size (candles)  Q = floor(S / base); price = base; depth_verified = False

Exit leg, the same model mirrored for a sale of a position of Q0:
selling YES crosses yes_bid and consumes the YES bids (bid_size); selling NO
crosses no_bid = 1 - yes_ask and consumes the YES asks (ask_size).
  base = bid(side) - k * spread
  depth known        Q = min(Q0, displayed bid depth); price = min(VWAP(Q), base)
  top size known     Q = floor(min(Q0, max_fraction_top * bid-side top size)); price = base
  no size (candles)  Q = Q0; price = base; depth_verified = False
The exit fee comes from the schedule in force at the exit time. Contracts that
cannot be sold are held to settlement, and the Position records why.

A Position is open from its entry fill until its exit fill or settlement; it
is the one round-trip P&L path, shared by the backtest and paper trading.
P&L semantics are those of the backtest_decisions table (src/storage/schemas.py).
Scalar functions use plain float math: the backtest calls them per observation.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from ..storage.schemas import EXIT_AT_OFFSET, EXIT_RULES, HOLD_TO_SETTLEMENT
from ..storage.schemas import FEE_SCHEDULES as FEE_SCHEDULES_TABLE

SIDES = ("yes", "no")
NO_PRICE_AT = 0.99  # a side costing this much or more is treated as unpriceable
_EPS = 1e-9  # float noise guard for floor/ceil (0.29 * 100 = 28.999999999999996)


class NoFeeSchedule(LookupError):
    """No (or more than one) fee schedule covers the requested date."""


# ---------------------------------------------------------------------------
# Fee schedules
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class FeeSchedule(ABC):
    fee_schedule_id: str
    venue: str = "exchange"
    maker_or_taker: str
    effective_from: date
    effective_to: date | None = None  # inclusive; None = still in force
    formula: str
    params: Mapping[str, float] = field(default_factory=dict, hash=False)
    verified: bool = False
    source_note: str = ""

    def __post_init__(self) -> None:
        if self.maker_or_taker not in ("maker", "taker"):
            raise ValueError(f"{self.fee_schedule_id}: maker_or_taker must be maker|taker")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError(f"{self.fee_schedule_id}: effective_to before effective_from")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @abstractmethod
    def fee(self, contracts: float, price: float) -> float:
        """Total fee in dollars for one order of `contracts` at `price`."""

    def covers(self, d: date) -> bool:
        return self.effective_from <= d and (self.effective_to is None or d <= self.effective_to)

    def row(self) -> dict[str, Any]:
        return {"fee_schedule_id": self.fee_schedule_id, "venue": self.venue,
                "maker_or_taker": self.maker_or_taker, "effective_from": self.effective_from,
                "effective_to": self.effective_to, "formula": self.formula,
                "params_json": json.dumps(dict(self.params), sort_keys=True), "verified": self.verified,
                "source_note": self.source_note}


@dataclass(frozen=True, kw_only=True)
class QuadraticFeeSchedule(FeeSchedule):
    """fee = rate * C * P * (1 - P), rounded up to the next cent, per order."""

    formula: str = "quadratic_ceil_cent_v1"

    def __post_init__(self) -> None:
        super().__post_init__()
        if "rate" not in self.params:
            raise ValueError(f"{self.fee_schedule_id}: params need 'rate'")

    def fee(self, contracts: float, price: float) -> float:
        if contracts < 0 or not 0 < price < 1:
            raise ValueError(f"fee needs contracts >= 0 and 0 < price < 1, got {contracts}, {price}")
        cents = self.params["rate"] * contracts * price * (1.0 - price) * 100.0
        return math.ceil(cents - _EPS) / 100.0


_UNVERIFIED = ("A quadratic trading-fee formula of the kind exchanges publish: round up to the next cent of "
               "rate x contracts x price x (1 - price). Illustrative: not checked against any venue's dated fee "
               "schedule; historical changes, per-series exceptions and fee caps are unverified.")

FEE_SCHEDULES: tuple[FeeSchedule, ...] = (
    QuadraticFeeSchedule(fee_schedule_id="quadratic-taker-v1", maker_or_taker="taker",
                         effective_from=date(2021, 1, 1), params={"rate": 0.07}, verified=False,
                         source_note=f"Taker, rate 0.07. {_UNVERIFIED}"),
    QuadraticFeeSchedule(fee_schedule_id="quadratic-maker-v1", maker_or_taker="maker",
                         effective_from=date(2021, 1, 1), params={"rate": 0.0175}, verified=False,
                         source_note=f"Maker, rate 0.0175. Maker fees apply only to some markets; whether "
                                     f"MLB game markets charge them is unverified. {_UNVERIFIED}"),
)


def fee_schedule_for(when: date | datetime, maker_or_taker: str = "taker",
                     registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> FeeSchedule:
    """The one schedule in force on `when`'s UTC date for this liquidity role."""
    d = _utc_date(when)
    found = [s for s in registry if s.maker_or_taker == maker_or_taker and s.covers(d)]
    if len(found) != 1:
        what = "no" if not found else f"ambiguous ({[s.fee_schedule_id for s in found]})"
        raise NoFeeSchedule(f"{what} {maker_or_taker} fee schedule for {d}")
    return found[0]


def fee_schedule_by_id(fee_schedule_id: str, registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> FeeSchedule:
    """The registry schedule a stored fill names (paper trades rebuilt from their log)."""
    found = [s for s in registry if s.fee_schedule_id == fee_schedule_id]
    if len(found) != 1:
        raise NoFeeSchedule(f"{len(found)} fee schedules with id {fee_schedule_id!r}")
    return found[0]


def fee_schedule_rows(registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> pd.DataFrame:
    """Rows for the fee_schedules table (a mirror of the registry)."""
    ids = [s.fee_schedule_id for s in registry]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate fee_schedule_id in registry: {ids}")
    return pd.DataFrame([s.row() for s in registry], columns=FEE_SCHEDULES_TABLE.column_names)


def _utc_date(when: date | datetime) -> date:
    if isinstance(when, datetime):
        return (when.astimezone(timezone.utc) if when.tzinfo else when).date()
    return when


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    yes_bid: float | None
    yes_ask: float | None
    bid_size: float | None = None
    ask_size: float | None = None
    depth_json: str | None = None  # {"yes_bids": [[p, q], ..], "yes_asks": [[p, q], ..]}, best first

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Quote:
        """From an observations row (market_yes_bid, ...) or a market_quotes row (yes_bid, ...)."""
        bid = row.get("market_yes_bid", row.get("yes_bid"))
        ask = row.get("market_yes_ask", row.get("yes_ask"))
        depth = row.get("depth_json")
        return cls(_num(bid), _num(ask), _num(row.get("bid_size")), _num(row.get("ask_size")),
                   depth if isinstance(depth, str) and depth else None)

    @property
    def is_valid(self) -> bool:
        """Two-sided with bid < ask (NaN compares false, so it is invalid too)."""
        b, a = self.yes_bid, self.yes_ask
        return b is not None and a is not None and 0 < b < a < 1


def _num(v: Any) -> float | None:
    return None if v is None or v != v else float(v)  # v != v: NaN


def _check_side(side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be yes|no, got {side!r}")


def quoted_price(side: str, quote: Quote) -> float:
    """Executable ask of `side`: yes_ask, or no_ask = 1 - yes_bid."""
    _check_side(side)
    return quote.yes_ask if side == "yes" else 1.0 - quote.yes_bid


def spread(quote: Quote) -> float:
    return quote.yes_ask - quote.yes_bid


def top_size(side: str, quote: Quote) -> float | None:
    """Displayed contracts at the best price `side` would buy from (None = unknown)."""
    _check_side(side)
    return _num(quote.ask_size if side == "yes" else quote.bid_size)


def depth_levels(side: str, quote: Quote) -> list[tuple[float, float]]:
    """Displayed asks for buying `side` as (price, contracts), cheapest first."""
    _check_side(side)
    if not quote.depth_json:
        return []
    book = json.loads(quote.depth_json)
    if side == "yes":
        levels = [(float(p), float(q)) for p, q in book.get("yes_asks") or []]
    else:
        levels = [(1.0 - float(p), float(q)) for p, q in book.get("yes_bids") or []]
    return sorted((p, q) for p, q in levels if 0 < p < 1 and q > 0)


def bid_price(side: str, quote: Quote) -> float:
    """Executable bid of `side` (what a sale receives): yes_bid, or no_bid = 1 - yes_ask."""
    _check_side(side)
    return quote.yes_bid if side == "yes" else 1.0 - quote.yes_ask


def bid_top_size(side: str, quote: Quote) -> float | None:
    """Displayed contracts at the best price `side` would sell into (None = unknown)."""
    _check_side(side)
    return _num(quote.bid_size if side == "yes" else quote.ask_size)


def bid_depth_levels(side: str, quote: Quote) -> list[tuple[float, float]]:
    """Displayed bids for selling `side` as (price, contracts), best (highest) first."""
    _check_side(side)
    if not quote.depth_json:
        return []
    book = json.loads(quote.depth_json)
    if side == "yes":
        levels = [(float(p), float(q)) for p, q in book.get("yes_bids") or []]
    else:
        levels = [(1.0 - float(p), float(q)) for p, q in book.get("yes_asks") or []]
    return sorted(((p, q) for p, q in levels if 0 < p < 1 and q > 0), reverse=True)


def _vwap(levels: list[tuple[float, float]], qty: int) -> float:
    remaining, cost = float(qty), 0.0
    for p, q in levels:
        take = min(q, remaining)
        cost += take * p
        remaining -= take
        if remaining <= 0:
            break
    return cost / qty


def _floor(x: float) -> int:
    return math.floor(x + _EPS)


# ---------------------------------------------------------------------------
# Fills and settlement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fill:
    side: str
    quoted_price: float
    execution_price: float
    quantity: float
    stake: float  # quantity * execution_price
    spread_penalty_frac: float
    fee: float
    fee_schedule: FeeSchedule
    slippage_cost: float  # quantity * (execution_price - quoted_price)
    depth_verified: bool

    def record(self) -> dict[str, Any]:
        """The backtest_decisions columns a fill determines."""
        s = self.fee_schedule
        return {"side": self.side, "quoted_price": self.quoted_price,
                "spread_penalty_frac": self.spread_penalty_frac, "execution_price": self.execution_price,
                "quantity": self.quantity, "stake": self.stake, "fee_schedule_id": s.fee_schedule_id,
                "fee_schedule_effective_date": s.effective_from, "maker_or_taker": s.maker_or_taker,
                "calculated_fee": self.fee, "slippage_cost": self.slippage_cost,
                "depth_verified": self.depth_verified}

    @classmethod
    def from_record(cls, rec: Mapping[str, Any], *, when: date | datetime | None = None,
                    registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> Fill:
        """Rebuild a fill from its `record()` columns (a logged paper trade).

        The stored fee, stake and slippage are recomputed from the named schedule and must agree;
        `when` (the entry time) must fall in that schedule's window.
        """
        schedule = _stored_schedule(rec, "", when, _str(rec.get("maker_or_taker")), registry)
        side, qty = str(rec["side"]), float(rec["quantity"])
        price, quoted = float(rec["execution_price"]), float(rec["quoted_price"])
        _check_side(side)
        if qty <= 0:
            raise ValueError(f"a fill needs quantity > 0, got {qty}")
        fill = cls(side=side, quoted_price=quoted, execution_price=price, quantity=qty, stake=qty * price,
                   spread_penalty_frac=float(rec["spread_penalty_frac"]), fee=schedule.fee(qty, price),
                   fee_schedule=schedule, slippage_cost=qty * (price - quoted),
                   depth_verified=bool(rec["depth_verified"]))
        _check_stored(rec, {"stake": fill.stake, "calculated_fee": fill.fee, "slippage_cost": fill.slippage_cost},
                      schedule)
        return fill


@dataclass(frozen=True)
class NoFill:
    side: str
    reason: str  # a backtest_decisions pass_reason: no_price | insufficient_depth


def simulate_fill(side: str, quote: Quote, *, penalty_frac: float, stake_dollars: float,
                  max_fraction_top: float, min_contracts: int, when: date | datetime,
                  maker_or_taker: str = "taker",
                  registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> Fill | NoFill:
    """Buy `side` with about `stake_dollars` (26.9 fill model; see module docstring)."""
    _check_side(side)
    if penalty_frac < 0:
        raise ValueError("penalty_frac must be >= 0")
    if not quote.is_valid:
        return NoFill(side, "no_price")
    quoted = quoted_price(side, quote)
    base = quoted + penalty_frac * spread(quote)
    if base >= NO_PRICE_AT:
        return NoFill(side, "no_price")
    levels = depth_levels(side, quote)
    top = top_size(side, quote)
    if levels:
        qty = min(_floor(stake_dollars / base), _floor(sum(q for _, q in levels)))
        price = max(_vwap(levels, qty), base) if qty > 0 else base
        verified = True
    elif top is not None:
        qty = _floor(min(stake_dollars / base, max_fraction_top * top))
        price, verified = base, True
    else:  # historical candles: bid/ask without displayed size
        qty = _floor(stake_dollars / base)
        price, verified = base, False
    if price >= NO_PRICE_AT:
        return NoFill(side, "no_price")
    if qty < max(min_contracts, 1):
        return NoFill(side, "insufficient_depth")
    schedule = fee_schedule_for(when, maker_or_taker, registry)
    return Fill(side=side, quoted_price=quoted, execution_price=price, quantity=float(qty),
                stake=qty * price, spread_penalty_frac=penalty_frac, fee=schedule.fee(qty, price),
                fee_schedule=schedule, slippage_cost=qty * (price - quoted), depth_verified=verified)


@dataclass(frozen=True)
class ExitFill:
    side: str  # the side sold (the side held)
    quoted_price: float  # executable bid of that side
    execution_price: float  # quoted_price - penalty x spread, or lower (VWAP down the bids)
    quantity: float
    proceeds: float  # quantity * execution_price
    spread_penalty_frac: float
    fee: float
    fee_schedule: FeeSchedule
    slippage_cost: float  # quantity * (quoted_price - execution_price)
    depth_verified: bool

    @classmethod
    def from_record(cls, rec: Mapping[str, Any], *, side: str, penalty_frac: float,
                    maker_or_taker: str | None = None, when: date | datetime | None = None,
                    registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> ExitFill:
        """Rebuild a sale from its `Position.record()` exit_* columns (a logged paper exit).

        As `Fill.from_record`: the stored fee, proceeds and slippage must agree with the schedule.
        """
        _check_side(side)
        if _str(rec.get("side")) not in (None, side):
            raise ValueError(f"exit record sold {rec.get('side')!r}, position holds {side!r}")
        schedule = _stored_schedule(rec, "exit_", when, maker_or_taker, registry)
        qty = float(rec["exit_quantity"])
        price, quoted = float(rec["exit_price"]), float(rec["exit_quoted_price"])
        if qty <= 0:
            raise ValueError(f"a sale needs quantity > 0, got {qty}")
        x = cls(side=side, quoted_price=quoted, execution_price=price, quantity=qty, proceeds=qty * price,
                spread_penalty_frac=penalty_frac, fee=schedule.fee(qty, price), fee_schedule=schedule,
                slippage_cost=qty * (quoted - price), depth_verified=bool(rec["exit_depth_verified"]))
        _check_stored(rec, {"exit_proceeds": x.proceeds, "exit_fee": x.fee, "exit_slippage_cost": x.slippage_cost},
                      schedule)
        return x


def _str(v: Any) -> str | None:
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


def _stored_schedule(rec: Mapping[str, Any], prefix: str, when: date | datetime | None,
                     maker_or_taker: str | None, registry: tuple[FeeSchedule, ...]) -> FeeSchedule:
    """The schedule a stored leg names, checked against its stored date, role and trade time."""
    s = fee_schedule_by_id(str(rec[f"{prefix}fee_schedule_id"]), registry)
    eff = rec.get(f"{prefix}fee_schedule_effective_date")
    if eff is None or pd.isna(eff) or pd.Timestamp(eff).date() != s.effective_from:
        raise ValueError(f"{s.fee_schedule_id}: stored effective date {eff!r} is not {s.effective_from}")
    if maker_or_taker is not None and maker_or_taker != s.maker_or_taker:
        raise ValueError(f"{s.fee_schedule_id} is a {s.maker_or_taker} schedule, not {maker_or_taker}")
    if when is not None and fee_schedule_for(when, s.maker_or_taker, registry).fee_schedule_id != s.fee_schedule_id:
        raise ValueError(f"{s.fee_schedule_id} was not in force on {_utc_date(when)}")
    return s


def _check_stored(rec: Mapping[str, Any], expected: Mapping[str, float], schedule: FeeSchedule) -> None:
    for key, value in expected.items():
        stored = _num(rec.get(key))
        if stored is None or abs(stored - value) > _EPS:
            raise ValueError(f"stored {key} {stored!r} disagrees with {value!r} recomputed under "
                             f"{schedule.fee_schedule_id}")


@dataclass(frozen=True)
class NoExit:
    side: str
    reason: str  # a backtest_decisions exit_reason: no_valid_exit_quote | no_exit_price | insufficient_exit_depth


def simulate_exit(side: str, quote: Quote, quantity: float, *, penalty_frac: float, max_fraction_top: float,
                  when: date | datetime, maker_or_taker: str = "taker",
                  registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> ExitFill | NoExit:
    """Sell up to `quantity` contracts of `side` into the bids (exit leg; see module docstring)."""
    _check_side(side)
    if penalty_frac < 0:
        raise ValueError("penalty_frac must be >= 0")
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if not quote.is_valid:
        return NoExit(side, "no_valid_exit_quote")
    quoted = bid_price(side, quote)
    base = quoted - penalty_frac * spread(quote)
    floor_price = 1.0 - NO_PRICE_AT
    if base <= floor_price:
        return NoExit(side, "no_exit_price")
    levels = bid_depth_levels(side, quote)
    top = bid_top_size(side, quote)
    if levels:
        qty = min(_floor(quantity), _floor(sum(q for _, q in levels)))
        price = min(_vwap(levels, qty), base) if qty > 0 else base
        verified = True
    elif top is not None:
        qty = _floor(min(quantity, max_fraction_top * top))
        price, verified = base, True
    else:  # historical candles: bid/ask without displayed size
        qty = _floor(quantity)
        price, verified = base, False
    if price <= floor_price:
        return NoExit(side, "no_exit_price")
    if qty < 1:
        return NoExit(side, "insufficient_exit_depth")
    schedule = fee_schedule_for(when, maker_or_taker, registry)
    return ExitFill(side=side, quoted_price=quoted, execution_price=price, quantity=float(qty),
                    proceeds=qty * price, spread_penalty_frac=penalty_frac, fee=schedule.fee(qty, price),
                    fee_schedule=schedule, slippage_cost=qty * (quoted - price), depth_verified=verified)


def side_won(side: str, outcome: int) -> int:
    """1 if the side bought paid out, given outcome = 1 when the contract resolved YES."""
    _check_side(side)
    if outcome not in (0, 1):
        raise ValueError(f"outcome must be 0|1, got {outcome!r}")
    return int(outcome) if side == "yes" else 1 - int(outcome)


def settle(fill: Fill, side_won: int) -> dict[str, float]:
    """A held position's gross_pnl (before fees and slippage) and net_pnl (after both)."""
    if side_won not in (0, 1):
        raise ValueError(f"side_won must be 0|1, got {side_won!r}")
    return {"side_won": int(side_won), **round_trip_pnl(fill, None, side_won)}


def round_trip_pnl(entry: Fill, exit: ExitFill | None, side_won: int | None) -> dict[str, float | None]:
    """gross_pnl and net_pnl of a position (backtest_decisions semantics); the one P&L formula.

    net = exit proceeds + held contracts x side_won - entry cost - entry fee - exit fee.
    None while held contracts are unsettled (side_won None).
    """
    held = entry.quantity - (exit.quantity if exit else 0.0)
    if held > 0 and side_won is None:
        return {"gross_pnl": None, "net_pnl": None}
    if held > 0 and side_won not in (0, 1):
        raise ValueError(f"side_won must be 0|1, got {side_won!r}")
    gross = held * (side_won - entry.quoted_price) if held > 0 else 0.0
    net = (held * side_won if held > 0 else 0.0) - entry.stake - entry.fee
    if exit is not None:
        gross += exit.quantity * (exit.quoted_price - entry.quoted_price)
        net += exit.proceeds - exit.fee
    return {"gross_pnl": gross, "net_pnl": net}


# ---------------------------------------------------------------------------
# Exit rule and positions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitRule:
    """hold_to_settlement, or exit_at_offset: sell at the T-`offset_min` pre-game snapshot."""

    rule: str = HOLD_TO_SETTLEMENT
    offset_min: int | None = None

    def __post_init__(self) -> None:
        if self.rule not in EXIT_RULES:
            raise ValueError(f"exit_rule must be one of {EXIT_RULES}, got {self.rule!r}")
        if self.rule == EXIT_AT_OFFSET:
            if not isinstance(self.offset_min, int) or isinstance(self.offset_min, bool) or self.offset_min <= 0:
                raise ValueError(f"{EXIT_AT_OFFSET} needs a positive integer exit_offset_min, got {self.offset_min!r}")
        elif self.offset_min is not None:
            raise ValueError(f"{HOLD_TO_SETTLEMENT} takes no exit offset, got {self.offset_min!r}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> ExitRule:
        e = cfg.get("execution") or {}
        rule = str(e.get("exit_rule", HOLD_TO_SETTLEMENT))
        return cls(rule, e.get("exit_offset_min") if rule == EXIT_AT_OFFSET else None)

    @property
    def exits(self) -> bool:
        return self.rule == EXIT_AT_OFFSET

    def exit_time(self, event_start: datetime) -> datetime | None:
        """The planned exit snapshot for a game starting at `event_start` (None: held to settlement)."""
        return event_start - timedelta(minutes=self.offset_min) if self.exits else None


@dataclass(frozen=True)
class Position:
    """One position: open from its entry fill until its exit fill or settlement.

    `close` applies the planned exit; contracts it cannot sell stay held to
    settlement with `exit_reason` saying why. Round-trip P&L:
      exited  exit proceeds - entry cost - entry fee - exit fee
      held    settlement payout - entry cost - entry fee
      partial both legs: exit proceeds + held_quantity * side_won - entry cost - both fees
    """

    entry: Fill
    entry_time: datetime
    exit_rule: ExitRule = ExitRule()
    exit: ExitFill | None = None
    exit_time: datetime | None = None  # planned exit time; set on fallbacks too
    exit_reason: str | None = None
    exit_ref: str | None = None  # observation that priced the exit

    @property
    def exit_quantity(self) -> float:
        return self.exit.quantity if self.exit else 0.0

    @property
    def held_quantity(self) -> float:
        return self.entry.quantity - self.exit_quantity

    @property
    def exit_status(self) -> str:
        if self.exit is None:
            return "held"
        return "exited" if self.held_quantity <= 0 else "partial"

    def close(self, quote: Quote | None, when: datetime, *, ref: str | None = None, penalty_frac: float,
              max_fraction_top: float, maker_or_taker: str = "taker",
              registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> Position:
        """Apply the planned exit at `when` against `quote` (None: no valid exit quote)."""
        if not self.exit_rule.exits:
            raise ValueError(f"{self.exit_rule.rule} positions are held to settlement; nothing to close")
        if self.exit is not None or self.exit_reason is not None:
            raise ValueError("position already closed")
        held = replace(self, exit_time=when, exit_ref=ref)
        if when <= self.entry_time:
            return replace(held, exit_reason="exit_not_after_entry")
        if quote is None:
            return replace(held, exit_reason="no_valid_exit_quote")
        x = simulate_exit(self.entry.side, quote, self.entry.quantity, penalty_frac=penalty_frac,
                          max_fraction_top=max_fraction_top, when=when, maker_or_taker=maker_or_taker,
                          registry=registry)
        if isinstance(x, NoExit):
            return replace(held, exit_reason=x.reason)
        partial = x.quantity < self.entry.quantity
        return replace(held, exit=x, exit_reason="insufficient_exit_depth" if partial else None)

    def open_quantity(self, at: datetime, settled_at: datetime | None = None) -> float:
        """Contracts open at `at`: all of them from entry to exit, the held rest until settlement."""
        if at < self.entry_time or (settled_at is not None and at >= settled_at):
            return 0.0
        if self.exit is not None and at >= self.exit_time:
            return self.held_quantity
        return self.entry.quantity

    def pnl(self, side_won: int | None) -> dict[str, float | None]:
        """Round-trip gross_pnl and net_pnl (None while held contracts are unsettled)."""
        return round_trip_pnl(self.entry, self.exit, side_won)

    def record(self, side_won: int | None) -> dict[str, Any]:
        """The backtest_decisions exit and P&L columns this position determines."""
        x = self.exit
        return {"exit_rule": self.exit_rule.rule, "exit_status": self.exit_status, "exit_reason": self.exit_reason,
                "exit_observation_id": self.exit_ref, "exit_snapshot_time": self.exit_time,
                "exit_quoted_price": x.quoted_price if x else None, "exit_price": x.execution_price if x else None,
                "exit_quantity": x.quantity if x else None, "held_quantity": self.held_quantity,
                "exit_proceeds": x.proceeds if x else None,
                "exit_fee_schedule_id": x.fee_schedule.fee_schedule_id if x else None,
                "exit_fee_schedule_effective_date": x.fee_schedule.effective_from if x else None,
                "exit_fee": x.fee if x else None, "exit_slippage_cost": x.slippage_cost if x else None,
                "exit_depth_verified": x.depth_verified if x else None, **self.pnl(side_won)}


def net_edge(p_conservative_side: float, execution_price: float, fee: float, quantity: float) -> float:
    """Per-contract edge after execution price and fee: p_conservative - price - fee / quantity."""
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    return p_conservative_side - execution_price - fee / quantity


def estimated_net_edge(p_side_conservative: float, quote: Quote, side: str, penalty_frac: float,
                       stake_dollars: float, when: date | datetime, maker_or_taker: str = "taker",
                       registry: tuple[FeeSchedule, ...] = FEE_SCHEDULES) -> float | None:
    """Screening net edge for `side`: prices Q = floor(stake / base) at base, with no size cap.

    None when the side can't be priced (invalid quote, base >= NO_PRICE_AT, or Q = 0).
    """
    _check_side(side)
    if not quote.is_valid:
        return None
    base = quoted_price(side, quote) + penalty_frac * spread(quote)
    qty = _floor(stake_dollars / base) if base < NO_PRICE_AT else 0
    if qty < 1:
        return None
    fee = fee_schedule_for(when, maker_or_taker, registry).fee(qty, base)
    return p_side_conservative - base - fee / qty


def scenario_name(penalty_frac: float) -> str:
    """backtest_decisions.scenario label: 0.25 -> pen_25."""
    return f"pen_{round(penalty_frac * 100)}"
