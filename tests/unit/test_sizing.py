"""Paper-trading sizing (Sec 15): fractional Kelly and exposure caps."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.portfolio.exposure import ExposureBook, apply_caps
from src.portfolio.sizing import kelly_fraction, kelly_stake

D1, D2 = date(2026, 9, 24), date(2026, 9, 25)


def test_kelly_fraction_matches_the_spec_formula():
    assert kelly_fraction(0.60, 0.50) == pytest.approx(0.20)          # b = 1: f* = 0.6 - 0.4
    assert kelly_fraction(0.62, 0.57) == pytest.approx(0.05 / 0.43)   # b = 0.43 / 0.57
    b = 0.43 / 0.57
    assert kelly_fraction(0.62, 0.57) == pytest.approx((b * 0.62 - 0.38) / b)


def test_kelly_fraction_is_floored_at_zero():
    assert kelly_fraction(0.50, 0.60) == 0.0
    assert kelly_fraction(0.57, 0.57) == 0.0


@pytest.mark.parametrize("p,q", [(0.5, 0.0), (0.5, 1.0), (1.2, 0.5), (-0.1, 0.5)])
def test_kelly_rejects_impossible_inputs(p, q):
    with pytest.raises(ValueError):
        kelly_fraction(p, q)


def test_kelly_stake_is_fractional_and_capped_per_position():
    assert kelly_stake(0.51, 0.50, 10_000) == pytest.approx(50.0)    # 0.25 * 0.02 * 10k
    assert kelly_stake(0.60, 0.50, 10_000) == pytest.approx(100.0)   # 500 uncapped -> 1% cap
    assert kelly_stake(0.60, 0.50, 10_000, kelly_fraction=0.1, max_position_frac=0.5) == pytest.approx(200.0)
    assert kelly_stake(0.40, 0.50, 10_000) == 0.0


def test_exposure_book_applies_game_and_daily_caps():
    book = ExposureBook(10_000)  # game cap 300, daily new cap 500
    assert book.allocate("g1", D1, 100) == pytest.approx(100)
    assert book.allocate("g1", D1, 250) == pytest.approx(200)  # game cap: 300 - 100
    assert book.allocate("g1", D1, 50) == 0.0
    assert book.allocate("g2", D1, 250) == pytest.approx(200)  # daily cap: 500 - 300
    assert book.allocate("g3", D1, 10) == 0.0
    assert book.allocate("g3", D2, 100) == pytest.approx(100)  # a new day has fresh daily capacity
    assert book.allocate("g1", D2, 100) == 0.0                 # ... but the game is still at its cap
    assert book.by_game == pytest.approx({"g1": 300, "g2": 200, "g3": 100})
    assert book.by_day == pytest.approx({D1: 500, D2: 100})


def test_exposure_book_treats_a_datetime_as_its_day():
    book = ExposureBook(10_000, max_daily_new_frac=0.02)
    assert book.allocate("g1", datetime(2026, 9, 24, 18, tzinfo=timezone.utc), 150) == pytest.approx(150)
    assert book.allocate("g2", D1, 150) == pytest.approx(50)


def test_apply_caps_is_first_come_first_served():
    orders = [
        {"game_id": "g1", "day": D1, "requested_dollars": 250, "contract_id": "A"},
        {"game_id": "g1", "day": D1, "requested_dollars": 150, "contract_id": "B"},  # other side of g1
        {"game_id": "g2", "day": D1, "requested_dollars": 90, "contract_id": "C"},
        {"game_id": "g3", "day": D1, "requested_dollars": 100, "contract_id": "D"},
        {"game_id": "g4", "day": D1, "requested_dollars": 100, "contract_id": "E"},
    ]
    caps = {"max_position_frac": 0.02, "max_game_frac": 0.03, "max_daily_new_frac": 0.05}
    out = apply_caps(orders, 10_000, caps)  # caps: position 200, game 300, daily 500
    assert [o["contract_id"] for o in out] == ["A", "B", "C", "D", "E"]
    assert [o["allowed_dollars"] for o in out] == pytest.approx([200, 100, 90, 100, 10])
    assert [o["capped_by"] for o in out] == ["position", "game", None, None, "daily"]
    assert orders[0]["requested_dollars"] == 250 and "allowed_dollars" not in orders[0]  # inputs untouched


def test_apply_caps_reports_the_game_cap():
    orders = [{"game_id": "g1", "day": D1, "requested_dollars": 200}] * 2
    out = apply_caps(orders, 10_000, {"max_game_frac": 0.03, "max_daily_new_frac": 0.05})
    assert [(o["allowed_dollars"], o["capped_by"]) for o in out] == [(200, None), (pytest.approx(100), "game")]
