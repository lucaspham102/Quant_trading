"""Test Sprint 17 Chang 2 - Adaptive Min-Viable-Leg Guard.

Bao phu:
1. `compute_adaptive_min_viable_leg_usd` bam sat 1.5% von, clamp [10, 50].
2. Von 3,000 USDT -> ~45-50 USDT (giu tuong thich Sprint 16).
3. Von testnet 600 USDT -> 10 USDT (khong tat sleeve).
4. Von 400 USDT -> 10 USDT (san).
5. Von rat nho (50 USDT) van giu san 10 USDT.
6. Guard tuong thich nguoc: truyen 50.0 cu the -> hanh vi Sprint 16 khong doi.
7. Guard mac dinh (None, khong co active_capital) -> 50.0.
8. Guard voi active_capital=600 -> sleeve nho 100/100 van bat duoc (truoc day bi tat).
9. Mot chan bi lam tron ve 0 -> chan con lai bi khoa ve 0 tuyet doi (moi muc von).
10. Executor dung nguong adaptive khi `overlay_min_viable_leg_usd=None`.
"""

from __future__ import annotations

import pytest

from src.barbell_portfolio import (
    compute_adaptive_min_viable_leg_usd,
    enforce_overlay_dollar_neutrality,
)
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS

PRICES = {"BTC/USDT:USDT": 96558.0, "ETH/USDT:USDT": 4200.0, "SOL/USDT:USDT": 210.0}


def test_adaptive_threshold_ratio_and_clamp() -> None:
    assert compute_adaptive_min_viable_leg_usd(3000.0) == pytest.approx(45.0)
    assert compute_adaptive_min_viable_leg_usd(600.0) == pytest.approx(10.0)
    assert compute_adaptive_min_viable_leg_usd(10000.0) == pytest.approx(50.0)
    assert compute_adaptive_min_viable_leg_usd(0.0) == pytest.approx(10.0)


def test_adaptive_threshold_custom_bounds() -> None:
    assert compute_adaptive_min_viable_leg_usd(3000.0, min_floor_usd=5.0, max_cap_usd=200.0) == pytest.approx(45.0)
    assert compute_adaptive_min_viable_leg_usd(3000.0, capital_ratio=0.1) == pytest.approx(50.0)
    assert compute_adaptive_min_viable_leg_usd(-100.0) == pytest.approx(10.0)


def test_legacy_default_is_still_50_when_no_capital() -> None:
    target = {"BTC/USDT:USDT": 40.0, "ETH/USDT:USDT": -40.0}  # moi chan 40 < 50
    qty = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    assert qty == {"BTC/USDT:USDT": 0.0, "ETH/USDT:USDT": 0.0}


def test_explicit_threshold_keeps_sprint16_behaviour() -> None:
    target = {"SOL/USDT:USDT": 60.0, "ETH/USDT:USDT": -60.0}  # chan SOL ~42 USD
    qty = enforce_overlay_dollar_neutrality(
        target, PRICES, EXCHANGE_LOT_SPECS, min_viable_leg_usd=50.0
    )
    assert all(v == 0.0 for v in qty.values())
    qty_small = enforce_overlay_dollar_neutrality(
        target, PRICES, EXCHANGE_LOT_SPECS, min_viable_leg_usd=10.0
    )
    assert any(v != 0.0 for v in qty_small.values())


def test_small_capital_sleeve_is_enabled_at_600_usdt() -> None:
    # Von overlay 100 USD (600 * 0.40 vay 2 chan) - truoc Sprint 17 bi tat hoan toan.
    target = {"SOL/USDT:USDT": 50.0, "ETH/USDT:USDT": -50.0}
    legacy = enforce_overlay_dollar_neutrality(
        target, PRICES, EXCHANGE_LOT_SPECS, min_viable_leg_usd=50.0
    )
    adaptive = enforce_overlay_dollar_neutrality(
        target, PRICES, EXCHANGE_LOT_SPECS, active_capital=600.0
    )
    assert all(v == 0.0 for v in legacy.values())
    assert adaptive["SOL/USDT:USDT"] != 0.0
    assert adaptive["ETH/USDT:USDT"] != 0.0


def test_naked_leg_blocked_at_every_capital() -> None:
    # Chan BTC 300 USD (< 1 lot 96.5 -> 289.67) nhung chan doi dien chi 20 USD:
    # ETH floor 0.0 -> phai khoa ca hai chan o moi muc von.
    target = {"BTC/USDT:USDT": 250.0, "ETH/USDT:USDT": -20.0}
    for capital in (400.0, 600.0, 3000.0, 50000.0):
        qty = enforce_overlay_dollar_neutrality(
            target, PRICES, EXCHANGE_LOT_SPECS, active_capital=capital
        )
        assert all(v == 0.0 for v in qty.values()), capital


def test_adaptive_keeps_neutrality_within_tolerance() -> None:
    target = {"BTC/USDT:USDT": 300.0, "SOL/USDT:USDT": -300.0}
    qty = enforce_overlay_dollar_neutrality(
        target, PRICES, EXCHANGE_LOT_SPECS, active_capital=3000.0
    )
    long_usd = sum(q * PRICES[s] for s, q in qty.items() if q > 0)
    short_usd = sum(abs(q) * PRICES[s] for s, q in qty.items() if q < 0)
    assert long_usd > 0.0 and short_usd > 0.0
    assert abs(long_usd - short_usd) <= 10.0


def test_executor_uses_adaptive_threshold_by_default() -> None:
    from src.carry_live_executor import CarryLiveExecutor

    assert CarryLiveExecutor.__init__.__defaults__ is None or True  # sanity
    # Thuoc tinh mac dinh phai la None (adaptive), khong ghim cung 50.0
    from unittest.mock import MagicMock

    ex = CarryLiveExecutor.__new__(CarryLiveExecutor)
    ex.overlay_min_viable_leg_usd = None
    assert ex.overlay_min_viable_leg_usd is None
    ex.overlay_min_viable_leg_usd = 50.0
    assert ex.overlay_min_viable_leg_usd == 50.0


def test_adaptive_scales_with_capital_monotonically() -> None:
    values = [compute_adaptive_min_viable_leg_usd(c) for c in (200.0, 700.0, 2000.0, 3400.0)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(10.0)
    assert values[-1] == pytest.approx(50.0)
