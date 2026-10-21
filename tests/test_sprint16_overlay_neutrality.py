"""Test Sprint 16 Chang 1 - Overlay Lot-Rounding Neutrality Guard.

Bao phu:
1. Tai hien dung case Sprint 15 (Long=331.06 vs Short=407.68) -> |NetDelta| <= 10 USD.
2. Nhieu ma moi chan -> can bang van giu |NetDelta| <= 10 USD.
3. Von qua nho (mot chan < 50 USD) -> tra 0.0 cho TAT CA (khong bao gio naked leg).
4. Chan BTC bi floor ve 0 vi duoi min_notional -> ca hai chan ve 0.
5. Input da can bang san -> giu nguyen qty da floor (khong tu y co them).
6. Nguong max_imbalance_usd rat lon -> tra ve dung qty floor tho.
7. Guard idempotent: chay 2 lan cho cung ket qua.
8. Gia khong hop le -> ma do ve 0 va khong lam vo chan con lai.
9. Tich hop `build_barbell_allocations` (tier1b) -> long/short lech <= 10 USD.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, enforce_overlay_dollar_neutrality
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"
BNB = "BNB/USDT:USDT"

PRICES = {BTC: 96_558.0, ETH: 4_200.0, SOL: 150.0, BNB: 600.0}


def _leg_usd(qty: dict[str, float], prices: dict[str, float]) -> tuple[float, float]:
    long_usd = sum(q * prices[s] for s, q in qty.items() if q > 0.0)
    short_usd = sum(abs(q) * prices[s] for s, q in qty.items() if q < 0.0)
    return long_usd, short_usd


# ---------------------------------------------------------------------------
# 1-2. Tai hien Sprint 15 va case nhieu ma
# ---------------------------------------------------------------------------


def test_sprint15_repro_long_short_imbalance_is_neutralized() -> None:
    target = {BTC: 331.06, ETH: -407.68}
    qty = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    long_usd, short_usd = _leg_usd(qty, PRICES)

    assert long_usd >= 50.0 and short_usd >= 50.0
    assert abs(long_usd - short_usd) <= 10.0
    # truoc khi sua, lech la -76.62 USD; sau khi sua phai nho hon nhieu
    assert abs(long_usd - short_usd) < 10.0


def test_multi_symbol_legs_are_balanced() -> None:
    target = {BTC: 200.0, ETH: 131.06, SOL: -207.68, BNB: -200.0}
    qty = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    long_usd, short_usd = _leg_usd(qty, PRICES)

    assert long_usd > 0.0 and short_usd > 0.0
    assert abs(long_usd - short_usd) <= 10.0


# ---------------------------------------------------------------------------
# 3-4. Chan naked leg
# ---------------------------------------------------------------------------


def test_tiny_capital_returns_zero_for_both_legs() -> None:
    target = {BTC: 30.0, ETH: -30.0}
    qty = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    assert qty == {BTC: 0.0, ETH: 0.0}


def test_btc_floor_to_zero_kills_counter_leg() -> None:
    # BTC 96.558 USD < 1 step (0.001 BTC = 96.558 USD) nhung van duoi min_notional
    target = {BTC: 50.0, ETH: -400.0}
    qty = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    assert qty[BTC] == 0.0
    assert all(q == 0.0 for q in qty.values())


# ---------------------------------------------------------------------------
# 5-8. Tinh chat ham
# ---------------------------------------------------------------------------


def test_already_balanced_input_keeps_floored_qty() -> None:
    # BTC long 0.002 = 193.12 USD, SOL short 1.3 = 195.0 USD -> lech 1.88 USD
    target = {BTC: 193.12, SOL: -195.0}
    qty = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    assert qty[BTC] == pytest.approx(0.002)
    assert qty[SOL] == pytest.approx(-1.3)
    long_usd, short_usd = _leg_usd(qty, PRICES)
    assert abs(long_usd - short_usd) <= 10.0


def test_large_imbalance_threshold_returns_raw_floor() -> None:
    target = {BTC: 331.06, ETH: -407.68}
    qty = enforce_overlay_dollar_neutrality(
        target, PRICES, EXCHANGE_LOT_SPECS, max_imbalance_usd=1000.0
    )
    assert qty[BTC] == pytest.approx(0.003)   # floor(331.06/96558) = 0.003
    assert qty[ETH] == pytest.approx(-0.09)   # floor(407.68/4200) = 0.09


def test_guard_is_deterministic() -> None:
    target = {BTC: 331.06, ETH: -407.68}
    first = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    second = enforce_overlay_dollar_neutrality(target, PRICES, EXCHANGE_LOT_SPECS)
    assert first == second
    long1, short1 = _leg_usd(first, PRICES)
    assert abs(long1 - short1) <= 10.0


def test_invalid_price_zeroes_symbol_without_breaking_other_leg() -> None:
    target = {BTC: 331.06, ETH: -407.68}
    bad_prices = dict(PRICES)
    bad_prices[BTC] = 0.0
    qty = enforce_overlay_dollar_neutrality(target, bad_prices, EXCHANGE_LOT_SPECS)
    assert qty[BTC] == 0.0
    assert all(q == 0.0 for q in qty.values())


# ---------------------------------------------------------------------------
# 9. Tich hop executor
# ---------------------------------------------------------------------------

CORE4 = (BTC, ETH, SOL, BNB)


def _overlay_universe(bars: int = 900) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2025-01-01", periods=bars, freq="1h", tz="UTC")
    base = {BTC: 96_000.0, ETH: 4_200.0, SOL: 150.0, BNB: 600.0}
    slopes = {BTC: 1.0, ETH: 2.5, SOL: -1.8, BNB: -2.0}
    out: dict[str, pd.DataFrame] = {}
    for sym in CORE4:
        close = base[sym] * (1.0 + slopes[sym] * np.arange(bars) / 1_000.0)
        out[sym] = pd.DataFrame({"perp_close": close, "spot_close": close * 0.9998}, index=idx)
    return out


def test_build_barbell_allocations_enforces_dollar_neutrality(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "state.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=raw_dir,
        state_file_path=state_path,
    )
    executor = CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(initial_capital=3000.0),
        mode="paper",
        symbols=list(CORE4),
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
        capital=3000.0,
        initial_capital=3000.0,
        overlay_config=BarbellConfig(initial_capital=3000.0, overlay_lookback_hours=336),
        tier1b_gate=True,
    )
    universe = _overlay_universe()
    alloc = executor.build_barbell_allocations(universe, {sym: 0.0 for sym in CORE4})
    overlay = alloc["momentum_overlay"]

    if overlay.get("allocations"):
        long_usd = sum(
            float(a["notional"]) for a in overlay["allocations"] if "long" in str(a["side"])
        )
        short_usd = sum(
            float(a["notional"]) for a in overlay["allocations"] if "short" in str(a["side"])
        )
        assert long_usd > 0.0 and short_usd > 0.0
        assert abs(long_usd - short_usd) <= 10.0
    delta = getattr(executor, "last_overlay_delta", {})
    assert abs(float(delta.get("net_delta_usd", 0.0))) <= 10.0
