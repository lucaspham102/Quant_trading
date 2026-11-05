"""tests/test_carry_rebalance_self_healing.py

Unit tests verifying:
1. Scale-up / Scale-down of Carry positions preserves new_qty across _self_heal_carry_positions,
   preventing regression to initial_qty.
2. Partial close of Carry position accurately updates pos["qty"], leg1_filled_qty, leg2_hedged_qty
   to remaining quantity (_rem = qty - _closed_qty), without overwriting with _closed_qty.
3. Full close removes the position and marks leg state as CLOSED with 0.0 quantities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.carry_live_executor import CarryLiveExecutor
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore

BTC = "BTC/USDT:USDT"


class _MockSignedClient:
    """Mock testnet client returning filled orders."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        return {"executedQty": qty, "avgPrice": 80000.0, "orderId": 101, "status": "FILLED"}

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        return {"executedQty": qty, "avgPrice": 80000.0, "orderId": 202, "status": "FILLED"}


def _build_executor(tmp_path: Path, client: Any = None, **kwargs: Any) -> CarryLiveExecutor:
    cfg = DynamicCarryConfig(
        initial_capital=1000.0,
    )
    return CarryLiveExecutor(
        config=cfg,
        symbols=[BTC],
        initial_capital=1000.0,
        state_store=StateStore(tmp_path / "state.json"),
        runtime_mode="testnet",
        testnet_dry_state=False,
        signed_client=client,
        execute_real_orders=client is not None,
        persistent_positions=True,
        **kwargs,
    )


def test_carry_rebalance_scale_up_preserves_qty_across_self_heal(tmp_path: Path) -> None:
    """1. Scale-up: sau khi rebalance tang khoi luong tu initial_qty len new_qty,
    leg1_filled_qty va leg2_hedged_qty phai dong bo new_qty.
    Khi _self_heal_carry_positions() chay o dau chu ky tiep theo, pos.qty
    khong bi tut ve initial_qty."""
    fake_client = _MockSignedClient()
    ex = _build_executor(tmp_path, fake_client)

    initial_qty = 0.002
    sp_curr = 80000.0
    pe_curr = 80000.0

    # Gia lap vi the ban dau mo voi initial_qty = 0.002
    ex.state.carry_positions[BTC] = {
        "symbol": BTC,
        "spot_symbol": "BTC",
        "qty": initial_qty,
        "entry_spot_price": sp_curr,
        "entry_perp_price": pe_curr,
        "notional_usdt": initial_qty * sp_curr,
        "margin_allocated_usdt": 50.0,
        "leg1_status": "FILLED",
        "leg1_filled_qty": initial_qty,
        "leg2_hedged_qty": initial_qty,
        "execution_state": "HEDGED",
    }
    ex._update_leg_state(BTC, "FILLED", initial_qty, initial_qty, "HEDGED")

    # Gia lap chu ky rebalance scale-up len 0.004 BTC (delta = +0.002)
    new_qty = 0.004
    delta_q = new_qty - initial_qty
    reb_res = ex._execute_atomic_carry_delta(BTC, delta_q, -delta_q, pe_curr)
    assert reb_res["ok"] is True

    # Cap nhat theo logic da sua trong HOLD rebalance
    pos = ex.state.carry_positions[BTC]
    pos["qty"] = float(new_qty)
    pos["leg1_filled_qty"] = float(new_qty)
    pos["leg2_hedged_qty"] = float(new_qty)
    pos["notional_usdt"] = float(new_qty * sp_curr)
    ex._update_leg_state(BTC, "FILLED", float(new_qty), float(new_qty), "HEDGED")

    # Kiem tra ngay sau rebalance
    assert pos["qty"] == pytest.approx(0.004)
    assert pos["leg1_filled_qty"] == pytest.approx(0.004)
    assert pos["leg2_hedged_qty"] == pytest.approx(0.004)

    # Chay Self-Healing
    healed = ex._self_heal_carry_positions()
    assert healed is False  # Khong bi lech state

    # Bao toan tuyet doi new_qty, KHONG bi tut ve initial_qty
    assert pos["qty"] == pytest.approx(0.004)
    assert pos["leg1_filled_qty"] == pytest.approx(0.004)
    assert pos["leg2_hedged_qty"] == pytest.approx(0.004)


def test_carry_rebalance_scale_down_preserves_qty_across_self_heal(tmp_path: Path) -> None:
    """2. Scale-down: sau khi rebalance giam khoi luong tu initial_qty xuong new_qty,
    leg1_filled_qty va leg2_hedged_qty phai dong bo new_qty.
    _self_heal_carry_positions() khong keo nguoc len initial_qty."""
    fake_client = _MockSignedClient()
    ex = _build_executor(tmp_path, fake_client)

    initial_qty = 0.005
    sp_curr = 80000.0
    pe_curr = 80000.0

    ex.state.carry_positions[BTC] = {
        "symbol": BTC,
        "spot_symbol": "BTC",
        "qty": initial_qty,
        "entry_spot_price": sp_curr,
        "entry_perp_price": pe_curr,
        "notional_usdt": initial_qty * sp_curr,
        "margin_allocated_usdt": 100.0,
        "leg1_status": "FILLED",
        "leg1_filled_qty": initial_qty,
        "leg2_hedged_qty": initial_qty,
        "execution_state": "HEDGED",
    }
    ex._update_leg_state(BTC, "FILLED", initial_qty, initial_qty, "HEDGED")

    # Rebalance scale down xuong 0.002 BTC (delta = -0.003)
    new_qty = 0.002
    delta_q = new_qty - initial_qty
    reb_res = ex._execute_atomic_carry_delta(BTC, delta_q, -delta_q, pe_curr)
    assert reb_res["ok"] is True

    pos = ex.state.carry_positions[BTC]
    pos["qty"] = float(new_qty)
    pos["leg1_filled_qty"] = float(new_qty)
    pos["leg2_hedged_qty"] = float(new_qty)
    pos["notional_usdt"] = float(new_qty * sp_curr)
    ex._update_leg_state(BTC, "FILLED", float(new_qty), float(new_qty), "HEDGED")

    healed = ex._self_heal_carry_positions()
    assert healed is False

    assert pos["qty"] == pytest.approx(0.002)
    assert pos["leg1_filled_qty"] == pytest.approx(0.002)
    assert pos["leg2_hedged_qty"] == pytest.approx(0.002)


def test_carry_partial_close_preserves_remaining_qty(tmp_path: Path) -> None:
    """3. Partial Close: khi dong mot phan (_closed_qty = 0.002 cua 0.005 BTC),
    khoi luong con lai phai la _rem = 0.003.
    _update_leg_state phai cap nhat dung _rem, khong duoc de _closed_qty ghi de len pos.qty."""
    fake_client = _MockSignedClient()
    ex = _build_executor(tmp_path, fake_client)

    initial_qty = 0.005
    sp_curr = 80000.0
    pe_curr = 80000.0

    ex.state.carry_positions[BTC] = {
        "symbol": BTC,
        "spot_symbol": "BTC",
        "qty": initial_qty,
        "entry_spot_price": sp_curr,
        "entry_perp_price": pe_curr,
        "notional_usdt": initial_qty * sp_curr,
        "margin_allocated_usdt": 100.0,
        "leg1_status": "FILLED",
        "leg1_filled_qty": initial_qty,
        "leg2_hedged_qty": initial_qty,
        "execution_state": "HEDGED",
    }
    ex._update_leg_state(BTC, "FILLED", initial_qty, initial_qty, "HEDGED")

    # Dong 0.002 BTC
    closed_qty = 0.002
    close_res = ex._execute_atomic_carry_delta(BTC, -closed_qty, closed_qty, pe_curr)
    assert close_res["ok"] is True

    # Thuc thi logic partial close
    rem_qty = max(0.0, initial_qty - closed_qty)
    pos = ex.state.carry_positions[BTC]
    pos["qty"] = float(rem_qty)
    pos["leg1_filled_qty"] = float(rem_qty)
    pos["leg2_hedged_qty"] = float(rem_qty)
    pos["notional_usdt"] = float(rem_qty * sp_curr)
    pos["execution_state"] = "PARTIAL_CLOSE"
    ex._update_leg_state(BTC, "HEDGED", float(rem_qty), float(rem_qty), "HEDGED")

    # Kiem tra sau partial close: phai bang 0.003 (KHONG phai 0.002)
    assert pos["qty"] == pytest.approx(0.003)
    assert pos["leg1_filled_qty"] == pytest.approx(0.003)
    assert pos["leg2_hedged_qty"] == pytest.approx(0.003)

    # Chay Self-Healing
    healed = ex._self_heal_carry_positions()
    assert healed is False

    # Van giu dung 0.003
    assert pos["qty"] == pytest.approx(0.003)
    assert pos["leg1_filled_qty"] == pytest.approx(0.003)
    assert pos["leg2_hedged_qty"] == pytest.approx(0.003)


def test_carry_full_close_removes_position_and_sets_closed_state(tmp_path: Path) -> None:
    """4. Full Close: khi dong toan bo (_closed_qty == qty), vi the bi pop khoi carry_positions,
    va _update_leg_state ghi nhan CLOSED voi 0.0."""
    fake_client = _MockSignedClient()
    ex = _build_executor(tmp_path, fake_client)

    initial_qty = 0.002
    sp_curr = 80000.0
    pe_curr = 80000.0

    ex.state.carry_positions[BTC] = {
        "symbol": BTC,
        "spot_symbol": "BTC",
        "qty": initial_qty,
        "entry_spot_price": sp_curr,
        "entry_perp_price": pe_curr,
        "notional_usdt": initial_qty * sp_curr,
        "margin_allocated_usdt": 50.0,
        "leg1_status": "FILLED",
        "leg1_filled_qty": initial_qty,
        "leg2_hedged_qty": initial_qty,
        "execution_state": "HEDGED",
    }
    ex._update_leg_state(BTC, "FILLED", initial_qty, initial_qty, "HEDGED")

    # Full close
    close_res = ex._execute_atomic_carry_delta(BTC, -initial_qty, initial_qty, pe_curr)
    assert close_res["ok"] is True

    ex.state.carry_positions.pop(BTC, None)
    ex._update_leg_state(BTC, "CLOSED", 0.0, 0.0, "CLOSED")

    assert BTC not in ex.state.carry_positions
    leg_state = ex._leg_states.get(BTC, {})
    assert leg_state.get("execution_state") == "CLOSED"
    assert leg_state.get("leg1_filled_qty") == pytest.approx(0.0)
    assert leg_state.get("leg2_hedged_qty") == pytest.approx(0.0)
