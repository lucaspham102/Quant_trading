"""Sprint 24: Unit tests cho fee-dust clamp, overlay auto-rollback va split-wallet plan.

Bao phu 8 kich ban bat buoc:
1. test_close_clamps_spot_sell_to_free_balance - Spot SELL bi clamp ve free balance thuc te.
2. test_close_unsellable_fee_dust_flattens_state - Fee dust < 1 lot step -> khong ban, dong state.
3. test_open_records_base_asset_commission - Phi tinh bang base asset duoc tru va log warning.
4. test_close_without_free_balance_helper_keeps_old_behaviour - Khong co helper -> giu hanh vi cu.
5. test_overlay_imbalance_auto_rollback_on_leg_error - Lenh overlay loi/lech -> rollback, phuc hoi state.
6. test_overlay_ok_when_balanced - Long/Short can bang -> status OK, ghi state overlay.
7. test_split_wallet_plan_400_usdt - Ke hoach chia vi cho 400 USDT dung so lieu.
8. test_split_wallet_readiness_shortfalls - Thieu hut Spot/Futures duoc tinh dung.

Bo sung regression cho audit Sprint 24 (9-14):
9. test_close_free_balance_unknown_does_not_skip_sell - Helper tra None (loi API) -> KHONG clamp.
10. test_open_hedges_on_net_spot_after_base_fee - Perp hedge theo net_spot_received.
11. test_overlay_accepts_state_dict_current - current dang dict {qty, side} khong con TypeError.
12. test_overlay_rollback_reduce_only_false_for_reduction_leg - Rollback leg giam vi the dung reduce_only=False.
13. test_overlay_partial_rollback_keeps_actual_state - Rollback that bai -> state theo vi the THAT.
14. test_overlay_close_partial_fill_keeps_remaining_qty - Fill mot phan -> giu qty con lai trong state.

Khong co network, khong dung wall-clock thuc, moi so sanh float dung pytest.approx.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.carry_live_executor import CarryLiveExecutor
from src.dynamic_carry_basis import DynamicCarryConfig
from src.quant_invariants import (
    compute_split_wallet_canary_plan,
    validate_wallet_split_readiness,
)
from src.state_store import StateStore

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
DOGE = "DOGE/USDT:USDT"
XRP = "XRP/USDT:USDT"


def _executor(tmp_path: Path, client: Any) -> CarryLiveExecutor:
    """Tao CarryLiveExecutor chuan canary 400 USDT, dinh tuyen lenh THAT."""
    return CarryLiveExecutor(
        carry_config=DynamicCarryConfig(initial_capital=400.0, top_k=4, eta_override=0.80),
        mode="testnet",
        capital=400.0,
        initial_capital=400.0,
        state_store=StateStore(tmp_path / "state.json"),
        runtime_mode="testnet",
        testnet_dry_state=False,
        signed_client=client,
        execute_real_orders=True,
        persistent_positions=True,
    )


class _FakeSignedClient:
    """Ghi lai moi lenh (tuple) va luon fill day du theo qty dat."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.events: list[tuple[Any, ...]] = []

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        self.events.append(("spot", symbol, side, qty))
        return {"executedQty": qty, "avgPrice": 81000.0, "orderId": 1}

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        self.events.append(("perp", symbol, side, qty, reduce_only))
        return {"executedQty": qty, "avgPrice": 81000.0, "orderId": 2}


class _ClampedSellClient(_FakeSignedClient):
    """Co get_spot_free_balance; perp BUY reduceOnly va spot SELL tra qty chi dinh."""

    def __init__(self, free_qty: float, perp_fill_qty: float, spot_fill_qty: float) -> None:
        super().__init__()
        self.free_qty = free_qty
        self.perp_fill_qty = perp_fill_qty
        self.spot_fill_qty = spot_fill_qty

    def get_spot_free_balance(self, symbol: str) -> float:
        return self.free_qty

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        self.events.append(("spot", symbol, side, qty))
        return {"executedQty": self.spot_fill_qty, "avgPrice": 81000.0, "orderId": 11}

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        self.events.append(("perp", symbol, side, qty, reduce_only))
        return {"executedQty": self.perp_fill_qty, "avgPrice": 81000.0, "orderId": 12}


class _OpenCommissionClient(_FakeSignedClient):
    """Spot BUY tra fills phi bang chinh base asset (BTC)."""

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        self.events.append(("spot", symbol, side, qty))
        return {
            "executedQty": 0.001,
            "avgPrice": 81000.0,
            "orderId": 1,
            "fills": [{"commission": 0.000002, "commissionAsset": "BTC"}],
        }


class _NoFreeHelperClient(_FakeSignedClient):
    """Khong co get_spot_free_balance -> khong clamp; spot SELL tra executedQty=0.002."""

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        self.events.append(("spot", symbol, side, qty))
        return {"executedQty": 0.002, "avgPrice": 81000.0, "orderId": 21}


class _OverlayLegErrorClient(_FakeSignedClient):
    """Futures BUY fill day du; SELL cho DOGE nem RuntimeError. query_order tra qty da dat."""

    def __init__(self) -> None:
        super().__init__()
        self._order_qty: dict[Any, float] = {}
        self._next_id = 100

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        if symbol == DOGE and side == "SELL":
            raise RuntimeError("api error")
        self._next_id += 1
        oid = self._next_id
        self._order_qty[oid] = float(qty)
        self.calls.append(("perp", side, qty, reduce_only))
        self.events.append(("perp", symbol, side, qty, reduce_only))
        return {"executedQty": qty, "avgPrice": 0.0, "orderId": oid}

    def query_order(
        self, venue: str, symbol: str, order_id: Any, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        return {
            "executedQty": self._order_qty.get(order_id, 0.0),
            "avgPrice": 0.0,
            "status": "FILLED",
            "orderId": order_id,
        }


class _OverlayFillClient(_FakeSignedClient):
    """Fill du moi lenh overlay, query_order tra qty da dat neu bi goi."""

    def __init__(self) -> None:
        super().__init__()
        self._order_qty: dict[Any, float] = {}
        self._next_id = 500

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self._next_id += 1
        oid = self._next_id
        self._order_qty[oid] = float(qty)
        self.calls.append(("perp", side, qty, reduce_only))
        self.events.append(("perp", symbol, side, qty, reduce_only))
        return {"executedQty": qty, "avgPrice": 0.0, "orderId": oid}

    def query_order(
        self, venue: str, symbol: str, order_id: Any, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        return {
            "executedQty": self._order_qty.get(order_id, 0.0),
            "avgPrice": 0.0,
            "status": "FILLED",
            "orderId": order_id,
        }


def test_close_clamps_spot_sell_to_free_balance(tmp_path: Path) -> None:
    """1. Free balance 0.001998 < 0.002 -> spot SELL bi clamp ve 0.001 (1 lot)."""
    fake = _ClampedSellClient(free_qty=0.001998, perp_fill_qty=0.002, spot_fill_qty=0.001)
    ex = _executor(tmp_path, fake)

    res = ex._execute_atomic_carry_delta(BTC, -0.002, 0.002, 81000.0)

    sell_calls = [c for c in fake.calls if c[0] == "spot" and c[1] == "SELL"]
    assert len(sell_calls) == 1
    assert sell_calls[0][2] == pytest.approx(0.001)
    assert res["ok"] is True
    assert res["event"] == "ATOMIC_CARRY_CLOSE_FILLED"
    assert res["spot_filled_qty"] == pytest.approx(0.001)


def test_close_unsellable_fee_dust_flattens_state(tmp_path: Path) -> None:
    """2. Free balance 0.0009 < 1 lot step -> khong gui spot SELL, coi nhu da dong."""
    fake = _ClampedSellClient(free_qty=0.0009, perp_fill_qty=0.001, spot_fill_qty=0.001)
    ex = _executor(tmp_path, fake)

    res = ex._execute_atomic_carry_delta(BTC, -0.002, 0.002, 81000.0)

    assert res["ok"] is True
    assert res["event"] == "ATOMIC_CLOSE_DUST_UNSELLABLE"
    assert res["reason"] == "SPOT_FEE_DUST_UNSELLABLE_LOT"
    assert res["dust_qty"] == pytest.approx(0.0009)
    assert not [c for c in fake.calls if c[0] == "spot"]


def test_open_records_base_asset_commission(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """3. Phi 0.000002 BTC duoc tru vao net_spot_received_qty va log warning."""
    fake = _OpenCommissionClient()
    ex = _executor(tmp_path, fake)

    with caplog.at_level(logging.WARNING):
        res = ex._execute_atomic_carry_delta(BTC, 0.001, -0.001, 81000.0)

    assert res["net_spot_received_qty"] == pytest.approx(0.000998)
    assert res["ok"] is True
    assert "SPOT_FEE_DEDUCTED_FROM_BASE_ASSET" in caplog.text


def test_close_without_free_balance_helper_keeps_old_behaviour(tmp_path: Path) -> None:
    """4. Khong co get_spot_free_balance -> ban du 0.002, khong clamp."""
    fake = _NoFreeHelperClient()
    ex = _executor(tmp_path, fake)

    res = ex._execute_atomic_carry_delta(BTC, -0.002, 0.002, 81000.0)

    sell_calls = [c for c in fake.calls if c[0] == "spot" and c[1] == "SELL"]
    assert len(sell_calls) == 1
    assert sell_calls[0][2] == pytest.approx(0.002)
    assert res["ok"] is True
    assert res["event"] == "ATOMIC_CARRY_CLOSE_FILLED"


def test_overlay_imbalance_auto_rollback_on_leg_error(tmp_path: Path) -> None:
    """5. Overlay lech/loi -> rollback nguoc lenh da fill va khoi phuc snapshot state."""
    fake = _OverlayLegErrorClient()
    ex = _executor(tmp_path, fake)
    snapshot = {XRP: {"qty": 5.0, "side": "short", "entry_price": 1.0}}
    ex.state.overlay_positions = {XRP: {"qty": 5.0, "side": "short", "entry_price": 1.0}}

    res = ex._execute_overlay_deltas_real(
        {ETH: 0.01, DOGE: -100.0},
        {},
        {ETH: 2400.0, DOGE: 0.08},
    )

    assert res["status"] == "ROLLED_BACK"
    assert res["restored_state"] is True
    assert res["rollback_qty"] > 0
    assert ex.state.overlay_positions == snapshot
    assert ETH not in ex.state.overlay_positions

    # Rollback phai nguoc chieu lenh da fill (ETH BUY -> SELL) va reduce_only=True.
    eth_rollbacks = [
        e for e in fake.events
        if e[0] == "perp" and e[1] == ETH and e[2] == "SELL" and e[4] is True
    ]
    assert eth_rollbacks
    assert eth_rollbacks[0][3] == pytest.approx(0.01)
    assert ex.state.overlay_positions[XRP]["qty"] == pytest.approx(5.0)


def test_overlay_ok_when_balanced(tmp_path: Path) -> None:
    """6. Long ETH 24 USD / Short DOGE 24 USD -> lech 0, status OK, ghi state."""
    fake = _OverlayFillClient()
    ex = _executor(tmp_path, fake)

    res = ex._execute_overlay_deltas_real(
        {ETH: 0.01, DOGE: -300.0},
        {},
        {ETH: 2400.0, DOGE: 0.08},
    )

    assert res["status"] == "OK"
    assert res["rollback_qty"] == 0.0
    assert res["restored_state"] is False
    assert ex.state.overlay_positions[ETH]["side"] == "long"
    assert ex.state.overlay_positions[ETH]["qty"] == pytest.approx(0.01)
    assert ex.state.overlay_positions[DOGE]["side"] == "short"
    assert ex.state.overlay_positions[DOGE]["qty"] == pytest.approx(300.0)


def test_split_wallet_plan_400_usdt() -> None:
    """7. Plan 400 USDT: Spot 260 / Futures 140 / Overlay 60 / leverage 2.29."""
    plan = compute_split_wallet_canary_plan(400.0)

    assert plan["recommended_spot_usdt"] == pytest.approx(260.0)
    assert plan["recommended_futures_usdt"] == pytest.approx(140.0)
    assert plan["overlay_gross_notional_usd"] == pytest.approx(60.0)
    assert plan["total_futures_notional_usd"] == pytest.approx(320.0)
    assert plan["effective_futures_leverage"] == pytest.approx(2.29)
    assert plan["is_safe_leverage"] is True
    assert plan["carry_slot_usdt"] == pytest.approx(130.0)
    assert plan["recommended_bnb_buffer_usd"] == pytest.approx(10.0)


def test_split_wallet_readiness_shortfalls() -> None:
    """8. Du vi -> ready; thieu vi -> tinh dung shortfall Spot 147 / Futures 83."""
    plan = compute_split_wallet_canary_plan(400.0)

    ok = validate_wallet_split_readiness(260.0, 140.0, plan)
    assert ok["ready"] is True
    assert ok["spot_shortfall_usd"] == pytest.approx(0.0)
    assert ok["futures_shortfall_usd"] == pytest.approx(0.0)

    r = validate_wallet_split_readiness(100.0, 50.0, plan)
    assert r["ready"] is False
    assert r["spot_shortfall_usd"] == pytest.approx(147.0)
    assert r["futures_shortfall_usd"] == pytest.approx(83.0)
    assert r["plan"] is plan


# ---------------------------------------------------------------------------
# Regression cho audit Sprint 24 (cac loi HIGH/MEDIUM da sua)
# ---------------------------------------------------------------------------

class _UnknownFreeClient(_FakeSignedClient):
    """get_spot_free_balance tra None = khong doc duoc so du (loi API/dry_run)."""

    def get_spot_free_balance(self, symbol: str) -> None:
        return None

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        return {"executedQty": qty, "avgPrice": 81000.0, "orderId": 31}


class _OpenNetSpotClient(_FakeSignedClient):
    """Spot BUY 0.003 BTC nhung phi 0.000002 BTC tru vao base asset."""

    def place_spot_market_order(
        self, symbol: str, side: str, qty: float, timeout_sec: float = 10.0
    ) -> dict[str, Any]:
        self.calls.append(("spot", side, qty))
        return {
            "executedQty": 0.003,
            "avgPrice": 81000.0,
            "orderId": 41,
            "fills": [{"commission": 0.000002, "commissionAsset": "BTC"}],
        }


class _SelectiveRaiseClient(_FakeSignedClient):
    """Fill moi lenh tru cac cap (symbol, side) nam trong raise_on.

    Co snapshot healthy de executor xac minh duoc vi the that (_pos_now).
    """

    def __init__(self, raise_on: set[tuple[str, str]], with_snapshot: bool = True) -> None:
        super().__init__()
        self.raise_on = set(raise_on)
        self.with_snapshot = with_snapshot
        self.positions: dict[str, float] = {}
        self._next_id = 700

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        if (symbol, side) in self.raise_on:
            raise RuntimeError("api error")
        self._next_id += 1
        self.calls.append(("perp", side, qty, reduce_only))
        self.events.append(("perp", symbol, side, qty, reduce_only))
        sign = 1.0 if side == "BUY" else -1.0
        self.positions[symbol] = float(self.positions.get(symbol, 0.0)) + sign * float(qty)
        return {"executedQty": qty, "avgPrice": 0.0, "orderId": self._next_id}

    def fetch_exchange_reconcile_snapshot(
        self, symbols: list[str], prices: dict[str, float] | None = None
    ) -> dict[str, Any]:
        if not self.with_snapshot:
            return {"snapshot_healthy": False, "perp_positions": {}}
        return {
            "snapshot_healthy": True,
            "perp_positions": {s: float(self.positions.get(s, 0.0)) for s in symbols},
        }


class _PartialCloseClient(_FakeSignedClient):
    """Lenh dong chi khop toi da 0.01 (fill mot phan)."""

    def __init__(self, max_fill: float) -> None:
        super().__init__()
        self.max_fill = max_fill

    def place_futures_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        self.calls.append(("perp", side, qty, reduce_only))
        self.events.append(("perp", symbol, side, qty, reduce_only))
        return {"executedQty": min(float(qty), self.max_fill), "avgPrice": 0.0, "orderId": 51}


def test_close_free_balance_unknown_does_not_skip_sell(tmp_path: Path) -> None:
    """9. Helper tra None -> khong biet so du -> van ban du qty yeu cau (khong clamp ve 0)."""
    fake = _UnknownFreeClient()
    ex = _executor(tmp_path, fake)

    res = ex._execute_atomic_carry_delta(BTC, -0.002, 0.002, 81000.0)

    sell_calls = [c for c in fake.calls if c[0] == "spot" and c[1] == "SELL"]
    assert len(sell_calls) == 1
    assert sell_calls[0][2] == pytest.approx(0.002)
    assert res["event"] == "ATOMIC_CARRY_CLOSE_FILLED"
    assert res["ok"] is True


def test_open_hedges_on_net_spot_after_base_fee(tmp_path: Path) -> None:
    """10. Hedge = floor(spot_filled) (giu delta-neutral), over-hedge chi bang phi base asset.

    Vong 2: floor(net_spot_received) tung lam ho TRON 1 lot step (BTC ~80 USD) nen doi lai
    chien luoc hedge theo spot_filled; sai lech con lai <= base_commission va duoc bao ro.
    """
    fake = _OpenNetSpotClient()
    ex = _executor(tmp_path, fake)

    res = ex._execute_atomic_carry_delta(BTC, 0.003, -0.003, 81000.0)

    perp_calls = [c for c in fake.calls if c[0] == "perp" and c[1] == "SELL"]
    assert len(perp_calls) == 1
    assert perp_calls[0][2] == pytest.approx(0.003)
    assert res["net_spot_received_qty"] == pytest.approx(0.002998)
    assert res["perp_over_hedge_qty"] == pytest.approx(0.000002)
    assert res["ok"] is True


def test_overlay_accepts_state_dict_current(tmp_path: Path) -> None:
    """11. current la dict {qty, side} -> chuan hoa duoc, khong TypeError, giam vi the dung."""
    fake = _OverlayFillClient()
    ex = _executor(tmp_path, fake)
    current = {XRP: {"qty": 100.0, "side": "short", "entry_price": 1.278}}

    res = ex._execute_overlay_deltas_real(
        {XRP: -40.0, ETH: 0.02}, current, {XRP: 1.278, ETH: 2400.0}
    )

    assert res["status"] == "OK"
    assert res["n_orders"] == 2
    assert ex.state.overlay_positions[XRP]["side"] == "short"
    assert ex.state.overlay_positions[XRP]["qty"] == pytest.approx(40.0)
    assert ex.state.overlay_positions[ETH]["side"] == "long"
    assert ex.state.overlay_positions[ETH]["qty"] == pytest.approx(0.02)


def test_overlay_rollback_reduce_only_false_for_reduction_leg(tmp_path: Path) -> None:
    """12. Leg giam vi the cu (cur != 0) -> rollback phai reduce_only=False."""
    fake = _SelectiveRaiseClient(raise_on={(ETH, "BUY")})
    ex = _executor(tmp_path, fake)
    ex.state.overlay_positions = {DOGE: {"qty": 300.0, "side": "short", "entry_price": 0.08}}

    res = ex._execute_overlay_deltas_real(
        {DOGE: -100.0, ETH: 0.01},
        dict(ex.state.overlay_positions),
        {DOGE: 0.08, ETH: 2400.0},
    )

    assert res["status"] == "ROLLED_BACK"
    rollbacks = [
        e for e in fake.events
        if e[0] == "perp" and e[1] == DOGE and e[2] == "SELL"
    ]
    assert rollbacks
    assert rollbacks[0][4] is False
    assert rollbacks[0][3] == pytest.approx(200.0)


def test_overlay_partial_rollback_keeps_actual_state(tmp_path: Path) -> None:
    """13. Rollback that bai -> status ROLLBACK_PARTIAL, state theo vi the THAT (khong snapshot)."""
    fake = _SelectiveRaiseClient(raise_on={(ETH, "BUY"), (DOGE, "SELL")})
    ex = _executor(tmp_path, fake)
    ex.state.overlay_positions = {DOGE: {"qty": 300.0, "side": "short", "entry_price": 0.08}}

    res = ex._execute_overlay_deltas_real(
        {DOGE: -100.0, ETH: 0.01},
        dict(ex.state.overlay_positions),
        {DOGE: 0.08, ETH: 2400.0},
    )

    assert res["status"] == "ROLLBACK_PARTIAL"
    assert res["restored_state"] is False
    assert res["rollback_failures"] == [DOGE]
    assert ex.state.overlay_positions[DOGE]["qty"] == pytest.approx(100.0)
    assert ETH not in ex.state.overlay_positions


def test_overlay_close_partial_fill_keeps_remaining_qty(tmp_path: Path) -> None:
    """14. Dong overlay fill mot phan -> giu qty con lai, zero_residual=False."""
    fake = _PartialCloseClient(max_fill=0.01)
    ex = _executor(tmp_path, fake)
    ex.state.overlay_positions = {ETH: {"qty": 0.02, "side": "long", "entry_price": 2400.0}}

    out = ex.close_all_persistent_overlay_positions({ETH: 2400.0})

    assert out["zero_residual"] is False
    assert out["closed_overlay"] == []
    assert ex.state.overlay_positions[ETH]["qty"] == pytest.approx(0.01)
    assert any("OVERLAY_CLOSE_PARTIAL" in e for e in out["errors"])


def test_overlay_unverified_position_when_snapshot_unavailable(tmp_path: Path) -> None:
    """15. Lenh loi + khong doc duoc vi the that -> UNVERIFIED_POSITION, KHONG restore snapshot."""
    fake = _SelectiveRaiseClient(raise_on={(ETH, "BUY")}, with_snapshot=False)
    ex = _executor(tmp_path, fake)
    ex.state.overlay_positions = {DOGE: {"qty": 300.0, "side": "short", "entry_price": 0.08}}

    res = ex._execute_overlay_deltas_real(
        {DOGE: -100.0, ETH: 0.01},
        dict(ex.state.overlay_positions),
        {DOGE: 0.08, ETH: 2400.0},
    )

    assert res["status"] == "UNVERIFIED_POSITION"
    assert res["unverified_symbols"] == [ETH]
    assert res["restored_state"] is False
    # DOGE rollback thanh cong -> ve -300; ETH chua xac minh nen state KHONG duoc restore
    # ve snapshot mot cach mu quang (restored_state False, co unverified_symbols).
    assert ex.state.overlay_positions[DOGE]["qty"] == pytest.approx(300.0)
