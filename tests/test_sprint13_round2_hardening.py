"""Unit test Sprint 13 — Chặng 1: vá 4 lỗ hổng edge-case vòng 2.

Phạm vi
-------
- **R2-1**: kill-switch chỉ xóa vị thế khi lệnh đóng khẩn cấp thành công; thất bại
  thì giữ nguyên vị thế + bật ``kill_switch_pending_close``; chu kỳ sau tự đóng nốt.
- **R2-2**: cờ ``drift_unrepairable`` tự phục hồi khi lệch vượt 1 lô, và drift về 0
  thì gỡ cờ.
- **R2-3**: ``reinvest_profits=False`` co vốn hoạt động theo equity khi đang lỗ.
- **R2-4**: orderbook lỗi/crossed book bị chặn ở MỌI chế độ.

Nguyên tắc offline: không gọi mạng, không sleep, mọi file nằm trong ``tmp_path``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import _simulate_barbell_equity
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.cross_exchange_carry import LOT_STEP_SIZES
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore

PERP_SYMBOL = "BTC/USDT:USDT"
SPOT_SYMBOL = "BTC/USDT"
STEP_BTC = LOT_STEP_SIZES[PERP_SYMBOL]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_frame(price: float) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    return pd.DataFrame({"close": float(price), "spot_close": float(price)}, index=idx).astype(
        "float64"
    )


class FakeExchange:
    """Sàn giả: ghi lại lệnh, có thể cấu hình để ném lỗi hoặc trả status lạ."""

    def __init__(
        self,
        bids: Sequence[Sequence[float]] | None = None,
        asks: Sequence[Sequence[float]] | None = None,
        with_order_book: bool = True,
        fail: bool = False,
        status: str = "closed",
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self.bids = [list(level) for level in (bids or [[99.9, 10]])]
        self.asks = [list(level) for level in (asks or [[100.1, 10]])]
        self.fail = fail
        self.status = status
        if with_order_book:
            self.fetch_order_book = self._fetch_order_book  # type: ignore[assignment]

    def _fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        return {
            "bids": [list(level) for level in self.bids],
            "asks": [list(level) for level in self.asks],
        }

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.orders.append(
            {
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": float(amount),
                "price": None if price is None else float(price),
                "params": dict(params) if params is not None else {},
            }
        )
        if self.fail:
            raise RuntimeError("exchange down")
        if type == "limit":
            return {"id": "leg1-1", "status": "open", "filled": 0.0}
        return {"id": f"m{len(self.orders)}", "status": self.status, "filled": float(amount)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        self.cancelled.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}


def _make_executor(
    tmp_path: Path,
    *,
    mode: str = "paper",
    capital: float = 3000.0,
    perp: Any = None,
    spot: Any = None,
) -> CarryLiveExecutor:
    state_path = tmp_path / "s.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=tmp_path / "raw",
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(),
        mode=mode,  # type: ignore[arg-type]
        symbols=[PERP_SYMBOL],
        capital=capital,
        perp_exchange=perp,
        spot_exchange=spot,
        exchange_map={"binance": perp, "binance_spot": spot},
        state_store=StateStore(state_path),
        raw_data_dir=tmp_path / "raw",
        max_allowed_spread_bps=10.0,
        max_live_drawdown_pct=0.06,
    )


def _seed_carry(executor: CarryLiveExecutor, qty: float = 0.1, price: float = 60000.0) -> dict[str, Any]:
    pos = {
        "qty": float(qty),
        "spot_qty": float(qty),
        "spot_symbol": SPOT_SYMBOL,
        "entry_spot_price": float(price),
        "entry_perp_price": float(price),
        "spot_price": float(price),
        "perp_price": float(price),
    }
    executor.state.carry_positions[PERP_SYMBOL] = dict(pos)
    return pos


def _seed_overlay(executor: CarryLiveExecutor, qty: float = 1.0, price: float = 60000.0) -> None:
    executor.state.overlay_positions[PERP_SYMBOL] = {
        "qty": float(qty),
        "side": "short",
        "entry_price": float(price),
        "ts": "2024-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# R2-1: Partial emergency flatten
# ---------------------------------------------------------------------------


def test_r2_1_close_failure_keeps_position_and_sets_pending_flag(tmp_path: Path) -> None:
    """Đóng thất bại -> vị thế còn nguyên trong state, bật pending_close, không pop."""
    perp = FakeExchange(fail=True)
    spot = FakeExchange(fail=True)
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _seed_carry(executor)
    executor.state.peak_equity = 3000.0
    executor.state.current_equity = 3000.0

    triggered = executor.check_kill_switch(2812.5)

    assert triggered is True
    assert PERP_SYMBOL in executor.state.carry_positions, "vi the phai con lai khi dong that bai"
    assert executor.state.kill_switch_pending_close is True
    assert executor.state.kill_switch_triggered is True


def test_r2_1_close_success_pops_position_and_clears_pending_flag(tmp_path: Path) -> None:
    """Đóng thành công -> vị thế bị xóa và pending_close = False."""
    perp = FakeExchange(status="closed")
    spot = FakeExchange(status="closed")
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _seed_carry(executor)
    _seed_overlay(executor)
    executor.state.peak_equity = 3000.0

    assert executor.check_kill_switch(2812.5) is True

    assert executor.state.carry_positions == {}
    assert executor.state.overlay_positions == {}
    assert executor.state.kill_switch_pending_close is False


def test_r2_1_partial_close_keeps_only_unclosed_position(tmp_path: Path) -> None:
    """Chỉ vị thế đóng được mới bị xóa; vị thế lỗi vẫn nằm trong state."""
    perp = FakeExchange(status="closed")
    spot = FakeExchange(status="closed")
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _seed_carry(executor)
    executor.state.carry_positions["ETH/USDT:USDT"] = {
        "qty": 1.0,
        "spot_qty": 1.0,
        "spot_symbol": "ETH/USDT",
        "entry_spot_price": 3000.0,
        "entry_perp_price": 3000.0,
    }
    # Chỉ ETH lỗi: spot exchange từ chối mọi lệnh có symbol ETH.
    original_create = spot.create_order

    def flaky_create(symbol: str, *a: Any, **kw: Any) -> dict[str, Any]:
        if symbol.startswith("ETH"):
            raise RuntimeError("reject ETH")
        return original_create(symbol, *a, **kw)

    spot.create_order = flaky_create  # type: ignore[assignment]
    executor.state.peak_equity = 3000.0

    executor.check_kill_switch(2812.5)

    assert PERP_SYMBOL not in executor.state.carry_positions
    assert "ETH/USDT:USDT" in executor.state.carry_positions, "vi the dong loi phai duoc giu lai"
    assert executor.state.kill_switch_pending_close is True


def test_r2_1_flush_pending_closes_stuck_positions(tmp_path: Path) -> None:
    """Chu kỳ sau gọi flush -> đóng nốt vị thế kẹt và tắt pending_close."""
    perp = FakeExchange(status="closed")
    spot = FakeExchange(status="closed")
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _seed_carry(executor)
    executor.state.kill_switch_triggered = True
    executor.state.kill_switch_pending_close = True

    closed_carry, closed_overlay = executor._flush_pending_emergency_close()

    assert closed_carry == 1
    assert closed_overlay == 0
    assert executor.state.carry_positions == {}
    assert executor.state.kill_switch_pending_close is False


def test_r2_1_pending_close_blocks_new_orders(tmp_path: Path) -> None:
    """pending_close = True (chưa đóng xong) -> reconcile_once không mở lệnh mới."""
    perp = FakeExchange(status="closed")
    spot = FakeExchange(status="closed")
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=spot)
    executor.state.kill_switch_triggered = True
    executor.state.kill_switch_pending_close = True
    executor.state.peak_equity = 3000.0
    executor.state.current_equity = 3000.0

    plan = executor.reconcile_once({})

    assert plan.get("kill_switch") is True
    assert plan.get("actions") == []


# ---------------------------------------------------------------------------
# R2-2: Self-healing sub-lot drift flag
# ---------------------------------------------------------------------------


def test_r2_2_sub_lot_drift_sets_flag_without_order(tmp_path: Path) -> None:
    """Lệch 0.0004 BTC (< 1 lô) -> gắn cờ drift_unrepairable, không đặt lệnh."""
    perp = FakeExchange()
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=FakeExchange())
    _seed_carry(executor, qty=0.1)
    executor.state.carry_positions[PERP_SYMBOL]["spot_qty"] = 0.1004

    items = executor.audit_delta_drift({PERP_SYMBOL: _price_frame(60000.0)})

    assert items == [], "dust duoi 1 lo khong duoc sinh lenh sua"
    pos = executor.state.carry_positions[PERP_SYMBOL]
    assert pos.get("drift_unrepairable") is True
    assert executor.state.drift_unrepairable_symbols.get(PERP_SYMBOL) is not None


def test_r2_2_drift_growing_above_one_lot_self_heals_flag(tmp_path: Path) -> None:
    """Lệch tăng lên 0.003 BTC -> tự gỡ cờ và sinh lệnh sửa >= 1 lô."""
    perp = FakeExchange()
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=FakeExchange())
    _seed_carry(executor, qty=0.1)
    executor.state.carry_positions[PERP_SYMBOL]["spot_qty"] = 0.1004
    executor.audit_delta_drift({PERP_SYMBOL: _price_frame(60000.0)})
    assert executor.state.carry_positions[PERP_SYMBOL].get("drift_unrepairable") is True

    executor.state.carry_positions[PERP_SYMBOL]["spot_qty"] = 0.103
    items = executor.audit_delta_drift({PERP_SYMBOL: _price_frame(60000.0)})

    assert len(items) == 1
    assert items[0]["repair_qty"] >= STEP_BTC
    assert "drift_unrepairable" not in executor.state.carry_positions[PERP_SYMBOL]
    assert PERP_SYMBOL not in executor.state.drift_unrepairable_symbols


def test_r2_2_zero_drift_clears_flag(tmp_path: Path) -> None:
    """Lệch về 0 -> cờ bị gỡ hoàn toàn."""
    executor = _make_executor(tmp_path, mode="paper", perp=FakeExchange(), spot=FakeExchange())
    _seed_carry(executor, qty=0.1)
    executor.state.carry_positions[PERP_SYMBOL]["drift_unrepairable"] = True
    executor.state.drift_unrepairable_symbols[PERP_SYMBOL] = 0.0004

    items = executor.audit_delta_drift({PERP_SYMBOL: _price_frame(60000.0)})

    assert items == []
    assert "drift_unrepairable" not in executor.state.carry_positions[PERP_SYMBOL]
    assert PERP_SYMBOL not in executor.state.drift_unrepairable_symbols


def test_r2_2_repair_uses_repair_qty_multiple_of_step(tmp_path: Path) -> None:
    """Lệnh sửa lệch dùng đúng ``repair_qty`` bội số của lot step."""
    executor = _make_executor(tmp_path, mode="paper", perp=FakeExchange(), spot=FakeExchange())
    _seed_carry(executor, qty=0.1)
    executor.state.carry_positions[PERP_SYMBOL]["spot_qty"] = 0.1035

    items = executor.audit_delta_drift({PERP_SYMBOL: _price_frame(60000.0)})
    assert len(items) == 1
    repair_qty = items[0]["repair_qty"]
    units = round(repair_qty / STEP_BTC)
    assert abs(repair_qty - units * STEP_BTC) < 1e-12


# ---------------------------------------------------------------------------
# R2-3: Active capital downside clamp
# ---------------------------------------------------------------------------


def _sim_kwargs(initial_capital: float, losses: bool) -> dict[str, Any]:
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    symbols = ["AAA", "BBB"]
    weights = pd.DataFrame(0.5, index=idx, columns=symbols)
    close = pd.DataFrame(100.0, index=idx, columns=symbols)
    funding = pd.DataFrame(0.0, index=idx, columns=symbols)
    carry_returns = np.full(len(idx), -0.02) if losses else np.full(len(idx), 0.005)
    return {
        "raw_weights": weights,
        "close_matrix": close,
        "funding_matrix": funding,
        "target_leverage": np.ones(len(idx)),
        "carry_returns": carry_returns,
        "initial_capital": float(initial_capital),
        "rebalance_hours": 1,
        "one_way_cost": 0.0,
        "quantize": False,
        "min_weight_change": 0.0,
        "dd_cut": 1.0,
        "dd_off": 1.0,
        "enable_dd_cushion": False,
    }


def test_r2_3_fixed_base_clamps_capital_when_losing() -> None:
    """Đang lỗ: vốn hoạt động phải co theo equity, không giữ 3000 tuyệt đối."""
    kwargs = _sim_kwargs(3000.0, losses=True)
    out_fixed = _simulate_barbell_equity(reinvest_profits=False, **kwargs)
    out_compound = _simulate_barbell_equity(reinvest_profits=True, **kwargs)

    # Nếu neo cứng 3.000 (không clamp), mỗi bar lỗ 2% của 3.000 = 60 -> tổng -360.
    # Clamp dùng min(3.000, equity) nên lỗ co lại theo vốn thực còn.
    net_fixed = float(out_fixed["equity"][-1]) - 3000.0
    naive_unclamped = -0.02 * 3000.0 * 6
    assert net_fixed > naive_unclamped, (
        f"lo neo tinh {net_fixed:.2f} khong duoc co lai so voi neo cung {naive_unclamped:.2f}"
    )
    assert float(out_fixed["equity"][-1]) > 0.0


def test_r2_3_fixed_base_identical_to_initial_when_profitable() -> None:
    """Đang lãi: neo tĩnh giữ đúng 3000 nên lãi tuyến tính theo vốn gốc."""
    kwargs = _sim_kwargs(3000.0, losses=False)
    out_fixed = _simulate_barbell_equity(reinvest_profits=False, **kwargs)
    out_compound = _simulate_barbell_equity(reinvest_profits=True, **kwargs)

    net_fixed = float(out_fixed["equity"][-1]) - 3000.0
    net_compound = float(out_compound["equity"][-1]) - 3000.0
    # Lãi: compounding >= tĩnh, và tĩnh không được vượt quá lãi của compounding.
    assert net_fixed > 0.0
    assert net_fixed <= net_compound + 1e-9


# ---------------------------------------------------------------------------
# R2-4: Crossed / invalid orderbook guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["paper", "testnet", "live"])
def test_r2_4_crossed_book_blocked_in_all_modes(tmp_path: Path, mode: str) -> None:
    """ask < bid bị chặn ở mọi chế độ, không đặt lệnh."""
    perp = FakeExchange(bids=[[100.5, 10]], asks=[[100.0, 10]])
    executor = _make_executor(
        tmp_path, mode=mode, perp=perp, spot=FakeExchange(bids=[[100.5, 10]], asks=[[100.0, 10]])
    )

    res = executor.place_maker_leg1(PERP_SYMBOL, "buy", 0.2, 100.2)

    assert res.get("status") == "skipped"
    assert res.get("reason") == "SPREAD_CROSSED_BOOK"
    assert perp.orders == []


@pytest.mark.parametrize("bids,asks", [([[0.0, 10]], [[100.0, 10]]), ([[99.0, 10]], [[0.0, 10]])])
def test_r2_4_nonpositive_price_blocked_in_paper(
    tmp_path: Path, bids: Sequence[Sequence[float]], asks: Sequence[Sequence[float]]
) -> None:
    """bid/ask <= 0 là dữ liệu lỗi -> chặn ngay cả ở paper."""
    perp = FakeExchange(bids=bids, asks=asks)
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=FakeExchange())

    res = executor.place_maker_leg1(PERP_SYMBOL, "buy", 0.2, 100.2)

    assert res.get("status") == "skipped"
    assert res.get("reason") == "SPREAD_INVALID_BOOK"
    assert perp.orders == []


def test_r2_4_valid_book_still_passes_in_paper(tmp_path: Path) -> None:
    """Book hợp lệ 5 bps -> lệnh vẫn được đặt bình thường (không chặn oan)."""
    perp = FakeExchange(bids=[[99.99, 10]], asks=[[100.01, 10]])
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=FakeExchange())

    res = executor.place_maker_leg1(PERP_SYMBOL, "buy", 0.2, 100.0)

    assert res.get("status") != "skipped"
    assert len(perp.orders) == 1


# ---------------------------------------------------------------------------
# State store: cờ an toàn bền vững qua restart
# ---------------------------------------------------------------------------


def test_kill_switch_pending_close_persisted_and_reset_clears(tmp_path: Path) -> None:
    """pending_close được lưu xuống file; --reset-state xóa sạch cả 3 cờ."""
    state_path = tmp_path / "s.json"
    store = StateStore(state_path)
    executor = _make_executor(tmp_path, mode="paper", perp=FakeExchange(), spot=FakeExchange())
    executor.state.kill_switch_triggered = True
    executor.state.kill_switch_pending_close = True
    executor.state.drift_unrepairable_symbols = {PERP_SYMBOL: 0.0004}
    executor.state_store.save(executor.state)

    reloaded = StateStore(state_path).load(default_equity=3000.0)
    assert reloaded.kill_switch_triggered is True
    assert reloaded.kill_switch_pending_close is True
    assert reloaded.drift_unrepairable_symbols.get(PERP_SYMBOL) == pytest.approx(0.0004)

    after_reset = StateStore(state_path).load(default_equity=3000.0, reset=True)
    assert after_reset.kill_switch_triggered is False
    assert after_reset.kill_switch_pending_close is False
    assert after_reset.drift_unrepairable_symbols == {}
