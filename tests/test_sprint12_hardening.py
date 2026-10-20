"""Unit test Sprint 12 cho 3 lớp bảo vệ vận hành trong ``src/carry_live_executor.py``.

Phạm vi
-------
- **PR-3 (Spread guard)**: ``place_maker_leg1`` từ chối đặt Chân 1 khi spread
  vượt ``max_allowed_spread_bps`` (mặc định 10,0 bps) và vẫn đặt lệnh khi spread
  hẹp; ``check_orderbook_spread`` trả đúng ``(spread_bps, mid)`` và fail-safe
  ``(0.0, 0.0)`` khi sàn không hỗ trợ order book.
- **PR-4 (Delta drift repair)**: ``audit_delta_drift`` chỉ sinh hành động khi
  ``|spot_qty - qty| * price >= min_notional_usd``; ``repair_delta_drift`` lượng
  hóa theo lot-size và đưa độ lệch về dưới ngưỡng.
- **PR-5 (Drawdown kill-switch)**: ``check_kill_switch`` bật khi drawdown vượt
  ngưỡng, hủy lệnh treo + đóng vị thế + xóa overlay, giữ cờ chặn các chu kỳ sau
  và cập nhật ``peak_equity``; ngưỡng cấu hình được.

Nguyên tắc offline
------------------
Không gọi mạng, không ``sleep`` thật, mọi file nằm trong ``tmp_path``. Sàn là
``FakeExchange`` thuần Python ghi lại mọi lời gọi ``create_order``/``cancel_order``.

Ghi chú hành vi thực tế của source (đã đối chiếu khi viết test)
---------------------------------------------------------------
- Nhánh **không skip** của ``place_maker_leg1`` trả dict **không có** key
  ``"status"`` (chỉ có ``"leg1_status"``). Vì vậy test dùng ``res.get("status")``
  để khẳng định "không bị skip" thay vì indexing trực tiếp.
- ``repair_delta_drift`` không sửa ``spot_qty`` mà **tăng/giảm ``qty`` (perp)**
  để khớp spot. Với ``drift_qty > 0`` (spot nhiều hơn perp) nó tăng ``qty`` lên
  ``spot_qty``, nên audit lại cho drift ≈ 0.
- Lệnh nhỏ hơn min-notional (10 USD) bị ``quantize_leg`` trả ``qty == 0``; do đó
  kịch bản spread hẹp dùng ``target_qty=0.2`` BTC (~20 USD) để thực sự chạm
  ``create_order``.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

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
    """DataFrame 1 dòng có ``close``/``spot_close`` để audit lấy giá hiện hành."""
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"close": float(price), "spot_close": float(price)},
        index=idx,
    ).astype("float64")


class FakeExchange:
    """Sàn giả ghi nhận lệnh và hủy lệnh, order book tĩnh do test điều khiển."""

    def __init__(
        self,
        bids: Sequence[Sequence[float]] | None = None,
        asks: Sequence[Sequence[float]] | None = None,
        with_order_book: bool = True,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self.bids = [list(level) for level in (bids or [[99.9, 10]])]
        self.asks = [list(level) for level in (asks or [[100.1, 10]])]
        if with_order_book:
            # Gắn method động để có thể mô phỏng sàn thiếu fetch_order_book.
            self.fetch_order_book = self._fetch_order_book  # type: ignore[assignment]

    def _fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """Trả order book tĩnh: mặc định best bid 99.9 / best ask 100.1 (20 bps)."""
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
        """Ghi nhận lệnh; limit trả về open, market trả về closed đã khớp đủ."""
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
        if type == "limit":
            return {"id": "leg1-1", "status": "open", "filled": 0.0}
        return {"id": f"m{len(self.orders)}", "status": "closed", "filled": float(amount)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Ghi nhận lệnh hủy để test kiểm tra không bỏ quên lệnh treo."""
        self.cancelled.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}


def _make_executor(
    tmp_path: Path,
    *,
    mode: str = "paper",
    capital: float = 3000.0,
    perp: Any = None,
    spot: Any = None,
    max_allowed_spread_bps: float = 10.0,
    max_live_drawdown_pct: float = 0.06,
) -> CarryLiveExecutor:
    """Tạo executor offline với state store nằm trong ``tmp_path``."""
    state_path = tmp_path / "s.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=tmp_path / "raw",
        state_file_path=state_path,
    )
    exchange_map = {"binance": perp, "binance_spot": spot}
    return CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(),
        mode=mode,  # type: ignore[arg-type]
        symbols=[PERP_SYMBOL],
        capital=capital,
        perp_exchange=perp,
        spot_exchange=spot,
        exchange_map=exchange_map,
        state_store=StateStore(state_path),
        raw_data_dir=tmp_path / "raw",
        max_allowed_spread_bps=max_allowed_spread_bps,
        max_live_drawdown_pct=max_live_drawdown_pct,
    )


def _assert_step_multiple(qty: float, step: float, abs_tol: float = 1e-12) -> None:
    """Khẳng định ``qty`` là bội số nguyên của ``step`` với sai số tuyệt đối nhỏ."""
    units = round(abs(float(qty)) / float(step))
    assert abs(abs(float(qty)) - units * float(step)) < abs_tol, (
        f"qty={qty!r} khong phai boi so cua step={step!r}"
    )


# ---------------------------------------------------------------------------
# PR-3: Spread guard
# ---------------------------------------------------------------------------


def test_pr3_wide_spread_skips_leg1_without_create_order(tmp_path: Path) -> None:
    """Book 20 bps > ngưỡng 10 bps -> skipped/SPREAD_TOO_WIDE, không đặt lệnh."""
    perp = FakeExchange(bids=[[99.9, 10]], asks=[[100.1, 10]])
    spot = FakeExchange()
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)

    res = executor.place_maker_leg1(PERP_SYMBOL, "sell", 0.002, 60000.0)

    assert res["status"] == "skipped"
    assert res["reason"] == "SPREAD_TOO_WIDE"
    assert res["spread_bps"] > 10.0
    assert len(perp.orders) == 0
    # Không có lệnh nào được gửi sang chân spot.
    assert len(spot.orders) == 0


def test_pr3_narrow_spread_still_places_leg1(tmp_path: Path) -> None:
    """Book 0,2 bps <= ngưỡng 10 bps -> không skip và create_order được gọi."""
    perp = FakeExchange(bids=[[99.999, 10]], asks=[[100.001, 10]])
    spot = FakeExchange()
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)

    res = executor.place_maker_leg1(PERP_SYMBOL, "sell", 0.2, 100.0)

    # Nhánh không skip không có key "status" (xem ghi chú đầu file).
    assert res.get("status") != "skipped"
    assert len(perp.orders) >= 1
    assert perp.orders[0]["type"] == "limit"
    assert perp.orders[0]["side"] == "sell"
    assert res["limit_price"] == pytest.approx(100.001)


def test_pr3_check_orderbook_spread_values_and_fail_safe(tmp_path: Path) -> None:
    """Book 20 bps -> (20,0 ± 0,5; 100,0 ± 0,01); sàn thiếu method -> (0,0; 0,0)."""
    perp = FakeExchange(bids=[[99.9, 10]], asks=[[100.1, 10]])
    executor = _make_executor(tmp_path, mode="paper", perp=perp, spot=FakeExchange())

    spread_bps, mid = executor.check_orderbook_spread(PERP_SYMBOL)

    assert spread_bps == pytest.approx(20.0, abs=0.5)
    assert mid == pytest.approx(100.0, abs=0.01)

    # Sàn không có fetch_order_book -> fail-safe, không raise.
    bare = FakeExchange(with_order_book=False)
    executor_bare = _make_executor(tmp_path / "bare", mode="paper", perp=bare, spot=bare)
    assert not hasattr(bare, "fetch_order_book")
    assert executor_bare.check_orderbook_spread(PERP_SYMBOL) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# PR-4: Delta drift repair
# ---------------------------------------------------------------------------


def test_pr4_small_drift_not_repaired(tmp_path: Path) -> None:
    """Lệch 0,001 BTC * 100 = 0,1 USD < 10 USD -> audit rỗng."""
    executor = _make_executor(tmp_path)
    executor.state.carry_positions = {
        PERP_SYMBOL: {"qty": 1.0, "spot_qty": 1.001, "entry_spot_price": 100.0}
    }

    items = executor.audit_delta_drift({PERP_SYMBOL: _price_frame(100.0)})

    assert items == []


def test_pr4_large_drift_is_repaired_below_threshold(tmp_path: Path) -> None:
    """Lệch 0,30 BTC * 100 = 30 USD >= 10 -> sinh REPAIR_DELTA_DRIFT và sửa về < 10."""
    executor = _make_executor(tmp_path)
    executor.state.carry_positions = {
        PERP_SYMBOL: {"qty": 1.0, "spot_qty": 1.30, "entry_spot_price": 100.0}
    }
    aligned = {PERP_SYMBOL: _price_frame(100.0)}

    items = executor.audit_delta_drift(aligned)

    assert len(items) == 1
    item = items[0]
    assert item["symbol"] == PERP_SYMBOL
    assert item["action"] == "REPAIR_DELTA_DRIFT"
    assert item["drift_usd"] == pytest.approx(30.0)

    res = executor.repair_delta_drift(item)

    assert res["status"] == "repaired"
    assert res["qty"] > 0.0
    # Source cập nhật qty (perp) lên bằng spot_qty, không sửa spot_qty.
    assert executor.state.carry_positions[PERP_SYMBOL]["qty"] == pytest.approx(1.30)

    remaining = executor.audit_delta_drift(aligned)
    assert remaining == []
    assert executor.state.carry_positions[PERP_SYMBOL]["qty"] * 100.0 == pytest.approx(130.0)


def test_pr4_repair_quantizes_odd_drift_to_lot_step(tmp_path: Path) -> None:
    """Drift lẻ 1,234567 BTC -> qty sửa là bội số step_size 0,001 (sai số < 1e-12)."""
    executor = _make_executor(tmp_path)
    item = {
        "symbol": PERP_SYMBOL,
        "action": "REPAIR_DELTA_DRIFT",
        "drift_qty": 1.234567,
        "drift_usd": 123.4567,
        "price": 100.0,
        "side": "sell_spot",
    }

    res = executor.repair_delta_drift(item)

    assert res["status"] == "repaired"
    assert res["qty"] > 0.0
    _assert_step_multiple(res["qty"], STEP_BTC)
    assert res["qty"] == pytest.approx(1.234, abs=1e-12)
    assert res["qty"] <= abs(item["drift_qty"])


# ---------------------------------------------------------------------------
# PR-5: Drawdown kill-switch
# ---------------------------------------------------------------------------


def _kill_switch_state(executor: CarryLiveExecutor) -> None:
    """Nạp 1 vị thế carry có lệnh treo + 1 overlay để kiểm tra dọn dẹp."""
    executor.state.peak_equity = 3000.0
    executor.state.kill_switch_triggered = False
    executor.kill_switch_triggered = False
    executor.state.carry_positions = {
        PERP_SYMBOL: {
            "symbol": PERP_SYMBOL,
            "spot_symbol": SPOT_SYMBOL,
            "qty": 1.0,
            "spot_qty": 1.0,
            "leg1_order_id": "leg1-1",
            "entry_spot_price": 100.0,
        }
    }
    executor.state.overlay_positions = {
        "ETH/USDT:USDT": {"qty": 1.0, "side": "long", "entry_price": 100.0}
    }


def test_pr5_triggers_at_drawdown_above_threshold(tmp_path: Path) -> None:
    """Equity 2800 / peak 3000 (6,67 % > 6 %) -> bật, hủy lệnh, đóng vị thế, xóa overlay."""
    perp = FakeExchange()
    spot = FakeExchange()
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _kill_switch_state(executor)

    triggered = executor.check_kill_switch(2800.0)

    assert triggered is True
    assert executor.state.kill_switch_triggered is True
    assert executor.kill_switch_triggered is True

    # cancel_maker_leg1 -> exchange.cancel_order
    assert len(perp.cancelled) >= 1
    assert perp.cancelled[0]["id"] == "leg1-1"

    # Đóng vị thế: spot market sell + perp market buy reduceOnly.
    market_orders = [o for o in spot.orders if o["type"] == "market"]
    assert len(market_orders) >= 1
    assert market_orders[0]["side"] == "sell"
    perp_market_orders = [o for o in perp.orders if o["type"] == "market"]
    assert len(perp_market_orders) >= 1
    assert perp_market_orders[0]["side"] == "buy"
    assert perp_market_orders[0]["params"].get("reduceOnly") is True

    assert executor.state.overlay_positions == {}
    # Kill-switch phai dong va go bo vi the carry: hoac xoa key, hoac qty ve 0
    # (ban sua CRITICAL chon xoa key de khong bi tinh la con vi the mo).
    if PERP_SYMBOL in executor.state.carry_positions:
        assert executor.state.carry_positions[PERP_SYMBOL]["qty"] == 0.0
        assert executor.state.carry_positions[PERP_SYMBOL]["closed_by"] == "KILL_SWITCH"
    else:
        assert PERP_SYMBOL not in executor.state.carry_positions


def test_pr5_does_not_trigger_at_three_percent(tmp_path: Path) -> None:
    """Equity 2910 / peak 3000 (3 % < 6 %) -> không bật, không hủy lệnh."""
    perp = FakeExchange()
    spot = FakeExchange()
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _kill_switch_state(executor)

    assert executor.check_kill_switch(2910.0) is False
    assert executor.state.kill_switch_triggered is False
    assert executor.kill_switch_triggered is False
    assert perp.cancelled == []
    assert [o for o in spot.orders if o["type"] == "market"] == []
    assert executor.state.carry_positions[PERP_SYMBOL]["qty"] == pytest.approx(1.0)


def test_pr5_remains_triggered_when_equity_recovers(tmp_path: Path) -> None:
    """Đã kích hoạt rồi thì equity hồi phục 3000 vẫn trả True và giữ cờ chặn."""
    perp = FakeExchange()
    spot = FakeExchange()
    executor = _make_executor(tmp_path, mode="live", perp=perp, spot=spot)
    _kill_switch_state(executor)

    assert executor.check_kill_switch(2800.0) is True

    # Chu kỳ sau: equity hồi phục hoàn toàn, kill-switch vẫn chặn.
    assert executor.check_kill_switch(3000.0) is True
    assert executor.state.kill_switch_triggered is True
    assert executor.kill_switch_triggered is True


def test_pr5_updates_peak_equity_on_new_high(tmp_path: Path) -> None:
    """Equity 3200 > peak 3000 -> không bật và peak_equity được nâng lên 3200."""
    executor = _make_executor(tmp_path)
    executor.state.peak_equity = 3000.0

    assert executor.check_kill_switch(3200.0) is False
    assert executor.state.peak_equity == 3200.0


def test_pr5_threshold_is_configurable(tmp_path: Path) -> None:
    """Ngưỡng 10 %: equity 2800 / peak 3000 (6,67 %) không kích hoạt."""
    executor = _make_executor(tmp_path, max_live_drawdown_pct=0.10)
    _kill_switch_state(executor)

    assert executor.check_kill_switch(2800.0) is False
    assert executor.state.kill_switch_triggered is False
