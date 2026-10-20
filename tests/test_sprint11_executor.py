"""Unit test Sprint 11 cho thực thi Maker->Taker và tái cân bằng Momentum Overlay.

Phạm vi
-------
- ``CarryLiveExecutor.execute_maker_then_taker_pair``: khớp đủ, khớp một phần,
  TTL hết hạn (hủy lệnh treo, tuyệt đối không hedge) và lượng hóa lot-size của
  chân hedge thứ hai.
- ``CarryLiveExecutor.execute_overlay_rebalance``: mở mới long/short, bỏ qua
  delta quá nhỏ, đóng toàn bộ vị thế, và bỏ qua symbol thiếu giá thay vì crash.
- Bốn trường state mới ``carry_orders``, ``overlay_orders``, ``overlay_positions``,
  ``executor_functional``: tương thích ngược với JSON cũ và round-trip save/load.

Nguyên tắc offline
------------------
Không gọi mạng, không ngủ thật: mọi lời gọi ``sleep_fn`` được thay bằng lambda
ghi nhận thời gian giả, sàn là ``FakeExchange`` thuần Python. Mọi file chỉ ghi
trong ``tmp_path``; mỗi test chạy dưới 3 giây.

Ghi chú nhãn trạng thái (theo code hiện tại của ``execute_maker_then_taker_pair``)
-------------------------------------------------------------------------------
- ``final_status`` lấy từ ``leg1_status`` của Chân 1 (FILLED / PARTIAL_FILLED /
  CANCELED / PENDING_MAKER ...), không phải nhãn tổng hợp.
- Nhãn tổng hợp ``filled`` / ``partial`` / ``unfilled`` chỉ nằm trong record của
  ``state.carry_orders``. Các test dưới đây assert theo đúng thực tế này.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import pytest

from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.cross_exchange_carry import LOT_STEP_SIZES
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import BotState, StateStore

PERP_SYMBOL = "BTC/USDT:USDT"
SPOT_SYMBOL = "BTC/USDT"
STEP_BTC = LOT_STEP_SIZES[PERP_SYMBOL]
STEP_ETH = LOT_STEP_SIZES["ETH/USDT:USDT"]


# ---------------------------------------------------------------------------
# Helpers dùng chung
# ---------------------------------------------------------------------------


def _price_frame(price: float, n_rows: int = 3) -> pd.DataFrame:
    """DataFrame tối giản có ``perp_close``/``close`` để overlay lấy giá cuối."""
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"close": float(price), "perp_close": float(price)},
        index=idx,
    ).astype("float64")


def _make_paper_executor(
    tmp_path: Path,
    symbols: Sequence[str] | None = None,
) -> CarryLiveExecutor:
    """Executor chế độ paper, state file và raw dir đều nằm trong ``tmp_path``."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "state.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=raw_dir,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(),
        symbols=list(symbols) if symbols is not None else [PERP_SYMBOL],
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
    )


def _make_live_executor(
    perp_fake: "FakeExchange",
    spot_fake: "FakeExchange",
    state_path: Path,
) -> CarryLiveExecutor:
    """Executor chế độ live nhưng sàn là FakeExchange, không chạm mạng."""
    return CarryLiveExecutor(
        mode="live",
        execution_style="maker_taker_hedge",
        capital=3000.0,
        exchange_map={"binance": perp_fake, "binance_spot": spot_fake},
        perp_exchange=perp_fake,
        spot_exchange=spot_fake,
        state_store=StateStore(state_path),
    )


def _assert_step_multiple(qty: float, step: float) -> None:
    """Khẳng định ``qty`` là bội số nguyên của ``step_size`` (sai số số thực nhỏ)."""
    units = abs(float(qty)) / float(step)
    assert math.isclose(units, round(units), rel_tol=0.0, abs_tol=1e-6), (
        f"qty={qty!r} khong phai boi so cua step_size={step!r}"
    )


def _market_orders(exchange: "FakeExchange") -> list[dict[str, Any]]:
    """Lọc các lệnh market đã ghi nhận trên FakeExchange."""
    return [order for order in exchange.orders if order["type"] == "market"]


class FakeExchange:
    """Sàn giả thuần Python cho máy trạng thái Chân 1 / Chân 2.

    Ghi lại mọi lệnh vào ``orders`` dưới dạng dict để test kiểm tra
    ``type``/``side``/``amount``/``params``. ``fetch_order`` trả đúng dict
    ``order_status`` do test điều khiển, cho phép mô phỏng open / partial /
    closed / canceled / rejected mà không cần đồng hồ hay mạng thật.
    """

    def __init__(
        self,
        order_status: dict[str, Any] | None = None,
        bids: Sequence[Sequence[float]] | None = None,
        asks: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.order_status: dict[str, Any] = (
            dict(order_status)
            if order_status is not None
            else {"id": "leg1-1", "status": "open", "filled": 0.0}
        )
        self.bids = [list(level) for level in (bids or [[99.9, 10]])]
        self.asks = [list(level) for level in (asks or [[100.1, 10]])]
        self.cancelled: list[dict[str, Any]] = []

    def milliseconds(self) -> int:
        """Mốc thời gian cố định, không phụ thuộc đồng hồ thật."""
        return 1_700_000_000_000

    def fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """Order book tĩnh: best bid 99.9, best ask 100.1."""
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
        """Ghi nhận lệnh; limit trả về open, market trả về closed đã khớp đủ amount."""
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

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Trả dữ liệu trạng thái do test điều khiển qua ``order_status``."""
        return dict(self.order_status)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Ghi nhận lệnh hủy để test chống bỏ quên lệnh treo (orphaned order)."""
        self.cancelled.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}


def _sleep_recorder() -> tuple[Callable[[float], None], list[float]]:
    """Trả về (sleep_fn giả, danh sách thời gian đã 'ngủ') — không sleep thật."""
    calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        calls.append(float(seconds))

    return _fake_sleep, calls


# ---------------------------------------------------------------------------
# 1. execute_maker_then_taker_pair — khớp đủ
# ---------------------------------------------------------------------------


def test_maker_then_taker_pair_full_fill_marks_functional(tmp_path: Path) -> None:
    """Chân 1 khớp đủ 1.0 -> hedge đúng 1.0, ghi 1 record và bật cờ functional.

    Đồng thời khóa việc polling có gọi ``sleep_fn`` (được truyền giả) ít nhất một
    lần, và state trên executor/state đều phản ánh ``executor_functional=True``.
    """
    perp_fake = FakeExchange(
        order_status={"id": "leg1-1", "status": "closed", "filled": 1.0}
    )
    spot_fake = FakeExchange()
    executor = _make_live_executor(perp_fake, spot_fake, tmp_path / "state.json")
    sleep_fn, sleep_calls = _sleep_recorder()

    result = executor.execute_maker_then_taker_pair(
        PERP_SYMBOL,
        "sell",
        1.0,
        100.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol=SPOT_SYMBOL,
        maker_ttl_sec=5.0,
        poll_interval_sec=1.0,
        sleep_fn=sleep_fn,
    )

    assert result["order_id"] == "leg1-1"
    assert result["filled_qty"] == pytest.approx(1.0)
    assert result["hedged_qty"] == pytest.approx(result["filled_qty"])
    assert result["leg2_hedged_qty"] == pytest.approx(result["filled_qty"])
    assert result["functional"] is True
    assert result["cancelled"] is False

    assert executor.executor_functional is True
    assert executor.state.executor_functional is True

    assert len(executor.state.carry_orders) == 1
    record = executor.state.carry_orders[0]
    assert record["symbol"] == PERP_SYMBOL
    assert record["status"] == "filled"
    assert record["filled_qty"] == pytest.approx(1.0)
    assert record["hedged_qty"] == pytest.approx(record["filled_qty"])

    market_orders = _market_orders(spot_fake)
    assert len(market_orders) == 1
    assert market_orders[0]["symbol"] == SPOT_SYMBOL
    assert market_orders[0]["side"] == "buy"
    assert market_orders[0]["amount"] == pytest.approx(1.0)

    assert len(sleep_calls) >= 1
    assert all(seconds == pytest.approx(1.0) for seconds in sleep_calls)


# ---------------------------------------------------------------------------
# 2. execute_maker_then_taker_pair — khớp một phần
# ---------------------------------------------------------------------------


def test_maker_then_taker_pair_partial_fill_hedges_only_filled_qty(tmp_path: Path) -> None:
    """Chân 1 chỉ khớp 40% -> hedge đúng 40%, không over-hedge lên target.

    Nhãn tổng hợp ``partial`` nằm trong record ``state.carry_orders`` (xem ghi chú
    đầu file: ``final_status`` là ``leg1_status`` của Chân 1 nên là
    ``PARTIAL_FILLED``).
    """
    target_qty = 1.0
    filled_partial = 0.4 * target_qty
    perp_fake = FakeExchange(
        order_status={"id": "leg1-1", "status": "open", "filled": filled_partial}
    )
    spot_fake = FakeExchange()
    executor = _make_live_executor(perp_fake, spot_fake, tmp_path / "state.json")
    sleep_fn, sleep_calls = _sleep_recorder()

    result = executor.execute_maker_then_taker_pair(
        PERP_SYMBOL,
        "sell",
        target_qty,
        100.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol=SPOT_SYMBOL,
        maker_ttl_sec=2.0,
        poll_interval_sec=1.0,
        sleep_fn=sleep_fn,
    )

    assert result["filled_qty"] == pytest.approx(filled_partial)
    assert result["hedged_qty"] == pytest.approx(filled_partial)
    assert result["hedged_qty"] <= target_qty

    market_orders = _market_orders(spot_fake)
    assert len(market_orders) == 1
    assert sum(order["amount"] for order in market_orders) == pytest.approx(filled_partial)
    assert abs(market_orders[0]["amount"] - target_qty) > 1e-9

    assert result["functional"] is True
    assert result["final_status"] == "PARTIAL_FILLED"

    assert len(executor.state.carry_orders) == 1
    assert executor.state.carry_orders[0]["status"] == "partial"
    assert executor.state.carry_orders[0]["filled_qty"] == pytest.approx(filled_partial)

    assert len(sleep_calls) >= 1
    assert _market_orders(perp_fake) == []


# ---------------------------------------------------------------------------
# 3. execute_maker_then_taker_pair — TTL hết hạn, không khớp
# ---------------------------------------------------------------------------


def test_maker_then_taker_pair_ttl_expiry_cancels_without_hedging(tmp_path: Path) -> None:
    """TTL 3s, lệnh luôn open/filled=0 -> hủy lệnh, không có market Chân 2, unfilled."""
    perp_fake = FakeExchange(
        order_status={"id": "leg1-1", "status": "open", "filled": 0.0}
    )
    spot_fake = FakeExchange()
    executor = _make_live_executor(perp_fake, spot_fake, tmp_path / "state.json")
    sleep_fn, sleep_calls = _sleep_recorder()

    result = executor.execute_maker_then_taker_pair(
        PERP_SYMBOL,
        "sell",
        1.0,
        100.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol=SPOT_SYMBOL,
        maker_ttl_sec=3.0,
        poll_interval_sec=1.0,
        sleep_fn=sleep_fn,
    )

    # 3 vòng polling giả, không sleep thật.
    assert sleep_calls == pytest.approx([1.0, 1.0, 1.0])
    assert len(perp_fake.cancelled) >= 1
    assert perp_fake.cancelled[0]["id"] == "leg1-1"

    assert _market_orders(spot_fake) == []
    assert _market_orders(perp_fake) == []

    assert result["cancelled"] is True
    assert result["filled_qty"] == pytest.approx(0.0)
    assert result["hedged_qty"] == pytest.approx(0.0)
    assert result["functional"] is False
    assert executor.executor_functional is False
    assert executor.state.executor_functional is False

    assert len(executor.state.carry_orders) == 1
    assert executor.state.carry_orders[0]["status"] == "unfilled"


# ---------------------------------------------------------------------------
# 4. execute_maker_then_taker_pair — lượng hóa qty hedge
# ---------------------------------------------------------------------------


def test_maker_then_taker_pair_quantizes_hedge_qty_to_lot_step(tmp_path: Path) -> None:
    """Filled lẻ 0.0012345 BTC -> qty hedge làm tròn xuống bội số step_size 0.001."""
    odd_filled = 0.0012345
    perp_fake = FakeExchange(
        order_status={"id": "leg1-1", "status": "closed", "filled": odd_filled}
    )
    spot_fake = FakeExchange()
    executor = _make_live_executor(perp_fake, spot_fake, tmp_path / "state.json")
    sleep_fn, _ = _sleep_recorder()

    result = executor.execute_maker_then_taker_pair(
        PERP_SYMBOL,
        "sell",
        0.5,
        100.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol=SPOT_SYMBOL,
        maker_ttl_sec=2.0,
        poll_interval_sec=1.0,
        sleep_fn=sleep_fn,
    )

    market_orders = _market_orders(spot_fake)
    assert len(market_orders) == 1
    hedge_qty = market_orders[0]["amount"]

    assert STEP_BTC == pytest.approx(0.001)
    _assert_step_multiple(hedge_qty, STEP_BTC)
    assert hedge_qty == pytest.approx(0.001)
    # Không hedge vượt phần đã khớp thực tế.
    assert hedge_qty <= odd_filled + 1e-12

    assert result["hedged_qty"] == pytest.approx(hedge_qty)
    _assert_step_multiple(result["hedged_qty"], STEP_BTC)


# ---------------------------------------------------------------------------
# 5. execute_overlay_rebalance — mở mới long + short
# ---------------------------------------------------------------------------


def test_overlay_rebalance_opens_new_long_and_short_positions(tmp_path: Path) -> None:
    """2 allocation (1 long, 1 short) notional > 10 USD -> 2 lệnh, 2 vị thế đúng dấu."""
    executor = _make_paper_executor(tmp_path, symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    executor.state.overlay_positions = {}

    overlay_allocations = {
        "overlay_active": True,
        "allocations": [
            {"symbol": "BTC/USDT:USDT", "side": "long_perp", "target_qty": 0.2},
            {"symbol": "ETH/USDT:USDT", "side": "short_perp", "qty": 1.0},
        ],
    }
    aligned_universe = {
        "BTC/USDT:USDT": _price_frame(100.0),
        "ETH/USDT:USDT": _price_frame(100.0),
    }

    result = executor.execute_overlay_rebalance(
        overlay_allocations,
        aligned_universe,
        timestamp="2025-01-01T00:00:00+00:00",
    )

    assert result["n_orders"] == 2
    assert len(result["orders"]) == 2
    assert result["skipped"] == 0
    assert result["functional"] is True

    assert len(executor.state.overlay_orders) == 2
    assert executor.state.executor_functional is True

    positions = executor.state.overlay_positions
    assert set(positions) == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    assert positions["BTC/USDT:USDT"]["qty"] == pytest.approx(0.2)
    assert positions["BTC/USDT:USDT"]["side"] == "long"
    assert positions["ETH/USDT:USDT"]["qty"] == pytest.approx(-1.0)
    assert positions["ETH/USDT:USDT"]["side"] == "short"

    _assert_step_multiple(positions["BTC/USDT:USDT"]["qty"], STEP_BTC)
    _assert_step_multiple(positions["ETH/USDT:USDT"]["qty"], STEP_ETH)

    orders_by_symbol = {order["symbol"]: order for order in result["orders"]}
    assert orders_by_symbol["BTC/USDT:USDT"]["side"] == "buy"
    assert orders_by_symbol["ETH/USDT:USDT"]["side"] == "sell"
    for order in result["orders"]:
        assert order["status"] == "paper_filled"
        assert order["notional"] > 10.0
        assert order["ts"] == "2025-01-01T00:00:00+00:00"

    assert result["turnover_notional"] == pytest.approx(
        sum(order["notional"] for order in result["orders"])
    )


# ---------------------------------------------------------------------------
# 6. execute_overlay_rebalance — delta quá nhỏ
# ---------------------------------------------------------------------------


def test_overlay_rebalance_skips_delta_below_min_notional(tmp_path: Path) -> None:
    """Chênh lệch 5 USD (< 10 USD) -> skipped, không đặt lệnh, không thêm record."""
    executor = _make_paper_executor(tmp_path)
    executor.state.overlay_positions = {
        "BTC/USDT:USDT": {"qty": 0.15, "side": "long", "entry_price": 100.0}
    }
    executor.state.overlay_orders = [{"symbol": "EXISTING", "status": "paper_filled"}]

    overlay_allocations = {
        "overlay_active": True,
        "allocations": [
            {"symbol": "BTC/USDT:USDT", "side": "long_perp", "target_qty": 0.2},
        ],
    }

    result = executor.execute_overlay_rebalance(
        overlay_allocations,
        {"BTC/USDT:USDT": _price_frame(100.0)},
    )

    assert result["skipped"] >= 1
    assert result["n_orders"] == 0
    assert result["orders"] == []
    assert result["turnover_notional"] == pytest.approx(0.0)
    assert len(executor.state.overlay_orders) == 1
    assert executor.state.overlay_positions["BTC/USDT:USDT"]["qty"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 7. execute_overlay_rebalance — đóng toàn bộ
# ---------------------------------------------------------------------------


def test_overlay_rebalance_closes_all_positions_when_inactive(tmp_path: Path) -> None:
    """overlay_active=False + allocations rỗng -> 1 lệnh bán đóng vị thế, positions rỗng."""
    executor = _make_paper_executor(tmp_path)
    executor.state.overlay_positions = {
        "BTC/USDT:USDT": {"qty": 0.5, "side": "long", "entry_price": 100.0}
    }

    result = executor.execute_overlay_rebalance(
        {"allocations": [], "overlay_active": False},
        {"BTC/USDT:USDT": _price_frame(100.0)},
    )

    assert result["n_orders"] == 1
    assert result["orders"][0]["symbol"] == "BTC/USDT:USDT"
    assert result["orders"][0]["side"] == "sell"
    assert result["orders"][0]["qty"] == pytest.approx(0.5)
    assert result["orders"][0]["reason"] == "close"

    assert executor.state.overlay_positions == {}
    assert len(executor.state.overlay_orders) == 1
    assert result["functional"] is True


# ---------------------------------------------------------------------------
# 8. execute_overlay_rebalance — thiếu giá
# ---------------------------------------------------------------------------


def test_overlay_rebalance_skips_symbol_missing_price_without_raising(tmp_path: Path) -> None:
    """Symbol có allocation nhưng không có trong aligned_universe -> skip, không raise."""
    executor = _make_paper_executor(tmp_path)
    executor.state.overlay_positions = {}

    overlay_allocations = {
        "overlay_active": True,
        "allocations": [
            {"symbol": "ZZZ/USDT:USDT", "side": "long_perp", "target_qty": 1.0},
        ],
    }

    result = executor.execute_overlay_rebalance(
        overlay_allocations,
        {"BTC/USDT:USDT": _price_frame(100.0)},
    )

    assert result["skipped"] >= 1
    assert result["n_orders"] == 0
    assert result["orders"] == []
    assert executor.state.overlay_positions == {}
    assert executor.state.overlay_orders == []


# ---------------------------------------------------------------------------
# 9. State JSON tương thích ngược + round-trip 4 trường mới
# ---------------------------------------------------------------------------


def test_state_store_legacy_json_defaults_and_roundtrip_new_fields(tmp_path: Path) -> None:
    """JSON cũ thiếu 4 key mới -> mặc định an toàn; save/load lại giữ đúng giá trị."""
    state_path = tmp_path / "state.json"
    legacy_payload: dict[str, Any] = {
        "open_position": None,
        "daily_realized_pnl": 0.0,
        "daily_start_equity": 400.0,
        "current_equity": 400.0,
        "current_utc_date": "2025-01-01",
        "consecutive_losses": 0,
        "circuit_breaker_tripped": False,
        "last_processed_candle_ts": None,
        "trade_history": [],
        "carry_positions": {},
        "carry_last_reconcile_ts": None,
        "carry_sleeve": {},
        "momentum_overlay": {},
    }
    assert not any(
        key in legacy_payload
        for key in ("carry_orders", "overlay_orders", "overlay_positions", "executor_functional")
    )
    state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    store = StateStore(state_path)
    loaded = store.load(default_equity=400.0)

    assert loaded.carry_orders == []
    assert loaded.overlay_orders == []
    assert loaded.overlay_positions == {}
    assert loaded.executor_functional is False

    carry_orders = [{"symbol": PERP_SYMBOL, "status": "filled", "filled_qty": 1.0}]
    overlay_orders = [
        {"symbol": "ETH/USDT:USDT", "side": "sell", "qty": 1.0, "status": "paper_filled"}
    ]
    overlay_positions = {
        "ETH/USDT:USDT": {"qty": -1.0, "side": "short", "entry_price": 100.0, "ts": ""}
    }
    loaded.carry_orders = carry_orders
    loaded.overlay_orders = overlay_orders
    loaded.overlay_positions = overlay_positions
    loaded.executor_functional = True
    store.save(loaded)

    assert list(tmp_path.glob("*.tmp")) == []

    reloaded = store.load(default_equity=400.0)
    assert reloaded.carry_orders == carry_orders
    assert reloaded.overlay_orders == overlay_orders
    assert reloaded.overlay_positions == overlay_positions
    assert reloaded.executor_functional is True

    # Mặc định của dataclass cũng phải an toàn (không dùng mutable dùng chung).
    fresh = BotState()
    assert fresh.carry_orders == []
    assert fresh.overlay_orders == []
    assert fresh.overlay_positions == {}
    assert fresh.executor_functional is False


# ---------------------------------------------------------------------------
# 10. execute_overlay_rebalance — mặc định an toàn
# ---------------------------------------------------------------------------


def test_overlay_rebalance_empty_inputs_returns_safe_default(tmp_path: Path) -> None:
    """``execute_overlay_rebalance({}, {})`` với state rỗng -> 0 lệnh, không raise."""
    executor = _make_paper_executor(tmp_path)
    assert executor.state.overlay_positions == {}
    assert executor.state.overlay_orders == []

    result = executor.execute_overlay_rebalance({}, {})

    assert result["n_orders"] == 0
    assert result["orders"] == []
    assert result["turnover_notional"] == pytest.approx(0.0)
    assert result["functional"] is False
    assert executor.state.overlay_orders == []
    assert executor.state.overlay_positions == {}
    assert executor.executor_functional is False
