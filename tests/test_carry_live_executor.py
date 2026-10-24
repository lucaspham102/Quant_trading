"""Unit test offline cho CarryLiveExecutor, StateStore carry fields và CLI run_live.

Toàn bộ test đều tất định, không gọi mạng, không đọc ``data/raw`` thật và chỉ ghi
file trong ``tmp_path``. Dữ liệu parquet tổng hợp được tạo trực tiếp trong test để
khóa các hành vi sau:

- Tương thích ngược StateStore khi file JSON cũ chưa có trường carry.
- Ghi nguyên tử và khôi phục ``carry_positions`` / ``carry_last_reconcile_ts``.
- Kế hoạch reconcile OPEN -> HOLD -> CLOSE với cache parquet đúng quy ước tên.
- Kiểm tra an toàn ký quỹ và bảng trạng thái thuần ASCII.
- CLI ``run_live.main([... "--strategy", "dynamic_carry", "--once"])`` offline.
- Các nhánh an toàn: cache rỗng ném FileNotFoundError, universe rỗng/thiếu dữ liệu
  không crash.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

import run_live
from src.carry_live_executor import CarryLiveExecutor, ReconcileAction
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import BotState, StateStore

_START = "2024-01-01 00:00:00"
_N_HOURS = 60
_HIGH_FUNDING = 0.001
_LOW_FUNDING = -0.001


def _hourly_index(n_hours: int = _N_HOURS, start: str = _START) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex 1h UTC cố định, không phụ thuộc thời điểm chạy test."""
    idx = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    return idx.as_unit("ns")


def _ohlcv_frame(index: pd.DatetimeIndex, close: float) -> pd.DataFrame:
    """Tạo DataFrame OHLCV tổng hợp với giá đứng yên quanh ``close``."""
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame.astype("float64")


def _aligned_frame(
    symbol: str,
    funding_rate: float,
    n_hours: int = _N_HOURS,
    spot_price: float = 100.0,
    perp_price: float = 100.05,
) -> pd.DataFrame:
    """Tạo DataFrame đã đồng bộ Spot/Perp/Funding giống đầu ra align_spot_perp_funding."""
    idx = _hourly_index(n_hours)
    frame = _ohlcv_frame(idx, perp_price)
    frame["spot_close"] = float(spot_price)
    frame["perp_close"] = float(perp_price)
    frame["basis_pct"] = (float(perp_price) - float(spot_price)) / float(spot_price)

    mask_8h = idx.hour.isin([0, 8, 16]) & (idx.minute == 0)
    frame["funding_rate_event"] = 0.0
    frame.loc[mask_8h, "funding_rate_event"] = float(funding_rate)
    frame["last_known_funding_rate"] = float(funding_rate)
    frame.attrs["symbol"] = symbol
    return frame


def _write_cache_bundle(
    raw_dir: Path,
    symbol: str,
    funding_rate: float,
    n_hours: int = _N_HOURS,
    spot_price: float = 100.0,
    perp_price: float = 100.05,
) -> None:
    """Ghi bộ cache parquet đúng quy ước tên của funding_loader / run_sprint7."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    idx = _hourly_index(n_hours)
    perp = _ohlcv_frame(idx, perp_price)
    spot = _ohlcv_frame(idx, spot_price)

    clean = symbol.replace("/", "_").replace(":", "_")
    spot_clean = symbol.split(":")[0].replace("/", "").replace(":", "").upper()
    if spot_clean.endswith("USDTUSDT"):
        spot_clean = spot_clean[:-4]

    funding_idx = pd.date_range(
        _START,
        periods=max(1, n_hours // 8 + 1),
        freq="8h",
        tz="UTC",
    ).as_unit("ns")
    funding = pd.DataFrame(
        {
            "timestamp": funding_idx.asi8 // 1_000_000,
            "datetime": funding_idx,
            "funding_rate": float(funding_rate),
            "mark_price": float(spot_price),
        }
    )

    perp.to_parquet(raw_dir / f"{clean}_1h.parquet", engine="pyarrow")
    spot.to_parquet(raw_dir / f"{spot_clean}_SPOT_1h.parquet", engine="pyarrow")
    funding.to_parquet(raw_dir / f"{clean}_funding.parquet", engine="pyarrow", index=False)


def _make_executor(
    tmp_path: Path,
    raw_dir: Path | None = None,
    symbols: Sequence[str] | None = None,
    carry_config: DynamicCarryConfig | None = None,
) -> CarryLiveExecutor:
    """Khởi tạo executor trỏ state file và raw dir vào thư mục tạm."""
    resolved_raw = Path(raw_dir) if raw_dir is not None else tmp_path / "raw"
    resolved_raw.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "state.json"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=resolved_raw,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=carry_config if carry_config is not None else DynamicCarryConfig(),
        symbols=symbols if symbols is not None else ["AAA/USDT:USDT"],
        state_store=StateStore(state_path),
        raw_data_dir=resolved_raw,
        max_allowed_spread_bps=10_000.0,  # fixture book rong 20 bps: tat guard spread de test state machine
    )


# ---------------------------------------------------------------------------
# 1. State store tương thích ngược + ghi nguyên tử
# ---------------------------------------------------------------------------


def test_state_store_backward_compatibility_with_carry_positions(tmp_path: Path) -> None:
    """File state kiểu cũ vẫn load được; save/load carry_positions giữ nguyên và không sót .tmp."""
    state_path = tmp_path / "state.json"
    old_payload: dict[str, Any] = {
        "open_position": None,
        "daily_realized_pnl": 1.5,
        "daily_start_equity": 500.0,
        "current_equity": 505.0,
        "current_utc_date": "2025-01-01",
        "consecutive_losses": 2,
        "circuit_breaker_tripped": False,
        "last_processed_candle_ts": "2025-01-01T00:00:00+00:00",
        "trade_history": [],
    }
    state_path.write_text(json.dumps(old_payload), encoding="utf-8")

    store = StateStore(state_path)
    loaded_old = store.load(default_equity=400.0)

    assert loaded_old.current_equity == pytest.approx(505.0)
    assert loaded_old.carry_positions == {}
    assert loaded_old.carry_last_reconcile_ts is None

    carry_positions = {
        "AAA/USDT:USDT": {
            "symbol": "AAA/USDT:USDT",
            "spot_symbol": "AAA/USDT",
            "qty": 1.5,
            "entry_spot_price": 100.0,
            "entry_perp_price": 100.05,
            "notional_usdt": 150.0,
            "margin_allocated_usdt": 49.75,
            "periods_held": 3,
            "funding_collected_usdt": 0.45,
        },
        "BBB/USDT:USDT": {
            "symbol": "BBB/USDT:USDT",
            "spot_symbol": "BBB/USDT",
            "qty": 2.0,
            "entry_spot_price": 50.0,
            "entry_perp_price": 50.02,
            "notional_usdt": 100.0,
            "margin_allocated_usdt": 99.0,
            "periods_held": 0,
            "funding_collected_usdt": 0.0,
        },
    }
    new_state = BotState(
        current_equity=420.0,
        carry_positions=carry_positions,
        carry_last_reconcile_ts="2025-01-02T00:00:00+00:00",
    )
    store.save(new_state)

    assert state_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []

    reloaded = store.load(default_equity=400.0)
    assert reloaded.carry_positions == carry_positions
    assert reloaded.carry_last_reconcile_ts == "2025-01-02T00:00:00+00:00"


# ---------------------------------------------------------------------------
# 2. Kế hoạch reconcile OPEN -> HOLD -> CLOSE
# ---------------------------------------------------------------------------


def test_reconcile_once_open_hold_close_plan(tmp_path: Path) -> None:
    """3 chu kỳ liên tiếp phải tạo OPEN cho 2 symbol funding cao, HOLD rồi CLOSE khi điểm giảm."""
    raw_dir = tmp_path / "raw"
    symbols = ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"]
    # Sprint 18: lich su phai >= 720h (cong bao hoa warmup) moi duoc mo vi the moi.
    _write_cache_bundle(raw_dir, "AAA/USDT:USDT", _HIGH_FUNDING, n_hours=720)
    _write_cache_bundle(raw_dir, "BBB/USDT:USDT", _HIGH_FUNDING, n_hours=720)
    _write_cache_bundle(raw_dir, "CCC/USDT:USDT", _LOW_FUNDING, n_hours=720)

    carry_config = DynamicCarryConfig(
        top_k=2,
        entry_threshold=0.00010,
        exit_threshold=0.00002,
        min_hold_periods=9,
    )
    executor = _make_executor(tmp_path, raw_dir=raw_dir, symbols=symbols, carry_config=carry_config)
    expected_high = {"AAA/USDT:USDT", "BBB/USDT:USDT"}

    # Lần 1: state rỗng -> mở các symbol điểm cao nhất.
    plan_open = executor.reconcile_once()
    actions_open = {action.symbol: action.action for action in plan_open["actions"]}

    assert {
        sym for sym, act in actions_open.items() if act == "OPEN_DELTA_NEUTRAL"
    } == expected_high
    assert set(plan_open["active_positions"]) == expected_high
    assert actions_open["CCC/USDT:USDT"] == "SKIP"
    assert executor.state_store.file_path.exists()

    saved_payload = json.loads(executor.state_store.file_path.read_text(encoding="utf-8"))
    assert set(saved_payload["carry_positions"]) == expected_high

    # Lần 2: cùng dữ liệu, chưa đủ min_hold_periods -> các vị thế đang mở phải HOLD.
    plan_hold = executor.reconcile_once()
    actions_hold = {action.symbol: action.action for action in plan_hold["actions"]}
    for sym in expected_high:
        assert actions_hold[sym] == "HOLD"

    # Lần 3: điểm của vị thế đang giữ rơi dưới exit_threshold và đã đủ min_hold_periods.
    for pos in executor.state.carry_positions.values():
        pos["periods_held"] = carry_config.min_hold_periods
    low_universe = {
        sym: _aligned_frame(sym, funding_rate=_LOW_FUNDING, n_hours=720) for sym in expected_high
    }
    plan_close = executor.reconcile_once(aligned_universe=low_universe)
    actions_close = {action.symbol: action.action for action in plan_close["actions"]}

    for sym in expected_high:
        assert actions_close[sym] == "CLOSE_DELTA_NEUTRAL"
        assert sym not in plan_close["active_positions"]
    assert executor.state.carry_positions == {}


# ---------------------------------------------------------------------------
# 3. Sức khỏe ký quỹ + bảng ASCII
# ---------------------------------------------------------------------------


def test_margin_health_check_and_ascii_console_table(tmp_path: Path) -> None:
    """Ký quỹ mỏng khi perp tăng vọt là CRITICAL/REBALANCE; vị thế chuẩn là HEALTHY; bảng thuần ASCII."""
    executor = _make_executor(tmp_path)

    thin_pos = {
        "qty": 1.0,
        "entry_spot_price": 100.0,
        "entry_perp_price": 100.0,
        "margin_allocated_usdt": 5.0,
    }
    ratio_low, health_low = executor.check_margin_health(thin_pos, spot_price=100.0, perp_price=200.0)
    assert ratio_low < executor.carry_config.maintenance_margin_ratio
    assert health_low in ("REBALANCE_NEEDED", "CRITICAL")

    healthy_pos = {
        "qty": 1.0,
        "entry_spot_price": 100.0,
        "entry_perp_price": 100.0,
        "margin_allocated_usdt": 25.0,
    }
    ratio_ok, health_ok = executor.check_margin_health(healthy_pos, spot_price=100.0, perp_price=100.0)
    assert ratio_ok == pytest.approx(0.25)
    assert health_ok == "HEALTHY"

    action = ReconcileAction(
        symbol="AAA/USDT:USDT",
        action="HOLD",
        reason="HOLD_MIN_PERIODS",
        score=0.001,
        f_ewma=0.0009,
        basis_pct=0.0005,
        basis_dev=0.0001,
        target_notional_usdt=150.0,
        spot_qty=1.5,
        perp_qty=1.5,
        margin_ratio=0.33,
        margin_health="HEALTHY",
    )
    table = executor.format_status_table(
        {
            "timestamp": "2024-01-03T11:00:00+00:00",
            "equity": 400.0,
            "actions": [action],
        }
    )
    assert isinstance(table, str)
    assert all(ord(ch) < 128 for ch in table)
    assert "HOLD" in table
    assert "AAA/USDT:USDT" in table


# ---------------------------------------------------------------------------
# 4. CLI run_live.py
# ---------------------------------------------------------------------------


def test_run_live_cli_dynamic_carry_paper_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI dynamic_carry --once không gọi mạng, trả mã 0 và ghi state JSON nguyên tử."""
    universe = {
        "BTC/USDT:USDT": _aligned_frame("BTC/USDT:USDT", funding_rate=_HIGH_FUNDING)
    }

    def fake_fetch(self: CarryLiveExecutor) -> dict[str, pd.DataFrame]:
        return {sym: frame.copy() for sym, frame in universe.items()}

    monkeypatch.setattr(
        CarryLiveExecutor,
        "fetch_latest_aligned_universe",
        fake_fetch,
    )

    state_file = tmp_path / "state.json"
    exit_code = run_live.main(
        [
            "--mode",
            "paper",
            "--strategy",
            "dynamic_carry",
            "--once",
            "--state-file",
            str(state_file),
        ]
    )

    assert exit_code == 0
    assert state_file.exists()
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert "carry_positions" in payload
    assert "carry_last_reconcile_ts" in payload


def test_parse_args_strategy_default_and_choices() -> None:
    """parse_args mặc định barbell, nhận dynamic_carry và từ chối giá trị lạ."""
    assert run_live.parse_args([]).strategy == "barbell"
    assert run_live.parse_args(["--strategy", "dynamic_carry"]).strategy == "dynamic_carry"

    with pytest.raises(SystemExit):
        run_live.parse_args(["--strategy", "not_a_strategy"])


# ---------------------------------------------------------------------------
# 5. Các nhánh an toàn
# ---------------------------------------------------------------------------


def test_fetch_latest_aligned_universe_empty_cache_raises_file_not_found(tmp_path: Path) -> None:
    """Cache rỗng và không có exchange phải ném đúng FileNotFoundError, không lỗi khác."""
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    executor = _make_executor(tmp_path, raw_dir=raw_dir, symbols=["AAA/USDT:USDT"])

    with pytest.raises(FileNotFoundError):
        executor.fetch_latest_aligned_universe()


def test_reconcile_once_empty_or_partial_universe_does_not_crash(tmp_path: Path) -> None:
    """Universe rỗng hoặc thiếu symbol phải trả plan hợp lệ, không ném ngoại lệ."""
    executor = _make_executor(
        tmp_path,
        symbols=["AAA/USDT:USDT", "BBB/USDT:USDT"],
    )

    empty_plan = executor.reconcile_once(aligned_universe={})
    assert empty_plan["actions"] == []
    assert empty_plan["active_positions"] == {}

    partial_universe = {
        "AAA/USDT:USDT": _aligned_frame("AAA/USDT:USDT", funding_rate=_HIGH_FUNDING)
    }
    partial_plan = executor.reconcile_once(aligned_universe=partial_universe)

    assert isinstance(partial_plan["actions"], list)
    partial_actions = {action.symbol: action.action for action in partial_plan["actions"]}
    assert partial_actions["BBB/USDT:USDT"] == "SKIP"


# ---------------------------------------------------------------------------
# 6. Máy trạng thái maker_then_taker hedge (GO-4)
# ---------------------------------------------------------------------------


class FakeExchange:
    """Sàn giả thuần Python, không gọi mạng, cho test máy trạng thái Chân 1/Chân 2.

    Ghi lại mọi lệnh vào ``self.orders`` dưới dạng dict để test kiểm tra
    ``type``/``side``/``amount``/``params``. Kết quả ``fetch_order`` được điều
    khiển qua ``self.order_status`` để mô phỏng open/partial/closed/canceled/rejected.
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
        return {"bids": [list(level) for level in self.bids], "asks": [list(level) for level in self.asks]}

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
        """Trả dữ liệu trạng thái do test điều khiển qua ``self.order_status``."""
        return dict(self.order_status)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Ghi nhận lệnh hủy để test chống bỏ quên lệnh treo (orphaned order)."""
        self.cancelled.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}


def test_maker_taker_hedge_state_machine_full_and_partial_fill(tmp_path: Path) -> None:
    """Chân 1 Post-Only (GTX) -> partial 0.6 hedge 0.6 -> fill đủ 1.0 hedge thêm 0.4.

    Khóa hành vi GO-4: không hedge vượt ``filled_qty``, partial fill chỉ hedge phần
    đã khớp, và state Chân 1/Chân 2 được ghi đúng vào ``carry_positions``.

    Lưu ý API thật: ``place_maker_leg1``/``verify_maker_fill_and_hedge_leg2`` chỉ
    cập nhật state trong bộ nhớ; chỉ ``reconcile_once`` mới gọi ``state_store.save``.
    Vì vậy test chủ động gọi ``save`` trước khi đọc JSON trên đĩa.
    """
    perp_fake = FakeExchange()
    spot_fake = FakeExchange()
    state_path = tmp_path / "state.json"
    executor = CarryLiveExecutor(
        mode="live",
        execution_style="maker_taker_hedge",
        capital=3000.0,
        exchange_map={"binance": perp_fake, "binance_spot": spot_fake},
        perp_exchange=perp_fake,
        spot_exchange=spot_fake,
        state_store=StateStore(state_path),
        max_allowed_spread_bps=10_000.0,  # fixture book rong 20 bps: tat guard spread de test state machine
    )

    leg1_res = executor.place_maker_leg1(
        "BTC/USDT:USDT",
        "sell",
        1.0,
        100.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol="BTC/USDT",
    )

    assert leg1_res["leg1_order_id"] == "leg1-1"
    assert leg1_res["leg1_status"] == "PENDING_MAKER"
    assert leg1_res["limit_price"] == pytest.approx(100.1)

    limit_orders = [o for o in perp_fake.orders if o["type"] == "limit"]
    assert len(limit_orders) == 1
    leg1_order = limit_orders[0]
    assert leg1_order["symbol"] == "BTC/USDT:USDT"
    assert leg1_order["side"] == "sell"
    assert leg1_order["amount"] == pytest.approx(1.0)
    assert leg1_order["params"].get("postOnly") is True
    assert leg1_order["params"].get("timeInForce") == "GTX"

    # Lần 1: Chân 1 mới khớp một phần 0.6 -> Chân 2 chỉ được hedge đúng 0.6.
    perp_fake.order_status = {"id": "leg1-1", "status": "open", "filled": 0.6}
    first = executor.verify_maker_fill_and_hedge_leg2(
        "BTC/USDT:USDT",
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol="BTC/USDT",
    )

    market_orders_first = [o for o in spot_fake.orders if o["type"] == "market"]
    assert len(market_orders_first) == 1
    assert market_orders_first[0]["symbol"] == "BTC/USDT"
    assert market_orders_first[0]["side"] == "buy"
    assert market_orders_first[0]["amount"] == pytest.approx(0.6)
    assert first["fired_market"] is True
    assert first["leg1_status"] == "PARTIAL_FILLED"
    assert first["leg1_filled_qty"] == pytest.approx(0.6)
    assert first["leg2_hedged_qty"] == pytest.approx(0.6)
    # Module gán HEDGED ngay khi delta đã hedge đủ filled_qty hiện tại
    # (MARKET_HEDGE_LEG_2 chỉ dùng khi vẫn còn delta chưa hedge).
    assert first["execution_state"] == "HEDGED"

    # Lần 2: Chân 1 khớp đủ 1.0 -> chỉ hedge phần chênh lệch 0.4, không hedge lại 1.0.
    perp_fake.order_status = {"id": "leg1-1", "status": "closed", "filled": 1.0}
    second = executor.verify_maker_fill_and_hedge_leg2(
        "BTC/USDT:USDT",
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol="BTC/USDT",
    )

    market_orders_second = [o for o in spot_fake.orders if o["type"] == "market"]
    assert len(market_orders_second) == 2
    assert market_orders_second[1]["amount"] == pytest.approx(0.4)
    assert second["fired_market"] is True
    assert second["leg1_status"] == "FILLED"
    assert second["leg1_filled_qty"] == pytest.approx(1.0)
    assert second["leg2_hedged_qty"] == pytest.approx(1.0)
    assert second["execution_state"] == "HEDGED"
    assert executor._leg_states["BTC/USDT:USDT"]["leg2_hedged_qty"] == pytest.approx(1.0)
    assert [o for o in perp_fake.orders if o["type"] == "market"] == []

    executor.state_store.save(executor.state)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    saved_pos = payload["carry_positions"]["BTC/USDT:USDT"]
    assert saved_pos["leg1_status"] == "FILLED"
    assert saved_pos["leg1_filled_qty"] == pytest.approx(1.0)
    assert saved_pos["leg2_hedged_qty"] == pytest.approx(1.0)
    assert saved_pos["execution_state"] == "HEDGED"


def test_maker_taker_hedge_leg1_rejected_or_unfilled_never_fires_market_leg2(
    tmp_path: Path,
) -> None:
    """Chân 1 canceled/rejected với filled=0 tuyệt đối không bắn Market Chân 2.

    Lặp cả hai trạng thái kết thúc ``canceled`` và ``rejected``; mỗi biến thể dùng
    state file riêng để không phụ thuộc thứ tự test.
    """
    expected_leg1_status = {"canceled": "CANCELED", "rejected": "REJECTED"}

    for terminal_status in ("canceled", "rejected"):
        perp_fake = FakeExchange(
            order_status={"id": "leg1-1", "status": terminal_status, "filled": 0.0}
        )
        spot_fake = FakeExchange()
        executor = CarryLiveExecutor(
            mode="live",
            execution_style="maker_taker_hedge",
            capital=3000.0,
            exchange_map={"binance": perp_fake, "binance_spot": spot_fake},
            perp_exchange=perp_fake,
            spot_exchange=spot_fake,
            state_store=StateStore(tmp_path / f"state_{terminal_status}.json"),
            max_allowed_spread_bps=10_000.0,  # fixture book rong 20 bps: tat guard spread de test state machine
        )

        leg1_res = executor.place_maker_leg1(
            "BTC/USDT:USDT",
            "sell",
            1.0,
            100.0,
            exchange_short="binance",
            exchange_long="binance_spot",
            spot_symbol="BTC/USDT",
        )
        assert leg1_res["leg1_order_id"] == "leg1-1"
        assert leg1_res["leg1_status"] == "PENDING_MAKER"

        verify_res = executor.verify_maker_fill_and_hedge_leg2(
            "BTC/USDT:USDT",
            exchange_short="binance",
            exchange_long="binance_spot",
            spot_symbol="BTC/USDT",
        )

        assert verify_res["leg1_status"] == expected_leg1_status[terminal_status]
        assert verify_res["leg1_filled_qty"] == pytest.approx(0.0)
        assert verify_res["leg2_hedged_qty"] == pytest.approx(0.0)
        assert verify_res["fired_market"] is False
        assert verify_res["execution_state"] == "IDLE"
        assert [o for o in perp_fake.orders if o["type"] == "market"] == []
        assert [o for o in spot_fake.orders if o["type"] == "market"] == []


def test_unfilled_maker_leg1_is_cancelled_and_never_orphaned(tmp_path: Path) -> None:
    """Lệnh Post-Only không khớp phải bị HỦY trên sàn, không để lệnh mồ côi.

    Rủi ro thực chiến: bắn GTX rồi xóa khỏi state mà không cancel thì lệnh vẫn
    treo trên sàn; khi giá quét qua sẽ tạo vị thế trần trụi ngoài tầm kiểm soát.
    """
    perp_fake = FakeExchange(
        order_status={"id": "leg1-1", "status": "open", "filled": 0.0}
    )
    spot_fake = FakeExchange()
    executor = CarryLiveExecutor(
        mode="live",
        execution_style="maker_taker_hedge",
        capital=3000.0,
        exchange_map={"binance": perp_fake, "binance_spot": spot_fake},
        perp_exchange=perp_fake,
        spot_exchange=spot_fake,
        state_store=StateStore(tmp_path / "state.json"),
        max_allowed_spread_bps=10_000.0,  # fixture book rong 20 bps: tat guard spread de test state machine
    )

    executor.place_maker_leg1(
        "BTC/USDT:USDT",
        "sell",
        1.0,
        100.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol="BTC/USDT",
    )

    assert executor.cancel_maker_leg1("BTC/USDT:USDT", exchange_name="binance") is True
    assert perp_fake.cancelled == [{"id": "leg1-1", "symbol": "BTC/USDT:USDT"}]
    assert executor.state.carry_positions["BTC/USDT:USDT"]["leg1_status"] == "CANCELED"
    assert executor.state.carry_positions["BTC/USDT:USDT"]["execution_state"] == "IDLE"
    assert [o for o in perp_fake.orders if o["type"] == "market"] == []
    assert [o for o in spot_fake.orders if o["type"] == "market"] == []
