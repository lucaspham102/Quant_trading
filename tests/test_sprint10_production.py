"""Unit tests cho Sprint 10 Chặng 1: lot-size, Basis MTM và horizon dài hạn.

Phạm vi khóa hành vi:

- ``quantize_order_qty`` làm tròn xuống đúng ``step_size`` của BTC/ETH/SOL/BNB,
  chặn lệnh dưới ``min_notional`` và giữ step mặc định cho symbol lạ.
- Basis MTM phải phát sinh PnL thực khi ``basis_pct`` dao động, đồng thời bảo
  toàn đẳng thức kế toán từng giờ ``funding + basis - fees == net_pnl``.
- Ngưỡng entry mới (0.50 bps/8h) cho phép vào lệnh với spread chỉ 1 bp/8h và
  hysteresis giữ vị thế tối thiểu 42 kỳ funding.
- Lot-size quantization phải giữ khối lượng giao dịch trên lưới step của sàn.

Toàn bộ dữ liệu được tổng hợp trong bộ nhớ qua
``align_multi_exchange_funding`` (đúng schema thật), không gọi mạng, không đọc
parquet. Mỗi test tự dựng dữ liệu riêng nên độc lập thứ tự thực thi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cross_exchange_carry import (
    DEFAULT_MIN_NOTIONAL,
    EXCHANGE_LOT_SPECS,
    LOT_STEP_SIZES,
    CrossExchangeCarryConfig,
    quantize_order_qty,
    run_cross_exchange_carry,
)
from src.multi_exchange_loader import align_multi_exchange_funding

SYMBOL = "BTC/USDT:USDT"
START = "2024-01-01 00:00:00"
INITIAL_CAPITAL = 3000.0


# ---------------------------------------------------------------------------
# Helpers dựng dữ liệu tổng hợp (offline, tất định)
# ---------------------------------------------------------------------------
def _hourly_index(periods: int) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC ns tên ``datetime`` với tần suất 1 giờ."""
    index = pd.date_range(START, periods=periods, freq="1h", tz="UTC").as_unit("ns")
    index.name = "datetime"
    return index


def _ohlcv_frame(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    basis_pct: np.ndarray | None = None,
) -> pd.DataFrame:
    """Dựng OHLCV giả quanh chuỗi ``close`` kèm perp/spot close và basis tùy chọn.

    Biên high/low +/-5 bps đủ rộng để lệnh Maker Post-Only (treo cao hơn close
    t-1 đúng 0.5 bps) khớp được khi giá đi ngang.
    """
    close_arr = np.asarray(close, dtype="float64")
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(len(close_arr), 1000.0, dtype="float64"),
            "perp_close": close_arr,
            "spot_close": close_arr,
        },
        index=index.copy(),
    )
    if basis_pct is not None:
        frame["basis_pct"] = np.asarray(basis_pct, dtype="float64")
    frame.index.name = "datetime"
    return frame


def _funding_frame(index: pd.DatetimeIndex, rate: float) -> pd.DataFrame:
    """Dựng funding 8h đúng schema ``timestamp/datetime/funding_rate/mark_price``.

    Các mốc sinh từ 00:00 UTC ngày đầu nên rơi đúng khung thanh toán 00/08/16.
    """
    start = index[0].floor("D")
    n_events = int(np.ceil(len(index) / 8.0)) + 2
    event_index = pd.date_range(start, periods=n_events, freq="8h", tz="UTC").as_unit("ns")
    return pd.DataFrame(
        {
            "timestamp": (event_index.asi8 // 1_000_000).astype("int64"),
            "datetime": event_index,
            "funding_rate": np.full(len(event_index), float(rate), dtype="float64"),
            "mark_price": np.full(len(event_index), 100.0, dtype="float64"),
        }
    )


def _aligned_symbol(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    binance_rate: float,
    bybit_rate: float | None,
    basis_pct: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Ghép OHLCV giả với funding Binance/Bybit qua loader thật.

    OKX cố ý vắng mặt để mô phỏng sàn chưa niêm yết, đúng kịch bản 2 sàn.
    """
    funding: dict[str, pd.DataFrame] = {
        "binance": _funding_frame(index, binance_rate),
    }
    if bybit_rate is not None:
        funding["bybit"] = _funding_frame(index, bybit_rate)

    aligned = align_multi_exchange_funding(
        SYMBOL,
        _ohlcv_frame(index, close, basis_pct),
        funding,
    )
    return {SYMBOL: aligned}


def _carry_config(**overrides: object) -> CrossExchangeCarryConfig:
    """Config Carry tối giản, mặc định ép Perp-vs-Perp để test tất định."""
    base: dict[str, object] = {
        "mode": "perp_perp_spread",
        "initial_capital": INITIAL_CAPITAL,
    }
    base.update(overrides)
    return CrossExchangeCarryConfig(**base)


def _flat_scenario(
    periods: int,
    binance_rate: float,
    bybit_rate: float,
    basis_pct: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Kịch bản giá phẳng 100 trong ``periods`` nến với funding 2 sàn cho trước."""
    index = _hourly_index(periods)
    close = np.full(periods, 100.0, dtype="float64")
    return _aligned_symbol(index, close, binance_rate, bybit_rate, basis_pct)


# ---------------------------------------------------------------------------
# 1. Lot-size của 4 cặp lõi
# ---------------------------------------------------------------------------
def test_quantize_order_qty_step_sizes_for_core4() -> None:
    """Khối lượng phải chia hết ``step_size`` và không vượt quá notional mục tiêu.

    Ở giá 100 và target 1000 USDT, BTC cho đúng 10.0 BTC. Với target 750 USDT,
    từng cặp lõi (BTC 65000, ETH 3000, SOL 150, BNB 600) phải làm tròn xuống
    thực tế: ``qty * price <= target`` và ``qty / step`` là số nguyên.
    """
    qty, actual = quantize_order_qty(SYMBOL, 1000.0, 100.0)
    assert qty == pytest.approx(10.0)
    assert actual == pytest.approx(1000.0)
    assert qty / EXCHANGE_LOT_SPECS[SYMBOL]["step_size"] == pytest.approx(10000.0)

    cases = [
        ("BTC/USDT:USDT", 65000.0),
        ("ETH/USDT:USDT", 3000.0),
        ("SOL/USDT:USDT", 150.0),
        ("BNB/USDT:USDT", 600.0),
    ]
    target = 750.0
    for symbol, price in cases:
        step = EXCHANGE_LOT_SPECS[symbol]["step_size"]
        assert LOT_STEP_SIZES[symbol] == pytest.approx(step)

        q, notional = quantize_order_qty(symbol, target, price)

        assert q > 0.0
        assert notional == pytest.approx(q * price)
        assert notional <= target + 1e-6
        units = q / step
        assert units == pytest.approx(round(units), abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Chặn lệnh dưới min_notional và symbol lạ
# ---------------------------------------------------------------------------
def test_quantize_order_qty_blocks_below_min_notional() -> None:
    """Dưới 10 USDT (hoặc tham số không hợp lệ) phải trả ``(0.0, 0.0)``.

    Symbol lạ như ``DOGE/USDT:USDT`` dùng step mặc định 0.001 và vẫn trả qty
    dương khi giá trị lệnh đủ ngưỡng min-notional.
    """
    assert DEFAULT_MIN_NOTIONAL == pytest.approx(10.0)

    # 5 USDT dưới sàn 10 USDT -> không đặt lệnh.
    assert quantize_order_qty(SYMBOL, 5.0, 65000.0) == (0.0, 0.0)
    # 9.99 USDT ở giá 1.0 cũng dưới ngưỡng sau khi làm tròn.
    assert quantize_order_qty(SYMBOL, 9.99, 1.0) == (0.0, 0.0)

    # Tham số không hợp lệ.
    assert quantize_order_qty(SYMBOL, 0.0, 65000.0) == (0.0, 0.0)
    assert quantize_order_qty(SYMBOL, 100.0, 0.0) == (0.0, 0.0)
    assert quantize_order_qty(SYMBOL, 100.0, -1.0) == (0.0, 0.0)

    unknown = "DOGE/USDT:USDT"
    q, notional = quantize_order_qty(unknown, 50.0, 1.0)
    assert q > 0.0
    assert notional == pytest.approx(50.0)
    assert q / 0.001 == pytest.approx(round(q / 0.001), abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Basis MTM thực sự phát sinh khi basis_pct dao động
# ---------------------------------------------------------------------------
def test_basis_mtm_is_nonzero_when_basis_pct_moves() -> None:
    """``basis_pct`` dao động sin +/-20 bps phải tạo Basis MTM khác 0.

    Vị thế được mở khi spread funding 4 bps > ngưỡng hiệu dụng. Mỗi nến 1h
    engine ghi ``basis_pnl = -notional * delta(basis)``; do đó tổng basis khác
    0, độ lệch chuẩn dương và đẳng thức kế toán từng giờ phải khớp tuyệt đối.
    """
    n = 300
    index = _hourly_index(n)
    steps = np.arange(n, dtype="float64")
    basis_pct = 0.002 * np.sin(2.0 * np.pi * steps / 24.0)
    data = _flat_scenario(
        n,
        binance_rate=0.0005,
        bybit_rate=0.0001,
        basis_pct=basis_pct,
    )

    config = _carry_config(
        top_k=1,
        max_payback_periods=42,
        entry_funding_threshold=0.0,
    )
    result = run_cross_exchange_carry(data, config)

    trades = result["trades_df"]
    assert (trades["action"].astype(str) == "open").any(), (
        "Kịch bản phải mở được vị thế để Basis MTM có ý nghĩa"
    )

    attribution = result["attribution_series"]
    assert abs(result["basis_pnl"]) > 1e-9
    assert result["basis_mtm_std"] > 0.0
    assert attribution["basis_pnl"].abs().sum() > 0.0

    np.testing.assert_allclose(
        attribution["funding_pnl"]
        + attribution["basis_pnl"]
        - attribution["fees"],
        attribution["net_pnl"],
        rtol=0.0,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 4. Horizon dài hạn cho phép entry với funding 1 bp/8h
# ---------------------------------------------------------------------------
def test_long_horizon_threshold_allows_1bp_funding_entry() -> None:
    """Spread 1 bp/8h vượt ngưỡng 0.50 bps/8h và phải được giữ >= 42 kỳ.

    Binance trả 1.0 bps, Bybit 0.0 -> spread 1.0 bp. Config mặc định mới có
    ``max_payback_periods=42`` (ngưỡng 0.476 bps) và
    ``entry_funding_threshold=0.5 bps`` nên vẫn mở lệnh. Funding thu về dương
    và không có close nào trước khi đủ 42 kỳ funding.
    """
    n = 400
    index = _hourly_index(n)
    data = _flat_scenario(n, binance_rate=0.0001, bybit_rate=0.0)

    config = _carry_config(
        top_k=1,
        initial_capital=INITIAL_CAPITAL,
        max_payback_periods=42,
        min_hold_periods=42,
        entry_funding_threshold=0.00005,
    )
    result = run_cross_exchange_carry(data, config)
    trades = result["trades_df"]

    opens = trades[trades["action"].astype(str) == "open"]
    assert not opens.empty, (
        "Spread 1 bp/8h phải vượt ngưỡng entry hiệu dụng 0.5 bps/8h"
    )

    open_time = pd.Timestamp(opens["time"].iloc[0])
    event_grid = pd.date_range(index[0], index[-1], freq="8h", tz="UTC")
    closes = trades[trades["action"].astype(str) == "close"]
    for close_time in closes["time"]:
        events_held = int(
            ((event_grid > open_time) & (event_grid <= pd.Timestamp(close_time))).sum()
        )
        assert events_held >= 42, "Không được đóng vị thế trước 42 kỳ funding"

    assert result["funding_pnl"] > 0.0
    assert result["equity_curve"].iloc[-1] > INITIAL_CAPITAL * 0.99


# ---------------------------------------------------------------------------
# 5. Lot-size quantization giữ khối lượng trên lưới step
# ---------------------------------------------------------------------------
def test_lot_size_quantization_keeps_qty_on_step_grid() -> None:
    """Mọi lệnh mở phải >= 10 USDT và khối lượng chia hết step 0.001 của BTC.

    Giá 65000, vốn 3000, top_k=1 và tắt maker-fill để entry khớp đúng giá
    65000. Nhờ đó ``notional / 65000 / 0.001`` là số nguyên tuyệt đối.
    """
    n = 400
    index = _hourly_index(n)
    close = np.full(n, 65000.0, dtype="float64")
    data = _aligned_symbol(
        index,
        close,
        binance_rate=0.0005,
        bybit_rate=0.0001,
    )

    config = _carry_config(
        top_k=1,
        initial_capital=INITIAL_CAPITAL,
        lot_size_quantization=True,
        enable_maker_fill_model=False,
    )
    result = run_cross_exchange_carry(data, config)

    opens = result["trades_df"]
    opens = opens[opens["action"].astype(str) == "open"]
    assert not opens.empty, (
        "Kịch bản phải mở được lệnh BTC để kiểm tra lot-size quantization"
    )

    notionals = opens["notional"].to_numpy(dtype="float64")
    assert (notionals >= DEFAULT_MIN_NOTIONAL).all()

    step = EXCHANGE_LOT_SPECS["BTC/USDT:USDT"]["step_size"]
    qty = notionals / 65000.0
    units = qty / step
    np.testing.assert_allclose(
        units,
        np.round(units),
        rtol=0.0,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 6. Mặc định cấu hình đúng đặc tả Sprint 10 Chặng 1
# ---------------------------------------------------------------------------
def test_default_config_matches_sprint10_spec() -> None:
    """Khóa các tham số mặc định mới của Sprint 10 Chặng 1."""
    config = CrossExchangeCarryConfig()

    assert config.ema_span == 21
    assert config.max_payback_periods == 42
    assert config.min_hold_periods == 42
    assert config.entry_funding_threshold == pytest.approx(0.00005)
    assert config.exit_funding_threshold == pytest.approx(-0.00001)
    assert config.lot_size_quantization is True
