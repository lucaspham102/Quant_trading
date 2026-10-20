"""Unit tests cho Cross-Exchange Carry và Barbell Portfolio (Sprint 9).

Phạm vi khóa hành vi:

- Ma trận phí ``maker_taker_hedge``: 10 bps/chiều Perp-Perp (20 bps khứ hồi),
  11.25 bps/chiều Spot-Perp (22.5 bps khứ hồi) và ngưỡng entry tối thiểu mỗi 8h.
- Rebalance ký quỹ liên sàn khi giá biến động +/-25%: có ghi nhận giao dịch
  ``rebalance`` và trừ phí 5 bps vào equity, không phát sinh margin call.
- Mô hình khớp lệnh Maker Post-Only: giá không đi qua mức treo thì Chân 1
  không khớp, không phát sinh phí, equity giữ nguyên vốn gốc.
- Barbell với ``enable_overlay=False`` phải trùng khít 100% equity curve của
  Carry và không tạo PnL overlay.

Toàn bộ dữ liệu được tổng hợp trong bộ nhớ thông qua
``align_multi_exchange_funding`` (đúng schema thật của module loader), không
gọi mạng và không đọc ``data/raw``. Mỗi test tự dựng dữ liệu riêng nên không
phụ thuộc thứ tự thực thi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.cross_exchange_carry import (
    CrossExchangeCarryConfig,
    run_cross_exchange_carry,
)
from src.multi_exchange_loader import align_multi_exchange_funding

SYMBOL = "AAA/USDT:USDT"
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


def _ohlcv_frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    """Dựng OHLCV giả quanh chuỗi ``close`` với biên high/low +/-5 bps.

    Biên high/low đủ rộng để lệnh Maker Post-Only (treo cao hơn close t-1
    đúng 0.5 bps) có thể khớp khi giá đi ngang, đồng thời vẫn giữ được khả
    năng "không khớp" khi giá giảm liên tục.
    """
    close_arr = np.asarray(close, dtype="float64")
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(len(close_arr), 1000.0, dtype="float64"),
        },
        index=index.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _funding_frame(index: pd.DatetimeIndex, rate: float) -> pd.DataFrame:
    """Dựng DataFrame funding 8h đúng schema ``timestamp/datetime/funding_rate/mark_price``.

    Các mốc được sinh từ 00:00 UTC của ngày đầu tiên, do đó rơi đúng vào
    khung thanh toán 00/08/16 UTC mà ``align_multi_exchange_funding`` kỳ vọng.
    """
    start = index[0].floor("D")
    n_events = int(np.ceil(len(index) / 8.0)) + 2
    event_index = pd.date_range(start, periods=n_events, freq="8h", tz="UTC").as_unit("ns")
    rates = np.full(len(event_index), float(rate), dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": (event_index.asi8 // 1_000_000).astype("int64"),
            "datetime": event_index,
            "funding_rate": rates,
            "mark_price": np.full(len(event_index), 100.0, dtype="float64"),
        }
    )


def _aligned_symbol(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    binance_rate: float,
    bybit_rate: float | None,
) -> dict[str, pd.DataFrame]:
    """Ghép OHLCV giả với funding Binance/Bybit qua loader thật.

    OKX cố ý vắng mặt để mô phỏng sàn chưa niêm yết (``ewma_funding_okx`` và
    ``funding_event_okx`` toàn NaN), đúng kịch bản 2 sàn của Sprint 9.
    """
    funding: dict[str, pd.DataFrame] = {
        "binance": _funding_frame(index, binance_rate),
    }
    if bybit_rate is not None:
        funding["bybit"] = _funding_frame(index, bybit_rate)

    aligned = align_multi_exchange_funding(
        SYMBOL,
        _ohlcv_frame(index, close),
        funding,
    )
    return {SYMBOL: aligned}


def _carry_config(**overrides: float | int | str) -> CrossExchangeCarryConfig:
    """Config Carry tối giản, mặc định ép chế độ Perp-vs-Perp để test tất định."""
    base: dict[str, float | int | str] = {
        "mode": "perp_perp_spread",
        "initial_capital": INITIAL_CAPITAL,
    }
    base.update(overrides)
    return CrossExchangeCarryConfig(**base)


def _rebalance_scenario() -> dict[str, pd.DataFrame]:
    """Kịch bản giá đi ngang 100, nhảy +26% lên 126 rồi +19% lên 150.

    Funding Binance 5 bps và Bybit 1 bps mỗi 8h tạo spread 4 bps -> đủ điều
    kiện mở vị thế Perp-Perp tại mốc 8h đầu tiên có EWMA hợp lệ.
    """
    n = 400
    index = _hourly_index(n)
    close = np.full(n, 100.0, dtype="float64")
    close[100:250] = 126.0  # +26% so với mốc tham chiếu 100 -> rebalance
    close[250:] = 150.0     # +19.05% so với 126 -> chưa đủ ngưỡng 25%
    return _aligned_symbol(index, close, binance_rate=0.0005, bybit_rate=0.0001)


def _no_fill_scenario() -> dict[str, pd.DataFrame]:
    """Kịch bản giá giảm 0.5% mỗi nến: high luôn thấp hơn giá treo lệnh short.

    Giá treo = close(t-1) * 1.00005, trong khi high(t) = close(t) * 1.0005 và
    close(t) = close(t-1) * 0.995, nên high(t) < giá treo tuyệt đối.
    """
    n = 200
    index = _hourly_index(n)
    steps = np.arange(n, dtype="float64")
    close = 100.0 * np.power(0.995, steps)
    return _aligned_symbol(index, close, binance_rate=0.0005, bybit_rate=0.0001)


# ---------------------------------------------------------------------------
# 1. Ma trận phí maker_taker_hedge
# ---------------------------------------------------------------------------
def test_maker_taker_hedge_cost_is_20bps_round_trip() -> None:
    """Chuẩn phí ``maker_taker_hedge`` đúng docs/go_no_go_sprint9.md.

    Chân 1 Maker 2 bps + adverse selection 1 bp, Chân 2 Taker 5 bps + slippage
    2 bps -> 10 bps/chiều, 20 bps khứ hồi. Spot-Perp dùng override 11.25
    bps/chiều -> 22.5 bps khứ hồi. Ngưỡng entry mỗi 8h = round-trip / 12 kỳ.
    """
    config = CrossExchangeCarryConfig()

    # Mặc định phải là maker_taker_hedge (chuẩn so sánh duy nhất khi GO/NO-GO).
    assert config.execution == "maker_taker_hedge"

    assert config.one_way_cost_perp_perp() == pytest.approx(0.0010)
    assert config.round_trip_cost_perp_perp() == pytest.approx(0.0020)
    assert config.one_way_cost_spot_perp() == pytest.approx(0.001125)
    assert config.round_trip_cost_spot_perp() == pytest.approx(0.00225)
    assert config.min_entry_rate_per_8h("perp_perp_spread") == pytest.approx(
        0.0020 / 12.0
    )

    # Đẳng thức kế toán: khứ hồi luôn gấp đôi một chiều.
    assert config.round_trip_cost_perp_perp() == pytest.approx(
        2.0 * config.one_way_cost_perp_perp()
    )
    assert config.round_trip_cost_spot_perp() == pytest.approx(
        2.0 * config.one_way_cost_spot_perp()
    )


# ---------------------------------------------------------------------------
# 2. Rebalance ký quỹ liên sàn khi giá biến động >= 25%
# ---------------------------------------------------------------------------
def test_cross_exchange_price_rebalance_at_25pct_charges_5bps_and_prevents_liquidation() -> None:
    """Giá vượt +/-25% phải kích hoạt rebalance, trừ 5 bps và không margin call.

    Vị thế Perp-Perp được mở khi giá phẳng 100. Sau cú nhảy +26% lên 126,
    engine phải ghi 1 dòng ``action='rebalance'``, cộng dồn ``rebalance_fees``
    dương và giữ ``Margin Calls = 0``. Cú nhảy +19.05% tiếp theo (126 -> 150)
    không đủ ngưỡng nên không tạo thêm rebalance bắt buộc.
    """
    data = _rebalance_scenario()
    result = run_cross_exchange_carry(data, _carry_config())
    metrics = result["metrics"]
    trades = result["trades_df"]

    # Vị thế phải thực sự được mở, nếu không phép thử rebalance vô nghĩa.
    assert metrics["Num Entries"] >= 1

    assert metrics["Price Rebalances"] >= 1
    assert metrics["Margin Calls"] == 0

    rebalance_rows = trades[
        trades["action"].astype(str).str.contains("rebalanc", case=False)
    ]
    assert len(rebalance_rows) >= 1
    assert rebalance_rows["fee"].gt(0.0).all()

    assert result["rebalance_fees"] > 0.0
    assert result["fees"] >= result["rebalance_fees"]
    assert np.isfinite(result["equity_curve"].to_numpy()).all()


# ---------------------------------------------------------------------------
# 3. Maker Post-Only không khớp khi giá không đi qua mức treo
# ---------------------------------------------------------------------------
def test_maker_post_only_no_fill_when_price_does_not_cross_limit() -> None:
    """Giá giảm liên tục: lệnh short Post-Only không khớp, không mất phí.

    Engine đặt lệnh Bán tại ``close(t-1) * (1 + 0.5 bps)`` với TTL 2 nến.
    Vì high(t) luôn thấp hơn mức treo, lệnh bị hủy sau TTL và được đếm vào
    ``Maker No-Fill Count``; không có entry nào và equity đứng yên ở vốn gốc.
    """
    data = _no_fill_scenario()
    result = run_cross_exchange_carry(data, _carry_config())
    metrics = result["metrics"]

    assert metrics["Maker No-Fill Count"] >= 1
    assert metrics["Num Entries"] == 0
    assert result["fees"] == pytest.approx(0.0)
    assert result["rebalance_fees"] == pytest.approx(0.0)
    assert result["equity_curve"].iloc[-1] == pytest.approx(INITIAL_CAPITAL)

    # Không có giao dịch mở/đóng nào được ghi nhận.
    trades = result["trades_df"]
    assert trades.empty or not trades["action"].astype(str).str.contains(
        "open|close", case=False, regex=True
    ).any()


# ---------------------------------------------------------------------------
# 4. Barbell tắt overlay phải trùng khít Carry
# ---------------------------------------------------------------------------
def test_barbell_overlay_off_matches_carry_exactly() -> None:
    """``enable_overlay=False``: equity Barbell trùng khít equity Carry.

    Đây là bất biến kế toán quan trọng: tắt overlay thì không được double-count
    vốn, không tạo PnL overlay và không phát sinh phí overlay.
    """
    data = _rebalance_scenario()
    carry_config = _carry_config()
    barbell = run_barbell_portfolio(
        data,
        BarbellConfig(enable_overlay=False, carry_config=carry_config),
    )

    pd.testing.assert_series_equal(
        barbell["equity_curve"],
        barbell["carry_result"]["equity_curve"],
        check_names=False,
    )
    np.testing.assert_allclose(
        barbell["equity_curve"].to_numpy(dtype="float64"),
        barbell["carry_result"]["equity_curve"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-9,
    )

    assert barbell["overlay_pnl"] == pytest.approx(0.0)
    assert barbell["fees"] == pytest.approx(barbell["carry_result"]["fees"])

    overlay_pnl_series = barbell["attribution_series"]["overlay_pnl"]
    assert overlay_pnl_series.abs().max() == pytest.approx(0.0)
