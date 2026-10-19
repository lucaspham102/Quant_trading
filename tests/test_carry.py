"""Unit tests cho chiến lược Funding Carry và Funding Loader (Sprint 6).

Đảm bảo hoàn toàn offline, deterministic, không gọi mạng, không look-ahead bias.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.carry_strategy import CarryConfig, run_carry_portfolio
from src.funding_loader import align_ohlcv_with_funding, fetch_funding_history


def test_align_no_lookahead() -> None:
    """Kiểm tra align_ohlcv_with_funding không bị rò rỉ dữ liệu tương lai (look-ahead bias).

    (i) Cột funding_rate_event và last_known_funding_rate tồn tại.
    (ii) last_known_funding_rate tại nến t chỉ nhận giá trị của kỳ funding có timestamp <= t.
         Đặc biệt, ngay trước mốc 08:00 (ví dụ 07:00), giá trị phải là của kỳ 00:00, không phải 08:00.
    (iii) funding_rate_event chỉ khác 0 tại đúng các nến 00:00, 08:00, 16:00 và khớp chính xác funding_rate.
    """
    # 1. Tạo OHLCV 1h gồm 48 nến (2 ngày: 2024-01-01 00:00 đến 2024-01-02 23:00 UTC)
    dates = pd.date_range("2024-01-01 00:00:00", periods=48, freq="1h", tz="UTC").as_unit("ns")
    ohlcv_df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "volume": 1000.0,
        },
        index=dates,
    )
    ohlcv_df.index.name = "datetime"

    # 2. Tạo funding_df có các mốc 00:00, 08:00, 16:00 với funding rate tăng dần
    funding_times = pd.date_range("2024-01-01 00:00:00", periods=6, freq="8h", tz="UTC").as_unit("ns")
    funding_rates = [0.0001 * (i + 1) for i in range(len(funding_times))]  # [0.0001, 0.0002, ..., 0.0006]
    funding_df = pd.DataFrame(
        {
            "timestamp": (funding_times.asi8 // 1_000_000).astype("int64"),
            "datetime": funding_times,
            "funding_rate": funding_rates,
            "mark_price": [100.0 + i for i in range(len(funding_times))],
        }
    )

    aligned = align_ohlcv_with_funding(ohlcv_df, funding_df)

    # (i) Kiểm tra các cột bắt buộc tồn tại
    assert "funding_rate_event" in aligned.columns
    assert "last_known_funding_rate" in aligned.columns
    assert str(aligned.index.tz) == "UTC"
    assert aligned.index.name == "datetime"

    # (ii) Kiểm tra last_known_funding_rate không look-ahead
    # Tại nến 07:00 (ngay trước mốc 08:00): phải là giá trị kỳ 00:00 (0.0001), KHÔNG phải 08:00 (0.0002)
    t_07 = pd.Timestamp("2024-01-01 07:00:00", tz="UTC")
    assert aligned.loc[t_07, "last_known_funding_rate"] == pytest.approx(0.0001)

    # Tại đúng nến 08:00: funding đã chốt lúc 08:00 nên được biết (0.0002)
    t_08 = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    assert aligned.loc[t_08, "last_known_funding_rate"] == pytest.approx(0.0002)

    # Tại nến 15:00 (ngay trước mốc 16:00): phải là giá trị kỳ 08:00 (0.0002), KHÔNG phải 16:00 (0.0003)
    t_15 = pd.Timestamp("2024-01-01 15:00:00", tz="UTC")
    assert aligned.loc[t_15, "last_known_funding_rate"] == pytest.approx(0.0002)

    # Tại nến 16:00: nhận giá trị 0.0003
    t_16 = pd.Timestamp("2024-01-01 16:00:00", tz="UTC")
    assert aligned.loc[t_16, "last_known_funding_rate"] == pytest.approx(0.0003)

    # Kiểm tra tổng quát trên toàn bộ 48 nến: tại nến t, last_known_funding_rate chỉ lấy từ mốc <= t
    for t_candle in aligned.index:
        rate = float(aligned.loc[t_candle, "last_known_funding_rate"])
        valid_f = funding_df[funding_df["datetime"] <= t_candle]
        if valid_f.empty:
            assert rate == 0.0
        else:
            expected_rate = float(valid_f.iloc[-1]["funding_rate"])
            assert rate == pytest.approx(expected_rate)

    # (iii) Kiểm tra funding_rate_event chỉ khác 0 tại 00:00, 08:00, 16:00
    for t_candle in aligned.index:
        event_rate = float(aligned.loc[t_candle, "funding_rate_event"])
        if t_candle.hour in (0, 8, 16):
            expected_f = funding_df[funding_df["datetime"] == t_candle]
            if not expected_f.empty:
                assert event_rate == pytest.approx(float(expected_f.iloc[0]["funding_rate"]))
            else:
                assert event_rate == 0.0
        else:
            assert event_rate == 0.0


def test_carry_equity_equals_funding_minus_fees() -> None:
    """Kiểm tra đường cong vốn: equity cuối = initial + tổng funding nhận - tổng phí.

    Sử dụng 1 symbol với cấu trúc coinm_1leg, triệt tiêu nhiễu cơ sở (basis_noise_std = 0.0).
    """
    dates = pd.date_range("2024-01-01 00:00:00", periods=48, freq="1h", tz="UTC").as_unit("ns")
    ohlcv = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000.0},
        index=dates,
    )
    ohlcv.index.name = "datetime"

    # Funding rate dương ổn định: 0.0002 mỗi 8h (đủ vượt entry_threshold 0.0001)
    funding_times = pd.date_range("2024-01-01 00:00:00", periods=6, freq="8h", tz="UTC").as_unit("ns")
    funding_df = pd.DataFrame(
        {
            "timestamp": (funding_times.asi8 // 1_000_000).astype("int64"),
            "datetime": funding_times,
            "funding_rate": [0.0002] * len(funding_times),
            "mark_price": [100.0] * len(funding_times),
        }
    )

    aligned = align_ohlcv_with_funding(ohlcv, funding_df)

    cfg = CarryConfig(
        structure="coinm_1leg",
        entry_threshold=0.0001,
        exit_threshold=0.0,
        ema_span=6,
        initial_capital=400.0,
        basis_noise_std=0.0,  # Không có nhiễu cơ sở để kiểm tra kế toán vốn chính xác
    )

    result = run_carry_portfolio({"BTC/USDT:USDT": aligned}, cfg)

    equity_curve = result["equity_curve"]
    metrics = result["metrics"]
    trades_df = result["trades_df"]

    initial = cfg.initial_capital
    final_equity = float(equity_curve.iloc[-1])
    total_funding = metrics["Total Funding Collected"]
    total_fees = metrics["Total Fees Paid"]

    # Kiểm tra bảo toàn kế toán: Final Equity = Initial + Funding - Fees
    assert final_equity == pytest.approx(initial + total_funding - total_fees, rel=1e-6)
    assert total_funding > 0.0
    assert metrics["Time in Market %"] > 0.0
    assert total_fees > 0.0
    assert len(trades_df) >= 1  # Đã mở ít nhất 1 vị thế


def test_hysteresis_prevents_churn() -> None:
    """Kiểm tra cơ chế hysteresis ngăn chặn giao dịch quá mức (churn).

    Khi funding dao động nhẹ quanh ngưỡng entry (vừa trên / vừa dưới entry_threshold
    nhưng luôn > exit_threshold 0.0), chiến lược carry không mở/đóng liên tục.
    Số lần mở lệnh trong chuỗi ~200 nến phải <= 2, trong khi chiến lược ngây thơ (naive)
    không có hysteresis sẽ churn gấp nhiều lần.
    """
    n_candles = 200
    dates = pd.date_range("2024-01-01 00:00:00", periods=n_candles, freq="1h", tz="UTC").as_unit("ns")
    ohlcv = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "volume": 1000.0},
        index=dates,
    )
    ohlcv.index.name = "datetime"

    # Tạo chuỗi funding rate dao động quanh 0.0001 (từ 0.000092 đến 0.000108), nhưng luôn > 0.0
    funding_times = pd.date_range("2024-01-01 00:00:00", periods=(n_candles // 8) + 2, freq="8h", tz="UTC").as_unit("ns")
    rng = np.random.default_rng(seed=42)
    # Dao động xen kẽ trên và dưới ngưỡng 0.0001
    rates = [0.0001 + 0.00002 * ((-1) ** i) + rng.uniform(-0.000005, 0.000005) for i in range(len(funding_times))]
    # Đảm bảo điểm đầu tiên vượt ngưỡng để mở lệnh
    rates[0] = 0.00015
    # Tất cả các tỷ lệ đều dương (lớn hơn exit_threshold 0.0)
    rates = [max(r, 0.00005) for r in rates]

    funding_df = pd.DataFrame(
        {
            "timestamp": (funding_times.asi8 // 1_000_000).astype("int64"),
            "datetime": funding_times,
            "funding_rate": rates,
            "mark_price": [100.0] * len(funding_times),
        }
    )

    aligned = align_ohlcv_with_funding(ohlcv, funding_df)

    cfg = CarryConfig(
        structure="coinm_1leg",
        entry_threshold=0.0001,
        exit_threshold=0.0,  # Ngưỡng exit thấp hơn nhiều so với entry -> vùng trễ (hysteresis)
        min_hold_periods=3,
        ema_span=4,
        initial_capital=400.0,
        basis_noise_std=0.0,
    )

    result = run_carry_portfolio({"BTC/USDT:USDT": aligned}, cfg)
    trades_df = result["trades_df"]

    open_trades = trades_df[trades_df["action"] == "open"]
    # Nhờ hysteresis, số lần mở lệnh rất ít (<= 2 lần trên 200 nến)
    assert len(open_trades) <= 2

    # Đối chứng với chiến lược ngây thơ (naive) không hysteresis:
    # Mở khi ema >= 0.0001 và đóng ngay khi ema < 0.0001
    ema_signal = aligned["last_known_funding_rate"].ewm(span=cfg.ema_span).mean().shift(1).fillna(0.0)
    naive_in_pos = False
    naive_opens = 0
    for val in ema_signal:
        if not naive_in_pos and val >= cfg.entry_threshold:
            naive_in_pos = True
            naive_opens += 1
        elif naive_in_pos and val < cfg.entry_threshold:
            naive_in_pos = False

    # Naive churn nhiều hơn hẳn do rung lắc quanh ngưỡng entry
    assert naive_opens > len(open_trades)


def test_carry_config_costs_and_eta() -> None:
    """Kiểm tra hàm tính chi phí one_way_cost và tỷ lệ đòn bẩy notional_eta trong CarryConfig."""
    # coinm_1leg
    cfg_1leg = CarryConfig(structure="coinm_1leg", perp_fee=0.0005, slippage=0.0002)
    assert cfg_1leg.one_way_cost() == pytest.approx(0.0007)
    assert cfg_1leg.notional_eta() == pytest.approx(1.0)

    # spot_perp_2leg
    cfg_2leg = CarryConfig(
        structure="spot_perp_2leg",
        spot_fee=0.00075,
        perp_fee=0.0005,
        slippage=0.0002,
    )
    assert cfg_2leg.one_way_cost() == pytest.approx(0.00165)
    assert cfg_2leg.notional_eta() == pytest.approx(2.0 / 3.0)

    # Cấu trúc không hợp lệ
    cfg_invalid = CarryConfig(structure="invalid_struct")
    with pytest.raises(ValueError, match="Cấu trúc thực thi không hợp lệ"):
        cfg_invalid.one_way_cost()
    with pytest.raises(ValueError, match="Cấu trúc thực thi không hợp lệ"):
        cfg_invalid.notional_eta()


class FakeFundingExchange:
    """Sàn giả lập trả về danh sách lịch sử funding rate phân trang cho kiểm thử offline."""

    def __init__(self, records: list[dict[str, Any]], server_time_ms: int) -> None:
        self.records = records
        self.server_time_ms = server_time_ms
        self.call_history: list[dict[str, Any]] = []

    def milliseconds(self) -> int:
        return self.server_time_ms

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.call_history.append({"symbol": symbol, "since": since, "limit": limit})
        filtered = [r for r in self.records if since is None or int(r["timestamp"]) >= since]
        if limit is not None:
            filtered = filtered[:limit]
        return [dict(r) for r in filtered]


def test_fetch_funding_history_pagination_and_cache(tmp_path: Path) -> None:
    """Kiểm tra fetch_funding_history: phân trang, schema chuẩn và cơ chế đọc/ghi cache offline."""
    base_ts = 1_700_000_000_000  # ms
    eight_h_ms = 8 * 3600 * 1000

    # Giả lập 5 bản ghi funding rate
    records = [
        {"timestamp": base_ts + i * eight_h_ms, "fundingRate": 0.0001 * (i + 1), "markPrice": 100.0 + i}
        for i in range(5)
    ]
    fake_exchange = FakeFundingExchange(records, server_time_ms=base_ts + 6 * eight_h_ms)

    start_date_str = pd.Timestamp(base_ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = pd.Timestamp(base_ts + 4 * eight_h_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")

    # Lần 1: Chưa có cache, phải tải từ fake exchange và ghi cache
    df1 = fetch_funding_history(
        symbol="BTC/USDT:USDT",
        start_date=start_date_str,
        end_date=end_date_str,
        data_dir=str(tmp_path),
        exchange=fake_exchange,
    )

    assert len(df1) == 5
    assert list(df1.columns) == ["timestamp", "datetime", "funding_rate", "mark_price"]
    assert df1["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert len(fake_exchange.call_history) >= 1

    # Kiểm tra file parquet đã được tạo tại đúng đường dẫn cache
    cache_file = tmp_path / "BTC_USDT_USDT_funding.parquet"
    assert cache_file.exists()

    # Lần 2: Đã có cache đủ [start, end], không được gọi exchange nữa
    fake_exchange_2 = FakeFundingExchange(records, server_time_ms=base_ts + 6 * eight_h_ms)
    df2 = fetch_funding_history(
        symbol="BTC/USDT:USDT",
        start_date=start_date_str,
        end_date=end_date_str,
        data_dir=str(tmp_path),
        exchange=fake_exchange_2,
    )

    assert len(df2) == 5
    assert len(fake_exchange_2.call_history) == 0  # Không có lệnh gọi mạng nào vì cache đã đủ
