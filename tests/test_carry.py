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
from src.funding_loader import (
    align_ohlcv_with_funding,
    align_spot_perp_funding,
    fetch_funding_history,
    fetch_spot_ohlcv,
)


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


# ---------------------------------------------------------------------------
# Sprint 7: Bar-by-bar MTM, margin/price rebalance, passive long-term mode,
# Spot OHLCV loader và đồng bộ Spot/Perp/Funding không look-ahead.
# ---------------------------------------------------------------------------


def test_spot_perp_mtm_tracks_basis_variation() -> None:
    """MTM spot_perp_2leg phản ánh biến động giá perp dù spot đứng yên.

    Spot giữ nguyên 100, perp dao động hình sin quanh 100. Vị thế mua spot /
    bán perp phải sinh lãi lỗ MTM từng nến, tạo drawdown và tổng MTM khác 0.
    """
    n = 200
    dates = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1h", tz="UTC").as_unit("ns")
    perp_close = 100.0 * (1.0 + 0.005 * np.sin(np.arange(n) / 7.0))
    spot_close = np.full(n, 100.0)
    basis_pct = (perp_close - spot_close) / spot_close
    funding_event = np.where(dates.hour % 8 == 0, 0.0002, 0.0)

    frame = pd.DataFrame(
        {
            "open": perp_close,
            "high": perp_close,
            "low": perp_close,
            "close": perp_close,
            "volume": 1000.0,
            "spot_close": spot_close,
            "perp_close": perp_close,
            "basis_pct": basis_pct,
            "funding_rate_event": funding_event,
            "last_known_funding_rate": 0.0002,
        },
        index=dates,
    )
    frame.index.name = "datetime"

    cfg = CarryConfig(
        structure="spot_perp_2leg",
        basis_noise_std=0.0,
        entry_threshold=0.0001,
        exit_threshold=0.0,
        min_hold_periods=3,
        initial_capital=400.0,
    )

    result = run_carry_portfolio({"BTC/USDT:USDT": frame}, cfg)
    equity_curve = result["equity_curve"]
    metrics = result["metrics"]

    assert metrics["MaxDD %"] > 0.0
    assert metrics["Total MTM PnL"] != 0.0
    # Phải có ít nhất một nến equity giảm so với nến trước.
    assert (equity_curve.diff().dropna() < 0.0).any()


def test_mtm_accounting_reconciles() -> None:
    """Bảo toàn kế toán: final = initial + funding + MTM - fees (spot_perp_2leg)."""
    n = 200
    dates = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1h", tz="UTC").as_unit("ns")
    perp_close = 100.0 * (1.0 + 0.005 * np.sin(np.arange(n) / 7.0))
    spot_close = np.full(n, 100.0)
    basis_pct = (perp_close - spot_close) / spot_close
    funding_event = np.where(dates.hour % 8 == 0, 0.0002, 0.0)

    frame = pd.DataFrame(
        {
            "open": perp_close,
            "high": perp_close,
            "low": perp_close,
            "close": perp_close,
            "volume": 1000.0,
            "spot_close": spot_close,
            "perp_close": perp_close,
            "basis_pct": basis_pct,
            "funding_rate_event": funding_event,
            "last_known_funding_rate": 0.0002,
        },
        index=dates,
    )
    frame.index.name = "datetime"

    cfg = CarryConfig(
        structure="spot_perp_2leg",
        basis_noise_std=0.0,
        entry_threshold=0.0001,
        exit_threshold=0.0,
        min_hold_periods=3,
        initial_capital=400.0,
    )

    result = run_carry_portfolio({"BTC/USDT:USDT": frame}, cfg)
    metrics = result["metrics"]
    final_equity = float(result["equity_curve"].iloc[-1])

    expected = (
        cfg.initial_capital
        + metrics["Total Funding Collected"]
        + metrics["Total MTM PnL"]
        - metrics["Total Fees Paid"]
    )
    assert final_equity == pytest.approx(expected, rel=1e-9)


def test_margin_rebalance_triggers_on_price_spike() -> None:
    """Giá spot tăng sốc >40% sau khi mở phải kích hoạt rebalance do giá."""
    n = 100
    dates = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1h", tz="UTC").as_unit("ns")
    funding_event = np.where(dates.hour % 8 == 0, 0.0002, 0.0)

    def _make_frame(price: np.ndarray) -> pd.DataFrame:
        f = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
                "spot_close": price,
                "perp_close": price,
                "basis_pct": np.zeros(n),
                "funding_rate_event": funding_event,
                "last_known_funding_rate": 0.0002,
            },
            index=dates,
        )
        f.index.name = "datetime"
        return f

    cfg = CarryConfig(
        structure="spot_perp_2leg",
        basis_noise_std=0.0,
        entry_threshold=0.0001,
        exit_threshold=0.0,
        min_hold_periods=3,
        initial_capital=400.0,
    )

    # Kịch bản sốc giá: giữ 100 đến nến 9, sau đó nhảy lên 150 và giữ nguyên.
    spiked_price = np.full(n, 100.0)
    spiked_price[10:] = 150.0
    spiked = run_carry_portfolio({"BTC/USDT:USDT": _make_frame(spiked_price)}, cfg)

    # Kịch bản đối chứng: giá phẳng 100 suốt.
    flat = run_carry_portfolio({"BTC/USDT:USDT": _make_frame(np.full(n, 100.0))}, cfg)

    trades_df = spiked["trades_df"]
    has_rebalance_row = bool(
        "action" in trades_df.columns and (trades_df["action"] == "rebalance").any()
    )

    assert spiked["metrics"]["Price Rebalances"] >= 1 or has_rebalance_row
    assert spiked["metrics"]["Total Fees Paid"] > flat["metrics"]["Total Fees Paid"]


def test_coinm_basis_mtm() -> None:
    """Coin-M 1 chân: basis mở rộng (perp cao hơn spot) làm short perp lỗ MTM."""
    n = 100
    dates = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1h", tz="UTC").as_unit("ns")
    basis_pct = np.linspace(0.0, 0.005, n)
    funding_event = np.where(dates.hour % 8 == 0, 0.0002, 0.0)

    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
            "basis_pct": basis_pct,
            "funding_rate_event": funding_event,
            "last_known_funding_rate": 0.0002,
        },
        index=dates,
    )
    frame.index.name = "datetime"

    cfg = CarryConfig(
        structure="coinm_1leg",
        basis_noise_std=0.0,
        entry_threshold=0.0001,
        exit_threshold=0.0,
        min_hold_periods=3,
        initial_capital=400.0,
    )

    result = run_carry_portfolio({"BTC/USDT:USDT": frame}, cfg)
    metrics = result["metrics"]

    assert metrics["MaxDD %"] > 0.0
    assert metrics["Total MTM PnL"] < 0.0


def test_passive_long_term_mode() -> None:
    """Passive long-term: mở một lần, giữ >=21 kỳ funding rồi thoát khi funding âm."""
    n = 300
    dates = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1h", tz="UTC").as_unit("ns")

    funding_event = np.zeros(n)
    last_known = np.zeros(n)
    for i in range(n):
        is_anchor = i % 8 == 0
        rate = 0.0002 if i < 184 else -0.0002
        if is_anchor:
            funding_event[i] = rate
        last_known[i] = 0.0002 if i < 184 else -0.0002

    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
            "funding_rate_event": funding_event,
            "last_known_funding_rate": last_known,
        },
        index=dates,
    )
    frame.index.name = "datetime"

    cfg = CarryConfig(
        structure="coinm_1leg",
        carry_mode="passive_long_term",
        passive_entry_ema_span=21,
        passive_exit_ema_span=42,
        passive_entry_threshold=5e-5,
        passive_exit_threshold=-2e-5,
        passive_min_hold_periods=21,
        initial_capital=400.0,
        basis_noise_std=0.0,
    )

    result = run_carry_portfolio({"BTC/USDT:USDT": frame}, cfg)
    trades_df = result["trades_df"]

    opens = trades_df[trades_df["action"] == "open"]
    closes = trades_df[trades_df["action"] == "close"]
    assert len(opens) <= 2
    assert len(closes) >= 1
    # Vị thế được giữ đủ số kỳ funding tối thiểu trước khi đóng.
    assert result["metrics"]["Total Funding Collected"] > 0.0


class FakeSpotExchange:
    """Sàn Spot giả lập trả tối đa 2 nến mỗi lần gọi để kiểm tra phân trang."""

    def __init__(self, records: list[list[float]], server_time_ms: int) -> None:
        self.records = records
        self.server_time_ms = server_time_ms
        self.call_history: list[dict[str, Any]] = []

    def milliseconds(self) -> int:
        return self.server_time_ms

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        self.call_history.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )
        filtered = [r for r in self.records if since is None or int(r[0]) >= since]
        cap = 2 if limit is None else min(2, int(limit))
        return [list(r) for r in filtered[:cap]]


def test_fetch_spot_ohlcv_pagination_and_cache(tmp_path: Path) -> None:
    """fetch_spot_ohlcv phân trang đúng và lần gọi thứ hai đọc hoàn toàn từ cache."""
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    records = [
        [
            int((start + pd.Timedelta(hours=i)).timestamp() * 1000),
            100.0 + i,
            101.0 + i,
            99.0 + i,
            100.5 + i,
            10.0 + i,
        ]
        for i in range(6)
    ]
    server_ms = int((start + pd.Timedelta(hours=7)).timestamp() * 1000)

    fake = FakeSpotExchange(records, server_time_ms=server_ms)
    df1 = fetch_spot_ohlcv(
        "BTC/USDT",
        data_dir=str(tmp_path),
        exchange=fake,
        start_date="2024-01-01 00:00:00",
        limit_per_call=2,
    )

    assert len(df1) == 6
    assert len(fake.call_history) >= 3  # Đã phân trang nhiều lần
    assert (tmp_path / "BTCUSDT_SPOT_1h.parquet").exists()

    fake2 = FakeSpotExchange(records, server_time_ms=server_ms)
    df2 = fetch_spot_ohlcv(
        "BTC/USDT",
        data_dir=str(tmp_path),
        exchange=fake2,
        start_date="2024-01-01 00:00:00",
        limit_per_call=2,
    )

    assert len(fake2.call_history) == 0  # Cache đã bao phủ đủ dải yêu cầu
    assert len(df2) == 6
    assert all(str(dtype) == "float64" for dtype in df2.dtypes)
    assert str(df2.index.tz) == "UTC"


def test_align_spot_perp_funding_no_lookahead() -> None:
    """Đồng bộ Spot/Perp/Funding: inner join chính xác và không nhìn trước funding."""
    perp_dates = pd.date_range(
        "2024-01-01 00:00:00", periods=24, freq="1h", tz="UTC"
    ).as_unit("ns")
    spot_dates = pd.date_range(
        "2024-01-01 01:00:00", periods=24, freq="1h", tz="UTC"
    ).as_unit("ns")

    perp_close = 100.0 + np.arange(len(perp_dates)) * 0.1
    spot_close = 99.0 + np.arange(len(spot_dates)) * 0.1

    perp_df = pd.DataFrame(
        {
            "open": perp_close,
            "high": perp_close + 0.5,
            "low": perp_close - 0.5,
            "close": perp_close,
            "volume": 1000.0,
        },
        index=perp_dates,
    )
    spot_df = pd.DataFrame(
        {
            "open": spot_close,
            "high": spot_close + 0.5,
            "low": spot_close - 0.5,
            "close": spot_close,
            "volume": 500.0,
        },
        index=spot_dates,
    )

    funding_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 08:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 16:00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 00:00:00", tz="UTC"),
        ]
    ).as_unit("ns")
    funding_df = pd.DataFrame(
        {
            "datetime": funding_times,
            "funding_rate": [0.0001, 0.0002, 0.0003, 0.0004],
        }
    )

    aligned = align_spot_perp_funding(perp_df, spot_df, funding_df)

    # Chỉ giữ giao timestamp chính xác (01:00..23:00), không ffill phần lệch.
    expected_index = perp_dates.intersection(spot_dates)
    assert len(aligned) == len(expected_index) == 23
    assert aligned.index.equals(expected_index)
    assert aligned.index.name == "datetime"
    assert str(aligned.index.tz) == "UTC"
    assert pd.Timestamp("2024-01-01 00:00:00", tz="UTC") not in aligned.index
    assert pd.Timestamp("2024-01-02 00:00:00", tz="UTC") not in aligned.index

    # Giữ đủ cột OHLCV của perp và tính basis_pct chính xác.
    for col in ("open", "high", "low", "close", "volume"):
        assert col in aligned.columns
    expected_basis = (aligned["perp_close"] - aligned["spot_close"]) / aligned["spot_close"]
    pd.testing.assert_series_equal(
        aligned["basis_pct"], expected_basis.astype("float64"), check_names=False
    )

    # funding_rate_event chỉ khác 0 tại đúng các nến 00/08/16.
    for ts, event_rate in aligned["funding_rate_event"].items():
        if ts.hour in (0, 8, 16) and ts.minute == 0:
            assert event_rate == pytest.approx(
                0.0001 if ts.hour == 0 else (0.0002 if ts.hour == 8 else 0.0003)
            )
        else:
            assert event_rate == 0.0

    # Ngay trước mốc funding thứ hai (08:00), chỉ được biết funding thứ nhất.
    t_07 = pd.Timestamp("2024-01-01 07:00:00", tz="UTC")
    assert float(aligned.loc[t_07, "last_known_funding_rate"]) == pytest.approx(0.0001)
    t_08 = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    assert float(aligned.loc[t_08, "last_known_funding_rate"]) == pytest.approx(0.0002)
