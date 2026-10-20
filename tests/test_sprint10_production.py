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


# ===========================================================================
# Sprint 10 — Chặng 2: SNR Regime Gate, Volatility Targeting, DD Cushion
# ===========================================================================
from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio  # noqa: E402

OVERLAY_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
]


def _overlay_index(periods: int) -> pd.DatetimeIndex:
    """DatetimeIndex 1h UTC ns tên ``datetime`` cho dữ liệu overlay."""
    index = pd.date_range(
        "2024-01-01", periods=periods, freq="h", tz="UTC"
    ).as_unit("ns")
    index.name = "datetime"
    return index


def _overlay_frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    """Dựng DataFrame đủ schema cho cả Carry và Overlay với funding/spread = 0.

    ``cross_ex_spread``, ``max_single_funding`` và toàn bộ funding event đều
    bằng 0 nên Core Carry không bao giờ mở vị thế; nhờ vậy mọi biến động vốn
    đo được đều quy về Tactical Overlay. Không có ``best_short_ex`` để chắc
    chắn Chế độ B cũng không có ứng viên.
    """
    close_arr = np.asarray(close, dtype="float64")
    n = len(close_arr)
    zeros = lambda: np.zeros(n, dtype="float64")  # noqa: E731
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(n, 1000.0, dtype="float64"),
            "perp_close": close_arr,
            "spot_close": close_arr,
            "basis_pct": zeros(),
            "funding_event_binance": zeros(),
            "funding_event_bybit": zeros(),
            "ewma_funding_binance": zeros(),
            "ewma_funding_bybit": zeros(),
            "max_single_funding": zeros(),
            "min_single_funding": zeros(),
            "cross_ex_spread": zeros(),
        },
        index=index.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _price_path(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    """Ghép chuỗi giá từ returns 1h cho trước (giá nến 0 = ``start*(1+r0)``)."""
    return start * np.cumprod(1.0 + np.asarray(returns, dtype="float64"))


def _four_symbol_data(
    close_map: dict[str, np.ndarray],
    index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Đóng gói 4 symbol Core-4 thành dict dữ liệu đã đồng bộ."""
    return {sym: _overlay_frame(index, close_map[sym]) for sym in OVERLAY_SYMBOLS}


def _trend_overlay_data(periods: int, noise_amp: float) -> dict[str, pd.DataFrame]:
    """Trend phân hóa mạnh để SNR gate bật, nhiễu xen kẽ đặt ngược pha.

    Coin 0 tăng 0.05%/h, coin 3 giảm 0.05%/h, 2 coin giữa đi ngang. Nhiễu
    ``±noise_amp`` xen kẽ và ngược dấu giữa chân long và chân short khiến độ
    lệch chuẩn của ``r_long - r_short`` tỷ lệ thuận với ``noise_amp`` - đúng
    biến mà Volatility Targeting dùng để nghịch đảo đòn bẩy.
    """
    index = _overlay_index(periods)
    steps = np.arange(periods, dtype="float64")
    alternator = np.where(steps % 2 == 0, 1.0, -1.0)
    base_noise = noise_amp * alternator

    close_map = {
        OVERLAY_SYMBOLS[0]: _price_path(0.0005 + base_noise),
        OVERLAY_SYMBOLS[3]: _price_path(-0.0005 - base_noise),
        OVERLAY_SYMBOLS[1]: _price_path(0.1 * base_noise),
        OVERLAY_SYMBOLS[2]: _price_path(-0.1 * base_noise),
    }
    return _four_symbol_data(close_map, index)


def _reversal_overlay_data(
    periods: int = 1600,
    reversal_start: int = 1200,
    noise_amp: float = 0.0005,
) -> dict[str, pd.DataFrame]:
    """Trend phân hóa mạnh 1200 nến rồi đảo chiều 0.5%/h trong phần còn lại.

    Cú đảo chiều tạo drawdown sâu cho cặp long-short đang giữ vị thế cũ, đủ để
    kích hoạt cả mức cắt 50% (DD > 4%) lẫn mức tắt hẳn (DD > 6%).
    """
    index = _overlay_index(periods)
    steps = np.arange(periods, dtype="float64")
    alternator = np.where(steps % 2 == 0, 1.0, -1.0)
    base_noise = noise_amp * alternator

    drift_long = np.where(steps < reversal_start, 0.0005, -0.005)
    drift_short = np.where(steps < reversal_start, -0.0005, 0.005)

    close_map = {
        OVERLAY_SYMBOLS[0]: _price_path(drift_long + base_noise),
        OVERLAY_SYMBOLS[3]: _price_path(drift_short - base_noise),
        OVERLAY_SYMBOLS[1]: _price_path(0.1 * base_noise),
        OVERLAY_SYMBOLS[2]: _price_path(-0.1 * base_noise),
    }
    return _four_symbol_data(close_map, index)


def _barbell_config(**overrides: object) -> BarbellConfig:
    """Config Barbell tất định: Carry tắt giao dịch, các gate bật mặc định."""
    base: dict[str, object] = {
        "initial_capital": INITIAL_CAPITAL,
        "carry_config": _carry_config(),
        "overlay_lookback_hours": 336,
        "overlay_rebalance_hours": 48,
    }
    base.update(overrides)
    return BarbellConfig(**base)


# ---------------------------------------------------------------------------
# 7. SNR Gate phải đóng hoàn toàn trong thị trường đi ngang
# ---------------------------------------------------------------------------
def test_sideways_market_disables_overlay_by_snr_gate() -> None:
    """Đi ngang: SNR dispersion < 1.35 nên gate khóa overlay, equity = Carry.

    Đây là bằng chứng lỗi "z-score hằng-đúng" đã bị diệt: 4 coin cùng một
    sóng sin biên độ 1e-4 và pha lệch rất ít, dispersion gần 0. Vì vậy
    ``overlay_active_ratio`` phải bằng đúng 0 và đường cong vốn Barbell phải
    trùng khít đường cong vốn Carry (overlay không được cấp đòn bẩy).
    """
    n = 1500
    index = _overlay_index(n)
    steps = np.arange(n, dtype="float64")

    close_map: dict[str, np.ndarray] = {}
    for i, sym in enumerate(OVERLAY_SYMBOLS):
        phase = 0.05 * i
        sin_wave = np.sin(2.0 * np.pi * steps / 24.0 + phase)
        close_map[sym] = 100.0 * (1.0 + 1e-4 * sin_wave)

    data = _four_symbol_data(close_map, index)
    config = _barbell_config(
        overlay_lookback_hours=336,
        overlay_rebalance_hours=48,
        overlay_min_snr_dispersion=1.35,
        overlay_exit_snr_dispersion=0.95,
    )
    res = run_barbell_portfolio(data, config)

    assert res["overlay_active_ratio"] == pytest.approx(0.0)

    np.testing.assert_allclose(
        res["equity_curve"].to_numpy(dtype="float64"),
        res["carry_result"]["equity_curve"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-9,
    )

    dispersion = res["snr_dispersion"].dropna()
    assert not dispersion.empty, "Cần ít nhất một nến có SNR hợp lệ để kiểm tra"
    assert float(dispersion.max()) < 1.35


# ---------------------------------------------------------------------------
# 8. Volatility Targeting: pair vol gấp đôi phải hạ đòn bẩy
# ---------------------------------------------------------------------------
def test_vol_targeting_reduces_leverage_when_pair_vol_doubles() -> None:
    """Biến động cặp long-short gấp đôi phải kéo leverage trung bình xuống < 85%.

    Hai dataset cùng trend phân hóa (gate ON) và chỉ khác biên độ nhiễu xen kẽ
    (±0.05% so với ±0.10%). Vol Targeting nghịch đảo vol nên B phải bị hạ đòn
    bẩy rõ rệt; mọi giá trị active vẫn nằm trong dải [0.05, 0.45].
    """
    n = 2000
    data_a = _trend_overlay_data(n, noise_amp=0.0005)
    data_b = _trend_overlay_data(n, noise_amp=0.0010)

    config = _barbell_config(
        enable_overlay=True,
        enable_snr_gate=True,
        enable_vol_targeting=True,
        overlay_target_vol_ann=0.065,
        overlay_vol_lookback=168,
        overlay_leverage_min=0.05,
        overlay_leverage_max=0.45,
        overlay_lookback_hours=336,
        overlay_rebalance_hours=48,
    )
    res_a = run_barbell_portfolio(data_a, config)
    res_b = run_barbell_portfolio(data_b, config)

    lev_a = res_a["overlay_leverage_series"]
    lev_b = res_b["overlay_leverage_series"]

    # Gate phải bật ở cả hai dataset, nếu không phép so sánh là vô nghĩa.
    assert res_a["metrics"]["overlay_gate_on_ratio"] > 0.0
    assert res_b["metrics"]["overlay_gate_on_ratio"] > 0.0

    assert float(lev_a.mean()) > 0.0
    assert float(lev_b.mean()) < 0.85 * float(lev_a.mean())

    for series in (lev_a, lev_b):
        active = series[series > 1e-12]
        assert not active.empty
        assert float(active.min()) >= 0.05 - 1e-12
        assert float(active.max()) <= 0.45 + 1e-12
        assert 0.05 <= float(series.mean()) <= 0.45


# ---------------------------------------------------------------------------
# 9. Drawdown Cushion: cắt rồi tắt hẳn đòn bẩy sau cú lỗ lớn
# ---------------------------------------------------------------------------
def test_drawdown_cushion_scales_down_leverage_after_loss() -> None:
    """Đảo chiều mạnh sau 1200 nến phải kích hoạt DD Cushion và tắt hẳn leverage.

    Giai đoạn đầu trend phân hóa giữ gate ON và leverage dương. Cú đảo chiều
    0.5%/h làm equity Barbell sụt quá 4% rồi quá 6% so với đỉnh, nên
    ``drawdown_cushion_events`` > 0 và đòn bẩy overlay về đúng 0.0 sau mốc đảo
    chiều, trong khi giai đoạn trước đó vẫn có leverage > 0.
    """
    n = 1600
    reversal_start = 1200
    data = _reversal_overlay_data(periods=n, reversal_start=reversal_start)
    config = _barbell_config(
        enable_overlay=True,
        enable_snr_gate=True,
        enable_vol_targeting=True,
        enable_drawdown_cushion=True,
        overlay_dd_cut_threshold=0.04,
        overlay_dd_off_threshold=0.06,
        overlay_target_vol_ann=0.065,
        overlay_vol_lookback=168,
        overlay_leverage_min=0.05,
        overlay_leverage_max=0.45,
        overlay_lookback_hours=336,
        overlay_rebalance_hours=48,
    )
    res = run_barbell_portfolio(data, config)

    metrics = res["metrics"]
    # Tên key thực tế trong src/barbell_portfolio.py là 'drawdown_cushion_events'.
    assert metrics["drawdown_cushion_events"] > 0

    lev = res["overlay_leverage_series"]
    assert float(lev.max()) > 0.0
    assert float(lev.min()) == pytest.approx(0.0)
    # Sau cú đảo chiều, cushion phải tắt hẳn overlay (leverage = 0 tuyệt đối).
    assert float(lev.iloc[reversal_start:].min()) == pytest.approx(0.0)
    assert float(lev.iloc[:reversal_start].max()) > 0.0


# ===========================================================================
# Sprint 10 — Chặng 3: Execute Live (quantize + hedge 2 sàn), cancel orphan,
# Barbell allocations và persist sleeve state
# ===========================================================================
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

from src.carry_live_executor import CarryLiveExecutor  # noqa: E402
from src.dynamic_carry_basis import DynamicCarryConfig  # noqa: E402
from src.state_store import StateStore  # noqa: E402

BTC = "BTC/USDT:USDT"
BTC_SPOT = "BTC/USDT"


class FakeExchange:
    """Sàn giả thuần Python cho test thực thi, không gọi mạng.

    Ghi lại lệnh vào ``orders`` và lệnh hủy vào ``cancelled`` để test kiểm tra
    đúng sàn nhận lệnh. ``order_status`` điều khiển kết quả ``fetch_order``
    (open/partial/closed/canceled/rejected). Order book mặc định bid 99.9 /
    ask 100.1; test nào cần giá lớn hơn có thể truyền ``bids``/``asks`` riêng.
    """

    def __init__(
        self,
        order_status: dict[str, Any] | None = None,
        bids: Sequence[Sequence[float]] | None = None,
        asks: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self.order_status: dict[str, Any] = (
            dict(order_status)
            if order_status is not None
            else {"id": "leg1-1", "status": "open", "filled": 0.0}
        )
        self.bids = [list(level) for level in (bids or [[99.9, 10]])]
        self.asks = [list(level) for level in (asks or [[100.1, 10]])]

    def fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """Order book tĩnh: bid 99.9 / ask 100.1 theo mặc định."""
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
        """Ghi nhận lệnh; limit trả open ``leg1-1``, market trả closed đã khớp đủ."""
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
        """Trả trạng thái do test điều khiển qua ``self.order_status``."""
        return dict(self.order_status)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        """Ghi nhận lệnh hủy để kiểm tra chống lệnh mồ côi."""
        self.cancelled.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}


def _s10c3_executor(
    state_path: Path,
    exchange_map: dict[str, Any],
    mode: str = "live",
    overlay_config: BarbellConfig | None = None,
) -> CarryLiveExecutor:
    """Tạo executor offline với state file nằm trong ``tmp_path``."""
    return CarryLiveExecutor(
        carry_config=DynamicCarryConfig(),
        mode=mode,  # type: ignore[arg-type]
        capital=3000.0,
        exchange_map=exchange_map,
        state_store=StateStore(state_path),
        execution_style="maker_taker_hedge",
        overlay_config=overlay_config,
    )


def _s10c3_universe(periods: int = 1200) -> dict[str, pd.DataFrame]:
    """Dựng 4 symbol Core-4 trend phân hóa mạnh kèm nhiễu xen kẽ rất nhỏ.

    Drift lần lượt +0.05%, +0.01%, -0.01%, -0.05% mỗi giờ. Nhiễu xen kẽ biên
    độ 0.02%/h chỉ để ``sigma_1h`` khác 0 (trend tuyến tính tuyệt đối sẽ khiến
    SNR = NaN vì chia cho 0), không làm đổi hướng trend. Nhờ đó
    ``raw_snr_dispersion`` rất lớn và SNR gate mở chắc chắn.
    """
    index = _overlay_index(periods)
    steps = np.arange(periods, dtype="float64")
    noise = 2e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    drifts = (0.0005, 0.0001, -0.0001, -0.0005)
    starts = (65000.0, 3000.0, 150.0, 600.0)

    return {
        sym: _overlay_frame(index, _price_path(drift + noise, start))
        for sym, drift, start in zip(OVERLAY_SYMBOLS, drifts, starts)
    }


def _s10c3_last_price(df: pd.DataFrame) -> float:
    """Giá perp cuối cùng dùng cho lượng hóa khối lượng."""
    return float(df["perp_close"].iloc[-1])


def _s10c3_assert_step_grid(symbol: str, qty: float) -> None:
    """Khối lượng phải nằm đúng lưới ``step_size`` của sàn (sai số 1e-6)."""
    step = EXCHANGE_LOT_SPECS[symbol]["step_size"]
    units = qty / step
    assert units == pytest.approx(round(units), abs=1e-6)
    assert qty > 0.0


# ---------------------------------------------------------------------------
# 10. Lượng hóa Chân 1 và hedge đúng khối lượng trên sàn thứ hai
# ---------------------------------------------------------------------------
def test_executor_quantizes_leg1_and_hedges_same_qty_on_second_venue(
    tmp_path: Path,
) -> None:
    """0.0187 BTC @ 65000 làm tròn 0.018; hedge market đúng 0.018 trên bybit.

    Chân 1 Post-Only nằm trên ``binance``, Chân 2 phải bắn trên ``bybit``
    (không phải binance) với khối lượng bằng đúng phần đã khớp, không hedge
    thừa và không bắn lệnh market trên sàn Chân 1.
    """
    fake1 = FakeExchange(bids=[[64999.9, 10]], asks=[[65000.1, 10]])
    fake2 = FakeExchange()
    executor = _s10c3_executor(
        tmp_path / "state_hedge.json",
        {"binance": fake1, "bybit": fake2},
    )

    leg1_res = executor.place_maker_leg1(
        BTC,
        "sell",
        0.0187,
        65000.0,
        exchange_short="binance",
        exchange_long="bybit",
        spot_symbol=BTC_SPOT,
    )

    limit_orders = [o for o in fake1.orders if o["type"] == "limit"]
    assert len(limit_orders) == 1
    assert limit_orders[0]["amount"] == pytest.approx(0.018)
    assert leg1_res["target_qty"] == pytest.approx(0.018)
    assert leg1_res["leg1_status"] == "PENDING_MAKER"
    assert fake2.orders == []

    fake1.order_status = {"id": "leg1-1", "status": "closed", "filled": 0.018}
    verify_res = executor.verify_maker_fill_and_hedge_leg2(
        BTC,
        exchange_short="binance",
        exchange_long="bybit",
        spot_symbol=BTC_SPOT,
    )

    assert verify_res["leg1_filled_qty"] == pytest.approx(0.018)
    assert verify_res["leg2_hedged_qty"] == pytest.approx(0.018)
    assert [o for o in fake1.orders if o["type"] == "market"] == []

    bybit_markets = [o for o in fake2.orders if o["type"] == "market"]
    assert len(bybit_markets) == 1
    assert bybit_markets[0]["amount"] == pytest.approx(0.018)
    assert bybit_markets[0]["side"] == "buy"


# ---------------------------------------------------------------------------
# 11. Chặn lệnh dưới min-notional ngay tại tầng thực thi
# ---------------------------------------------------------------------------
def test_executor_rejects_order_below_min_notional(tmp_path: Path) -> None:
    """1e-7 BTC @ 65000 chỉ đáng 0.0065 USDT nên bị REJECTED, không gửi lệnh.

    Executor phải trả ``error='below_min_notional'``, không tạo lệnh limit trên
    sàn, không bắn market hedge, và giữ vị thế đang theo dõi ở trạng thái IDLE
    thay vì để lại state rác.
    """
    fake1 = FakeExchange()
    fake2 = FakeExchange()
    executor = _s10c3_executor(
        tmp_path / "state_reject.json",
        {"binance": fake1, "binance_spot": fake2},
    )
    # Mô phỏng symbol đã được executor theo dõi từ chu kỳ trước.
    executor.state.carry_positions[BTC] = {
        "symbol": BTC,
        "spot_symbol": BTC_SPOT,
        "qty": 0.0,
        "leg1_status": "NONE",
        "leg1_filled_qty": 0.0,
        "leg2_hedged_qty": 0.0,
        "execution_state": "IDLE",
    }

    res = executor.place_maker_leg1(
        BTC,
        "sell",
        1e-7,
        65000.0,
        exchange_short="binance",
        exchange_long="binance_spot",
        spot_symbol=BTC_SPOT,
    )

    assert res["leg1_status"] == "REJECTED"
    assert res["error"] == "below_min_notional"
    assert res["target_qty"] == pytest.approx(0.0)
    assert fake1.orders == []
    assert [o for o in fake2.orders if o["type"] == "market"] == []
    assert executor.state.carry_positions[BTC]["execution_state"] == "IDLE"
    assert executor.state.carry_positions[BTC]["leg1_status"] == "REJECTED"


# ---------------------------------------------------------------------------
# 12. Hủy lệnh mồ côi trên đúng sàn Chân 1 (short venue)
# ---------------------------------------------------------------------------
def test_executor_cancels_orphan_order_on_short_venue(tmp_path: Path) -> None:
    """Limit chưa khớp trên bybit phải bị hủy tại bybit, không đụng binance.

    Rủi ro thực chiến: bỏ quên lệnh treo sẽ tạo vị thế trần trụi khi giá quét
    qua. Hủy phải gọi đúng sàn đã đặt Chân 1 (``bybit``), không phải sàn mặc
    định ``binance``, và không bắn market hedge.
    """
    fake1 = FakeExchange()
    fake2 = FakeExchange(
        order_status={"id": "leg1-1", "status": "open", "filled": 0.0}
    )
    executor = _s10c3_executor(
        tmp_path / "state_cancel.json",
        {"binance": fake1, "bybit": fake2},
    )

    leg1_res = executor.place_maker_leg1(
        BTC,
        "sell",
        1.0,
        100.0,
        exchange_short="bybit",
        exchange_long="binance_spot",
        spot_symbol=BTC_SPOT,
    )
    assert leg1_res["leg1_order_id"] == "leg1-1"
    assert [o for o in fake2.orders if o["type"] == "limit"] != []

    assert executor.cancel_maker_leg1(BTC, exchange_name="bybit") is True
    assert fake2.cancelled == [{"id": "leg1-1", "symbol": BTC}]
    assert fake1.cancelled == []
    assert executor.state.carry_positions[BTC]["leg1_status"] == "CANCELED"
    assert [o for o in fake1.orders if o["type"] == "market"] == []
    assert [o for o in fake2.orders if o["type"] == "market"] == []


# ---------------------------------------------------------------------------
# 13. Phân bổ Barbell: lượng hóa carry và mở overlay long/short khi gate ON
# ---------------------------------------------------------------------------
def test_build_barbell_allocations_quantizes_and_gates_overlay(
    tmp_path: Path,
) -> None:
    """Trend phân hóa mạnh mở SNR gate; carry 2 mã và overlay 1 long 1 short.

    Mọi ``target_qty`` phải nằm đúng lưới step của cặp (BTC 0.001 / ETH 0.01 /
    SOL 0.1 / BNB 0.01), notional carry >= 10 USDT, và overlay phải có đúng một
    chân long và một chân short sau khi lượng hóa.
    """
    universe = _s10c3_universe(periods=1200)
    overlay_config = BarbellConfig(
        initial_capital=3000.0,
        overlay_min_snr_dispersion=0.5,
        overlay_lookback_hours=336,
    )
    executor = _s10c3_executor(
        tmp_path / "state_alloc.json",
        {"binance": FakeExchange()},
        mode="paper",
        overlay_config=overlay_config,
    )
    scores_at_t = {sym: 1.0 for sym in OVERLAY_SYMBOLS}

    alloc = executor.build_barbell_allocations(universe, scores_at_t)

    assert alloc["overlay_active"] is True
    assert alloc["raw_snr_dispersion"] >= 0.5

    carry_allocs = alloc["carry_sleeve"]["allocations"]
    assert len(carry_allocs) == 2
    for item in carry_allocs:
        _s10c3_assert_step_grid(item["symbol"], item["target_qty"])
        assert item["notional"] >= 10.0
        price = _s10c3_last_price(universe[item["symbol"]])
        assert item["notional"] == pytest.approx(item["target_qty"] * price)

    overlay_allocs = alloc["momentum_overlay"]["allocations"]
    assert len(overlay_allocs) == 2
    sides = [str(item["side"]) for item in overlay_allocs]
    assert sum("long" in side for side in sides) == 1
    assert sum("short" in side for side in sides) == 1
    for item in overlay_allocs:
        _s10c3_assert_step_grid(item["symbol"], item["target_qty"])
        price = _s10c3_last_price(universe[item["symbol"]])
        assert item["notional"] == pytest.approx(item["target_qty"] * price)


# ---------------------------------------------------------------------------
# 14. Persist carry_sleeve / momentum_overlay sống sót qua vòng save -> load
# ---------------------------------------------------------------------------
def test_persist_sleeve_state_roundtrip(tmp_path: Path) -> None:
    """Persist gắn timestamp, load lại giữ overlay_active và không xóa carry_positions."""
    state_path = tmp_path / "state.json"
    executor = _s10c3_executor(state_path, {}, mode="paper")
    executor.state.carry_positions[BTC] = {
        "symbol": BTC,
        "spot_symbol": BTC_SPOT,
        "qty": 0.5,
        "leg1_status": "FILLED",
        "leg1_filled_qty": 0.5,
        "leg2_hedged_qty": 0.5,
        "execution_state": "HEDGED",
    }

    timestamp = "2026-10-19T00:00:00+00:00"
    allocations = {
        "carry_sleeve": {
            "allocations": [
                {
                    "symbol": BTC,
                    "side": "short_perp",
                    "target_qty": 0.01,
                    "notional": 650.0,
                }
            ],
            "total_notional": 650.0,
        },
        "momentum_overlay": {
            "allocations": [
                {
                    "symbol": "ETH/USDT:USDT",
                    "side": "long_perp",
                    "target_qty": 0.1,
                    "notional": 300.0,
                }
            ],
            "total_notional": 300.0,
        },
        "raw_snr_dispersion": 2.0,
        "overlay_active": True,
    }

    executor.persist_sleeve_state(allocations, timestamp)
    executor.state_store.save(executor.state)

    loaded = StateStore(state_path).load(default_equity=3000.0)

    assert loaded.carry_sleeve["timestamp"] == timestamp
    assert loaded.carry_sleeve["raw_snr_dispersion"] == pytest.approx(2.0)
    assert len(loaded.carry_sleeve["allocations"]) == 1
    assert loaded.momentum_overlay["timestamp"] == timestamp
    assert loaded.momentum_overlay["overlay_active"] is True
    assert len(loaded.momentum_overlay["allocations"]) == 1

    # carry_positions cũ không bị ghi đè bởi persist sleeve state.
    assert loaded.carry_positions[BTC]["execution_state"] == "HEDGED"
    assert loaded.carry_positions[BTC]["qty"] == pytest.approx(0.5)


# ===========================================================================
# Sprint 10 — Chặng 4: Attribution theo kỳ, GO/NO-GO 5 tiêu chí, executor
# parity và mô phỏng 4 Tier + bảng tổng hợp 21 cột
# ===========================================================================
from src.sprint10_eval import (  # noqa: E402
    build_sprint10_summary_rows,
    check_executor_parity,
    compute_period_contributions,
    evaluate_sprint10_go_no_go,
    run_sprint10_tiers,
)

SUMMARY_COLUMNS = (
    "tier",
    "tier_name",
    "final_equity",
    "annualized_return_pct",
    "sharpe",
    "maxdd_pct",
    "beta_btc",
    "net_pnl",
    "positive_periods",
    "total_periods",
    "holdout_annualized_pct",
    "worst_period_pct",
    "fees",
    "funding_pnl",
    "basis_pnl",
    "basis_mtm_std",
    "overlay_active_ratio",
    "overlay_mean_leverage",
    "snr_dispersion_mean",
    "verdict",
    "failed_conditions",
)


def _s10c4_attribution_index(periods: int = 10) -> pd.DatetimeIndex:
    """DatetimeIndex 1 giờ UTC dùng cho chuỗi attribution giả lập."""
    return pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")


def _s10c4_go_payload() -> dict[str, Any]:
    """Payload Tier 4 giả lập thỏa cả 5 tiêu chí GO của Sprint 10.

    ``periods`` có 7 kỳ: 6 kỳ +2.0% và kỳ holdout cuối -1.0% nhưng annualized
    +9.0% (vẫn qua ngưỡng 8%). ``attribution_series`` dương ở mọi thành phần
    trong kỳ holdout để cond2 (nhánh Barbell) đạt, kèm ``basis_mtm_std`` và
    ``overlay_active_ratio`` nằm trong dải hợp lệ cho cond4.
    """
    attr_index = _s10c4_attribution_index(8)
    attribution = pd.DataFrame(
        {
            "funding_pnl": np.full(8, 1.0, dtype="float64"),
            "basis_pnl": np.full(8, 2.0, dtype="float64"),
            "overlay_pnl": np.full(8, 3.0, dtype="float64"),
            "fees": np.full(8, 0.5, dtype="float64"),
            "net_pnl": np.full(8, 5.5, dtype="float64"),
        },
        index=attr_index,
    )

    net_returns = [2.0] * 6 + [-1.0]
    period_rows = []
    for i, net_return in enumerate(net_returns):
        is_holdout = i == len(net_returns) - 1
        period_rows.append(
            {
                "period": f"P{i + 1}",
                "period_id": f"P{i + 1}",
                "start": attr_index[i],
                "end": attr_index[i + 1],
                "net_return_pct": net_return,
                "annualized_return_pct": 9.0 if is_holdout else 12.0,
                "positive": net_return > 0.0,
            }
        )

    return {
        "tier_name": "Tier 4 — Production Barbell (test)",
        "row": {
            "annualized_return_pct": 12.0,
            "sharpe": 2.0,
            "maxdd_pct": 5.0,
            "beta_btc": 0.05,
        },
        "periods": pd.DataFrame(period_rows),
        "result": {
            "attribution_series": attribution,
            "basis_mtm_std": 0.1,
            "overlay_active_ratio": 0.5,
            "metrics": {"Margin Calls": 0},
        },
    }


def _s10c4_synthetic_frame(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    basis_pct: np.ndarray,
    funding_binance: float,
    funding_bybit: float,
) -> pd.DataFrame:
    """Dựng schema đa sàn đầy đủ cho 4 Tier, funding chỉ trả tại mốc 8h UTC.

    ``ewma_funding_*`` và ``max_single_funding`` giữ hằng số để yield kỳ vọng
    ổn định; ``funding_event_*`` chỉ khác 0 tại 00/08/16 UTC đúng quy ước.
    """
    close_arr = np.asarray(close, dtype="float64")
    n = len(close_arr)
    is_event = ((index.hour % 8) == 0) & (index.minute == 0)
    event_binance = np.where(is_event, funding_binance, 0.0)
    event_bybit = np.where(is_event, funding_bybit, 0.0)
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(n, 1000.0, dtype="float64"),
            "perp_close": close_arr,
            "spot_close": close_arr,
            "basis_pct": np.asarray(basis_pct, dtype="float64"),
            "funding_event_binance": event_binance,
            "funding_event_bybit": event_bybit,
            "ewma_funding_binance": np.full(n, funding_binance, dtype="float64"),
            "ewma_funding_bybit": np.full(n, funding_bybit, dtype="float64"),
            "max_single_funding": np.full(n, funding_binance, dtype="float64"),
            "min_single_funding": np.full(n, funding_bybit, dtype="float64"),
            "cross_ex_spread": np.full(n, funding_binance - funding_bybit, dtype="float64"),
        },
        index=index.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _s10c4_synthetic_data(periods: int = 700) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Dựng 4 symbol × ``periods`` nến 1h với basis sin ±10 bps và funding 2 sàn.

    Giá phân hóa nhẹ (drift ±1-2 bps/giờ) để maker Post-Only luôn khớp, funding
    Binance 1.5 bps / Bybit 0.5 bps đủ vượt ngưỡng entry 0.5 bps/8h. Trả về
    ``(multi_aligned_data, btc_benchmark_returns)``.
    """
    index = pd.date_range(
        "2024-01-01", periods=periods, freq="h", tz="UTC"
    ).as_unit("ns")
    index.name = "datetime"

    steps = np.arange(periods, dtype="float64")
    basis_pct = 0.001 * np.sin(2.0 * np.pi * steps / 48.0)

    starts = (65000.0, 3000.0, 150.0, 600.0)
    drifts = (0.0001, 0.00005, -0.00005, -0.0001)

    data: dict[str, pd.DataFrame] = {}
    close_map: dict[str, np.ndarray] = {}
    for symbol, start, drift in zip(OVERLAY_SYMBOLS, starts, drifts):
        close = start * np.cumprod(1.0 + np.full(periods, drift, dtype="float64"))
        close_map[symbol] = close
        data[symbol] = _s10c4_synthetic_frame(
            index,
            close,
            basis_pct,
            funding_binance=0.00015,
            funding_bybit=0.00005,
        )

    btc_returns = (
        pd.Series(close_map[OVERLAY_SYMBOLS[0]], index=index)
        .pct_change()
        .fillna(0.0)
        .astype("float64")
    )
    return data, btc_returns


# ---------------------------------------------------------------------------
# 15. Bóc tách đóng góp PnL theo cửa sổ thời gian [start, end)
# ---------------------------------------------------------------------------
def test_compute_period_contributions_slices_attribution_correctly() -> None:
    """Chỉ cộng các bar trong [index[2], index[5]) và chịu được cột thiếu.

    Mỗi bar có funding 1.0 / basis 2.0 / overlay 3.0 / fees 0.5 / net 5.5 nên
    cửa sổ 3 bar phải cho 3.0 / 6.0 / 9.0 / 1.5 / 16.5. Khi attribution chỉ có
    ``funding_pnl``, các thành phần còn lại trả 0.0 thay vì ném KeyError.
    """
    index = _s10c4_attribution_index(10)
    attribution = pd.DataFrame(
        {
            "funding_pnl": np.full(10, 1.0, dtype="float64"),
            "basis_pnl": np.full(10, 2.0, dtype="float64"),
            "overlay_pnl": np.full(10, 3.0, dtype="float64"),
            "fees": np.full(10, 0.5, dtype="float64"),
            "net_pnl": np.full(10, 5.5, dtype="float64"),
        },
        index=index,
    )

    contrib = compute_period_contributions(
        {"attribution_series": attribution},
        index[2],
        index[5],
    )

    assert contrib["funding_pnl"] == pytest.approx(3.0)
    assert contrib["basis_pnl"] == pytest.approx(6.0)
    assert contrib["overlay_pnl"] == pytest.approx(9.0)
    assert contrib["fees"] == pytest.approx(1.5)
    assert contrib["net_pnl"] == pytest.approx(16.5)

    partial = pd.DataFrame(
        {"funding_pnl": np.full(10, 1.0, dtype="float64")},
        index=index,
    )
    partial_contrib = compute_period_contributions(
        {"attribution_series": partial},
        index[2],
        index[5],
    )

    assert partial_contrib["funding_pnl"] == pytest.approx(3.0)
    assert partial_contrib["basis_pnl"] == pytest.approx(0.0)
    assert partial_contrib["overlay_pnl"] == pytest.approx(0.0)
    assert partial_contrib["fees"] == pytest.approx(0.0)
    assert partial_contrib["net_pnl"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 16. GO/NO-GO: đủ 5 tiêu chí -> GO; hạ Sharpe hoặc lệch regime -> NO-GO
# ---------------------------------------------------------------------------
def test_evaluate_sprint10_go_no_go_all_conditions_pass_and_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload chuẩn phải cho GO; mỗi vi phạm đơn lẻ phải chỉ đúng tiêu chí fail.

    Executor parity được mock để cô lập 4 tiêu chí định lượng. Bản sao hạ
    ``sharpe`` xuống 1.0 phải fail ``cond3_risk``; bản sao đẩy
    ``overlay_active_ratio`` lên 0.95 phải fail ``cond4_regime_and_mtm``.
    """
    monkeypatch.setattr(
        "src.sprint10_eval.check_executor_parity",
        lambda: {"passed": True, "checks": {}, "missing": []},
    )

    payload = _s10c4_go_payload()
    ok = evaluate_sprint10_go_no_go(payload, is_barbell=True)

    assert ok["verdict"] == "GO"
    assert ok["failed"] == []
    assert all(cond["passed"] for cond in ok["conditions"].values())

    bad_risk = _s10c4_go_payload()
    bad_risk["row"]["sharpe"] = 1.0
    risk_verdict = evaluate_sprint10_go_no_go(bad_risk, is_barbell=True)

    assert risk_verdict["verdict"] == "NO-GO"
    assert risk_verdict["failed"] == ["cond3_risk"]
    assert risk_verdict["conditions"]["cond3_risk"]["passed"] is False

    bad_regime = _s10c4_go_payload()
    bad_regime["row"]["overlay_active_ratio"] = 0.95
    regime_verdict = evaluate_sprint10_go_no_go(bad_regime, is_barbell=True)

    assert regime_verdict["verdict"] == "NO-GO"
    assert regime_verdict["failed"] == ["cond4_regime_and_mtm"]
    assert regime_verdict["conditions"]["cond4_regime_and_mtm"]["passed"] is False


# ---------------------------------------------------------------------------
# 17. Executor parity: đủ thành phần quantize/cancel/barbell/sleeve/overlay
# ---------------------------------------------------------------------------
def test_check_executor_parity_detects_required_components() -> None:
    """Quét source thực tế phải thấy đủ 5 thành phần GO-5 và không thiếu gì.

    Đây là kiểm tra tích hợp trên ``src/carry_live_executor.py`` và
    ``src/state_store.py``: thiếu bất kỳ hàm/khóa nào sẽ khiến ``passed`` False
    và tên đó xuất hiện trong ``missing``.
    """
    parity = check_executor_parity()

    assert parity["passed"] is True
    assert parity["missing"] == []

    checks = parity["checks"]
    for required in (
        "quantize_leg",
        "cancel_maker_leg1",
        "build_barbell_allocations",
        "carry_sleeve",
        "momentum_overlay",
    ):
        assert required in checks
        assert checks[required] is True


# ---------------------------------------------------------------------------
# 18. Chạy 4 Tier trên dữ liệu tổng hợp và xuất bảng 21 cột
# ---------------------------------------------------------------------------
def test_run_sprint10_tiers_and_summary_rows_on_synthetic_data() -> None:
    """4 Tier phải trả kỳ hợp lệ, bảng tổng hợp đủ 21 cột và True Basis MTM chạy.

    Dữ liệu 700 nến với basis sin ±10 bps để Tier 1 (single Binance) thực sự
    giữ vị thế và phát sinh Basis MTM; nhờ đó ``basis_mtm_std > 0`` chứng minh
    MTM không còn là hằng 0. Tier 4 phải có tỷ lệ overlay hoạt động trong
    [0, 1] và không dòng nào có ``final_equity`` NaN.
    """
    data, btc_returns = _s10c4_synthetic_data(periods=700)

    tiers = run_sprint10_tiers(
        data,
        btc_returns,
        initial_capital=3000.0,
        n_periods=2,
        period_days=14,
    )

    expected_tiers = {
        "tier1_single_binance",
        "tier2_best_venue",
        "tier3_hybrid_perp_perp",
        "tier4_production_barbell",
    }
    assert set(tiers) == expected_tiers

    for tier_key in expected_tiers:
        periods = tiers[tier_key]["periods"]
        assert isinstance(periods, pd.DataFrame)
        assert not periods.empty, f"{tier_key} phải có ít nhất một kỳ"
        assert len(periods) == 2

    rows = build_sprint10_summary_rows(tiers)
    assert len(rows) == 4
    assert len(SUMMARY_COLUMNS) == 21

    by_tier = {str(row["tier"]): row for row in rows}
    for row in rows:
        for column in SUMMARY_COLUMNS:
            assert column in row, f"Thiếu cột {column} ở {row.get('tier')}"
        assert np.isfinite(float(row["final_equity"]))

    tier1 = by_tier["tier1_single_binance"]
    assert float(tier1["basis_mtm_std"]) > 0.0, (
        "True Basis MTM phải phát sinh khi basis_pct dao động và có vị thế"
    )

    tier4 = by_tier["tier4_production_barbell"]
    overlay_ratio = float(tier4["overlay_active_ratio"])
    assert 0.0 <= overlay_ratio <= 1.0
