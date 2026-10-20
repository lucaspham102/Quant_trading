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
