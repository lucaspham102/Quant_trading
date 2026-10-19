from dataclasses import dataclass
from pathlib import Path
from typing import Literal


CORE_4_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
EXTRA_8_SYMBOLS = [
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "DOT/USDT:USDT",
    "LTC/USDT:USDT",
    "TRX/USDT:USDT",
]
UNIVERSE_12_SYMBOLS = CORE_4_SYMBOLS + EXTRA_8_SYMBOLS


@dataclass(frozen=True)
class TradingConfig:
    # Định danh thị trường & khung thời gian
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "15m"
    candles_per_year: int = 35040          # 365 * 24 * 4 (khung 15m)

    # Vốn & Quản trị rủi ro
    initial_capital: float = 400.0         # NAV khởi tạo (USDT)
    risk_per_trade_pct: float = 0.01       # 1% NAV mỗi lệnh (= 4.0 USDT khi NAV=400)
    max_daily_drawdown_pct: float = 0.03   # 3% NAV lỗ tối đa trong ngày UTC (= 12.0 USDT khi NAV=400)
    max_effective_leverage: float = 2.0    # Đòn bẩy hiệu dụng tối đa (Notional / NAV <= 2.0)

    # Phí giao dịch, Trượt giá & Ràng buộc sàn Binance Futures
    taker_fee: float = 0.0005              # 0.05% mỗi chiều
    maker_fee: float = 0.0002              # 0.02% mỗi chiều
    slippage: float = 0.0002               # 0.02% trượt giá mỗi chiều
    min_notional: float = 5.0              # Giá trị danh nghĩa tối thiểu trên Binance Futures (USDT)
    step_size: float = 0.001               # Bước làm tròn khối lượng (BTC lot step size)

    # Tham số chỉ báo kỹ thuật (Features)
    ema_fast_period: int = 50              # Chu kỳ EMA nhanh
    ema_slow_period: int = 200             # Chu kỳ EMA chậm (dùng cho Regime Filter)
    atr_period: int = 14                   # Chu kỳ ATR chuẩn Wilder
    rsi_period: int = 14                   # Chu kỳ RSI chuẩn Wilder
    donchian_window: int = 20              # Cửa sổ kênh Donchian
    vol_zscore_window: int = 20            # Cửa sổ chuẩn hóa Z-Score khối lượng
    regime_slope_lookback: int = 5         # Số nến tính độ dốc EMA slow cho Regime Filter

    # Tham số Chiến lược & TP/SL động
    vol_zscore_threshold: float = 1.5      # Ngưỡng đột biến khối lượng xác nhận breakout
    rsi_long_max: float = 75.0             # Ngưỡng chống mua đuổi quá mức
    rsi_short_min: float = 25.0            # Ngưỡng chống bán đuổi quá mức
    sl_atr_multiple: float = 2.0           # Khoảng cách Stop Loss = 2.0 * ATR
    tp_atr_multiple: float = 3.0           # Khoảng cách Take Profit = 3.0 * ATR

    # Đường dẫn lưu trữ
    raw_data_dir: Path = Path("data/raw")

    # Tham số đặc trưng vi cấu trúc & chu kỳ (Sprint 2 - Phần A1)
    bb_window: int = 20                              # Cửa sổ Bollinger Bands cho bb_width_squeeze
    bb_std_mult: float = 2.0                         # Hệ số độ lệch chuẩn Bollinger Bands
    return_lags: tuple[int, int, int] = (1, 4, 16)   # Các bậc trễ lợi suất close.pct_change(k)

    # Tham số Breakeven Stop & Time-based Exit (Sprint 2 - Phần A2, A3)
    use_breakeven: bool = True                       # Bật/tắt dời SL về hòa vốn
    breakeven_trigger_r: float = 1.0                 # Hệ số R kích hoạt breakeven (1.0 * initial_risk)
    max_holding_bars: int = 48                       # Số nến giữ lệnh tối đa (48 nến 15m = 12 giờ); <= 0 là tắt

    # Tham số ML Meta-Labeling & Purged Walk-Forward (Sprint 2 - Phần B)
    use_ml: bool = False                             # Bật/tắt bộ lọc ML trong Backtester
    ml_prob_threshold: float = 0.55                  # Ngưỡng xác suất thắng tối thiểu để duyệt lệnh
    ml_n_splits: int = 5                             # Số fold cho PurgedWalkForwardSplit
    ml_embargo_bars: int = 4                         # Số nến embargo sau tập test
    ml_min_train_samples: int = 30                   # Số mẫu tối thiểu để fit model (dưới ngưỡng -> fallback)

    # Tham số StateStore, Notifier & LiveExecutor (Sprint 2 - Phần C)
    state_file_path: Path = Path("data/state/bot_state.json")
    live_lookback_candles: int = 300                 # Số nến lịch sử tối thiểu tải mỗi chu kỳ để tính đủ warmup
    poll_interval_seconds: float = 5.0               # Chu kỳ ngủ giữa các lần kiểm tra nến mới

    # Sprint 3 - Phần A: Fee-aware sizing, Min Stop Distance, Limit Pullback, Trailing Stop
    use_fee_aware_sizing: bool = False
    min_stop_distance_pct: float = 0.0
    min_stop_action: Literal["widen", "reject"] = "widen"
    entry_mode: Literal["market", "limit_pullback"] = "market"
    limit_pullback_atr_mult: float = 0.5
    limit_order_ttl_bars: int = 3
    use_trailing_stop: bool = False
    trailing_activation_r: float = 0.75          # Ngưỡng kích hoạt trailing (× R0); phải thỏa trailing_activation_r*sl_atr_multiple < tp_atr_multiple
    trailing_atr_mult: float = 2.0

    # Sprint 3 - Phần B: Đa chiến lược & chế độ thị trường
    strategy_mode: Literal["breakout", "trend_pullback", "mean_reversion", "regime_adaptive"] = "breakout"
    ema_pullback_period: int = 20
    adx_period: int = 14
    adx_threshold: float = 25.0
    pullback_rsi_long_max: float = 50.0
    pullback_rsi_short_min: float = 50.0
    pullback_rr_ratio: float = 2.0
    mr_bb_std_mult: float = 2.0
    mr_rsi_oversold: float = 30.0
    mr_rsi_overbought: float = 70.0
    mr_rr_ratio: float = 1.5
    adaptive_trend_mode: Literal["trend_pullback", "breakout"] = "trend_pullback"

    def __post_init__(self) -> None:
        """Fail-fast: chặn cấu hình Trailing Stop không bao giờ có thể kích hoạt."""
        if not self.use_trailing_stop:
            return
        if self.trailing_activation_r <= 0:
            raise ValueError(
                f"trailing_activation_r phải > 0 khi bật trailing, nhận: {self.trailing_activation_r}"
            )
        if self.sl_atr_multiple <= 0:
            raise ValueError(
                f"sl_atr_multiple phải > 0 khi bật trailing, nhận: {self.sl_atr_multiple}"
            )
        if self.trailing_atr_mult <= 0:
            raise ValueError(
                f"trailing_atr_mult phải > 0 khi bật trailing, nhận: {self.trailing_atr_mult}"
            )
        if self.tp_atr_multiple > 0:
            activation_dist = self.trailing_activation_r * self.sl_atr_multiple
            if activation_dist >= self.tp_atr_multiple:
                raise ValueError(
                    f"Cấu hình Trailing Stop vô hiệu: trailing_activation_r "
                    f"({self.trailing_activation_r}) * sl_atr_multiple ({self.sl_atr_multiple}) = "
                    f"{activation_dist:.4f} >= tp_atr_multiple ({self.tp_atr_multiple}). "
                    f"Giá sẽ chạm TP trước khi trailing kích hoạt; hãy giảm trailing_activation_r "
                    f"hoặc tăng tp_atr_multiple."
                )


def candles_per_year_for_timeframe(timeframe: str) -> int:
    """Số nến mỗi năm (365 ngày) theo khung thời gian: 15m -> 35040, 1h -> 8760."""
    from src.data_loader import timeframe_to_ms  # import muộn để tránh vòng lặp import

    return int((365 * 86_400_000) // timeframe_to_ms(timeframe))

