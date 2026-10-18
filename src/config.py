from dataclasses import dataclass
from pathlib import Path


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
