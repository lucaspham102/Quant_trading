"""Tính toán chỉ báo kỹ thuật vector hóa (thuần Pandas/NumPy), không look-ahead bias."""
import numpy as np
import pandas as pd

from src.config import TradingConfig

FEATURE_COLUMNS = [
    "ema_fast",
    "ema_slow",
    "atr",
    "rsi",
    "donchian_upper",
    "donchian_lower",
    "donchian_mid",
    "vol_zscore",
    "regime",
]


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """EMA: ewm(span=period, adjust=False, min_periods=period). Chỉ dùng dữ liệu [0..t]."""
    out = series.astype("float64").ewm(span=period, adjust=False, min_periods=period).mean()
    out.name = f"ema_{period}"
    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR làm mượt Wilder (RMA) trên True Range."""
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    prev_close = df["close"].astype("float64").shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    atr.name = "atr"
    return atr


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI chuẩn Wilder. avg_loss == 0 -> RSI = 100 (nếu cả gain cũng 0 -> 50)."""
    s = series.astype("float64")
    delta = s.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    valid = avg_gain.notna() & avg_loss.notna()
    rsi = rsi.where(~(valid & (avg_loss == 0.0)), 100.0)
    rsi = rsi.where(~(valid & (avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    rsi = rsi.where(valid)
    rsi.name = "rsi"
    return rsi.astype("float64")


def compute_donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Kênh Donchian trên [t-window : t-1]; .shift(1) trước rolling để loại high/low của nến t."""
    upper = df["high"].astype("float64").shift(1).rolling(window=window, min_periods=window).max()
    lower = df["low"].astype("float64").shift(1).rolling(window=window, min_periods=window).min()
    mid = 0.5 * (upper + lower)
    return pd.DataFrame(
        {"donchian_upper": upper, "donchian_lower": lower, "donchian_mid": mid},
        index=df.index,
    )


def compute_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Z-Score khối lượng trên cửa sổ [t-window+1 : t]; std == 0 -> NaN."""
    vol = df["volume"].astype("float64")
    mean = vol.rolling(window=window, min_periods=window).mean()
    std = vol.rolling(window=window, min_periods=window).std(ddof=1)
    z = (vol - mean) / std.replace(0.0, np.nan)
    z.name = "vol_zscore"
    return z


def compute_regime_filter(
    df: pd.DataFrame,
    ema_fast_period: int = 50,
    ema_slow_period: int = 200,
    slope_lookback: int = 5,
) -> pd.Series:
    """Bull=+1, Bear=-1, Neutral=0 (int8). Warmup (NaN) -> 0."""
    close = df["close"].astype("float64")
    ema_fast = compute_ema(close, ema_fast_period)
    ema_slow = compute_ema(close, ema_slow_period)
    slope = ema_slow - ema_slow.shift(slope_lookback)
    bull = (close > ema_slow) & (ema_fast > ema_slow) & (slope > 0)
    bear = (close < ema_slow) & (ema_fast < ema_slow) & (slope < 0)
    regime = pd.Series(
        np.where(bull, 1, np.where(bear, -1, 0)).astype(np.int8),
        index=df.index,
        name="regime",
    )
    return regime


def add_all_features(
    df: pd.DataFrame,
    config: TradingConfig = TradingConfig(),
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Trả bản sao df kèm các cột đặc trưng. drop_warmup=True loại các nến chưa đủ chu kỳ."""
    out = df.copy()
    out["ema_fast"] = compute_ema(out["close"], config.ema_fast_period)
    out["ema_slow"] = compute_ema(out["close"], config.ema_slow_period)
    out["atr"] = compute_atr(out, config.atr_period)
    out["rsi"] = compute_rsi(out["close"], config.rsi_period)
    donchian = compute_donchian(out, config.donchian_window)
    for col in donchian.columns:
        out[col] = donchian[col]
    out["vol_zscore"] = compute_volume_zscore(out, config.vol_zscore_window)
    out["regime"] = compute_regime_filter(
        out,
        config.ema_fast_period,
        config.ema_slow_period,
        config.regime_slope_lookback,
    )
    if drop_warmup:
        # regime chưa hợp lệ khi độ dốc EMA slow còn NaN
        slope_ok = out["ema_slow"].shift(config.regime_slope_lookback).notna()
        out = out.loc[slope_ok].dropna()
    return out
