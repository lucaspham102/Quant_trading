"""Tính toán chỉ báo kỹ thuật vector hóa (thuần Pandas/NumPy), không look-ahead bias."""
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import TradingConfig

CORE_FEATURE_COLUMNS = [
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

MICROSTRUCTURE_COLUMNS = [
    "body_to_range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "bb_width_squeeze",
    "return_lag_1",
    "return_lag_4",
    "return_lag_16",
    "hour_sin",
    "hour_cos",
]

FEATURE_COLUMNS = [
    *CORE_FEATURE_COLUMNS,
    *MICROSTRUCTURE_COLUMNS,
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


def compute_candle_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Tỷ lệ thân nến và râu nến trên biên độ [low, high]. Khi high == low -> 0.0 (không NaN/inf)."""
    open_ = df["open"].astype("float64").to_numpy()
    high = df["high"].astype("float64").to_numpy()
    low = df["low"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()

    rng = high - low
    valid = np.isfinite(rng) & (rng > 0.0) & np.isfinite(open_) & np.isfinite(close)
    safe_rng = np.where(valid, rng, 1.0)

    body = np.abs(close - open_)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    body_ratio = np.where(valid, body / safe_rng, 0.0)
    upper_ratio = np.where(valid, upper_wick / safe_rng, 0.0)
    lower_ratio = np.where(valid, lower_wick / safe_rng, 0.0)

    return pd.DataFrame(
        {
            "body_to_range_ratio": body_ratio,
            "upper_wick_ratio": upper_ratio,
            "lower_wick_ratio": lower_ratio,
        },
        index=df.index,
        dtype="float64",
    )


def compute_bb_width_squeeze(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std_mult: float = 2.0,
    atr_period: int = 14,
) -> pd.Series:
    """Độ rộng dải Bollinger chuẩn hóa: (upper - lower) / mid trên cửa sổ bb_window (ddof=1)."""
    _ = atr_period
    close = df["close"].astype("float64") if isinstance(df, pd.DataFrame) else df.astype("float64")
    mid = close.rolling(window=bb_window, min_periods=bb_window).mean()
    std = close.rolling(window=bb_window, min_periods=bb_window).std(ddof=1)
    upper = mid + bb_std_mult * std
    lower = mid - bb_std_mult * std
    squeeze = (upper - lower) / mid.replace(0.0, np.nan)
    squeeze.name = "bb_width_squeeze"
    return squeeze.astype("float64")


def compute_return_lags(
    series: pd.Series,
    lags: Sequence[int] = (1, 4, 16),
) -> pd.DataFrame:
    """Lợi suất trễ close.pct_change(periods=k) (tương thích Pandas 3.0, không truyền fill_method)."""
    s = series["close"].astype("float64") if isinstance(series, pd.DataFrame) else series.astype("float64")
    cols = {f"return_lag_{int(k)}": s.pct_change(periods=int(k)) for k in lags}
    return pd.DataFrame(cols, index=s.index, dtype="float64")


def compute_time_cyclical(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Mã hóa chu kỳ giờ UTC: sin(2*pi*hour/24), cos(2*pi*hour/24)."""
    idx = index.index if isinstance(index, pd.DataFrame) else pd.DatetimeIndex(index)
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    hours = idx_utc.hour.to_numpy(dtype="float64")
    angles = 2.0 * np.pi * hours / 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(angles),
            "hour_cos": np.cos(angles),
        },
        index=idx,
        dtype="float64",
    )


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

    micro = compute_candle_microstructure(out)
    for col in micro.columns:
        out[col] = micro[col]
    out["bb_width_squeeze"] = compute_bb_width_squeeze(
        out, config.bb_window, config.bb_std_mult, config.atr_period
    )
    ret_lags = compute_return_lags(out["close"], config.return_lags)
    for col in ret_lags.columns:
        out[col] = ret_lags[col]
    cyclical = compute_time_cyclical(out.index)
    for col in cyclical.columns:
        out[col] = cyclical[col]

    if drop_warmup:
        # Chỉ cắt đoạn warmup ở đầu chuỗi khi các chỉ báo cốt lõi chưa đủ chu kỳ;
        # nếu giữa chuỗi có vol_zscore == NaN (do std == 0) thì điền 0.0 để giữ index liên tục.
        lag_cols = [f"return_lag_{int(k)}" for k in config.return_lags]
        core_cols = [
            "ema_fast",
            "ema_slow",
            "atr",
            "rsi",
            "donchian_upper",
            "donchian_lower",
            "donchian_mid",
            "bb_width_squeeze",
            *lag_cols,
        ]
        slope_ok = out["ema_slow"].shift(config.regime_slope_lookback).notna()
        vol_warmup_ok = pd.Series(
            np.arange(len(out)) >= config.vol_zscore_window - 1,
            index=out.index,
        )
        bb_warmup_ok = pd.Series(
            np.arange(len(out)) >= config.bb_window - 1,
            index=out.index,
        )
        valid_mask = (
            slope_ok
            & out[core_cols].notna().all(axis=1)
            & vol_warmup_ok
            & bb_warmup_ok
        )
        if valid_mask.any():
            first_valid_pos = int(np.argmax(valid_mask.to_numpy()))
            out = out.iloc[first_valid_pos:].copy()
        else:
            out = out.iloc[0:0].copy()
        out["vol_zscore"] = out["vol_zscore"].fillna(0.0)
        out["bb_width_squeeze"] = out["bb_width_squeeze"].fillna(0.0)
    return out

