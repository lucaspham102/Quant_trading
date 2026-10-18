"""Chiến lược Donchian Breakout + Volume Z-Score + Regime Filter, SL/TP động theo ATR."""
import numpy as np
import pandas as pd

from src import features
from src.config import TradingConfig


class Strategy:
    def __init__(self, config: TradingConfig = TradingConfig()) -> None:
        self.config = config

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sinh cột 'signal' (1/-1/0), 'stop_loss', 'take_profit' (NaN khi flat).

        Tín hiệu tại nến t chỉ dùng dữ liệu đã đóng đến t; thực thi ở open[t+1] (Backtester).
        """
        cfg = self.config
        if not all(col in df.columns for col in features.FEATURE_COLUMNS):
            out = features.add_all_features(df, cfg, drop_warmup=True)
        else:
            out = df.copy()

        close = out["close"].astype("float64")
        atr = out["atr"].astype("float64")

        vol_ok = out["vol_zscore"] >= cfg.vol_zscore_threshold
        long_cond = (
            (close > out["donchian_upper"])
            & vol_ok
            & (out["regime"] == 1)
            & (out["rsi"] < cfg.rsi_long_max)
        ).to_numpy(dtype=bool)
        short_cond = (
            (close < out["donchian_lower"])
            & vol_ok
            & (out["regime"] == -1)
            & (out["rsi"] > cfg.rsi_short_min)
        ).to_numpy(dtype=bool)

        close_v = close.to_numpy()
        atr_v = atr.to_numpy()
        signal = np.where(long_cond, 1, np.where(short_cond, -1, 0)).astype(np.int64)
        stop_loss = np.where(
            long_cond,
            close_v - cfg.sl_atr_multiple * atr_v,
            np.where(short_cond, close_v + cfg.sl_atr_multiple * atr_v, np.nan),
        ).astype("float64")
        take_profit = np.where(
            long_cond,
            close_v + cfg.tp_atr_multiple * atr_v,
            np.where(short_cond, close_v - cfg.tp_atr_multiple * atr_v, np.nan),
        ).astype("float64")

        out["signal"] = signal
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        return out
