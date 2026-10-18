"""Chiến lược Donchian Breakout + Volume Z-Score + Regime Filter, SL/TP động theo ATR."""
from typing import Literal

import numpy as np
import pandas as pd

from src import features
from src.config import TradingConfig
from src.risk_manager import Position


def compute_breakeven_price(
    entry_price: float,
    side: Literal[1, -1],
    config: TradingConfig = TradingConfig(),
) -> float:
    """Tính mức SL hòa vốn đã cộng bù phí taker 2 chiều + slippage 2 chiều."""
    cost_buffer = 2.0 * config.taker_fee + 2.0 * config.slippage
    if side == 1:
        return float(entry_price * (1.0 + cost_buffer))
    return float(entry_price * (1.0 - cost_buffer))


compute_breakeven_sl = compute_breakeven_price


def is_breakeven_triggered(
    pos: Position,
    candle_high: float,
    candle_low: float,
    config: TradingConfig = TradingConfig(),
) -> bool:
    """Kiểm tra nến hiện tại đã đạt ngưỡng kích hoạt Breakeven (breakeven_trigger_r * initial_risk) chưa."""
    initial_sl = pos.initial_sl if pos.initial_sl != 0.0 else pos.stop_loss
    initial_risk = abs(pos.entry_price - initial_sl)
    if initial_risk <= 0.0:
        return False
    trigger_dist = config.breakeven_trigger_r * initial_risk
    if pos.side == 1:
        return bool(candle_high >= pos.entry_price + trigger_dist)
    return bool(candle_low <= pos.entry_price - trigger_dist)


class Strategy:
    def __init__(self, config: TradingConfig = TradingConfig()) -> None:
        self.config = config

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sinh cột 'signal' (1/-1/0), 'stop_loss', 'take_profit' (NaN khi flat).

        Tín hiệu tại nến t chỉ dùng dữ liệu đã đóng đến t; thực thi ở open[t+1] (Backtester).
        Hai cột 'stop_loss' và 'take_profit' ở đây là mức tham chiếu neo theo close[t];
        khi khớp thực tế tại open[t+1], Backtester tính lại SL/TP theo giá entry đã gồm slippage.
        """
        cfg = self.config
        if not all(col in df.columns for col in features.CORE_FEATURE_COLUMNS):
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

