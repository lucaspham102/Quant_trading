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


def is_trailing_triggered(
    pos: Position,
    candle_high: float,
    candle_low: float,
    config: TradingConfig = TradingConfig(),
) -> bool:
    """Kiểm tra cực trị kể từ lúc vào lệnh (gồm nến hiện tại) đã đạt trailing_activation_r * R0 chưa."""
    r0 = abs(pos.entry_price - pos.initial_sl)
    if r0 <= 0.0:
        return False
    d_act = config.trailing_activation_r * r0
    if pos.side == 1:
        return bool(max(pos.highest_price_since_entry, candle_high) >= pos.entry_price + d_act)
    return bool(min(pos.lowest_price_since_entry, candle_low) <= pos.entry_price - d_act)


def compute_trailing_stop(
    pos: Position,
    candle_high: float,
    candle_low: float,
    atr_val: float,
    config: TradingConfig = TradingConfig(),
) -> float:
    """SL trailing theo ATR kèm bất biến ratchet (chỉ siết một chiều có lợi). Gọi SAU khi nến đóng."""
    peak = max(pos.highest_price_since_entry, candle_high)
    trough = min(pos.lowest_price_since_entry, candle_low)
    if pos.side == 1:
        cand = peak - config.trailing_atr_mult * atr_val
        return float(max(pos.stop_loss, cand))
    cand = trough + config.trailing_atr_mult * atr_val
    return float(min(pos.stop_loss, cand))


def _shift(s: pd.Series, k: int = 1) -> pd.Series:
    return s.shift(k)


class Strategy:
    def __init__(self, config: TradingConfig = TradingConfig()) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Các nhánh chiến lược: mỗi nhánh trả (long, short, sl, tp) dạng ndarray
    # (sl/tp = NaN ở nơi không có tín hiệu). Chỉ dùng dữ liệu đến hết nến t.
    # ------------------------------------------------------------------
    def _breakout_branch(self, out: pd.DataFrame):
        cfg = self.config
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
        return long_cond, short_cond, stop_loss, take_profit

    def _trend_pullback_branch(self, out: pd.DataFrame):
        cfg = self.config
        close = out["close"].astype("float64")
        open_ = out["open"].astype("float64")
        high = out["high"].astype("float64")
        low = out["low"].astype("float64")
        atr = out["atr"].astype("float64")
        rsi = out["rsi"].astype("float64")
        ema_fast = out["ema_fast"].astype("float64")
        ema_slow = out["ema_slow"].astype("float64")
        ema20 = out["ema_20"].astype("float64")
        bb_mid = out["bb_mid"].astype("float64")
        slope = out["ema_slow_slope"].astype("float64")
        adx = out["adx"].astype("float64")

        trend_ok = adx >= cfg.adx_threshold
        bull = (ema_fast > ema_slow) & (slope > 0) & trend_ok & (close > ema_slow)
        bear = (ema_fast < ema_slow) & (slope < 0) & trend_ok & (close < ema_slow)

        support = pd.concat([ema20, bb_mid], axis=1).max(axis=1, skipna=False)
        resist = pd.concat([ema20, bb_mid], axis=1).min(axis=1, skipna=False)
        rsi_prev = _shift(rsi)

        long_touch = (
            (low <= support)
            | (_shift(low) <= _shift(support))
            | (rsi_prev < cfg.pullback_rsi_long_max)
        )
        long_turn = (
            (close > open_)
            & (close >= ema20)
            & (rsi > rsi_prev)
            & (rsi < cfg.rsi_long_max)
        )
        short_touch = (
            (high >= resist)
            | (_shift(high) >= _shift(resist))
            | (rsi_prev > cfg.pullback_rsi_short_min)
        )
        short_turn = (
            (close < open_)
            & (close <= ema20)
            & (rsi < rsi_prev)
            & (rsi > cfg.rsi_short_min)
        )
        long_cond = (bull & long_touch & long_turn).fillna(False).to_numpy(dtype=bool)
        short_cond = (bear & short_touch & short_turn).fillna(False).to_numpy(dtype=bool)

        close_v = close.to_numpy()
        atr_v = atr.to_numpy()
        d_sl = np.maximum(cfg.sl_atr_multiple * atr_v, close_v * cfg.min_stop_distance_pct)
        rr = max(float(cfg.pullback_rr_ratio), 2.0)
        d_tp = np.maximum(cfg.tp_atr_multiple * atr_v, rr * d_sl)
        stop_loss = np.where(
            long_cond, close_v - d_sl, np.where(short_cond, close_v + d_sl, np.nan)
        ).astype("float64")
        take_profit = np.where(
            long_cond, close_v + d_tp, np.where(short_cond, close_v - d_tp, np.nan)
        ).astype("float64")
        return long_cond, short_cond, stop_loss, take_profit

    def _mean_reversion_branch(self, out: pd.DataFrame):
        cfg = self.config
        close = out["close"].astype("float64")
        open_ = out["open"].astype("float64")
        high = out["high"].astype("float64")
        low = out["low"].astype("float64")
        atr = out["atr"].astype("float64")
        rsi = out["rsi"].astype("float64")
        adx = out["adx"].astype("float64")

        if (
            "bb_mid" in out.columns
            and "bb_lower" in out.columns
            and "bb_upper" in out.columns
            and (cfg.mr_bb_std_mult == cfg.bb_std_mult or len(out) < cfg.bb_window)
        ):
            bb_mid = out["bb_mid"].astype("float64")
            bb_lower = out["bb_lower"].astype("float64")
            bb_upper = out["bb_upper"].astype("float64")
        else:
            bands = features.compute_bollinger_bands(close, cfg.bb_window, cfg.mr_bb_std_mult)
            bb_mid = bands["bb_mid"]
            bb_lower = bands["bb_lower"]
            bb_upper = bands["bb_upper"]

        sideways = adx < cfg.adx_threshold
        rsi_prev = _shift(rsi)
        close_prev = _shift(close)

        long_cond_s = (
            sideways
            & ((_shift(low) < _shift(bb_lower)) | (low < bb_lower))
            & (pd.concat([rsi_prev, rsi], axis=1).min(axis=1, skipna=False) <= cfg.mr_rsi_oversold)
            & (close > open_)
            & (close > close_prev)
        )
        short_cond_s = (
            sideways
            & ((_shift(high) > _shift(bb_upper)) | (high > bb_upper))
            & (pd.concat([rsi_prev, rsi], axis=1).max(axis=1, skipna=False) >= cfg.mr_rsi_overbought)
            & (close < open_)
            & (close < close_prev)
        )
        long_cond = long_cond_s.fillna(False).to_numpy(dtype=bool)
        short_cond = (short_cond_s.fillna(False).to_numpy(dtype=bool)) & ~long_cond

        close_v = close.to_numpy()
        atr_v = atr.to_numpy()
        mid_v = bb_mid.to_numpy()
        d_sl = np.maximum(cfg.sl_atr_multiple * atr_v, close_v * cfg.min_stop_distance_pct)
        rr = max(float(cfg.mr_rr_ratio), 1.5)
        d_tp = np.maximum(np.abs(mid_v - close_v), rr * d_sl)
        d_tp = np.where(np.isfinite(d_tp), d_tp, rr * d_sl)
        stop_loss = np.where(
            long_cond, close_v - d_sl, np.where(short_cond, close_v + d_sl, np.nan)
        ).astype("float64")
        take_profit = np.where(
            long_cond, close_v + d_tp, np.where(short_cond, close_v - d_tp, np.nan)
        ).astype("float64")
        return long_cond, short_cond, stop_loss, take_profit

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sinh cột 'signal' (1/-1/0), 'stop_loss', 'take_profit', 'limit_price' (NaN khi flat).

        Tín hiệu tại nến t chỉ dùng dữ liệu đã đóng đến t; thực thi ở open[t+1] (Backtester).
        Hai cột 'stop_loss' và 'take_profit' ở đây là mức tham chiếu neo theo close[t];
        khi khớp thực tế tại open[t+1], Backtester tính lại SL/TP theo giá entry đã gồm slippage.
        'limit_price' là mức đặt Limit pullback: close[t] -/+ limit_pullback_atr_mult * atr[t].
        """
        cfg = self.config
        mode = cfg.strategy_mode
        need_regime_cols = mode != "breakout"
        if not all(col in df.columns for col in features.CORE_FEATURE_COLUMNS) or (
            need_regime_cols
            and not all(col in df.columns for col in features.REGIME_STRATEGY_COLUMNS)
        ):
            out = features.add_all_features(df, cfg, drop_warmup=True)
        else:
            out = df.copy()

        close = out["close"].astype("float64")
        atr = out["atr"].astype("float64")

        if mode == "trend_pullback":
            long_cond, short_cond, stop_loss, take_profit = self._trend_pullback_branch(out)
        elif mode == "mean_reversion":
            long_cond, short_cond, stop_loss, take_profit = self._mean_reversion_branch(out)
        elif mode == "regime_adaptive":
            if cfg.adaptive_trend_mode == "breakout":
                t_long, t_short, t_sl, t_tp = self._breakout_branch(out)
            else:
                t_long, t_short, t_sl, t_tp = self._trend_pullback_branch(out)
            m_long, m_short, m_sl, m_tp = self._mean_reversion_branch(out)
            trend_mask = (out["adx"].astype("float64") >= cfg.adx_threshold).to_numpy(dtype=bool)
            long_cond = np.where(trend_mask, t_long, m_long)
            short_cond = np.where(trend_mask, t_short, m_short)
            stop_loss = np.where(trend_mask, t_sl, m_sl).astype("float64")
            take_profit = np.where(trend_mask, t_tp, m_tp).astype("float64")
        else:
            long_cond, short_cond, stop_loss, take_profit = self._breakout_branch(out)

        signal = np.where(long_cond, 1, np.where(short_cond, -1, 0)).astype(np.int64)

        close_v = close.to_numpy()
        atr_v = atr.to_numpy()
        limit_price = np.where(
            long_cond,
            close_v - cfg.limit_pullback_atr_mult * atr_v,
            np.where(short_cond, close_v + cfg.limit_pullback_atr_mult * atr_v, np.nan),
        ).astype("float64")

        out["signal"] = signal
        out["stop_loss"] = stop_loss
        out["take_profit"] = take_profit
        out["limit_price"] = limit_price
        return out

