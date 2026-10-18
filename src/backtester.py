"""Backtester: khớp lệnh next-bar open, SL/TP trong nến (SL ưu tiên), phí taker 2 chiều + slippage."""
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from src.config import TradingConfig
from src.risk_manager import Position, RiskManager
from src.strategy import (
    Strategy,
    compute_breakeven_price,
    compute_trailing_stop,
    is_breakeven_triggered,
    is_trailing_triggered,
)

ExitReason = Literal["SL", "TP", "TIME_EXIT", "SIGNAL_REVERSE", "END_OF_DATA"]


@dataclass(frozen=True)
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Literal[1, -1]
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    total_fee: float
    net_pnl: float
    exit_reason: ExitReason
    entry_order_type: Literal["market", "limit"] = "market"


@dataclass
class PendingLimitOrder:
    side: Literal[1, -1]
    limit_price: float
    atr_at_signal: float
    created_bar: int
    expires_bar: int          # created_bar + config.limit_order_ttl_bars


@dataclass(frozen=True)
class BacktestReport:
    total_return_pct: float
    final_equity: float
    max_drawdown_pct: float        # số dương: 5.25 nghĩa là sụt giảm 5.25%
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    total_fees_paid: float
    equity_curve: pd.Series
    trades: list[TradeRecord]


class Backtester:
    def __init__(
        self,
        config: TradingConfig = TradingConfig(),
        strategy: Optional[Strategy] = None,
        risk_manager: Optional[RiskManager] = None,
        ml_filter: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.strategy = strategy if strategy is not None else Strategy(config)
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager(config)
        self.ml_filter = ml_filter

    # ------------------------------------------------------------------
    def _close_position(
        self,
        pos: Position,
        raw_exit: float,
        exit_time: pd.Timestamp,
        reason: str,
    ) -> TradeRecord:
        cfg = self.config
        side = pos.side
        exit_price = raw_exit * (1.0 - cfg.slippage) if side == 1 else raw_exit * (1.0 + cfg.slippage)
        gross = pos.qty * (exit_price - pos.entry_price) * side
        fee = pos.entry_fee + pos.qty * exit_price * cfg.taker_fee
        net = gross - fee
        self.risk_manager.update_after_trade_close(net, exit_time)
        return TradeRecord(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            side=side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            gross_pnl=gross,
            total_fee=fee,
            net_pnl=net,
            exit_reason=reason,  # type: ignore[arg-type]
            entry_order_type=pos.entry_order_type,
        )

    def _try_open(
        self, side: int, open_px: float, atr_prev: float, ts: pd.Timestamp
    ) -> Optional[Position]:
        cfg = self.config
        if not np.isfinite(atr_prev) or atr_prev <= 0 or not np.isfinite(open_px):
            return None
        if side == 1:
            entry = open_px * (1.0 + cfg.slippage)
            sl = entry - cfg.sl_atr_multiple * atr_prev
            tp = entry + cfg.tp_atr_multiple * atr_prev
        else:
            entry = open_px * (1.0 - cfg.slippage)
            sl = entry + cfg.sl_atr_multiple * atr_prev
            tp = entry - cfg.tp_atr_multiple * atr_prev
        return self.risk_manager.build_position(side, entry, sl, tp, ts)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> BacktestReport:
        sig_df = self.strategy.generate_signals(df)

        index = sig_df.index
        opens = sig_df["open"].to_numpy(dtype="float64")
        highs = sig_df["high"].to_numpy(dtype="float64")
        lows = sig_df["low"].to_numpy(dtype="float64")
        closes = sig_df["close"].to_numpy(dtype="float64")
        signals = sig_df["signal"].to_numpy()
        atrs = sig_df["atr"].to_numpy(dtype="float64")
        n = len(sig_df)

        ml_probs: Optional[np.ndarray] = None
        ml_threshold: float = float(self.config.ml_prob_threshold)
        if self.config.use_ml and n > 0:
            from src.ml_filter import _ensure_ml_feature_columns, compute_walk_forward_probs

            sig_df = _ensure_ml_feature_columns(sig_df, self.config)
            if self.ml_filter is not None and getattr(self.ml_filter, "is_fitted_", True):
                ml_threshold = float(getattr(self.ml_filter, "prob_threshold", ml_threshold))
                if hasattr(self.ml_filter, "predict_proba"):
                    ml_probs = np.asarray(self.ml_filter.predict_proba(sig_df), dtype="float64")
            else:
                ml_probs = compute_walk_forward_probs(sig_df, self.config).to_numpy(dtype="float64")

        trades: list[TradeRecord] = []
        equity_vals = np.empty(n, dtype="float64")
        pos: Optional[Position] = None
        pending: Optional[PendingLimitOrder] = None
        cfg = self.config
        limit_mode = cfg.entry_mode == "limit_pullback"
        limit_prices: Optional[np.ndarray] = (
            sig_df["limit_price"].to_numpy(dtype="float64") if "limit_price" in sig_df.columns else None
        )

        def _ml_allows(bar: int) -> bool:
            """Duyệt ML cho tín hiệu phát ra ở nến `bar`."""
            if not cfg.use_ml:
                return True
            if ml_probs is not None:
                prob_win = float(ml_probs[bar])
                return bool(np.isfinite(prob_win) and prob_win >= ml_threshold)
            if self.ml_filter is not None and hasattr(self.ml_filter, "should_take"):
                return bool(self.ml_filter.should_take(sig_df.iloc[bar]))
            return True

        for t in range(n):
            ts = index[t]
            self.risk_manager.on_new_Timestamp(ts)
            prev_sig = int(signals[t - 1]) if t > 0 and not pd.isna(signals[t - 1]) else 0

            # a) Đảo chiều tại open[t] nếu tín hiệu t-1 ngược vị thế đang mở
            if pos is not None and prev_sig != 0 and prev_sig == -pos.side:
                trades.append(self._close_position(pos, opens[t], ts, "SIGNAL_REVERSE"))
                pos = None

            # b) Mở vị thế
            opened_this_bar = False
            opened_by_limit = False
            if not limit_mode:
                # b1) Market: khớp tại open[t] khi tín hiệu t-1 != 0
                if pos is None and prev_sig != 0:
                    if _ml_allows(t - 1):
                        pos = self._try_open(prev_sig, opens[t], atrs[t - 1], ts)
                        opened_this_bar = pos is not None
            else:
                # b2) Limit pullback: tạo/thay lệnh chờ từ tín hiệu t-1, rồi kiểm tra khớp trong nến t
                if prev_sig != 0 and pos is None:
                    if pending is not None and prev_sig == -pending.side:
                        pending = None  # tín hiệu ngược chiều: hủy lệnh chờ cũ
                    if pending is None and _ml_allows(t - 1):
                        atr_sig = float(atrs[t - 1])
                        if np.isfinite(atr_sig) and atr_sig > 0.0:
                            lp = (
                                float(limit_prices[t - 1])
                                if limit_prices is not None and np.isfinite(limit_prices[t - 1])
                                else float("nan")
                            )
                            if not np.isfinite(lp):
                                ref = float(closes[t - 1])
                                lp = (
                                    ref - cfg.limit_pullback_atr_mult * atr_sig
                                    if prev_sig == 1
                                    else ref + cfg.limit_pullback_atr_mult * atr_sig
                                )
                            if lp > 0.0:
                                pending = PendingLimitOrder(
                                    side=1 if prev_sig == 1 else -1,
                                    limit_price=lp,
                                    atr_at_signal=atr_sig,
                                    created_bar=t,
                                    expires_bar=t + int(cfg.limit_order_ttl_bars),
                                )
                if pending is not None and pos is None:
                    if pending.side == 1:
                        touched = lows[t] <= pending.limit_price
                        fill_px = min(float(opens[t]), pending.limit_price)
                        sl_f = fill_px - cfg.sl_atr_multiple * pending.atr_at_signal
                        tp_f = fill_px + cfg.tp_atr_multiple * pending.atr_at_signal
                    else:
                        touched = highs[t] >= pending.limit_price
                        fill_px = max(float(opens[t]), pending.limit_price)
                        sl_f = fill_px + cfg.sl_atr_multiple * pending.atr_at_signal
                        tp_f = fill_px - cfg.tp_atr_multiple * pending.atr_at_signal
                    if touched:
                        # Maker: không slippage chiều vào, phí maker
                        pos = self.risk_manager.build_position(
                            pending.side, fill_px, sl_f, tp_f, ts, is_maker_entry=True  # type: ignore[arg-type]
                        )
                        pending = None
                        opened_this_bar = pos is not None
                        opened_by_limit = opened_this_bar
                    elif t + 1 >= pending.expires_bar:
                        pending = None  # hết TTL, hủy lệnh chờ

            # c) Kiểm tra SL/TP trong [low[t], high[t]] (SL ưu tiên, xử lý gap qua open[t])
            #    Dùng mức SL CŨ (đã có từ trước khi nến t đóng).
            if pos is not None:
                if pos.side == 1:
                    hit_sl = lows[t] <= pos.stop_loss
                    hit_tp = highs[t] >= pos.take_profit
                    raw_sl = pos.stop_loss if opened_this_bar else min(opens[t], pos.stop_loss)
                    raw_tp = pos.take_profit if opened_this_bar else max(opens[t], pos.take_profit)
                else:
                    hit_sl = highs[t] >= pos.stop_loss
                    hit_tp = lows[t] <= pos.take_profit
                    raw_sl = pos.stop_loss if opened_this_bar else max(opens[t], pos.stop_loss)
                    raw_tp = pos.take_profit if opened_this_bar else min(opens[t], pos.take_profit)
                if hit_sl:
                    trades.append(self._close_position(pos, raw_sl, ts, "SL"))
                    pos = None
                elif hit_tp:
                    trades.append(self._close_position(pos, raw_tp, ts, "TP"))
                    pos = None

            # d) SAU KHI nến t đóng: cập nhật cực trị, Breakeven, ATR Trailing (chỉ hiệu lực từ nến t+1)
            if pos is not None:
                eff_high = float(highs[t])
                eff_low = float(lows[t])
                if opened_by_limit:
                    # Nến khớp Limit: không biết đỉnh/đáy xảy ra trước hay sau lúc khớp ->
                    # bảo thủ, chỉ tính từ giá khớp và close[t].
                    eff_high = max(pos.entry_price, float(closes[t]))
                    eff_low = min(pos.entry_price, float(closes[t]))
                pos.highest_price_since_entry = max(pos.highest_price_since_entry, eff_high)
                pos.lowest_price_since_entry = min(pos.lowest_price_since_entry, eff_low)

                if cfg.use_breakeven and not pos.breakeven_activated:
                    if is_breakeven_triggered(pos, eff_high, eff_low, cfg):
                        be_sl = compute_breakeven_price(pos.entry_price, pos.side, cfg)
                        if pos.side == 1:
                            pos.stop_loss = max(pos.stop_loss, be_sl)
                        else:
                            pos.stop_loss = min(pos.stop_loss, be_sl)
                        pos.breakeven_activated = True

                if cfg.use_trailing_stop and np.isfinite(atrs[t]) and atrs[t] > 0.0:
                    if pos.trailing_activated or is_trailing_triggered(pos, eff_high, eff_low, cfg):
                        pos.trailing_activated = True
                        pos.stop_loss = compute_trailing_stop(
                            pos, eff_high, eff_low, float(atrs[t]), cfg
                        )

            # e) Kiểm tra Time-based Exit tại close[t] (nếu vị thế vẫn mở)
            if pos is not None:
                pos.bars_held += 1
                if self.config.max_holding_bars > 0 and pos.bars_held >= self.config.max_holding_bars:
                    trades.append(self._close_position(pos, closes[t], ts, "TIME_EXIT"))
                    pos = None

            # f) Đóng cưỡng bức ở nến cuối
            if pos is not None and t == n - 1:
                trades.append(self._close_position(pos, closes[t], ts, "END_OF_DATA"))
                pos = None

            equity_vals[t] = self.risk_manager.current_equity

        equity_curve = pd.Series(equity_vals, index=index, name="equity", dtype="float64")
        return self._build_report(equity_curve, trades)


    # ------------------------------------------------------------------
    def _build_report(self, equity_curve: pd.Series, trades: list[TradeRecord]) -> BacktestReport:
        cfg = self.config
        final_equity = float(self.risk_manager.current_equity)
        total_return_pct = (final_equity - cfg.initial_capital) / cfg.initial_capital * 100.0

        if len(equity_curve) > 0:
            peak = np.maximum.accumulate(np.concatenate([[cfg.initial_capital], equity_curve.to_numpy()]))
            vals = np.concatenate([[cfg.initial_capital], equity_curve.to_numpy()])
            dd = (peak - vals) / peak
            max_dd_pct = float(dd.max() * 100.0)
        else:
            max_dd_pct = 0.0

        returns = equity_curve.pct_change().dropna()
        if len(returns) < 2:
            sharpe = 0.0
        else:
            std = float(returns.std(ddof=1))
            if not np.isfinite(std) or std <= 1e-12:
                sharpe = 0.0
            else:
                sharpe = float(returns.mean() / std * math.sqrt(cfg.candles_per_year))

        total_trades = len(trades)
        wins = [tr.net_pnl for tr in trades if tr.net_pnl > 0]
        losses = [tr.net_pnl for tr in trades if tr.net_pnl < 0]
        win_rate = len(wins) / total_trades * 100.0 if total_trades else 0.0
        gross_win = float(sum(wins))
        gross_loss = abs(float(sum(losses)))
        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        else:
            profit_factor = float("inf") if gross_win > 0 else 0.0

        return BacktestReport(
            total_return_pct=float(total_return_pct),
            final_equity=final_equity,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            win_rate_pct=float(win_rate),
            profit_factor=float(profit_factor),
            total_trades=total_trades,
            total_fees_paid=float(sum(tr.total_fee for tr in trades)),
            equity_curve=equity_curve,
            trades=trades,
        )
