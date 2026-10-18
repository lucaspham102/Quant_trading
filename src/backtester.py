"""Backtester: khớp lệnh next-bar open, SL/TP trong nến (SL ưu tiên), phí taker 2 chiều + slippage."""
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from src.config import TradingConfig
from src.risk_manager import Position, RiskManager
from src.strategy import Strategy, compute_breakeven_price, is_breakeven_triggered

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

        trades: list[TradeRecord] = []
        equity_vals = np.empty(n, dtype="float64")
        pos: Optional[Position] = None

        for t in range(n):
            ts = index[t]
            prev_sig = int(signals[t - 1]) if t > 0 and not pd.isna(signals[t - 1]) else 0

            # a) Đảo chiều tại open[t] nếu tín hiệu t-1 ngược vị thế đang mở
            if pos is not None and prev_sig != 0 and prev_sig == -pos.side:
                trades.append(self._close_position(pos, opens[t], ts, "SIGNAL_REVERSE"))
                pos = None

            # b) Mở lệnh tại open[t] khi tín hiệu t-1 != 0
            opened_this_bar = False
            if pos is None and prev_sig != 0:
                pos = self._try_open(prev_sig, opens[t], atrs[t - 1], ts)
                opened_this_bar = pos is not None

            # c) Kiểm tra SL/TP trong [low[t], high[t]] (SL ưu tiên, xử lý gap qua open[t])
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

            # d) Cập nhật Breakeven Stop cho nến t+1 (nếu vị thế vẫn mở sau bước c)
            if pos is not None and self.config.use_breakeven and not pos.breakeven_activated:
                if is_breakeven_triggered(pos, highs[t], lows[t], self.config):
                    be_sl = compute_breakeven_price(pos.entry_price, pos.side, self.config)
                    if pos.side == 1:
                        pos.stop_loss = max(pos.stop_loss, be_sl)
                    else:
                        pos.stop_loss = min(pos.stop_loss, be_sl)
                    pos.breakeven_activated = True

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
