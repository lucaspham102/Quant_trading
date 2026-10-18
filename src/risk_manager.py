"""Quản trị rủi ro: tính khối lượng lệnh, kẹp đòn bẩy, min_notional và Circuit Breaker theo ngày UTC."""
import math
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import pandas as pd

from src.config import TradingConfig

_EPS = 1e-9


@dataclass
class Position:
    symbol: str
    side: Literal[1, -1]              # 1: Long, -1: Short
    entry_price: float                # Giá khớp thực tế (đã tính slippage)
    stop_loss: float                  # Mức giá cắt lỗ
    take_profit: float                # Mức giá chốt lời
    qty: float                        # Khối lượng hợp đồng (đã làm tròn theo step_size)
    entry_time: pd.Timestamp          # Thời điểm mở lệnh (tz-aware UTC)
    entry_fee: float                  # Phí mở lệnh đã trả (USDT)
    notional: float                   # Giá trị danh nghĩa lúc vào lệnh = qty * entry_price


def _to_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class RiskManager:
    def __init__(self, config: TradingConfig = TradingConfig()) -> None:
        self.config = config
        self.current_equity: float = config.initial_capital
        self.current_utc_date: Optional[date] = None
        self.daily_start_equity: float = config.initial_capital
        self.daily_realized_pnl: float = 0.0
        self.circuit_breaker_tripped: bool = False

    def _daily_loss_limit(self) -> float:
        return self.daily_start_equity * self.config.max_daily_drawdown_pct

    def _limit_hit(self) -> bool:
        return self.daily_realized_pnl <= -self._daily_loss_limit() + _EPS

    def on_new_Timestamp(self, ts: pd.Timestamp) -> None:
        """Sang ngày UTC mới thì reset bộ đếm ngày và mở khóa circuit breaker."""
        ts = _to_utc_timestamp(ts)
        day = ts.date()
        if self.current_utc_date != day:
            self.current_utc_date = day
            self.daily_start_equity = self.current_equity
            self.daily_realized_pnl = 0.0
            self.circuit_breaker_tripped = False

    def can_open_trade(self, current_time: pd.Timestamp) -> bool:
        self.on_new_Timestamp(current_time)
        if self.current_equity <= 0.0:
            return False
        if self.circuit_breaker_tripped:
            return False
        if self._limit_hit():
            self.circuit_breaker_tripped = True
            return False
        return True

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> Optional[float]:
        if self.current_equity <= 0.0:
            return None
        if entry_price <= 0 or stop_loss <= 0:
            return None
        distance = abs(entry_price - stop_loss)
        if distance == 0:
            return None
        risk_amount = self.current_equity * self.config.risk_per_trade_pct
        raw_qty = risk_amount / distance
        max_qty_by_leverage = (self.current_equity * self.config.max_effective_leverage) / entry_price
        clamped_qty = min(raw_qty, max_qty_by_leverage)  # CLAMP, không từ chối

        step = self.config.step_size
        steps = math.floor(clamped_qty / step + _EPS)
        qty = round(steps * step, 12)
        # Phòng sai số dấu phẩy động làm vượt trần đòn bẩy
        while qty > 0 and qty > max_qty_by_leverage + 1e-12:
            steps -= 1
            qty = round(steps * step, 12)

        if qty <= 0 or qty * entry_price < self.config.min_notional:
            return None
        return qty

    def build_position(
        self,
        side: Literal[1, -1],
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_time: pd.Timestamp,
    ) -> Optional[Position]:
        utc_entry_time = _to_utc_timestamp(entry_time)
        if not self.can_open_trade(utc_entry_time):
            return None
        qty = self.calculate_position_size(entry_price, stop_loss)
        if qty is None:
            return None
        notional = qty * entry_price
        return Position(
            symbol=self.config.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            qty=qty,
            entry_time=utc_entry_time,
            entry_fee=notional * self.config.taker_fee,
            notional=notional,
        )

    def update_after_trade_close(self, net_pnl: float, exit_time: pd.Timestamp) -> None:
        self.on_new_Timestamp(exit_time)
        self.current_equity += net_pnl
        self.daily_realized_pnl += net_pnl
        if self._limit_hit():
            self.circuit_breaker_tripped = True
