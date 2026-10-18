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
    stop_loss: float                  # Mức giá cắt lỗ hiện hành (có thể được dời về breakeven)
    take_profit: float                # Mức giá chốt lời
    qty: float                        # Khối lượng hợp đồng (đã làm tròn theo step_size)
    entry_time: pd.Timestamp          # Thời điểm mở lệnh (tz-aware UTC)
    entry_fee: float                  # Phí mở lệnh đã trả (USDT)
    notional: float                   # Giá trị danh nghĩa lúc vào lệnh = qty * entry_price
    initial_sl: float = 0.0           # Mức SL ban đầu lúc mở lệnh (để tính initial_risk = |entry - initial_sl|)
    breakeven_activated: bool = False # Cờ đánh dấu đã kích hoạt dời SL về hòa vốn
    bars_held: int = 0                # Số nến đã nắm giữ vị thế
    trailing_activated: bool = False          # Cờ đánh dấu đã vượt ngưỡng trailing_activation_r
    highest_price_since_entry: float = 0.0    # Đỉnh cao nhất kể từ lúc vào lệnh (cho Long trailing)
    lowest_price_since_entry: float = 0.0     # Đáy thấp nhất kể từ lúc vào lệnh (cho Short trailing)
    entry_order_type: Literal["market", "limit"] = "market"

    def __post_init__(self) -> None:
        if self.initial_sl == 0.0:
            self.initial_sl = self.stop_loss
        if self.highest_price_since_entry == 0.0:
            self.highest_price_since_entry = self.entry_price
        if self.lowest_price_since_entry == 0.0:
            self.lowest_price_since_entry = self.entry_price


def enforce_min_stop_distance(
    side: Literal[1, -1],
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    config: TradingConfig = TradingConfig(),
) -> Optional[tuple[float, float]]:
    """Áp khoảng cách SL tối thiểu. Trả (sl, tp) đã điều chỉnh hoặc None nếu bị từ chối/không hợp lệ."""
    if entry_price <= 0 or stop_loss <= 0:
        return None
    if side == 1 and not stop_loss < entry_price:
        return None
    if side == -1 and not stop_loss > entry_price:
        return None
    d_raw = abs(entry_price - stop_loss)
    d_min = entry_price * config.min_stop_distance_pct
    if config.min_stop_distance_pct <= 0.0 or d_raw >= d_min - 1e-12:
        return stop_loss, take_profit
    if config.min_stop_action == "reject":
        return None
    sl_new = entry_price - d_min if side == 1 else entry_price + d_min
    if sl_new <= 0:
        return None
    rr = abs(take_profit - entry_price) / d_raw
    tp_new = take_profit
    if abs(take_profit - entry_price) < rr * d_min:
        tp_new = entry_price + side * rr * d_min
    return sl_new, tp_new


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
        self.consecutive_losses: int = 0

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

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        is_maker_entry: bool = False,
    ) -> Optional[float]:
        if self.current_equity <= 0.0:
            return None
        if entry_price <= 0 or stop_loss <= 0:
            return None
        distance = abs(entry_price - stop_loss)
        if distance == 0:
            return None
        cfg = self.config
        risk_amount = self.current_equity * cfg.risk_per_trade_pct
        if cfg.use_fee_aware_sizing:
            if is_maker_entry:
                cost_rate = cfg.maker_fee + cfg.taker_fee + 2.0 * cfg.slippage
            else:
                cost_rate = 2.0 * (cfg.taker_fee + cfg.slippage)
            effective_risk = distance + entry_price * cost_rate
        else:
            effective_risk = distance
        raw_qty = risk_amount / effective_risk
        max_qty_by_leverage = (self.current_equity * cfg.max_effective_leverage) / entry_price
        clamped_qty = min(raw_qty, max_qty_by_leverage)  # CLAMP, không từ chối

        step = cfg.step_size
        steps = math.floor(clamped_qty / step + _EPS)
        qty = round(steps * step, 12)
        # Phòng sai số dấu phẩy động làm vượt trần đòn bẩy (và ngân sách rủi ro khi fee-aware)
        while qty > 0 and (
            qty > max_qty_by_leverage + 1e-12
            or (cfg.use_fee_aware_sizing and qty * effective_risk > risk_amount + 1e-9)
        ):
            steps -= 1
            qty = round(steps * step, 12)

        if qty <= 0 or qty * entry_price < cfg.min_notional:
            return None
        return qty

    def build_position(
        self,
        side: Literal[1, -1],
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_time: pd.Timestamp,
        is_maker_entry: bool = False,
    ) -> Optional[Position]:
        utc_entry_time = _to_utc_timestamp(entry_time)
        if not self.can_open_trade(utc_entry_time):
            return None
        adjusted = enforce_min_stop_distance(side, entry_price, stop_loss, take_profit, self.config)
        if adjusted is None:
            return None
        sl_adj, tp_adj = adjusted
        qty = self.calculate_position_size(entry_price, sl_adj, is_maker_entry=is_maker_entry)
        if qty is None:
            return None
        notional = qty * entry_price
        fee_rate = self.config.maker_fee if is_maker_entry else self.config.taker_fee
        return Position(
            symbol=self.config.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=sl_adj,
            take_profit=tp_adj,
            qty=qty,
            entry_time=utc_entry_time,
            entry_fee=notional * fee_rate,
            notional=notional,
            initial_sl=sl_adj,
            breakeven_activated=False,
            bars_held=0,
            trailing_activated=False,
            highest_price_since_entry=entry_price,
            lowest_price_since_entry=entry_price,
            entry_order_type="limit" if is_maker_entry else "market",
        )

    def update_after_trade_close(self, net_pnl: float, exit_time: pd.Timestamp) -> None:
        self.on_new_Timestamp(exit_time)
        self.current_equity += net_pnl
        self.daily_realized_pnl += net_pnl
        if net_pnl < 0.0:
            self.consecutive_losses += 1
        elif net_pnl > 0.0:
            self.consecutive_losses = 0
        if self._limit_hit():
            self.circuit_breaker_tripped = True

