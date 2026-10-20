"""Lưu trữ trạng thái nguyên tử (Atomic JSON StateStore) cho vận hành 24/7."""
import dataclasses
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.backtester import TradeRecord
from src.risk_manager import Position, RiskManager


def _to_utc_timestamp(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def position_to_dict(pos: Position) -> dict[str, Any]:
    """Chuyển đối tượng Position sang dict tuần tự hóa JSON được."""
    raw = dataclasses.asdict(pos)
    entry_ts = _to_utc_timestamp(pos.entry_time)
    return {
        "symbol": str(raw["symbol"]),
        "side": int(raw["side"]),
        "entry_price": float(raw["entry_price"]),
        "stop_loss": float(raw["stop_loss"]),
        "take_profit": float(raw["take_profit"]),
        "qty": float(raw["qty"]),
        "entry_time": entry_ts.isoformat(),
        "entry_fee": float(raw["entry_fee"]),
        "notional": float(raw["notional"]),
        "initial_sl": float(raw.get("initial_sl", raw["stop_loss"])),
        "breakeven_activated": bool(raw.get("breakeven_activated", False)),
        "bars_held": int(raw.get("bars_held", 0)),
        "trailing_activated": bool(raw.get("trailing_activated", False)),
        "highest_price_since_entry": float(raw.get("highest_price_since_entry", raw["entry_price"])),
        "lowest_price_since_entry": float(raw.get("lowest_price_since_entry", raw["entry_price"])),
        "entry_order_type": "limit" if str(raw.get("entry_order_type", "market")) == "limit" else "market",
    }


def position_from_dict(data: Optional[dict[str, Any]]) -> Optional[Position]:
    """Khôi phục đối tượng Position từ dict (entry_time là pd.Timestamp tz-aware UTC)."""
    if not data:
        return None
    side_val = 1 if int(data["side"]) == 1 else -1
    stop_loss = float(data["stop_loss"])
    initial_sl = float(data.get("initial_sl", stop_loss))
    if initial_sl == 0.0:
        initial_sl = stop_loss
    return Position(
        symbol=str(data["symbol"]),
        side=side_val,
        entry_price=float(data["entry_price"]),
        stop_loss=stop_loss,
        take_profit=float(data["take_profit"]),
        qty=float(data["qty"]),
        entry_time=_to_utc_timestamp(data["entry_time"]),
        entry_fee=float(data["entry_fee"]),
        notional=float(data["notional"]),
        initial_sl=initial_sl,
        breakeven_activated=bool(data.get("breakeven_activated", False)),
        bars_held=int(data.get("bars_held", 0)),
        trailing_activated=bool(data.get("trailing_activated", False)),
        highest_price_since_entry=float(data.get("highest_price_since_entry", 0.0) or 0.0),
        lowest_price_since_entry=float(data.get("lowest_price_since_entry", 0.0) or 0.0),
        entry_order_type="limit" if str(data.get("entry_order_type", "market")) == "limit" else "market",
    )


def trade_record_to_dict(record: TradeRecord | dict[str, Any]) -> dict[str, Any]:
    """Chuyển TradeRecord sang dict JSON-serializable."""
    raw = dataclasses.asdict(record) if isinstance(record, TradeRecord) else dict(record)
    return {
        "entry_time": _to_utc_timestamp(raw["entry_time"]).isoformat(),
        "exit_time": _to_utc_timestamp(raw["exit_time"]).isoformat(),
        "side": int(raw["side"]),
        "entry_price": float(raw["entry_price"]),
        "exit_price": float(raw["exit_price"]),
        "qty": float(raw["qty"]),
        "gross_pnl": float(raw["gross_pnl"]),
        "total_fee": float(raw["total_fee"]),
        "net_pnl": float(raw["net_pnl"]),
        "exit_reason": str(raw["exit_reason"]),
    }


def trade_record_from_dict(data: dict[str, Any]) -> TradeRecord:
    """Khôi phục TradeRecord từ dict."""
    side_val = 1 if int(data["side"]) == 1 else -1
    return TradeRecord(
        entry_time=_to_utc_timestamp(data["entry_time"]),
        exit_time=_to_utc_timestamp(data["exit_time"]),
        side=side_val,
        entry_price=float(data["entry_price"]),
        exit_price=float(data["exit_price"]),
        qty=float(data["qty"]),
        gross_pnl=float(data["gross_pnl"]),
        total_fee=float(data["total_fee"]),
        net_pnl=float(data["net_pnl"]),
        exit_reason=str(data["exit_reason"]),  # type: ignore[arg-type]
    )


def default_leg_state(
    capital: float = 3000.0,
    exchange_long: str = "binance_spot",
    exchange_short: str = "binance",
) -> dict[str, Any]:
    """Trả về dict chứa các trường Chân 1/Chân 2 ở trạng thái IDLE mặc định."""
    return {
        "leg1_status": "NONE",
        "leg1_order_id": None,
        "leg1_filled_qty": 0.0,
        "leg2_hedged_qty": 0.0,
        "exchange_long": str(exchange_long),
        "exchange_short": str(exchange_short),
        "capital": float(capital),
        "execution_state": "IDLE",
    }


def normalize_carry_position(
    pos: dict[str, Any],
    default_capital: float = 3000.0,
) -> dict[str, Any]:
    """Bổ sung các trường Chân 1/Chân 2 còn thiếu (giữ nguyên giá trị cũ nếu có), không xóa trường lạ."""
    out = dict(pos)
    defaults = default_leg_state(
        capital=float(out.get("capital", default_capital)),
        exchange_long=str(out.get("exchange_long", "binance_spot")),
        exchange_short=str(out.get("exchange_short", "binance")),
    )
    for k, v in defaults.items():
        if k not in out:
            out[k] = v
    return out


@dataclass
class BotState:
    open_position: Optional[dict[str, Any]] = None
    daily_realized_pnl: float = 0.0
    daily_start_equity: float = 400.0
    current_equity: float = 400.0
    current_utc_date: Optional[str] = None
    consecutive_losses: int = 0
    circuit_breaker_tripped: bool = False
    last_processed_candle_ts: Optional[str] = None
    trade_history: list[dict[str, Any]] = field(default_factory=list)
    carry_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    carry_last_reconcile_ts: Optional[str] = None


class StateStore:
    """Quản lý lưu trữ và khôi phục trạng thái bot với cơ chế ghi nguyên tử (.tmp -> os.replace)."""

    def __init__(self, file_path: Path | str = Path("data/state/bot_state.json")) -> None:
        self.file_path = Path(file_path)

    def save(self, state: BotState | dict[str, Any]) -> Path:
        """Ghi nguyên tử trạng thái ra file JSON qua file tạm cùng thư mục."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dataclasses.asdict(state) if isinstance(state, BotState) else dict(state)
        tmp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, self.file_path)
        return self.file_path

    def load(self, default_equity: float = 400.0, normalize_carry: bool = False) -> BotState:
        """Đọc trạng thái từ đĩa; nếu file chưa tồn tại thì trả về BotState mặc định."""
        if not self.file_path.exists():
            eq = float(default_equity)
            return BotState(daily_start_equity=eq, current_equity=eq)

        with open(self.file_path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        eq = float(default_equity)
        open_pos = raw.get("open_position")
        utc_date = raw.get("current_utc_date")
        last_ts = raw.get("last_processed_candle_ts")
        raw_history = raw.get("trade_history") or []

        raw_carry = raw.get("carry_positions") or {}
        carry_positions: dict[str, dict[str, Any]] = {}
        if isinstance(raw_carry, dict):
            for k, v in raw_carry.items():
                if isinstance(v, dict):
                    pos_dict = dict(v)
                    if normalize_carry:
                        pos_dict = normalize_carry_position(pos_dict, default_capital=eq)
                    carry_positions[str(k)] = pos_dict
        raw_carry_ts = raw.get("carry_last_reconcile_ts")
        carry_last_reconcile_ts = str(raw_carry_ts) if raw_carry_ts is not None else None

        return BotState(
            open_position=dict(open_pos) if isinstance(open_pos, dict) else None,
            daily_realized_pnl=float(raw.get("daily_realized_pnl", 0.0)),
            daily_start_equity=float(raw.get("daily_start_equity", eq)),
            current_equity=float(raw.get("current_equity", eq)),
            current_utc_date=str(utc_date) if utc_date is not None else None,
            consecutive_losses=int(raw.get("consecutive_losses", 0)),
            circuit_breaker_tripped=bool(raw.get("circuit_breaker_tripped", False)),
            last_processed_candle_ts=str(last_ts) if last_ts is not None else None,
            trade_history=[trade_record_to_dict(item) for item in raw_history],
            carry_positions=carry_positions,
            carry_last_reconcile_ts=carry_last_reconcile_ts,
        )

    def sync_to_risk_manager(self, state: BotState, risk_manager: RiskManager) -> Optional[Position]:
        """Khôi phục 100% trạng thái vào risk_manager và trả về Position đang mở (nếu có)."""
        risk_manager.current_equity = float(state.current_equity)
        risk_manager.daily_start_equity = float(state.daily_start_equity)
        risk_manager.daily_realized_pnl = float(state.daily_realized_pnl)
        risk_manager.current_utc_date = (
            date.fromisoformat(state.current_utc_date)
            if state.current_utc_date is not None
            else None
        )
        risk_manager.consecutive_losses = int(state.consecutive_losses)
        risk_manager.circuit_breaker_tripped = bool(state.circuit_breaker_tripped)
        return position_from_dict(state.open_position)

    def snapshot_from_runtime(
        self,
        risk_manager: RiskManager,
        open_position: Optional[Position],
        last_processed_candle_ts: Optional[pd.Timestamp | str],
        trade_history: Sequence[TradeRecord | dict[str, Any]] = (),
    ) -> BotState:
        """Tạo snapshot BotState từ trạng thái runtime hiện hành."""
        last_ts_str: Optional[str] = None
        if last_processed_candle_ts is not None:
            last_ts_str = _to_utc_timestamp(last_processed_candle_ts).isoformat()

        utc_date_str: Optional[str] = None
        if risk_manager.current_utc_date is not None:
            utc_date_str = risk_manager.current_utc_date.isoformat()

        return BotState(
            open_position=position_to_dict(open_position) if open_position is not None else None,
            daily_realized_pnl=float(risk_manager.daily_realized_pnl),
            daily_start_equity=float(risk_manager.daily_start_equity),
            current_equity=float(risk_manager.current_equity),
            current_utc_date=utc_date_str,
            consecutive_losses=int(risk_manager.consecutive_losses),
            circuit_breaker_tripped=bool(risk_manager.circuit_breaker_tripped),
            last_processed_candle_ts=last_ts_str,
            trade_history=[trade_record_to_dict(rec) for rec in trade_history],
        )
