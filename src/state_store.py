"""Lưu trữ trạng thái nguyên tử (Atomic JSON StateStore) cho vận hành 24/7."""
import dataclasses
import json
import logging
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.backtester import TradeRecord
from src.risk_manager import Position, RiskManager

logger = logging.getLogger(__name__)


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
    carry_sleeve: dict[str, Any] = field(default_factory=dict)  # Phân bổ nhánh yield sleeve
    momentum_overlay: dict[str, Any] = field(default_factory=dict)  # Phân bổ nhánh alpha momentum overlay
    carry_orders: list[dict[str, Any]] = field(default_factory=list)
    overlay_orders: list[dict[str, Any]] = field(default_factory=list)
    overlay_positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    executor_functional: bool = False
    initial_capital: float = 400.0
    peak_equity: float = 400.0
    strategy: Optional[str] = None
    kill_switch_triggered: bool = False
    kill_switch_pending_close: bool = False
    drift_unrepairable_symbols: dict[str, float] = field(default_factory=dict)


class StateStore:
    """Quản lý lưu trữ và khôi phục trạng thái bot với cơ chế ghi nguyên tử (.tmp -> os.replace)."""

    def __init__(self, file_path: Path | str = Path("data/state/bot_state.json")) -> None:
        self.file_path = Path(file_path)

    def save(self, state: BotState | dict[str, Any]) -> Path:
        """Ghi nguyên tử trạng thái ra file JSON qua file tạm cùng thư mục."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(state, BotState):
            payload = dataclasses.asdict(state)
        else:
            payload = dict(state)
            if "kill_switch_pending_close" not in payload:
                payload["kill_switch_pending_close"] = False
            if "drift_unrepairable_symbols" not in payload:
                payload["drift_unrepairable_symbols"] = {}
        tmp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        bak_path = self.file_path.with_suffix(self.file_path.suffix + ".bak")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        if self.file_path.exists():
            try:
                if self.file_path.stat().st_size > 0:
                    with open(self.file_path, "r", encoding="utf-8") as existing_f:
                        existing_obj = json.load(existing_f)
                    if isinstance(existing_obj, dict):
                        shutil.copy2(self.file_path, bak_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip backing up existing state file %s: %s", self.file_path, exc)

        for attempt in range(5):
            try:
                os.replace(tmp_path, self.file_path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)

        if not bak_path.exists() and self.file_path.exists():
            try:
                shutil.copy2(self.file_path, bak_path)
            except Exception:  # noqa: BLE001
                pass

        return self.file_path

    def load(
        self,
        default_equity: float = 400.0,
        normalize_carry: bool = False,
        initial_capital: Optional[float] = None,
        strategy: Optional[str] = None,
        reset: bool = False,
    ) -> BotState:
        """Đọc trạng thái từ đĩa; nếu file chưa tồn tại thì trả về BotState mặc định."""
        cap = float(initial_capital) if initial_capital is not None else float(default_equity)
        bak_path = self.file_path.with_suffix(self.file_path.suffix + ".bak")

        if reset or (not self.file_path.exists() and not bak_path.exists()):
            return BotState(
                daily_start_equity=cap,
                current_equity=cap,
                initial_capital=cap,
                peak_equity=cap,
                strategy=strategy,
                kill_switch_triggered=False,
                kill_switch_pending_close=False,
                drift_unrepairable_symbols={},
            )

        raw: Optional[dict[str, Any]] = None
        if self.file_path.exists():
            try:
                if self.file_path.stat().st_size == 0:
                    raise ValueError("Empty state file (0 bytes)")
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("State file root is not a JSON object")
                raw = loaded
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                logger.warning(
                    "Primary state file %s is corrupted or unreadable (%s); checking backup %s",
                    self.file_path,
                    exc,
                    bak_path,
                )

        if raw is None:
            if bak_path.exists():
                try:
                    if bak_path.stat().st_size == 0:
                        raise ValueError("Empty backup state file (0 bytes)")
                    with open(bak_path, "r", encoding="utf-8") as f:
                        loaded_bak = json.load(f)
                    if not isinstance(loaded_bak, dict):
                        raise ValueError("Backup state file root is not a JSON object")
                    raw = loaded_bak
                    logger.warning(
                        "WARNING: RECOVERED_STATE_FROM_BACKUP (%s -> %s)",
                        bak_path,
                        self.file_path,
                    )
                    try:
                        shutil.copy2(bak_path, self.file_path)
                    except OSError:
                        pass
                except (json.JSONDecodeError, ValueError, OSError) as bak_exc:
                    logger.error(
                        "Backup state file %s is also corrupted (%s); returning default BotState",
                        bak_path,
                        bak_exc,
                    )
                    return BotState(
                        daily_start_equity=cap,
                        current_equity=cap,
                        initial_capital=cap,
                        peak_equity=cap,
                        strategy=strategy,
                        kill_switch_triggered=False,
                        kill_switch_pending_close=False,
                        drift_unrepairable_symbols={},
                    )
            else:
                logger.warning(
                    "No valid backup file %s found for corrupted %s; returning default BotState",
                    bak_path,
                    self.file_path,
                )
                return BotState(
                    daily_start_equity=cap,
                    current_equity=cap,
                    initial_capital=cap,
                    peak_equity=cap,
                    strategy=strategy,
                    kill_switch_triggered=False,
                    kill_switch_pending_close=False,
                    drift_unrepairable_symbols={},
                )

        had_explicit_initial = "initial_capital" in raw

        raw_unrepairable = raw.get("drift_unrepairable_symbols", {})
        drift_unrepairable: dict[str, float] = {}
        if isinstance(raw_unrepairable, dict):
            for k, v in raw_unrepairable.items():
                try:
                    drift_unrepairable[str(k)] = float(v)
                except (ValueError, TypeError):
                    continue

        if strategy is not None and raw.get("strategy") is not None and raw.get("strategy") != strategy:
            sticky_kill = bool(raw.get("kill_switch_triggered", False))
            sticky_pending_close = bool(raw.get("kill_switch_pending_close", False))
            if sticky_kill or sticky_pending_close:
                logger.error("State cua chien luoc khac dang co kill-switch/pending-close=True -> giu nguyen co")
            logger.warning(
                "Strategy mismatch in state file: expected %s, found %s. Resetting state.",
                strategy,
                raw.get("strategy"),
            )
            return BotState(
                daily_start_equity=cap,
                current_equity=cap,
                initial_capital=cap,
                peak_equity=cap,
                strategy=strategy,
                kill_switch_triggered=sticky_kill,
                kill_switch_pending_close=sticky_pending_close,
                drift_unrepairable_symbols=drift_unrepairable,
            )

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
                        pos_dict = normalize_carry_position(pos_dict, default_capital=cap)
                    carry_positions[str(k)] = pos_dict
        raw_carry_ts = raw.get("carry_last_reconcile_ts")
        carry_last_reconcile_ts = str(raw_carry_ts) if raw_carry_ts is not None else None

        raw_carry_sleeve = raw.get("carry_sleeve") or {}
        carry_sleeve: dict[str, Any] = dict(raw_carry_sleeve) if isinstance(raw_carry_sleeve, dict) else {}
        raw_momentum_overlay = raw.get("momentum_overlay") or {}
        momentum_overlay: dict[str, Any] = dict(raw_momentum_overlay) if isinstance(raw_momentum_overlay, dict) else {}

        raw_carry_orders = raw.get("carry_orders") or []
        carry_orders: list[dict[str, Any]] = [
            dict(x) for x in raw_carry_orders if isinstance(x, dict)
        ] if isinstance(raw_carry_orders, list) else []

        raw_overlay_orders = raw.get("overlay_orders") or []
        overlay_orders: list[dict[str, Any]] = [
            dict(x) for x in raw_overlay_orders if isinstance(x, dict)
        ] if isinstance(raw_overlay_orders, list) else []

        raw_overlay_positions = raw.get("overlay_positions") or {}
        overlay_positions: dict[str, dict[str, Any]] = {
            str(k): dict(v) for k, v in raw_overlay_positions.items() if isinstance(v, dict)
        } if isinstance(raw_overlay_positions, dict) else {}

        executor_functional: bool = bool(raw.get("executor_functional", False))

        raw_init_cap = raw.get("initial_capital")
        raw_peak_eq = raw.get("peak_equity")
        raw_strat = raw.get("strategy")
        raw_kill = raw.get("kill_switch_triggered", False)
        raw_pending_close = raw.get("kill_switch_pending_close", False)

        cur_eq = float(raw.get("current_equity", cap))
        init_cap = float(raw_init_cap) if raw_init_cap is not None else cap
        peak_eq = float(raw_peak_eq) if raw_peak_eq is not None else max(cur_eq, init_cap)
        strat = str(raw_strat) if raw_strat is not None else None
        kill_switch = bool(raw_kill)
        kill_switch_pending = bool(raw_pending_close)

        state = BotState(
            open_position=dict(open_pos) if isinstance(open_pos, dict) else None,
            daily_realized_pnl=float(raw.get("daily_realized_pnl", 0.0)),
            daily_start_equity=float(raw.get("daily_start_equity", cap)),
            current_equity=cur_eq,
            current_utc_date=str(utc_date) if utc_date is not None else None,
            consecutive_losses=int(raw.get("consecutive_losses", 0)),
            circuit_breaker_tripped=bool(raw.get("circuit_breaker_tripped", False)),
            last_processed_candle_ts=str(last_ts) if last_ts is not None else None,
            trade_history=[trade_record_to_dict(item) for item in raw_history],
            carry_positions=carry_positions,
            carry_last_reconcile_ts=carry_last_reconcile_ts,
            carry_sleeve=carry_sleeve,
            momentum_overlay=momentum_overlay,
            carry_orders=carry_orders,
            overlay_orders=overlay_orders,
            overlay_positions=overlay_positions,
            executor_functional=executor_functional,
            initial_capital=init_cap,
            peak_equity=peak_eq,
            strategy=strat,
            kill_switch_triggered=kill_switch,
            kill_switch_pending_close=kill_switch_pending,
            drift_unrepairable_symbols=drift_unrepairable,
        )

        if initial_capital is not None:
            has_open_positions = bool(state.carry_positions or state.overlay_positions or state.open_position)
            if not has_open_positions:
                if not had_explicit_initial:
                    state.initial_capital = cap
                    state.current_equity = cap
                    state.daily_start_equity = cap
                    state.peak_equity = max(state.peak_equity, cap)
                elif abs(state.current_equity - state.initial_capital) < 1e-9:
                    state.initial_capital = cap
                    state.current_equity = cap
                    state.daily_start_equity = cap
                    state.peak_equity = max(state.peak_equity, cap)
                else:
                    old_init = state.initial_capital
                    state.initial_capital = cap
                    state.peak_equity = max(state.peak_equity, state.current_equity)
                    logger.warning(
                        "Khong dong bo equity vi da co lai/lo thuc hien (current_equity=%.2f != initial_capital cu=%.2f). Giu nguyen current_equity va daily_start_equity, cap nhat initial_capital=%.2f",
                        state.current_equity,
                        old_init,
                        cap,
                    )
            else:
                raw_init = raw.get("initial_capital")
                try:
                    file_init_cap = float(raw_init) if raw_init is not None else None
                except (ValueError, TypeError):
                    file_init_cap = None

                if file_init_cap is not None and file_init_cap > 0:
                    state.initial_capital = file_init_cap
                else:
                    state.initial_capital = state.current_equity

                state.peak_equity = max(state.peak_equity, state.current_equity)

                if abs(state.current_equity - cap) > 1e-9 or abs(state.initial_capital - cap) > 1e-9:
                    logger.warning(
                        "State has open positions; keeping existing equity %.2f and capital %.2f (differs from CLI capital %.2f)",
                        state.current_equity,
                        state.initial_capital,
                        cap,
                    )

        if strategy is not None and state.strategy is None:
            state.strategy = strategy

        return state

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
