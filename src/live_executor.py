"""Bộ thực thi Paper/Live (LiveExecutor) đảm bảo Idempotency, Atomic State và Fault Tolerance."""
import logging
import time
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import numpy as np
import pandas as pd

from src import features
from src.backtester import TradeRecord
from src.config import TradingConfig
from src.data_loader import (
    DataLoader,
    ExchangeProtocol,
    drop_incomplete_last_candle,
    raw_ohlcv_to_dataframe,
)
from src.ml_filter import (
    MetaLabelFilter,
    _ensure_ml_feature_columns,
    build_meta_dataset,
)
from src.notifier import TelegramNotifier
from src.risk_manager import Position, RiskManager
from src.state_store import StateStore, _to_utc_timestamp, trade_record_from_dict
from src.strategy import Strategy, compute_breakeven_price, is_breakeven_triggered

logger = logging.getLogger(__name__)


class LiveExchangeProtocol(ExchangeProtocol, Protocol):
    """Giao diện sàn mở rộng cho chế độ Paper/Live."""

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]: ...


class LiveExecutor:
    """Thực thi chiến lược trên nến đóng hoàn chỉnh với cơ chế khôi phục trạng thái nguyên tử."""

    def __init__(
        self,
        config: TradingConfig = TradingConfig(),
        mode: Literal["paper", "live"] = "paper",
        data_loader: Optional[DataLoader] = None,
        strategy: Optional[Strategy] = None,
        risk_manager: Optional[RiskManager] = None,
        state_store: Optional[StateStore] = None,
        notifier: Optional[TelegramNotifier] = None,
        ml_filter: Optional[MetaLabelFilter] = None,
    ) -> None:
        if mode not in ("paper", "live"):
            raise ValueError(f"mode phải là 'paper' hoặc 'live' (nhận {mode!r}).")
        self.config = config
        self.mode: Literal["paper", "live"] = mode
        self.data_loader: Optional[DataLoader] = data_loader
        self.strategy: Strategy = strategy if strategy is not None else Strategy(config)
        self.risk_manager: RiskManager = (
            risk_manager if risk_manager is not None else RiskManager(config)
        )
        self.state_store: StateStore = (
            state_store if state_store is not None else StateStore(config.state_file_path)
        )
        self.notifier: TelegramNotifier = (
            notifier if notifier is not None else TelegramNotifier()
        )
        self.ml_filter: Optional[MetaLabelFilter] = ml_filter

        state = self.state_store.load(default_equity=config.initial_capital)
        self.open_position: Optional[Position] = self.state_store.sync_to_risk_manager(
            state, self.risk_manager
        )
        self.last_processed_candle_ts: Optional[pd.Timestamp] = (
            _to_utc_timestamp(state.last_processed_candle_ts)
            if state.last_processed_candle_ts is not None
            else None
        )
        self.trade_history: list[TradeRecord] = [
            trade_record_from_dict(item) for item in state.trade_history
        ]

    def _get_data_loader(self) -> DataLoader:
        if self.data_loader is None:
            self.data_loader = DataLoader(self.config)
        return self.data_loader

    def save_state(self) -> Path:
        """Chụp snapshot runtime hiện tại và ghi nguyên tử xuống đĩa."""
        snapshot = self.state_store.snapshot_from_runtime(
            risk_manager=self.risk_manager,
            open_position=self.open_position,
            last_processed_candle_ts=self.last_processed_candle_ts,
            trade_history=self.trade_history,
        )
        return self.state_store.save(snapshot)

    def _execute_live_order(
        self,
        side_str: str,
        qty: float,
        reduce_only: bool = False,
        max_retries: int = 2,
    ) -> Optional[dict[str, Any]]:
        loader = self._get_data_loader()
        exchange = loader.exchange
        if not hasattr(exchange, "create_order"):
            raise RuntimeError("Exchange object không hỗ trợ create_order cho chế độ live.")

        params: dict[str, Any] = {"reduceOnly": True} if reduce_only else {}
        last_exc: Optional[BaseException] = None
        for _ in range(max(1, max_retries)):
            try:
                order = exchange.create_order(  # type: ignore[attr-defined]
                    self.config.symbol,
                    "market",
                    side_str,
                    float(qty),
                    None,
                    params,
                )
                return order if isinstance(order, dict) else {}
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return None

    def _close_position(
        self,
        pos: Position,
        raw_exit: float,
        exit_time: pd.Timestamp,
        reason: str,
    ) -> TradeRecord:
        cfg = self.config
        side = pos.side
        exit_price = (
            raw_exit * (1.0 - cfg.slippage)
            if side == 1
            else raw_exit * (1.0 + cfg.slippage)
        )

        if self.mode == "live":
            order_side = "sell" if side == 1 else "buy"
            order = self._execute_live_order(order_side, pos.qty, reduce_only=True)
            if order:
                avg_px = order.get("average") or order.get("price")
                if avg_px is not None and float(avg_px) > 0:
                    exit_price = float(avg_px)

        gross = pos.qty * (exit_price - pos.entry_price) * side
        fee = pos.entry_fee + pos.qty * exit_price * cfg.taker_fee
        net = gross - fee

        self.risk_manager.update_after_trade_close(net, exit_time)
        record = TradeRecord(
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
        self.trade_history.append(record)
        self.open_position = None
        self.save_state()
        self.notifier.notify_trade_close(
            record, equity=self.risk_manager.current_equity, mode=self.mode
        )
        return record

    def _try_open_position(
        self,
        side: Literal[1, -1],
        ref_price: float,
        atr_val: float,
        ts: pd.Timestamp,
    ) -> Optional[Position]:
        cfg = self.config
        if (
            not np.isfinite(ref_price)
            or ref_price <= 0.0
            or not np.isfinite(atr_val)
            or atr_val <= 0.0
        ):
            return None

        if side == 1:
            entry = ref_price * (1.0 + cfg.slippage)
            sl = entry - cfg.sl_atr_multiple * atr_val
            tp = entry + cfg.tp_atr_multiple * atr_val
        else:
            entry = ref_price * (1.0 - cfg.slippage)
            sl = entry + cfg.sl_atr_multiple * atr_val
            tp = entry - cfg.tp_atr_multiple * atr_val

        pos = self.risk_manager.build_position(side, entry, sl, tp, ts)
        if pos is None:
            return None

        if self.mode == "live":
            order_side = "buy" if side == 1 else "sell"
            order = self._execute_live_order(order_side, pos.qty, reduce_only=False)
            if order:
                avg_px = order.get("average") or order.get("price")
                if avg_px is not None and float(avg_px) > 0:
                    fill_px = float(avg_px)
                    if side == 1:
                        new_sl = fill_px - cfg.sl_atr_multiple * atr_val
                        new_tp = fill_px + cfg.tp_atr_multiple * atr_val
                    else:
                        new_sl = fill_px + cfg.sl_atr_multiple * atr_val
                        new_tp = fill_px - cfg.tp_atr_multiple * atr_val
                    pos.entry_price = fill_px
                    pos.stop_loss = new_sl
                    pos.initial_sl = new_sl
                    pos.take_profit = new_tp
                    pos.notional = pos.qty * fill_px
                    pos.entry_fee = pos.notional * cfg.taker_fee

        self.open_position = pos
        self.save_state()
        self.notifier.notify_trade_open(pos, mode=self.mode)
        return pos

    def process_closed_candles(self, df: pd.DataFrame) -> bool:
        """Xử lý DataFrame các nến ĐÃ ĐÓNG hoàn chỉnh.

        Trả về True nếu có nến mới được xử lý, False nếu bị bỏ qua (rỗng, trùng nến, hoặc lỗi).
        """
        try:
            if df is None or len(df) == 0:
                return False

            latest_ts = _to_utc_timestamp(df.index[-1])
            if (
                self.last_processed_candle_ts is not None
                and latest_ts <= self.last_processed_candle_ts
            ):
                return False

            if (
                type(self.strategy) is Strategy
                and "signal" in df.columns
                and "atr" in df.columns
                and not all(col in df.columns for col in features.CORE_FEATURE_COLUMNS)
            ):
                sig_df = df.copy()
            else:
                sig_df = self.strategy.generate_signals(df)

            if len(sig_df) > 0:
                if self.config.use_ml:
                    sig_df = _ensure_ml_feature_columns(sig_df, self.config)
                last_row = sig_df.iloc[-1]
            else:
                last_row = df.iloc[-1].copy()

            self.risk_manager.on_new_Timestamp(latest_ts)

            c_open = float(last_row["open"])
            c_high = float(last_row["high"])
            c_low = float(last_row["low"])
            c_close = float(last_row["close"])

            raw_sig = last_row["signal"] if "signal" in last_row.index else 0
            sig = 0 if pd.isna(raw_sig) else int(raw_sig)

            raw_atr = last_row["atr"] if "atr" in last_row.index else np.nan
            if pd.isna(raw_atr) and all(
                c in df.columns for c in ("high", "low", "close")
            ):
                atr_series = features.compute_atr(df, self.config.atr_period)
                raw_atr = atr_series.iloc[-1] if len(atr_series) > 0 else np.nan
            atr_val = float(raw_atr) if not pd.isna(raw_atr) else float("nan")

            # 1) Quản trị vị thế đang mở (SL/TP -> Breakeven -> Time Exit -> Đảo chiều)
            if self.open_position is not None:
                pos = self.open_position
                if pos.side == 1:
                    hit_sl = c_low <= pos.stop_loss
                    hit_tp = c_high >= pos.take_profit
                    raw_sl = min(c_open, pos.stop_loss)
                    raw_tp = max(c_open, pos.take_profit)
                else:
                    hit_sl = c_high >= pos.stop_loss
                    hit_tp = c_low <= pos.take_profit
                    raw_sl = max(c_open, pos.stop_loss)
                    raw_tp = min(c_open, pos.take_profit)

                if hit_sl:
                    self._close_position(pos, raw_sl, latest_ts, "SL")
                elif hit_tp:
                    self._close_position(pos, raw_tp, latest_ts, "TP")
                else:
                    if self.config.use_breakeven and not pos.breakeven_activated:
                        if is_breakeven_triggered(pos, c_high, c_low, self.config):
                            be_sl = compute_breakeven_price(
                                pos.entry_price, pos.side, self.config
                            )
                            if pos.side == 1:
                                pos.stop_loss = max(pos.stop_loss, be_sl)
                            else:
                                pos.stop_loss = min(pos.stop_loss, be_sl)
                            pos.breakeven_activated = True

                    pos.bars_held += 1
                    if (
                        self.config.max_holding_bars > 0
                        and pos.bars_held >= self.config.max_holding_bars
                    ):
                        self._close_position(pos, c_close, latest_ts, "TIME_EXIT")
                    elif sig in (1, -1) and sig == -pos.side:
                        self._close_position(pos, c_close, latest_ts, "SIGNAL_REVERSE")

            # 2) Mở vị thế mới nếu đang flat và có tín hiệu hợp lệ
            if self.open_position is None and sig in (1, -1):
                allow_open = True
                if self.config.use_ml:
                    if self.ml_filter is None:
                        self.ml_filter = MetaLabelFilter(self.config)
                    if (
                        not getattr(self.ml_filter, "is_fitted_", False)
                        and len(sig_df) > 1
                    ):
                        history_df = sig_df.iloc[:-1]
                        X_hist, y_hist, exit_pos_hist = build_meta_dataset(
                            history_df, self.config
                        )
                        if len(X_hist) > 0:
                            valid_mask = exit_pos_hist.to_numpy(dtype=np.int64) < (
                                len(history_df) - 1
                            )
                            if int(valid_mask.sum()) > 0:
                                self.ml_filter.fit(
                                    X_hist.iloc[valid_mask], y_hist.iloc[valid_mask]
                                )
                    allow_open = bool(self.ml_filter.should_take(last_row))
                if allow_open and self.risk_manager.can_open_trade(latest_ts):
                    side_lit: Literal[1, -1] = 1 if sig == 1 else -1
                    self._try_open_position(side_lit, c_close, atr_val, latest_ts)

            self.last_processed_candle_ts = latest_ts
            self.save_state()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("LiveExecutor.process_closed_candles failed: %s", exc)
            self.notifier.notify_error("LiveExecutor.process_closed_candles", exc)
            return False

    def step(self) -> bool:
        """Thực hiện 1 chu kỳ tải nến -> lọc nến chưa đóng -> xử lý. Không bao giờ crash."""
        try:
            loader = self._get_data_loader()
            server_now_ms = int(loader.exchange.milliseconds())
            tf_ms = loader.timeframe_ms
            lookback = max(1, int(self.config.live_lookback_candles))
            since_ms = max(0, server_now_ms - (lookback + 2) * tf_ms)

            df = loader.fetch_historical_ohlcv(
                since_ms=since_ms,
                until_ms=server_now_ms,
                limit_per_call=max(1000, lookback + 5),
            )
            if len(df) == 0:
                raw_rows = loader.exchange.fetch_ohlcv(
                    self.config.symbol,
                    timeframe=self.config.timeframe,
                    since=None,
                    limit=lookback + 2,
                )
                df = raw_ohlcv_to_dataframe(
                    raw_rows, self.config.symbol, self.config.timeframe
                )

            closed_df = drop_incomplete_last_candle(
                df, self.config.timeframe, server_now_ms
            )
            return self.process_closed_candles(closed_df)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LiveExecutor.step failed: %s", exc)
            self.notifier.notify_error("LiveExecutor.step", exc)
            return False

    def run_loop(
        self,
        once: bool = False,
        max_iterations: Optional[int] = None,
    ) -> int:
        """Vòng lặp thực thi 24/7. Nếu once=True thì gọi step() đúng 1 lần rồi trả về 0."""
        if once:
            self.step()
            return 0

        iterations = 0
        try:
            while True:
                self.step()
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                time.sleep(max(0.0, float(self.config.poll_interval_seconds)))
        except KeyboardInterrupt:
            logger.info("Nhận tín hiệu dừng từ người dùng (KeyboardInterrupt).")
        return 0
