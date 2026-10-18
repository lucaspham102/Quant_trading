"""Unit test cho StateStore, TelegramNotifier, LiveExecutor và CLI run_live.py (100% offline)."""
import dataclasses
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

import run_live
from src.config import TradingConfig
from src.data_loader import DataLoader, timeframe_to_ms
from src.live_executor import LiveExecutor
from src.ml_filter import ML_FEATURE_COLUMNS, MetaLabelFilter
from src.notifier import TelegramNotifier
from src.risk_manager import RiskManager
from src.state_store import StateStore
from src.strategy import compute_breakeven_price

_TF_15M_MS = timeframe_to_ms("15m")
_T0_MS = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC


class PassthroughStrategy:
    """Chiến lược giả lập trả nguyên DataFrame đã gắn sẵn signal và atr để kiểm thử định sẵn."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


def _make_signal_df(rows: list[tuple[float, float, float, float, int, float]]) -> pd.DataFrame:
    """rows: (open, high, low, close, signal, atr)."""
    idx = pd.date_range(
        "2025-01-01 00:00:00",
        periods=len(rows),
        freq="15min",
        tz="UTC",
        name="timestamp",
    ).as_unit("ns")
    df = pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "signal", "atr"],
        index=idx,
    )
    df["volume"] = 100.0
    for col in ("open", "high", "low", "close", "atr", "volume"):
        df[col] = df[col].astype("float64")
    df["signal"] = df["signal"].astype("int64")
    return df


class FailingExchange:
    """Exchange giả lập luôn ném RuntimeError khi gọi fetch_ohlcv."""

    def milliseconds(self) -> int:
        return _T0_MS + 10 * _TF_15M_MS

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> list[list[float | int]]:
        raise RuntimeError("Network timeout")


class StaticExchange:
    """Exchange giả lập trả danh sách nến cố định và hỗ trợ create_order offline."""

    def __init__(self, raw_rows: list[list[float | int]], now_ms: int) -> None:
        self.raw_rows = raw_rows
        self.now_ms = now_ms
        self.orders: list[dict[str, Any]] = []

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> list[list[float | int]]:
        out = [r for r in self.raw_rows if since is None or int(r[0]) >= int(since)]
        if limit is not None:
            out = out[: int(limit)]
        return [list(r) for r in out]

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        order = {
            "id": f"ord_{len(self.orders) + 1}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": float(amount),
            "price": price,
            "params": dict(params or {}),
        }
        self.orders.append(order)
        return order


def test_atomic_persistence_restores_open_position_and_daily_pnl(tmp_path: Path) -> None:
    """(a) Kiểm tra ghi nguyên tử (.tmp -> os.replace) và khôi phục 100% vị thế + lỗ trong ngày."""
    state_file = tmp_path / "nested_state" / "state.json"
    cfg = dataclasses.replace(TradingConfig(), state_file_path=state_file)
    sent_messages: list[str] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> Any:
        sent_messages.append(str(json.get("text", "")))
        return type("Resp", (), {"status_code": 200, "ok": True})()

    notifier = TelegramNotifier(
        bot_token="test_token",
        chat_id="test_chat",
        env_file=None,
        http_post=fake_post,
    )

    executor1 = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        risk_manager=RiskManager(cfg),
        state_store=StateStore(state_file),
        notifier=notifier,
    )

    # Nến 1: Mở vị thế Long
    rows = [(50000.0, 50050.0, 49950.0, 50000.0, 1, 500.0)]
    assert executor1.process_closed_candles(_make_signal_df(rows)) is True
    assert executor1.open_position is not None
    assert executor1.open_position.side == 1

    # Nến 2: Giá sập chạm SL -> ghi nhận khoản lỗ thực hiện trong ngày (daily_realized_pnl < 0)
    rows.append((49800.0, 49900.0, 48800.0, 48900.0, 0, 500.0))
    assert executor1.process_closed_candles(_make_signal_df(rows)) is True
    assert executor1.open_position is None
    assert executor1.risk_manager.daily_realized_pnl < 0.0
    assert executor1.risk_manager.consecutive_losses == 1
    assert len(executor1.trade_history) == 1

    # Nến 3: Mở vị thế Short mới
    rows.append((48900.0, 49000.0, 48800.0, 48900.0, -1, 400.0))
    assert executor1.process_closed_candles(_make_signal_df(rows)) is True
    assert executor1.open_position is not None
    assert executor1.open_position.side == -1

    # Nến 4: Giữ vị thế Short qua 1 nến đóng (bars_held tăng lên 1)
    rows.append((48850.0, 48950.0, 48700.0, 48800.0, 0, 400.0))
    assert executor1.process_closed_candles(_make_signal_df(rows)) is True
    assert executor1.open_position is not None
    assert executor1.open_position.bars_held == 1

    expected_pos = dataclasses.replace(executor1.open_position)
    expected_daily_pnl = executor1.risk_manager.daily_realized_pnl
    expected_equity = executor1.risk_manager.current_equity
    expected_daily_start = executor1.risk_manager.daily_start_equity
    expected_losses = executor1.risk_manager.consecutive_losses
    expected_date = executor1.risk_manager.current_utc_date
    expected_last_ts = executor1.last_processed_candle_ts

    # Đảm bảo file chính tồn tại và không còn file .tmp dư thừa
    assert state_file.exists()
    assert not state_file.with_suffix(state_file.suffix + ".tmp").exists()
    assert list(state_file.parent.glob("*.tmp")) == []
    assert len(sent_messages) == 3  # 1 open Long + 1 close Long + 1 open Short

    # Hủy đối tượng cũ và tạo mới từ cùng file state
    del executor1

    executor2 = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        risk_manager=RiskManager(cfg),
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    restored_pos = executor2.open_position
    assert restored_pos is not None
    assert restored_pos.symbol == expected_pos.symbol
    assert restored_pos.side == expected_pos.side == -1
    assert restored_pos.entry_price == pytest.approx(expected_pos.entry_price)
    assert restored_pos.stop_loss == pytest.approx(expected_pos.stop_loss)
    assert restored_pos.take_profit == pytest.approx(expected_pos.take_profit)
    assert restored_pos.qty == pytest.approx(expected_pos.qty)
    assert restored_pos.entry_fee == pytest.approx(expected_pos.entry_fee)
    assert restored_pos.notional == pytest.approx(expected_pos.notional)
    assert restored_pos.initial_sl == pytest.approx(expected_pos.initial_sl)
    assert restored_pos.breakeven_activated is expected_pos.breakeven_activated
    assert restored_pos.bars_held == expected_pos.bars_held == 1
    assert restored_pos.entry_time == expected_pos.entry_time
    assert str(restored_pos.entry_time.tz) == "UTC"

    assert executor2.risk_manager.daily_realized_pnl == pytest.approx(expected_daily_pnl)
    assert executor2.risk_manager.daily_realized_pnl < 0.0
    assert executor2.risk_manager.current_equity == pytest.approx(expected_equity)
    assert executor2.risk_manager.daily_start_equity == pytest.approx(expected_daily_start)
    assert executor2.risk_manager.consecutive_losses == expected_losses == 1
    assert executor2.risk_manager.current_utc_date == expected_date
    assert executor2.last_processed_candle_ts == expected_last_ts
    assert len(executor2.trade_history) == 1
    assert executor2.trade_history[0].exit_reason == "SL"

    # Nến 5: Chạy tiếp trên executor2 để xác nhận vị thế khôi phục đóng lệnh TP bình thường
    rows.append((48800.0, 48850.0, 47500.0, 47600.0, 0, 400.0))
    assert executor2.process_closed_candles(_make_signal_df(rows)) is True
    assert executor2.open_position is None
    assert len(executor2.trade_history) == 2
    assert executor2.trade_history[-1].exit_reason == "TP"
    assert executor2.risk_manager.consecutive_losses == 0



def test_fault_tolerance_survives_network_and_notifier_exceptions(tmp_path: Path) -> None:
    """(b) Kiểm tra khả năng chịu lỗi khi exchange ném RuntimeError và notifier ném ConnectionError."""
    state_file = tmp_path / "fault_state.json"
    cfg = dataclasses.replace(TradingConfig(), state_file_path=state_file)

    call_count = 0

    def failing_http_post(url: str, json: dict[str, Any], timeout: float) -> Any:
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Telegram API unreachable")

    notifier = TelegramNotifier(
        bot_token="fake_bot_token",
        chat_id="fake_chat_id",
        timeout_sec=0.1,
        max_retries=2,
        env_file=None,
        http_post=failing_http_post,
    )

    # Notifier bắt gọn ConnectionError, thử đủ max_retries và trả về False
    assert notifier.send_message("Test message") is False
    assert call_count == 2

    # Khi thiếu cấu hình, notifier fallback sang logging và trả về False mà không gọi mạng
    fallback_notifier = TelegramNotifier(bot_token=None, chat_id=None, env_file=None, http_post=failing_http_post)
    assert fallback_notifier.send_message("Fallback message") is False
    assert call_count == 2

    loader = DataLoader(cfg, exchange=FailingExchange())
    executor = LiveExecutor(
        config=cfg,
        mode="paper",
        data_loader=loader,
        state_store=StateStore(state_file),
        notifier=notifier,
    )

    # step() gặp RuntimeError từ exchange + ConnectionError từ notifier nhưng không crash, trả False
    assert executor.step() is False
    assert executor.open_position is None
    assert call_count == 4  # Thêm 2 lần retry từ notify_error bên trong step()


def test_idempotency_processes_same_closed_candle_only_once(tmp_path: Path) -> None:
    """(c) Kiểm tra tính Idempotency: cùng nến đóng chỉ xử lý đúng 1 lần."""
    state_file = tmp_path / "idempotent_state.json"
    cfg = dataclasses.replace(TradingConfig(), state_file_path=state_file)

    raw_rows = [
        [_T0_MS + 0 * _TF_15M_MS, 50000.0, 50050.0, 49950.0, 50000.0, 100.0],
        [_T0_MS + 1 * _TF_15M_MS, 50000.0, 50100.0, 49900.0, 50050.0, 120.0],
        # Nến thứ 3 đang mở (chưa đóng tại now_ms = _T0_MS + 2 * _TF_15M_MS + 30_000)
        [_T0_MS + 2 * _TF_15M_MS, 50050.0, 50900.0, 49000.0, 50800.0, 500.0],
    ]
    exchange = StaticExchange(raw_rows=raw_rows, now_ms=_T0_MS + 2 * _TF_15M_MS + 30_000)
    loader = DataLoader(cfg, exchange=exchange)

    sig_df = _make_signal_df(
        [
            (50000.0, 50050.0, 49950.0, 50000.0, 0, 500.0),
            (50000.0, 50100.0, 49900.0, 50050.0, 1, 500.0),
        ]
    )

    class TwoBarStrategy:
        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["atr"] = 500.0
            out["signal"] = 0
            if len(out) >= 2:
                out.iloc[-1, out.columns.get_loc("signal")] = 1
            return out

    executor = LiveExecutor(
        config=cfg,
        mode="paper",
        data_loader=loader,
        strategy=TwoBarStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    # Lần 1 gọi process_closed_candles(sig_df): xử lý nến mới -> mở lệnh, trả True
    assert executor.process_closed_candles(sig_df) is True
    assert executor.open_position is not None
    assert executor.open_position.bars_held == 0
    first_entry_price = executor.open_position.entry_price
    assert len(executor.trade_history) == 0

    # Lần 2 gọi process_closed_candles(sig_df) với cùng timestamp cuối: bỏ qua -> trả False
    assert executor.process_closed_candles(sig_df) is False
    assert executor.open_position is not None
    assert executor.open_position.bars_held == 0
    assert executor.open_position.entry_price == pytest.approx(first_entry_price)
    assert len(executor.trade_history) == 0

    # Gọi step() khi sàn chưa có nến đóng mới (nến thứ 3 chưa đóng bị drop_incomplete_last_candle loại): trả False
    assert executor.step() is False
    assert executor.open_position.bars_held == 0


def test_live_mode_orders_breakeven_time_exit_and_cli(tmp_path: Path) -> None:
    """Kiểm tra thêm: nạp .env, breakeven + time exit trong LiveExecutor, mode='live' và CLI run_live.py."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# Comment line\nTELEGRAM_BOT_TOKEN='env_token_123'\nTELEGRAM_CHAT_ID=\"987654\"\n",
        encoding="utf-8",
    )
    notifier = TelegramNotifier(env_file=env_path)
    assert notifier.bot_token == "env_token_123"
    assert notifier.chat_id == "987654"

    state_file = tmp_path / "live_mode_state.json"
    cfg = dataclasses.replace(
        TradingConfig(),
        state_file_path=state_file,
        use_breakeven=True,
        breakeven_trigger_r=1.0,
        max_holding_bars=2,
    )
    exchange = StaticExchange(raw_rows=[], now_ms=_T0_MS)
    loader = DataLoader(cfg, exchange=exchange)

    executor = LiveExecutor(
        config=cfg,
        mode="live",
        data_loader=loader,
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    # Nến 1: Mở Long ở mode='live' -> gọi create_order('buy')
    rows = [(50000.0, 50050.0, 49950.0, 50000.0, 1, 500.0)]
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    assert len(exchange.orders) == 1
    assert exchange.orders[0]["side"] == "buy"
    assert executor.open_position is not None

    # Nến 2: Đạt >= +1R (high=51100 >= entry + 1000) -> kích hoạt breakeven, bars_held=1
    rows.append((50050.0, 51100.0, 50000.0, 50800.0, 0, 500.0))
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    assert executor.open_position is not None
    assert executor.open_position.breakeven_activated is True
    expected_be = compute_breakeven_price(executor.open_position.entry_price, 1, cfg)
    assert executor.open_position.stop_loss == pytest.approx(expected_be)
    assert executor.open_position.bars_held == 1

    # Nến 3: Đạt max_holding_bars=2 -> đóng TIME_EXIT và gửi lệnh sell reduceOnly=True
    rows.append((50800.0, 50900.0, 50700.0, 50850.0, 0, 500.0))
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    assert executor.open_position is None
    assert len(exchange.orders) == 2
    assert exchange.orders[1]["side"] == "sell"
    assert exchange.orders[1]["params"].get("reduceOnly") is True
    assert executor.trade_history[-1].exit_reason == "TIME_EXIT"

    # Kiểm tra CLI run_live.py --mode paper --once chạy offline trả về 0
    cli_state = tmp_path / "cli_state.json"
    exit_code = run_live.main(
        ["--mode", "paper", "--once", "--state-file", str(cli_state), "--synthetic"]
    )
    assert exit_code == 0
    assert cli_state.exists()


def test_live_executor_auto_trains_unfitted_ml_filter_on_closed_history(tmp_path: Path) -> None:
    """[HIGH-01] Khi config.use_ml=True và ml_filter chưa fit, LiveExecutor phải tự train trên cửa sổ lịch sử đã đóng."""
    state_file = tmp_path / "ml_auto_fit_state.json"
    cfg = dataclasses.replace(
        TradingConfig(),
        state_file_path=state_file,
        use_ml=True,
        ml_prob_threshold=0.55,
        ml_min_train_samples=30,
        max_holding_bars=48,
    )
    ml_filter = MetaLabelFilter(cfg, random_state=42)
    assert ml_filter.is_fitted_ is False

    n_bars = 200
    rows = [(50000.0, 50100.0, 49900.0, 50000.0, 0, 500.0) for _ in range(n_bars)]
    df = _make_signal_df(rows)
    for col in ML_FEATURE_COLUMNS:
        if col != "signal":
            df[col] = 0.0

    # Đặt 40 tín hiệu lịch sử đã đóng hoàn toàn (t = 2, 5, ..., 119; 119 + 48 < 199):
    # Chẵn: rsi=65, vol_zscore=3.0 -> chạm TP ở t+1 (high=51600) -> label=1
    # Lẻ:   rsi=35, vol_zscore=1.0 -> chạm SL ở t+1 (low=48800)  -> label=0
    for k in range(40):
        t = 2 + k * 3
        df.iloc[t, df.columns.get_loc("signal")] = 1
        if k % 2 == 0:
            df.iloc[t, df.columns.get_loc("rsi")] = 65.0
            df.iloc[t, df.columns.get_loc("vol_zscore")] = 3.0
            df.iloc[t + 1, df.columns.get_loc("high")] = 51600.0
        else:
            df.iloc[t, df.columns.get_loc("rsi")] = 35.0
            df.iloc[t, df.columns.get_loc("vol_zscore")] = 1.0
            df.iloc[t + 1, df.columns.get_loc("low")] = 48800.0

    # Nến cuối cùng (t=199) phát tín hiệu Long nhưng có đặc trưng xác suất thắng thấp (rsi=35, vol_zscore=1.0)
    df.iloc[-1, df.columns.get_loc("signal")] = 1
    df.iloc[-1, df.columns.get_loc("rsi")] = 35.0
    df.iloc[-1, df.columns.get_loc("vol_zscore")] = 1.0

    executor = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
        ml_filter=ml_filter,
    )

    assert executor.process_closed_candles(df) is True
    assert ml_filter.is_fitted_ is True
    # Vì ml_filter đã được fit, tín hiệu chất lượng thấp ở nến cuối phải bị từ chối
    assert executor.open_position is None


def test_live_mode_open_position_reanchors_sl_initial_sl_and_tp_to_fill_price(tmp_path: Path) -> None:
    """[HIGH-02] Khi sàn khớp giá thực tế fill_px lệch khỏi ref_price ở mode='live', phải tính lại initial_sl/stop_loss/take_profit theo fill_px."""

    class FillDeviationExchange(StaticExchange):
        def __init__(self, fill_price: float) -> None:
            super().__init__(raw_rows=[], now_ms=_T0_MS)
            self.fill_price = fill_price

        def create_order(
            self,
            symbol: str,
            type: str,
            side: str,
            amount: float,
            price: Optional[float] = None,
            params: Optional[dict[str, Any]] = None,
        ) -> dict[str, Any]:
            order = super().create_order(symbol, type, side, amount, price, params)
            order["average"] = self.fill_price
            return order

    state_file = tmp_path / "fill_reanchor_state.json"
    cfg = dataclasses.replace(TradingConfig(), state_file_path=state_file)
    fill_px = 50250.0  # lệch rõ so với ref_price=50000.0
    atr_val = 500.0

    exchange = FillDeviationExchange(fill_price=fill_px)
    loader = DataLoader(cfg, exchange=exchange)
    executor = LiveExecutor(
        config=cfg,
        mode="live",
        data_loader=loader,
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    rows = [(50000.0, 50050.0, 49950.0, 50000.0, 1, atr_val)]
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    pos = executor.open_position
    assert pos is not None
    assert pos.entry_price == pytest.approx(fill_px)
    assert pos.stop_loss == pytest.approx(fill_px - cfg.sl_atr_multiple * atr_val)
    assert pos.initial_sl == pytest.approx(fill_px - cfg.sl_atr_multiple * atr_val)
    assert pos.take_profit == pytest.approx(fill_px + cfg.tp_atr_multiple * atr_val)
    assert abs(pos.entry_price - pos.initial_sl) == pytest.approx(cfg.sl_atr_multiple * atr_val)


def test_live_executor_time_exit_takes_precedence_over_signal_reverse_on_same_bar(tmp_path: Path) -> None:
    """[MEDIUM-02] Khi nến đóng đồng thời đạt max_holding_bars và có tín hiệu ngược chiều, lý do thoát phải là TIME_EXIT nhất quán với Backtester."""
    state_file = tmp_path / "time_exit_vs_reverse.json"
    cfg = dataclasses.replace(
        TradingConfig(),
        state_file_path=state_file,
        max_holding_bars=2,
    )
    executor = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    # Nến 1: Mở vị thế Long (bars_held = 0)
    rows = [(50000.0, 50050.0, 49950.0, 50000.0, 1, 500.0)]
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    assert executor.open_position is not None
    assert executor.open_position.bars_held == 0

    # Nến 2: Giữ vị thế Long (bars_held = 1 < max_holding_bars=2)
    rows.append((50000.0, 50100.0, 49900.0, 50020.0, 0, 500.0))
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    assert executor.open_position is not None
    assert executor.open_position.bars_held == 1

    # Nến 3: Đồng thời đạt bars_held=2 == max_holding_bars VÀ xuất hiện tín hiệu ngược chiều (sig=-1)
    rows.append((50020.0, 50100.0, 49920.0, 50050.0, -1, 500.0))
    assert executor.process_closed_candles(_make_signal_df(rows)) is True
    assert len(executor.trade_history) == 1
    assert executor.trade_history[0].exit_reason == "TIME_EXIT"


def test_live_executor_utc_day_rollover_resets_daily_pnl_and_unlocks_circuit_breaker(tmp_path: Path) -> None:
    """[LOW-01(c)] Đổi ngày UTC trong LiveExecutor phải reset daily_realized_pnl về 0 và mở khóa circuit_breaker_tripped."""
    state_file = tmp_path / "utc_rollover_state.json"
    # Đặt max_daily_drawdown_pct=0.01 (4 USDT trên vốn 400 USDT) để 1 lệnh thua SL (~ -4.4 USDT) kích hoạt circuit breaker
    cfg = dataclasses.replace(
        TradingConfig(),
        state_file_path=state_file,
        max_daily_drawdown_pct=0.01,
    )
    executor = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    def _df_with_timestamps(
        items: list[tuple[str, float, float, float, float, int, float]]
    ) -> pd.DataFrame:
        idx = pd.DatetimeIndex([pd.Timestamp(x[0], tz="UTC") for x in items], name="timestamp").as_unit("ns")
        df = pd.DataFrame(
            [x[1:] for x in items],
            columns=["open", "high", "low", "close", "signal", "atr"],
            index=idx,
        )
        df["volume"] = 100.0
        for col in ("open", "high", "low", "close", "atr", "volume"):
            df[col] = df[col].astype("float64")
        df["signal"] = df["signal"].astype("int64")
        return df

    # Nến 1 (2025-01-01 23:15 UTC): Mở Long
    items = [("2025-01-01 23:15:00", 50000.0, 50050.0, 49950.0, 50000.0, 1, 500.0)]
    assert executor.process_closed_candles(_df_with_timestamps(items)) is True
    assert executor.open_position is not None

    # Nến 2 (2025-01-01 23:30 UTC): Chạm SL -> lỗ > 4 USDT -> kích hoạt circuit_breaker_tripped
    items.append(("2025-01-01 23:30:00", 49900.0, 49950.0, 48800.0, 48900.0, 0, 500.0))
    assert executor.process_closed_candles(_df_with_timestamps(items)) is True
    assert executor.open_position is None
    assert executor.risk_manager.daily_realized_pnl < -4.0
    assert executor.risk_manager.circuit_breaker_tripped is True

    # Nến 3 (2025-01-01 23:45 UTC - cùng ngày UTC): Có tín hiệu Long mới nhưng bị chặn bởi circuit breaker
    items.append(("2025-01-01 23:45:00", 48900.0, 49000.0, 48850.0, 48950.0, 1, 500.0))
    assert executor.process_closed_candles(_df_with_timestamps(items)) is True
    assert executor.open_position is None
    assert executor.risk_manager.circuit_breaker_tripped is True

    # Nến 4 (2025-01-02 00:00:00 UTC - sang ngày UTC mới): Reset daily_realized_pnl=0, mở khóa circuit breaker và vào lệnh
    items.append(("2025-01-02 00:00:00", 48950.0, 49050.0, 48900.0, 49000.0, 1, 500.0))
    assert executor.process_closed_candles(_df_with_timestamps(items)) is True
    assert executor.risk_manager.daily_realized_pnl == pytest.approx(0.0)
    assert executor.risk_manager.circuit_breaker_tripped is False
    assert executor.open_position is not None
    assert executor.open_position.side == 1


def test_live_executor_trailing_stop_and_cycle_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Kiểm tra LiveExecutor cập nhật Trailing Stop, khôi phục qua StateStore và ghi heartbeat logging."""
    import logging
    caplog.set_level(logging.INFO)

    state_file = tmp_path / "trailing_cycle_state.json"
    cfg = dataclasses.replace(
        TradingConfig(),
        state_file_path=state_file,
        use_trailing_stop=True,
        trailing_activation_r=1.5,
        trailing_atr_mult=2.0,
        tp_atr_multiple=5.0,
        use_breakeven=False,
    )
    executor = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )

    # Nến 1: Mở vị thế Long
    # entry_price = 50000 * (1 + 0.0002) = 50010.0, atr = 500.0, sl = 50010 - 2*500 = 49010.0
    # initial_risk = 1000.0 -> trigger dist = 1.5 * 1000 = 1500 -> trigger price = 51510.0
    rows = [(50000.0, 50100.0, 49900.0, 50000.0, 1, 500.0)]
    df1 = _make_signal_df(rows)
    assert executor.process_closed_candles(df1) is True
    assert executor.open_position is not None
    pos = executor.open_position
    assert pos.side == 1
    assert pos.trailing_activated is False
    assert pos.highest_price_since_entry == pytest.approx(pos.entry_price)

    # Nến 2: Giá tăng mạnh vượt 1.5R: high = 52000.0 (>= 51510.0), close = 51600.0
    # Trailing Stop kích hoạt: peak = 52000.0
    # new_sl = 52000.0 - 2.0 * 500.0 = 51000.0 > initial_sl (49010.0)
    rows.append((50050.0, 52000.0, 50000.0, 51600.0, 0, 500.0))
    df2 = _make_signal_df(rows)
    assert executor.process_closed_candles(df2) is True
    assert executor.open_position is not None
    pos2 = executor.open_position
    assert pos2.trailing_activated is True
    assert pos2.highest_price_since_entry == pytest.approx(52000.0)
    assert pos2.stop_loss == pytest.approx(51000.0)

    # Khôi phục trạng thái từ StateStore để đảm bảo trailing fields được persist
    del executor
    executor_restored = LiveExecutor(
        config=cfg,
        mode="paper",
        strategy=PassthroughStrategy(),  # type: ignore[arg-type]
        state_store=StateStore(state_file),
        notifier=TelegramNotifier(env_file=None),
    )
    restored_pos = executor_restored.open_position
    assert restored_pos is not None
    assert restored_pos.trailing_activated is True
    assert restored_pos.highest_price_since_entry == pytest.approx(52000.0)
    assert restored_pos.stop_loss == pytest.approx(51000.0)

    # Nến 2 gửi lại (trùng nến): phải log status=WAITING_NEXT_CANDLE và trả về False
    assert executor_restored.process_closed_candles(df2) is False

    # Kiểm tra Heartbeat logging
    log_records = [rec.message for rec in caplog.records if "[Heartbeat]" in rec.message]
    assert len(log_records) >= 3
    # Phải có bản ghi NEW_CANDLE_PROCESSED
    processed_logs = [m for m in log_records if "status=NEW_CANDLE_PROCESSED" in m]
    assert len(processed_logs) >= 2
    assert any("LONG qty=" in m for m in processed_logs)
    assert any("mode=paper" in m for m in processed_logs)
    # Phải có bản ghi WAITING_NEXT_CANDLE
    waiting_logs = [m for m in log_records if "status=WAITING_NEXT_CANDLE" in m]
    assert len(waiting_logs) >= 1


