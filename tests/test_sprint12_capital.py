"""Unit test offline cho Sprint 12 - Chặng 1: capital parity, reset state và daemon cycles.

Phạm vi khóa hành vi (đối chiếu ``docs/go_no_go_sprint12.md``):

- PR-1a: State mới (chưa có file) khởi tạo đúng ``initial_capital`` / ``current_equity``
  / ``daily_start_equity`` / ``peak_equity`` theo vốn CLI.
- PR-1b: State cũ còn equity ``400.0`` và **không** có vị thế mở được đồng bộ lên vốn CLI.
- PR-1c: State có vị thế mở **không** bị ghi đè equity, chỉ cảnh báo.
- PR-1d: ``reset=True`` xóa sạch vị thế/lệnh và đặt lại vốn, tắt kill-switch.
- Đổi ``strategy`` phải trả state mới tinh, không kế thừa vị thế của chiến lược cũ.
- Round-trip ``BotState`` bảo toàn ``initial_capital`` / ``peak_equity`` /
  ``kill_switch_triggered`` khi load lại không truyền ``initial_capital``.
- PR-1 + PR-6: ``CarryLiveExecutor`` nhận capital, phản chiếu qua ``current_equity``
  và ghi đúng xuống file state trên đĩa.
- PR-6 + PR-7: CLI ``run_live.parse_args`` hỗ trợ ``--reset-state``, ``--cycles``,
  ``--interval-sec``, ``--capital``, ``--mode testnet`` và từ chối mode lạ.
- PR-6: ``run_loop(cycles=N)`` gọi ``reconcile_once`` đúng N lần, trả mã int 0 và
  không ngủ thật (``interval_sec=0.0`` + ``time.sleep`` được monkeypatch).

Toàn bộ test tất định, không gọi mạng, không đọc ``data/raw`` thật và chỉ ghi file
trong ``tmp_path``. Mỗi test chạy dưới 3 giây.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

import run_live
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import BotState, StateStore

CAPITAL = 3000.0
LEGACY_EQUITY = 400.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_state(path: Path, payload: dict[str, Any]) -> Path:
    """Ghi một file state JSON kiểu cũ/giả lập vào ``path`` (không qua StateStore)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _legacy_payload(**overrides: Any) -> dict[str, Any]:
    """Payload state tối thiểu kiểu cũ: equity 400, thiếu ``initial_capital``."""
    payload: dict[str, Any] = {
        "open_position": None,
        "daily_realized_pnl": 0.0,
        "daily_start_equity": LEGACY_EQUITY,
        "current_equity": LEGACY_EQUITY,
        "current_utc_date": "2025-01-01",
        "consecutive_losses": 0,
        "circuit_breaker_tripped": False,
        "last_processed_candle_ts": None,
        "trade_history": [],
        "carry_positions": {},
        "overlay_positions": {},
    }
    payload.update(overrides)
    return payload


def _open_carry_payload(equity: float = LEGACY_EQUITY) -> dict[str, Any]:
    """Payload state cũ có một vị thế carry đang mở (phải không bị ghi đè equity)."""
    return _legacy_payload(
        current_equity=equity,
        carry_positions={
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT",
                "qty": 0.1,
                "capital": 500.0,
                "entry_spot_price": 100.0,
                "entry_perp_price": 100.05,
            }
        },
    )


def _make_executor(
    tmp_path: Path,
    capital: float | None = None,
    initial_capital: float | None = None,
    strategy: str | None = None,
    reset_state: bool = False,
) -> CarryLiveExecutor:
    """Khởi tạo ``CarryLiveExecutor`` trỏ state file và raw dir vào ``tmp_path``.

    Bản sao tối giản của helper trong ``tests/test_carry_live_executor.py`` nhưng
    bổ sung các tham số capital mới của Sprint 12 (``capital``, ``initial_capital``,
    ``strategy``, ``reset_state``).
    """
    state_path = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=raw_dir,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(),
        symbols=["AAA/USDT:USDT"],
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
        capital=capital,
        initial_capital=initial_capital,
        strategy=strategy,
        reset_state=reset_state,
    )


def _read_state_file(path: Path) -> dict[str, Any]:
    """Đọc lại JSON state đã ghi xuống đĩa."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PR-1a: State hoàn toàn mới
# ---------------------------------------------------------------------------


def test_fresh_state_uses_initial_capital(tmp_path: Path) -> None:
    """File chưa tồn tại -> state mới có equity/capital/peak đều bằng vốn CLI."""
    state = StateStore(tmp_path / "s.json").load(initial_capital=CAPITAL)

    assert state.current_equity == pytest.approx(CAPITAL)
    assert state.initial_capital == pytest.approx(CAPITAL)
    assert state.daily_start_equity == pytest.approx(CAPITAL)
    assert state.peak_equity >= CAPITAL


# ---------------------------------------------------------------------------
# PR-1b / PR-1c / PR-1d: Đồng bộ, không ghi đè, reset
# ---------------------------------------------------------------------------


def test_legacy_equity_synced_when_no_open_positions(tmp_path: Path) -> None:
    """State cũ 400.0 không có vị thế mở phải tự đồng bộ lên 3000.0 và peak 3000.0."""
    state_path = _write_state(tmp_path / "s.json", _legacy_payload())

    state = StateStore(state_path).load(initial_capital=CAPITAL)

    assert state.current_equity == pytest.approx(CAPITAL)
    assert state.initial_capital == pytest.approx(CAPITAL)
    assert state.peak_equity == pytest.approx(CAPITAL)
    assert state.daily_start_equity == pytest.approx(CAPITAL)


def test_open_positions_are_not_overwritten(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Có vị thế mở -> giữ equity 400.0, chỉ log cảnh báo và không ném lỗi."""
    state_path = _write_state(tmp_path / "s.json", _open_carry_payload())

    with caplog.at_level("WARNING", logger="src.state_store"):
        state = StateStore(state_path).load(initial_capital=CAPITAL)

    assert state.current_equity == pytest.approx(LEGACY_EQUITY)
    assert "BTC/USDT:USDT" in state.carry_positions
    assert any("open positions" in record.message for record in caplog.records)


def test_reset_clears_positions_and_restores_capital(tmp_path: Path) -> None:
    """``reset=True`` -> xóa toàn bộ vị thế, đặt lại vốn và tắt kill-switch."""
    payload = _open_carry_payload(equity=500.0)
    payload["overlay_positions"] = {"ETH/USDT:USDT": {"qty": 0.2, "side": "long"}}
    payload["kill_switch_triggered"] = True
    state_path = _write_state(tmp_path / "s.json", payload)

    state = StateStore(state_path).load(initial_capital=CAPITAL, reset=True)

    assert state.current_equity == pytest.approx(CAPITAL)
    assert state.initial_capital == pytest.approx(CAPITAL)
    assert state.carry_positions == {}
    assert state.overlay_positions == {}
    assert state.kill_switch_triggered is False


# ---------------------------------------------------------------------------
# Strategy mismatch + round-trip
# ---------------------------------------------------------------------------


def test_strategy_mismatch_resets_state(tmp_path: Path) -> None:
    """Đổi strategy (dynamic_carry -> barbell) phải trả state mới, không giữ vị thế cũ."""
    payload = _open_carry_payload(equity=520.0)
    payload["strategy"] = "dynamic_carry"
    state_path = _write_state(tmp_path / "s.json", payload)

    state = StateStore(state_path).load(strategy="barbell", initial_capital=CAPITAL)

    assert state.current_equity == pytest.approx(CAPITAL)
    assert state.initial_capital == pytest.approx(CAPITAL)
    assert state.strategy == "barbell"
    assert state.carry_positions == {}
    assert state.overlay_positions == {}


def test_state_round_trip_preserves_new_fields(tmp_path: Path) -> None:
    """Save rồi load lại (không truyền capital) giữ nguyên capital/peak/kill-switch."""
    state_path = tmp_path / "s.json"
    store = StateStore(state_path)
    original = BotState(
        current_equity=CAPITAL,
        daily_start_equity=CAPITAL,
        initial_capital=CAPITAL,
        peak_equity=3200.0,
        strategy="barbell",
        kill_switch_triggered=True,
    )

    store.save(original)
    restored = StateStore(state_path).load()

    assert restored.initial_capital == pytest.approx(CAPITAL)
    assert restored.peak_equity == pytest.approx(3200.0)
    assert restored.kill_switch_triggered is True
    assert restored.strategy == "barbell"


# ---------------------------------------------------------------------------
# Executor nhận capital
# ---------------------------------------------------------------------------


def test_executor_initial_capital_reflected_in_state_and_disk(tmp_path: Path) -> None:
    """Executor nhận capital 3000 -> current_equity và file state trên đĩa đều 3000."""
    executor = _make_executor(
        tmp_path, capital=CAPITAL, initial_capital=CAPITAL
    )

    assert executor.current_equity == pytest.approx(CAPITAL)
    assert executor.state.current_equity == pytest.approx(CAPITAL)
    assert executor.state.initial_capital == pytest.approx(CAPITAL)

    on_disk = _read_state_file(tmp_path / "state.json")
    assert float(on_disk["current_equity"]) == pytest.approx(CAPITAL)


# ---------------------------------------------------------------------------
# CLI flags (PR-6, PR-7)
# ---------------------------------------------------------------------------


def test_cli_accepts_sprint12_flags() -> None:
    """``--reset-state`` / ``--cycles`` / ``--interval-sec`` / ``--capital`` parse đúng."""
    args = run_live.parse_args(
        [
            "--strategy",
            "barbell",
            "--capital",
            "3000",
            "--reset-state",
            "--cycles",
            "2",
            "--interval-sec",
            "1",
        ]
    )

    assert args.reset_state is True
    assert args.cycles == 2
    assert args.interval_sec == pytest.approx(1.0)
    assert args.capital == pytest.approx(CAPITAL)
    assert args.strategy == "barbell"


def test_cli_accepts_testnet_mode_and_rejects_unknown_mode() -> None:
    """``--mode testnet`` hợp lệ; mode lạ phải làm argparse ném ``SystemExit``."""
    args = run_live.parse_args(["--mode", "testnet"])
    assert args.mode == "testnet"

    with pytest.raises(SystemExit):
        run_live.parse_args(["--mode", "not_a_mode"])


# ---------------------------------------------------------------------------
# run_loop nhiều chu kỳ (PR-6)
# ---------------------------------------------------------------------------


def test_run_loop_calls_reconcile_exact_cycles_without_real_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cycles=3`` gọi reconcile đúng 3 lần, trả 0 và chỉ sleep với duration 0.0."""
    import src.carry_live_executor as cle

    executor = _make_executor(tmp_path, capital=CAPITAL, initial_capital=CAPITAL)
    calls: list[int] = []

    def fake_reconcile(aligned_universe: Any = None) -> dict[str, Any]:
        """Thay ``reconcile_once``: đếm lần gọi và trả plan rỗng."""
        calls.append(len(calls) + 1)
        return {
            "timestamp": "N/A",
            "equity": executor.current_equity,
            "actions": [],
            "active_positions": {},
            "margin_health_summary": "fake",
            "table_ascii": "",
            "sleeves": {},
        }

    sleep_durations: list[float] = []

    def fake_sleep(seconds: float) -> None:
        """Ghi nhận thời lượng sleep nhưng không ngủ thật."""
        sleep_durations.append(float(seconds))

    monkeypatch.setattr(executor, "reconcile_once", fake_reconcile)
    monkeypatch.setattr(cle.time, "sleep", fake_sleep)

    result = executor.run_loop(once=False, cycles=3, interval_sec=0.0)

    assert isinstance(result, int)
    assert result == 0
    assert len(calls) == 3
    # run_loop kiểm tra điều kiện dừng trước khi ngủ -> 2 lần sleep giữa các chu kỳ.
    assert len(sleep_durations) == 2
    assert all(duration == 0.0 for duration in sleep_durations)
