"""Tests for Anti-Hang Hardening and System Resilience.

Verifies:
1. Global Socket & HTTP Timeout Guard
2. Maker Wait-Fill Monotonic Wall-Clock Ceiling
3. Windows File Lock Retry on Atomic StateStore Save
4. Tournament Heartbeat & Interruptible Sleep Slicing
5. Cycle-Level Watchdog Timeout with Forced os._exit(2)
6. Safe Windows QuickEdit Mode toggling
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any
import urllib.error

import pytest

from run_live import disable_windows_quick_edit as rl_disable_quick_edit
from scripts.run_parallel_live_demo import (
    DEFAULT_MAX_CYCLE_DURATION_SEC,
    HEARTBEAT_FILENAME,
    CycleWatchdog,
    disable_windows_quick_edit as demo_disable_quick_edit,
    interruptible_sleep,
    write_heartbeat,
)
from src.binance_public_feed import fetch_binance_shadow_snapshot
from src.binance_signed_client import (
    VENUE_FUTURES,
    VENUE_SPOT,
    BinanceHttpError,
    BinanceSignedTestnetClient,
)
from src.state_store import BotState, StateStore


# ============================================================================
# 1. Test Socket / HTTP Timeout Guard
# ============================================================================
def test_global_socket_default_timeout() -> None:
    """Verify that socket default timeout is set to a safe finite value (25.0s)."""
    current_timeout = socket.getdefaulttimeout()
    assert current_timeout is not None
    assert current_timeout <= 25.0


def test_binance_signed_client_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that client HTTP requests timeout instead of hanging indefinitely."""
    client = BinanceSignedTestnetClient(api_key="mock", api_secret="mock")

    def mock_urlopen(req: Any, timeout: float = 10.0) -> Any:
        # Simulate socket/network timeout
        raise urllib.error.URLError(socket.timeout("Operation timed out"))

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    with pytest.raises(BinanceHttpError) as exc_info:
        client._http_request("GET", "https://testnet.binance.vision/api/v3/time", timeout_sec=0.1)

    assert "Network error" in str(exc_info.value) or "timed out" in str(exc_info.value)


def test_binance_public_feed_timeout_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that public feed gracefully fails-closed when network times out."""
    def mock_urlopen(url: Any, timeout: float = 5.0) -> Any:
        raise urllib.error.URLError(socket.timeout("Socket connection timed out"))

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    snapshot = fetch_binance_shadow_snapshot(["BTC/USDT:USDT"], timeout_sec=0.1)
    assert snapshot["healthy"] is False
    assert snapshot["symbols"] == {}
    assert any("NETWORK_FATAL_ERROR" in err for err in snapshot["errors"])


# ============================================================================
# 2. Test Maker Wait-Fill Wall-Clock Ceiling
# ============================================================================
def test_maker_await_fill_monotonic_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _await_fill exits strictly within max_wait_sec when order stays NEW."""
    client = BinanceSignedTestnetClient(api_key="mock", api_secret="mock")

    poll_count = 0

    def mock_query_order(venue: str, symbol: str, order_id: Any, timeout_sec: float = 5.0) -> dict[str, Any]:
        nonlocal poll_count
        poll_count += 1
        return {"orderId": order_id, "status": "NEW", "executedQty": "0.0"}

    monkeypatch.setattr(client, "query_order", mock_query_order)

    events: list[dict[str, Any]] = []
    initial_res = {"orderId": 99999, "status": "NEW", "executedQty": "0.0"}

    start_mono = time.monotonic()
    final_res = client._await_fill(
        venue=VENUE_SPOT,
        symbol="BTCUSDT",
        res=initial_res,
        max_wait_sec=0.15,
        poll_interval_sec=0.03,
        events=events,
    )
    elapsed = time.monotonic() - start_mono

    # Wall-clock ceiling must have prevented hanging
    assert elapsed < 0.60
    assert final_res.get("status") == "NEW"
    assert poll_count >= 1


# ============================================================================
# 3. Test Windows File Lock Retry (StateStore)
# ============================================================================
def test_state_store_file_lock_retry_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify StateStore.save retries on transient PermissionError and succeeds."""
    state_file = tmp_path / "test_state.json"
    store = StateStore(state_file)

    call_count = 0
    real_replace = os.replace

    def mock_replace(src: Path | str, dst: Path | str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            # Simulate Windows transient file lock by antivirus/file indexer
            raise PermissionError(13, "WinError 32: The process cannot access the file because it is being used")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)

    saved_path = store.save(BotState(current_equity=1234.56))
    assert saved_path == state_file
    assert state_file.exists()
    assert call_count == 4  # Failed 3 times, succeeded on 4th

    loaded = store.load()
    assert loaded.current_equity == pytest.approx(1234.56)


def test_state_store_file_lock_retry_exhausted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify StateStore.save raises PermissionError if all 5 retries fail."""
    state_file = tmp_path / "test_state_fail.json"
    store = StateStore(state_file)

    call_count = 0

    def mock_always_fail_replace(src: Path | str, dst: Path | str) -> None:
        nonlocal call_count
        call_count += 1
        raise PermissionError(13, "WinError 5: Access denied permanently")

    monkeypatch.setattr(os, "replace", mock_always_fail_replace)

    with pytest.raises(PermissionError):
        store.save(BotState())

    assert call_count == 5


# ============================================================================
# 4. Test Heartbeat & Interruptible Sleep
# ============================================================================
def test_write_heartbeat_atomic(tmp_path: Path) -> None:
    """Verify write_heartbeat creates a valid atomic JSON heartbeat file."""
    demo_dir = str(tmp_path / "state")
    hb_path = write_heartbeat(
        demo_dir=demo_dir,
        cycle=103,
        phase="TEST_PHASE",
        extra={"custom_metric": 42},
    )

    assert os.path.exists(hb_path)
    with open(hb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["pid"] == os.getpid()
    assert data["cycle"] == 103
    assert data["phase"] == "TEST_PHASE"
    assert data["custom_metric"] == 42
    assert "timestamp_utc" in data
    assert abs(time.time() - data["epoch_sec"]) < 5.0


def test_interruptible_sleep_updates_heartbeat(tmp_path: Path) -> None:
    """Verify interruptible_sleep slices sleep duration and updates heartbeat per step."""
    demo_dir = str(tmp_path / "state")
    hb_file = os.path.join(demo_dir, HEARTBEAT_FILENAME)

    start_mono = time.monotonic()
    interruptible_sleep(
        total_seconds=0.25,
        cycle=103,
        demo_dir=demo_dir,
        chunk_seconds=0.08,
    )
    elapsed = time.monotonic() - start_mono

    assert 0.20 <= elapsed < 0.60
    assert os.path.exists(hb_file)
    with open(hb_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["cycle"] == 103
    assert data["phase"] == "SLEEPING"
    assert "sleep_remaining_sec" in data
    assert data["sleep_total_sec"] == pytest.approx(0.25, rel=1e-2)


# ============================================================================
# 5. Test Cycle Watchdog Timeout
# ============================================================================
def test_cycle_watchdog_timeout_triggers_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CycleWatchdog triggers os._exit(2) when cycle duration exceeds ceiling."""
    exit_calls: list[int] = []

    def mock_os_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(os, "_exit", mock_os_exit)

    # Short ceiling of 0.15s for the test
    watchdog = CycleWatchdog(max_duration_sec=0.15)
    try:
        watchdog.start_cycle(cycle=105)
        # Wait for watchdog to detect timeout
        time.sleep(0.30)
        assert 2 in exit_calls
    finally:
        watchdog.shutdown()


def test_cycle_watchdog_normal_cycle_does_not_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CycleWatchdog does not trigger when cycle finishes within ceiling."""
    exit_calls: list[int] = []

    def mock_os_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(os, "_exit", mock_os_exit)

    watchdog = CycleWatchdog(max_duration_sec=0.30)
    try:
        watchdog.start_cycle(cycle=105)
        time.sleep(0.05)
        watchdog.stop_cycle()  # Normal cycle completion
        time.sleep(0.30)
        assert len(exit_calls) == 0  # No abort triggered
    finally:
        watchdog.shutdown()


# ============================================================================
# 6. Test Windows QuickEdit Mode Functions
# ============================================================================
def test_disable_windows_quick_edit_safe_invocation() -> None:
    """Verify disable_windows_quick_edit runs safely without raising exceptions."""
    res1 = rl_disable_quick_edit()
    res2 = demo_disable_quick_edit()
    assert isinstance(res1, bool)
    assert isinstance(res2, bool)
