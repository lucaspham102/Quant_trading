"""Tests for Anti-Hang Hardening and System Resilience.

Verifies:
1. Global Socket & HTTP Timeout Guard
2. Maker Wait-Fill Monotonic Wall-Clock Ceiling
3. Windows File Lock Retry on Atomic StateStore Save
4. Tournament Heartbeat & Interruptible Sleep Slicing
5. Cycle-Level Watchdog Timeout with Forced os._exit(2)
6. Safe Windows QuickEdit & Prevent-Sleep Mode toggling
7. Backup (.bak) Auto-Recovery on Corrupted / 0-Byte State & Ledger JSON
8. Network Outage Resilience (wait_for_network_connectivity with WAITING_FOR_NETWORK heartbeat)
9. Rich Terminal Scoreboard Table (Unicode Box, ANSI Colors, Trend CB N/A (No-Trend), Summary Footer)
10. Smart Telegram HTML Report Formatting, Fallback & 5 Milestone Triggers
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any
import urllib.error

import pytest

from run_live import (
    disable_windows_quick_edit as rl_disable_quick_edit,
    prevent_windows_sleep as rl_prevent_sleep,
)
from scripts.run_parallel_live_demo import (
    DEFAULT_MAX_CYCLE_DURATION_SEC,
    HEARTBEAT_FILENAME,
    CycleWatchdog,
    disable_windows_quick_edit as demo_disable_quick_edit,
    format_scoreboard_table,
    interruptible_sleep,
    load_ledger,
    prevent_windows_sleep as demo_prevent_sleep,
    print_scoreboard,
    save_ledger_atomic,
    wait_for_network_connectivity,
    write_heartbeat,
)
from src.binance_public_feed import fetch_binance_shadow_snapshot
from src.binance_signed_client import (
    VENUE_FUTURES,
    VENUE_SPOT,
    BinanceHttpError,
    BinanceSignedTestnetClient,
)
from src.notifier import (
    TelegramNotifier,
    format_tournament_telegram_report,
    positions_changed,
    should_send_tournament_telegram,
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
    """Verify write_heartbeat creates a valid atomic JSON heartbeat file and .bak backup."""
    demo_dir = str(tmp_path / "state")
    hb_path = write_heartbeat(
        demo_dir=demo_dir,
        cycle=103,
        phase="TEST_PHASE",
        extra={"custom_metric": 42},
    )

    assert os.path.exists(hb_path)
    assert os.path.exists(f"{hb_path}.bak")
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
# 6. Test Windows QuickEdit & Prevent-Sleep Mode Functions
# ============================================================================
def test_disable_windows_quick_edit_and_prevent_sleep_safe_invocation() -> None:
    """Verify disable_windows_quick_edit and prevent_windows_sleep run safely."""
    res1 = rl_disable_quick_edit()
    res2 = demo_disable_quick_edit()
    sleep1 = rl_prevent_sleep()
    sleep2 = demo_prevent_sleep()
    assert isinstance(res1, bool)
    assert isinstance(res2, bool)
    assert isinstance(sleep1, bool)
    assert isinstance(sleep2, bool)


# ============================================================================
# 7. Test .bak Auto-Recovery on Corrupted / 0-Byte State & Ledger Files
# ============================================================================
def test_state_store_bak_recovery_on_corrupted_and_zero_byte_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify StateStore automatically recovers from .bak on corrupted or 0-byte JSON."""
    state_file = tmp_path / "bot_state.json"
    bak_file = tmp_path / "bot_state.json.bak"
    store = StateStore(state_file)

    # Save v1 (1100.0) and v2 (1250.5) so .bak holds v1 and .json holds v2
    store.save(BotState(current_equity=1100.0, consecutive_losses=2))
    assert bak_file.exists()
    store.save(BotState(current_equity=1250.5, consecutive_losses=3))
    assert bak_file.exists()

    # 1) Corrupt the primary .json file (simulating power loss mid-write)
    state_file.write_text("{corrupted_json_payload...", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        recovered = store.load()
    assert recovered.current_equity == pytest.approx(1100.0)
    assert recovered.consecutive_losses == 2
    assert "WARNING: RECOVERED_STATE_FROM_BACKUP" in caplog.text

    # 2) Truncate primary .json to 0 bytes
    caplog.clear()
    state_file.write_text("", encoding="utf-8")
    assert state_file.stat().st_size == 0
    with caplog.at_level(logging.WARNING):
        recovered_zero = store.load()
    assert recovered_zero.current_equity == pytest.approx(1100.0)
    assert "WARNING: RECOVERED_STATE_FROM_BACKUP" in caplog.text


def test_ledger_bak_recovery_on_corrupted_and_zero_byte_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify save_ledger_atomic creates .bak and load_ledger recovers from .bak on corruption."""
    ledger_path = str(tmp_path / "demo_regime_omni_1k.json")
    bak_path = f"{ledger_path}.bak"

    ledger_v1 = load_ledger(ledger_path, "regime_omni_1k")
    ledger_v1["cycle"] = 104
    ledger_v1["pure_trading_equity_usd"] = 1005.25
    save_ledger_atomic(ledger_path, ledger_v1)
    assert os.path.exists(bak_path)

    # Corrupt primary file
    Path(ledger_path).write_text("<<<NOT_JSON>>>", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        recovered = load_ledger(ledger_path, "regime_omni_1k")
    assert recovered["cycle"] == 104
    assert recovered["pure_trading_equity_usd"] == pytest.approx(1005.25)
    assert "WARNING: RECOVERED_STATE_FROM_BACKUP" in caplog.text

    # 0-byte primary file
    caplog.clear()
    Path(ledger_path).write_text("", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        recovered_empty = load_ledger(ledger_path, "regime_omni_1k")
    assert recovered_empty["cycle"] == 104
    assert recovered_empty["pure_trading_equity_usd"] == pytest.approx(1005.25)
    assert "WARNING: RECOVERED_STATE_FROM_BACKUP" in caplog.text


# ============================================================================
# 8. Test Network Outage Resilience (wait_for_network_connectivity)
# ============================================================================
def test_wait_for_network_connectivity_recovers_and_updates_heartbeat(tmp_path: Path) -> None:
    """Verify wait_for_network_connectivity retries, writes WAITING_FOR_NETWORK heartbeat, and recovers."""
    demo_dir = str(tmp_path / "state")
    hb_file = os.path.join(demo_dir, HEARTBEAT_FILENAME)
    attempts = 0
    seen_waiting_hb = False

    def flaky_probe() -> bool:
        nonlocal attempts, seen_waiting_hb
        attempts += 1
        if attempts >= 2 and os.path.exists(hb_file):
            with open(hb_file, "r", encoding="utf-8") as f:
                hb_data = json.load(f)
            if hb_data.get("phase") == "WAITING_FOR_NETWORK":
                seen_waiting_hb = True
        return attempts >= 3

    status_out: dict[str, Any] = {}
    ok = wait_for_network_connectivity(
        max_wait_sec=2.0,
        check_interval_sec=0.05,
        demo_dir=demo_dir,
        cycle=105,
        probe_fn=flaky_probe,
        status_out=status_out,
    )

    assert ok is True
    assert attempts == 3
    assert seen_waiting_hb is True
    assert status_out["connected"] is True
    assert status_out["recovered"] is True
    assert status_out["attempts"] == 3


# ============================================================================
# 9. Test Rich Terminal Scoreboard Table & Telegram HTML Formatter
# ============================================================================
def _sample_tournament_records() -> list[dict[str, Any]]:
    return [
        {
            "profile": "regime_omni_1k",
            "mode": "TESTNET_REAL",
            "cycle": 105,
            "timestamp": "2026-10-31T12:19:29+00:00",
            "pure_trading_equity_usd": 999.78,
            "yield_stacked_equity_usd": 1000.39,
            "net_pnl_usd": -0.22,
            "funding_pnl_usd": 0.07,
            "overlay_pnl_usd": -1.62,
            "trend_pnl_usd": 0.0,
            "basis_pnl_usd": 3.79,
            "fees_paid_usd": 1.26,
            "max_dd_pct": 0.28,
            "carry_slots": 3,
            "overlay_long_usd": 141.0,
            "overlay_short_usd": 144.0,
            "futures_leverage": 2.21,
            "circuit_breaker_state": "NORMAL",
            "exchange_recon_equity_usd": 994.31,
            "recon_drift_usd": -5.4646,
            "reconcile_status": "SYNCED",
            "status": "OK",
        },
        {
            "profile": "carry_max_eta_1k",
            "mode": "SHADOW_LIVE",
            "cycle": 105,
            "pure_trading_equity_usd": 1001.14,
            "yield_stacked_equity_usd": 1001.85,
            "net_pnl_usd": 1.14,
            "funding_pnl_usd": 0.12,
            "overlay_pnl_usd": 0.0,
            "trend_pnl_usd": 0.0,
            "basis_pnl_usd": 2.15,
            "fees_paid_usd": 1.13,
            "max_dd_pct": 0.05,
            "carry_slots": 4,
            "overlay_long_usd": 0.0,
            "overlay_short_usd": 0.0,
            "futures_leverage": 1.80,
            "circuit_breaker_state": "DISABLED",
            "exchange_recon_equity_usd": None,
            "recon_drift_usd": None,
            "reconcile_status": "NOT_APPLICABLE",
            "status": "OK",
        },
        {
            "profile": "flexible_regime_1k",
            "mode": "SHADOW_LIVE",
            "cycle": 105,
            "pure_trading_equity_usd": 997.70,
            "yield_stacked_equity_usd": 999.44,
            "net_pnl_usd": -2.30,
            "funding_pnl_usd": 0.05,
            "overlay_pnl_usd": -1.50,
            "trend_pnl_usd": 0.0,
            "basis_pnl_usd": 0.40,
            "fees_paid_usd": 1.25,
            "max_dd_pct": 0.34,
            "carry_slots": 2,
            "overlay_long_usd": 140.0,
            "overlay_short_usd": 140.0,
            "futures_leverage": 1.95,
            "circuit_breaker_state": "NORMAL",
            "exchange_recon_equity_usd": None,
            "recon_drift_usd": None,
            "reconcile_status": "NOT_APPLICABLE",
            "status": "OK",
        },
        {
            "profile": "baseline_1k",
            "mode": "SHADOW_LIVE",
            "cycle": 105,
            "pure_trading_equity_usd": 998.59,
            "yield_stacked_equity_usd": 1000.33,
            "net_pnl_usd": -1.41,
            "funding_pnl_usd": 0.05,
            "overlay_pnl_usd": -1.10,
            "trend_pnl_usd": 0.0,
            "basis_pnl_usd": 0.90,
            "fees_paid_usd": 1.26,
            "max_dd_pct": 0.21,
            "carry_slots": 2,
            "overlay_long_usd": 140.0,
            "overlay_short_usd": 140.0,
            "futures_leverage": 1.90,
            "circuit_breaker_state": "DISABLED",
            "exchange_recon_equity_usd": None,
            "recon_drift_usd": None,
            "reconcile_status": "NOT_APPLICABLE",
            "status": "OK",
        },
    ]


def test_format_scoreboard_table_unicode_ansi_and_ascii_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify Rich Terminal scoreboard renders Unicode box, ANSI colors, Trend CB N/A (No-Trend), and summary."""
    records = _sample_tournament_records()
    live_ledger = {
        "positions_snapshot": {
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT",
                "kind": "carry",
                "qty": 0.0027,
                "perp_qty": -0.0027,
                "signed_usd": 211.4,
            },
            "BNB/USDT:USDT|overlay": {
                "symbol": "BNB/USDT:USDT",
                "kind": "overlay",
                "qty": 0.2,
                "signed_usd": 141.1,
            },
        }
    }

    rich_out = format_scoreboard_table(
        records,
        use_unicode=True,
        use_color=True,
        ledgers={"regime_omni_1k": live_ledger},
        live_profile="regime_omni_1k",
        network_status="ONLINE",
        tg_status="TG: SENT",
        next_cycle_sec=3600.0,
    )
    assert "╔═" in rich_out
    assert "╚═" in rich_out
    assert "║" in rich_out
    assert "\033[" in rich_out  # ANSI color codes
    assert "Trend CB" in rich_out
    assert "N/A (No-Trend)" in rich_out
    assert "[POSITIONS regime_omni_1k]" in rich_out
    assert "BTC($211)" in rich_out
    assert "TG: SENT" in rich_out
    assert "3600s (60m 00s)" in rich_out

    # Test ASCII fallback
    ascii_out = format_scoreboard_table(
        records,
        use_unicode=False,
        use_color=False,
        ledgers={"regime_omni_1k": live_ledger},
        live_profile="regime_omni_1k",
        network_status="ONLINE",
        tg_status="TG: SKIPPED_NO_CHANGE",
        next_cycle_sec=1800.0,
    )
    assert all(ord(ch) < 128 for ch in ascii_out)
    assert "+-" in ascii_out
    assert "N/A (No-Trend)" in ascii_out
    assert "TG: SKIPPED_NO_CHANGE" in ascii_out

    # Ensure print_scoreboard executes without error
    print_scoreboard(records, use_unicode=True, use_color=True, ledgers={"regime_omni_1k": live_ledger})
    captured = capsys.readouterr().out
    assert "regime_omni_1k" in captured


# ============================================================================
# 10. Test Smart Telegram HTML Report & Milestone Triggers
# ============================================================================
def test_format_tournament_telegram_report_html_and_fallback() -> None:
    """Verify Telegram HTML formatter includes all required metrics and falls back to plain text on 400."""
    records = _sample_tournament_records()
    live_ledger = {
        "reconcile_status": "SYNCED",
        "recon_drift_usd": -5.4646,
        "positions_snapshot": {
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT",
                "kind": "carry",
                "qty": 0.002726,
                "perp_qty": -0.002726,
                "signed_usd": 211.42,
            },
            "AVAX/USDT:USDT|overlay": {
                "symbol": "AVAX/USDT:USDT",
                "kind": "overlay",
                "qty": -17.0,
                "signed_usd": -143.64,
            },
        },
    }

    html_msg = format_tournament_telegram_report(
        records,
        live_profile="regime_omni_1k",
        live_ledger=live_ledger,
        reasons=["PERIODIC_4C"],
    )

    assert "<b>" in html_msg and "<pre>" in html_msg and "<code>" in html_msg
    assert "Cycle #105" in html_msg
    assert "🟢 SYNCED" in html_msg
    assert "-5.4646" in html_msg
    for pname in ("regime_omni_1k", "carry_max_eta_1k", "flexible_regime_1k", "baseline_1k"):
        assert pname in html_msg
    assert "Funding:" in html_msg
    assert "Basis:" in html_msg
    assert "Overlay:" in html_msg
    assert "Trend:" in html_msg and "(FLAT)" in html_msg
    assert "Fees:" in html_msg
    assert "Fut Leverage:" in html_msg
    assert "BTC" in html_msg and "AVAX" in html_msg

    # Verify TelegramNotifier sends with parse_mode='HTML' and falls back to plain text if HTML parse fails
    payloads: list[dict[str, Any]] = []

    def mock_http_post_html_fail_then_plain_ok(url: str, json: dict[str, Any], timeout: float) -> Any:
        payloads.append(dict(json))
        if json.get("parse_mode") == "HTML":
            return type("Resp", (), {"status_code": 400, "ok": False})()
        return type("Resp", (), {"status_code": 200, "ok": True})()

    notifier = TelegramNotifier(
        bot_token="mock_token",
        chat_id="mock_chat",
        max_retries=1,
        env_file=None,
        http_post=mock_http_post_html_fail_then_plain_ok,
    )
    assert notifier.send_message(html_msg, parse_mode="HTML") is True
    assert len(payloads) == 2
    assert payloads[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in payloads[1]
    assert "<b>" not in payloads[1]["text"]
    assert "Cycle #105" in payloads[1]["text"]


def test_should_send_tournament_telegram_five_milestones() -> None:
    """Verify all 5 smart milestone triggers and skip behavior on unchanged intermediate cycles."""
    records = _sample_tournament_records()
    t_mid = datetime(2026, 10, 31, 14, 20, tzinfo=timezone.utc)

    # Normal intermediate cycle (e.g. Cycle 105, report every 4 -> 105 % 4 != 0, no changes) -> SKIP
    send, reasons = should_send_tournament_telegram(
        cycle=105,
        is_startup=False,
        trade_or_position_changed=False,
        network_recovered=False,
        records=records,
        tg_report_every=4,
        now_utc=t_mid,
        prev_utc_date="2026-10-31",
    )
    assert send is False
    assert reasons == []

    # 1. Startup / Resume -> SEND
    send1, r1 = should_send_tournament_telegram(
        cycle=105, is_startup=True, records=records, tg_report_every=4, now_utc=t_mid, prev_utc_date="2026-10-31"
    )
    assert send1 is True and "STARTUP_RESUME" in r1

    # 2. Trade Executed / Position Changed -> SEND
    prev_pos = {"BTC/USDT:USDT": {"qty": 0.002}}
    curr_pos = {"BTC/USDT:USDT": {"qty": 0.003}}
    assert positions_changed(prev_pos, curr_pos) is True
    send2, r2 = should_send_tournament_telegram(
        cycle=105,
        trade_or_position_changed=True,
        records=records,
        tg_report_every=4,
        now_utc=t_mid,
        prev_utc_date="2026-10-31",
    )
    assert send2 is True and "TRADE_OR_POSITION_CHANGED" in r2

    # 3. Network Recovered -> SEND
    send3, r3 = should_send_tournament_telegram(
        cycle=105,
        network_recovered=True,
        records=records,
        tg_report_every=4,
        now_utc=t_mid,
        prev_utc_date="2026-10-31",
    )
    assert send3 is True and "NETWORK_RECOVERED" in r3

    # 4. Risk / Reconcile Alert -> SEND
    alert_records = [dict(r) for r in records]
    alert_records[0]["reconcile_status"] = "DRIFT_WARNING"
    send4, r4 = should_send_tournament_telegram(
        cycle=105,
        records=alert_records,
        tg_report_every=4,
        now_utc=t_mid,
        prev_utc_date="2026-10-31",
    )
    assert send4 is True and any("RECON_ALERT" in x for x in r4)

    # 5a. Periodic Milestone (Cycle 108 % 4 == 0) -> SEND
    send5a, r5a = should_send_tournament_telegram(
        cycle=108,
        records=records,
        tg_report_every=4,
        now_utc=t_mid,
        prev_utc_date="2026-10-31",
    )
    assert send5a is True and "PERIODIC_4C" in r5a

    # 5b. Daily Milestone (00:00 UTC date transition) -> SEND
    t_new_day = datetime(2026, 11, 1, 0, 15, tzinfo=timezone.utc)
    send5b, r5b = should_send_tournament_telegram(
        cycle=105,
        records=records,
        tg_report_every=4,
        now_utc=t_new_day,
        prev_utc_date="2026-10-31",
    )
    assert send5b is True and "DAILY_MILESTONE_00UTC" in r5b
