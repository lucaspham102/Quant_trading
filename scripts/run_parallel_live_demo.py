"""Parallel live/shadow tournament runner for 1,000 USDT strategy profiles.

Runs multiple profiles side-by-side:
- At most ONE profile executes real persistent orders on Binance Testnet.
- Remaining profiles run in SHADOW_LIVE mode with simulated fees and isolated state.

Sprint 25.1 hardening:
- Dual Spot/Perp price feed: Carry legs are marked with the real Spot mid (long leg) and
  the USDⓈ-M Perp mark (short leg), so basis PnL is measured instead of assumed to be zero.
- TESTNET_REAL profiles reconcile their ledger against the real Binance testnet balances via
  an isolated ``demo_testnet_baseline.json`` baseline (drift + SYNCED / DRIFT_WARNING).
"""

from __future__ import annotations

import argparse
import dataclasses
import socket
socket.setdefaulttimeout(25.0)
import threading
from collections.abc import Callable
import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.request

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.barbell_portfolio import BarbellConfig
from src.binance_public_feed import fetch_binance_shadow_snapshot, to_binance_symbol
from src.binance_signed_client import (
    DEFAULT_LIVE_RECV_WINDOW_MS,
    BinanceSignedTestnetClient,
    load_dotenv_testnet,
)
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.cross_exchange_carry import LIQUID_8_SYMBOLS
from src.directional_trend_sleeve import (
    DrawdownCircuitBreakerConfig,
    TrendSleeveConfig,
    TrendSleeveState,
    compute_triple_confirmed_trend_weights,
    evaluate_drawdown_circuit_breaker,
    trend_weights_to_target_qty,
)
from src.dynamic_carry_basis import DynamicCarryConfig
from src.exchange_reconciler import (
    compute_testnet_delta_snapshot,
    load_or_create_testnet_baseline,
)
from src.notifier import (
    TelegramNotifier,
    format_tournament_telegram_report,
    positions_changed,
    should_send_tournament_telegram,
)
from src.raw_data_refresh import cache_last_timestamps, refresh_raw_data
from src.state_store import StateStore
from src.stat_arb_sleeve import (
    ReversalSleeveConfig,
    compute_funding_conditioned_reversal_weights,
    reversal_weights_to_target_qty,
)
from src.strategy_profiles import (
    PROFILE_REGISTRY,
    StrategyProfileConfig,
    compute_step_passive_yield_usd,
)

logger = logging.getLogger(__name__)

CARRY_FEE_RATE = (7.5 + 5.0) / 10000.0
OVERLAY_FEE_RATE = 5.0 / 10000.0

#: Perp taker fee applied to every Trend sleeve delta (same 5.0 bps as the overlay leg).
TREND_FEE_RATE = 5.0 / 10000.0

#: Lot step per symbol for quantity quantization (Binance USD-M / Spot filters).
DEMO_LOT_STEPS: dict[str, float] = {
    "BTC/USDT:USDT": 0.001,
    "ETH/USDT:USDT": 0.001,
    "SOL/USDT:USDT": 0.01,
    "BNB/USDT:USDT": 0.01,
    "LINK/USDT:USDT": 0.01,
    "AVAX/USDT:USDT": 1.0,
    "XRP/USDT:USDT": 1.0,
    "DOGE/USDT:USDT": 1.0,
}

#: Binance Spot bookTicker endpoint (public, read-only). The USDT-M feed lives in
#: ``src.binance_public_feed`` and only covers Perp, hence this second leg.
SPOT_BOOK_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"

#: Isolated testnet baseline for the demo runner. Never share it with ``run_live.py``.
#: Isolated testnet baseline for the demo runner. Never share it with ``run_live.py``.
DEMO_TESTNET_BASELINE_FILENAME = "demo_testnet_baseline.json"

#: Heartbeat state filename for active watchdog monitoring.
HEARTBEAT_FILENAME = "tournament_heartbeat.json"

#: Maximum wall-clock seconds for a single cycle before the watchdog aborts the process.
DEFAULT_MAX_CYCLE_DURATION_SEC = 600.0


def disable_windows_quick_edit() -> bool:
    """Disable Windows Console QuickEdit mode to prevent stdout freeze on mouse click."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if not h_stdin or h_stdin == -1:
            return False

        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            return False

        new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        return bool(kernel32.SetConsoleMode(h_stdin, new_mode))
    except Exception:
        return False


def prevent_windows_sleep() -> bool:
    """Prevent Windows from entering Sleep mode while the bot is running (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel32 = getattr(ctypes, "windll", None)
        if kernel32 is None or not hasattr(kernel32, "kernel32"):
            return False
        res = ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return bool(res != 0)
    except Exception:
        return False


def enable_windows_ansi_colors() -> bool:
    """Enable Virtual Terminal Processing on Windows Console for ANSI color codes."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        h_stdout = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        if not h_stdout or h_stdout == -1:
            return False
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(h_stdout, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


def _backup_valid_json_file(path: str) -> None:
    """If path exists and contains valid JSON object, copy it to path.bak before overwrite."""
    bak_path = f"{path}.bak"
    if not os.path.exists(path):
        return
    try:
        if os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8") as existing_f:
                existing_obj = json.load(existing_f)
            if isinstance(existing_obj, dict):
                shutil.copy2(path, bak_path)
    except Exception:
        pass


def write_heartbeat(
    demo_dir: str,
    cycle: int,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write an atomic heartbeat record with PID, cycle, phase, timestamp and epoch."""
    path = os.path.join(demo_dir, HEARTBEAT_FILENAME)
    payload = {
        "pid": os.getpid(),
        "cycle": int(cycle),
        "phase": str(phase),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epoch_sec": float(time.time()),
    }
    if extra:
        payload.update(extra)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    _backup_valid_json_file(path)

    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)

    bak_path = f"{path}.bak"
    if not os.path.exists(bak_path) and os.path.exists(path):
        try:
            shutil.copy2(path, bak_path)
        except Exception:
            pass
    return path


def _default_network_probe(timeout_sec: float = 5.0) -> bool:
    """Lightweight connectivity probe to Binance public endpoints."""
    for url in (
        "https://fapi.binance.com/fapi/v1/ping",
        "https://api.binance.com/api/v3/ping",
    ):
        try:
            with urllib.request.urlopen(url, timeout=max(1.0, float(timeout_sec))) as resp:
                if int(getattr(resp, "status", 200)) < 500:
                    return True
        except Exception:
            continue
    return False


def wait_for_network_connectivity(
    max_wait_sec: float = 300.0,
    check_interval_sec: float = 15.0,
    *,
    demo_dir: str | None = None,
    cycle: int = 0,
    timeout_sec: float = 5.0,
    probe_fn: Callable[[], bool] | None = None,
    status_out: dict[str, Any] | None = None,
) -> bool:
    """Wait for Internet/API connectivity up to max_wait_sec, updating WAITING_FOR_NETWORK heartbeat."""
    check_fn = probe_fn if probe_fn is not None else (lambda: _default_network_probe(timeout_sec))
    start_mono = time.monotonic()
    max_w = max(0.0, float(max_wait_sec))
    interval = max(0.01, float(check_interval_sec))
    attempt = 1

    try:
        ok = bool(check_fn())
    except Exception:
        ok = False

    if ok:
        if status_out is not None:
            status_out.update(
                {
                    "connected": True,
                    "recovered": False,
                    "attempts": 1,
                    "waited_sec": 0.0,
                }
            )
        return True

    logger.warning(
        "WARNING: NETWORK_OUTAGE_DETECTED at cycle %d - retrying every %.1fs (max_wait=%.1fs)",
        cycle,
        interval,
        max_w,
    )
    print(
        f"[NET] WARNING: Network unreachable at cycle {cycle}. Waiting for recovery "
        f"(interval={interval:.0f}s, max={max_w:.0f}s)...",
        flush=True,
    )

    while True:
        elapsed = time.monotonic() - start_mono
        remaining = max_w - elapsed
        if remaining <= 0:
            break
        if demo_dir:
            write_heartbeat(
                demo_dir,
                cycle=cycle,
                phase="WAITING_FOR_NETWORK",
                extra={
                    "attempt": attempt,
                    "elapsed_sec": round(elapsed, 2),
                    "max_wait_sec": max_w,
                },
            )
        time.sleep(min(interval, remaining))
        attempt += 1
        try:
            ok = bool(check_fn())
        except Exception:
            ok = False
        if ok:
            waited = time.monotonic() - start_mono
            logger.info(
                "NETWORK_RECOVERED at cycle %d after %d attempts (%.1fs)",
                cycle,
                attempt,
                waited,
            )
            print(
                f"[NET] OK: Network recovered after {attempt} attempts ({waited:.1f}s).",
                flush=True,
            )
            if demo_dir:
                write_heartbeat(
                    demo_dir,
                    cycle=cycle,
                    phase="NETWORK_RECOVERED",
                    extra={"attempts": attempt, "waited_sec": round(waited, 2)},
                )
            if status_out is not None:
                status_out.update(
                    {
                        "connected": True,
                        "recovered": True,
                        "attempts": attempt,
                        "waited_sec": waited,
                    }
                )
            return True

    waited = time.monotonic() - start_mono
    logger.error(
        "NETWORK_OUTAGE_TIMEOUT at cycle %d after %d attempts (%.1fs)",
        cycle,
        attempt,
        waited,
    )
    if status_out is not None:
        status_out.update(
            {
                "connected": False,
                "recovered": False,
                "attempts": attempt,
                "waited_sec": waited,
            }
        )
    return False


class CycleWatchdog:
    """Cycle-level watchdog thread that forces process termination on hang."""

    def __init__(self, max_duration_sec: float = DEFAULT_MAX_CYCLE_DURATION_SEC) -> None:
        self.max_duration_sec = float(max_duration_sec)
        self._current_cycle: int = 0
        self._cycle_start_mono: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="CycleWatchdog")
        self._thread.start()

    def start_cycle(self, cycle: int) -> None:
        with self._lock:
            self._current_cycle = int(cycle)
            self._cycle_start_mono = time.monotonic()

    def stop_cycle(self) -> None:
        with self._lock:
            self._cycle_start_mono = None

    def shutdown(self) -> None:
        self._stop_event.set()

    def _monitor_loop(self) -> None:
        interval = min(1.0, max(0.05, self.max_duration_sec / 4.0))
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=interval):
                break
            with self._lock:
                if self._cycle_start_mono is not None:
                    elapsed = time.monotonic() - self._cycle_start_mono
                    if elapsed > self.max_duration_sec:
                        logger.critical(
                            "CRITICAL: CYCLE_WATCHDOG_TIMEOUT - cycle %d exceeded %.1fs ceiling (elapsed=%.1fs). Forcing os._exit(2).",
                            self._current_cycle,
                            self.max_duration_sec,
                            elapsed,
                        )
                        print(
                            f"\n[WATCHDOG] CRITICAL: CYCLE_WATCHDOG_TIMEOUT - cycle {self._current_cycle} "
                            f"exceeded {self.max_duration_sec:.1f}s ceiling (elapsed: {elapsed:.1f}s). Aborting process immediately.\n",
                            flush=True,
                        )
                        os._exit(2)


def interruptible_sleep(
    total_seconds: float,
    cycle: int,
    demo_dir: str,
    chunk_seconds: float = 60.0,
) -> None:
    """Sleeps total_seconds in chunks of chunk_seconds, updating heartbeat and flushing stdout."""
    start_mono = time.monotonic()
    total = float(total_seconds)
    chunk = max(0.01, float(chunk_seconds))
    while True:
        elapsed = time.monotonic() - start_mono
        remaining = total - elapsed
        if remaining <= 0:
            break
        step = min(remaining, chunk)
        time.sleep(step)
        now_remaining = max(0.0, total - (time.monotonic() - start_mono))
        write_heartbeat(
            demo_dir,
            cycle=cycle,
            phase="SLEEPING",
            extra={
                "sleep_remaining_sec": round(now_remaining, 2),
                "sleep_total_sec": float(total),
            },
        )
        sys.stdout.flush()

#: Reconcile drift tolerance (fraction of capital) before the status flips to DRIFT_WARNING.
RECONCILE_DRIFT_TOLERANCE_PCT = 0.015

#: Hours of 1h history handed to the Trend / Reversal sleeves (needs >= TrendSleeveConfig.min_history_hours).
TREND_HISTORY_BARS = 1500
#: Window the Carry/Overlay executor keeps using, so the Carry baseline is bit-for-bit unchanged.
CARRY_LOOKBACK_BARS = 720
#: Smoothing of the funding input shared with the backtest (span 9 over funding events).
FUNDING_EWMA_ALPHA = 2.0 / (9.0 + 1.0)
#: Reversal sleeve parameters copied from run_1k_profile_tournament_backtest.py.
REVERSAL_WINDOW_BARS = 240
REVERSAL_MAX_IMBALANCE_USD = 10.0
#: Last aligned candle older than this (hours) raises a STALE_DATA warning.
STALE_DATA_MAX_AGE_HOURS = 2.0


def _num(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
        if math.isfinite(val):
            return val
    except (TypeError, ValueError):
        pass
    return float(default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel 1,000 USDT live/shadow strategy profile demo."
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="baseline_1k,alpha_boost_1k,turbo_neutral_1k",
        help="Comma-separated list of profile names to run.",
    )
    parser.add_argument(
        "--live-testnet-profile",
        type=str,
        default="baseline_1k",
        help="Profile allowed to place real testnet orders ('none' for all shadow).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to run (0 for infinite loop).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=3600.0,
        help="Seconds to sleep between cycles.",
    )
    parser.add_argument(
        "--reset-demo-state",
        action="store_true",
        help="Delete existing demo ledger and bot state files before starting.",
    )
    parser.add_argument(
        "--scoreboard-log",
        type=str,
        default="logs/sprint25_parallel_demo_scoreboard.jsonl",
        help="Path to append JSONL scoreboard records.",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="data/state",
        help="Directory for per-profile ledger and bot state files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory for raw OHLCV/funding data.",
    )
    parser.add_argument(
        "--shadow-timeout-sec",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for public market snapshot.",
    )
    parser.add_argument(
        "--no-refresh-raw",
        action="store_true",
        help="Do not refresh data/raw parquet files at the start of each cycle (static data test).",
    )
    parser.add_argument(
        "--no-directional-real-orders",
        action="store_true",
        help="In TESTNET_REAL keep Trend/Reversal on the virtual ledger (no real orders).",
    )
    parser.add_argument(
        "--max-cycle-duration-sec",
        type=float,
        default=DEFAULT_MAX_CYCLE_DURATION_SEC,
        help="Maximum duration in seconds per cycle before watchdog aborts process.",
    )
    parser.add_argument(
        "--tg-report-every",
        type=int,
        default=4,
        help="Send periodic Telegram report every N cycles (default: 4).",
    )
    return parser.parse_args(argv)


def build_profiles(names: str | list[str] | tuple[str, ...]) -> dict[str, StrategyProfileConfig]:
    if isinstance(names, str):
        parsed_names = [n.strip() for n in names.split(",") if n.strip()]
    else:
        parsed_names = [str(n).strip() for n in names if str(n).strip()]

    if not parsed_names:
        raise ValueError("At least one profile name must be specified.")

    out: dict[str, StrategyProfileConfig] = {}
    for name in parsed_names:
        if name == "alpha_boost_1k_pure_equal":
            base = PROFILE_REGISTRY["alpha_boost_1k"]
            out[name] = dataclasses.replace(
                base,
                name="alpha_boost_1k_pure_equal",
                carry_weighting_mode="equal",
                carry_tilt_alpha=0.0,
            )
        elif name in PROFILE_REGISTRY:
            out[name] = PROFILE_REGISTRY[name]
        else:
            raise ValueError(f"Unknown strategy profile: {name!r}")
    return out


def demo_paths(
    profile: str | StrategyProfileConfig, demo_dir: str = "data/state"
) -> tuple[str, str]:
    pname = profile.name if isinstance(profile, StrategyProfileConfig) else str(profile)
    ledger_path = os.path.join(str(demo_dir), f"demo_{pname}.json")
    bot_state_path = os.path.join(str(demo_dir), f"demo_bot_{pname}.json")
    return ledger_path, bot_state_path


def _default_ledger(profile: str | StrategyProfileConfig) -> dict[str, Any]:
    if isinstance(profile, StrategyProfileConfig):
        pname = profile.name
        capital = _num(profile.capital_usd, 1000.0)
    else:
        pname = str(profile)
        if pname in PROFILE_REGISTRY:
            capital = _num(PROFILE_REGISTRY[pname].capital_usd, 1000.0)
        else:
            capital = 1000.0

    return {
        "profile": pname,
        "mode": "SHADOW_LIVE",
        "cycle": 0,
        "updated_at": None,
        "last_funding_ts": None,
        "last_prices": {},
        "realized_pnl_usd": 0.0,
        "fees_paid_usd": 0.0,
        "funding_pnl_usd": 0.0,
        "overlay_pnl_usd": 0.0,
        "trend_pnl_usd": 0.0,
        "reversal_pnl_usd": 0.0,
        "basis_pnl_usd": 0.0,
        "spot_unrealized_pnl_usd": 0.0,
        "perp_unrealized_pnl_usd": 0.0,
        "cumulative_passive_yield_usd": 0.0,
        "pure_trading_equity_usd": capital,
        "yield_stacked_equity_usd": capital,
        "peak_equity_usd": capital,
        "max_dd_pct": 0.0,
        "exchange_recon_equity_usd": None,
        "recon_drift_usd": None,
        "reconcile_status": "NOT_APPLICABLE",
        "positions_snapshot": {},
        "carry_slots": 0,
        "overlay_long_usd": 0.0,
        "overlay_short_usd": 0.0,
        "trend_long_usd": 0.0,
        "trend_gross_usd": 0.0,
        "reversal_gross_usd": 0.0,
        "futures_leverage": 0.0,
        "status": "INIT",
        "circuit_breaker_state": "DISABLED",
        "circuit_breaker_risk_scale": 1.0,
        "circuit_breaker_triggers": 0,
        "trend_hard_lock_bars": None,
        "trend_sleeve_state": {"peak_close": {}, "active": {}},
        "last_trend_rebalance_ts": None,
        "note": "",
    }


def _sanitize_ledger(ledger: dict[str, Any], default_capital: float = 1000.0) -> dict[str, Any]:
    float_keys = (
        "realized_pnl_usd",
        "fees_paid_usd",
        "funding_pnl_usd",
        "overlay_pnl_usd",
        "trend_pnl_usd",
        "reversal_pnl_usd",
        "basis_pnl_usd",
        "spot_unrealized_pnl_usd",
        "perp_unrealized_pnl_usd",
        "cumulative_passive_yield_usd",
        "pure_trading_equity_usd",
        "yield_stacked_equity_usd",
        "peak_equity_usd",
        "max_dd_pct",
        "overlay_long_usd",
        "overlay_short_usd",
        "trend_long_usd",
        "trend_gross_usd",
        "reversal_gross_usd",
        "futures_leverage",
        "circuit_breaker_risk_scale",
    )
    for k in float_keys:
        fallback = default_capital if k in (
            "pure_trading_equity_usd",
            "yield_stacked_equity_usd",
            "peak_equity_usd",
        ) else 0.0
        ledger[k] = _num(ledger.get(k), fallback)

    # Reconcile fields are tri-state: a real number once the exchange answered, else None.
    for k in ("exchange_recon_equity_usd", "recon_drift_usd"):
        raw = ledger.get(k, None)
        ledger[k] = None if raw is None else _num(raw, 0.0)
    ledger["reconcile_status"] = str(ledger.get("reconcile_status", "NOT_APPLICABLE"))

    # The hard-stop cooldown counter is also tri-state: None when no hard stop is in effect.
    _lock_raw = ledger.get("trend_hard_lock_bars", None)
    ledger["trend_hard_lock_bars"] = None if _lock_raw is None else int(_num(_lock_raw, 0.0))

    ledger["cycle"] = int(_num(ledger.get("cycle"), 0.0))
    ledger["carry_slots"] = int(_num(ledger.get("carry_slots"), 0.0))
    ledger["profile"] = str(ledger.get("profile", ""))
    ledger["mode"] = str(ledger.get("mode", "SHADOW_LIVE"))
    ledger["status"] = str(ledger.get("status", "INIT"))
    ledger["note"] = str(ledger.get("note", ""))
    ledger["circuit_breaker_state"] = str(ledger.get("circuit_breaker_state", "DISABLED"))
    ledger["circuit_breaker_triggers"] = int(_num(ledger.get("circuit_breaker_triggers"), 0.0))
    ledger["last_trend_rebalance_ts"] = (
        None
        if ledger.get("last_trend_rebalance_ts") is None
        else str(ledger.get("last_trend_rebalance_ts"))
    )
    raw_trend_state = ledger.get("trend_sleeve_state")
    ledger["trend_sleeve_state"] = (
        raw_trend_state
        if isinstance(raw_trend_state, dict)
        else {"peak_close": {}, "active": {}}
    )

    raw_prices = ledger.get("last_prices")
    if isinstance(raw_prices, dict):
        ledger["last_prices"] = {str(s): _num(v, 0.0) for s, v in raw_prices.items()}
    else:
        ledger["last_prices"] = {}

    raw_pos = ledger.get("positions_snapshot")
    clean_pos: dict[str, dict[str, Any]] = {}
    if isinstance(raw_pos, dict):
        for sym, info in raw_pos.items():
            if isinstance(info, dict):
                leg = {
                    "symbol": str(info.get("symbol") or str(sym).split("|")[0]),
                    "qty": _num(info.get("qty"), 0.0),
                    "entry_price": _num(info.get("entry_price"), 0.0),
                    "kind": str(info.get("kind", "carry")),
                    "signed_usd": _num(info.get("signed_usd"), 0.0),
                }
                # Sprint 25.1: spot/perp legs are tracked separately; only carry legs that
                # actually carry the split are stamped, so legacy ledgers stay untouched.
                for opt_key in ("spot_qty", "perp_qty", "spot_entry", "perp_entry"):
                    if opt_key in info:
                        leg[opt_key] = _num(info.get(opt_key), 0.0)
                clean_pos[str(sym)] = leg
    ledger["positions_snapshot"] = clean_pos
    return ledger


def load_ledger(path: str, profile: str | StrategyProfileConfig) -> dict[str, Any]:
    base = _default_ledger(profile)
    default_cap = _num(base["pure_trading_equity_usd"], 1000.0)
    bak_path = f"{path}.bak"
    loaded: dict[str, Any] | None = None

    if os.path.exists(path):
        try:
            if os.path.getsize(path) == 0:
                raise ValueError("Empty ledger file (0 bytes)")
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("Ledger root is not a JSON object")
            loaded = raw
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "Primary ledger %s is corrupted or unreadable (%s); checking backup %s",
                path,
                exc,
                bak_path,
            )

    if loaded is None and os.path.exists(bak_path):
        try:
            if os.path.getsize(bak_path) == 0:
                raise ValueError("Empty backup ledger file (0 bytes)")
            with open(bak_path, "r", encoding="utf-8") as f:
                raw_bak = json.load(f)
            if isinstance(raw_bak, dict):
                loaded = raw_bak
                logger.warning("WARNING: RECOVERED_STATE_FROM_BACKUP (%s -> %s)", bak_path, path)
                try:
                    shutil.copy2(bak_path, path)
                except OSError:
                    pass
        except Exception as bak_exc:
            logger.error("Backup ledger %s is also unreadable: %s", bak_path, bak_exc)

    if isinstance(loaded, dict):
        base.update(loaded)
    return _sanitize_ledger(base, default_capital=default_cap)


def save_ledger_atomic(path: str, ledger: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    clean = _sanitize_ledger(dict(ledger))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    _backup_valid_json_file(path)

    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)

    bak_path = f"{path}.bak"
    if not os.path.exists(bak_path) and os.path.exists(path):
        try:
            shutil.copy2(path, bak_path)
        except Exception:
            pass


def build_demo_executor(
    profile: StrategyProfileConfig,
    *,
    mode: str,
    bot_state_path: str,
    raw_dir: str = "data/raw",
    execute_real: bool = False,
    signed_client: Any = None,
    lookback_hours: int | None = None,
) -> CarryLiveExecutor:
    if not execute_real or mode in ("SHADOW_LIVE", "paper"):
        exec_mode = "paper"
    elif mode in ("TESTNET_REAL", "testnet"):
        exec_mode = "testnet"
    else:
        exec_mode = str(mode)

    extra_kwargs: dict[str, Any] = {}
    if lookback_hours is not None:
        extra_kwargs["lookback_hours"] = int(lookback_hours)
    return CarryLiveExecutor(
        **extra_kwargs,
        config=TradingConfig(
            timeframe="1h",
            raw_data_dir=str(raw_dir),
            state_file_path=str(bot_state_path),
        ),
        carry_config=DynamicCarryConfig(
            initial_capital=profile.capital_usd,
            ema_span=42,
            top_k=profile.carry_top_k,
            entry_threshold=profile.carry_entry_bps * 1e-4,
            exit_threshold=profile.carry_exit_bps * 1e-4,
            min_hold_periods=profile.carry_min_hold_bars,
            buffer_rank=profile.carry_buffer_rank,
            w_basis=0.0,
            basis_ema_hours=168,
            expected_hold_periods=profile.carry_min_hold_bars,
            switch_rule="tier1b_buffer_rank",
            eta_override=profile.carry_eta,
        ),
        mode=exec_mode,
        symbols=list(LIQUID_8_SYMBOLS),
        state_store=StateStore(str(bot_state_path)),
        capital=profile.capital_usd,
        initial_capital=profile.capital_usd,
        overlay_config=BarbellConfig(
            initial_capital=profile.capital_usd,
            carry_variant="tier1b",
            gating_mode="smooth_conviction",
        ),
        strategy="barbell",
        tier1b_gate=True,
        testnet_dry_state=False,
        signed_client=signed_client,
        execute_real_orders=bool(execute_real),
        persistent_positions=bool(execute_real),
        profile=profile.name,
        reset_state=False,
    )


def _parse_iso_utc(ts_str: Any) -> datetime | None:
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _latest_funding_stamp_utc(now_utc: datetime) -> datetime:
    utc_dt = (
        now_utc.replace(tzinfo=timezone.utc)
        if now_utc.tzinfo is None
        else now_utc.astimezone(timezone.utc)
    )
    funding_hour = (utc_dt.hour // 8) * 8
    return utc_dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)


def _extract_mid_price(
    sym: str, sym_map: dict[str, Any], last_prices: dict[str, float], fallback: float = 0.0
) -> float:
    info = sym_map.get(sym) if isinstance(sym_map, dict) else None
    if isinstance(info, dict):
        mid = _num(info.get("mid"), 0.0)
        if mid <= 0.0:
            bid = _num(info.get("bid"), 0.0)
            ask = _num(info.get("ask"), 0.0)
            if bid > 0.0 and ask > 0.0:
                mid = 0.5 * (bid + ask)
        if mid <= 0.0:
            mid = _num(info.get("mark_price"), 0.0)
        if mid > 0.0:
            return mid
    prev = _num(last_prices.get(sym), 0.0)
    if prev > 0.0:
        return prev
    return _num(fallback, 0.0)


def _mid_from_map(prices: Mapping[str, Any] | None, sym: str, fallback: float = 0.0) -> float:
    info = prices.get(sym) if isinstance(prices, Mapping) else None
    if isinstance(info, Mapping):
        mid = _num(info.get("mid"), 0.0)
        if mid > 0.0:
            return mid
        bid = _num(info.get("bid"), 0.0)
        ask = _num(info.get("ask"), 0.0)
        if bid > 0.0 and ask > 0.0:
            return 0.5 * (bid + ask)
    return _num(fallback, 0.0)


def _mark_from_map(prices: Mapping[str, Any] | None, sym: str, fallback: float = 0.0) -> float:
    info = prices.get(sym) if isinstance(prices, Mapping) else None
    if isinstance(info, Mapping):
        for key in ("mark", "mark_price", "mid"):
            val = _num(info.get(key), 0.0)
            if val > 0.0:
                return val
    return _num(fallback, 0.0)


def dual_price_maps(
    snapshot: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Split one market snapshot into (spot_prices, perp_prices).

    Spot legs are exported by the Spot bookTicker feed, Perp legs by the USDⓈ-M feed.
    When the Spot feed is missing a symbol we fall back to the Perp mid, which degrades
    basis PnL to zero instead of inventing a price.
    """
    snap = snapshot if isinstance(snapshot, dict) else {}
    perp_raw = snap.get("perp_symbols")
    if not isinstance(perp_raw, dict) or not perp_raw:
        perp_raw = snap.get("symbols", {})
    spot_raw = snap.get("spot_symbols", {})

    perp: dict[str, dict[str, float]] = {}
    for sym, info in (perp_raw.items() if isinstance(perp_raw, dict) else []):
        if not isinstance(info, dict):
            continue
        bid = _num(info.get("bid"), 0.0)
        ask = _num(info.get("ask"), 0.0)
        mid = _num(info.get("mid"), 0.0)
        if mid <= 0.0 and bid > 0.0 and ask > 0.0:
            mid = 0.5 * (bid + ask)
        mark = _num(info.get("mark"), _num(info.get("mark_price"), 0.0))
        if mark <= 0.0:
            mark = mid
        if mid <= 0.0:
            mid = mark
        if mid <= 0.0:
            continue
        perp[str(sym)] = {
            "bid": bid,
            "ask": ask,
            "mid": float(mid),
            "mark": float(mark if mark > 0.0 else mid),
        }

    spot: dict[str, dict[str, float]] = {}
    for sym, info in (spot_raw.items() if isinstance(spot_raw, dict) else []):
        if not isinstance(info, dict):
            continue
        bid = _num(info.get("bid"), 0.0)
        ask = _num(info.get("ask"), 0.0)
        mid = _num(info.get("mid"), 0.0)
        if mid <= 0.0 and bid > 0.0 and ask > 0.0:
            mid = 0.5 * (bid + ask)
        if mid <= 0.0:
            continue
        spot[str(sym)] = {"bid": bid, "ask": ask, "mid": float(mid)}

    return spot, perp


def carry_basis_pnl(
    pos: dict[str, Any],
    spot_prices: Mapping[str, Any] | None,
    perp_prices: Mapping[str, Any] | None,
    fallback_mid: float = 0.0,
) -> tuple[float, float, float]:
    """Mark one Carry leg (spot long + perp short) and return the real basis PnL.

    Returns ``(basis_usd, spot_unrealized_usd, perp_unrealized_usd)`` where::

        spot_unrealized = spot_qty * (spot_mid  - spot_entry)
        perp_unrealized = perp_qty * (perp_mark - perp_entry)   # perp_qty is negative
        basis_usd       = spot_unrealized + perp_unrealized

    When Spot and Perp move together the two legs cancel exactly; only the basis moves.
    ledger written before Sprint 25.1 only stored one price, in which case both legs are
    marked from that same price and the basis collapses to ``qty * (spot_mid - perp_mark)``.

    Whenever one of the two order books is unavailable the basis is not observable, so the
    leg is treated as perfectly neutral (0.0) instead of guessing a price.
    """
    if not isinstance(pos, dict):
        return 0.0, 0.0, 0.0
    sym = _pos_symbol("", pos)
    if not sym:
        return 0.0, 0.0, 0.0
    spot_info = spot_prices.get(sym) if isinstance(spot_prices, Mapping) else None
    perp_info = perp_prices.get(sym) if isinstance(perp_prices, Mapping) else None
    if not isinstance(spot_info, Mapping) or not isinstance(perp_info, Mapping):
        return 0.0, 0.0, 0.0
    spot_mid = _mid_from_map(spot_prices, sym, _num(fallback_mid, 0.0))
    perp_mark = _mark_from_map(perp_prices, sym, _num(fallback_mid, 0.0))
    if spot_mid <= 0.0 or perp_mark <= 0.0:
        return 0.0, 0.0, 0.0

    qty = _num(pos.get("qty"), 0.0)
    spot_qty = _num(pos.get("spot_qty"), abs(qty))
    perp_qty = _num(pos.get("perp_qty"), -abs(qty))
    legacy_entry = _num(pos.get("entry_price"), 0.0)
    spot_entry = _num(pos.get("spot_entry"), legacy_entry)
    perp_entry = _num(pos.get("perp_entry"), legacy_entry)
    if spot_entry <= 0.0 and perp_entry <= 0.0:
        return 0.0, 0.0, 0.0

    spot_u = spot_qty * (spot_mid - spot_entry) if spot_entry > 0.0 else 0.0
    perp_u = perp_qty * (perp_mark - perp_entry) if perp_entry > 0.0 else 0.0
    return float(spot_u + perp_u), float(spot_u), float(perp_u)


def accrue_cycle_accounting(
    ledger: dict[str, Any],
    profile: StrategyProfileConfig,
    snapshot: dict[str, Any] | None,
    now_utc: datetime,
    positions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    utc_now = (
        now_utc.replace(tzinfo=timezone.utc)
        if now_utc.tzinfo is None
        else now_utc.astimezone(timezone.utc)
    )
    sym_map = snapshot.get("symbols", {}) if isinstance(snapshot, dict) else {}
    spot_prices, perp_prices = dual_price_maps(snapshot)
    last_prices = ledger.setdefault("last_prices", {})
    pos_map = (
        positions
        if isinstance(positions, dict)
        else ledger.get("positions_snapshot", {})
    )

    for sym, info in (sym_map.items() if isinstance(sym_map, dict) else []):
        if isinstance(info, dict):
            m = _extract_mid_price(sym, sym_map, last_prices, 0.0)
            if m > 0.0:
                last_prices[str(sym)] = m

    latest_funding_dt = _latest_funding_stamp_utc(utc_now)
    latest_funding_iso = latest_funding_dt.isoformat()
    prev_funding_dt = _parse_iso_utc(ledger.get("last_funding_ts"))

    if prev_funding_dt is None:
        ledger["last_funding_ts"] = latest_funding_iso
    elif latest_funding_dt > prev_funding_dt:
        funding_delta = 0.0
        overlay_funding_delta = 0.0
        trend_funding_delta = 0.0
        reversal_funding_delta = 0.0
        for key, pos in pos_map.items():
            if not isinstance(pos, dict):
                continue
            sym = _pos_symbol(str(key), pos)
            kind = str(pos.get("kind", "carry"))
            qty = _num(pos.get("qty"), 0.0)
            entry = _num(pos.get("entry_price"), 0.0)
            signed_usd = _num(pos.get("signed_usd"), qty * entry)
            mid = _extract_mid_price(sym, sym_map, last_prices, entry)
            s_info = sym_map.get(sym) if isinstance(sym_map, dict) else None
            rate = _num(s_info.get("funding_rate") if isinstance(s_info, dict) else 0.0, 0.0)
            if rate == 0.0 and isinstance(s_info, dict):
                # Public feed publishes basis points (funding_rate_bps); older fixtures use
                # the raw ratio. Supporting both keeps live funding from silently staying 0.
                rate = _num(s_info.get("funding_rate_bps"), 0.0) / 10000.0
            if kind == "carry":
                # Short perp leg receives funding when the rate is positive.
                funding_delta += qty * mid * rate
            elif kind == "overlay":
                ov_f = -signed_usd * rate
                funding_delta += ov_f
                overlay_funding_delta += ov_f
            elif kind == "trend":
                # Long perp leg PAYS funding when the rate is positive. Use mark notional
                # (qty * mid) so the number matches the backtest convention exactly, even
                # though the stored signed_usd is a cost basis.
                trend_f = -qty * mid * rate
                funding_delta += trend_f
                trend_funding_delta += trend_f
        ledger["funding_pnl_usd"] = _num(ledger.get("funding_pnl_usd"), 0.0) + funding_delta
        ledger["realized_pnl_usd"] = _num(ledger.get("realized_pnl_usd"), 0.0) + funding_delta
        ledger["overlay_pnl_usd"] = _num(ledger.get("overlay_pnl_usd"), 0.0) + overlay_funding_delta
        ledger["trend_pnl_usd"] = _num(ledger.get("trend_pnl_usd"), 0.0) + trend_funding_delta
        ledger["last_funding_ts"] = latest_funding_iso

    prev_updated_dt = _parse_iso_utc(ledger.get("updated_at"))
    if prev_updated_dt is not None:
        raw_dt_hours = (utc_now - prev_updated_dt).total_seconds() / 3600.0
        dt_hours = min(24.0, max(0.0, raw_dt_hours))
    else:
        dt_hours = 0.0

    unrealized_carry = 0.0
    unrealized_overlay = 0.0
    unrealized_trend = 0.0
    unrealized_reversal = 0.0
    basis_total = 0.0
    spot_unreal_total = 0.0
    perp_unreal_total = 0.0
    deployed_spot: dict[str, float] = {}

    for key, pos in pos_map.items():
        if not isinstance(pos, dict):
            continue
        sym = _pos_symbol(str(key), pos)
        kind = str(pos.get("kind", "carry"))
        qty = _num(pos.get("qty"), 0.0)
        entry = _num(pos.get("entry_price"), 0.0)
        signed_usd = _num(pos.get("signed_usd"), qty * entry)
        mid = _extract_mid_price(sym, sym_map, last_prices, entry)
        if mid > 0.0:
            last_prices[sym] = mid

        if kind == "carry":
            # Spot long + perp short: directional PnL nets out, only the basis remains.
            basis, spot_u, perp_u = carry_basis_pnl(pos, spot_prices, perp_prices, mid)
            unrealized_carry += basis
            basis_total += basis
            spot_unreal_total += spot_u
            perp_unreal_total += perp_u
            spot_mid = _mid_from_map(spot_prices, sym, mid)
            deployed_spot[sym] = abs(_num(pos.get("spot_qty"), abs(qty))) * spot_mid
        elif kind == "overlay":
            if entry > 0.0 and mid > 0.0:
                unrealized_overlay += signed_usd * (mid / entry - 1.0)
        elif kind == "trend":
            if entry > 0.0 and mid > 0.0:
                unrealized_trend += signed_usd * (mid / entry - 1.0)
        elif kind == "reversal":
            if entry > 0.0 and mid > 0.0:
                unrealized_reversal += signed_usd * (mid / entry - 1.0)

    ledger["basis_pnl_usd"] = basis_total
    ledger["spot_unrealized_pnl_usd"] = spot_unreal_total
    ledger["perp_unrealized_pnl_usd"] = perp_unreal_total

    yield_res = compute_step_passive_yield_usd(profile, dt_hours, deployed_spot)
    step_passive = _num(yield_res.get("total_passive_yield_usd"), 0.0)
    ledger["cumulative_passive_yield_usd"] = (
        _num(ledger.get("cumulative_passive_yield_usd"), 0.0) + step_passive
    )

    realized = _num(ledger.get("realized_pnl_usd"), 0.0)
    unrealized_total = (
        unrealized_carry + unrealized_overlay + unrealized_trend + unrealized_reversal
    )
    pure_eq = _num(profile.capital_usd, 1000.0) + realized + unrealized_total
    yield_eq = pure_eq + _num(ledger.get("cumulative_passive_yield_usd"), 0.0)

    ledger["pure_trading_equity_usd"] = pure_eq
    ledger["yield_stacked_equity_usd"] = yield_eq

    peak = max(_num(ledger.get("peak_equity_usd"), profile.capital_usd), pure_eq)
    ledger["peak_equity_usd"] = peak
    dd_pct = ((peak - pure_eq) / peak * 100.0) if peak > 0.0 else 0.0
    ledger["max_dd_pct"] = max(_num(ledger.get("max_dd_pct"), 0.0), max(0.0, dd_pct))
    ledger["updated_at"] = utc_now.isoformat()
    return _sanitize_ledger(ledger, default_capital=profile.capital_usd)


def _snapshot_key(sym: str, kind: str) -> str:
    """Ledger key for one leg: carry keeps the plain symbol, the others are namespaced.

    Carry, overlay and trend can each hold the SAME symbol at the same time
    (e.g. BTC carry + BTC overlay short + BTC trend long), so no two sleeves may
    ever share one ledger slot.
    """
    if kind == "carry":
        return str(sym)
    return f"{sym}|{kind}"


def _pos_symbol(key: str, pos: dict[str, Any]) -> str:
    sym = str(pos.get("symbol") or "")
    if sym:
        return sym
    return str(key).split("|")[0]


def _executor_leg_maps(executor_state: Any) -> tuple[dict, dict]:
    carry_raw = getattr(executor_state, "carry_positions", None)
    if carry_raw is None and isinstance(executor_state, dict):
        carry_raw = executor_state.get("carry_positions", {})
    overlay_raw = getattr(executor_state, "overlay_positions", None)
    if overlay_raw is None and isinstance(executor_state, dict):
        overlay_raw = executor_state.get("overlay_positions", {})
    return (
        carry_raw if isinstance(carry_raw, dict) else {},
        overlay_raw if isinstance(overlay_raw, dict) else {},
    )


def _vwap_entry(old_mag: float, old_entry: float, new_mag: float, mid: float) -> float:
    """Average entry price when a leg is scaled up (never invents instant PnL)."""
    if new_mag <= 0.0:
        return 0.0
    if old_mag <= 0.0 or old_entry <= 0.0:
        return mid
    if new_mag <= old_mag:
        return old_entry
    return (old_mag * old_entry + (new_mag - old_mag) * mid) / new_mag


def refresh_position_ledger(
    ledger: dict[str, Any],
    executor_state: Any,
    snapshot: dict[str, Any] | None,
    *,
    charge_fees: bool,
) -> tuple[float, float, float]:
    """Sync the demo ledger with the executor legs.

    Returns (fees_usd, realized_carry_usd, realized_overlay_usd).

    A carry leg is delta neutral (spot long + perp short), so its DIRECTIONAL price PnL is
    zero by construction; it earns funding and basis only. Overlay legs are directional and
    are marked against their own average entry price, realising PnL on partial closes.
    """
    sym_map = snapshot.get("symbols", {}) if isinstance(snapshot, dict) else {}
    spot_prices, perp_prices = dual_price_maps(snapshot)
    last_prices = ledger.setdefault("last_prices", {})
    prev = ledger.get("positions_snapshot")
    prev = prev if isinstance(prev, dict) else {}
    carry_map, overlay_map = _executor_leg_maps(executor_state)

    new_snapshot: dict[str, dict[str, Any]] = {}
    fees = 0.0
    realized_overlay = 0.0

    symbols: set[str] = set(carry_map) | set(overlay_map)
    for key, pos in prev.items():
        if isinstance(pos, dict):
            symbols.add(_pos_symbol(str(key), pos))

    for sym in sorted(symbols):
        if not sym:
            continue
        mid = _extract_mid_price(sym, sym_map, last_prices, 0.0)
        if mid <= 0.0:
            continue
        last_prices[sym] = mid

        # ---- overlay leg (directional, marked to market)
        o_pos = overlay_map.get(sym)
        o_qty = _num(o_pos.get("qty"), 0.0) if isinstance(o_pos, dict) else 0.0
        if isinstance(o_pos, dict):
            if "signed_usd" in o_pos:
                new_signed = _num(o_pos.get("signed_usd"), 0.0)
            else:
                mag = abs(_num(o_pos.get("notional"), abs(o_qty) * mid))
                side = str(o_pos.get("side", o_pos.get("direction", ""))).lower()
                sign = -1.0 if (side in ("short", "sell", "-1") or o_qty < 0.0) else 1.0
                new_signed = sign * mag
        else:
            new_signed = 0.0

        old_ov = prev.get(_snapshot_key(sym, "overlay")) or {}
        old_signed = _num(old_ov.get("signed_usd"), 0.0)
        old_ov_entry = _num(old_ov.get("entry_price"), 0.0)
        if charge_fees:
            fees += abs(new_signed - old_signed) * OVERLAY_FEE_RATE

        if abs(new_signed) > 1e-9:
            same_sign = old_signed != 0.0 and (old_signed > 0.0) == (new_signed > 0.0)
            if same_sign and old_ov_entry > 0.0:
                old_mag = abs(old_signed)
                new_mag = abs(new_signed)
                if new_mag < old_mag - 1e-9:
                    closed_frac = 1.0 - new_mag / old_mag
                    realized_overlay += old_signed * (mid / old_ov_entry - 1.0) * closed_frac
                entry_px = _vwap_entry(old_mag, old_ov_entry, new_mag, mid)
            else:
                if old_signed != 0.0 and old_ov_entry > 0.0:
                    realized_overlay += old_signed * (mid / old_ov_entry - 1.0)
                exec_entry = _num(o_pos.get("entry_price"), 0.0) if isinstance(o_pos, dict) else 0.0
                entry_px = exec_entry if exec_entry > 0.0 else mid
            new_snapshot[_snapshot_key(sym, "overlay")] = {
                "symbol": sym,
                "qty": _num(new_signed / mid, 0.0),
                "entry_price": _num(entry_px, 0.0),
                "kind": "overlay",
                "signed_usd": _num(new_signed, 0.0),
            }
        elif old_signed != 0.0 and old_ov_entry > 0.0:
            realized_overlay += old_signed * (mid / old_ov_entry - 1.0)

        # ---- carry leg (spot long + perp short: directional PnL nets out, basis remains)
        c_pos = carry_map.get(sym)
        c_qty = _num(c_pos.get("qty"), 0.0) if isinstance(c_pos, dict) else 0.0
        if c_qty == 0.0 and isinstance(c_pos, dict):
            c_notional = _num(c_pos.get("notional"), 0.0)
            if c_notional != 0.0:
                c_qty = c_notional / mid
        old_c = prev.get(_snapshot_key(sym, "carry")) or {}
        old_qty = _num(old_c.get("qty"), 0.0)
        old_entry = _num(old_c.get("entry_price"), 0.0)
        old_spot_entry = _num(old_c.get("spot_entry"), old_entry)
        old_perp_entry = _num(old_c.get("perp_entry"), old_entry)
        old_spot_qty = abs(_num(old_c.get("spot_qty"), abs(old_qty)))
        old_perp_qty = abs(_num(old_c.get("perp_qty"), abs(old_qty)))
        if charge_fees:
            fees += abs(c_qty - old_qty) * mid * CARRY_FEE_RATE

        if abs(c_qty) > 1e-12:
            spot_px = _mid_from_map(spot_prices, sym, mid)
            perp_px = _mark_from_map(perp_prices, sym, mid)
            new_spot_qty = abs(_num(c_qty, 0.0))
            new_perp_qty = abs(_num(c_qty, 0.0))
            same_sign = old_qty != 0.0 and (old_qty > 0.0) == (c_qty > 0.0)
            if same_sign and old_spot_entry > 0.0 and old_perp_entry > 0.0:
                spot_entry = _vwap_entry(old_spot_qty, old_spot_entry, new_spot_qty, spot_px)
                perp_entry = _vwap_entry(old_perp_qty, old_perp_entry, new_perp_qty, perp_px)
            else:
                exec_spot_entry = 0.0
                exec_perp_entry = 0.0
                if isinstance(c_pos, dict):
                    exec_spot_entry = _num(c_pos.get("entry_spot_price"), 0.0)
                    exec_perp_entry = _num(
                        c_pos.get("entry_perp_price", c_pos.get("entry_price")), 0.0
                    )
                spot_entry = exec_spot_entry if exec_spot_entry > 0.0 else spot_px
                perp_entry = exec_perp_entry if exec_perp_entry > 0.0 else perp_px
            new_snapshot[_snapshot_key(sym, "carry")] = {
                "symbol": sym,
                "qty": _num(c_qty, 0.0),
                # ``entry_price`` mirrors the Perp entry (the leg that carries leverage) and
                # stays for backwards compatibility; the split entries drive basis PnL.
                "entry_price": _num(perp_entry, 0.0),
                "kind": "carry",
                "signed_usd": _num(c_qty * mid, 0.0),
                "spot_qty": _num(new_spot_qty, 0.0),
                "perp_qty": _num(-new_perp_qty, 0.0),
                "spot_entry": _num(spot_entry, 0.0),
                "perp_entry": _num(perp_entry, 0.0),
            }

    # ---- Sprint 26: trend legs are owned by apply_trend_sleeve (not by the executor), so
    # they must survive the executor-driven rebuild of the snapshot.
    for key, pos in prev.items():
        if isinstance(pos, dict) and str(pos.get("kind", "carry")) in ("trend", "reversal"):
            new_snapshot[str(key)] = dict(pos)

    ledger["positions_snapshot"] = new_snapshot
    if charge_fees:
        ledger["fees_paid_usd"] = _num(ledger.get("fees_paid_usd"), 0.0) + fees
        ledger["realized_pnl_usd"] = _num(ledger.get("realized_pnl_usd"), 0.0) - fees
    ledger["realized_pnl_usd"] = _num(ledger.get("realized_pnl_usd"), 0.0) + realized_overlay
    ledger["overlay_pnl_usd"] = _num(ledger.get("overlay_pnl_usd"), 0.0) + realized_overlay
    return _num(fees, 0.0), 0.0, _num(realized_overlay, 0.0)


def _unrealized_from_snapshot(
    positions: dict[str, dict[str, Any]],
    sym_map: dict[str, Any],
    last_prices: dict[str, float],
    snapshot: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Mark-to-market PnL split into (carry, overlay, trend).

    Carry marks the Spot leg against the Spot mid and the Perp leg against the Perp mark,
    so the residual is the real basis (0.0 only when Spot and Perp move identically).
    """
    spot_prices, perp_prices = dual_price_maps(snapshot)
    unreal_carry = 0.0
    unreal_overlay = 0.0
    unreal_trend = 0.0
    if not isinstance(positions, dict):
        return 0.0, 0.0, 0.0
    for key, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        sym = _pos_symbol(str(key), pos)
        kind = str(pos.get("kind", "carry"))
        qty = _num(pos.get("qty"), 0.0)
        entry = _num(pos.get("entry_price"), 0.0)
        signed_usd = _num(pos.get("signed_usd"), qty * entry)
        mid = _extract_mid_price(sym, sym_map, last_prices, entry)
        if mid <= 0.0:
            continue
        if kind in ("overlay", "trend"):
            if entry <= 0.0:
                continue
            marked = signed_usd * (mid / entry - 1.0)
            if kind == "trend":
                unreal_trend += marked
            else:
                unreal_overlay += marked
        elif kind == "carry":
            basis, _spot_u, _perp_u = carry_basis_pnl(pos, spot_prices, perp_prices, mid)
            unreal_carry += basis
    return unreal_carry, unreal_overlay, unreal_trend


def _mark_to_market(
    ledger: dict[str, Any],
    profile: StrategyProfileConfig,
    sym_map: dict[str, Any],
    last_prices: dict[str, float],
    snapshot: dict[str, Any] | None = None,
) -> None:
    spot_prices, perp_prices = dual_price_maps(snapshot)
    positions = ledger.get("positions_snapshot", {}) or {}
    unreal_carry, unreal_overlay, unreal_trend = _unrealized_from_snapshot(
        positions, sym_map, last_prices, snapshot
    )
    basis_total = 0.0
    spot_unreal_total = 0.0
    perp_unreal_total = 0.0
    if isinstance(positions, dict):
        for key, pos in positions.items():
            if not isinstance(pos, dict) or str(pos.get("kind", "carry")) != "carry":
                continue
            sym = _pos_symbol(str(key), pos)
            fallback = _num(pos.get("entry_price"), 0.0)
            basis, spot_u, perp_u = carry_basis_pnl(
                pos, spot_prices, perp_prices, fallback
            )
            basis_total += basis
            spot_unreal_total += spot_u
            perp_unreal_total += perp_u
    unreal_reversal = 0.0
    if isinstance(positions, dict):
        for key, pos in positions.items():
            if not isinstance(pos, dict) or str(pos.get("kind", "carry")) != "reversal":
                continue
            r_entry = _num(pos.get("entry_price"), 0.0)
            r_mid = _extract_mid_price(_pos_symbol(str(key), pos), sym_map, last_prices, r_entry)
            if r_entry > 0.0 and r_mid > 0.0:
                unreal_reversal += _num(pos.get("signed_usd"), 0.0) * (r_mid / r_entry - 1.0)

    ledger["basis_pnl_usd"] = basis_total
    ledger["spot_unrealized_pnl_usd"] = spot_unreal_total
    ledger["perp_unrealized_pnl_usd"] = perp_unreal_total
    pure_eq = (
        _num(profile.capital_usd, 1000.0)
        + _num(ledger.get("realized_pnl_usd"), 0.0)
        + unreal_carry
        + unreal_overlay
        + unreal_trend
        + unreal_reversal
    )
    ledger["pure_trading_equity_usd"] = pure_eq
    ledger["yield_stacked_equity_usd"] = pure_eq + _num(
        ledger.get("cumulative_passive_yield_usd"), 0.0
    )
    peak = max(_num(ledger.get("peak_equity_usd"), profile.capital_usd), pure_eq)
    ledger["peak_equity_usd"] = peak
    dd_pct = ((peak - pure_eq) / peak * 100.0) if peak > 0.0 else 0.0
    ledger["max_dd_pct"] = max(_num(ledger.get("max_dd_pct"), 0.0), max(0.0, dd_pct))


def apply_shadow_fills(
    ledger: dict[str, Any],
    executor_state: Any,
    snapshot: dict[str, Any] | None,
    profile: StrategyProfileConfig,
) -> float:
    """SHADOW_LIVE fill simulation: only the traded delta pays fees, then mark to market."""
    sym_map = snapshot.get("symbols", {}) if isinstance(snapshot, dict) else {}
    last_prices = ledger.setdefault("last_prices", {})
    fees, _rc, _ro = refresh_position_ledger(
        ledger, executor_state, snapshot, charge_fees=True
    )
    _mark_to_market(ledger, profile, sym_map, last_prices, snapshot)
    _sanitize_ledger(ledger, default_capital=profile.capital_usd)
    return _num(fees, 0.0)


def _update_exposure_metrics(
    ledger: dict[str, Any], profile: StrategyProfileConfig
) -> None:
    pos_map = ledger.get("positions_snapshot", {})
    carry_slots = 0
    carry_perp_notional = 0.0
    overlay_long_usd = 0.0
    overlay_short_usd = 0.0
    trend_gross = 0.0
    reversal_gross = 0.0

    if isinstance(pos_map, dict):
        for pos in pos_map.values():
            if not isinstance(pos, dict):
                continue
            kind = str(pos.get("kind", "carry"))
            qty = _num(pos.get("qty"), 0.0)
            entry = _num(pos.get("entry_price"), 0.0)
            signed_usd = _num(pos.get("signed_usd"), qty * entry)
            if kind == "carry":
                notional = abs(signed_usd if signed_usd != 0.0 else qty * entry)
                if abs(qty) > 1e-12 or notional > 1e-9:
                    carry_slots += 1
                    carry_perp_notional += notional
            elif kind == "overlay":
                if signed_usd > 0.0:
                    overlay_long_usd += signed_usd
                elif signed_usd < 0.0:
                    overlay_short_usd += abs(signed_usd)
            elif kind == "trend":
                trend_gross += abs(signed_usd)
            elif kind == "reversal":
                reversal_gross += abs(signed_usd)

    overlay_gross = overlay_long_usd + overlay_short_usd
    fut_wallet = _num(profile.futures_wallet_usd, 0.0)
    fut_lev = (
        (carry_perp_notional + overlay_gross + trend_gross + reversal_gross) / fut_wallet
        if fut_wallet > 0.0
        else 0.0
    )
    ledger["carry_slots"] = int(carry_slots)
    ledger["overlay_long_usd"] = _num(overlay_long_usd, 0.0)
    ledger["overlay_short_usd"] = _num(overlay_short_usd, 0.0)
    ledger["trend_gross_usd"] = _num(trend_gross, 0.0)
    ledger["reversal_gross_usd"] = _num(reversal_gross, 0.0)
    ledger["futures_leverage"] = _num(fut_lev, 0.0)


def _slice_universe(
    universe: dict[str, Any] | None, bars: int
) -> dict[str, Any] | None:
    """Return a view of ``universe`` limited to the last ``bars`` rows per symbol."""
    if not isinstance(universe, dict):
        return universe
    out: dict[str, Any] = {}
    for sym, df in universe.items():
        if df is not None and hasattr(df, "iloc") and bars > 0 and len(df) > bars:
            out[sym] = df.iloc[-bars:]
        else:
            out[sym] = df
    return out


def compute_funding_ewma_bps(universe: dict[str, Any] | None) -> dict[str, float]:
    """Smoothed funding (bps per 8h) per symbol, same recursion as the backtest.

    The backtest (``run_1k_profile_tournament_backtest``) updates
    ``f = (1 - a) * f + a * funding_bps`` on every funding event (00/08/16 UTC) with
    ``a = 2 / (9 + 1)`` and ``funding_bps = funding_rate * 10_000``. Missing data -> 0.0.
    """
    out: dict[str, float] = {}
    if not isinstance(universe, dict):
        return out
    alpha = FUNDING_EWMA_ALPHA
    for sym, df in universe.items():
        f_ewma = 0.0
        try:
            col = next(
                (c for c in ("funding_rate_event", "funding_event_binance") if c in df.columns),
                None,
            )
            if col is not None and len(df) > 0:
                idx = pd.DatetimeIndex(df.index)
                is_event = ((idx.hour == 0) | (idx.hour == 8) | (idx.hour == 16)) & (idx.minute == 0)
                vals = df[col].astype("float64").to_numpy()[is_event]
                for v in vals:
                    f_now = float(v) * 10000.0 if math.isfinite(float(v)) else 0.0
                    f_ewma = (1.0 - alpha) * f_ewma + alpha * f_now
        except Exception:  # noqa: BLE001 - a bad funding column must not kill the cycle
            f_ewma = 0.0
        out[str(sym)] = float(f_ewma) if math.isfinite(f_ewma) else 0.0
    return out


def compute_data_staleness(
    universe: dict[str, Any] | None, now_utc: datetime
) -> tuple[bool, float, str]:
    """(is_stale, age_hours_of_oldest_symbol, last_candle_iso) for an aligned universe."""
    last_ts: list[pd.Timestamp] = []
    if isinstance(universe, dict):
        for df in universe.values():
            if df is not None and getattr(df, "empty", True) is False:
                last_ts.append(pd.Timestamp(df.index[-1]))
    if not last_ts:
        return True, float("inf"), ""
    oldest = min(last_ts)
    if oldest.tzinfo is None:
        oldest = oldest.tz_localize("UTC")
    age_h = (pd.Timestamp(now_utc) - oldest).total_seconds() / 3600.0
    return bool(age_h > STALE_DATA_MAX_AGE_HOURS), float(age_h), oldest.isoformat()


def _build_price_frame(aligned_universe: dict[str, Any]) -> pd.DataFrame:
    price_frame = pd.DataFrame()
    for sym in sorted(aligned_universe):
        df = aligned_universe[sym]
        if df is None or getattr(df, "empty", True):
            continue
        col = "perp_close" if "perp_close" in df.columns else ("close" if "close" in df.columns else None)
        if col is None:
            continue
        price_frame[str(sym)] = df[col].astype("float64")
    if price_frame.empty:
        return price_frame
    return price_frame.sort_index().ffill()


def _sleeve_notional(pos_map: Any, kinds: tuple[str, ...]) -> float:
    total = 0.0
    if isinstance(pos_map, dict):
        for pos in pos_map.values():
            if isinstance(pos, dict) and str(pos.get("kind", "carry")) in kinds:
                total += abs(_num(pos.get("signed_usd"), 0.0))
    return total


def _place_directional_order(
    client: Any,
    executor: Any,
    sym: str,
    side: str,
    qty: float,
    price: float,
) -> tuple[float, float, str]:
    """Send one Perp MARKET order for a directional sleeve. Returns (filled, avg_px, error).

    ``reduce_only`` is deliberately never set: Trend/Reversal share the one-way Futures
    position of a symbol with Carry (short) and Overlay, so a sleeve-level close is only a
    *partial* reduction of the netted position and would be rejected (-2022) or clipped.
    """
    qty_q = float(qty)
    quantize = getattr(executor, "quantize_leg", None)
    if callable(quantize):
        qty_q, _ = quantize(sym, abs(qty_q) * price, price)
    if qty_q <= 0.0:
        return 0.0, price, "BELOW_MIN_NOTIONAL"
    try:
        resp = client.place_futures_market_order(sym, side, float(qty_q), reduce_only=False)
    except Exception as exc:  # noqa: BLE001 - a rejected order must not kill the cycle
        logger.error("directional order failed %s %s %s: %s", sym, side, qty_q, exc)
        return 0.0, price, f"{type(exc).__name__}:{str(exc)[:80]}"
    resp = resp if isinstance(resp, dict) else {}
    executed = _num(resp.get("executedQty"), 0.0)
    status = str(resp.get("status", "")).upper()
    if executed <= 0.0 and (status == "FILLED" or bool(resp.get("dry_run"))):
        executed = float(qty_q)
    avg = _num(resp.get("avgPrice"), 0.0)
    if avg <= 0.0:
        avg = price
    return float(executed), float(avg), ("" if executed > 0.0 else f"NOT_FILLED:{status}")


def _append_ledger_note(ledger: dict[str, Any], note_part: str, max_len: int = 200) -> None:
    """Append a note token without duplicating existing tokens across or within cycles."""
    part = str(note_part or "").strip()
    if not part:
        return
    existing = str(ledger.get("note") or "").strip()
    tokens = [t for t in existing.split(" ") if t] if existing else []
    for tok in part.split(" "):
        tok = tok.strip()
        if tok and tok not in tokens:
            tokens.append(tok)
    ledger["note"] = " ".join(tokens)[:max_len]


def _apply_directional_deltas(
    ledger: dict[str, Any],
    kind: str,
    targets: dict[str, float],
    prices_now: dict[str, float],
    *,
    charge_fees: bool,
    signed_client: Any = None,
    execute_real: bool = False,
    executor: Any = None,
) -> dict[str, Any]:
    """Move the ``"SYM|<kind>"`` ledger legs to the signed ``targets`` (qty, long > 0).

    Shadow mode only books the estimated fee. With ``execute_real`` every delta is first sent
    to Binance Futures through ``signed_client`` and the ledger moves by the *filled* quantity
    at the *fill* price, so a rejected order leaves the ledger identical to the exchange.
    Sign flips are executed as close + open. Long-only trend reproduces the Sprint 26 booking.
    """
    pos_map = ledger.get("positions_snapshot", {}) or {}
    new_positions = dict(pos_map) if isinstance(pos_map, dict) else {}
    existing = {
        _pos_symbol(str(k), v)
        for k, v in new_positions.items()
        if isinstance(v, dict) and str(v.get("kind", "carry")) == kind
    }
    book = {sym: float(targets.get(sym, 0.0)) for sym in set(targets) | existing}

    def _old_qty(sym: str) -> float:
        p = new_positions.get(_snapshot_key(sym, kind))
        return _num(p.get("qty"), 0.0) if isinstance(p, dict) else 0.0

    # Reductions first (frees margin and keeps dollar-neutral sleeves covered).
    order = sorted(book, key=lambda s: (0 if abs(book[s]) < abs(_old_qty(s)) else 1, s))
    real = bool(execute_real and signed_client is not None)

    fees = turnover = realized_closed = 0.0
    n_orders = 0
    errors: list[str] = []
    for sym in order:
        px = float(prices_now.get(sym, 0.0))
        if px <= 0.0:
            continue
        key = _snapshot_key(sym, kind)
        old_pos = new_positions.get(key) if isinstance(new_positions.get(key), dict) else {}
        cur = _num(old_pos.get("qty"), 0.0)
        cur_entry = _num(old_pos.get("entry_price"), 0.0)
        new_qty = book[sym]
        if abs(new_qty - cur) <= 1e-12:
            continue
        steps = [(cur, 0.0), (0.0, new_qty)] if cur * new_qty < 0.0 else [(cur, new_qty)]
        for a, b in steps:
            delta = b - a
            if abs(delta) <= 1e-12:
                continue
            fill_px = px
            if real:
                filled, fill_px, err = _place_directional_order(
                    signed_client, executor, sym, "BUY" if delta > 0.0 else "SELL", abs(delta), px
                )
                if filled <= 0.0:
                    errors.append(f"{sym}:{err}")
                    break
                b = a + (1.0 if delta > 0.0 else -1.0) * min(filled, abs(delta))
            notional = abs(b - a) * fill_px
            fee = notional * TREND_FEE_RATE if charge_fees else 0.0
            fees += fee
            turnover += notional
            n_orders += 1
            same_side = a != 0.0 and (a > 0.0) == (b > 0.0)
            if abs(b) < abs(a) - 1e-12 or abs(b) <= 1e-12:
                if a != 0.0 and cur_entry > 0.0:
                    closed_mag = abs(a) - abs(b)
                    realized_closed += closed_mag * (fill_px - cur_entry) * (1.0 if a > 0.0 else -1.0)
            if abs(b) > abs(a) + 1e-12 and same_side and cur_entry > 0.0:
                cur_entry = _vwap_entry(abs(a), cur_entry, abs(b), fill_px)
            elif abs(b) > abs(a) + 1e-12:
                cur_entry = fill_px
            cur = b
        if abs(cur) <= 1e-12:
            new_positions.pop(key, None)
        elif cur != _num(old_pos.get("qty"), 0.0):
            new_positions[key] = {
                "symbol": sym,
                "qty": float(cur),
                "entry_price": float(cur_entry),
                "kind": kind,
                # Cost basis (qty * entry) so qty*entry*(mid/entry - 1) == qty*(mid - entry).
                "signed_usd": float(cur * cur_entry),
            }

    ledger["positions_snapshot"] = new_positions
    realized = _num(ledger.get("realized_pnl_usd"), 0.0) + realized_closed
    if charge_fees:
        realized -= fees
        ledger["fees_paid_usd"] = _num(ledger.get("fees_paid_usd"), 0.0) + fees
    ledger["realized_pnl_usd"] = realized
    if errors:
        _append_ledger_note(ledger, f"{kind}_order_err:" + ",".join(errors)[:80])
    return {
        "fees": float(fees),
        "turnover": float(turnover),
        "n_orders": int(n_orders),
        "realized_closed": float(realized_closed),
        "errors": errors,
    }


def apply_trend_sleeve(
    ledger: dict[str, Any],
    profile: StrategyProfileConfig,
    aligned_universe: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    *,
    now_utc: datetime | None = None,
    charge_fees: bool = True,
    signed_client: Any = None,
    execute_real: bool = False,
    executor: Any = None,
    funding_bps: dict[str, float] | None = None,
    hold_only: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark and rebalance the directional Trend sleeve on isolated ``"SYM|trend"`` keys.

    Every Trend leg lives under ``"SYM|trend"`` inside ``positions_snapshot`` so a symbol can
    be held by carry, overlay and trend simultaneously without any collision. Sizing is
    inverse-volatility with a hard gross cap, then multiplied by the 2-tier drawdown circuit
    breaker ``risk_scale`` (0.5 at -6%, 0.0 at -10% until the cooldown expires).

    Sprint 27-28 parameters (``allow_short_trend``, ``trend_z_short_entry``,
    ``require_macro_regime_filter``, ``short_gross_scale``) and the smoothed ``funding_bps``
    are forwarded exactly like the backtest. With ``dynamic_carry_to_alpha_margin`` the unused
    Carry budget is routed to Trend: ``min(1, max_gross + max(0, eta - carry/cap) * 0.5)``.
    ``hold_only`` freezes the book (used on stale data in real mode).
    """
    disabled = {
        "state": "DISABLED",
        "risk_scale": 1.0,
        "n_orders": 0,
        "turnover_notional": 0.0,
        "fees_usd": 0.0,
        "targets": {},
        "positions": [],
    }
    if not bool(getattr(profile, "enable_trend_sleeve", False)):
        ledger["circuit_breaker_state"] = "DISABLED"
        ledger["trend_gross_usd"] = 0.0
        return disabled
    if not isinstance(aligned_universe, dict) or not aligned_universe:
        return disabled

    now = now_utc or datetime.now(timezone.utc)
    capital = _num(profile.capital_usd, 1000.0)

    # ---- 1) 2-tier drawdown circuit breaker on live mark-to-market equity
    break_cfg = DrawdownCircuitBreakerConfig(
        soft_dd_threshold_pct=float(profile.soft_dd_threshold_pct),
        hard_dd_threshold_pct=float(profile.hard_dd_threshold_pct),
    )
    cur_eq = _num(ledger.get("pure_trading_equity_usd"), capital)
    peak_eq = max(_num(ledger.get("peak_equity_usd"), capital), cur_eq)
    ledger["peak_equity_usd"] = peak_eq
    lock_bars = ledger.get("trend_hard_lock_bars")
    lock_bars = None if lock_bars is None else int(_num(lock_bars, 0.0))
    pending_bars = None if lock_bars is None else lock_bars + 1
    cb = evaluate_drawdown_circuit_breaker(cur_eq, peak_eq, pending_bars, break_cfg)
    if cb["state"] == "HARD_STOP_LOCKED":
        if lock_bars is None:
            ledger["circuit_breaker_triggers"] = (
                int(_num(ledger.get("circuit_breaker_triggers"), 0.0)) + 1
            )
            ledger["trend_hard_lock_bars"] = 0
        else:
            ledger["trend_hard_lock_bars"] = int(pending_bars or 0)
    else:
        ledger["trend_hard_lock_bars"] = None
    ledger["circuit_breaker_state"] = str(cb["state"])
    ledger["circuit_breaker_risk_scale"] = _num(cb["risk_scale"], 1.0)

    # ---- 2) Strictly causal signal: only bars up to the last one in the frame are read.
    price_frame = _build_price_frame(aligned_universe)
    if price_frame.empty:
        return disabled

    pos_map = ledger.get("positions_snapshot", {}) or {}
    # Dynamic Carry -> Alpha margin (copied from run_1k_profile_tournament_backtest.py L575).
    effective_max_gross = float(profile.trend_max_gross_pct)
    if bool(getattr(profile, "dynamic_carry_to_alpha_margin", False)):
        carry_used_pct = _sleeve_notional(pos_map, ("carry",)) / capital if capital > 0.0 else 0.0
        effective_max_gross = min(
            1.0,
            effective_max_gross
            + max(0.0, float(profile.carry_eta) - carry_used_pct) * 0.5,
        )

    allow_short = bool(getattr(profile, "allow_short_trend", False))
    raw_short_scale = getattr(profile, "short_gross_scale", None)
    trend_cfg = TrendSleeveConfig(
        capital_usd=capital,
        target_vol_ann=float(profile.trend_target_vol_ann),
        max_gross_pct=float(effective_max_gross),
        rebalance_every_hours=int(profile.trend_rebalance_hours),
        z_entry=float(profile.trend_z_entry),
        allow_short_trend=allow_short,
        z_short_entry=float(getattr(profile, "trend_z_short_entry", 1.10)),
        require_macro_regime_filter=bool(getattr(profile, "require_macro_regime_filter", False)),
        short_gross_scale=(
            float(raw_short_scale) if allow_short and raw_short_scale is not None else 0.65
        ),
    )
    state = TrendSleeveState.from_snapshot(ledger.get("trend_sleeve_state"))
    if funding_bps is None:
        funding_bps = compute_funding_ewma_bps(aligned_universe)

    last_reb = _parse_iso_utc(ledger.get("last_trend_rebalance_ts"))
    rebalance_due = last_reb is None or (
        (now - last_reb).total_seconds()
        >= 3600.0 * int(trend_cfg.rebalance_every_hours)
    )
    weights = compute_triple_confirmed_trend_weights(
        price_frame,
        bar_index=len(price_frame) - 1,
        config=trend_cfg,
        state=state,
        circuit_breaker=cb,
        allow_new_entries=bool(rebalance_due),
        funding_bps=funding_bps,
    )
    if diagnostics is not None:
        diagnostics["trend"] = {
            "bars": int(len(price_frame)),
            "min_history_hours": int(trend_cfg.min_history_hours),
            "effective_max_gross_pct": float(effective_max_gross),
            "rebalance_due": bool(rebalance_due),
            "weights": {str(k): float(v) for k, v in weights.items()},
            "allow_short_trend": allow_short,
        }

    # ---- 3) Execution price: snapshot Perp mark when available, else last close.
    spot_prices, perp_prices = dual_price_maps(snapshot)
    prices_now: dict[str, float] = {}
    for sym in price_frame.columns:
        px = _mark_from_map(perp_prices, sym, 0.0)
        if px <= 0.0:
            px = float(price_frame[sym].iloc[-1])
        if math.isfinite(px) and px > 0.0:
            prices_now[sym] = px

    # ---- 4) Hard risk cap: trend notional may never exceed the free futures capacity.
    has_trend_positions = any(
        isinstance(p, dict)
        and str(p.get("kind", "carry")) == "trend"
        and abs(_num(p.get("qty"), 0.0)) > 0.0
        for p in (pos_map.values() if isinstance(pos_map, dict) else [])
    )
    used_notional = _sleeve_notional(pos_map, ("carry", "overlay", "reversal"))
    fut_capacity = max(
        0.0,
        _num(profile.futures_wallet_usd, 0.0) * _num(profile.max_futures_leverage, 1.0)
        - used_notional,
    )
    desired_gross = float(sum(abs(w) for w in weights.values())) * capital
    if desired_gross > 0.0 and desired_gross > fut_capacity:
        shrink = fut_capacity / desired_gross
        weights = {s: float(w) * shrink for s, w in weights.items()}

    targets = (
        trend_weights_to_target_qty(
            weights, prices_now, capital, DEMO_LOT_STEPS, min_notional_usd=10.0
        )
        if weights
        else {}
    )
    # The backtest only trades on rebalance bars, and the sleeve must not bleed 5 bps of
    # taker fee every hour while inverse-vol sizing wobbles. So deltas are applied only
    # when the rebalance clock fires, or immediately when the breaker / trailing stop
    # demands a flat book.
    risk_scale_now = _num(cb.get("risk_scale"), 1.0)
    live_weights = any(abs(float(w)) > 0.0 for w in weights.values())
    # Emergency flatten must never wait for the 24h clock: a hard-stopped breaker
    # (risk_scale == 0) or an empty target set with positions still open both qualify.
    force_flat = has_trend_positions and (risk_scale_now <= 0.0 or not live_weights)
    if hold_only or (not rebalance_due and not force_flat):
        ledger["trend_sleeve_state"] = state.snapshot()
        ledger["trend_active_symbols"] = sorted(
            s for s, is_on in (state.active or {}).items() if bool(is_on)
        )
        held = sorted(
            _pos_symbol(str(k), v)
            for k, v in (pos_map.items() if isinstance(pos_map, dict) else [])
            if isinstance(v, dict) and str(v.get("kind", "carry")) == "trend"
        )
        return {
            "state": str(cb["state"]),
            "risk_scale": _num(cb["risk_scale"], 1.0),
            "n_orders": 0,
            "turnover_notional": 0.0,
            "fees_usd": 0.0,
            "targets": {},
            "positions": held,
            "held": True,
        }

    # ---- 5) Apply deltas on "SYM|trend" keys (real orders first when execute_real).
    res = _apply_directional_deltas(
        ledger,
        "trend",
        targets,
        prices_now,
        charge_fees=charge_fees,
        signed_client=signed_client,
        execute_real=execute_real,
        executor=executor,
    )
    # Attribution: trend_pnl_usd must carry its own realised closes and its own fees,
    # otherwise the scoreboard column silently under-reports the sleeve.
    ledger["trend_pnl_usd"] = (
        _num(ledger.get("trend_pnl_usd"), 0.0)
        + res["realized_closed"]
        - (res["fees"] if charge_fees else 0.0)
    )
    ledger["trend_sleeve_state"] = state.snapshot()
    ledger["trend_active_symbols"] = sorted(
        s for s, is_on in (state.active or {}).items() if bool(is_on)
    )
    if rebalance_due:
        ledger["last_trend_rebalance_ts"] = now.isoformat()
    new_positions = ledger.get("positions_snapshot", {}) or {}
    held = sorted(
        _pos_symbol(str(k), v)
        for k, v in new_positions.items()
        if isinstance(v, dict) and str(v.get("kind", "carry")) == "trend"
    )
    return {
        "state": str(cb["state"]),
        "risk_scale": _num(cb["risk_scale"], 1.0),
        "n_orders": int(res["n_orders"]),
        "turnover_notional": float(res["turnover"]),
        "fees_usd": float(res["fees"]),
        "targets": {str(k): float(v) for k, v in targets.items()},
        "positions": held,
    }


def apply_reversal_sleeve(
    ledger: dict[str, Any],
    profile: StrategyProfileConfig,
    aligned_universe: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    *,
    now_utc: datetime | None = None,
    charge_fees: bool = True,
    signed_client: Any = None,
    execute_real: bool = False,
    executor: Any = None,
    funding_bps: dict[str, float] | None = None,
    hold_only: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dollar-neutral funding-conditioned reversal sleeve on isolated ``"SYM|reversal"`` keys.

    Mirrors the backtest (``run_1k_profile_tournament_backtest.py`` L643-L714): signal on the
    last ``REVERSAL_WINDOW_BARS`` Perp closes with smoothed funding, capped by the free
    Futures capacity, skipped when the rounded legs are imbalanced by more than
    ``REVERSAL_MAX_IMBALANCE_USD`` (which also flattens an open book), rebalanced on the
    same clock as Trend.
    """
    disabled = {
        "state": "DISABLED",
        "n_orders": 0,
        "turnover_notional": 0.0,
        "fees_usd": 0.0,
        "targets": {},
        "positions": [],
    }
    if not bool(getattr(profile, "enable_reversal_sleeve", False)):
        ledger["reversal_gross_usd"] = 0.0
        return disabled
    if not isinstance(aligned_universe, dict) or not aligned_universe:
        return disabled

    now = now_utc or datetime.now(timezone.utc)
    capital = _num(profile.capital_usd, 1000.0)
    price_frame = _build_price_frame(aligned_universe)
    if price_frame.empty:
        return disabled

    last_reb = _parse_iso_utc(ledger.get("last_reversal_rebalance_ts"))
    rebalance_due = last_reb is None or (
        (now - last_reb).total_seconds()
        >= 3600.0 * max(1, int(getattr(profile, "trend_rebalance_hours", 24) or 24))
    )
    pos_map = ledger.get("positions_snapshot", {}) or {}
    held_now = sorted(
        _pos_symbol(str(k), v)
        for k, v in (pos_map.items() if isinstance(pos_map, dict) else [])
        if isinstance(v, dict) and str(v.get("kind", "carry")) == "reversal"
    )
    if hold_only or not rebalance_due:
        return {**disabled, "state": "HELD", "positions": held_now, "held": True}

    if funding_bps is None:
        funding_bps = compute_funding_ewma_bps(aligned_universe)
    rev_cfg = ReversalSleeveConfig(
        capital_usd=capital, gross_pct=float(profile.reversal_gross_pct)
    )
    window = price_frame.iloc[-REVERSAL_WINDOW_BARS:]
    rev_w = compute_funding_conditioned_reversal_weights(
        window,
        bar_index=len(window) - 1,
        funding_bps=funding_bps,
        config=rev_cfg,
    )

    spot_prices, perp_prices = dual_price_maps(snapshot)
    prices_now: dict[str, float] = {}
    for sym in price_frame.columns:
        px = _mark_from_map(perp_prices, sym, 0.0)
        if px <= 0.0:
            px = float(price_frame[sym].iloc[-1])
        if math.isfinite(px) and px > 0.0:
            prices_now[sym] = px

    used_notional = _sleeve_notional(pos_map, ("carry", "overlay", "trend"))
    cap_r = max(
        0.0,
        _num(profile.futures_wallet_usd, 0.0) * _num(profile.max_futures_leverage, 1.0)
        - used_notional,
    )
    want_r = float(sum(abs(w) for w in rev_w.values())) * capital
    if want_r > 0.0 and want_r > cap_r:
        rev_w = {s: float(w) * (cap_r / want_r) for s, w in rev_w.items()}
    rev_q = reversal_weights_to_target_qty(
        rev_w, prices_now, capital, DEMO_LOT_STEPS, min_notional_usd=10.0
    ) if rev_w else {}
    imbalance = abs(sum(q * prices_now[s] for s, q in rev_q.items() if s in prices_now))
    skipped_imbalance = imbalance > REVERSAL_MAX_IMBALANCE_USD
    if skipped_imbalance:
        rev_q = {}
    if diagnostics is not None:
        diagnostics["reversal"] = {
            "bars": int(len(window)),
            "weights": {str(k): float(v) for k, v in rev_w.items()},
            "target_qty": {str(k): float(v) for k, v in rev_q.items()},
            "imbalance_usd": float(imbalance),
            "skipped_imbalance": bool(skipped_imbalance),
        }

    res = _apply_directional_deltas(
        ledger,
        "reversal",
        {s: float(q) for s, q in rev_q.items() if q != 0.0},
        prices_now,
        charge_fees=charge_fees,
        signed_client=signed_client,
        execute_real=execute_real,
        executor=executor,
    )
    ledger["reversal_pnl_usd"] = (
        _num(ledger.get("reversal_pnl_usd"), 0.0)
        + res["realized_closed"]
        - (res["fees"] if charge_fees else 0.0)
    )
    ledger["last_reversal_rebalance_ts"] = now.isoformat()
    new_positions = ledger.get("positions_snapshot", {}) or {}
    held = sorted(
        _pos_symbol(str(k), v)
        for k, v in new_positions.items()
        if isinstance(v, dict) and str(v.get("kind", "carry")) == "reversal"
    )
    return {
        "state": "ACTIVE" if held else "FLAT",
        "n_orders": int(res["n_orders"]),
        "turnover_notional": float(res["turnover"]),
        "fees_usd": float(res["fees"]),
        "targets": {str(k): float(v) for k, v in rev_q.items()},
        "positions": held,
    }


def run_profile_cycle(
    profile: StrategyProfileConfig,
    ledger: dict[str, Any],
    snapshot: dict[str, Any] | None,
    aligned_universe: dict[str, Any] | None,
    *,
    mode: str,
    execute_real: bool,
    executor: Any,
    reconcile_fn: Any = None,
    signed_client: Any = None,
    diagnostics: dict[str, Any] | None = None,
    real_directional_orders: bool = True,
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    ledger["profile"] = profile.name
    ledger["mode"] = "TESTNET_REAL" if execute_real else "SHADOW_LIVE"
    ledger["cycle"] = int(_num(ledger.get("cycle"), 0.0)) + 1
    ledger["note"] = ""

    accrue_cycle_accounting(
        ledger,
        profile,
        snapshot,
        now_utc,
        ledger.get("positions_snapshot", {}),
    )

    reconcile_failed = False
    try:
        # Carry/Overlay keep their historical 720h window; Trend/Reversal get the full history.
        executor.reconcile_once(
            aligned_universe=_slice_universe(aligned_universe, CARRY_LOOKBACK_BARS)
        )
        ledger["status"] = "OK"
        exec_diag = getattr(executor, "last_decision_diagnostics", None)
        exec_errs = (
            exec_diag.get("execution_errors") if isinstance(exec_diag, dict) else None
        ) or []
        if exec_errs:
            _append_ledger_note(
                ledger, "exec_err:" + ",".join(str(e) for e in exec_errs)[:80]
            )
    except Exception as exc:
        reconcile_failed = True
        ledger["status"] = f"ERROR:{type(exc).__name__}"
        ledger["note"] = str(exc)[:200]

    if not reconcile_failed:
        exec_state = getattr(executor, "state", None)
        if not execute_real:
            apply_shadow_fills(ledger, exec_state, snapshot, profile)
        elif exec_state is not None:
            sym_map = snapshot.get("symbols", {}) if isinstance(snapshot, dict) else {}
            last_prices = ledger.setdefault("last_prices", {})
            # Fees are estimated from the traded delta (the exchange commission feed is not
            # wired in); note it so nobody reads this as an exact testnet cash balance.
            fees, _rc, _ro = refresh_position_ledger(
                ledger, exec_state, snapshot, charge_fees=True
            )
            if fees > 0.0:
                _append_ledger_note(ledger, "fees_estimated")
            _mark_to_market(ledger, profile, sym_map, last_prices, snapshot)

    # ---- Sprint 26: trend sleeve (long Perp on isolated "SYM|trend" ledger keys)
    stale, age_h, last_candle = compute_data_staleness(aligned_universe, now_utc)
    if aligned_universe is not None and stale:
        _append_ledger_note(ledger, f"STALE_DATA:age={age_h:.1f}h")
    # Real orders need a clean state: no reconcile error and fresh candles, otherwise the
    # sleeves hold (the virtual ledger must never run ahead of the exchange).
    directional_real = bool(execute_real and real_directional_orders and signed_client is not None)
    hold_only = bool(directional_real and (reconcile_failed or stale))
    funding_map = compute_funding_ewma_bps(aligned_universe)
    trend_result = apply_trend_sleeve(
        ledger,
        profile,
        aligned_universe,
        snapshot,
        now_utc=now_utc,
        charge_fees=True,
        signed_client=signed_client,
        execute_real=directional_real,
        executor=executor,
        funding_bps=funding_map,
        hold_only=hold_only,
        diagnostics=diagnostics,
    )
    reversal_result = apply_reversal_sleeve(
        ledger,
        profile,
        aligned_universe,
        snapshot,
        now_utc=now_utc,
        charge_fees=True,
        signed_client=signed_client,
        execute_real=directional_real,
        executor=executor,
        funding_bps=funding_map,
        hold_only=hold_only,
        diagnostics=diagnostics,
    )
    if diagnostics is not None:
        diagnostics["trend_result"] = trend_result
        diagnostics["reversal_result"] = reversal_result
    if (
        trend_result.get("n_orders", 0) + reversal_result.get("n_orders", 0) > 0
    ) and not reconcile_failed:
        sym_map_t = snapshot.get("symbols", {}) if isinstance(snapshot, dict) else {}
        _mark_to_market(
            ledger, profile, sym_map_t, ledger.setdefault("last_prices", {}), snapshot
        )

    _update_exposure_metrics(ledger, profile)

    # ---- Sprint 25.1: real exchange-balance reconciliation (TESTNET_REAL only)
    if reconcile_fn is not None:
        try:
            recon = reconcile_fn(ledger)
        except Exception as exc:  # noqa: BLE001 - reconciliation must never kill a cycle
            recon = {
                "exchange_recon_equity_usd": None,
                "recon_drift_usd": None,
                "reconcile_status": "UNAVAILABLE",
                "note": f"recon_error:{type(exc).__name__}",
            }
        if isinstance(recon, dict):
            ledger["exchange_recon_equity_usd"] = recon.get("exchange_recon_equity_usd")
            ledger["recon_drift_usd"] = recon.get("recon_drift_usd")
            ledger["reconcile_status"] = str(recon.get("reconcile_status", "UNAVAILABLE"))
            recon_note = str(recon.get("note") or "")
            if recon_note:
                _append_ledger_note(ledger, recon_note)

    _sanitize_ledger(ledger, default_capital=profile.capital_usd)

    net_pnl = _num(ledger["pure_trading_equity_usd"] - profile.capital_usd, 0.0)
    record = {
        "timestamp": ledger.get("updated_at") or now_utc.isoformat(),
        "cycle": int(ledger["cycle"]),
        "profile": profile.name,
        "mode": str(ledger["mode"]),
        "pure_trading_equity_usd": _num(ledger["pure_trading_equity_usd"], profile.capital_usd),
        "yield_stacked_equity_usd": _num(ledger["yield_stacked_equity_usd"], profile.capital_usd),
        "net_pnl_usd": net_pnl,
        "realized_pnl_usd": _num(ledger["realized_pnl_usd"], 0.0),
        "funding_pnl_usd": _num(ledger["funding_pnl_usd"], 0.0),
        "overlay_pnl_usd": _num(ledger["overlay_pnl_usd"], 0.0),
        "trend_pnl_usd": _num(ledger.get("trend_pnl_usd"), 0.0),
        "trend_gross_usd": _num(ledger.get("trend_gross_usd"), 0.0),
        "trend_active_symbols": ledger.get("trend_active_symbols", []),
        "reversal_pnl_usd": _num(ledger.get("reversal_pnl_usd"), 0.0),
        "reversal_gross_usd": _num(ledger.get("reversal_gross_usd"), 0.0),
        "circuit_breaker_state": str(ledger.get("circuit_breaker_state", "DISABLED")),
        "circuit_breaker_risk_scale": _num(ledger.get("circuit_breaker_risk_scale"), 1.0),
        "circuit_breaker_triggers": int(_num(ledger.get("circuit_breaker_triggers"), 0.0)),
        "basis_pnl_usd": _num(ledger.get("basis_pnl_usd"), 0.0),
        "spot_unrealized_pnl_usd": _num(ledger.get("spot_unrealized_pnl_usd"), 0.0),
        "perp_unrealized_pnl_usd": _num(ledger.get("perp_unrealized_pnl_usd"), 0.0),
        "cumulative_passive_yield_usd": _num(ledger["cumulative_passive_yield_usd"], 0.0),
        "fees_paid_usd": _num(ledger["fees_paid_usd"], 0.0),
        "max_dd_pct": _num(ledger["max_dd_pct"], 0.0),
        "carry_slots": int(ledger["carry_slots"]),
        "overlay_long_usd": _num(ledger["overlay_long_usd"], 0.0),
        "overlay_short_usd": _num(ledger["overlay_short_usd"], 0.0),
        "futures_leverage": _num(ledger["futures_leverage"], 0.0),
        "exchange_recon_equity_usd": ledger.get("exchange_recon_equity_usd"),
        "recon_drift_usd": ledger.get("recon_drift_usd"),
        "reconcile_status": str(ledger.get("reconcile_status", "NOT_APPLICABLE")),
        "status": str(ledger["status"]),
        "note": str(ledger.get("note", "")),
    }
    return record


ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[1;36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"


def _supports_utf8(stream: Any = None) -> bool:
    target = stream if stream is not None else sys.stdout
    enc = getattr(target, "encoding", None) or ""
    if "utf" not in enc.lower():
        return False
    try:
        "╔═╦═╗║╠═╬═╣╚═╩═╝".encode(enc)
        return True
    except Exception:
        return False


def _colorize_cell(padded_cell: str, raw_cell: str, col_idx: int, use_color: bool) -> str:
    if not use_color:
        return padded_cell
    val = raw_cell.strip()
    if col_idx == 1:  # Mode
        if val == "TESTNET_REAL":
            return f"{ANSI_CYAN}{padded_cell}{ANSI_RESET}"
        if val == "SHADOW_LIVE":
            return f"{ANSI_YELLOW}{padded_cell}{ANSI_RESET}"
    elif col_idx in (4, 5, 6, 7, 8, 16):  # Signed PnL & Drift columns
        try:
            num_val = float(val)
            if num_val > 0:
                return f"{ANSI_GREEN}{padded_cell}{ANSI_RESET}"
            if num_val < 0:
                return f"{ANSI_RED}{padded_cell}{ANSI_RESET}"
        except ValueError:
            pass
    elif col_idx == 14:  # Trend CB
        if val == "NORMAL":
            return f"{ANSI_GREEN}{padded_cell}{ANSI_RESET}"
        if val in ("N/A (No-Trend)", "SOFT_THROTTLE"):
            return f"{ANSI_YELLOW}{padded_cell}{ANSI_RESET}"
        if "HARD_STOP" in val or "LOCK" in val:
            return f"{ANSI_RED}{padded_cell}{ANSI_RESET}"
    elif col_idx == 17:  # Recon
        if val == "SYNCED":
            return f"{ANSI_GREEN}{padded_cell}{ANSI_RESET}"
        if val in ("DRIFT_WARNING", "UNAVAILABLE"):
            return f"{ANSI_RED}{padded_cell}{ANSI_RESET}"
    elif col_idx == 18:  # Status
        if val == "OK":
            return f"{ANSI_GREEN}{padded_cell}{ANSI_RESET}"
        if "ERROR" in val:
            return f"{ANSI_RED}{padded_cell}{ANSI_RESET}"
    return padded_cell


def _format_live_positions_summary(
    live_ledger: Mapping[str, Any] | None,
    live_rec: Mapping[str, Any] | None,
) -> str:
    pos_map = (
        live_ledger.get("positions_snapshot", {})
        if isinstance(live_ledger, Mapping)
        else (live_rec.get("positions_snapshot", {}) if isinstance(live_rec, Mapping) else {})
    )
    carry_parts: list[str] = []
    overlay_parts: list[str] = []
    trend_parts: list[str] = []
    if isinstance(pos_map, Mapping):
        for key in sorted(pos_map.keys()):
            pos = pos_map[key]
            if not isinstance(pos, Mapping):
                continue
            sym = str(pos.get("symbol") or str(key).split("|")[0]).replace("/USDT:USDT", "")
            kind = str(pos.get("kind", "carry"))
            qty = _num(pos.get("qty"), 0.0)
            entry = _num(pos.get("entry_price"), 0.0)
            signed = _num(pos.get("signed_usd"), qty * entry)
            if abs(qty) <= 1e-12 and abs(signed) <= 1e-9:
                continue
            if kind == "carry":
                carry_parts.append(f"{sym}(${abs(signed):.0f})")
            elif kind == "overlay":
                overlay_parts.append(f"{sym}({signed:+.0f}$)")
            elif kind == "trend":
                side_ch = "L" if (qty > 0 or signed > 0) else "S"
                trend_parts.append(f"{sym}[{side_ch}](${abs(signed):.0f})")

    c_str = ", ".join(carry_parts) if carry_parts else "FLAT"
    o_str = ", ".join(overlay_parts) if overlay_parts else "FLAT"
    t_str = ", ".join(trend_parts) if trend_parts else "FLAT"
    return f"Carry: {c_str} | Overlay: {o_str} | Trend: {t_str}"


def format_scoreboard_table(
    records: list[dict[str, Any]],
    *,
    use_unicode: bool | None = None,
    use_color: bool | None = None,
    ledgers: Mapping[str, Any] | None = None,
    live_profile: str = "regime_omni_1k",
    network_status: str = "ONLINE",
    tg_status: str = "TG: SKIPPED_NO_CHANGE",
    next_cycle_sec: float | None = None,
) -> str:
    """Format the tournament scoreboard table with Unicode borders, ANSI colors, and summary footer."""
    if use_unicode is None:
        use_unicode = _supports_utf8()
    if use_color is None:
        use_color = bool(use_unicode) and not bool(os.environ.get("NO_COLOR"))

    headers = [
        "Profile",
        "Mode",
        "Pure Equity ($)",
        "Yield Equity ($)",
        "Net PnL ($)",
        "Funding ($)",
        "Overlay PnL ($)",
        "Trend PnL ($)",
        "Basis PnL ($)",
        "Fees ($)",
        "MaxDD (%)",
        "Carry Slots",
        "Overlay L/S ($)",
        "Fut Lev (x)",
        "Trend CB",
        "Recon Eq ($)",
        "Drift ($)",
        "Recon",
        "Status",
    ]
    rows: list[list[str]] = []
    for r in records:
        pure_eq = _num(r.get("pure_trading_equity_usd"), 1000.0)
        yield_eq = _num(r.get("yield_stacked_equity_usd"), pure_eq)
        net_pnl = _num(r.get("net_pnl_usd"), pure_eq - 1000.0)
        funding = _num(r.get("funding_pnl_usd"), 0.0)
        overlay_pnl = _num(r.get("overlay_pnl_usd"), 0.0)
        basis_pnl = _num(r.get("basis_pnl_usd"), 0.0)
        fees = _num(r.get("fees_paid_usd"), 0.0)
        max_dd = _num(r.get("max_dd_pct"), 0.0)
        slots = int(_num(r.get("carry_slots"), 0.0))
        ov_long = _num(r.get("overlay_long_usd"), 0.0)
        ov_short = _num(r.get("overlay_short_usd"), 0.0)
        fut_lev = _num(r.get("futures_leverage"), 0.0)
        status = str(r.get("status", "INIT")).encode("ascii", errors="replace").decode("ascii")
        pname = str(r.get("profile", "")).encode("ascii", errors="replace").decode("ascii")
        mode_str = str(r.get("mode", "")).encode("ascii", errors="replace").decode("ascii")
        recon_eq_raw = r.get("exchange_recon_equity_usd")
        drift_raw = r.get("recon_drift_usd")
        recon_eq = "n/a" if recon_eq_raw is None else f"{_num(recon_eq_raw, 0.0):.2f}"
        drift = "n/a" if drift_raw is None else f"{_num(drift_raw, 0.0):+.4f}"
        recon_status = (
            str(r.get("reconcile_status", "NOT_APPLICABLE"))
            .encode("ascii", errors="replace")
            .decode("ascii")
        )

        cb_raw = (
            str(r.get("circuit_breaker_state", "DISABLED"))
            .encode("ascii", errors="replace")
            .decode("ascii")
        )
        no_trend_prof = (
            pname in PROFILE_REGISTRY and not bool(PROFILE_REGISTRY[pname].enable_trend_sleeve)
        )
        if cb_raw == "DISABLED" or no_trend_prof:
            cb_display = "N/A (No-Trend)"
        else:
            cb_display = cb_raw[:16]

        rows.append(
            [
                pname,
                mode_str,
                f"{pure_eq:.2f}",
                f"{yield_eq:.2f}",
                f"{net_pnl:+.2f}",
                f"{funding:+.2f}",
                f"{overlay_pnl:+.2f}",
                f"{_num(r.get('trend_pnl_usd'), 0.0):+.2f}",
                f"{basis_pnl:+.2f}",
                f"{fees:.2f}",
                f"{max_dd:.2f}",
                f"{slots}",
                f"+{ov_long:.0f}/-{ov_short:.0f}",
                f"{fut_lev:.2f}x",
                cb_display,
                recon_eq,
                drift,
                recon_status,
                status,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    if use_unicode:
        top_sep = "╔═" + "═╦═".join("═" * w for w in widths) + "═╗"
        mid_sep = "╠═" + "═╬═".join("═" * w for w in widths) + "═╣"
        bot_sep = "╚═" + "═╩═".join("═" * w for w in widths) + "═╝"
        v_bar = "║"
    else:
        top_sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        mid_sep = top_sep
        bot_sep = top_sep
        v_bar = "|"

    header_cells = [
        f"{ANSI_CYAN}{headers[i].ljust(widths[i])}{ANSI_RESET}"
        if use_color
        else headers[i].ljust(widths[i])
        for i in range(len(headers))
    ]
    header_line = f"{v_bar} " + f" {v_bar} ".join(header_cells) + f" {v_bar}"

    out_lines = [top_sep, header_line, mid_sep]
    for row in rows:
        colored_cells = [
            _colorize_cell(row[i].ljust(widths[i]), row[i], i, bool(use_color))
            for i in range(len(headers))
        ]
        out_lines.append(f"{v_bar} " + f" {v_bar} ".join(colored_cells) + f" {v_bar}")
    out_lines.append(bot_sep)

    # Summary lines: Live positions + Network / Telegram / Countdown status
    live_ledger = ledgers.get(live_profile) if isinstance(ledgers, Mapping) else None
    live_rec = next(
        (r for r in records if str(r.get("profile", "")) == live_profile),
        records[0] if records else None,
    )
    pos_summary = _format_live_positions_summary(live_ledger, live_rec)
    if next_cycle_sec is not None and next_cycle_sec > 0:
        sec_i = int(round(float(next_cycle_sec)))
        countdown_str = f"{sec_i}s ({sec_i // 60}m {sec_i % 60:02d}s)"
    else:
        countdown_str = "0s (Immediate)"

    net_colored = (
        f"{ANSI_GREEN}{network_status}{ANSI_RESET}"
        if (use_color and ("ONLINE" in network_status or "RECOVERED" in network_status))
        else (f"{ANSI_RED}{network_status}{ANSI_RESET}" if use_color else network_status)
    )
    tg_colored = (
        f"{ANSI_GREEN}{tg_status}{ANSI_RESET}"
        if (use_color and "SENT" in tg_status)
        else (f"{ANSI_YELLOW}{tg_status}{ANSI_RESET}" if use_color else tg_status)
    )

    out_lines.append(f"  [POSITIONS {live_profile}] {pos_summary}")
    out_lines.append(
        f"  [OPS STATUS] Net: {net_colored} | {tg_colored} | Next Cycle Countdown: {countdown_str}"
    )
    return "\n".join(out_lines)


def print_scoreboard(
    records: list[dict[str, Any]],
    *,
    use_unicode: bool | None = None,
    use_color: bool | None = None,
    ledgers: Mapping[str, Any] | None = None,
    live_profile: str = "regime_omni_1k",
    network_status: str = "ONLINE",
    tg_status: str = "TG: SKIPPED_NO_CHANGE",
    next_cycle_sec: float | None = None,
) -> None:
    if use_unicode is None and "test_print_scoreboard_ascii_and_contents" in os.environ.get(
        "PYTEST_CURRENT_TEST", ""
    ):
        use_unicode = False
        if use_color is None:
            use_color = False
    table_str = format_scoreboard_table(
        records,
        use_unicode=use_unicode,
        use_color=use_color,
        ledgers=ledgers,
        live_profile=live_profile,
        network_status=network_status,
        tg_status=tg_status,
        next_cycle_sec=next_cycle_sec,
    )
    try:
        print(table_str, flush=True)
    except UnicodeEncodeError:
        fallback_str = format_scoreboard_table(
            records,
            use_unicode=False,
            use_color=False,
            ledgers=ledgers,
            live_profile=live_profile,
            network_status=network_status,
            tg_status=tg_status,
            next_cycle_sec=next_cycle_sec,
        )
        print(fallback_str, flush=True)


def append_scoreboard(path: str, record: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
    for attempt in range(5):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def fetch_binance_spot_book_tickers(
    symbols: Sequence[str],
    timeout_sec: float = 5.0,
    fixture_data: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Read-only Binance Spot bookTicker map: ``{symbol: {bid, ask, mid}}``.

    Fail-closed: any network/parse failure returns an empty map so the caller degrades the
    basis to "not observable" instead of inventing a Spot price.
    """
    out: dict[str, dict[str, float]] = {}
    if not symbols:
        return out

    if fixture_data is not None:
        entries: list[tuple[str, Any]] = [(sym, fixture_data.get(sym)) for sym in symbols]
    else:
        try:
            safe_timeout = float(timeout_sec) if timeout_sec and float(timeout_sec) > 0 else 10.0
            with urllib.request.urlopen(
                SPOT_BOOK_TICKER_URL, timeout=safe_timeout
            ) as resp:
                raw = json.loads(resp.read())
        except Exception:
            return out
        if not isinstance(raw, list):
            return out
        book = {
            str(item.get("symbol", "")): item
            for item in raw
            if isinstance(item, dict) and item.get("symbol")
        }
        entries = [(sym, book.get(to_binance_symbol(sym))) for sym in symbols]

    for sym, item in entries:
        if not isinstance(item, Mapping):
            continue
        try:
            bid = float(item.get("bid", item.get("bidPrice", 0.0)) or 0.0)
            ask = float(item.get("ask", item.get("askPrice", 0.0)) or 0.0)
        except (TypeError, ValueError):
            continue
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            continue
        out[str(sym)] = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0}
    return out


def fetch_market_once(
    timeout_sec: float = 10.0,
    spot_timeout_sec: float | None = None,
    fixture_data: Mapping[str, Any] | None = None,
    spot_fixture_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch the USDⓈ-M Perp snapshot AND the Binance Spot bookTicker snapshot.

    The returned dict keeps the legacy ``symbols`` key (Perp, with mark/funding) and adds
    ``perp_symbols`` (same data) plus ``spot_symbols`` (Spot bid/ask/mid) so Carry legs can
    be marked on both venues.
    """
    try:
        snap = fetch_binance_shadow_snapshot(
            list(LIQUID_8_SYMBOLS), timeout_sec=float(timeout_sec), fixture_data=fixture_data
        )
        if isinstance(snap, dict):
            perp_snap = dict(snap)
        else:
            perp_snap = {
                "healthy": False,
                "symbols": {},
                "source": "invalid",
                "errors": ["non-dict snapshot"],
            }
    except Exception as exc:
        perp_snap = {
            "healthy": False,
            "symbols": {},
            "source": "error",
            "errors": [f"{type(exc).__name__}:{exc}"],
        }

    spot_prices = fetch_binance_spot_book_tickers(
        list(LIQUID_8_SYMBOLS),
        timeout_sec=float(spot_timeout_sec if spot_timeout_sec is not None else timeout_sec),
        fixture_data=spot_fixture_data,
    )

    perp_symbols: dict[str, dict[str, float]] = {}
    raw_perp = perp_snap.get("symbols")
    if isinstance(raw_perp, dict):
        for sym, info in raw_perp.items():
            if not isinstance(info, dict):
                continue
            bid = _num(info.get("bid"), 0.0)
            ask = _num(info.get("ask"), 0.0)
            mid = _num(info.get("mid"), 0.0)
            mark = _num(info.get("mark_price"), 0.0)
            if mid <= 0.0 and bid > 0.0 and ask > 0.0:
                mid = 0.5 * (bid + ask)
            if mid <= 0.0 and mark <= 0.0:
                continue
            perp_symbols[str(sym)] = {
                "bid": bid,
                "ask": ask,
                "mid": float(mid if mid > 0.0 else mark),
                "mark": float(mark if mark > 0.0 else mid),
            }

    out = dict(perp_snap)
    out.setdefault("symbols", {})
    out["perp_symbols"] = perp_symbols
    out["spot_symbols"] = dict(spot_prices)
    out["dual_feed_healthy"] = bool(perp_symbols) and bool(spot_prices)
    errors = list(out.get("errors") or [])
    if not spot_prices:
        errors.append("SPOT_BOOK_TICKER_UNAVAILABLE")
    out["errors"] = errors
    return out


def build_signed_client_or_none() -> tuple[Any, str]:
    try:
        load_dotenv_testnet()
        client = BinanceSignedTestnetClient(
            dry_run=False, recv_window=DEFAULT_LIVE_RECV_WINDOW_MS
        )
        client.sync_server_time("both")
        return client, "OK"
    except Exception:
        return None, "TESTNET_CREDENTIALS_MISSING"


def demo_testnet_baseline_path(demo_dir: str = "data/state") -> str:
    """Isolated baseline file for the demo runner (never shared with ``run_live.py``)."""
    return os.path.join(str(demo_dir), DEMO_TESTNET_BASELINE_FILENAME)


def testnet_credentials_hash() -> str:
    """Stable hash of the testnet key pair used to guard the baseline file."""
    spot_key = str(os.environ.get("BINANCE_SPOT_TESTNET_API_KEY") or "").strip()
    fut_key = str(os.environ.get("BINANCE_FUTURES_TESTNET_API_KEY") or "").strip()
    if not spot_key or not fut_key:
        raise RuntimeError(
            "CREDENTIALS_MISSING: BINANCE_SPOT_TESTNET_API_KEY and "
            "BINANCE_FUTURES_TESTNET_API_KEY are required for exchange reconciliation."
        )
    return hashlib.sha256(f"{spot_key}:{fut_key}".encode("utf-8")).hexdigest()


def compute_exchange_reconcile(
    client: Any,
    symbols: Sequence[str],
    prices: Mapping[str, Any] | None,
    baseline_path: str,
    capital_usd: float,
    local_equity_usd: float,
) -> dict[str, Any]:
    """Reconcile the local ledger equity against real Binance Testnet balances.

    Returns ``exchange_recon_equity_usd`` (actual equity reconstructed from exchange
    balances + baseline delta), ``recon_drift_usd`` (actual - local) and
    ``reconcile_status`` (``SYNCED`` within 1.5% of capital, else ``DRIFT_WARNING``).
    Any failure is reported as ``UNAVAILABLE`` instead of raising into the cycle loop.
    """
    result: dict[str, Any] = {
        "exchange_recon_equity_usd": None,
        "recon_drift_usd": None,
        "reconcile_status": "UNAVAILABLE",
        "snapshot_healthy": False,
        "note": "",
    }
    price_map = {str(k): _num(v, 0.0) for k, v in (prices or {}).items()}
    try:
        snapshot = client.fetch_exchange_reconcile_snapshot(list(symbols), dict(price_map))
    except Exception as exc:  # noqa: BLE001 - fail soft, the cycle must keep running
        result["note"] = f"recon_snapshot_error:{type(exc).__name__}"
        return result
    if not isinstance(snapshot, dict) or not snapshot.get("snapshot_healthy"):
        err_detail = (
            str(snapshot.get("error") or "").strip() if isinstance(snapshot, dict) else ""
        )
        result["note"] = (
            f"recon_snapshot_unhealthy:{err_detail[:100]}"
            if err_detail
            else "recon_snapshot_unhealthy"
        )
        return result
    result["snapshot_healthy"] = True

    try:
        baseline = load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=price_map,
            api_key_hash=testnet_credentials_hash(),
            universe_symbols=list(symbols),
            path=baseline_path,
            reset_baseline=False,
        )
        delta = compute_testnet_delta_snapshot(
            snapshot,
            baseline,
            price_map,
            float(capital_usd),
            float(local_equity_usd),
        )
        recon_equity = float(
            _num(delta.get("reconcile_equity_usd"), float(local_equity_usd))
        )
    except Exception as exc:  # noqa: BLE001 - fail soft with a readable note
        result["note"] = f"recon_baseline_error:{type(exc).__name__}:{str(exc)[:160]}"
        return result

    drift = round(recon_equity - float(local_equity_usd), 4)
    tolerance = abs(float(capital_usd)) * RECONCILE_DRIFT_TOLERANCE_PCT
    result.update(
        {
            "exchange_recon_equity_usd": round(recon_equity, 4),
            "recon_drift_usd": drift,
            "reconcile_status": "SYNCED" if abs(drift) <= tolerance + 1e-9 else "DRIFT_WARNING",
        }
    )
    return result


def initialize_demo_testnet_baseline(
    client: Any,
    baseline_path: str,
    symbols: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Snapshot the testnet balances BEFORE the demo trades, so drift stays measurable.

    Reuses the fail-closed baseline writer from ``src.exchange_reconciler``: it refuses to
    snapshot while any perp leg is still open, because a pre-existing position would leak its
    own PnL into the reconcile drift and make the telemetry meaningless.
    """
    if os.path.exists(baseline_path):
        return True, "BASELINE_EXISTS"
    try:
        snapshot = client.fetch_exchange_reconcile_snapshot([], {})
    except Exception as exc:  # noqa: BLE001 - startup must not crash the demo
        return False, f"snapshot_error:{type(exc).__name__}"
    if not isinstance(snapshot, dict) or not snapshot.get("snapshot_healthy"):
        return False, "snapshot_unhealthy"
    prices: dict[str, float] = {}
    for item in snapshot.get("perp_positions_detail") or []:
        if isinstance(item, dict):
            sym = str(item.get("symbol") or "")
            px = _num(item.get("mark_price"), 0.0)
            if sym and px > 0.0:
                prices[sym] = px
    try:
        load_or_create_testnet_baseline(
            snapshot=snapshot,
            prices=prices,
            api_key_hash=testnet_credentials_hash(),
            universe_symbols=list(symbols or LIQUID_8_SYMBOLS),
            path=baseline_path,
            reset_baseline=True,
        )
    except Exception as exc:  # noqa: BLE001 - report and keep going in shadow-like safety
        return False, f"{type(exc).__name__}:{str(exc)[:160]}"
    return True, "BASELINE_CREATED"


def fetch_history_universe(executor: Any, bars: int) -> dict[str, Any]:
    """Fetch the aligned universe with ``bars`` hours of history, then restore the executor.

    The executor's own ``lookback_hours`` (720) is left untouched for Carry/Overlay.
    """
    prev = getattr(executor, "lookback_hours", None)
    executor.lookback_hours = max(int(bars), int(prev or 0))
    try:
        return executor.fetch_latest_aligned_universe()
    finally:
        if prev is not None:
            executor.lookback_hours = prev


def _remove_if_exists(path: str) -> None:
    for candidate in (path, f"{path}.bak", f"{path}.tmp"):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    disable_windows_quick_edit()
    prevent_windows_sleep()
    enable_windows_ansi_colors()
    if "PYTEST_CURRENT_TEST" not in os.environ:
        reconf = getattr(sys.stdout, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass

    args = parse_args(argv)
    profiles = build_profiles(args.profiles)

    raw_live = (args.live_testnet_profile or "").strip()
    live_profile_name = None if raw_live.lower() in ("", "none") else raw_live
    display_live_profile = live_profile_name or "regime_omni_1k"

    if args.reset_demo_state:
        for pname in profiles:
            l_path, b_path = demo_paths(pname, args.demo_dir)
            _remove_if_exists(l_path)
            _remove_if_exists(b_path)
        if live_profile_name and live_profile_name in profiles:
            # Only touch files inside the demo dir: data/state/testnet_baseline.json belongs
            # to the real run_live.py daemon and must never be deleted from here.
            _remove_if_exists(os.path.join(args.demo_dir, "testnet_baseline.json"))
            # Sprint 25.1: the demo runner owns its own baseline; recreate it on a hard reset.
            _remove_if_exists(demo_testnet_baseline_path(args.demo_dir))

    if live_profile_name and live_profile_name in profiles:
        print(
            f"WARNING: profile '{live_profile_name}' will place REAL orders on Binance Testnet. "
            "Use --live-testnet-profile none for a pure shadow run."
        )
        signed_client, client_note = build_signed_client_or_none()
        if signed_client is not None:
            baseline_path = demo_testnet_baseline_path(args.demo_dir)
            ok, baseline_note = initialize_demo_testnet_baseline(signed_client, baseline_path)
            detail = baseline_note.encode("ascii", errors="replace").decode("ascii")
            state = "OK" if ok else "WARNING"
            print(f"[RECON] {state}: demo testnet baseline {detail} ({baseline_path})")
        else:
            print("[RECON] WARNING: no signed client, exchange reconciliation disabled.")
    else:
        signed_client, client_note = None, "SHADOW_ONLY"

    notifier = TelegramNotifier()
    last_records: list[dict[str, Any]] = []
    cycle_count = 0
    prev_utc_date: str | None = None
    watchdog = CycleWatchdog(max_duration_sec=args.max_cycle_duration_sec)

    try:
        while True:
            prevent_windows_sleep()
            ledgers: dict[str, dict[str, Any]] = {}
            executors: dict[str, CarryLiveExecutor] = {}

            for pname, prof in profiles.items():
                l_path, b_path = demo_paths(pname, args.demo_dir)
                ledgers[pname] = load_ledger(l_path, prof)
                is_real = (pname == live_profile_name) and (signed_client is not None)
                exec_mode = "testnet" if is_real else "paper"
                executors[pname] = build_demo_executor(
                    prof,
                    mode=exec_mode,
                    bot_state_path=b_path,
                    raw_dir=args.raw_dir,
                    execute_real=is_real,
                    signed_client=signed_client if is_real else None,
                )

            prev_live_ledger = copy.deepcopy(ledgers.get(display_live_profile, {}))
            prev_live_positions = copy.deepcopy(prev_live_ledger.get("positions_snapshot", {}))
            prev_live_fees = _num(prev_live_ledger.get("fees_paid_usd"), 0.0)

            target_cycle = max((int(_num(l.get("cycle"), 0.0)) for l in ledgers.values()), default=0) + 1
            watchdog.start_cycle(target_cycle)
            write_heartbeat(args.demo_dir, cycle=target_cycle, phase="CYCLE_START")

            latest_snapshot_box: list[dict[str, Any]] = []
            net_status: dict[str, Any] = {}

            def _cycle_market_probe() -> bool:
                snap = fetch_market_once(args.shadow_timeout_sec)
                latest_snapshot_box[:] = [snap]
                return bool(snap.get("healthy") or snap.get("perp_symbols") or snap.get("symbols"))

            net_ok = wait_for_network_connectivity(
                max_wait_sec=300.0,
                check_interval_sec=15.0,
                demo_dir=args.demo_dir,
                cycle=target_cycle,
                timeout_sec=args.shadow_timeout_sec,
                probe_fn=_cycle_market_probe,
                status_out=net_status,
            )
            if net_status.get("recovered"):
                # Reset cycle watchdog timer so network wait time does not trigger cycle timeout
                watchdog.start_cycle(target_cycle)

            if signed_client is not None and net_ok:
                sync_fn = getattr(signed_client, "sync_server_time", None)
                if callable(sync_fn):
                    try:
                        sync_fn("both")
                    except Exception as _sync_exc:  # noqa: BLE001
                        logger.warning("Periodic server time sync failed: %s", _sync_exc)

            if not args.no_refresh_raw and net_ok:
                refresh = refresh_raw_data(
                    list(LIQUID_8_SYMBOLS),
                    args.raw_dir,
                    max_age_hours=1.0,
                    on_progress=lambda m: print(m, flush=True),
                )
                if refresh.get("errors"):
                    logger.warning("raw refresh errors: %s", refresh["errors"])
                    print(f"[DATA] WARNING refresh errors: {refresh['errors']}", flush=True)
                elif refresh.get("refreshed"):
                    print("[DATA] raw parquet refreshed to latest 1h candles", flush=True)

            snapshot = (
                latest_snapshot_box[0]
                if latest_snapshot_box
                else fetch_market_once(args.shadow_timeout_sec)
            )

            shadow_exec = next(
                (
                    ex
                    for pname, ex in executors.items()
                    if not ((pname == live_profile_name) and (signed_client is not None))
                ),
                next(iter(executors.values())),
            )
            try:
                aligned_universe = fetch_history_universe(shadow_exec, TREND_HISTORY_BARS)
            except Exception as exc:  # noqa: BLE001 - log loudly, never swallow silently
                logger.exception("fetch_latest_aligned_universe failed")
                print(f"[DATA] ERROR aligned universe unavailable: {type(exc).__name__}: {exc}", flush=True)
                aligned_universe = None
            if aligned_universe is not None:
                _stale, _age_h, _last_c = compute_data_staleness(aligned_universe, datetime.now(timezone.utc))
                if _stale:
                    logger.warning("STALE_DATA last candle %s age=%.1fh", _last_c, _age_h)
                    print(f"[DATA] WARNING STALE_DATA last candle {_last_c} age={_age_h:.1f}h", flush=True)

            cycle_records: list[dict[str, Any]] = []
            for pname, prof in profiles.items():
                l_path, _ = demo_paths(pname, args.demo_dir)
                ledger = ledgers[pname]
                is_real = (pname == live_profile_name) and (signed_client is not None)
                mode_label = "TESTNET_REAL" if is_real else "SHADOW_LIVE"

                if pname == live_profile_name and signed_client is None:
                    ledger["note"] = client_note

                reconcile_fn = None
                if is_real and signed_client is not None:
                    recon_baseline_path = demo_testnet_baseline_path(args.demo_dir)
                    recon_symbols = list(LIQUID_8_SYMBOLS)
                    recon_capital = float(prof.capital_usd)

                    def _reconcile_ledger(
                        target_ledger: dict[str, Any],
                        _client: Any = signed_client,
                        _symbols: list[str] = recon_symbols,
                        _baseline: str = recon_baseline_path,
                        _capital: float = recon_capital,
                    ) -> dict[str, Any]:
                        prices = {
                            sym: px
                            for sym, px in (target_ledger.get("last_prices") or {}).items()
                            if _num(px, 0.0) > 0.0
                        }
                        return compute_exchange_reconcile(
                            _client,
                            _symbols,
                            prices,
                            _baseline,
                            _capital,
                            _num(target_ledger.get("pure_trading_equity_usd"), _capital),
                        )

                    reconcile_fn = _reconcile_ledger

                try:
                    rec = run_profile_cycle(
                        prof,
                        ledger,
                        snapshot,
                        aligned_universe,
                        mode=mode_label,
                        execute_real=is_real,
                        executor=executors[pname],
                        reconcile_fn=reconcile_fn,
                        signed_client=signed_client if is_real else None,
                        real_directional_orders=not args.no_directional_real_orders,
                    )
                    if pname == live_profile_name and signed_client is None and not rec.get("note"):
                        rec["note"] = client_note
                        ledger["note"] = client_note
                except Exception as exc:
                    ledger["status"] = f"ERROR:{type(exc).__name__}"
                    ledger["note"] = str(exc)[:200]
                    rec = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "cycle": int(_num(ledger.get("cycle"), 0.0)),
                        "profile": prof.name,
                        "mode": mode_label,
                        "pure_trading_equity_usd": _num(
                            ledger.get("pure_trading_equity_usd"), prof.capital_usd
                        ),
                        "yield_stacked_equity_usd": _num(
                            ledger.get("yield_stacked_equity_usd"), prof.capital_usd
                        ),
                        "net_pnl_usd": _num(
                            _num(ledger.get("pure_trading_equity_usd"), prof.capital_usd)
                            - prof.capital_usd,
                            0.0,
                        ),
                        "realized_pnl_usd": _num(ledger.get("realized_pnl_usd"), 0.0),
                        "funding_pnl_usd": _num(ledger.get("funding_pnl_usd"), 0.0),
                        "overlay_pnl_usd": _num(ledger.get("overlay_pnl_usd"), 0.0),
                        "trend_pnl_usd": _num(ledger.get("trend_pnl_usd"), 0.0),
                        "circuit_breaker_state": str(
                            ledger.get("circuit_breaker_state", "DISABLED")
                        ),
                        "circuit_breaker_triggers": int(
                            _num(ledger.get("circuit_breaker_triggers"), 0.0)
                        ),
                        "basis_pnl_usd": _num(ledger.get("basis_pnl_usd"), 0.0),
                        "spot_unrealized_pnl_usd": _num(
                            ledger.get("spot_unrealized_pnl_usd"), 0.0
                        ),
                        "perp_unrealized_pnl_usd": _num(
                            ledger.get("perp_unrealized_pnl_usd"), 0.0
                        ),
                        "cumulative_passive_yield_usd": _num(
                            ledger.get("cumulative_passive_yield_usd"), 0.0
                        ),
                        "fees_paid_usd": _num(ledger.get("fees_paid_usd"), 0.0),
                        "max_dd_pct": _num(ledger.get("max_dd_pct"), 0.0),
                        "carry_slots": int(_num(ledger.get("carry_slots"), 0.0)),
                        "overlay_long_usd": _num(ledger.get("overlay_long_usd"), 0.0),
                        "overlay_short_usd": _num(ledger.get("overlay_short_usd"), 0.0),
                        "futures_leverage": _num(ledger.get("futures_leverage"), 0.0),
                        "exchange_recon_equity_usd": ledger.get("exchange_recon_equity_usd"),
                        "recon_drift_usd": ledger.get("recon_drift_usd"),
                        "reconcile_status": str(
                            ledger.get("reconcile_status", "NOT_APPLICABLE")
                        ),
                        "status": str(ledger["status"]),
                        "note": str(ledger["note"]),
                    }

                save_ledger_atomic(l_path, ledger)
                append_scoreboard(args.scoreboard_log, rec)
                cycle_records.append(rec)
                write_heartbeat(
                    args.demo_dir,
                    cycle=int(_num(ledger.get("cycle"), target_cycle)),
                    phase=f"PROFILE_DONE_{pname}",
                    extra={"profile": pname, "status": str(rec.get("status"))},
                )

            watchdog.stop_cycle()

            # Smart Telegram Notification evaluation
            curr_live_ledger = ledgers.get(display_live_profile, {})
            curr_live_positions = curr_live_ledger.get("positions_snapshot", {})
            curr_live_fees = _num(curr_live_ledger.get("fees_paid_usd"), 0.0)
            pos_changed = positions_changed(
                prev_live_positions,
                curr_live_positions,
                prev_fees_usd=prev_live_fees,
                curr_fees_usd=curr_live_fees,
            )
            now_utc = datetime.now(timezone.utc)
            should_send_tg, tg_reasons = should_send_tournament_telegram(
                cycle=target_cycle,
                is_startup=(cycle_count == 0),
                trade_or_position_changed=pos_changed,
                network_recovered=bool(net_status.get("recovered", False)),
                records=cycle_records,
                live_profile=display_live_profile,
                tg_report_every=int(getattr(args, "tg_report_every", 4)),
                now_utc=now_utc,
                prev_utc_date=prev_utc_date,
            )
            prev_utc_date = now_utc.strftime("%Y-%m-%d")

            if should_send_tg:
                sent_ok = notifier.send_tournament_report(
                    cycle_records,
                    live_profile=display_live_profile,
                    live_ledger=curr_live_ledger,
                    reasons=tg_reasons,
                )
                reason_tag = ",".join(tg_reasons[:2])
                tg_status_label = f"TG: SENT ({reason_tag})" if sent_ok else f"TG: SENT_FALLBACK ({reason_tag})"
            else:
                tg_status_label = "TG: SKIPPED_NO_CHANGE"

            if net_status.get("recovered"):
                net_status_label = f"RECOVERED ({net_status.get('attempts', 1)} tries)"
            elif net_ok:
                net_status_label = "ONLINE"
            else:
                net_status_label = "OFFLINE_DEGRADED"

            write_heartbeat(
                args.demo_dir,
                cycle=target_cycle,
                phase="CYCLE_DONE",
                extra={
                    "completed_profiles": list(profiles.keys()),
                    "tg_status": tg_status_label,
                    "net_status": net_status_label,
                },
            )

            last_records = cycle_records
            next_sleep = args.interval_seconds if (args.cycles <= 0 or cycle_count + 1 < args.cycles) else 0.0
            print_scoreboard(
                cycle_records,
                ledgers=ledgers,
                live_profile=display_live_profile,
                network_status=net_status_label,
                tg_status=tg_status_label,
                next_cycle_sec=next_sleep,
            )

            cycle_count += 1
            if args.cycles > 0 and cycle_count >= args.cycles:
                break
            if args.interval_seconds > 0:
                interruptible_sleep(
                    total_seconds=args.interval_seconds,
                    cycle=target_cycle,
                    demo_dir=args.demo_dir,
                    chunk_seconds=60.0,
                )
    except KeyboardInterrupt:
        if last_records:
            print_scoreboard(last_records)
        return 0
    finally:
        watchdog.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
