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
from datetime import datetime, timezone
import hashlib
import json
import math
import os
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
from src.state_store import StateStore
from src.strategy_profiles import (
    PROFILE_REGISTRY,
    StrategyProfileConfig,
    compute_step_passive_yield_usd,
)

CARRY_FEE_RATE = (7.5 + 5.0) / 10000.0
OVERLAY_FEE_RATE = 5.0 / 10000.0

#: Perp taker fee applied to every Trend sleeve delta (same 5.0 bps as the overlay leg).
TREND_FEE_RATE = 5.0 / 10000.0

#: Lot step per symbol for quantity quantization (Binance USD-M / Spot filters).
DEMO_LOT_STEPS: dict[str, float] = {
    "BTC/USDT:USDT": 0.001,
    "ETH/USDT:USDT": 0.001,
    "SOL/USDT:USDT": 0.01,
    "BNB/USDT:USDT": 0.001,
    "LINK/USDT:USDT": 0.01,
    "AVAX/USDT:USDT": 0.01,
    "XRP/USDT:USDT": 1.0,
    "DOGE/USDT:USDT": 1.0,
}

#: Binance Spot bookTicker endpoint (public, read-only). The USDT-M feed lives in
#: ``src.binance_public_feed`` and only covers Perp, hence this second leg.
SPOT_BOOK_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"

#: Isolated testnet baseline for the demo runner. Never share it with ``run_live.py``.
DEMO_TESTNET_BASELINE_FILENAME = "demo_testnet_baseline.json"

#: Reconcile drift tolerance (fraction of capital) before the status flips to DRIFT_WARNING.
RECONCILE_DRIFT_TOLERANCE_PCT = 0.015


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
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                base.update(loaded)
        except Exception:
            pass
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
    os.replace(tmp_path, path)


def build_demo_executor(
    profile: StrategyProfileConfig,
    *,
    mode: str,
    bot_state_path: str,
    raw_dir: str = "data/raw",
    execute_real: bool = False,
    signed_client: Any = None,
) -> CarryLiveExecutor:
    if not execute_real or mode in ("SHADOW_LIVE", "paper"):
        exec_mode = "paper"
    elif mode in ("TESTNET_REAL", "testnet"):
        exec_mode = "testnet"
    else:
        exec_mode = str(mode)

    return CarryLiveExecutor(
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

    ledger["basis_pnl_usd"] = basis_total
    ledger["spot_unrealized_pnl_usd"] = spot_unreal_total
    ledger["perp_unrealized_pnl_usd"] = perp_unreal_total

    yield_res = compute_step_passive_yield_usd(profile, dt_hours, deployed_spot)
    step_passive = _num(yield_res.get("total_passive_yield_usd"), 0.0)
    ledger["cumulative_passive_yield_usd"] = (
        _num(ledger.get("cumulative_passive_yield_usd"), 0.0) + step_passive
    )

    realized = _num(ledger.get("realized_pnl_usd"), 0.0)
    unrealized_total = unrealized_carry + unrealized_overlay + unrealized_trend
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
        if isinstance(pos, dict) and str(pos.get("kind", "carry")) == "trend":
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
    ledger["basis_pnl_usd"] = basis_total
    ledger["spot_unrealized_pnl_usd"] = spot_unreal_total
    ledger["perp_unrealized_pnl_usd"] = perp_unreal_total
    pure_eq = (
        _num(profile.capital_usd, 1000.0)
        + _num(ledger.get("realized_pnl_usd"), 0.0)
        + unreal_carry
        + unreal_overlay
        + unreal_trend
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

    overlay_gross = overlay_long_usd + overlay_short_usd
    fut_wallet = _num(profile.futures_wallet_usd, 0.0)
    fut_lev = (
        (carry_perp_notional + overlay_gross + trend_gross) / fut_wallet
        if fut_wallet > 0.0
        else 0.0
    )
    ledger["carry_slots"] = int(carry_slots)
    ledger["overlay_long_usd"] = _num(overlay_long_usd, 0.0)
    ledger["overlay_short_usd"] = _num(overlay_short_usd, 0.0)
    ledger["trend_gross_usd"] = _num(trend_gross, 0.0)
    ledger["futures_leverage"] = _num(fut_lev, 0.0)


def apply_trend_sleeve(
    ledger: dict[str, Any],
    profile: StrategyProfileConfig,
    aligned_universe: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    *,
    now_utc: datetime | None = None,
    charge_fees: bool = True,
) -> dict[str, Any]:
    """Mark and rebalance the directional Trend sleeve (long Perp) on isolated keys.

    Every Trend leg lives under ``"SYM|trend"`` inside ``positions_snapshot`` so a symbol can
    be held by carry, overlay and trend simultaneously without any collision. Sizing is
    inverse-volatility with a hard gross cap, then multiplied by the 2-tier drawdown circuit
    breaker ``risk_scale`` (0.5 at -6%, 0.0 at -10% until the cooldown expires).
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
        return disabled
    price_frame = price_frame.sort_index().ffill()

    trend_cfg = TrendSleeveConfig(
        capital_usd=capital,
        target_vol_ann=float(profile.trend_target_vol_ann),
        max_gross_pct=float(profile.trend_max_gross_pct),
        rebalance_every_hours=int(profile.trend_rebalance_hours),
        z_entry=float(profile.trend_z_entry),
    )
    state = TrendSleeveState.from_snapshot(ledger.get("trend_sleeve_state"))

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
    )

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
    pos_map = ledger.get("positions_snapshot", {}) or {}
    used_notional = 0.0
    has_trend_positions = False
    for _key, pos in (pos_map.items() if isinstance(pos_map, dict) else []):
        if not isinstance(pos, dict):
            continue
        _kind = str(pos.get("kind", "carry"))
        if _kind in ("carry", "overlay"):
            used_notional += abs(_num(pos.get("signed_usd"), 0.0))
        elif _kind == "trend" and abs(_num(pos.get("qty"), 0.0)) > 0.0:
            has_trend_positions = True
    fut_capacity = max(
        0.0,
        _num(profile.futures_wallet_usd, 0.0) * _num(profile.max_futures_leverage, 1.0)
        - used_notional,
    )
    desired_gross = float(sum(weights.values())) * capital
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
    live_weights = any(float(w) > 0.0 for w in weights.values())
    # Emergency flatten must never wait for the 24h clock: a hard-stopped breaker
    # (risk_scale == 0) or an empty target set with positions still open both qualify.
    force_flat = has_trend_positions and (risk_scale_now <= 0.0 or not live_weights)
    if not rebalance_due and not force_flat:
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

    # ---- 5) Apply deltas on "SYM|trend" keys.
    new_positions = dict(pos_map) if isinstance(pos_map, dict) else {}
    existing_trend_syms = {
        _pos_symbol(str(k), v)
        for k, v in new_positions.items()
        if isinstance(v, dict) and str(v.get("kind", "carry")) == "trend"
    }
    for sym in sorted(existing_trend_syms):
        targets.setdefault(sym, 0.0)

    fees = 0.0
    turnover = 0.0
    n_orders = 0
    trend_realized = 0.0
    realized = _num(ledger.get("realized_pnl_usd"), 0.0)
    for sym in sorted(targets):
        px = float(prices_now.get(sym, 0.0))
        if px <= 0.0:
            continue
        key = _snapshot_key(sym, "trend")
        old_pos = new_positions.get(key) if isinstance(new_positions.get(key), dict) else {}
        old_qty = _num(old_pos.get("qty"), 0.0)
        old_entry = _num(old_pos.get("entry_price"), 0.0)
        new_qty = float(targets.get(sym, 0.0))
        delta = new_qty - old_qty
        if abs(delta) <= 1e-12:
            continue
        notional = abs(delta) * px
        fee = notional * TREND_FEE_RATE if charge_fees else 0.0
        fees += fee
        turnover += notional
        n_orders += 1
        if charge_fees:
            realized -= fee
        if new_qty <= 0.0:
            if old_qty > 0.0 and old_entry > 0.0:
                closed_pnl = old_qty * (px - old_entry)
                trend_realized += closed_pnl
                realized += closed_pnl
            new_positions.pop(key, None)
            continue
        if old_qty > 0.0 and old_entry > 0.0 and new_qty > old_qty:
            entry_px = _vwap_entry(old_qty, old_entry, new_qty, px)
        elif old_qty > 0.0 and old_entry > 0.0 and new_qty < old_qty:
            closed_pnl = (old_qty - new_qty) * (px - old_entry)
            trend_realized += closed_pnl
            realized += closed_pnl
            entry_px = old_entry
        else:
            entry_px = px
        new_positions[key] = {
            "symbol": sym,
            "qty": float(new_qty),
            "entry_price": float(entry_px),
            "kind": "trend",
            # Cost basis (qty * entry) so the unrealized mark qty*entry*(mid/entry - 1)
            # is exactly qty*(mid - entry); using the live price here would mis-scale PnL.
            "signed_usd": float(new_qty * entry_px),
        }

    ledger["positions_snapshot"] = new_positions
    ledger["realized_pnl_usd"] = realized
    if charge_fees:
        ledger["fees_paid_usd"] = _num(ledger.get("fees_paid_usd"), 0.0) + fees
    # Attribution: trend_pnl_usd must carry its own realised closes and its own fees,
    # otherwise the scoreboard column silently under-reports the sleeve.
    ledger["trend_pnl_usd"] = (
        _num(ledger.get("trend_pnl_usd"), 0.0) + trend_realized - (fees if charge_fees else 0.0)
    )
    ledger["trend_sleeve_state"] = state.snapshot()
    ledger["trend_active_symbols"] = sorted(
        s for s, is_on in (state.active or {}).items() if bool(is_on)
    )
    if rebalance_due:
        ledger["last_trend_rebalance_ts"] = now.isoformat()
    held = sorted(
        _pos_symbol(str(k), v)
        for k, v in new_positions.items()
        if isinstance(v, dict) and str(v.get("kind", "carry")) == "trend"
    )
    return {
        "state": str(cb["state"]),
        "risk_scale": _num(cb["risk_scale"], 1.0),
        "n_orders": int(n_orders),
        "turnover_notional": float(turnover),
        "fees_usd": float(fees),
        "targets": {str(k): float(v) for k, v in targets.items()},
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
) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    ledger["profile"] = profile.name
    ledger["mode"] = "TESTNET_REAL" if execute_real else "SHADOW_LIVE"
    ledger["cycle"] = int(_num(ledger.get("cycle"), 0.0)) + 1

    accrue_cycle_accounting(
        ledger,
        profile,
        snapshot,
        now_utc,
        ledger.get("positions_snapshot", {}),
    )

    reconcile_failed = False
    try:
        executor.reconcile_once(aligned_universe=aligned_universe)
        ledger["status"] = "OK"
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
                ledger["note"] = (ledger.get("note") or "")[:120] + " fees_estimated"
            _mark_to_market(ledger, profile, sym_map, last_prices, snapshot)

    # ---- Sprint 26: trend sleeve (long Perp on isolated "SYM|trend" ledger keys)
    trend_result = apply_trend_sleeve(
        ledger, profile, aligned_universe, snapshot, now_utc=now_utc, charge_fees=True
    )
    if trend_result.get("n_orders", 0) > 0 and not reconcile_failed:
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
                ledger["note"] = (
                    (ledger.get("note") or "")[:120] + " " + recon_note
                ).strip()

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


def print_scoreboard(records: list[dict[str, Any]]) -> None:

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
        "CB State",
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
                str(r.get("circuit_breaker_state", "DISABLED"))
                .encode("ascii", errors="replace")
                .decode("ascii")[:16],
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

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        print(line)
    print(sep)


def append_scoreboard(path: str, record: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        f.flush()


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
            with urllib.request.urlopen(
                SPOT_BOOK_TICKER_URL, timeout=float(timeout_sec)
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
        result["note"] = "recon_snapshot_unhealthy"
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


def _remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profiles = build_profiles(args.profiles)

    raw_live = (args.live_testnet_profile or "").strip()
    live_profile_name = None if raw_live.lower() in ("", "none") else raw_live

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

    last_records: list[dict[str, Any]] = []
    cycle_count = 0

    try:
        while True:
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

            snapshot = fetch_market_once(args.shadow_timeout_sec)

            shadow_exec = next(
                (
                    ex
                    for pname, ex in executors.items()
                    if not ((pname == live_profile_name) and (signed_client is not None))
                ),
                next(iter(executors.values())),
            )
            try:
                aligned_universe = shadow_exec.fetch_latest_aligned_universe()
            except Exception:
                aligned_universe = None

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

            last_records = cycle_records
            print_scoreboard(cycle_records)

            cycle_count += 1
            if args.cycles > 0 and cycle_count >= args.cycles:
                break
            if args.interval_seconds > 0:
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        if last_records:
            print_scoreboard(last_records)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
