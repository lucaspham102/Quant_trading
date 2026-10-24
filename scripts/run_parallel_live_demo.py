"""Parallel live/shadow tournament runner for 1,000 USDT strategy profiles.

Runs multiple 1,000 USDT profiles side-by-side:
- At most ONE profile executes real persistent orders on Binance Testnet.
- Remaining profiles run in SHADOW_LIVE mode with simulated fees and isolated state.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
import sys
import time
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.barbell_portfolio import BarbellConfig
from src.binance_public_feed import fetch_binance_shadow_snapshot
from src.binance_signed_client import (
    DEFAULT_LIVE_RECV_WINDOW_MS,
    BinanceSignedTestnetClient,
    load_dotenv_testnet,
)
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.cross_exchange_carry import LIQUID_8_SYMBOLS
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore
from src.strategy_profiles import (
    PROFILE_REGISTRY,
    StrategyProfileConfig,
    compute_step_passive_yield_usd,
)

CARRY_FEE_RATE = (7.5 + 5.0) / 10000.0
OVERLAY_FEE_RATE = 5.0 / 10000.0


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
        default="alpha_boost_1k",
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
        "cumulative_passive_yield_usd": 0.0,
        "pure_trading_equity_usd": capital,
        "yield_stacked_equity_usd": capital,
        "peak_equity_usd": capital,
        "max_dd_pct": 0.0,
        "positions_snapshot": {},
        "carry_slots": 0,
        "overlay_long_usd": 0.0,
        "overlay_short_usd": 0.0,
        "futures_leverage": 0.0,
        "status": "INIT",
        "note": "",
    }


def _sanitize_ledger(ledger: dict[str, Any], default_capital: float = 1000.0) -> dict[str, Any]:
    float_keys = (
        "realized_pnl_usd",
        "fees_paid_usd",
        "funding_pnl_usd",
        "overlay_pnl_usd",
        "cumulative_passive_yield_usd",
        "pure_trading_equity_usd",
        "yield_stacked_equity_usd",
        "peak_equity_usd",
        "max_dd_pct",
        "overlay_long_usd",
        "overlay_short_usd",
        "futures_leverage",
    )
    for k in float_keys:
        fallback = default_capital if k in (
            "pure_trading_equity_usd",
            "yield_stacked_equity_usd",
            "peak_equity_usd",
        ) else 0.0
        ledger[k] = _num(ledger.get(k), fallback)

    ledger["cycle"] = int(_num(ledger.get("cycle"), 0.0))
    ledger["carry_slots"] = int(_num(ledger.get("carry_slots"), 0.0))
    ledger["profile"] = str(ledger.get("profile", ""))
    ledger["mode"] = str(ledger.get("mode", "SHADOW_LIVE"))
    ledger["status"] = str(ledger.get("status", "INIT"))
    ledger["note"] = str(ledger.get("note", ""))

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
                clean_pos[str(sym)] = {
                    "symbol": str(info.get("symbol") or str(sym).split("|")[0]),
                    "qty": _num(info.get("qty"), 0.0),
                    "entry_price": _num(info.get("entry_price"), 0.0),
                    "kind": str(info.get("kind", "carry")),
                    "signed_usd": _num(info.get("signed_usd"), 0.0),
                }
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
            if kind == "carry":
                # Short perp leg receives funding when the rate is positive.
                funding_delta += qty * mid * rate
            elif kind == "overlay":
                ov_f = -signed_usd * rate
                funding_delta += ov_f
                overlay_funding_delta += ov_f
        ledger["funding_pnl_usd"] = _num(ledger.get("funding_pnl_usd"), 0.0) + funding_delta
        ledger["realized_pnl_usd"] = _num(ledger.get("realized_pnl_usd"), 0.0) + funding_delta
        ledger["overlay_pnl_usd"] = _num(ledger.get("overlay_pnl_usd"), 0.0) + overlay_funding_delta
        ledger["last_funding_ts"] = latest_funding_iso

    prev_updated_dt = _parse_iso_utc(ledger.get("updated_at"))
    if prev_updated_dt is not None:
        raw_dt_hours = (utc_now - prev_updated_dt).total_seconds() / 3600.0
        dt_hours = min(24.0, max(0.0, raw_dt_hours))
    else:
        dt_hours = 0.0

    unrealized_carry = 0.0
    unrealized_overlay = 0.0
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
            # Delta neutral (spot long + perp short): directional price PnL is zero.
            deployed_spot[sym] = abs(qty) * mid
        elif kind == "overlay":
            if entry > 0.0 and mid > 0.0:
                unrealized_overlay += signed_usd * (mid / entry - 1.0)

    yield_res = compute_step_passive_yield_usd(profile, dt_hours, deployed_spot)
    step_passive = _num(yield_res.get("total_passive_yield_usd"), 0.0)
    ledger["cumulative_passive_yield_usd"] = (
        _num(ledger.get("cumulative_passive_yield_usd"), 0.0) + step_passive
    )

    realized = _num(ledger.get("realized_pnl_usd"), 0.0)
    unrealized_total = unrealized_carry + unrealized_overlay
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
    """Ledger key for one leg: carry keeps the plain symbol, overlay is namespaced.

    A symbol can be held by both sleeves at once (e.g. BTC carry + BTC overlay short),
    so the two legs must never share one ledger slot.
    """
    return str(sym) if kind == "carry" else f"{sym}|overlay"


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

        # ---- carry leg (delta neutral: no directional price PnL)
        c_pos = carry_map.get(sym)
        c_qty = _num(c_pos.get("qty"), 0.0) if isinstance(c_pos, dict) else 0.0
        if c_qty == 0.0 and isinstance(c_pos, dict):
            c_notional = _num(c_pos.get("notional"), 0.0)
            if c_notional != 0.0:
                c_qty = c_notional / mid
        old_c = prev.get(_snapshot_key(sym, "carry")) or {}
        old_qty = _num(old_c.get("qty"), 0.0)
        old_entry = _num(old_c.get("entry_price"), 0.0)
        if charge_fees:
            fees += abs(c_qty - old_qty) * mid * CARRY_FEE_RATE

        if abs(c_qty) > 1e-12:
            same_sign = old_qty != 0.0 and (old_qty > 0.0) == (c_qty > 0.0)
            if same_sign and old_entry > 0.0:
                entry_px = _vwap_entry(abs(old_qty), old_entry, abs(c_qty), mid)
            else:
                exec_entry = 0.0
                if isinstance(c_pos, dict):
                    exec_entry = _num(
                        c_pos.get("entry_perp_price", c_pos.get("entry_price")), 0.0
                    )
                entry_px = exec_entry if exec_entry > 0.0 else mid
            new_snapshot[_snapshot_key(sym, "carry")] = {
                "symbol": sym,
                "qty": _num(c_qty, 0.0),
                "entry_price": _num(entry_px, 0.0),
                "kind": "carry",
                "signed_usd": _num(c_qty * mid, 0.0),
            }

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
) -> tuple[float, float]:
    """Mark-to-market PnL split into (carry, overlay).

    Carry contributes 0: the spot long and the perp short offset each other, so only
    funding/basis move the book (basis is not observable from a single mid price).
    """
    unreal_carry = 0.0
    unreal_overlay = 0.0
    if not isinstance(positions, dict):
        return 0.0, 0.0
    for key, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        sym = _pos_symbol(str(key), pos)
        kind = str(pos.get("kind", "carry"))
        qty = _num(pos.get("qty"), 0.0)
        entry = _num(pos.get("entry_price"), 0.0)
        signed_usd = _num(pos.get("signed_usd"), qty * entry)
        mid = _extract_mid_price(sym, sym_map, last_prices, entry)
        if mid <= 0.0 or entry <= 0.0:
            continue
        if kind == "overlay":
            unreal_overlay += signed_usd * (mid / entry - 1.0)
        # carry: delta neutral -> no directional mark
    return unreal_carry, unreal_overlay


def _mark_to_market(
    ledger: dict[str, Any],
    profile: StrategyProfileConfig,
    sym_map: dict[str, Any],
    last_prices: dict[str, float],
) -> None:
    unreal_carry, unreal_overlay = _unrealized_from_snapshot(
        ledger.get("positions_snapshot", {}) or {}, sym_map, last_prices
    )
    pure_eq = (
        _num(profile.capital_usd, 1000.0)
        + _num(ledger.get("realized_pnl_usd"), 0.0)
        + unreal_carry
        + unreal_overlay
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
    _mark_to_market(ledger, profile, sym_map, last_prices)
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

    overlay_gross = overlay_long_usd + overlay_short_usd
    fut_wallet = _num(profile.futures_wallet_usd, 0.0)
    fut_lev = (
        (carry_perp_notional + overlay_gross) / fut_wallet if fut_wallet > 0.0 else 0.0
    )
    ledger["carry_slots"] = int(carry_slots)
    ledger["overlay_long_usd"] = _num(overlay_long_usd, 0.0)
    ledger["overlay_short_usd"] = _num(overlay_short_usd, 0.0)
    ledger["futures_leverage"] = _num(fut_lev, 0.0)


def run_profile_cycle(
    profile: StrategyProfileConfig,
    ledger: dict[str, Any],
    snapshot: dict[str, Any] | None,
    aligned_universe: dict[str, Any] | None,
    *,
    mode: str,
    execute_real: bool,
    executor: Any,
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
            _mark_to_market(ledger, profile, sym_map, last_prices)

    _update_exposure_metrics(ledger, profile)
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
        "cumulative_passive_yield_usd": _num(ledger["cumulative_passive_yield_usd"], 0.0),
        "fees_paid_usd": _num(ledger["fees_paid_usd"], 0.0),
        "max_dd_pct": _num(ledger["max_dd_pct"], 0.0),
        "carry_slots": int(ledger["carry_slots"]),
        "overlay_long_usd": _num(ledger["overlay_long_usd"], 0.0),
        "overlay_short_usd": _num(ledger["overlay_short_usd"], 0.0),
        "futures_leverage": _num(ledger["futures_leverage"], 0.0),
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
        "Fees ($)",
        "MaxDD (%)",
        "Carry Slots",
        "Overlay L/S ($)",
        "Fut Lev (x)",
        "Status",
    ]
    rows: list[list[str]] = []
    for r in records:
        pure_eq = _num(r.get("pure_trading_equity_usd"), 1000.0)
        yield_eq = _num(r.get("yield_stacked_equity_usd"), pure_eq)
        net_pnl = _num(r.get("net_pnl_usd"), pure_eq - 1000.0)
        funding = _num(r.get("funding_pnl_usd"), 0.0)
        overlay_pnl = _num(r.get("overlay_pnl_usd"), 0.0)
        fees = _num(r.get("fees_paid_usd"), 0.0)
        max_dd = _num(r.get("max_dd_pct"), 0.0)
        slots = int(_num(r.get("carry_slots"), 0.0))
        ov_long = _num(r.get("overlay_long_usd"), 0.0)
        ov_short = _num(r.get("overlay_short_usd"), 0.0)
        fut_lev = _num(r.get("futures_leverage"), 0.0)
        status = str(r.get("status", "INIT")).encode("ascii", errors="replace").decode("ascii")
        pname = str(r.get("profile", "")).encode("ascii", errors="replace").decode("ascii")
        mode_str = str(r.get("mode", "")).encode("ascii", errors="replace").decode("ascii")

        rows.append(
            [
                pname,
                mode_str,
                f"{pure_eq:.2f}",
                f"{yield_eq:.2f}",
                f"{net_pnl:+.2f}",
                f"{funding:+.2f}",
                f"{overlay_pnl:+.2f}",
                f"{fees:.2f}",
                f"{max_dd:.2f}",
                f"{slots}",
                f"+{ov_long:.0f}/-{ov_short:.0f}",
                f"{fut_lev:.2f}x",
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


def fetch_market_once(timeout_sec: float = 10.0) -> dict[str, Any]:
    try:
        snap = fetch_binance_shadow_snapshot(
            list(LIQUID_8_SYMBOLS), timeout_sec=float(timeout_sec)
        )
        if isinstance(snap, dict):
            return snap
        return {"healthy": False, "symbols": {}, "source": "invalid", "errors": ["non-dict snapshot"]}
    except Exception as exc:
        return {
            "healthy": False,
            "symbols": {},
            "source": "error",
            "errors": [f"{type(exc).__name__}:{exc}"],
        }


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

    if live_profile_name and live_profile_name in profiles:
        print(
            f"WARNING: profile '{live_profile_name}' will place REAL orders on Binance Testnet. "
            "Use --live-testnet-profile none for a pure shadow run."
        )
        signed_client, client_note = build_signed_client_or_none()
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

                try:
                    rec = run_profile_cycle(
                        prof,
                        ledger,
                        snapshot,
                        aligned_universe,
                        mode=mode_label,
                        execute_real=is_real,
                        executor=executors[pname],
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
                        "cumulative_passive_yield_usd": _num(
                            ledger.get("cumulative_passive_yield_usd"), 0.0
                        ),
                        "fees_paid_usd": _num(ledger.get("fees_paid_usd"), 0.0),
                        "max_dd_pct": _num(ledger.get("max_dd_pct"), 0.0),
                        "carry_slots": int(_num(ledger.get("carry_slots"), 0.0)),
                        "overlay_long_usd": _num(ledger.get("overlay_long_usd"), 0.0),
                        "overlay_short_usd": _num(ledger.get("overlay_short_usd"), 0.0),
                        "futures_leverage": _num(ledger.get("futures_leverage"), 0.0),
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
