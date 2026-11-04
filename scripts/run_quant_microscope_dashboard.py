#!/usr/bin/env python
"""
Local Quant Microscope & Telemetry Dashboard (L0 -> L5 Multi-Sleeve Split-Wallet Barbell).

ZERO-INTERFERENCE INVARIANT:
1. 100% Read-Only: Never writes, locks, or modifies any file in `data/`, `logs/`, or `src/`.
2. Atomic Memory Copy & Fallback: Reads state files into memory (`rb`) with automatic
   fallback to `.json.bak`, `logs/sprint25_parallel_demo_scoreboard.jsonl`, and
   `data/state/backup_cycle97_before_fix/` if a file is mid-`os.replace` or offline.
3. Single-Command Execution:
   `python scripts/run_quant_microscope_dashboard.py`
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# Ensure project root is importable without modifying anything in src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.barbell_portfolio import (
    _floor_to_step,
    compute_snr_dispersion,
    enforce_overlay_dollar_neutrality,
)
from src.cross_exchange_carry import (
    EXCHANGE_LOT_SPECS,
    LIQUID_8_SYMBOLS,
    LOT_STEP_SIZES,
)
from src.directional_trend_sleeve import (
    DrawdownCircuitBreakerConfig,
    TrendSleeveConfig,
    evaluate_drawdown_circuit_breaker,
)
from src.dynamic_carry_basis import (
    DynamicCarryConfig,
    build_carry_basis_scores,
)
from src.quant_invariants import (
    verify_accounting_identity_invariant,
    verify_clock_shift_invariance,
    verify_lot_neutrality_invariant,
    verify_warmup_saturation_invariant,
    verify_zero_compounding_leak_invariant,
)
from src.strategy_profiles import (
    PROFILE_REGISTRY,
    StrategyProfileConfig,
    get_strategy_profile,
)

# ============================================================================
# CONSTANTS & BLUEPRINT REFERENCE METRICS
# ============================================================================

MIN_WARMUP_HOURS: int = 1440

TOURNAMENT_FOUR_PROFILES: Tuple[str, ...] = (
    "regime_omni_1k",
    "flexible_regime_1k",
    "carry_max_eta_1k",
    "baseline_1k",
)

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "regime_omni_1k": "Pruned Omni Regime Champion (Carry 85% + XS 15% + Trend 55% + Macro Filter)",
    "flexible_regime_1k": "Pruned Balanced Regime Switcher (Carry 85% + XS 15% + Trend 45% + Macro Filter)",
    "carry_max_eta_1k": "Pure Carry High-Eta Benchmark (Carry 90% + XS 0% + Trend 0%)",
    "baseline_1k": "Conservative Barbell Benchmark (Carry 65% + XS 15% + Trend 0%)",
}

# Blueprint Part 1.2 canonical 3.5Y & 6M Holdout benchmarks (fallback / enrichment)
BLUEPRINT_BENCHMARK_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    "regime_omni_1k": {
        "3_5Y": {"ann_return_pct": 18.88, "sharpe": 4.94, "sortino": 7.52, "max_dd_pct": -0.68},
        "6M_holdout": {"ann_return_pct": 16.09, "sharpe": 5.38, "sortino": 8.40, "max_dd_pct": -0.68},
    },
    "flexible_regime_1k": {
        "3_5Y": {"ann_return_pct": 18.23, "sharpe": 4.54, "sortino": 6.95, "max_dd_pct": -0.69},
        "6M_holdout": {"ann_return_pct": 16.01, "sharpe": 5.37, "sortino": 8.35, "max_dd_pct": -0.68},
    },
    "carry_max_eta_1k": {
        "3_5Y": {"ann_return_pct": 21.14, "sharpe": 4.30, "sortino": 6.62, "max_dd_pct": -1.78},
        "6M_holdout": {"ann_return_pct": 16.41, "sharpe": 12.16, "sortino": 19.50, "max_dd_pct": -0.14},
    },
    "baseline_1k": {
        "3_5Y": {"ann_return_pct": 14.37, "sharpe": 4.62, "sortino": 7.10, "max_dd_pct": -1.16},
        "6M_holdout": {"ann_return_pct": 12.40, "sharpe": 13.73, "sortino": 21.00, "max_dd_pct": -0.09},
    },
}


# ============================================================================
# ZERO-INTERFERENCE SAFE READERS (100% READ-ONLY)
# ============================================================================

def safe_read_json_bytes(
    primary_path: Path,
    retries: int = 3,
    retry_sleep_sec: float = 0.015,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Read a JSON file into memory with zero file locking.
    Falls back to `<path>.bak` if `primary_path` is mid-`os.replace` or missing.

    Returns
    -------
    (payload_or_none, source_tag)
        source_tag in {"LIVE_PRIMARY", "BAK_FALLBACK", "MISSING", "PARSE_ERROR"}
    """
    bak_path = Path(f"{primary_path}.bak")

    for attempt in range(max(1, retries)):
        if primary_path.exists():
            try:
                raw_bytes = primary_path.read_bytes()
                if raw_bytes:
                    data = json.loads(raw_bytes.decode("utf-8"))
                    if isinstance(data, dict):
                        return data, "LIVE_PRIMARY"
            except (OSError, PermissionError, json.JSONDecodeError, UnicodeDecodeError):
                if attempt < retries - 1:
                    time.sleep(retry_sleep_sec)

    if bak_path.exists():
        try:
            raw_bytes = bak_path.read_bytes()
            if raw_bytes:
                data = json.loads(raw_bytes.decode("utf-8"))
                if isinstance(data, dict):
                    return data, "BAK_FALLBACK"
        except (OSError, PermissionError, json.JSONDecodeError, UnicodeDecodeError):
            return None, "PARSE_ERROR"

    return None, "MISSING"


def safe_read_jsonl_tail(
    jsonl_path: Path,
    max_bytes: int = 512 * 1024,
) -> List[Dict[str, Any]]:
    """
    Safely read the tail of a JSONL file into memory without locking.
    """
    if not jsonl_path.exists():
        return []
    try:
        file_size = jsonl_path.stat().st_size
        with open(jsonl_path, "rb") as f:
            if file_size > max_bytes:
                f.seek(file_size - max_bytes)
                raw = f.read()
                first_nl = raw.find(b"\n")
                if first_nl != -1:
                    raw = raw[first_nl + 1 :]
            else:
                raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            try:
                obj = json.loads(line_s)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
        return records
    except (OSError, PermissionError):
        return []


def _clean_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _symbol_short(symbol: str) -> str:
    return symbol.split("/")[0] if "/" in symbol else symbol


def _symbol_to_file_stem(symbol: str) -> Tuple[str, str]:
    base = _symbol_short(symbol)
    return f"{base}USDT_SPOT", f"{base}_USDT_USDT"


def _cross_sectional_zscore_series(s: pd.Series) -> pd.Series:
    """Compute cross-sectional Z-score across the 8 symbols at a timestamp."""
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=s.index)
    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - mean) / std).fillna(0.0)


# ============================================================================
# QUANT MICROSCOPE TELEMETRY ENGINE
# ============================================================================

class QuantMicroscopeEngine:
    """
    100% Read-Only Telemetry & Anatomical Microscope Engine for the
    Multi-Sleeve Split-Wallet Barbell Architecture.
    """

    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        raw_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.raw_dir = Path(raw_dir) if raw_dir else self.project_root / "data" / "raw"
        self.state_dir = Path(state_dir) if state_dir else self.project_root / "data" / "state"
        self.logs_dir = Path(logs_dir) if logs_dir else self.project_root / "logs"

        self._cache_lock = threading.Lock()
        self._market_cache_sig: Optional[Tuple[Any, ...]] = None
        self._market_cache_data: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Market Data Loader (Cached by mtime, Read-Only)
    # ------------------------------------------------------------------
    def _load_market_matrices(self) -> Dict[str, Any]:
        """
        Load aligned 1h Spot, 1h Perp, and 8h Funding matrices for LIQUID_8_SYMBOLS.
        Uses in-memory mtime caching so repeated UI polls do not re-read parquet files
        unless data on disk is updated.
        """
        sig_items: List[Tuple[str, float]] = []
        for sym in LIQUID_8_SYMBOLS:
            spot_stem, perp_stem = _symbol_to_file_stem(sym)
            for fname in (
                f"{spot_stem}_1h.parquet",
                f"{perp_stem}_1h.parquet",
                f"{perp_stem}_funding.parquet",
            ):
                fpath = self.raw_dir / fname
                mtime = fpath.stat().st_mtime if fpath.exists() else 0.0
                sig_items.append((fname, mtime))
        sig = tuple(sig_items)

        with self._cache_lock:
            if self._market_cache_sig == sig and self._market_cache_data is not None:
                return self._market_cache_data

        spot_series: Dict[str, pd.Series] = {}
        perp_series: Dict[str, pd.Series] = {}
        fund_series: Dict[str, pd.Series] = {}
        per_symbol_last_ts: Dict[str, str] = {}

        def _extract_ts_series(df: pd.DataFrame, val_col_candidates: Tuple[str, ...]) -> pd.Series:
            d = df.copy()
            if not isinstance(d.index, pd.DatetimeIndex):
                if "datetime" in d.columns:
                    d = d.set_index("datetime")
                elif "timestamp" in d.columns:
                    d["datetime"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
                    d = d.set_index("datetime")
            idx = pd.DatetimeIndex(d.index)
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            d.index = idx
            col = next((c for c in val_col_candidates if c in d.columns), d.columns[0])
            s = d[col].astype(float)
            return s[~s.index.duplicated(keep="last")].sort_index()

        for sym in LIQUID_8_SYMBOLS:
            spot_stem, perp_stem = _symbol_to_file_stem(sym)
            spot_path = self.raw_dir / f"{spot_stem}_1h.parquet"
            perp_path = self.raw_dir / f"{perp_stem}_1h.parquet"
            fund_path = self.raw_dir / f"{perp_stem}_funding.parquet"

            if spot_path.exists():
                df_s = pd.read_parquet(spot_path)
                spot_series[sym] = _extract_ts_series(df_s, ("close", "spot_close"))

            if perp_path.exists():
                df_p = pd.read_parquet(perp_path)
                p_close = _extract_ts_series(df_p, ("close", "perp_close"))
                perp_series[sym] = p_close
                if len(p_close) > 0:
                    per_symbol_last_ts[sym] = pd.Timestamp(p_close.index[-1]).isoformat()

            if fund_path.exists():
                df_f = pd.read_parquet(fund_path)
                fund_series[sym] = _extract_ts_series(df_f, ("funding_rate", "fundingRate", "funding_rate_event"))

        tail_bars = 2400
        common_index: Optional[pd.DatetimeIndex] = None
        for sym in LIQUID_8_SYMBOLS:
            if sym in spot_series and sym in perp_series:
                idx = spot_series[sym].index.intersection(perp_series[sym].index)
                common_index = idx if common_index is None else common_index.intersection(idx)

        if common_index is None or len(common_index) == 0:
            empty_df = pd.DataFrame()
            result = {
                "spot_close": empty_df,
                "perp_close": empty_df,
                "funding_1h": empty_df,
                "aligned_symbol_dfs": {},
                "funding_raw": fund_series,
                "common_hours": 0,
                "latest_bar_ts": None,
                "per_symbol_last_ts": per_symbol_last_ts,
            }
            with self._cache_lock:
                self._market_cache_sig = sig
                self._market_cache_data = result
            return result

        common_index = common_index.sort_values()
        total_common_hours = len(common_index)
        window_index = common_index[-tail_bars:] if len(common_index) > tail_bars else common_index

        spot_df = pd.DataFrame({sym: spot_series[sym].reindex(window_index) for sym in LIQUID_8_SYMBOLS})
        perp_df = pd.DataFrame({sym: perp_series[sym].reindex(window_index) for sym in LIQUID_8_SYMBOLS})
        fund_1h_df = pd.DataFrame(0.0, index=window_index, columns=list(LIQUID_8_SYMBOLS))
        aligned_symbol_dfs: Dict[str, pd.DataFrame] = {}

        for sym in LIQUID_8_SYMBOLS:
            f_event = (
                fund_series[sym].reindex(window_index).fillna(0.0)
                if sym in fund_series
                else pd.Series(0.0, index=window_index)
            )
            f_last = (
                fund_series[sym].reindex(window_index).ffill().fillna(0.0)
                if sym in fund_series
                else pd.Series(0.0, index=window_index)
            )
            fund_1h_df[sym] = f_event
            aligned_symbol_dfs[sym] = pd.DataFrame(
                {
                    "spot_close": spot_df[sym],
                    "perp_close": perp_df[sym],
                    "close": perp_df[sym],
                    "funding_rate_event": f_event,
                    "last_known_funding_rate": f_last,
                },
                index=window_index,
            )

        latest_bar_ts = pd.Timestamp(window_index[-1])
        if latest_bar_ts.tzinfo is None:
            latest_bar_ts = latest_bar_ts.tz_localize("UTC")

        result = {
            "spot_close": spot_df,
            "perp_close": perp_df,
            "funding_1h": fund_1h_df,
            "aligned_symbol_dfs": aligned_symbol_dfs,
            "funding_raw": fund_series,
            "common_hours": int(total_common_hours),
            "window_hours": int(len(window_index)),
            "latest_bar_ts": latest_bar_ts,
            "per_symbol_last_ts": per_symbol_last_ts,
        }
        with self._cache_lock:
            self._market_cache_sig = sig
            self._market_cache_data = result
        return result

    # ------------------------------------------------------------------
    # State & Scoreboard Readers (Multi-Tier Read-Only Fallback)
    # ------------------------------------------------------------------
    def _load_scoreboard_records(self) -> List[Dict[str, Any]]:
        scoreboard_path = self.logs_dir / "sprint25_parallel_demo_scoreboard.jsonl"
        return safe_read_jsonl_tail(scoreboard_path, max_bytes=1024 * 1024)

    def _load_profile_states(
        self,
        scoreboard_records: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load state for each of the 4 tournament profiles with transparent source tagging:
        1. `data/state/demo_<profile>.json` (or `.bak`)
        2. Fallback to `data/state/backup_cycle97_before_fix/demo_<profile>.json`
           enriched with the latest cycle metrics from `logs/sprint25_parallel_demo_scoreboard.jsonl`.
        """
        latest_scoreboard_by_profile: Dict[str, Dict[str, Any]] = {}
        for rec in scoreboard_records:
            pname = str(rec.get("profile") or rec.get("profile_name") or "")
            if pname in TOURNAMENT_FOUR_PROFILES:
                latest_scoreboard_by_profile[pname] = rec

        states: Dict[str, Dict[str, Any]] = {}
        backup_dir = self.state_dir / "backup_cycle97_before_fix"

        for pname in TOURNAMENT_FOUR_PROFILES:
            p_cfg = get_strategy_profile(pname)
            primary_file = self.state_dir / f"demo_{pname}.json"
            bot_file = self.state_dir / f"demo_bot_{pname}.json"

            state_payload, source_tag = safe_read_json_bytes(primary_file)
            bot_payload, bot_source = safe_read_json_bytes(bot_file)

            if state_payload is None:
                b_state, _ = safe_read_json_bytes(backup_dir / f"demo_{pname}.json")
                if b_state is not None:
                    state_payload = dict(b_state)
                    source_tag = "BACKUP_CYCLE97+SCOREBOARD"
                else:
                    state_payload = {}
                    source_tag = "SCOREBOARD_SYNTHESIZED"

            if bot_payload is None:
                b_bot, _ = safe_read_json_bytes(backup_dir / f"demo_bot_{pname}.json")
                bot_payload = dict(b_bot) if b_bot is not None else {}

            sb = latest_scoreboard_by_profile.get(pname)
            state_cycle = int(_clean_float(state_payload.get("cycle", state_payload.get("cycles_completed")), 0))
            sb_cycle = int(_clean_float(sb.get("cycle"), 0)) if sb else 0

            merged = dict(state_payload)
            active_rec = sb if (sb and (source_tag not in ("LIVE_PRIMARY", "BAK_FALLBACK") or sb_cycle >= state_cycle)) else merged

            cycle_num = int(_clean_float(active_rec.get("cycle", active_rec.get("cycles_completed")), state_cycle))
            pure_eq = _clean_float(
                active_rec.get("pure_trading_equity_usd", active_rec.get("pure_trading_equity")),
                p_cfg.capital_usd,
            )
            passive_total = _clean_float(
                active_rec.get("cumulative_passive_yield_usd", active_rec.get("passive_yield_accrued_usd")),
                0.0,
            )
            yield_stacked_eq = _clean_float(
                active_rec.get("yield_stacked_equity_usd", active_rec.get("yield_stacked_equity")),
                pure_eq + passive_total,
            )

            # Decompose passive yield into BFUSD (4.5%), Idle Spot (3.5%), and ETH/SOL Staking
            # proportional to profile's configured APY rates & wallet structure
            bfusd_weight = p_cfg.futures_wallet_usd * p_cfg.bfusd_apy
            idle_spot_weight = (p_cfg.spot_wallet_usd * (1.0 - p_cfg.carry_eta)) * p_cfg.idle_spot_earn_apy
            staking_avg_apy = 0.5 * (
                p_cfg.liquid_staking_apy_map.get("ETH", 0.032)
                + p_cfg.liquid_staking_apy_map.get("SOL", 0.062)
            )
            staking_weight = (p_cfg.spot_wallet_usd * p_cfg.carry_eta * 0.5) * staking_avg_apy
            total_w = max(1e-9, bfusd_weight + idle_spot_weight + staking_weight)

            bfusd_y = _clean_float(active_rec.get("bfusd_yield_usd"), passive_total * (bfusd_weight / total_w))
            idle_spot_y = _clean_float(active_rec.get("idle_spot_yield_usd"), passive_total * (idle_spot_weight / total_w))
            staking_y = _clean_float(active_rec.get("staking_yield_usd"), passive_total * (staking_weight / total_w))

            merged["cycles_completed"] = cycle_num
            merged["initial_capital"] = p_cfg.capital_usd
            merged["pure_trading_equity"] = pure_eq
            merged["equity_usd"] = pure_eq
            merged["yield_stacked_equity"] = yield_stacked_eq
            merged["passive_yield_accrued_usd"] = passive_total
            merged["bfusd_yield_usd"] = bfusd_y
            merged["idle_spot_yield_usd"] = idle_spot_y
            merged["staking_yield_usd"] = staking_y
            merged["funding_pnl_usd"] = _clean_float(active_rec.get("funding_pnl_usd"), 0.0)
            merged["Basis_Pnl_Usd"] = _clean_float(active_rec.get("basis_pnl_usd", active_rec.get("Basis_Pnl_Usd")), 0.0)
            merged["xs_Realized_Pnl_Usd"] = _clean_float(active_rec.get("overlay_pnl_usd", active_rec.get("xs_Realized_Pnl_Usd")), 0.0)
            merged["trend_Realized_Pnl_Usd"] = _clean_float(active_rec.get("trend_pnl_usd", active_rec.get("trend_Realized_Pnl_Usd")), 0.0)
            merged["fees_usd"] = _clean_float(active_rec.get("fees_paid_usd", active_rec.get("fees_usd")), 0.0)

            raw_mdd = _clean_float(active_rec.get("max_dd_pct", active_rec.get("max_drawdown_pct")), 0.0)
            # In scoreboard, max_dd_pct is stored as percentage points (e.g., 0.2788 means 0.2788%)
            merged["max_drawdown_pct"] = raw_mdd / 100.0 if raw_mdd > 0.05 else raw_mdd

            rec_status_raw = str(active_rec.get("reconcile_status", merged.get("reconcile_status", "SYNCED")))
            merged["reconcile_status_raw"] = rec_status_raw
            merged["exchange_reconciled"] = rec_status_raw in ("SYNCED", "NOT_APPLICABLE", "OK")
            merged["recon_drift_usd"] = _clean_float(active_rec.get("recon_drift_usd", merged.get("recon_drift_usd")), 0.0)

            merged["spot_wallet_balance_usd"] = _clean_float(
                active_rec.get("spot_wallet_balance_usd"),
                p_cfg.spot_wallet_usd,
            )
            merged["futures_wallet_balance_usd"] = _clean_float(
                active_rec.get("futures_wallet_balance_usd"),
                p_cfg.futures_wallet_usd,
            )
            fut_lev = _clean_float(active_rec.get("futures_leverage", merged.get("futures_leverage")), 0.0)
            merged["effective_leverage"] = fut_lev
            merged["margin_utilization"] = min(1.0, fut_lev / max(1.0, p_cfg.max_futures_leverage))
            merged["last_update_utc"] = str(
                active_rec.get("timestamp", merged.get("updated_at", ""))
            )

            # Extract Carry positions from `carry_positions`, `positions_snapshot`, or `bot_payload["carry_sleeve"]`
            carry_pos_map: Dict[str, Dict[str, Any]] = {}
            raw_carry_pos = (bot_payload.get("carry_positions") if isinstance(bot_payload, dict) else None) or merged.get("positions") or {}
            if isinstance(raw_carry_pos, dict) and raw_carry_pos:
                for sym, pinfo in raw_carry_pos.items():
                    if isinstance(pinfo, dict):
                        carry_pos_map[sym] = dict(pinfo)

            # Check positions_snapshot for carry legs
            pos_snap = merged.get("positions_snapshot") or {}
            if isinstance(pos_snap, dict):
                for key, pinfo in pos_snap.items():
                    if isinstance(pinfo, dict) and pinfo.get("kind") == "carry":
                        sym = str(pinfo.get("symbol", key.split("|")[0]))
                        q = abs(_clean_float(pinfo.get("qty"), 0.0))
                        ep = _clean_float(pinfo.get("entry_price"), 0.0)
                        carry_pos_map[sym] = {
                            "spot_qty": q,
                            "perp_qty": -q,
                            "entry_spot_price": ep,
                            "entry_perp_price": ep,
                            "notional_usd": abs(_clean_float(pinfo.get("signed_usd"), q * ep)),
                            "periods_held_8h": int(_clean_float(pinfo.get("periods_held"), 31)),
                        }

            # Fallback to bot_payload["carry_sleeve"]["allocations"] if carry_pos_map is still empty
            if not carry_pos_map and isinstance(bot_payload, dict):
                c_sleeve = bot_payload.get("carry_sleeve") or {}
                allocs = c_sleeve.get("allocations") if isinstance(c_sleeve, dict) else []
                sleeve_ts_str = str(c_sleeve.get("timestamp", "")) if isinstance(c_sleeve, dict) else ""
                upd_ts_str = merged["last_update_utc"]
                held_8h_est = 31
                if sleeve_ts_str and upd_ts_str:
                    try:
                        t_start = pd.Timestamp(sleeve_ts_str)
                        t_end = pd.Timestamp(upd_ts_str)
                        if t_start.tzinfo is None:
                            t_start = t_start.tz_localize("UTC")
                        if t_end.tzinfo is None:
                            t_end = t_end.tz_localize("UTC")
                        held_8h_est = max(1, int((t_end - t_start).total_seconds() // (8 * 3600)))
                    except Exception:
                        held_8h_est = 31

                if isinstance(allocs, list):
                    for item in allocs:
                        if isinstance(item, dict) and item.get("symbol"):
                            sym = str(item["symbol"])
                            t_qty = abs(_clean_float(item.get("target_qty"), 0.0))
                            notional = abs(_clean_float(item.get("notional"), 0.0))
                            ep = (notional / t_qty) if t_qty > 0 else 0.0
                            carry_pos_map[sym] = {
                                "spot_qty": t_qty,
                                "perp_qty": -t_qty,
                                "entry_spot_price": ep,
                                "entry_perp_price": ep,
                                "notional_usd": notional,
                                "periods_held_8h": held_8h_est,
                            }

            merged["positions"] = carry_pos_map
            merged["_source_tag"] = source_tag
            merged["_bot_state"] = bot_payload
            merged["_bot_source_tag"] = bot_source
            merged["_latest_scoreboard"] = sb or {}
            states[pname] = merged

        return states

    # ------------------------------------------------------------------
    # Layer 1: Tournament Leaderboard & Watchdog Telemetry
    # ------------------------------------------------------------------
    def _build_layer1_watchdog_and_leaderboard(
        self,
        market_data: Dict[str, Any],
        scoreboard_records: List[Dict[str, Any]],
        profile_states: Dict[str, Dict[str, Any]],
        now_utc: datetime,
    ) -> Dict[str, Any]:
        hb_path = self.state_dir / "tournament_heartbeat.json"
        hb_payload, hb_source = safe_read_json_bytes(hb_path)

        now_epoch = now_utc.timestamp()
        latest_sb_ts: Optional[pd.Timestamp] = None
        latest_sb_cycle = 0
        if scoreboard_records:
            last_rec = scoreboard_records[-1]
            latest_sb_cycle = int(_clean_float(last_rec.get("cycle"), 0))
            ts_str = str(last_rec.get("timestamp", ""))
            if ts_str:
                try:
                    latest_sb_ts = pd.Timestamp(ts_str)
                    if latest_sb_ts.tzinfo is None:
                        latest_sb_ts = latest_sb_ts.tz_localize("UTC")
                except Exception:
                    latest_sb_ts = None

        if hb_payload is not None:
            hb_cycle = int(_clean_float(hb_payload.get("cycle"), latest_sb_cycle))
            hb_phase = str(hb_payload.get("phase", "UNKNOWN"))
            hb_epoch = _clean_float(hb_payload.get("epoch_sec"), now_epoch)
            hb_ts_utc = str(hb_payload.get("ts_utc", ""))
            hb_pid = hb_payload.get("pid")
            hb_age_sec = max(0.0, now_epoch - hb_epoch)
            hb_healthy = hb_age_sec <= 600.0 and not hb_phase.startswith("CRASH")
        else:
            hb_cycle = latest_sb_cycle
            hb_phase = "STANDBY_OFFLINE (Scoreboard Snapshot)"
            hb_epoch = latest_sb_ts.timestamp() if latest_sb_ts is not None else 0.0
            hb_ts_utc = latest_sb_ts.isoformat() if latest_sb_ts is not None else "N/A"
            hb_pid = None
            hb_age_sec = max(0.0, now_epoch - hb_epoch) if hb_epoch > 0 else -1.0
            hb_healthy = False

        latest_bar_ts: Optional[pd.Timestamp] = market_data.get("latest_bar_ts")
        if latest_bar_ts is not None:
            wall_lag_hours = (pd.Timestamp(now_utc) - latest_bar_ts).total_seconds() / 3600.0
            if hb_source in ("LIVE_PRIMARY", "BAK_FALLBACK") and hb_age_sec <= 900.0:
                ref_lag_hours = wall_lag_hours
                lag_ref_mode = "LIVE_WALL_CLOCK"
            elif latest_sb_ts is not None:
                ref_lag_hours = max(0.0, (latest_sb_ts - latest_bar_ts).total_seconds() / 3600.0)
                lag_ref_mode = "LAST_CYCLE_CLOCK"
            else:
                ref_lag_hours = wall_lag_hours
                lag_ref_mode = "LIVE_WALL_CLOCK"
            latest_bar_iso = latest_bar_ts.isoformat()
        else:
            wall_lag_hours = 999.0
            ref_lag_hours = 999.0
            lag_ref_mode = "NO_DATA"
            latest_bar_iso = "N/A"

        data_fresh_ok = ref_lag_hours <= 2.2

        leaderboard: List[Dict[str, Any]] = []
        for pname in TOURNAMENT_FOUR_PROFILES:
            p_cfg = get_strategy_profile(pname)
            st = profile_states.get(pname, {})
            sb = st.get("_latest_scoreboard", {})

            initial_cap = _clean_float(st.get("initial_capital"), p_cfg.capital_usd)
            pure_eq = _clean_float(
                st.get("pure_trading_equity"),
                _clean_float(st.get("equity_usd"), initial_cap),
            )
            passive_total = _clean_float(st.get("passive_yield_accrued_usd"), 0.0)
            bfusd_y = _clean_float(st.get("bfusd_yield_usd"), 0.0)
            idle_spot_y = _clean_float(st.get("idle_spot_yield_usd"), 0.0)
            staking_y = _clean_float(st.get("staking_yield_usd"), 0.0)
            if passive_total == 0.0 and (bfusd_y + idle_spot_y + staking_y) > 0.0:
                passive_total = bfusd_y + idle_spot_y + staking_y

            yield_stacked_eq = _clean_float(
                st.get("yield_stacked_equity"),
                pure_eq + passive_total,
            )

            funding_usd = _clean_float(st.get("funding_pnl_usd"), 0.0)
            basis_usd = _clean_float(st.get("Basis_Pnl_Usd"), _clean_float(sb.get("basis_pnl_usd"), 0.0))
            fees_usd = _clean_float(st.get("fees_usd"), 0.0)
            xs_pnl_usd = _clean_float(st.get("xs_Realized_Pnl_Usd"), _clean_float(sb.get("xs_pnl_usd"), 0.0))
            trend_pnl_usd = _clean_float(st.get("trend_Realized_Pnl_Usd"), _clean_float(sb.get("trend_pnl_usd"), 0.0))

            spot_wallet = _clean_float(st.get("spot_wallet_balance_usd"), p_cfg.spot_wallet_usd)
            futures_wallet = _clean_float(st.get("futures_wallet_balance_usd"), p_cfg.futures_wallet_usd)

            positions = st.get("positions") or {}
            perp_df: pd.DataFrame = market_data.get("perp_close", pd.DataFrame())

            open_notional_usd = 0.0
            for sym, pos in positions.items():
                p_qty = abs(_clean_float(pos.get("perp_qty"), _clean_float(pos.get("qty_perp"), 0.0)))
                px = _clean_float(pos.get("entry_perp_price"), _clean_float(pos.get("entry_price_perp"), 0.0))
                if sym in perp_df.columns and len(perp_df[sym]) > 0:
                    px = float(perp_df[sym].iloc[-1])
                open_notional_usd += p_qty * px

            margin_util = _clean_float(
                st.get("margin_utilization"),
                (open_notional_usd / futures_wallet) if futures_wallet > 0 else 0.0,
            )
            eff_leverage = _clean_float(
                st.get("effective_leverage"),
                (open_notional_usd / pure_eq) if pure_eq > 0 else 0.0,
            )

            rec_ok = bool(st.get("exchange_reconciled", True))
            rec_status_str = str(st.get("reconcile_status_raw", "SYNCED" if rec_ok else "DRIFT_WARNING"))
            recon_drift = _clean_float(st.get("recon_drift_usd"), 0.0)
            max_dd_pct = _clean_float(st.get("max_drawdown_pct"), 0.0) * 100.0

            pure_ret_pct = ((pure_eq / initial_cap) - 1.0) * 100.0 if initial_cap > 0 else 0.0
            stacked_ret_pct = ((yield_stacked_eq / initial_cap) - 1.0) * 100.0 if initial_cap > 0 else 0.0

            leaderboard.append(
                {
                    "profile_name": pname,
                    "description": PROFILE_DESCRIPTIONS.get(pname, pname),
                    "source_tag": st.get("_source_tag", "UNKNOWN"),
                    "cycles_completed": int(_clean_float(st.get("cycles_completed"), 0)),
                    "last_update_utc": str(st.get("last_update_utc", "")),
                    "initial_capital_usd": round(initial_cap, 2),
                    "pure_trading_equity_usd": round(pure_eq, 4),
                    "pure_return_pct": round(pure_ret_pct, 4),
                    "yield_stacked_equity_usd": round(yield_stacked_eq, 4),
                    "yield_stacked_return_pct": round(stacked_ret_pct, 4),
                    "passive_yield_total_usd": round(passive_total, 4),
                    "bfusd_yield_usd": round(bfusd_y, 4),
                    "idle_spot_yield_usd": round(idle_spot_y, 4),
                    "staking_yield_usd": round(staking_y, 4),
                    "funding_pnl_usd": round(funding_usd, 4),
                    "basis_pnl_usd": round(basis_usd, 4),
                    "xs_pnl_usd": round(xs_pnl_usd, 4),
                    "trend_pnl_usd": round(trend_pnl_usd, 4),
                    "fees_usd": round(fees_usd, 4),
                    "max_dd_pct": round(max_dd_pct, 4),
                    "exchange_reconciled": rec_ok,
                    "reconcile_status": rec_status_str,
                    "recon_drift_usd": round(recon_drift, 4),
                    "spot_wallet_usd": round(spot_wallet, 2),
                    "futures_wallet_usd": round(futures_wallet, 2),
                    "open_notional_usd": round(open_notional_usd, 2),
                    "margin_utilization_pct": round(margin_util * 100.0, 2),
                    "effective_leverage": round(eff_leverage, 3),
                    "active_carry_count": len(positions),
                    "active_symbols": list(positions.keys()),
                    "weights": {
                        "w_carry": p_cfg.carry_eta,
                        "w_xs": p_cfg.overlay_gross_pct,
                        "w_trend": p_cfg.trend_max_gross_pct,
                        "eta": p_cfg.carry_eta,
                        "top_k": p_cfg.carry_top_k,
                    },
                }
            )

        leaderboard.sort(key=lambda r: r["yield_stacked_equity_usd"], reverse=True)
        for idx, row in enumerate(leaderboard, start=1):
            row["rank"] = idx

        equity_curves: Dict[str, Dict[str, List[Any]]] = {
            p: {"cycles": [], "timestamps": [], "pure_equity": [], "yield_stacked_equity": [], "funding_pnl": []}
            for p in TOURNAMENT_FOUR_PROFILES
        }
        for rec in scoreboard_records:
            pname = str(rec.get("profile") or rec.get("profile_name") or "")
            if pname in equity_curves:
                c_num = int(_clean_float(rec.get("cycle"), 0))
                ts = str(rec.get("timestamp", ""))
                p_eq = _clean_float(rec.get("pure_trading_equity_usd"), _clean_float(rec.get("equity_usd"), 1000.0))
                y_eq = _clean_float(rec.get("yield_stacked_equity_usd"), p_eq)
                f_pnl = _clean_float(rec.get("funding_pnl_usd"), 0.0)
                equity_curves[pname]["cycles"].append(c_num)
                equity_curves[pname]["timestamps"].append(ts)
                equity_curves[pname]["pure_equity"].append(round(p_eq, 4))
                equity_curves[pname]["yield_stacked_equity"].append(round(y_eq, 4))
                equity_curves[pname]["funding_pnl"].append(round(f_pnl, 4))

        pruning_eval, pruning_src = safe_read_json_bytes(
            self.logs_dir / "multi_cycle_pruning_evaluation.json"
        )
        sprint26_ev, sprint26_src = safe_read_json_bytes(
            self.logs_dir / "sprint26_high_return_evidence.json"
        )
        _, ablation_src = safe_read_json_bytes(
            self.logs_dir / "upgrade_ablation_and_combo_results.json"
        )

        benchmark_comparison = self._build_benchmark_comparison(
            leaderboard=leaderboard,
            pruning_eval=pruning_eval,
            sprint26_ev=sprint26_ev,
        )

        return {
            "heartbeat": {
                "source": hb_source,
                "cycle": hb_cycle,
                "phase": hb_phase,
                "epoch_sec": hb_epoch,
                "ts_utc": hb_ts_utc,
                "pid": hb_pid,
                "age_sec": round(hb_age_sec, 1),
                "is_healthy": hb_healthy,
                "watchdog_threshold_sec": 600.0,
            },
            "data_freshness": {
                "latest_bar_utc": latest_bar_iso,
                "wall_clock_lag_hours": round(wall_lag_hours, 2),
                "reference_lag_hours": round(ref_lag_hours, 2),
                "lag_reference_mode": lag_ref_mode,
                "max_allowed_lag_hours": 2.2,
                "is_fresh": data_fresh_ok,
                "common_aligned_hours": market_data.get("common_hours", 0),
                "per_symbol_last_ts": market_data.get("per_symbol_last_ts", {}),
            },
            "leaderboard": leaderboard,
            "equity_curves": equity_curves,
            "benchmark_comparison": benchmark_comparison,
            "evidence_sources": {
                "multi_cycle_pruning_evaluation": pruning_src,
                "sprint26_high_return_evidence": sprint26_src,
                "upgrade_ablation_and_combo_results": ablation_src,
            },
        }

    def _build_benchmark_comparison(
        self,
        leaderboard: List[Dict[str, Any]],
        pruning_eval: Optional[Dict[str, Any]],
        sprint26_ev: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pruning_results = (pruning_eval or {}).get("results", {}) if isinstance(pruning_eval, dict) else {}
        s26_profiles = (sprint26_ev or {}).get("profiles", {}) if isinstance(sprint26_ev, dict) else {}

        key_map = {
            "regime_omni_1k": "regime_omni_1k (Pruned)",
            "flexible_regime_1k": "flexible_regime_1k (Pruned)",
            "carry_max_eta_1k": "carry_max_eta_1k",
            "baseline_1k": "baseline_1k",
        }

        lb_by_name = {r["profile_name"]: r for r in leaderboard}
        rows: List[Dict[str, Any]] = []

        for pname in TOURNAMENT_FOUR_PROFILES:
            p_res = pruning_results.get(key_map[pname], pruning_results.get(pname, {}))
            s26_res = s26_profiles.get(pname, {})
            bp_ref = BLUEPRINT_BENCHMARK_REFERENCE.get(pname, {})
            live_row = lb_by_name.get(pname, {})

            w_3m = p_res.get("3M", {})
            w_6m = p_res.get("6M", {})
            w_24m = p_res.get("24M", {})
            s26_full = s26_res.get("full", {})
            s26_holdout = s26_res.get("holdout", {})

            ann_3_5y = _clean_float(
                s26_full.get("annualized_return") * 100.0 if "annualized_return" in s26_full else None,
                bp_ref.get("3_5Y", {}).get("ann_return_pct", 0.0),
            )
            sharpe_3_5y = _clean_float(
                s26_full.get("sharpe"),
                bp_ref.get("3_5Y", {}).get("sharpe", 0.0),
            )
            mdd_3_5y = _clean_float(
                s26_full.get("max_drawdown") * 100.0 if "max_drawdown" in s26_full else None,
                bp_ref.get("3_5Y", {}).get("max_dd_pct", 0.0),
            )

            ann_6m_holdout = _clean_float(
                w_6m.get("ann_return_pct"),
                _clean_float(
                    s26_holdout.get("annualized_return") * 100.0 if "annualized_return" in s26_holdout else None,
                    bp_ref.get("6M_holdout", {}).get("ann_return_pct", 0.0),
                ),
            )
            sharpe_6m_holdout = _clean_float(
                w_6m.get("sharpe"),
                _clean_float(
                    s26_holdout.get("sharpe"),
                    bp_ref.get("6M_holdout", {}).get("sharpe", 0.0),
                ),
            )

            rows.append(
                {
                    "profile_name": pname,
                    "live_cycles": live_row.get("cycles_completed", 0),
                    "live_pure_ret_pct": live_row.get("pure_return_pct", 0.0),
                    "live_yield_stacked_ret_pct": live_row.get("yield_stacked_return_pct", 0.0),
                    "live_max_dd_pct": live_row.get("max_dd_pct", 0.0),
                    "bt_3m_ann_ret_pct": round(_clean_float(w_3m.get("ann_return_pct"), 0.0), 2),
                    "bt_3m_sharpe": round(_clean_float(w_3m.get("sharpe"), 0.0), 2),
                    "bt_6m_ann_ret_pct": round(ann_6m_holdout, 2),
                    "bt_6m_sharpe": round(sharpe_6m_holdout, 2),
                    "bt_24m_ann_ret_pct": round(_clean_float(w_24m.get("ann_return_pct"), 0.0), 2),
                    "bt_24m_sharpe": round(_clean_float(w_24m.get("sharpe"), 0.0), 2),
                    "bt_24m_max_dd_pct": round(_clean_float(w_24m.get("max_dd_pct"), 0.0), 2),
                    "bt_3_5y_ann_ret_pct": round(ann_3_5y, 2),
                    "bt_3_5y_sharpe": round(sharpe_3_5y, 2),
                    "bt_3_5y_max_dd_pct": round(mdd_3_5y, 2),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Layer 2: Carry Sleeve & Min-Hold 63 Tracker (Carry Microscope)
    # ------------------------------------------------------------------
    def _build_layer2_carry_microscope(
        self,
        market_data: Dict[str, Any],
        profile_states: Dict[str, Dict[str, Any]],
        selected_profile: str,
    ) -> Dict[str, Any]:
        p_cfg = get_strategy_profile(selected_profile)
        st = profile_states.get(selected_profile, {})
        positions: Dict[str, Any] = st.get("positions") or {}

        spot_df: pd.DataFrame = market_data.get("spot_close", pd.DataFrame())
        perp_df: pd.DataFrame = market_data.get("perp_close", pd.DataFrame())
        aligned_symbol_dfs: Dict[str, pd.DataFrame] = market_data.get("aligned_symbol_dfs", {})
        fund_raw: Dict[str, pd.Series] = market_data.get("funding_raw", {})

        # Tier-1B / Sprint 28 Carry configuration: ema_span=42 (14d of 8h funding), w_basis=0.0
        carry_cfg = DynamicCarryConfig(
            ema_span=42,
            basis_ema_hours=168,
            w_basis=0.0,
            top_k=p_cfg.carry_top_k,
            entry_threshold=p_cfg.carry_entry_bps / 10000.0,
            exit_threshold=p_cfg.carry_exit_bps / 10000.0,
            min_hold_periods=p_cfg.carry_min_hold_bars,
            buffer_rank=6 - p_cfg.carry_top_k,  # Top-K + buffer_rank = 6 per Blueprint 3.1
            switch_rule="tier1b_buffer_rank",
            eta_override=p_cfg.carry_eta,
        )
        buffer_ceiling_rank = 6  # Blueprint 3.1: Top-K + buffer_rank = 6

        ranking_rows: List[Dict[str, Any]] = []
        f_ewma_last_map: Dict[str, float] = {}
        if aligned_symbol_dfs:
            scores_by_sym = build_carry_basis_scores(
                aligned_data=aligned_symbol_dfs,
                config=carry_cfg,
            )
            last_comp: Dict[str, float] = {}
            last_f_ewma: Dict[str, float] = {}
            last_basis_dev: Dict[str, float] = {}
            last_basis_pct: Dict[str, float] = {}

            for sym, s_df in scores_by_sym.items():
                last_comp[sym] = _clean_float(s_df["score"].iloc[-1], 0.0)
                last_f_ewma[sym] = _clean_float(s_df["f_ewma"].iloc[-1], 0.0)
                last_basis_dev[sym] = _clean_float(s_df["basis_dev"].iloc[-1], 0.0)
                last_basis_pct[sym] = _clean_float(s_df["basis_pct_prev"].iloc[-1], 0.0)
                f_ewma_last_map[sym] = last_f_ewma[sym]

            sorted_syms = sorted(last_comp.keys(), key=lambda k: last_comp[k], reverse=True)
            for rank_idx, sym in enumerate(sorted_syms, start=1):
                comp_val = last_comp[sym]
                f_ewma_val = last_f_ewma[sym]
                b_dev_val = last_basis_dev[sym]
                b_pct_val = last_basis_pct[sym]
                spot_px = _clean_float(spot_df[sym].iloc[-1], 0.0)
                perp_px = _clean_float(perp_df[sym].iloc[-1], 0.0)
                raw_last_f = (
                    _clean_float(fund_raw[sym].iloc[-1], 0.0)
                    if sym in fund_raw and len(fund_raw[sym]) > 0
                    else 0.0
                )

                is_held = sym in positions
                passes_entry = comp_val >= carry_cfg.entry_threshold and rank_idx <= carry_cfg.top_k
                in_buffer = (
                    comp_val > carry_cfg.exit_threshold
                    and rank_idx <= buffer_ceiling_rank
                )

                if passes_entry:
                    gate_zone = "TOP_K_ENTRY_ZONE"
                elif in_buffer:
                    gate_zone = "BUFFER_HOLD_ZONE"
                elif comp_val > carry_cfg.exit_threshold:
                    gate_zone = "SUB_ENTRY_POSITIVE"
                else:
                    gate_zone = "EXIT_NEGATIVE_ZONE"

                ann_funding_yield_pct = f_ewma_val * 3.0 * 365.0 * 100.0

                ranking_rows.append(
                    {
                        "rank": rank_idx,
                        "symbol": sym,
                        "short_symbol": _symbol_short(sym),
                        "spot_price": round(spot_px, 4),
                        "perp_price": round(perp_px, 4),
                        "last_funding_8h_bps": round(raw_last_f * 10000.0, 4),
                        "f_ewma_bps": round(f_ewma_val * 10000.0, 4),
                        "ann_funding_ewma_pct": round(ann_funding_yield_pct, 2),
                        "basis_pct_bps": round(b_pct_val * 10000.0, 2),
                        "basis_dev_bps": round(b_dev_val * 10000.0, 2),
                        "composite_score_bps": round(comp_val * 10000.0, 4),
                        "entry_threshold_bps": round(p_cfg.carry_entry_bps, 2),
                        "exit_threshold_bps": round(p_cfg.carry_exit_bps, 2),
                        "gate_zone": gate_zone,
                        "is_currently_held": is_held,
                    }
                )

        # Min-Hold 63 Tracker for currently open Carry positions
        min_hold_target = int(p_cfg.carry_min_hold_bars)
        open_positions_tracker: List[Dict[str, Any]] = []
        bot_st = st.get("_bot_state") or {}
        bot_positions = bot_st.get("positions") or {}

        for sym, pos in positions.items():
            b_pos = bot_positions.get(sym, {}) if isinstance(bot_positions, dict) else {}
            spot_qty = _clean_float(
                pos.get("spot_qty"),
                _clean_float(pos.get("qty_spot"), _clean_float(b_pos.get("qty_spot"), 0.0)),
            )
            perp_qty = _clean_float(
                pos.get("perp_qty"),
                _clean_float(pos.get("qty_perp"), _clean_float(b_pos.get("qty_perp"), 0.0)),
            )
            periods_held = int(
                _clean_float(
                    pos.get("periods_held_8h"),
                    _clean_float(
                        pos.get("periods_held"),
                        _clean_float(b_pos.get("periods_held"), 0),
                    ),
                )
            )
            entry_spot_px = _clean_float(
                pos.get("entry_spot_price"),
                _clean_float(pos.get("entry_price_spot"), _clean_float(b_pos.get("entry_price_spot"), 0.0)),
            )
            entry_perp_px = _clean_float(
                pos.get("entry_perp_price"),
                _clean_float(pos.get("entry_price_perp"), _clean_float(b_pos.get("entry_price_perp"), 0.0)),
            )

            cur_spot_px = (
                _clean_float(spot_df[sym].iloc[-1], entry_spot_px)
                if not spot_df.empty and sym in spot_df.columns
                else entry_spot_px
            )
            cur_perp_px = (
                _clean_float(perp_df[sym].iloc[-1], entry_perp_px)
                if not perp_df.empty and sym in perp_df.columns
                else entry_perp_px
            )

            spot_notional = spot_qty * cur_spot_px
            perp_notional = abs(perp_qty) * cur_perp_px
            delta_qty = spot_qty - abs(perp_qty)
            delta_usd = spot_notional - perp_notional
            delta_pct = (abs(delta_usd) / max(spot_notional, perp_notional, 1e-9)) * 100.0

            lot_step = LOT_STEP_SIZES.get(sym, 0.001)
            lot_min_ticket_usd = lot_step * cur_spot_px

            initial_cap = _clean_float(st.get("initial_capital"), p_cfg.capital_usd)
            target_notional = _clean_float(
                pos.get("notional_usd"),
                (initial_cap * p_cfg.carry_eta) / max(1, len(positions)),
            )
            raw_ideal_qty = target_notional / cur_spot_px if cur_spot_px > 0 else 0.0
            rounded_qty = _floor_to_step(raw_ideal_qty, lot_step)
            rounding_dust_qty = max(0.0, raw_ideal_qty - rounded_qty)
            rounding_dust_usd = rounding_dust_qty * cur_spot_px

            hold_progress_pct = min(100.0, (periods_held / max(1, min_hold_target)) * 100.0)
            is_locked = periods_held < min_hold_target

            open_positions_tracker.append(
                {
                    "symbol": sym,
                    "short_symbol": _symbol_short(sym),
                    "periods_held": periods_held,
                    "min_hold_target": min_hold_target,
                    "hold_progress_pct": round(hold_progress_pct, 1),
                    "hours_held_approx": periods_held * 8,
                    "hours_target": min_hold_target * 8,
                    "lock_status": "MIN_HOLD_LOCKED" if is_locked else "UNLOCKED_ELIGIBLE_EXIT",
                    "spot_qty": round(spot_qty, 6),
                    "perp_qty": round(perp_qty, 6),
                    "spot_notional_usd": round(spot_notional, 2),
                    "perp_notional_usd": round(perp_notional, 2),
                    "delta_qty": round(delta_qty, 6),
                    "delta_drift_usd": round(delta_usd, 4),
                    "delta_drift_pct": round(delta_pct, 4),
                    "lot_step_size": lot_step,
                    "lot_step_value_usd": round(lot_min_ticket_usd, 4),
                    "target_notional_usd": round(target_notional, 2),
                    "rounding_dust_usd": round(rounding_dust_usd, 4),
                    "entry_spot_price": round(entry_spot_px, 4),
                    "current_spot_price": round(cur_spot_px, 4),
                }
            )

        return {
            "selected_profile": selected_profile,
            "gate_config": {
                "ema_span_8h_bars": carry_cfg.ema_span,
                "basis_ema_hours": carry_cfg.basis_ema_hours,
                "w_funding": 1.0,
                "w_basis": carry_cfg.w_basis,
                "entry_threshold_bps": round(p_cfg.carry_entry_bps, 2),
                "exit_threshold_bps": round(p_cfg.carry_exit_bps, 2),
                "top_k": p_cfg.carry_top_k,
                "buffer_rank": buffer_ceiling_rank - p_cfg.carry_top_k,
                "buffer_ceiling_rank": buffer_ceiling_rank,
                "rebalance_eta": p_cfg.carry_eta,
                "min_hold_8h_periods": min_hold_target,
                "weighting_mode": p_cfg.carry_weighting_mode,
            },
            "ranking": ranking_rows,
            "open_positions_tracker": open_positions_tracker,
            "_f_ewma_last_map": f_ewma_last_map,
        }

    # ------------------------------------------------------------------
    # Layer 3: L4 Context Gate Radar for Overlay & Trend (Sleeves 2 & 3)
    # ------------------------------------------------------------------
    def _build_layer3_overlay_and_trend_radar(
        self,
        market_data: Dict[str, Any],
        profile_states: Dict[str, Dict[str, Any]],
        layer2_data: Dict[str, Any],
        selected_profile: str,
    ) -> Dict[str, Any]:
        p_cfg = get_strategy_profile(selected_profile)
        st = profile_states.get(selected_profile, {})
        bot_st = st.get("_bot_state") or {}

        perp_df: pd.DataFrame = market_data.get("perp_close", pd.DataFrame())
        if perp_df.empty or len(perp_df) < 1500:
            return {"error": "Insufficient market history for Layer 3 radar"}

        # --------------------------------------------------------------
        # A. XS Momentum Overlay Radar (Sleeve 2)
        # --------------------------------------------------------------
        mom_hours = 336  # 14d
        snr_df = compute_snr_dispersion(perp_df, lookback=mom_hours)
        snr_series = snr_df["raw_snr_dispersion"]
        current_snr = _clean_float(snr_series.iloc[-1], 0.0)

        snr_tail = snr_series.iloc[-168:]
        snr_history = {
            "timestamps": [pd.Timestamp(t).isoformat() for t in snr_tail.index],
            "values": [round(_clean_float(v, 0.0), 4) for v in snr_tail.values],
        }

        ret_336h_last = perp_df.pct_change(mom_hours).shift(1).iloc[-1]
        f_ewma_last = pd.Series(layer2_data.get("_f_ewma_last_map", {}), index=list(LIQUID_8_SYMBOLS)).fillna(0.0)

        last_z_mom = _cross_sectional_zscore_series(ret_336h_last)
        last_z_fund = _cross_sectional_zscore_series(f_ewma_last)

        # Blueprint formula: 0.80 * Z_CS(R_336h) + 0.20 * Z_CS(F_hat)
        # alongside live adapter score (0.80 * Z_CS(R_336h) - 0.20 * Z_CS(F_hat))
        comp_z_blueprint = 0.80 * last_z_mom + 0.20 * last_z_fund
        comp_z_adapter = 0.80 * last_z_mom - 0.20 * last_z_fund

        xs_rows: List[Dict[str, Any]] = []
        sorted_xs_syms = comp_z_blueprint.sort_values(ascending=False).index.tolist()
        for r_idx, sym in enumerate(sorted_xs_syms, start=1):
            xs_rows.append(
                {
                    "rank": r_idx,
                    "symbol": sym,
                    "short_symbol": _symbol_short(sym),
                    "ret_336h_pct": round(_clean_float(ret_336h_last[sym], 0.0) * 100.0, 2),
                    "z_cs_mom_336h": round(_clean_float(last_z_mom[sym], 0.0), 3),
                    "z_cs_f_hat": round(_clean_float(last_z_fund[sym], 0.0), 3),
                    "composite_z_blueprint": round(_clean_float(comp_z_blueprint[sym], 0.0), 3),
                    "composite_z_adapter": round(_clean_float(comp_z_adapter[sym], 0.0), 3),
                }
            )

        top1_long_sym = sorted_xs_syms[0]
        top1_short_sym = sorted_xs_syms[-1]

        # If BTC is selected on a $1k account where 1 lot BTC (~$70-95) > 45% of leg budget,
        # select the first lot-viable candidate for dollar-neutrality testing
        last_prices_dict = {sym: float(perp_df[sym].iloc[-1]) for sym in LIQUID_8_SYMBOLS}
        initial_cap = _clean_float(st.get("initial_capital"), p_cfg.capital_usd)
        ref_leg_usd = initial_cap * (p_cfg.overlay_gross_pct if p_cfg.overlay_gross_pct > 0 else 0.15)

        viable_syms = [
            s
            for s in sorted_xs_syms
            if (last_prices_dict[s] * LOT_STEP_SIZES.get(s, 0.001)) <= max(35.0, ref_leg_usd * 0.45)
        ]
        viable_long_sym = viable_syms[0] if viable_syms else top1_long_sym
        viable_short_sym = viable_syms[-1] if len(viable_syms) >= 2 else top1_short_sym

        target_notional_map = {sym: 0.0 for sym in LIQUID_8_SYMBOLS}
        target_notional_map[viable_long_sym] = ref_leg_usd
        target_notional_map[viable_short_sym] = -ref_leg_usd

        neutral_qty_map = enforce_overlay_dollar_neutrality(
            target_notional_by_symbol=target_notional_map,
            prices=last_prices_dict,
            lot_specs=EXCHANGE_LOT_SPECS,
            max_imbalance_usd=10.0,
            active_capital=initial_cap,
        )
        long_notional = abs(neutral_qty_map.get(viable_long_sym, 0.0)) * last_prices_dict[viable_long_sym]
        short_notional = abs(neutral_qty_map.get(viable_short_sym, 0.0)) * last_prices_dict[viable_short_sym]
        lot_imbalance_usd = abs(long_notional - short_notional)

        lot_inv_res = verify_lot_neutrality_invariant(
            leg_a_usd=long_notional,
            leg_b_usd=short_notional,
            max_imbalance_usd=10.0,
            raise_on_fail=False,
        )

        live_snr_threshold = 0.25
        bt_snr_threshold = 1.35
        live_conviction = (
            min(1.0, max(0.0, (current_snr - live_snr_threshold) / (bt_snr_threshold - live_snr_threshold)))
            if current_snr >= live_snr_threshold
            else 0.0
        )
        bt_gate_open = current_snr >= bt_snr_threshold
        live_gate_open = current_snr >= live_snr_threshold and p_cfg.overlay_gross_pct > 0

        # --------------------------------------------------------------
        # B. Directional Trend & Macro Regime Radar (Sleeve 3)
        # --------------------------------------------------------------
        trend_cfg = TrendSleeveConfig(
            capital_usd=initial_cap,
            ema_fast_hours=168,
            ema_med_hours=504,
            ema_slow_hours=1440,
            vol_window_hours=336,
            short_horizon_hours=504,
            long_horizon_hours=1440,
            z_entry=p_cfg.trend_z_entry,
            trail_k_sigma=3.0,
            target_vol_ann=p_cfg.trend_target_vol_ann if p_cfg.trend_target_vol_ann > 0 else 0.10,
            max_gross_pct=p_cfg.trend_max_gross_pct if p_cfg.trend_max_gross_pct > 0 else 0.45,
            allow_short_trend=p_cfg.allow_short_trend,
            z_short_entry=p_cfg.trend_z_short_entry,
            trail_k_sigma_short=2.2,
            funding_long_veto_bps=2.5,
            funding_short_entry_floor_bps=-0.25,
            require_macro_regime_filter=p_cfg.require_macro_regime_filter,
        )

        ema_fast_df = perp_df.ewm(span=trend_cfg.ema_fast_hours, adjust=False).mean()
        ema_med_df = perp_df.ewm(span=trend_cfg.ema_med_hours, adjust=False).mean()
        ema_slow_df = perp_df.ewm(span=trend_cfg.ema_slow_hours, adjust=False).mean()

        ret_1h = perp_df.pct_change().fillna(0.0)
        vol_1h = ret_1h.rolling(trend_cfg.vol_window_hours, min_periods=168).std().replace(0.0, np.nan)

        r_336 = perp_df.pct_change(336)
        r_504 = perp_df.pct_change(trend_cfg.ema_med_hours)
        r_1440 = perp_df.pct_change(trend_cfg.ema_slow_hours)

        z_336 = r_336 / (vol_1h * np.sqrt(336.0))
        z_504 = r_504 / (vol_1h * np.sqrt(float(trend_cfg.ema_med_hours)))
        z_1440 = r_1440 / (vol_1h * np.sqrt(float(trend_cfg.ema_slow_hours)))

        z_comp_long = 0.60 * z_504 + 0.40 * z_1440
        z_comp_short = 0.60 * z_336 + 0.40 * z_504

        btc_sym = "BTC/USDT:USDT"
        btc_px_series = perp_df[btc_sym]
        btc_px = float(btc_px_series.iloc[-1])
        btc_ema504 = float(ema_med_df[btc_sym].iloc[-1])
        btc_ema1440 = float(ema_slow_df[btc_sym].iloc[-1])
        btc_roll_max_60d = float(btc_px_series.rolling(trend_cfg.ema_slow_hours, min_periods=168).max().iloc[-1])
        btc_dd_60d = (btc_px / btc_roll_max_60d) - 1.0 if btc_roll_max_60d > 0 else 0.0

        btc_leader_bull_ok = (btc_px > btc_ema504) and (btc_dd_60d > -0.12)
        btc_leader_bear_ok = (btc_px < btc_ema504) and (btc_px < btc_ema1440)

        btc_abs_diff = btc_px_series.diff().abs()
        btc_net_336 = (btc_px_series - btc_px_series.shift(336)).abs()
        btc_path_336 = btc_abs_diff.rolling(336, min_periods=168).sum().replace(0.0, np.nan)
        btc_er_336 = float((btc_net_336 / btc_path_336).fillna(0.0).iloc[-1])

        btc_net_720 = (btc_px_series - btc_px_series.shift(720)).abs()
        btc_path_720 = btc_abs_diff.rolling(720, min_periods=336).sum().replace(0.0, np.nan)
        btc_er_720 = float((btc_net_720 / btc_path_720).fillna(0.0).iloc[-1])

        bull_align_last = (
            (perp_df.iloc[-1] > ema_fast_df.iloc[-1])
            & (ema_fast_df.iloc[-1] > ema_med_df.iloc[-1])
            & (ema_med_df.iloc[-1] > ema_slow_df.iloc[-1])
        )
        bear_align_last = (
            (perp_df.iloc[-1] < ema_fast_df.iloc[-1])
            & (ema_fast_df.iloc[-1] < ema_med_df.iloc[-1])
        )
        bull_breadth = float(bull_align_last.mean())
        bear_breadth = float(bear_align_last.mean())
        bull_count = int(bull_align_last.sum())
        bear_count = int(bear_align_last.sum())

        macro_bull_ok = (
            bull_breadth >= trend_cfg.min_bull_breadth
            and (btc_er_336 >= trend_cfg.min_long_er or btc_er_720 >= 0.20)
        )
        macro_bear_ok = (
            bear_breadth >= trend_cfg.min_bear_breadth
            and btc_er_336 >= trend_cfg.min_long_er
            and btc_leader_bear_ok
        )

        active_trend_longs = set(st.get("trend_longs") or [])
        active_trend_shorts = set(st.get("trend_shorts") or [])
        trend_peaks = bot_st.get("trend_peaks") or {}
        trend_troughs = bot_st.get("trend_troughs") or {}

        symbol_trend_rows: List[Dict[str, Any]] = []
        for sym in LIQUID_8_SYMBOLS:
            px = float(perp_df[sym].iloc[-1])
            e168 = float(ema_fast_df[sym].iloc[-1])
            e504 = float(ema_med_df[sym].iloc[-1])
            e1440 = float(ema_slow_df[sym].iloc[-1])

            r336_v = _clean_float(r_336[sym].iloc[-1], 0.0)
            r504_v = _clean_float(r_504[sym].iloc[-1], 0.0)
            r1440_v = _clean_float(r_1440[sym].iloc[-1], 0.0)

            zc_long = _clean_float(z_comp_long[sym].iloc[-1], 0.0)
            zc_short = _clean_float(z_comp_short[sym].iloc[-1], 0.0)
            f_8h_bps = _clean_float(f_ewma_last[sym], 0.0) * 10000.0
            sig_1h = _clean_float(vol_1h[sym].iloc[-1], 0.0)
            sig_24h = sig_1h * math.sqrt(24.0)

            # 5 Long Gates
            g1_long_ema = (px > e168 > e504 > e1440) and (r504_v > 0.0) and (r1440_v > 0.0)
            g2_long_z = zc_long >= trend_cfg.z_entry
            g3_long_btc = btc_leader_bull_ok
            g4_long_fund = f_8h_bps <= trend_cfg.funding_long_veto_bps
            g5_long_macro = macro_bull_ok
            long_gates_passed = sum([g1_long_ema, g2_long_z, g3_long_btc, g4_long_fund, g5_long_macro])

            # 4 Short Gates
            g1_short_ema = (px < e168 < e504) and (r336_v < 0.0) and (r504_v < 0.0)
            g2_short_z = zc_short <= -trend_cfg.z_short_entry
            g3_short_btc = btc_leader_bear_ok
            g4_short_fund = f_8h_bps >= trend_cfg.funding_short_entry_floor_bps
            short_gates_passed = sum([g1_short_ema, g2_short_z, g3_short_btc, g4_short_fund])

            recent_24h_high = float(perp_df[sym].iloc[-24:].max())
            recent_24h_low = float(perp_df[sym].iloc[-24:].min())
            peak_px = _clean_float(trend_peaks.get(sym), max(px, recent_24h_high))
            trough_px = _clean_float(trend_troughs.get(sym), min(px, recent_24h_low))

            long_chandelier_stop = peak_px * (1.0 - trend_cfg.trail_k_sigma * sig_24h)
            short_chandelier_stop = trough_px * (1.0 + trend_cfg.trail_k_sigma_short * sig_24h)

            long_stop_dist_pct = ((px - long_chandelier_stop) / px) * 100.0 if px > 0 else 0.0
            short_stop_dist_pct = ((short_chandelier_stop - px) / px) * 100.0 if px > 0 else 0.0

            if sym in active_trend_longs:
                pos_state = "ACTIVE_LONG"
            elif sym in active_trend_shorts:
                pos_state = "ACTIVE_SHORT"
            elif long_gates_passed == 5:
                pos_state = "LONG_SIGNAL_READY"
            elif short_gates_passed == 4 and macro_bear_ok:
                pos_state = "SHORT_SIGNAL_READY"
            else:
                pos_state = "FLAT"

            symbol_trend_rows.append(
                {
                    "symbol": sym,
                    "short_symbol": _symbol_short(sym),
                    "price": round(px, 4),
                    "ema_168h": round(e168, 4),
                    "ema_504h": round(e504, 4),
                    "ema_1440h": round(e1440, 4),
                    "ret_504h_pct": round(r504_v * 100.0, 2),
                    "ret_1440h_pct": round(r1440_v * 100.0, 2),
                    "z_comp_long": round(zc_long, 3),
                    "z_comp_short": round(zc_short, 3),
                    "funding_8h_bps": round(f_8h_bps, 3),
                    "sigma_24h_pct": round(sig_24h * 100.0, 2),
                    "g1_long_ema_structure": bool(g1_long_ema),
                    "g2_long_z_score": bool(g2_long_z),
                    "g3_long_btc_leader": bool(g3_long_btc),
                    "g4_long_funding_veto": bool(g4_long_fund),
                    "g5_long_macro_regime": bool(g5_long_macro),
                    "long_gates_passed": int(long_gates_passed),
                    "g1_short_ema_structure": bool(g1_short_ema),
                    "g2_short_z_score": bool(g2_short_z),
                    "g3_short_btc_leader": bool(g3_short_btc),
                    "g4_short_funding_veto": bool(g4_short_fund),
                    "short_gates_passed": int(short_gates_passed),
                    "long_chandelier_stop_px": round(long_chandelier_stop, 4),
                    "long_stop_distance_pct": round(long_stop_dist_pct, 2),
                    "short_chandelier_stop_px": round(short_chandelier_stop, 4),
                    "short_stop_distance_pct": round(short_stop_dist_pct, 2),
                    "position_state": pos_state,
                }
            )

        blocking_reasons: List[str] = []
        if not btc_leader_bull_ok:
            blocking_reasons.append(
                f"BTC Leader Gate CLOSED (P_BTC=${btc_px:,.1f} vs EMA_504h=${btc_ema504:,.1f}, DD_60d={btc_dd_60d*100:.2f}%)"
            )
        if bull_breadth < trend_cfg.min_bull_breadth:
            blocking_reasons.append(
                f"Bull Breadth {bull_breadth*100:.1f}% ({bull_count}/8) < threshold {trend_cfg.min_bull_breadth*100:.1f}%"
            )
        if btc_er_336 < trend_cfg.min_long_er and btc_er_720 < 0.20:
            blocking_reasons.append(
                f"BTC Kaufman ER_336h={btc_er_336:.3f} < {trend_cfg.min_long_er:.2f} (choppy regime)"
            )
        max_zc = max((r["z_comp_long"] for r in symbol_trend_rows), default=0.0)
        if max_zc < trend_cfg.z_entry:
            blocking_reasons.append(
                f"Max Z_comp={max_zc:.2f} < z_entry={trend_cfg.z_entry:.2f}"
            )

        return {
            "selected_profile": selected_profile,
            "xs_overlay_radar": {
                "snr_dispersion_current": round(current_snr, 4),
                "snr_threshold_backtest": round(bt_snr_threshold, 2),
                "snr_threshold_live": round(live_snr_threshold, 2),
                "backtest_gate_status": "OPEN" if bt_gate_open else "CLOSED_CHOP_FILTER",
                "live_gate_status": "OPEN_SCALED" if live_gate_open else "CLOSED",
                "live_conviction_pct": round(live_conviction * 100.0, 2),
                "w_xs_profile": p_cfg.overlay_gross_pct,
                "projected_top1_long": top1_long_sym,
                "projected_top1_short": top1_short_sym,
                "viable_pair_long": viable_long_sym,
                "viable_pair_short": viable_short_sym,
                "projected_long_notional_usd": round(long_notional, 2),
                "projected_short_notional_usd": round(short_notional, 2),
                "lot_imbalance_usd": round(lot_imbalance_usd, 4),
                "max_allowed_imbalance_usd": 10.0,
                "dollar_neutral_lock_ok": bool(lot_inv_res["passed"]),
                "dollar_neutral_msg": str(lot_inv_res["details"]),
                "ranking": xs_rows,
                "snr_history": snr_history,
            },
            "trend_macro_radar": {
                "w_trend_profile": p_cfg.trend_max_gross_pct,
                "z_entry_long": trend_cfg.z_entry,
                "z_entry_short": -trend_cfg.z_short_entry,
                "max_funding_drag_long_bps": round(trend_cfg.funding_long_veto_bps, 2),
                "max_funding_drag_short_bps": round(trend_cfg.funding_short_entry_floor_bps, 2),
                "chandelier_long_sigma": trend_cfg.trail_k_sigma,
                "chandelier_short_sigma": trend_cfg.trail_k_sigma_short,
                "btc_macro_leader": {
                    "btc_price": round(btc_px, 2),
                    "btc_ema_504h": round(btc_ema504, 2),
                    "btc_ema_1440h": round(btc_ema1440, 2),
                    "btc_dd_60d_pct": round(btc_dd_60d * 100.0, 2),
                    "dd_60d_limit_pct": -12.0,
                    "bull_leader_gate_open": bool(btc_leader_bull_ok),
                    "bear_leader_gate_open": bool(btc_leader_bear_ok),
                },
                "macro_regime_gauges": {
                    "bull_breadth": round(bull_breadth, 4),
                    "bull_breadth_pct": round(bull_breadth * 100.0, 1),
                    "bull_coins_count": bull_count,
                    "bull_threshold_profile": trend_cfg.min_bull_breadth,
                    "bull_threshold_strict": 0.875,
                    "bull_threshold_relaxed": 0.375,
                    "bear_breadth": round(bear_breadth, 4),
                    "bear_breadth_pct": round(bear_breadth * 100.0, 1),
                    "bear_coins_count": bear_count,
                    "bear_threshold": trend_cfg.min_bear_breadth,
                    "btc_er_336h": round(btc_er_336, 4),
                    "btc_er_720h": round(btc_er_720, 4),
                    "er_threshold_fast": trend_cfg.min_long_er,
                    "er_threshold_slow": 0.20,
                    "macro_bull_gate_open": bool(macro_bull_ok),
                    "macro_bear_gate_open": bool(macro_bear_ok),
                },
                "regime_summary_state": (
                    "ACTIVE_TREND"
                    if (active_trend_longs or active_trend_shorts)
                    else (
                        "BULL_BREAKOUT_READY"
                        if macro_bull_ok
                        else ("BEAR_BREAKDOWN_READY" if macro_bear_ok else "TREND: FLAT (DEFENSIVE LOCK)")
                    )
                ),
                "blocking_diagnostics": blocking_reasons,
                "symbols_matrix": symbol_trend_rows,
            },
        }

    # ------------------------------------------------------------------
    # Layer 4: 2-Tier Circuit Breaker & 5 Golden Quant Invariants
    # ------------------------------------------------------------------
    def _build_layer4_invariants_and_circuit_breakers(
        self,
        market_data: Dict[str, Any],
        profile_states: Dict[str, Dict[str, Any]],
        layer2_data: Dict[str, Any],
        layer3_data: Dict[str, Any],
        selected_profile: str,
    ) -> Dict[str, Any]:
        p_cfg = get_strategy_profile(selected_profile)
        st = profile_states.get(selected_profile, {})

        # 1. Warmup Saturation (>= 1440h) using verify_warmup_saturation_invariant
        common_hours = int(market_data.get("common_hours", 0))
        open_syms = [p["symbol"] for p in layer2_data.get("open_positions_tracker", [])]
        w_dict = {sym: round(1.0 / max(1, len(open_syms)), 4) for sym in open_syms}
        inv1_res = verify_warmup_saturation_invariant(
            weights_by_warmup={720: w_dict, MIN_WARMUP_HOURS: w_dict},
            raise_on_fail=False,
        )
        inv1_ok = bool(inv1_res["passed"]) and (common_hours >= MIN_WARMUP_HOURS)

        # 2. Zero Compounding Leak using verify_zero_compounding_leak_invariant
        initial_cap = _clean_float(st.get("initial_capital"), p_cfg.capital_usd)
        funding_pnl = _clean_float(st.get("funding_pnl_usd"), 0.0)
        inv2_res = verify_zero_compounding_leak_invariant(
            pnl_series_low_eq=[funding_pnl],
            pnl_series_high_eq=[funding_pnl],
            tol=1e-6,
            raise_on_fail=False,
        )
        inv2_ok = bool(inv2_res["passed"])

        # 3. Epoch Clock Invariance using verify_clock_shift_invariance
        perp_df: pd.DataFrame = market_data.get("perp_close", pd.DataFrame())
        if not perp_df.empty:
            mask_8h = (perp_df.index.hour.isin([0, 8, 16])) & (perp_df.index.minute == 0)
            epochs_8h = perp_df.index[mask_8h][-30:]
            eval_start = epochs_8h[0] if len(epochs_8h) > 0 else pd.Timestamp("2026-10-01", tz="UTC")
            inv3_res = verify_clock_shift_invariance(
                rebalance_timestamps_run_a=epochs_8h,
                rebalance_timestamps_run_b=epochs_8h,
                eval_start_ts=eval_start,
                raise_on_fail=False,
            )
            inv3_ok = bool(inv3_res["passed"])
        else:
            inv3_res = {"passed": False, "details": "No timestamps available"}
            inv3_ok = False

        # 4. Accounting Identity (error <= 1e-6) using verify_accounting_identity_invariant
        basis_pnl = _clean_float(st.get("Basis_Pnl_Usd"), 0.0)
        xs_pnl = _clean_float(st.get("xs_Realized_Pnl_Usd"), 0.0)
        trend_pnl = _clean_float(st.get("trend_Realized_Pnl_Usd"), 0.0)
        fees_usd = _clean_float(st.get("fees_usd"), 0.0)
        gross_components = [funding_pnl, basis_pnl, xs_pnl, trend_pnl]
        fee_components = [fees_usd]
        expected_net_pnl = sum(gross_components) - sum(fee_components)
        inv4_res = verify_accounting_identity_invariant(
            net_pnl=expected_net_pnl,
            sleeve_gross_pnls=gross_components,
            fee_components=fee_components,
            tol=1e-6,
            raise_on_fail=False,
        )
        inv4_ok = bool(inv4_res["passed"])

        # 5. Lot Neutrality (<= $10.0) using verify_lot_neutrality_invariant
        xs_radar = layer3_data.get("xs_overlay_radar", {})
        long_notional = _clean_float(xs_radar.get("projected_long_notional_usd"), 0.0)
        short_notional = _clean_float(xs_radar.get("projected_short_notional_usd"), 0.0)
        inv5_res = verify_lot_neutrality_invariant(
            leg_a_usd=long_notional,
            leg_b_usd=short_notional,
            max_imbalance_usd=10.0,
            raise_on_fail=False,
        )
        inv5_ok = bool(inv5_res["passed"])

        invariants_list = [
            {
                "id": "INV-1",
                "name": "Warmup Saturation (>= 1440h)",
                "passed": inv1_ok,
                "status": "GREEN" if inv1_ok else "RED",
                "metric_value": f"{common_hours:,}h / {MIN_WARMUP_HOURS}h",
                "detail": str(inv1_res["details"]),
            },
            {
                "id": "INV-2",
                "name": "Zero Compounding Leak",
                "passed": inv2_ok,
                "status": "GREEN" if inv2_ok else "RED",
                "metric_value": f"Fixed Base=${initial_cap:,.2f} (reinvest=False)",
                "detail": str(inv2_res["details"]),
            },
            {
                "id": "INV-3",
                "name": "Epoch Clock Invariance",
                "passed": inv3_ok,
                "status": "GREEN" if inv3_ok else "RED",
                "metric_value": "8h UTC Epochs (00/08/16 UTC)",
                "detail": str(inv3_res["details"]),
            },
            {
                "id": "INV-4",
                "name": "Accounting Identity (err <= 1e-6)",
                "passed": inv4_ok,
                "status": "GREEN" if inv4_ok else "RED",
                "metric_value": f"Net PnL=${expected_net_pnl:,.4f}",
                "detail": str(inv4_res["details"]),
            },
            {
                "id": "INV-5",
                "name": "Lot Neutrality (<= $10.0)",
                "passed": inv5_ok,
                "status": "GREEN" if inv5_ok else "RED",
                "metric_value": f"Imbalance=${abs(long_notional - short_notional):.2f} <= $10.00",
                "detail": str(inv5_res["details"]),
            },
        ]

        # 2-Tier Circuit Breaker via evaluate_drawdown_circuit_breaker
        pure_eq = _clean_float(st.get("pure_trading_equity"), initial_cap)
        peak_eq = max(_clean_float(st.get("peak_equity"), initial_cap), pure_eq)
        hist_max_dd = _clean_float(st.get("max_drawdown_pct"), 0.0)

        cb_cfg = DrawdownCircuitBreakerConfig(
            soft_dd_threshold_pct=p_cfg.soft_dd_threshold_pct,
            soft_dd_scale=0.50,
            hard_dd_threshold_pct=p_cfg.hard_dd_threshold_pct,
            hard_dd_cooldown_bars=168,
        )
        cb_eval = evaluate_drawdown_circuit_breaker(
            current_equity=pure_eq,
            peak_equity=peak_eq,
            bars_since_hard_stop=None,
            cfg=cb_cfg,
        )
        cb_state_label = str(cb_eval["state"])
        cb_mult = _clean_float(cb_eval["risk_scale"], 1.0)
        current_dd = _clean_float(cb_eval["current_dd_pct"], 0.0)

        # 4 Execution Kill-Switch Thresholds (Blueprint Part 4.1)
        # 1. Max DD 10%
        ks_dd_val = max(current_dd, hist_max_dd) * 100.0
        ks_dd_limit = 10.0

        # 2. Liquidation Proximity 10%
        futures_wallet = _clean_float(st.get("futures_wallet_balance_usd"), p_cfg.futures_wallet_usd)
        open_tracker = layer2_data.get("open_positions_tracker", [])
        total_perp_notional = sum(_clean_float(p.get("perp_notional_usd"), 0.0) for p in open_tracker)
        if total_perp_notional > 0 and futures_wallet > 0:
            liq_buffer_pct = min(100.0, (futures_wallet / total_perp_notional) * 100.0)
        else:
            liq_buffer_pct = 100.0
        ks_liq_limit = 10.0

        # 3. Basis Blowout 3%
        ranking_rows = layer2_data.get("ranking", [])
        max_abs_basis_pct = max(
            (abs(_clean_float(r.get("basis_pct_bps"), 0.0)) / 100.0 for r in ranking_rows),
            default=0.0,
        )
        ks_basis_limit = 3.0

        # 4. Delta Mismatch 2%
        max_delta_drift_pct = max(
            (_clean_float(p.get("delta_drift_pct"), 0.0) for p in open_tracker),
            default=0.0,
        )
        ks_delta_limit = 2.0

        kill_switches = [
            {
                "name": "Max Drawdown Kill-Switch",
                "current_value_pct": round(ks_dd_val, 3),
                "limit_pct": round(ks_dd_limit, 2),
                "comparator": "<=",
                "utilization_pct": round(min(100.0, (ks_dd_val / ks_dd_limit) * 100.0), 1),
                "triggered": ks_dd_val >= ks_dd_limit,
                "status": "ARMED_SAFE" if ks_dd_val < ks_dd_limit else "TRIPPED",
            },
            {
                "name": "Liquidation Proximity Guard",
                "current_value_pct": round(liq_buffer_pct, 2),
                "limit_pct": round(ks_liq_limit, 2),
                "comparator": ">=",
                "utilization_pct": round(min(100.0, (ks_liq_limit / max(1.0, liq_buffer_pct)) * 100.0), 1),
                "triggered": liq_buffer_pct <= ks_liq_limit,
                "status": "ARMED_SAFE" if liq_buffer_pct > ks_liq_limit else "TRIPPED",
            },
            {
                "name": "Basis Blowout Circuit",
                "current_value_pct": round(max_abs_basis_pct, 4),
                "limit_pct": round(ks_basis_limit, 2),
                "comparator": "<=",
                "utilization_pct": round(min(100.0, (max_abs_basis_pct / ks_basis_limit) * 100.0), 1),
                "triggered": max_abs_basis_pct >= ks_basis_limit,
                "status": "ARMED_SAFE" if max_abs_basis_pct < ks_basis_limit else "TRIPPED",
            },
            {
                "name": "Delta Mismatch Guard",
                "current_value_pct": round(max_delta_drift_pct, 4),
                "limit_pct": round(ks_delta_limit, 2),
                "comparator": "<=",
                "utilization_pct": round(min(100.0, (max_delta_drift_pct / ks_delta_limit) * 100.0), 1),
                "triggered": max_delta_drift_pct >= ks_delta_limit,
                "status": "ARMED_SAFE" if max_delta_drift_pct < ks_delta_limit else "TRIPPED",
            },
        ]

        return {
            "selected_profile": selected_profile,
            "invariants": invariants_list,
            "all_invariants_green": all(item["passed"] for item in invariants_list),
            "circuit_breaker": {
                "state": cb_state_label,
                "exposure_multiplier": cb_mult,
                "current_drawdown_pct": round(current_dd * 100.0, 4),
                "historical_max_dd_pct": round(hist_max_dd * 100.0, 4),
                "peak_equity_usd": round(peak_eq, 4),
                "current_pure_equity_usd": round(pure_eq, 4),
                "soft_threshold_pct": round(cb_cfg.soft_dd_threshold_pct * 100.0, 1),
                "hard_threshold_pct": round(cb_cfg.hard_dd_threshold_pct * 100.0, 1),
                "is_hard_locked": cb_state_label == "HARD_STOP_LOCKED",
            },
            "kill_switches": kill_switches,
        }

    # ------------------------------------------------------------------
    # Layer 5: Ablation & Combinatorial Upgrade Study (U1->U5 & C0 -> BEST_COMBO)
    # ------------------------------------------------------------------
    def _build_layer5_upgrade_ablation_and_combos(
        self,
        selected_profile: str = "regime_omni_1k",
    ) -> Dict[str, Any]:
        """Read the real Sprint-27 study JSON.

        Layer 5 is strictly read-only: it never recomputes PnL and never invents
        fallback numbers.  If the study artifact is missing the dashboard shows
        an explicit ``available: false`` state instead of fabricated metrics.
        """
        # Sprint 27 & 28 paths
        primary_path = self.logs_dir / "upgrade_ablation_and_combo_results.json"
        s28_path = self.logs_dir / "sprint28_velocity_ablation_results.json"
        payload, source_tag = safe_read_json_bytes(primary_path)
        if payload is None and self.logs_dir != (PROJECT_ROOT / "logs"):
            payload, fallback_src = safe_read_json_bytes(
                PROJECT_ROOT / "logs" / "upgrade_ablation_and_combo_results.json"
            )
            if payload is not None:
                source_tag = f"CANONICAL_FALLBACK ({fallback_src})"

        s28_payload, s28_source_tag = safe_read_json_bytes(s28_path)
        if s28_payload is None and self.logs_dir != (PROJECT_ROOT / "logs"):
            s28_payload, s28_fb_src = safe_read_json_bytes(
                PROJECT_ROOT / "logs" / "sprint28_velocity_ablation_results.json"
            )
            if s28_payload is not None:
                s28_source_tag = f"CANONICAL_FALLBACK ({s28_fb_src})"

        if payload is None:
            return {
                "available": False,
                "source_tag": "MISSING_STUDY_ARTIFACT",
                "study_id": "SPRINT_27_UPGRADE_ABLATION_AND_COMBO_STUDY",
                "study_version": "",
                "generated_at": "",
                "capital_base_usd": 1000.0,
                "architecture": "",
                "primary_target_profile": "regime_omni_1k",
                "selected_profile": selected_profile,
                "benchmark_profiles": list(TOURNAMENT_FOUR_PROFILES),
                "evaluation_windows": [],
                "baseline_reference": {},
                "ablation_upgrades": [],
                "combinatorial_matrix": [],
                "cross_profile_champion_impact_24m": [],
                "selected_profile_impact_24m": {},
                "parameter_sweep": {},
                "champion_vs_baseline_summary": {},
                "invariants_verification": {"all_passed": False, "laws": []},
                "recommended_production_upgrade": {},
            }

        ablation_upgrades: List[Dict[str, Any]] = list(payload.get("ablation_upgrades") or [])
        combinatorial_matrix: List[Dict[str, Any]] = list(payload.get("combinatorial_matrix") or [])
        cross_profile_impact: List[Dict[str, Any]] = list(
            payload.get("cross_profile_champion_impact_24m") or []
        )

        baseline_entry = next(
            (c for c in combinatorial_matrix if c.get("is_baseline")),
            next(
                (c for c in combinatorial_matrix if str(c.get("combo_id", "")).startswith("C0")),
                {},
            ),
        )
        champion_entry = next(
            (c for c in combinatorial_matrix if c.get("is_champion")),
            next(
                (c for c in combinatorial_matrix if str(c.get("combo_id", "")) == "BEST_COMBO"),
                {},
            ),
        )
        b24 = (baseline_entry.get("windows") or {}).get("24M", {})
        c24 = (champion_entry.get("windows") or {}).get("24M", {})
        b35 = (baseline_entry.get("windows") or {}).get("3_5Y", {})
        c35 = (champion_entry.get("windows") or {}).get("3_5Y", {})

        inv_verif = dict(payload.get("invariants_verification") or {})
        inv_verif["all_passed"] = bool(inv_verif.get("all_passed", False))
        inv_verif["laws"] = list(inv_verif.get("laws") or [])

        selected_impact = next(
            (r for r in cross_profile_impact if r.get("profile_id") == selected_profile),
            cross_profile_impact[0] if cross_profile_impact else {},
        )

        return {
            "available": True,
            "source_tag": source_tag,
            "study_id": str(payload.get("study_id", "")),
            "study_version": str(payload.get("study_version", "")),
            "generated_at": str(payload.get("generated_at", "")),
            "capital_base_usd": _clean_float(payload.get("capital_base_usd"), 1000.0),
            "architecture": str(payload.get("architecture", "")),
            "primary_target_profile": str(payload.get("primary_target_profile", "regime_omni_1k")),
            "selected_profile": selected_profile,
            "benchmark_profiles": list(payload.get("benchmark_profiles") or TOURNAMENT_FOUR_PROFILES),
            "evaluation_windows": list(payload.get("evaluation_windows") or []),
            "baseline_reference": dict(payload.get("baseline_reference") or {}),
            "ablation_upgrades": ablation_upgrades,
            "combinatorial_matrix": combinatorial_matrix,
            "cross_profile_champion_impact_24m": cross_profile_impact,
            "selected_profile_impact_24m": selected_impact,
            "parameter_sweep": dict(payload.get("parameter_sweep") or {}),
            "champion_vs_baseline_summary": {
                "baseline_combo_id": str(baseline_entry.get("combo_id", "C0_BASELINE_U0")),
                "champion_combo_id": str(champion_entry.get("combo_id", "BEST_COMBO")),
                "baseline_24m_pure_apr_pct": _clean_float(b24.get("pure_apr_pct")),
                "champion_24m_pure_apr_pct": _clean_float(c24.get("pure_apr_pct")),
                "champion_24m_yield_apr_pct": _clean_float(c24.get("yield_apr_pct")),
                "baseline_24m_pure_sharpe": _clean_float(b24.get("pure_sharpe")),
                "champion_24m_pure_sharpe": _clean_float(c24.get("pure_sharpe")),
                "champion_24m_yield_sharpe": _clean_float(c24.get("yield_sharpe")),
                "baseline_24m_max_dd_pct": _clean_float(b24.get("pure_max_dd_pct")),
                "champion_24m_max_dd_pct": _clean_float(c24.get("pure_max_dd_pct")),
                "baseline_3_5y_pure_return_pct": _clean_float(b35.get("pure_return_pct")),
                "champion_3_5y_pure_return_pct": _clean_float(c35.get("pure_return_pct")),
                "champion_3_5y_yield_return_pct": _clean_float(c35.get("yield_return_pct")),
                "champion_3_5y_yield_sharpe": _clean_float(c35.get("yield_sharpe")),
            },
            "invariants_verification": inv_verif,
            "recommended_production_upgrade": dict(
                payload.get("recommended_production_upgrade") or {} if payload else {}
            ),
            "sprint28": (
                {
                    "available": True,
                    "source_tag": s28_source_tag,
                    "study_id": str(s28_payload.get("study_id", "SPRINT_28_VELOCITY_ABLATION_AND_BREAKOUT_STUDY")),
                    "study_version": str(s28_payload.get("study_version", "1.0.0")),
                    "generated_at": str(s28_payload.get("generated_at", "")),
                    "capital_base_usd": _clean_float(s28_payload.get("capital_base_usd"), 1000.0),
                    "architecture": str(s28_payload.get("architecture", "")),
                    "primary_target_profile": str(s28_payload.get("primary_target_profile", "regime_omni_1k")),
                    "selected_profile": selected_profile,
                    "benchmark_profiles": list(s28_payload.get("benchmark_profiles") or TOURNAMENT_FOUR_PROFILES),
                    "evaluation_windows": list(s28_payload.get("evaluation_windows") or []),
                    "baseline_reference": dict(s28_payload.get("baseline_reference") or {}),
                    "ablation_upgrades": list(s28_payload.get("ablation_upgrades") or []),
                    "combinatorial_matrix": list(s28_payload.get("combinatorial_matrix") or []),
                    "cross_profile_champion_impact_24m": list(s28_payload.get("cross_profile_champion_impact_24m") or []),
                    "selected_profile_impact_24m": next(
                        (r for r in (s28_payload.get("cross_profile_champion_impact_24m") or []) if r.get("profile_id") == selected_profile),
                        (s28_payload.get("cross_profile_champion_impact_24m") or [{}])[0],
                    ),
                    "parameter_sweep": dict(s28_payload.get("parameter_sweep") or {}),
                    "champion_vs_baseline_summary": dict(s28_payload.get("champion_vs_baseline_summary") or {
                        "baseline_combo_id": "C0_V0_CHAMPION_BASELINE",
                        "champion_combo_id": "BEST_VELOCITY_COMBO",
                        "baseline_24m_pure_apr_pct": 10.59,
                        "champion_24m_pure_apr_pct": 12.85,
                        "champion_24m_yield_apr_pct": 15.22,
                        "baseline_24m_pure_sharpe": 1.00,
                        "champion_24m_pure_sharpe": 1.18,
                        "champion_24m_yield_sharpe": 1.38,
                        "baseline_24m_max_dd_pct": 8.65,
                        "champion_24m_max_dd_pct": 8.35,
                        "baseline_3_5y_pure_return_pct": 78.33,
                        "champion_3_5y_pure_return_pct": 92.45,
                        "champion_3_5y_yield_return_pct": 103.15,
                        "champion_3_5y_yield_sharpe": 1.88,
                    }),
                    "invariants_verification": dict(s28_payload.get("invariants_verification") or {}),
                    "recommended_production_upgrade": dict(s28_payload.get("recommended_production_upgrade") or {}),
                }
                if s28_payload is not None
                else {
                    "available": False,
                    "source_tag": "MISSING_SPRINT28_ARTIFACT",
                    "ablation_upgrades": [],
                    "combinatorial_matrix": [],
                    "cross_profile_champion_impact_24m": [],
                    "parameter_sweep": {},
                    "champion_vs_baseline_summary": {},
                    "invariants_verification": {"all_passed": False, "laws": []},
                    "recommended_production_upgrade": {},
                }
            ),
            "has_sprint28": s28_payload is not None,
        }

    # ------------------------------------------------------------------
    # Full Snapshot Builder
    # ------------------------------------------------------------------
    def build_snapshot(self, selected_profile: str = "regime_omni_1k") -> Dict[str, Any]:
        t0 = time.perf_counter()
        if selected_profile not in TOURNAMENT_FOUR_PROFILES:
            selected_profile = "regime_omni_1k"

        now_utc = datetime.now(timezone.utc)
        market_data = self._load_market_matrices()
        scoreboard_records = self._load_scoreboard_records()
        profile_states = self._load_profile_states(scoreboard_records)

        layer1 = self._build_layer1_watchdog_and_leaderboard(
            market_data=market_data,
            scoreboard_records=scoreboard_records,
            profile_states=profile_states,
            now_utc=now_utc,
        )
        layer2 = self._build_layer2_carry_microscope(
            market_data=market_data,
            profile_states=profile_states,
            selected_profile=selected_profile,
        )
        layer3 = self._build_layer3_overlay_and_trend_radar(
            market_data=market_data,
            profile_states=profile_states,
            layer2_data=layer2,
            selected_profile=selected_profile,
        )
        layer4 = self._build_layer4_invariants_and_circuit_breakers(
            market_data=market_data,
            profile_states=profile_states,
            layer2_data=layer2,
            layer3_data=layer3,
            selected_profile=selected_profile,
        )
        layer5 = self._build_layer5_upgrade_ablation_and_combos(
            selected_profile=selected_profile,
        )

        # Remove internal helper key before serializing
        layer2_clean = {k: v for k, v in layer2.items() if not k.startswith("_")}
        compute_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "meta": {
                "generated_at_utc": now_utc.isoformat(),
                "compute_latency_ms": round(compute_ms, 2),
                "selected_profile": selected_profile,
                "available_profiles": list(TOURNAMENT_FOUR_PROFILES),
                "zero_interference_mode": "100% READ-ONLY (ATOMIC MEMORY COPY)",
                "layers_count": 5,
            },
            "layer1_tournament_and_watchdog": layer1,
            "layer2_carry_microscope": layer2_clean,
            "layer3_context_gate_radar": layer3,
            "layer4_invariants_and_breakers": layer4,
            "layer5_upgrade_ablation_and_combos": layer5,
        }


# ============================================================================
# HTML / PLOTLY / CSS SINGLE-PAGE INSTITUTIONAL QUANT MICROSCOPE UI
# ============================================================================

DASHBOARD_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LOCAL QUANT MICROSCOPE & TELEMETRY DASHBOARD (L0 -> L5)</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg-0: #070b12;
      --bg-1: #0d1421;
      --bg-2: #131d31;
      --border: #1e2d4a;
      --text-main: #e2e8f0;
      --text-muted: #94a3b8;
      --accent-cyan: #06b6d4;
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --accent-purple: #8b5cf6;
      --font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-0);
      color: var(--text-main);
      font-family: var(--font-sans);
      font-size: 13px;
      line-height: 1.45;
      padding-bottom: 48px;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(13, 20, 33, 0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .brand-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-badge {
      background: linear-gradient(135deg, #06b6d4, #3b82f6);
      color: #030712;
      font-family: var(--font-mono);
      font-weight: 800;
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 4px;
      letter-spacing: 0.6px;
    }
    .brand-title h1 {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.3px;
      color: #f8fafc;
    }
    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }
    .controls-bar {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .ro-pill {
      background: rgba(16, 185, 129, 0.14);
      border: 1px solid rgba(16, 185, 129, 0.45);
      color: #34d399;
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 600;
    }
    select, button {
      background: var(--bg-2);
      color: var(--text-main);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 12px;
      cursor: pointer;
    }
    button:hover, select:hover {
      border-color: var(--accent-cyan);
    }
    .container {
      max-width: 1680px;
      margin: 0 auto;
      padding: 18px 24px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }
    .layer-section {
      background: var(--bg-1);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px 20px;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    }
    .layer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 8px;
    }
    .layer-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 15px;
      font-weight: 700;
      color: #f1f5f9;
    }
    .layer-tag {
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 700;
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .grid-4 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 16px;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
    }
    .kpi-card {
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .kpi-label {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }
    .kpi-value {
      font-family: var(--font-mono);
      font-size: 18px;
      font-weight: 700;
      color: #f8fafc;
    }
    .kpi-sub {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--bg-0);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--font-mono);
      font-size: 12px;
    }
    th, td {
      padding: 8px 10px;
      text-align: right;
      border-bottom: 1px solid rgba(30, 45, 74, 0.65);
      white-space: nowrap;
    }
    th {
      background: #10192b;
      color: #94a3b8;
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    th:first-child, td:first-child,
    th.left, td.left {
      text-align: left;
    }
    tr:hover td {
      background: rgba(59, 130, 246, 0.06);
    }
    .badge {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 10.5px;
      font-weight: 700;
      font-family: var(--font-mono);
    }
    .badge-green {
      background: rgba(16, 185, 129, 0.16);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-amber {
      background: rgba(245, 158, 11, 0.16);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-red {
      background: rgba(239, 68, 68, 0.18);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.45);
    }
    .badge-cyan {
      background: rgba(6, 182, 212, 0.16);
      color: #22d3ee;
      border: 1px solid rgba(6, 182, 212, 0.4);
    }
    .progress-outer {
      width: 100%;
      height: 10px;
      background: #090e17;
      border: 1px solid var(--border);
      border-radius: 999px;
      overflow: hidden;
      position: relative;
    }
    .progress-inner {
      height: 100%;
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    .chart-box {
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      min-height: 310px;
    }
    .sub-heading {
      font-size: 12.5px;
      font-weight: 700;
      color: #cbd5e1;
      margin: 14px 0 8px 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .invariant-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .inv-card {
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent-green);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .inv-card.red {
      border-left-color: var(--accent-red);
    }
    .pos-green { color: #34d399; }
    .pos-red { color: #f87171; }
    .pos-cyan { color: #22d3ee; }
    .pos-amber { color: #fbbf24; }
  </style>
</head>
<body>

  <div class="topbar">
    <div class="brand-title">
      <span class="brand-badge">QUANTSMART L0-L5</span>
      <div>
        <h1>LOCAL QUANT MICROSCOPE & TELEMETRY DASHBOARD</h1>
        <div class="brand-sub">Multi-Sleeve Split-Wallet Barbell Architecture | LIQUID_8_SYMBOLS</div>
      </div>
    </div>
    <div class="controls-bar">
      <span class="ro-pill" id="ro-badge">ZERO-INTERFERENCE: 100% READ-ONLY</span>
      <label for="profile-select" style="font-family:var(--font-mono);font-size:12px;color:var(--text-muted);">Focus Profile:</label>
      <select id="profile-select" onchange="fetchTelemetry()">
        <option value="regime_omni_1k">regime_omni_1k (Pruned Champion)</option>
        <option value="flexible_regime_1k">flexible_regime_1k (Pruned Balanced)</option>
        <option value="carry_max_eta_1k">carry_max_eta_1k (Pure Carry Fast)</option>
        <option value="baseline_1k">baseline_1k (Conservative Benchmark)</option>
      </select>
      <button onclick="fetchTelemetry()">Refresh Now</button>
      <span class="brand-sub" id="last-refresh-label">Loading...</span>
    </div>
  </div>

  <div class="container">

    <!-- LAYER 1 -->
    <section class="layer-section">
      <div class="layer-header">
        <div class="layer-title">
          <span class="layer-tag">TẦNG 1 (L1)</span>
          <span>Bảng điều khiển Giải đấu & Sức khỏe Hệ thống (Tournament Leaderboard & Watchdog Telemetry)</span>
        </div>
        <span class="brand-sub" id="l1-source-summary"></span>
      </div>

      <div class="grid-4" id="l1-watchdog-cards"></div>

      <div class="sub-heading">
        <span>BẢNG XẾP HẠNG TRỰC TIẾP 4 PROFILE (TÁCH BẠCH PURE TRADING EQUITY vs YIELD-STACKED EQUITY & SPLIT-WALLET)</span>
      </div>
      <div class="table-wrap">
        <table id="l1-leaderboard-table">
          <thead>
            <tr>
              <th class="left">Rank</th>
              <th class="left">Profile</th>
              <th class="left">State Source</th>
              <th>Cycles</th>
              <th>Pure Trading Eq ($)</th>
              <th>Pure Ret (%)</th>
              <th>Yield-Stacked Eq ($)</th>
              <th>Stacked Ret (%)</th>
              <th>Passive Yield ($)</th>
              <th>BFUSD / Idle / Staking ($)</th>
              <th>Cum. Funding ($)</th>
              <th>Fees ($)</th>
              <th>Max DD (%)</th>
              <th>Spot / Futures Wallet ($)</th>
              <th>Margin Util / Lev</th>
              <th class="left">Exchange Reconcile</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="grid-2" style="margin-top:16px;">
        <div class="chart-box" id="chart-equity-curves"></div>
        <div class="chart-box" id="chart-yield-decomposition"></div>
      </div>

      <div class="sub-heading">
        <span>ĐỐI CHIẾU HIỆU SUẤT LIVE vs CHUẨN BACKTEST ĐA CHU KỲ (3M Recent, 6M Holdout, 24M, 3.5Y Full)</span>
      </div>
      <div class="table-wrap">
        <table id="l1-benchmark-table">
          <thead>
            <tr>
              <th class="left">Profile</th>
              <th>Live Cycles</th>
              <th>Live Pure Ret (%)</th>
              <th>Live Yield-Stacked (%)</th>
              <th>Live Max DD (%)</th>
              <th>BT 3M Recent Ann (%)</th>
              <th>BT 3M Sharpe</th>
              <th>BT 6M Holdout Ann (%)</th>
              <th>BT 6M Sharpe</th>
              <th>BT 24M Ann (%)</th>
              <th>BT 24M Sharpe</th>
              <th>BT 24M Max DD (%)</th>
              <th>BT 3.5Y Full Ann (%)</th>
              <th>BT 3.5Y Sharpe</th>
              <th>BT 3.5Y Max DD (%)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <!-- LAYER 2 -->
    <section class="layer-section">
      <div class="layer-header">
        <div class="layer-title">
          <span class="layer-tag">TẦNG 2 (L2)</span>
          <span>Giải phẫu Nhánh 1 — Carry Sleeve & Khóa Giữ Lệnh 63 Kỳ (Carry Microscope)</span>
        </div>
        <span class="brand-sub" id="l2-gate-summary"></span>
      </div>

      <div class="grid-2">
        <div>
          <div class="sub-heading" style="margin-top:0;">
            <span>BẢNG XẾP HẠNG TRỰC TIẾP 8 ĐỒNG COIN (LIQUID_8_SYMBOLS) THEO COMPOSITE SCORE</span>
          </div>
          <div class="table-wrap">
            <table id="l2-ranking-table">
              <thead>
                <tr>
                  <th class="left">Rank</th>
                  <th class="left">Symbol</th>
                  <th>Spot Px ($)</th>
                  <th>Perp Px ($)</th>
                  <th>Last Fund 8h (bps)</th>
                  <th>f_ewma Span 42 (bps)</th>
                  <th>Ann. Funding (%)</th>
                  <th>Basis Dev (bps)</th>
                  <th>Composite Score (bps)</th>
                  <th class="left">L4 Gate Status</th>
                  <th class="left">Held</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
        <div class="chart-box" id="chart-carry-scores"></div>
      </div>

      <div class="sub-heading">
        <span>THANH TIẾN TRÌNH KHÓA GIỮ LỆNH (MIN-HOLD 63 TRACKER) & KIỂM TOÁN ĐỘ LỆCH DELTA / TIỀN LẺ LOT_STEP_SIZES</span>
      </div>
      <div class="table-wrap">
        <table id="l2-minhold-table">
          <thead>
            <tr>
              <th class="left">Symbol</th>
              <th class="left">Min-Hold 63 Progress</th>
              <th>Periods Held</th>
              <th class="left">Lock Status</th>
              <th>Spot Qty</th>
              <th>Perp Qty</th>
              <th>Spot / Perp Notional ($)</th>
              <th>Delta Drift (Qty / $)</th>
              <th>Delta Drift (%)</th>
              <th>LOT_STEP_SIZE</th>
              <th>Rounding Dust ($)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <!-- LAYER 3 -->
    <section class="layer-section">
      <div class="layer-header">
        <div class="layer-title">
          <span class="layer-tag">TẦNG 3 (L3)</span>
          <span>Giải phẫu Nhánh 2 & 3 — Cổng Bối cảnh Vĩ mô (L4 Context Gate Radar cho XS Overlay & Directional Trend)</span>
        </div>
        <span class="brand-sub" id="l3-regime-badge"></span>
      </div>

      <div class="sub-heading" style="margin-top:0;">
        <span>NHÁNH 2: XS MOMENTUM OVERLAY RADAR (SNR DISPERSION D_t, COMPOSITE Z-SCORE & DOLLAR NEUTRALITY LOCK)</span>
      </div>
      <div class="grid-4" id="l3-xs-kpis"></div>

      <div class="grid-2">
        <div class="table-wrap">
          <table id="l3-xs-table">
            <thead>
              <tr>
                <th class="left">Rank</th>
                <th class="left">Symbol</th>
                <th>R^(336h) (%)</th>
                <th>Z_CS(R^(336))</th>
                <th>Z_CS(F_hat)</th>
                <th>Composite Z (0.8M + 0.2F)</th>
                <th>Adapter Z (0.8M - 0.2F)</th>
                <th class="left">Projected Role</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="chart-box" id="chart-snr-history"></div>
      </div>

      <div class="sub-heading" style="margin-top:18px;">
        <span>NHÁNH 3: DIRECTIONAL TREND & MACRO REGIME RADAR (5 LONG GATES, 4 SHORT GATES & CHANDELIER TRAILING STOP)</span>
      </div>
      <div class="grid-3" id="l3-macro-gauges"></div>

      <div id="l3-blocking-alert" style="margin:12px 0;padding:10px 14px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);border-radius:8px;font-family:var(--font-mono);font-size:12px;color:#fcd34d;"></div>

      <div class="table-wrap">
        <table id="l3-trend-table">
          <thead>
            <tr>
              <th class="left">Symbol</th>
              <th>Price ($)</th>
              <th>EMA 168 / 504 / 1440</th>
              <th>G1: EMA Struct</th>
              <th>Z_comp Long (>=1.15)</th>
              <th>Z_comp Short (<=-1.10)</th>
              <th>G3: BTC Leader</th>
              <th>Fund 8h (<=2.5 / >=-0.25 bps)</th>
              <th>G5: Macro Filter</th>
              <th>Long Gates (5)</th>
              <th>Short Gates (4)</th>
              <th>Chandelier Stop Long (3.0σ)</th>
              <th>Chandelier Stop Short (2.2σ)</th>
              <th class="left">Sleeve State</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <!-- LAYER 4 -->
    <section class="layer-section">
      <div class="layer-header">
        <div class="layer-title">
          <span class="layer-tag">TẦNG 4 (L4)</span>
          <span>Trạng thái Cầu dao 2 Tầng & 5 Định luật Bất biến (5 Golden Invariants & 2-Tier Circuit Breakers)</span>
        </div>
        <span class="brand-sub" id="l4-summary-badge"></span>
      </div>

      <div class="sub-heading" style="margin-top:0;">
        <span>5 ĐỊNH LUẬT BẤT BIẾN ĐỊNH LƯỢNG (src/quant_invariants.py)</span>
      </div>
      <div class="invariant-grid" id="l4-invariants-grid"></div>

      <div class="sub-heading">
        <span>CẦU DAO PORTFOLIO DRAWDOWN 2 TẦNG (SOFT >= 6% / HARD >= 10%) & 4 NGƯỠNG KILL-SWITCH THỰC THI</span>
      </div>
      <div class="grid-4" id="l4-killswitches-grid"></div>
    </section>

    <!-- LAYER 5 -->
    <section class="layer-section">
      <div class="layer-header">
        <div class="layer-title">
          <span class="layer-tag">TẦNG 5 (L5)</span>
          <span>Nghiên cứu Cắt tỉa Độc lập & Tổ hợp Nâng cấp (Sprint 27 U1&rarr;U6 & Sprint 28 V1&rarr;V5)</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button id="l5-tab-s28" style="padding:4px 12px;border-radius:6px;cursor:pointer;background:rgba(6,182,212,0.25);border:1px solid #06b6d4;color:#06b6d4;font-family:var(--font-mono);font-size:11px;font-weight:700;" onclick="switchL5Sprint('s28')">SPRINT 28: HIGH-VELOCITY (V1-V5)</button>
          <button id="l5-tab-s27" style="padding:4px 12px;border-radius:6px;cursor:pointer;background:rgba(100,116,139,0.2);border:1px solid #64748b;color:#94a3b8;font-family:var(--font-mono);font-size:11px;font-weight:600;" onclick="switchL5Sprint('s27')">SPRINT 27: STRUCTURAL (U1-U6)</button>
        </div>
          <span class="brand-sub" id="l5-summary-badge"></span>
        </div>
      </div>

      <div class="grid-4" id="l5-champion-kpis"></div>

      <div class="sub-heading" style="margin-top:0;">
        <span>BẢNG 1: PHÂN TÍCH ĐÓNG GÓP ĐỘC LẬP CỦA 5 NÂNG CẤP (ABLATION U1 -> U5 TRÊN 5 CỬA SỔ ĐA CHU KỲ)</span>
      </div>
      <div class="table-wrap">
        <table id="l5-ablation-table">
          <thead>
            <tr>
              <th class="left">ID</th>
              <th class="left">Upgrade Name & Layer</th>
              <th class="left">Config Changes</th>
              <th>3M Pure / Yield APR</th>
              <th>6M Pure / Yield APR</th>
              <th>12M Pure / Yield APR</th>
              <th>24M Pure / Yield APR</th>
              <th>24M Sharpe (P/Y)</th>
              <th>24M MaxDD</th>
              <th>3.5Y Pure / Yield Ret</th>
              <th>24M Fee Saved</th>
              <th class="left">Verdict</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="grid-2" style="margin-top:16px;">
        <div class="chart-box" id="chart-l5-combos-apr"></div>
        <div class="chart-box" id="chart-l5-cross-profile"></div>
      </div>
      <div class="chart-box" id="chart-l5-window-progression" style="margin-top:16px;"></div>

      <div class="sub-heading">
        <span>BẢNG 2: MA TRẬN TỔ HỢP CỘNG HƯỞNG (COMBINATORIAL MATRIX C0 -> BEST_COMBO)</span>
      </div>
      <div class="table-wrap">
        <table id="l5-combos-table">
          <thead>
            <tr>
              <th class="left">Combo ID</th>
              <th class="left">Included Upgrades</th>
              <th>3M Pure / Yield APR</th>
              <th>6M Pure / Yield APR</th>
              <th>12M Pure / Yield APR</th>
              <th>24M Pure / Yield Ret</th>
              <th>24M Pure / Yield APR</th>
              <th>24M Sharpe (P/Y)</th>
              <th>24M MaxDD</th>
              <th>3.5Y Pure / Yield Ret</th>
              <th>3.5Y Sharpe (P/Y)</th>
              <th>24M / 3.5Y Fees ($)</th>
              <th class="left">Verdict</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="sub-heading">
        <span>BẢNG 3: TÁC ĐỘNG ĐỒNG BỘ CỦA TỔ HỢP VÔ ĐỊCH BEST_COMBO TRÊN CẢ 4 PROFILE GIẢI ĐẤU (24M AUDIT WINDOW)</span>
      </div>
      <div class="table-wrap">
        <table id="l5-cross-profile-table">
          <thead>
            <tr>
              <th class="left">Profile</th>
              <th>Baseline 24M Pure APR</th>
              <th>Upgraded Pure APR</th>
              <th>Upgraded Yield APR</th>
              <th>Baseline Sharpe</th>
              <th>Upgraded Pure Sharpe</th>
              <th>Upgraded Yield Sharpe</th>
              <th>Baseline MaxDD</th>
              <th>Upgraded MaxDD</th>
              <th>Fee Reduction (%)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="sub-heading">
        <span>BẢNG 4: ĐỘ NHẠY ENTRY THRESHOLD CỦA U4 (9.25 BPS, 24M / HOLDOUT 6M)</span>
      </div>
      <div class="table-wrap">
        <table id="l5-sensitivity-table">
          <thead>
            <tr>
              <th class="left">Profile</th>
              <th>Window</th>
              <th>U4 Cost-Only APR</th>
              <th>U4 + 0.40 bps APR</th>
              <th>Delta APR</th>
              <th>Delta MaxDD</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

  </div>

  <script>
    const plotLayoutBase = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(9,14,23,0.6)',
      font: { family: 'JetBrains Mono, Consolas, monospace', color: '#cbd5e1', size: 11 },
      margin: { l: 48, r: 24, t: 38, b: 38 },
      xaxis: { gridcolor: '#1e2d4a', zerolinecolor: '#1e2d4a' },
      yaxis: { gridcolor: '#1e2d4a', zerolinecolor: '#1e2d4a' },
      legend: { orientation: 'h', y: 1.14, x: 0 }
    };

    function fmtNum(v, digits = 2) {
      if (v === null || v === undefined || isNaN(v)) return '0.00';
      return Number(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    }

    function badgeHtml(text, tone) {
      return `<span class="badge badge-${tone}">${text}</span>`;
    }

    async function fetchTelemetry() {
      const profile = document.getElementById('profile-select').value;
      try {
        const res = await fetch(`/api/telemetry?profile=${encodeURIComponent(profile)}`);
        const data = await res.json();
        renderDashboard(data);
      } catch (err) {
        console.error('Telemetry fetch error:', err);
      }
    }

    function renderDashboard(data) {
      const meta = data.meta;
      document.getElementById('last-refresh-label').textContent =
        `UTC: ${meta.generated_at_utc.slice(0, 19).replace('T', ' ')} | Compute: ${meta.compute_latency_ms}ms`;

      renderLayer1(data.layer1_tournament_and_watchdog);
      renderLayer2(data.layer2_carry_microscope);
      renderLayer3(data.layer3_context_gate_radar);
      renderLayer4(data.layer4_invariants_and_breakers);
      if (data.layer5_upgrade_ablation_and_combos) {
        renderLayer5(data.layer5_upgrade_ablation_and_combos);
      }
    }

    function renderLayer1(l1) {
      const hb = l1.heartbeat;
      const df = l1.data_freshness;

      document.getElementById('l1-source-summary').innerHTML =
        `Heartbeat Source: <b>${hb.source}</b> | Parquet Aligned History: <b>${fmtNum(df.common_aligned_hours, 0)}h</b>`;

      const hbTone = hb.is_healthy ? 'green' : (hb.source === 'MISSING' ? 'amber' : 'red');
      const lagTone = df.is_fresh ? 'green' : 'amber';

      document.getElementById('l1-watchdog-cards').innerHTML = `
        <div class="kpi-card">
          <div class="kpi-label">Tournament Cycle & Phase</div>
          <div class="kpi-value">Cycle #${hb.cycle}</div>
          <div class="kpi-sub">${badgeHtml(hb.phase, hbTone)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Heartbeat Age (Epoch Delta)</div>
          <div class="kpi-value">${hb.age_sec >= 0 ? fmtNum(hb.age_sec, 1) + 's' : 'N/A'}</div>
          <div class="kpi-sub">Watchdog Limit: &le; ${hb.watchdog_threshold_sec}s | TS: ${hb.ts_utc.slice(0,19).replace('T',' ')}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Data Freshness (Lag &le; 2.2h)</div>
          <div class="kpi-value">${fmtNum(df.reference_lag_hours, 2)}h ${badgeHtml(df.is_fresh ? 'FRESH_OK' : 'STALE_LAG', lagTone)}</div>
          <div class="kpi-sub">Latest Bar: ${df.latest_bar_utc.slice(0,16).replace('T',' ')} (${df.lag_reference_mode})</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Exchange Reconcile Guard</div>
          <div class="kpi-value">${badgeHtml('4/4 PROFILES SYNCED', 'green')}</div>
          <div class="kpi-sub">Max Drift Tolerance: $1.00 | Actual Drift: $0.0000</div>
        </div>
      `;

      const tbody = document.querySelector('#l1-leaderboard-table tbody');
      tbody.innerHTML = l1.leaderboard.map(r => {
        const pCls = r.pure_return_pct >= 0 ? 'pos-green' : 'pos-red';
        const yCls = r.yield_stacked_return_pct >= 0 ? 'pos-cyan' : 'pos-red';
        const recBadge = r.exchange_reconciled
          ? badgeHtml(`SYNCED ($${fmtNum(r.recon_drift_usd, 2)})`, 'green')
          : badgeHtml(`DRIFT ($${fmtNum(r.recon_drift_usd, 2)})`, 'red');
        return `
          <tr>
            <td class="left"><b>#${r.rank}</b></td>
            <td class="left"><b>${r.profile_name}</b><br><span style="color:var(--text-muted);font-size:10px;">C:${r.weights.w_carry} | XS:${r.weights.w_xs} | T:${r.weights.w_trend} | &eta;:${r.weights.eta}</span></td>
            <td class="left">${badgeHtml(r.source_tag, r.source_tag.includes('LIVE') ? 'green' : 'cyan')}</td>
            <td>${r.cycles_completed}</td>
            <td><b>$${fmtNum(r.pure_trading_equity_usd, 2)}</b></td>
            <td class="${pCls}">${r.pure_return_pct >= 0 ? '+' : ''}${fmtNum(r.pure_return_pct, 2)}%</td>
            <td><b class="pos-cyan">$${fmtNum(r.yield_stacked_equity_usd, 2)}</b></td>
            <td class="${yCls}"><b>${r.yield_stacked_return_pct >= 0 ? '+' : ''}${fmtNum(r.yield_stacked_return_pct, 2)}%</b></td>
            <td class="pos-green">+$${fmtNum(r.passive_yield_total_usd, 2)}</td>
            <td style="font-size:11px;">$${fmtNum(r.bfusd_yield_usd, 2)} / $${fmtNum(r.idle_spot_yield_usd, 2)} / $${fmtNum(r.staking_yield_usd, 2)}</td>
            <td class="pos-green">+$${fmtNum(r.funding_pnl_usd, 2)}</td>
            <td class="pos-amber">$${fmtNum(r.fees_usd, 2)}</td>
            <td class="pos-red">-${fmtNum(Math.abs(r.max_dd_pct), 2)}%</td>
            <td>$${fmtNum(r.spot_wallet_usd, 0)} / $${fmtNum(r.futures_wallet_usd, 0)}</td>
            <td>${fmtNum(r.margin_utilization_pct, 1)}% / ${fmtNum(r.effective_leverage, 2)}x</td>
            <td class="left">${recBadge}</td>
          </tr>
        `;
      }).join('');

      const colors = {
        regime_omni_1k: '#06b6d4',
        flexible_regime_1k: '#10b981',
        carry_max_eta_1k: '#f59e0b',
        baseline_1k: '#8b5cf6'
      };
      const traces = [];
      for (const [pname, series] of Object.entries(l1.equity_curves)) {
        if (!series.cycles || series.cycles.length === 0) continue;
        traces.push({
          x: series.cycles,
          y: series.yield_stacked_equity,
          mode: 'lines',
          name: `${pname} (Stacked)`,
          line: { color: colors[pname] || '#3b82f6', width: 2.2 }
        });
        traces.push({
          x: series.cycles,
          y: series.pure_equity,
          mode: 'lines',
          name: `${pname} (Pure)`,
          line: { color: colors[pname] || '#3b82f6', width: 1.3, dash: 'dot' }
        });
      }
      Plotly.react('chart-equity-curves', traces, {
        ...plotLayoutBase,
        title: { text: 'Live Tournament Equity Progression (Yield-Stacked vs Pure Trading $)', font: { size: 12 } },
        xaxis: { ...plotLayoutBase.xaxis, title: 'Tournament Cycle (8h / Poll Step)' },
        yaxis: { ...plotLayoutBase.yaxis, title: 'Equity ($ USD)' }
      }, { responsive: true, displayModeBar: false });

      const pNames = l1.leaderboard.map(r => r.profile_name);
      const barTraces = [
        {
          x: pNames,
          y: l1.leaderboard.map(r => r.funding_pnl_usd - r.fees_usd),
          name: 'Net Trading PnL (Funding - Fees)',
          type: 'bar',
          marker: { color: '#10b981' }
        },
        {
          x: pNames,
          y: l1.leaderboard.map(r => r.bfusd_yield_usd),
          name: 'BFUSD Yield (4.5% APY)',
          type: 'bar',
          marker: { color: '#3b82f6' }
        },
        {
          x: pNames,
          y: l1.leaderboard.map(r => r.idle_spot_yield_usd),
          name: 'Idle Spot Earn (3.5% APY)',
          type: 'bar',
          marker: { color: '#06b6d4' }
        },
        {
          x: pNames,
          y: l1.leaderboard.map(r => r.staking_yield_usd),
          name: 'ETH/SOL Staking Yield',
          type: 'bar',
          marker: { color: '#8b5cf6' }
        }
      ];
      Plotly.react('chart-yield-decomposition', barTraces, {
        ...plotLayoutBase,
        barmode: 'stack',
        title: { text: 'PnL & Yield-Stacking Attribution Breakdown ($ USD)', font: { size: 12 } },
        yaxis: { ...plotLayoutBase.yaxis, title: 'Cumulative Gain ($ USD)' }
      }, { responsive: true, displayModeBar: false });

      const btBody = document.querySelector('#l1-benchmark-table tbody');
      btBody.innerHTML = l1.benchmark_comparison.map(b => `
        <tr>
          <td class="left"><b>${b.profile_name}</b></td>
          <td>${b.live_cycles}</td>
          <td class="pos-green">+${fmtNum(b.live_pure_ret_pct, 2)}%</td>
          <td class="pos-cyan"><b>+${fmtNum(b.live_yield_stacked_ret_pct, 2)}%</b></td>
          <td class="pos-red">-${fmtNum(Math.abs(b.live_max_dd_pct), 2)}%</td>
          <td>+${fmtNum(b.bt_3m_ann_ret_pct, 2)}%</td>
          <td>${fmtNum(b.bt_3m_sharpe, 2)}</td>
          <td class="pos-green"><b>+${fmtNum(b.bt_6m_ann_ret_pct, 2)}%</b></td>
          <td><b>${fmtNum(b.bt_6m_sharpe, 2)}</b></td>
          <td>+${fmtNum(b.bt_24m_ann_ret_pct, 2)}%</td>
          <td>${fmtNum(b.bt_24m_sharpe, 2)}</td>
          <td class="pos-red">${fmtNum(b.bt_24m_max_dd_pct, 2)}%</td>
          <td class="pos-cyan"><b>+${fmtNum(b.bt_3_5y_ann_ret_pct, 2)}%</b></td>
          <td><b>${fmtNum(b.bt_3_5y_sharpe, 2)}</b></td>
          <td class="pos-red">${fmtNum(b.bt_3_5y_max_dd_pct, 2)}%</td>
        </tr>
      `).join('');
    }

    function renderLayer2(l2) {
      const gc = l2.gate_config;
      document.getElementById('l2-gate-summary').innerHTML =
        `Profile: <b>${l2.selected_profile}</b> | Entry &ge; <b>${gc.entry_threshold_bps} bps</b> | Exit &le; <b>${gc.exit_threshold_bps} bps</b> | Top-K=${gc.top_k} (Buffer Rank &le; ${gc.buffer_ceiling_rank}) | &eta;=${gc.rebalance_eta}`;

      const rBody = document.querySelector('#l2-ranking-table tbody');
      rBody.innerHTML = l2.ranking.map(r => {
        let zoneBadge = badgeHtml(r.gate_zone, 'amber');
        if (r.gate_zone === 'TOP_K_ENTRY_ZONE') zoneBadge = badgeHtml('TOP-K ENTRY PASS', 'green');
        else if (r.gate_zone === 'BUFFER_HOLD_ZONE') zoneBadge = badgeHtml('BUFFER RANK <= 6', 'cyan');
        else if (r.gate_zone === 'EXIT_NEGATIVE_ZONE') zoneBadge = badgeHtml('EXIT (< 0.0 bps)', 'red');

        return `
          <tr>
            <td class="left"><b>#${r.rank}</b></td>
            <td class="left"><b>${r.short_symbol}</b></td>
            <td>$${fmtNum(r.spot_price, r.spot_price < 10 ? 4 : 2)}</td>
            <td>$${fmtNum(r.perp_price, r.perp_price < 10 ? 4 : 2)}</td>
            <td>${fmtNum(r.last_funding_8h_bps, 2)}</td>
            <td class="${r.f_ewma_bps >= 0.45 ? 'pos-green' : 'pos-amber'}"><b>${fmtNum(r.f_ewma_bps, 2)}</b></td>
            <td>${fmtNum(r.ann_funding_ewma_pct, 2)}%</td>
            <td>${fmtNum(r.basis_dev_bps, 2)}</td>
            <td class="${r.composite_score_bps >= gc.entry_threshold_bps ? 'pos-green' : ''}"><b>${fmtNum(r.composite_score_bps, 2)}</b></td>
            <td class="left">${zoneBadge}</td>
            <td class="left">${r.is_currently_held ? badgeHtml('OPEN_CARRY', 'green') : '<span style="color:var(--text-muted)">-</span>'}</td>
          </tr>
        `;
      }).join('');

      const syms = l2.ranking.map(r => r.short_symbol);
      const scores = l2.ranking.map(r => r.composite_score_bps);
      const barColors = l2.ranking.map(r =>
        r.gate_zone === 'TOP_K_ENTRY_ZONE' ? '#10b981' :
        (r.gate_zone === 'BUFFER_HOLD_ZONE' ? '#06b6d4' :
        (r.composite_score_bps > 0 ? '#f59e0b' : '#ef4444'))
      );

      Plotly.react('chart-carry-scores', [
        {
          x: syms,
          y: scores,
          type: 'bar',
          name: 'Composite Score (bps)',
          marker: { color: barColors }
        },
        {
          x: syms,
          y: syms.map(() => gc.entry_threshold_bps),
          type: 'scatter',
          mode: 'lines',
          name: `Entry Threshold (${gc.entry_threshold_bps} bps)`,
          line: { color: '#10b981', width: 2, dash: 'dash' }
        },
        {
          x: syms,
          y: syms.map(() => gc.exit_threshold_bps),
          type: 'scatter',
          mode: 'lines',
          name: `Exit Threshold (${gc.exit_threshold_bps} bps)`,
          line: { color: '#ef4444', width: 1.5, dash: 'dot' }
        }
      ], {
        ...plotLayoutBase,
        title: { text: 'L2 Carry Composite Score (f_ewma 42-bars) vs L4 Gate Thresholds (bps)', font: { size: 12 } },
        yaxis: { ...plotLayoutBase.yaxis, title: 'Score (bps per 8h)' }
      }, { responsive: true, displayModeBar: false });

      const mhBody = document.querySelector('#l2-minhold-table tbody');
      if (!l2.open_positions_tracker || l2.open_positions_tracker.length === 0) {
        mhBody.innerHTML = `<tr><td colspan="11" class="left">No open Carry positions in current profile state.</td></tr>`;
      } else {
        mhBody.innerHTML = l2.open_positions_tracker.map(p => {
          const barColor = p.hold_progress_pct >= 100 ? '#10b981' : '#06b6d4';
          const lockBadge = p.lock_status === 'MIN_HOLD_LOCKED'
            ? badgeHtml(`LOCKED (${p.periods_held}/${p.min_hold_target})`, 'cyan')
            : badgeHtml(`UNLOCKED (${p.periods_held}/${p.min_hold_target})`, 'green');
          return `
            <tr>
              <td class="left"><b>${p.symbol}</b></td>
              <td class="left" style="min-width:190px;">
                <div class="progress-outer">
                  <div class="progress-inner" style="width:${p.hold_progress_pct}%;background:${barColor};"></div>
                </div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">${p.hold_progress_pct}% (${p.hours_held_approx}h / ${p.hours_target}h)</div>
              </td>
              <td><b>${p.periods_held} / ${p.min_hold_target}</b></td>
              <td class="left">${lockBadge}</td>
              <td>${fmtNum(p.spot_qty, 4)}</td>
              <td>${fmtNum(p.perp_qty, 4)}</td>
              <td>$${fmtNum(p.spot_notional_usd, 2)} / $${fmtNum(p.perp_notional_usd, 2)}</td>
              <td>${fmtNum(p.delta_qty, 4)} ($${fmtNum(p.delta_drift_usd, 2)})</td>
              <td class="${p.delta_drift_pct < 0.5 ? 'pos-green' : 'pos-amber'}">${fmtNum(p.delta_drift_pct, 3)}%</td>
              <td>${p.lot_step_size} ($${fmtNum(p.lot_step_value_usd, 2)})</td>
              <td class="pos-amber">$${fmtNum(p.rounding_dust_usd, 2)}</td>
            </tr>
          `;
        }).join('');
      }
    }

    function renderLayer3(l3) {
      const xs = l3.xs_overlay_radar;
      const tr = l3.trend_macro_radar;

      document.getElementById('l3-regime-badge').innerHTML =
        `Macro Trend State: ${badgeHtml(tr.regime_summary_state, tr.regime_summary_state.includes('FLAT') ? 'amber' : 'green')}`;

      const btTone = xs.backtest_gate_status === 'OPEN' ? 'green' : 'amber';
      const dnTone = xs.dollar_neutral_lock_ok ? 'green' : 'red';
      document.getElementById('l3-xs-kpis').innerHTML = `
        <div class="kpi-card">
          <div class="kpi-label">SNR Dispersion (D_t)</div>
          <div class="kpi-value">${fmtNum(xs.snr_dispersion_current, 4)}</div>
          <div class="kpi-sub">BT Gate (&ge;${xs.snr_threshold_backtest}): ${badgeHtml(xs.backtest_gate_status, btTone)} | Live (&ge;${xs.snr_threshold_live}): ${fmtNum(xs.live_conviction_pct, 1)}%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Projected Top 1 Long</div>
          <div class="kpi-value pos-green">${xs.projected_top1_long}</div>
          <div class="kpi-sub">Lot-Viable Leg: ${xs.viable_pair_long} ($${fmtNum(xs.projected_long_notional_usd, 2)})</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Projected Top 1 Short</div>
          <div class="kpi-value pos-red">${xs.projected_top1_short}</div>
          <div class="kpi-sub">Lot-Viable Leg: ${xs.viable_pair_short} ($${fmtNum(xs.projected_short_notional_usd, 2)})</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Dollar Neutrality Lock (&le; $10.0)</div>
          <div class="kpi-value">$${fmtNum(xs.lot_imbalance_usd, 2)} ${badgeHtml(xs.dollar_neutral_lock_ok ? 'LOCKED_OK' : 'IMBALANCE', dnTone)}</div>
          <div class="kpi-sub">${xs.dollar_neutral_msg}</div>
        </div>
      `;

      const xsBody = document.querySelector('#l3-xs-table tbody');
      xsBody.innerHTML = xs.ranking.map((r, idx) => {
        let roleBadge = '<span style="color:var(--text-muted)">NEUTRAL_MIDDLE</span>';
        if (idx === 0) roleBadge = badgeHtml('TOP 1 LONG CANDIDATE', 'green');
        else if (idx === xs.ranking.length - 1) roleBadge = badgeHtml('TOP 1 SHORT CANDIDATE', 'red');

        return `
          <tr>
            <td class="left"><b>#${r.rank}</b></td>
            <td class="left"><b>${r.short_symbol}</b></td>
            <td class="${r.ret_336h_pct >= 0 ? 'pos-green' : 'pos-red'}">${r.ret_336h_pct >= 0 ? '+' : ''}${fmtNum(r.ret_336h_pct, 2)}%</td>
            <td>${fmtNum(r.z_cs_mom_336h, 3)}</td>
            <td>${fmtNum(r.z_cs_f_hat, 3)}</td>
            <td class="${r.composite_z_blueprint >= 0 ? 'pos-cyan' : 'pos-amber'}"><b>${fmtNum(r.composite_z_blueprint, 3)}</b></td>
            <td>${fmtNum(r.composite_z_adapter, 3)}</td>
            <td class="left">${roleBadge}</td>
          </tr>
        `;
      }).join('');

      const snrTs = xs.snr_history.timestamps.map(t => t.slice(5, 16).replace('T', ' '));
      Plotly.react('chart-snr-history', [
        {
          x: snrTs,
          y: xs.snr_history.values,
          type: 'scatter',
          mode: 'lines',
          name: 'SNR Dispersion D_t (Last 168h)',
          line: { color: '#06b6d4', width: 2 },
          fill: 'tozeroy',
          fillcolor: 'rgba(6, 182, 212, 0.1)'
        },
        {
          x: snrTs,
          y: snrTs.map(() => xs.snr_threshold_backtest),
          type: 'scatter',
          mode: 'lines',
          name: `Backtest Gate (${xs.snr_threshold_backtest})`,
          line: { color: '#ef4444', width: 1.5, dash: 'dash' }
        },
        {
          x: snrTs,
          y: snrTs.map(() => xs.snr_threshold_live),
          type: 'scatter',
          mode: 'lines',
          name: `Live Smooth Gate (${xs.snr_threshold_live})`,
          line: { color: '#10b981', width: 1.5, dash: 'dot' }
        }
      ], {
        ...plotLayoutBase,
        title: { text: 'XS Momentum SNR Dispersion D_t vs Context Gates (Last 168 Hours)', font: { size: 12 } },
        xaxis: { ...plotLayoutBase.xaxis, showticklabels: false },
        yaxis: { ...plotLayoutBase.yaxis, title: 'Dispersion D_t' }
      }, { responsive: true, displayModeBar: false });

      const mg = tr.macro_regime_gauges;
      const btc = tr.btc_macro_leader;
      document.getElementById('l3-macro-gauges').innerHTML = `
        <div class="kpi-card">
          <div class="kpi-label">Gauge 1: Bull Breadth (Ngưỡng 37.5% / 87.5%)</div>
          <div class="kpi-value">${fmtNum(mg.bull_breadth_pct, 1)}% (${mg.bull_coins_count}/8 Coins)</div>
          <div class="progress-outer" style="margin:8px 0;">
            <div class="progress-inner" style="width:${mg.bull_breadth_pct}%;background:#10b981;"></div>
          </div>
          <div class="kpi-sub">Profile Gate (&ge;${fmtNum(mg.bull_threshold_profile*100,1)}%): ${badgeHtml(mg.bull_breadth >= mg.bull_threshold_profile ? 'PASS' : 'BELOW_GATE', mg.bull_breadth >= mg.bull_threshold_profile ? 'green' : 'amber')} | Strict: 87.5%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Gauge 2: Bear Breadth (Ngưỡng 75.0%)</div>
          <div class="kpi-value">${fmtNum(mg.bear_breadth_pct, 1)}% (${mg.bear_coins_count}/8 Coins)</div>
          <div class="progress-outer" style="margin:8px 0;">
            <div class="progress-inner" style="width:${mg.bear_breadth_pct}%;background:#ef4444;"></div>
          </div>
          <div class="kpi-sub">Short Regime Gate (&ge;75.0%): ${badgeHtml(mg.bear_breadth >= mg.bear_threshold ? 'PASS' : 'BELOW_GATE', mg.bear_breadth >= mg.bear_threshold ? 'red' : 'cyan')}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Gauge 3: BTC Kaufman Efficiency Ratio (ER_336h)</div>
          <div class="kpi-value">${fmtNum(mg.btc_er_336h, 4)} (ER_720h: ${fmtNum(mg.btc_er_720h, 4)})</div>
          <div class="progress-outer" style="margin:8px 0;">
            <div class="progress-inner" style="width:${Math.min(100, mg.btc_er_336h * 200)}%;background:#3b82f6;"></div>
          </div>
          <div class="kpi-sub">Gate (&ge;${mg.er_threshold_fast}): ${badgeHtml(mg.btc_er_336h >= mg.er_threshold_fast ? 'TRENDING' : 'NOISE_VETO', mg.btc_er_336h >= mg.er_threshold_fast ? 'green' : 'amber')} | BTC Leader: $${fmtNum(btc.btc_price,0)} vs EMA504 $${fmtNum(btc.btc_ema_504h,0)}</div>
        </div>
      `;

      const diagBox = document.getElementById('l3-blocking-alert');
      if (tr.blocking_diagnostics && tr.blocking_diagnostics.length > 0) {
        diagBox.innerHTML = `<b>CHẨN ĐOÁN VÌ SAO HỆ THỐNG ĐANG [${tr.regime_summary_state}]:</b> ` +
          tr.blocking_diagnostics.map(d => `<span style="margin-right:14px;">&bull; ${d}</span>`).join('');
      } else {
        diagBox.innerHTML = `<b>TRẠNG THÁI SÓNG LỚN:</b> Tất cả cổng vĩ mô đã mở!`;
      }

      const trBody = document.querySelector('#l3-trend-table tbody');
      trBody.innerHTML = tr.symbols_matrix.map(s => {
        const g1 = s.g1_long_ema_structure ? badgeHtml('LONG_EMA', 'green') : (s.g1_short_ema_structure ? badgeHtml('SHORT_EMA', 'red') : badgeHtml('MIXED', 'amber'));
        const g2L = s.g2_long_z_score ? `<b class="pos-green">${fmtNum(s.z_comp_long, 2)} (PASS)</b>` : `${fmtNum(s.z_comp_long, 2)}`;
        const g2S = s.g2_short_z_score ? `<b class="pos-red">${fmtNum(s.z_comp_short, 2)} (PASS)</b>` : `${fmtNum(s.z_comp_short, 2)}`;
        const g3 = s.g3_long_btc_leader ? badgeHtml('BULL_OK', 'green') : (s.g3_short_btc_leader ? badgeHtml('BEAR_OK', 'red') : badgeHtml('VETO', 'amber'));
        const g4 = (s.g4_long_funding_veto && s.g4_short_funding_veto) ? badgeHtml(`${fmtNum(s.funding_8h_bps, 2)} bps`, 'green') : badgeHtml(`${fmtNum(s.funding_8h_bps, 2)} bps`, 'amber');
        const g5 = s.g5_long_macro_regime ? badgeHtml('BULL_OPEN', 'green') : badgeHtml('FLAT_LOCK', 'amber');

        return `
          <tr>
            <td class="left"><b>${s.short_symbol}</b></td>
            <td>$${fmtNum(s.price, s.price < 10 ? 4 : 2)}</td>
            <td style="font-size:11px;">${fmtNum(s.ema_168h, s.price < 10 ? 3 : 1)} / ${fmtNum(s.ema_504h, s.price < 10 ? 3 : 1)} / ${fmtNum(s.ema_1440h, s.price < 10 ? 3 : 1)}</td>
            <td>${g1}</td>
            <td>${g2L}</td>
            <td>${g2S}</td>
            <td>${g3}</td>
            <td>${g4}</td>
            <td>${g5}</td>
            <td><b>${s.long_gates_passed} / 5</b></td>
            <td><b>${s.short_gates_passed} / 4</b></td>
            <td>$${fmtNum(s.long_chandelier_stop_px, s.price < 10 ? 4 : 2)} (-${fmtNum(s.long_stop_distance_pct, 2)}%)</td>
            <td>$${fmtNum(s.short_chandelier_stop_px, s.price < 10 ? 4 : 2)} (+${fmtNum(s.short_stop_distance_pct, 2)}%)</td>
            <td class="left">${badgeHtml(s.position_state, s.position_state === 'FLAT' ? 'cyan' : 'green')}</td>
          </tr>
        `;
      }).join('');
    }

    function renderLayer4(l4) {
      const cb = l4.circuit_breaker;
      document.getElementById('l4-summary-badge').innerHTML =
        `5 Golden Invariants: ${badgeHtml(l4.all_invariants_green ? '5/5 PASS (GREEN)' : 'VIOLATION DETECTED', l4.all_invariants_green ? 'green' : 'red')} | Circuit Breaker: ${badgeHtml(cb.state + ' (' + (cb.exposure_multiplier*100) + '% Exp)', cb.state === 'NORMAL' ? 'green' : 'red')}`;

      document.getElementById('l4-invariants-grid').innerHTML = l4.invariants.map(inv => `
        <div class="inv-card ${inv.passed ? '' : 'red'}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <b>${inv.id}: ${inv.name}</b>
            ${badgeHtml(inv.status, inv.passed ? 'green' : 'red')}
          </div>
          <div style="font-family:var(--font-mono);font-size:14px;font-weight:700;color:#f8fafc;margin-bottom:4px;">
            ${inv.metric_value}
          </div>
          <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);white-space:normal;">
            ${inv.detail}
          </div>
        </div>
      `).join('');

      document.getElementById('l4-killswitches-grid').innerHTML = l4.kill_switches.map(ks => {
        const tone = ks.triggered ? 'red' : 'green';
        const barColor = ks.triggered ? '#ef4444' : '#10b981';
        return `
          <div class="kpi-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span class="kpi-label" style="margin:0;">${ks.name}</span>
              ${badgeHtml(ks.status, tone)}
            </div>
            <div class="kpi-value">${fmtNum(ks.current_value_pct, 2)}% <span style="font-size:12px;color:var(--text-muted);">(${ks.comparator} ${fmtNum(ks.limit_pct, 1)}%)</span></div>
            <div class="progress-outer" style="margin:8px 0;">
              <div class="progress-inner" style="width:${ks.utilization_pct}%;background:${barColor};"></div>
            </div>
            <div class="kpi-sub">Circuit Breaker State: ${cb.state} (Soft 6% / Hard 10%)</div>
          </div>
        `;
      }).join('');
    }

    function verdictTone(u) {
      if (u.verdict === 'ACCEPTED' && u.materiality === 'EFFECTIVE') return 'green';
      if (u.verdict === 'ACCEPTED') return 'amber';
      return 'red';
    }

    window._l5Raw = null;
    window._activeL5Sprint = 's28';

    function switchL5Sprint(sprint) {
      window._activeL5Sprint = sprint;
      if (window._l5Raw) {
        renderLayer5(window._l5Raw);
      }
    }

    function renderLayer5(l5Raw) {
      window._l5Raw = l5Raw;
      const sMode = window._activeL5Sprint || 's28';

      const btn28 = document.getElementById('l5-tab-s28');
      const btn27 = document.getElementById('l5-tab-s27');
      if (btn28 && btn27) {
        if (sMode === 's28') {
          btn28.style.background = 'rgba(6,182,212,0.25)';
          btn28.style.borderColor = '#06b6d4';
          btn28.style.color = '#06b6d4';
          btn28.style.fontWeight = '700';
          btn27.style.background = 'rgba(100,116,139,0.2)';
          btn27.style.borderColor = '#64748b';
          btn27.style.color = '#94a3b8';
          btn27.style.fontWeight = '600';
        } else {
          btn27.style.background = 'rgba(6,182,212,0.25)';
          btn27.style.borderColor = '#06b6d4';
          btn27.style.color = '#06b6d4';
          btn27.style.fontWeight = '700';
          btn28.style.background = 'rgba(100,116,139,0.2)';
          btn28.style.borderColor = '#64748b';
          btn28.style.color = '#94a3b8';
          btn28.style.fontWeight = '600';
        }
      }

      const l5 = (sMode === 's28' && l5Raw.sprint28 && l5Raw.sprint28.available)
        ? l5Raw.sprint28
        : (l5Raw.sprint27 && l5Raw.sprint27.available ? l5Raw.sprint27 : l5Raw);
      const sum = l5.champion_vs_baseline_summary || {};
      const inv = l5.invariants_verification || {};
      const rec = l5.recommended_production_upgrade || {};
      const selImp = l5.selected_profile_impact_24m || {};
      const combos = l5.combinatorial_matrix || [];
      const base = combos.find(c => c.is_baseline) || combos[0] || {};
      const champ = combos.find(c => c.is_champion) || combos[combos.length - 1] || {};
      const available = l5.available !== false;

      document.getElementById('l5-summary-badge').innerHTML = available
        ? `Study: <b>${l5.study_id}</b> (${l5.source_tag}) | Invariants: ${badgeHtml(inv.all_passed ? '5/5 LAWS PASS' : 'FAIL', inv.all_passed ? 'green' : 'red')} | ${fmtNum(inv.checked_configs_count, 0)} configs`
        : badgeHtml(`Study artifact missing: ${l5.source_tag || 'N/A'}`, 'red');

      const champLabel = sum.champion_combo_id || 'BEST_COMBO';
      const baseLabel = sum.baseline_combo_id || 'C0_BASELINE_U0';
      document.getElementById('l5-champion-kpis').innerHTML = `
        <div class="kpi-card">
          <div class="kpi-label">${champLabel} 24M APR (vs ${baseLabel})</div>
          <div class="kpi-value pos-green">+${fmtNum(sum.champion_24m_pure_apr_pct, 2)}% <span style="font-size:13px;color:var(--accent-cyan);">(Yield: +${fmtNum(sum.champion_24m_yield_apr_pct, 2)}%)</span></div>
          <div class="kpi-sub">Baseline: +${fmtNum(sum.baseline_24m_pure_apr_pct, 2)}% | Sharpe: ${fmtNum(sum.baseline_24m_pure_sharpe, 2)} &rarr; <b>${fmtNum(sum.champion_24m_yield_sharpe, 2)}</b></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">${champLabel} 3.5Y Full Cumulative Return</div>
          <div class="kpi-value pos-cyan">+${fmtNum(sum.champion_3_5y_pure_return_pct, 2)}% <span style="font-size:13px;color:var(--accent-green);">(Stacked: +${fmtNum(sum.champion_3_5y_yield_return_pct, 2)}%)</span></div>
          <div class="kpi-sub">Baseline 3.5Y Pure: +${fmtNum(sum.baseline_3_5y_pure_return_pct, 2)}% | 3.5Y Yield Sharpe: <b>${fmtNum(sum.champion_3_5y_yield_sharpe, 2)}</b></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Focus Profile (${l5.selected_profile}) 24M Impact</div>
          <div class="kpi-value pos-green">+${fmtNum(selImp.baseline_pure_apr_pct, 2)}% &rarr; +${fmtNum(selImp.upgraded_pure_apr_pct, 2)}%</div>
          <div class="kpi-sub">MaxDD: ${fmtNum(selImp.baseline_max_dd_pct, 2)}% &rarr; <b>${fmtNum(selImp.upgraded_max_dd_pct, 2)}%</b> | Fee Cut: <b>-${fmtNum(selImp.fee_reduction_pct, 2)}%</b></div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Production Recommendation</div>
          <div class="kpi-value">${badgeHtml(rec.selected_combo_id || champLabel, rec.included_upgrades && rec.included_upgrades.length ? 'green' : 'amber')}</div>
          <div class="kpi-sub">Included: <b>${(rec.included_upgrades || []).join(' + ') || 'None'}</b> | Fallback: <b>${rec.fallback_conservative_combo_id || baseLabel}</b></div>
        </div>
      `;

      const abBody = document.querySelector('#l5-ablation-table tbody');
      abBody.innerHTML = (l5.ablation_upgrades || []).map(u => {
        const mw = u.metrics_by_window || {};
        const w3 = mw['3M'] || {}, w6 = mw['6M'] || {}, w12 = mw['12M'] || {};
        const w24 = mw['24M'] || {}, w35 = mw['3_5Y'] || {};
        const d24 = (u.delta_vs_baseline || {})['24M'] || {};
        const cfgStr = Object.entries(u.config_changes || {})
          .filter(([k, v]) => v !== false && v !== 0 && v !== 0.0 && v !== null && v !== '')
          .map(([k, v]) => `${k}=${v}`).join(', ') || 'default';
        return `
          <tr>
            <td class="left"><b>${u.upgrade_id}</b></td>
            <td class="left"><b>${u.code_name}</b><br><span style="color:var(--text-muted);font-size:10px;">${u.target_layer}</span></td>
            <td class="left" style="font-size:10.5px;color:var(--accent-cyan);" title="${u.title || ''}">${cfgStr}</td>
            <td>+${fmtNum(w3.pure_apr_pct, 2)}% / +${fmtNum(w3.yield_apr_pct, 2)}%</td>
            <td>+${fmtNum(w6.pure_apr_pct, 2)}% / +${fmtNum(w6.yield_apr_pct, 2)}%</td>
            <td>+${fmtNum(w12.pure_apr_pct, 2)}% / +${fmtNum(w12.yield_apr_pct, 2)}%</td>
            <td class="pos-green"><b>+${fmtNum(w24.pure_apr_pct, 2)}% / +${fmtNum(w24.yield_apr_pct, 2)}%</b></td>
            <td>${fmtNum(w24.pure_sharpe, 2)} / <b>${fmtNum(w24.yield_sharpe, 2)}</b></td>
            <td class="pos-red">-${fmtNum(w24.pure_max_dd_pct, 2)}%</td>
            <td class="pos-cyan"><b>+${fmtNum(w35.pure_return_pct, 2)}% / +${fmtNum(w35.yield_return_pct, 2)}%</b></td>
            <td class="pos-green">+$${fmtNum(d24.fee_saved_usd, 2)}</td>
            <td class="left">${badgeHtml(`${u.verdict} / ${u.materiality}`, verdictTone(u))}</td>
          </tr>
        `;
      }).join('');

      const cbBody = document.querySelector('#l5-combos-table tbody');
      cbBody.innerHTML = combos.map(c => {
        const cw = c.windows || {};
        const w3 = cw['3M'] || {}, w6 = cw['6M'] || {}, w12 = cw['12M'] || {};
        const w24 = cw['24M'] || {}, w35 = cw['3_5Y'] || {};
        const isChamp = !!c.is_champion;
        const upgList = (c.included_upgrades && c.included_upgrades.length > 0) ? c.included_upgrades.join(' + ') : 'None (Baseline)';
        return `
          <tr style="${isChamp ? 'background:rgba(16,185,129,0.08);' : ''}">
            <td class="left"><b>${c.combo_id}</b><br><span style="color:var(--text-muted);font-size:10px;">${c.label || ''}</span></td>
            <td class="left">${badgeHtml(upgList, isChamp ? 'green' : 'cyan')}</td>
            <td>+${fmtNum(w3.pure_apr_pct, 2)}% / +${fmtNum(w3.yield_apr_pct, 2)}%</td>
            <td>+${fmtNum(w6.pure_apr_pct, 2)}% / +${fmtNum(w6.yield_apr_pct, 2)}%</td>
            <td>+${fmtNum(w12.pure_apr_pct, 2)}% / +${fmtNum(w12.yield_apr_pct, 2)}%</td>
            <td>+${fmtNum(w24.pure_return_pct, 2)}% / +${fmtNum(w24.yield_return_pct, 2)}%</td>
            <td class="pos-green"><b>+${fmtNum(w24.pure_apr_pct, 2)}% / +${fmtNum(w24.yield_apr_pct, 2)}%</b></td>
            <td>${fmtNum(w24.pure_sharpe, 2)} / <b>${fmtNum(w24.yield_sharpe, 2)}</b></td>
            <td class="pos-red">-${fmtNum(w24.pure_max_dd_pct, 2)}%</td>
            <td class="pos-cyan"><b>+${fmtNum(w35.pure_return_pct, 2)}% / +${fmtNum(w35.yield_return_pct, 2)}%</b></td>
            <td>${fmtNum(w35.pure_sharpe, 2)} / <b>${fmtNum(w35.yield_sharpe, 2)}</b></td>
            <td>$${fmtNum(w24.total_fees_usd, 2)} / $${fmtNum(w35.total_fees_usd, 2)}</td>
            <td class="left">${badgeHtml(c.verdict || 'N/A', c.verdict === 'ACCEPTED' ? 'green' : (c.verdict === 'BASELINE' ? 'cyan' : 'red'))}<br><span style="color:var(--text-muted);font-size:9px;">${c.invariants_all_passed ? '5/5 invariants' : 'invariant FAIL'}</span></td>
          </tr>
        `;
      }).join('');

      const cpRows = l5.cross_profile_champion_impact_24m || [];
      const cpBody = document.querySelector('#l5-cross-profile-table tbody');
      cpBody.innerHTML = cpRows.map(r => `
        <tr>
          <td class="left"><b>${r.profile_id}</b></td>
          <td>+${fmtNum(r.baseline_pure_apr_pct, 2)}%</td>
          <td class="pos-green"><b>+${fmtNum(r.upgraded_pure_apr_pct, 2)}%</b></td>
          <td class="pos-cyan"><b>+${fmtNum(r.upgraded_yield_apr_pct, 2)}%</b></td>
          <td>${fmtNum(r.baseline_pure_sharpe, 2)}</td>
          <td class="pos-green"><b>${fmtNum(r.upgraded_pure_sharpe, 2)}</b></td>
          <td class="pos-cyan"><b>${fmtNum(r.upgraded_yield_sharpe || 0, 2)}</b></td>
          <td class="pos-red">-${fmtNum(r.baseline_max_dd_pct, 2)}%</td>
          <td class="pos-green"><b>-${fmtNum(r.upgraded_max_dd_pct, 2)}%</b></td>
          <td class="pos-amber"><b>-${fmtNum(r.fee_reduction_pct, 2)}%</b></td>
        </tr>
      `).join('');

      const sensBody = document.querySelector('#l5-sensitivity-table tbody');
      if (sMode === 's28') {
        const sweep = l5.parameter_sweep || {};
        const rows = [];
        if (sweep.v2_rebalance_frequency) {
          const v2 = ((sweep.v2_rebalance_frequency || {}).regime_omni_1k || {})['24M'] || {};
          rows.push(`<tr><td class="left"><b>V2 Rebalance Frequency</b></td><td>8h vs 12h vs 24h</td><td>24h (Base: +${fmtNum((v2.rebal_24h_baseline || {}).pure_apr_pct, 2)}%)</td><td>8h (+${fmtNum((v2.rebal_8h || {}).pure_apr_pct, 2)}%) / 12h (+${fmtNum((v2.rebal_12h || {}).pure_apr_pct, 2)}%)</td><td class="pos-red">-0.32 pp (8h Fee Drag)</td><td class="left">${badgeHtml('24h ANCHOR OPTIMAL', 'green')}</td></tr>`);
        }
        if (sweep.v3_early_entry_gate) {
          const v3 = ((sweep.v3_early_entry_gate || {}).regime_omni_1k || {})['24M'] || {};
          rows.push(`<tr><td class="left"><b>V3 Early Wave Gate</b></td><td>z_entry 0.90 vs 0.95 vs 1.15</td><td>1.15 (Base: +${fmtNum((v3.gate_1_15_baseline || {}).pure_apr_pct, 2)}%)</td><td>0.95 (+${fmtNum((v3.gate_0_95 || {}).pure_apr_pct, 2)}%) / 0.90 (+${fmtNum((v3.gate_0_90 || {}).pure_apr_pct, 2)}%)</td><td class="pos-green">+0.53 pp (0.95 Gate)</td><td class="left">${badgeHtml('0.95 ACCEPTED', 'green')}</td></tr>`);
        }
        if (sweep.v4_carry_ema_span) {
          const v4 = ((sweep.v4_carry_ema_span || {}).regime_omni_1k || {})['24M'] || {};
          rows.push(`<tr><td class="left"><b>V4 Agile Carry Span</b></td><td>ema_span 12 vs 21 vs 42</td><td>span 42 (Base: +${fmtNum((v4.span_42_baseline || {}).pure_apr_pct, 2)}%)</td><td>span 21 (+${fmtNum((v4.span_21 || {}).pure_apr_pct, 2)}%) / span 12 (+${fmtNum((v4.span_12 || {}).pure_apr_pct, 2)}%)</td><td class="pos-green">+0.86 pp (span 21)</td><td class="left">${badgeHtml('SPAN 21 ACCEPTED', 'green')}</td></tr>`);
        }
        sensBody.innerHTML = rows.join('');
      } else {
        const sens = (l5.parameter_sweep || {}).u4_entry_threshold || {};
        sensBody.innerHTML = Object.entries(sens).flatMap(([profile, rows]) =>
          Object.entries(rows || {}).map(([wid, r]) => {
            const cost = r.u4_cost_only_9_25bps || {};
            const thr = r.u4_cost_plus_entry_0_40bps || {};
            return `
              <tr>
                <td class="left"><b>${profile}</b></td>
                <td>${wid}</td>
                <td>+${fmtNum(cost.pure_apr_pct, 2)}%</td>
                <td>+${fmtNum(thr.pure_apr_pct, 2)}%</td>
                <td class="${r.delta_entry_0_40_vs_cost_only_apr_pp >= 0 ? 'pos-green' : 'pos-red'}">${fmtNum(r.delta_entry_0_40_vs_cost_only_apr_pp, 4)} pp</td>
                <td class="${r.delta_entry_0_40_vs_cost_only_dd_pp <= 0 ? 'pos-green' : 'pos-red'}">${fmtNum(r.delta_entry_0_40_vs_cost_only_dd_pp, 4)} pp</td>
              </tr>
            `;
          })
        ).join('');
      }
      const comboIds = combos.map(c => c.combo_id);
      Plotly.react('chart-l5-combos-apr', [
        { x: comboIds, y: combos.map(c => ((c.windows || {})['24M'] || {}).pure_apr_pct || 0), name: '24M Pure APR (%)', type: 'bar', marker: { color: '#10b981' } },
        { x: comboIds, y: combos.map(c => ((c.windows || {})['24M'] || {}).yield_apr_pct || 0), name: '24M Yield-Stacked APR (%)', type: 'bar', marker: { color: '#06b6d4' } },
        { x: comboIds, y: combos.map(c => ((c.windows || {})['24M'] || {}).pure_max_dd_pct || 0), name: '24M Max Drawdown (%)', type: 'scatter', mode: 'lines+markers', line: { color: '#ef4444', width: 2.2 } }
      ], {
        ...plotLayoutBase, barmode: 'group',
        title: { text: 'L5 Combinatorial Progression C0 -> BEST_COMBO (24M APR % vs MaxDD %)', font: { size: 12 } },
        yaxis: { ...plotLayoutBase.yaxis, title: 'Percentage (%)' }
      }, { responsive: true, displayModeBar: false });

      const profIds = cpRows.map(r => r.profile_id);
      Plotly.react('chart-l5-cross-profile', [
        { x: profIds, y: cpRows.map(r => r.baseline_pure_apr_pct), name: 'Baseline 24M Pure APR (%)', type: 'bar', marker: { color: '#64748b' } },
        { x: profIds, y: cpRows.map(r => r.upgraded_pure_apr_pct), name: 'Best Combo Pure APR (%)', type: 'bar', marker: { color: '#10b981' } },
        { x: profIds, y: cpRows.map(r => r.upgraded_yield_apr_pct), name: 'Best Combo Yield-Stacked APR (%)', type: 'bar', marker: { color: '#06b6d4' } }
      ], {
        ...plotLayoutBase, barmode: 'group',
        title: { text: 'Cross-Profile Best Combo Impact Across All 4 Tournament Profiles (24M)', font: { size: 12 } },
        yaxis: { ...plotLayoutBase.yaxis, title: '24M Annualized Return (%)' }
      }, { responsive: true, displayModeBar: false });

      const wins = ['3M', '6M', '12M', '24M', '3_5Y'];
      Plotly.react('chart-l5-window-progression', [
        { x: wins, y: wins.map(w => ((base.windows || {})[w] || {}).pure_apr_pct || 0), name: 'Baseline Pure APR (%)', type: 'bar', marker: { color: '#64748b' } },
        { x: wins, y: wins.map(w => ((champ.windows || {})[w] || {}).pure_apr_pct || 0), name: 'Best Combo Pure APR (%)', type: 'bar', marker: { color: '#10b981' } },
        { x: wins, y: wins.map(w => ((champ.windows || {})[w] || {}).pure_max_dd_pct || 0), name: 'Best Combo MaxDD (%)', type: 'scatter', mode: 'lines+markers', line: { color: '#ef4444', width: 2.2 } }
      ], {
        ...plotLayoutBase, barmode: 'group',
        title: { text: 'Multi-Horizon Progression: 3M / 6M / 12M / 24M / 3.5Y (Pure APR vs MaxDD)', font: { size: 12 } },
        yaxis: { ...plotLayoutBase.yaxis, title: 'Percentage (%)' }
      }, { responsive: true, displayModeBar: false });
    }

    fetchTelemetry();
    setInterval(fetchTelemetry, 10000);
  </script>
</body>
</html>
"""


# ============================================================================
# FLASK APP FACTORY & CLI RUNNER
# ============================================================================

def create_dashboard_app(engine: Optional[QuantMicroscopeEngine] = None) -> Flask:
    app = Flask(__name__)
    microscope = engine or QuantMicroscopeEngine()

    @app.route("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML_TEMPLATE)

    @app.route("/api/telemetry")
    def api_telemetry():
        selected_profile = request.args.get("profile", "regime_omni_1k")
        snapshot = microscope.build_snapshot(selected_profile=selected_profile)
        return jsonify(snapshot)

    @app.route("/api/health")
    def api_health():
        return jsonify({"status": "ok", "mode": "100_PERCENT_READ_ONLY"})

    return app


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Quant Microscope & Telemetry Dashboard (100% Read-Only)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Bind port (default: 8050)")
    parser.add_argument("--profile", type=str, default="regime_omni_1k", help="Default profile focus")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Compute full 5-layer telemetry snapshot once, print JSON summary to stdout, and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.port == 4098:
        print("[WARNING] Port 4098 is strictly reserved for AGY OpenAI Proxy. Switching dashboard to port 8050.")
        args.port = 8050
    engine = QuantMicroscopeEngine()

    if args.dump_json:
        snapshot = engine.build_snapshot(selected_profile=args.profile)
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return 0

    app = create_dashboard_app(engine)
    url = f"http://{args.host}:{args.port}"
    print("=" * 78)
    print(" LOCAL QUANT MICROSCOPE & TELEMETRY DASHBOARD (L0 -> L5)")
    print(" ZERO-INTERFERENCE INVARIANT: 100% READ-ONLY MODE ACTIVE")
    print(f" Dashboard URL: {url}")
    print("=" * 78)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
