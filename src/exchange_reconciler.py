"""
Module doi soat trang thai danh muc giua local bot state va real exchange snapshot.
Dung cho pre-flight reconciliation truoc khi bat dat lenh testnet that.
Thiet ke hoan toan deterministic, khong side effects, khong mang/network I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SYNCED = "SYNCED"
ORPHAN_ON_EXCHANGE = "ORPHAN_ON_EXCHANGE"
PHANTOM_IN_STATE = "PHANTOM_IN_STATE"
QTY_DRIFT = "QTY_DRIFT"
EQUITY_MISMATCH = "EQUITY_MISMATCH"
MARGIN_DANGER = "MARGIN_DANGER"
BLOCKED = "BLOCKED"

# Sprint 21: canh bao tai san la tren Mainnet (vi the Perp / Spot ngoai danh muc).
UNTRACKED_MAINNET_ASSET_WARNING = "UNTRACKED_MAINNET_ASSET_WARNING"
# Sprint 21: canh bao vi the Perp ngoai danh muc tren Testnet (khong chan, nhung KHONG
# duoc im lang tra ve SYNCED - de nguoi van hanh thay ro rui ro delta).
UNTRACKED_TESTNET_PERP_WARNING = "UNTRACKED_TESTNET_PERP_POSITION"

RECONCILE_STATUSES: tuple[str, ...] = (
    SYNCED,
    ORPHAN_ON_EXCHANGE,
    PHANTOM_IN_STATE,
    QTY_DRIFT,
    EQUITY_MISMATCH,
    MARGIN_DANGER,
    BLOCKED,
)

CASH_ASSETS: frozenset[str] = frozenset(
    {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD"}
)


def _is_cash_asset_local(asset: str) -> bool:
    """Sprint 22: True neu asset nam trong CASH_ASSETS (khop chinh xac, khong dung endswith)."""
    a = str(asset or "").strip().upper()
    return a in CASH_ASSETS


DEFAULT_TESTNET_BASELINE_PATH = Path("data/state/testnet_baseline.json")
DEFAULT_DUST_NOTIONAL_USD = 1.0
REQUIRED_BASELINE_KEYS: tuple[str, ...] = (
    "credentials_sha256",
    "initial_recon_equity",
    "spot_balances",
    "perp_positions",
    "prices",
    "created_at",
)

# Fallback khi khong import duoc src.cross_exchange_carry (tranh circular import).
_FALLBACK_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
)


def default_universe_symbols() -> list[str]:
    """Sprint 21: ro mac dinh day du (LIQUID_8_SYMBOLS) de doi soat toan ro."""
    try:
        from src.cross_exchange_carry import LIQUID_8_SYMBOLS

        symbols = [normalize_symbol(str(s)) for s in LIQUID_8_SYMBOLS]
        if symbols:
            return symbols
    except Exception:  # noqa: BLE001 - fallback tinh (khong phu thuoc import)
        pass
    return list(_FALLBACK_UNIVERSE)


def normalize_symbol(symbol: str) -> str:
    """Chuan hoa ma giao dich sang dang canonical f'{base}/USDT:USDT'."""
    if not symbol:
        return ""
    s = str(symbol).strip().upper()
    if not s:
        return ""
    if "/" in s:
        return s if ":" in s else f"{s}:USDT"
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT:USDT"
    return f"{s}/USDT:USDT"


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Doc file JSON exchange snapshot. Yeu cau root phai la dict.

    Dung ``utf-8-sig`` de chap nhan ca file co BOM (Windows/PowerShell hay sinh ra).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be a dict")
    return data


def _to_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _has_field(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj and obj[key] is not None
    return hasattr(obj, key) and getattr(obj, key) is not None


def _lookup_price(prices: Any, sym: str, base: str) -> float:
    if not isinstance(prices, dict):
        return 0.0
    concat = sym.replace(":USDT", "").replace("/", "")
    for k in (sym, concat, f"{base}/USDT", base):
        if k in prices:
            p = _to_float(prices[k])
            if p > 0.0:
                return p
    for pk, pv in prices.items():
        if normalize_symbol(str(pk)) == sym:
            p = _to_float(pv)
            if p > 0.0:
                return p
    return 0.0


def _lookup_price_testnet_delta(prices: Any, sym: str, base: str) -> float:
    """Sprint 22: Tra gia mo rong chi dung cho testnet delta snapshot.

    Ho tro ca base/USDT:USDT va baseUSDT de tra gia ca coin universe lan coin faucet.
    """
    if not isinstance(prices, dict):
        return 0.0
    concat = sym.replace(":USDT", "").replace("/", "")
    for k in (sym, concat, f"{base}/USDT:USDT", f"{base}/USDT", f"{base}USDT", base):
        if k in prices:
            p = _to_float(prices[k])
            if p > 0.0:
                return p
    for pk, pv in prices.items():
        if normalize_symbol(str(pk)) in (sym, normalize_symbol(sym), f"{base}/USDT:USDT"):
            p = _to_float(pv)
            if p > 0.0:
                return p
    return 0.0


def _lookup_lot_spec(lot_specs: Any, sym: str, base: str) -> tuple[float, float]:
    if not isinstance(lot_specs, dict):
        return 0.0, 0.0
    concat = sym.replace(":USDT", "").replace("/", "")
    spec = None
    for k in (sym, concat, f"{base}/USDT", base):
        if k in lot_specs:
            spec = lot_specs[k]
            break
    if spec is None:
        for lk, lv in lot_specs.items():
            if normalize_symbol(str(lk)) == sym:
                spec = lv
                break
    if spec is None:
        return 0.0, 0.0
    return _to_float(_get_field(spec, "step_size", 0.0)), _to_float(_get_field(spec, "min_notional", 0.0))


def _classify_leg(
    loc_qty: float, ex_qty: float, side: str, price: float, step: float, min_notional: float, tol_usd: float
) -> tuple[str, str | None, float, str]:
    # Sprint 23 FIX: sai so float khi tru baseline (1.001 - 1.0 = 0.0009999999999998899)
    # tung lam mot vi the THAT bi bao PHANTOM_IN_STATE -> dung epsilon tuong doi theo step.
    _eps = max(float(step) * 1e-9, 1e-15)
    loc_nz = abs(loc_qty) + _eps >= step
    ex_nz = abs(ex_qty) + _eps >= step
    if not loc_nz and not ex_nz:
        return SYNCED, None, 0.0, ""
    if not loc_nz and ex_nz:
        if side == "spot" and ex_qty < 0:
            diff = abs(ex_qty)
            return QTY_DRIFT, "RESYNC_QTY", diff, f"Spot balance dropped below baseline by {diff}"
        is_orphan = (abs(ex_qty) * price >= min_notional) if (price > 0.0 and min_notional > 0.0) else True
        if is_orphan:
            act = "CLOSE_ORPHAN_SPOT" if side == "spot" else "CLOSE_ORPHAN_PERP"
            return ORPHAN_ON_EXCHANGE, act, abs(ex_qty), f"Orphan {side} on exchange: {abs(ex_qty)}"
        return SYNCED, None, 0.0, ""
    if loc_nz and not ex_nz:
        return PHANTOM_IN_STATE, "PURGE_PHANTOM_STATE", abs(loc_qty), f"Phantom {side} in state: {abs(loc_qty)}"
    diff = abs(loc_qty - ex_qty)
    usd_gap = diff * price if price > 0.0 else 0.0
    is_drift = (usd_gap > tol_usd) if price > 0.0 else (diff >= step)
    if is_drift:
        return QTY_DRIFT, "RESYNC_QTY", diff, f"{side.capitalize()} qty drift: diff={diff}"
    return SYNCED, None, 0.0, ""


def analyze_equity_and_margin_health(
    local_state: Any,
    exchange_snapshot: Any,
    prices: dict[str, Any],
    lot_specs: dict[str, Any] | None = None,
    tolerance_usd: float = 10.0,
    equity_tolerance_pct: float = 1.5,
    max_maint_margin_ratio: float = 0.50,
    min_liq_buffer_pct: float = 25.0,
) -> dict[str, Any]:
    """Phan tich suc khoe ky quy Perp, unrealized PnL va khoang cach thanh ly.

    Deterministic, khong side effects, khong mang/network I/O.
    """
    # 1. Chuan hoa danh sach vi the chi tiet perp_positions_detail
    raw_details = _get_field(exchange_snapshot, "perp_positions_detail", None)
    unnorm_items: list[dict[str, Any]] = []

    if raw_details is not None:
        if isinstance(raw_details, dict):
            for k, v in raw_details.items():
                if isinstance(v, dict):
                    item = dict(v)
                    if "symbol" not in item:
                        item["symbol"] = k
                    unnorm_items.append(item)
                elif hasattr(v, "__dict__"):
                    item = {attr: getattr(v, attr) for attr in dir(v) if not attr.startswith("_")}
                    if "symbol" not in item:
                        item["symbol"] = k
                    unnorm_items.append(item)
        elif isinstance(raw_details, list):
            for entry in raw_details:
                if isinstance(entry, dict):
                    unnorm_items.append(dict(entry))
                elif hasattr(entry, "__dict__"):
                    unnorm_items.append({attr: getattr(entry, attr) for attr in dir(entry) if not attr.startswith("_")})
    else:
        # Fallback neu khong co perp_positions_detail
        perp_positions = _get_field(exchange_snapshot, "perp_positions", None) or {}
        if isinstance(perp_positions, dict):
            for k, v in perp_positions.items():
                unnorm_items.append({
                    "symbol": k,
                    "qty": _to_float(v),
                    "liquidation_price": 0.0,
                })

    standardized_positions: list[dict[str, Any]] = []
    for entry in unnorm_items:
        raw_sym = str(_get_field(entry, "symbol", "") or "").strip()
        canon_sym = normalize_symbol(raw_sym) if raw_sym else ""
        base = canon_sym.split("/")[0] if "/" in canon_sym else canon_sym

        raw_qty = _get_field(entry, "qty", None)
        if raw_qty is None:
            raw_qty = _get_field(entry, "positionAmt", 0.0)
        qty = _to_float(raw_qty)

        mark_price = _to_float(_get_field(entry, "mark_price", 0.0))
        if mark_price <= 0.0 and canon_sym:
            mark_price = _lookup_price(prices, canon_sym, base)

        entry_price = _to_float(_get_field(entry, "entry_price", 0.0))
        liq_price = _to_float(_get_field(entry, "liquidation_price", 0.0))
        maint_margin = _to_float(_get_field(entry, "maint_margin_usd", 0.0))
        margin_bal = _to_float(_get_field(entry, "margin_balance_usd", 0.0))

        standardized_positions.append({
            "symbol": canon_sym if canon_sym else raw_sym,
            "qty": float(qty),
            "mark_price": float(mark_price),
            "entry_price": float(entry_price),
            "liquidation_price": float(liq_price),
            "maint_margin_usd": float(maint_margin),
            "margin_balance_usd": float(margin_bal),
        })

    # 2. Tinh toan chi so suc khoe ky quy (margin_health)
    snap_maint = _to_float(_get_field(exchange_snapshot, "maint_margin_usd", 0.0))
    snap_margin_bal = _to_float(_get_field(exchange_snapshot, "margin_balance_usd", 0.0))
    if snap_maint == 0.0:
        sum_maint = sum(p["maint_margin_usd"] for p in standardized_positions)
        if sum_maint > 0.0:
            snap_maint = sum_maint
    if snap_margin_bal == 0.0:
        sum_mb = sum(p["margin_balance_usd"] for p in standardized_positions)
        if sum_mb > 0.0:
            snap_margin_bal = sum_mb

    maint_margin_ratio = (snap_maint / snap_margin_bal) if snap_margin_bal > 0.0 else 0.0

    liq_distances: list[float] = []
    for p in standardized_positions:
        if abs(p["qty"]) > 0.0 and p["liquidation_price"] > 0.0 and p["mark_price"] > 0.0:
            d = abs(p["mark_price"] - p["liquidation_price"]) / p["mark_price"] * 100.0
            liq_distances.append(d)

    min_liq_distance_pct = min(liq_distances) if liq_distances else None

    margin_danger = bool(
        (maint_margin_ratio > max_maint_margin_ratio)
        or (min_liq_distance_pct is not None and min_liq_distance_pct < min_liq_buffer_pct)
    )

    margin_health = {
        "maint_margin_ratio": float(maint_margin_ratio),
        "min_liq_distance_pct": min_liq_distance_pct,
        "max_maint_margin_ratio": float(max_maint_margin_ratio),
        "min_liq_buffer_pct": float(min_liq_buffer_pct),
        "margin_danger": margin_danger,
        "perp_positions_detail": standardized_positions,
    }

    # 3. Hanh dong khac phuc khan cap khi vi pham an toan ky quy (repair_actions)
    health_repair_actions: list[dict[str, Any]] = []
    if margin_danger:
        dangerous_symbols: set[str] = set()
        open_positions = [p for p in standardized_positions if abs(p["qty"]) > 0.0]

        if min_liq_distance_pct is not None and min_liq_distance_pct < min_liq_buffer_pct:
            for p in open_positions:
                if p["liquidation_price"] > 0.0 and p["mark_price"] > 0.0:
                    dist = abs(p["mark_price"] - p["liquidation_price"]) / p["mark_price"] * 100.0
                    if dist < min_liq_buffer_pct:
                        dangerous_symbols.add(p["symbol"])
                        health_repair_actions.append({
                            "symbol": p["symbol"],
                            "action": "EMERGENCY_DELEVERAGE_PERP",
                            "side": "perp",
                            "qty": abs(p["qty"]),
                            "reason": f"Perp liq distance {dist:.1f}% below min buffer {min_liq_buffer_pct:.1f}%",
                        })

        if maint_margin_ratio > max_maint_margin_ratio:
            if open_positions:
                for p in open_positions:
                    if p["symbol"] not in dangerous_symbols:
                        dangerous_symbols.add(p["symbol"])
                        health_repair_actions.append({
                            "symbol": p["symbol"],
                            "action": "EMERGENCY_DELEVERAGE_PERP",
                            "side": "perp",
                            "qty": abs(p["qty"]),
                            "reason": f"Maintenance margin ratio {maint_margin_ratio:.2f} exceeds max {max_maint_margin_ratio:.2f}",
                        })
            else:
                total_qty = sum(abs(p["qty"]) for p in standardized_positions)
                health_repair_actions.append({
                    "symbol": "PORTFOLIO",
                    "action": "EMERGENCY_DELEVERAGE_PERP",
                    "side": "perp",
                    "qty": float(total_qty),
                    "reason": f"Maintenance margin ratio {maint_margin_ratio:.2f} exceeds max {max_maint_margin_ratio:.2f}",
                })

    # 4. Phan ra nguyen nhan lech equity (equity_drift_attribution)
    has_attr_fields = (
        _has_field(exchange_snapshot, "spot_cash_usd")
        or _has_field(exchange_snapshot, "perp_wallet_balance_usd")
        or _has_field(exchange_snapshot, "perp_unrealized_pnl_usd")
    )

    if not has_attr_fields:
        equity_drift_attribution = None
    else:
        local_unrealized_pnl_usd = 0.0
        carry_positions = _get_field(local_state, "carry_positions") or {}
        overlay_positions = _get_field(local_state, "overlay_positions") or {}

        def _get_pos_items(positions: Any) -> list[tuple[str, Any]]:
            if isinstance(positions, dict):
                return [(str(k), v) for k, v in positions.items()]
            if isinstance(positions, list):
                return [(str(_get_field(x, "symbol", "") or ""), x) for x in positions]
            return []

        for sym_key, pos in _get_pos_items(carry_positions):
            c_sym = normalize_symbol(sym_key)
            base = c_sym.split("/")[0] if "/" in c_sym else c_sym
            p_price = _lookup_price(prices, c_sym, base)
            leg2 = _get_field(pos, "leg2_hedged_qty", None)
            c_qty = _to_float(leg2) if leg2 is not None else _to_float(_get_field(pos, "qty", 0.0))
            c_qty = abs(c_qty)
            entry_perp = _to_float(
                _get_field(pos, "entry_perp_price", _get_field(pos, "entry_price", _get_field(pos, "perp_entry_price", 0.0)))
            )
            if p_price > 0.0 and entry_perp > 0.0 and c_qty > 0.0:
                local_unrealized_pnl_usd += (entry_perp - p_price) * c_qty

        for sym_key, pos in _get_pos_items(overlay_positions):
            o_sym = normalize_symbol(sym_key)
            base = o_sym.split("/")[0] if "/" in o_sym else o_sym
            p_price = _lookup_price(prices, o_sym, base)
            o_qty = abs(_to_float(_get_field(pos, "qty", 0.0)))
            entry_price = _to_float(_get_field(pos, "entry_price", 0.0))
            side = str(_get_field(pos, "side", "") or "").strip().lower()
            if p_price > 0.0 and entry_price > 0.0 and o_qty > 0.0:
                if side == "short":
                    local_unrealized_pnl_usd += (entry_price - p_price) * o_qty
                else:
                    local_unrealized_pnl_usd += (p_price - entry_price) * o_qty

        if local_unrealized_pnl_usd == 0.0:
            local_unrealized_pnl_usd = _to_float(
                _get_field(local_state, "local_unrealized_pnl_usd", _get_field(local_state, "unrealized_pnl_usd", 0.0))
            )

        spot_cash_usd = _to_float(_get_field(exchange_snapshot, "spot_cash_usd", 0.0))
        perp_wallet_balance_usd = _to_float(_get_field(exchange_snapshot, "perp_wallet_balance_usd", 0.0))
        perp_unrealized_pnl_usd = _to_float(_get_field(exchange_snapshot, "perp_unrealized_pnl_usd", 0.0))

        raw_recon_eq = _get_field(exchange_snapshot, "reconcile_equity_usd", None)
        if raw_recon_eq is not None:
            exchange_equity_usd = _to_float(raw_recon_eq)
        else:
            exchange_equity_usd = _to_float(_get_field(exchange_snapshot, "exchange_equity_usd", 0.0))
            if exchange_equity_usd == 0.0 and (spot_cash_usd != 0.0 or perp_wallet_balance_usd != 0.0 or perp_unrealized_pnl_usd != 0.0):
                exchange_equity_usd = spot_cash_usd + perp_wallet_balance_usd + perp_unrealized_pnl_usd

        local_expected_equity_usd = _to_float(_get_field(local_state, "current_equity", 0.0))

        total_drift = exchange_equity_usd - local_expected_equity_usd
        cash_drift = (spot_cash_usd + perp_wallet_balance_usd) - (local_expected_equity_usd - local_unrealized_pnl_usd)
        upnl_drift = perp_unrealized_pnl_usd - local_unrealized_pnl_usd
        residual_drift = total_drift - cash_drift - upnl_drift
        if abs(residual_drift) < 1e-9:
            residual_drift = 0.0

        c_abs = abs(cash_drift)
        u_abs = abs(upnl_drift)
        r_abs = abs(residual_drift)
        max_drift = max(c_abs, u_abs, r_abs)

        if max_drift < 1e-4:
            primary_cause = None
        elif c_abs == max_drift:
            primary_cause = "CASH_OR_FUNDING_DRIFT"
        elif u_abs == max_drift:
            primary_cause = "UNREALIZED_PNL_DRIFT"
        else:
            primary_cause = "UNEXPLAINED_EQUITY_DRIFT"

        equity_drift_attribution = {
            "total_equity_drift_usd": float(total_drift),
            "cash_and_funding_drift_usd": float(cash_drift),
            "unrealized_pnl_drift_usd": float(upnl_drift),
            "unexplained_residual_drift_usd": float(residual_drift),
            "primary_mismatch_cause": primary_cause,
        }

    return {
        "equity_drift_attribution": equity_drift_attribution,
        "margin_health": margin_health,
        "margin_danger": margin_danger,
        "repair_actions": health_repair_actions,
    }


def reconcile_portfolio_state(
    local_state: Any,
    exchange_snapshot: Any,
    prices: dict[str, Any],
    lot_specs: dict[str, Any],
    tolerance_usd: float = 10.0,
    equity_tolerance_pct: float = 1.5,
    max_maint_margin_ratio: float = 0.50,
    min_liq_buffer_pct: float = 25.0,
    universe_symbols: Sequence[str] | None = None,
    environment: str = "testnet",
    ignored_spot_assets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Doi soat danh muc local vs san.

    Sprint 21:
    - Quet **toan ro** ``universe_symbols`` (mac dinh ``LIQUID_8_SYMBOLS``) thay vi
      chi cac ma dang giu vi the, nho do phat hien vi the mo coi tren bat ky ma nao.
    - ``environment="mainnet"`` bat canh bao ``UNTRACKED_MAINNET_ASSET_WARNING`` khi
      co tai san Spot/Perp ngoai danh muc; vi the Perp la -> ``BLOCKED`` (cam giao dich).
      O ``"testnet"`` cac tai san faucet chi duoc thong ke, khong chan giao dich.
    """
    env = str(environment or "testnet").strip().lower()
    if env not in ("testnet", "mainnet"):
        raise ValueError(f"environment must be 'testnet' or 'mainnet', got {environment!r}")

    if universe_symbols is None:
        universe = default_universe_symbols()
    else:
        universe = [normalize_symbol(str(s)) for s in universe_symbols if str(s).strip()]
    universe_set = set(universe)
    universe_bases = {sym.split("/")[0].upper() for sym in universe_set if "/" in sym}
    ignored_assets = {str(a).strip().upper() for a in (ignored_spot_assets or ()) if str(a).strip()}

    carry_positions = _get_field(local_state, "carry_positions") or {}
    overlay_positions = _get_field(local_state, "overlay_positions") or {}
    spot_balances = _get_field(exchange_snapshot, "spot_balances") or {}
    perp_positions = _get_field(exchange_snapshot, "perp_positions") or {}
    perp_positions_detail = _get_field(exchange_snapshot, "perp_positions_detail") or {}

    canonical_symbols: set[str] = set(universe_set)

    def _add_snapshot_symbol(raw: Any) -> None:
        """Chi them symbol THUOC RO vao danh sach doi soat.

        Sprint 21: tai san Spot/Perp ngoai ro (faucet testnet, dust) duoc thong ke
        rieng trong ``untracked_spot_assets`` / ``untracked_perp_positions`` va
        KHONG duoc bien thanh ORPHAN_ON_EXCHANGE (tru khi o mainnet -> BLOCKED).
        """
        sym = normalize_symbol(str(raw))
        if not sym:
            return
        if universe_set:
            base = sym.split("/")[0].upper()
            if sym not in universe_set and f"{base}/USDT:USDT" not in universe_set:
                return
        canonical_symbols.add(sym)

    if isinstance(carry_positions, dict):
        for k in carry_positions:
            s = normalize_symbol(str(k))
            if s:
                canonical_symbols.add(s)
    if isinstance(overlay_positions, dict):
        for k in overlay_positions:
            s = normalize_symbol(str(k))
            if s:
                canonical_symbols.add(s)
    if isinstance(spot_balances, dict):
        for b in spot_balances:
            b_str = str(b).strip()
            if b_str:
                _add_snapshot_symbol(b_str)
    if isinstance(perp_positions, dict):
        for k in perp_positions:
            _add_snapshot_symbol(k)
    if isinstance(perp_positions_detail, dict):
        for k in perp_positions_detail:
            _add_snapshot_symbol(k)
    elif isinstance(perp_positions_detail, list):
        for item in perp_positions_detail:
            _add_snapshot_symbol(_get_field(item, "symbol", ""))

    norm_carry = {normalize_symbol(str(k)): v for k, v in carry_positions.items()} if isinstance(carry_positions, dict) else {}
    norm_overlay = {normalize_symbol(str(k)): v for k, v in overlay_positions.items()} if isinstance(overlay_positions, dict) else {}

    # --- Sprint 21: phat hien tai san ngoai danh muc (untracked) ---
    untracked_spot_assets: dict[str, float] = {}
    raw_untracked = _get_field(exchange_snapshot, "untracked_spot_assets", None)
    if isinstance(raw_untracked, dict):
        for k, v in raw_untracked.items():
            asset = str(k).strip().upper()
            qty = _to_float(v)
            if asset and qty > 0.0:
                untracked_spot_assets[asset] = qty
    if isinstance(spot_balances, dict):
        for k, v in spot_balances.items():
            asset = str(k).split("/")[0].strip().upper()
            if not asset or asset in universe_bases:
                continue
            qty = _to_float(v)
            if qty > 0.0:
                untracked_spot_assets.setdefault(asset, qty)

    untracked_perp_positions: dict[str, float] = {}
    if isinstance(perp_positions, dict):
        for k, v in perp_positions.items():
            canon = normalize_symbol(str(k))
            qty = _to_float(v)
            if canon and canon not in universe_set and abs(qty) > 0.0:
                untracked_perp_positions[canon] = qty
    if isinstance(perp_positions_detail, dict):
        for k, det in perp_positions_detail.items():
            canon = normalize_symbol(str(k))
            qty = _to_float(_get_field(det, "qty", _get_field(det, "positionAmt", 0.0)))
            if canon and canon not in universe_set and abs(qty) > 0.0:
                untracked_perp_positions.setdefault(canon, qty)
    elif isinstance(perp_positions_detail, list):
        for item in perp_positions_detail:
            canon = normalize_symbol(str(_get_field(item, "symbol", "")))
            qty = _to_float(_get_field(item, "qty", _get_field(item, "positionAmt", 0.0)))
            if canon and canon not in universe_set and abs(qty) > 0.0:
                untracked_perp_positions.setdefault(canon, qty)

    untracked_spot_flagged = {
        a: q for a, q in untracked_spot_assets.items() if a not in ignored_assets
    }
    warnings: list[str] = []
    mainnet_untracked_warning = False
    if env == "mainnet" and (untracked_spot_flagged or untracked_perp_positions):
        mainnet_untracked_warning = True
        warnings.append(UNTRACKED_MAINNET_ASSET_WARNING)
    elif env == "testnet" and untracked_perp_positions:
        # Testnet KHONG chan giao dich (chinh sach faucet), nhung phai canh bao de
        # khong bao "SYNCED" mot cach im lang khi co vi the Perp ngoai danh muc.
        warnings.append(UNTRACKED_TESTNET_PERP_WARNING)
    blocked_by_untracked_perp = bool(env == "mainnet" and untracked_perp_positions)

    items: list[dict[str, Any]] = []
    repair_actions: list[dict[str, Any]] = []


    for sym in sorted(canonical_symbols):
        base = sym.split("/")[0]
        concat = sym.replace(":USDT", "").replace("/", "")

        # Local quantities
        c_pos = norm_carry.get(sym)
        c_qty = _to_float(_get_field(c_pos, "qty", 0.0)) if c_pos else 0.0
        loc_spot = c_qty
        leg2 = _get_field(c_pos, "leg2_hedged_qty", None) if c_pos else None
        c_perp = -_to_float(leg2) if (c_pos and leg2 is not None) else -c_qty

        o_pos = norm_overlay.get(sym)
        o_qty = _to_float(_get_field(o_pos, "qty", 0.0)) if o_pos else 0.0
        side_val = str(_get_field(o_pos, "side", "") or "").strip().lower() if o_pos else ""
        o_perp = -abs(o_qty) if side_val == "short" else (abs(o_qty) if side_val == "long" else o_qty)

        loc_perp = c_perp + o_perp

        # Exchange quantities
        ex_spot = 0.0
        if isinstance(spot_balances, dict):
            for k in (base, sym, concat):
                if k in spot_balances:
                    ex_spot = _to_float(spot_balances[k])
                    break

        ex_perp = 0.0
        if isinstance(perp_positions, dict):
            for k in (sym, concat, base):
                if k in perp_positions:
                    ex_perp = _to_float(perp_positions[k])
                    break
        if ex_perp == 0.0 and perp_positions_detail:
            if isinstance(perp_positions_detail, dict):
                for k in (sym, concat, base):
                    if k in perp_positions_detail:
                        det = perp_positions_detail[k]
                        q = _get_field(det, "qty", _get_field(det, "positionAmt", 0.0))
                        ex_perp = _to_float(q)
                        break
            elif isinstance(perp_positions_detail, list):
                for item in perp_positions_detail:
                    item_sym = normalize_symbol(str(_get_field(item, "symbol", "")))
                    if item_sym == sym:
                        q = _get_field(item, "qty", _get_field(item, "positionAmt", 0.0))
                        ex_perp = _to_float(q)
                        break

        price = _lookup_price(prices, sym, base)
        step_sz, min_notional = _lookup_lot_spec(lot_specs, sym, base)
        step = step_sz if step_sz > 0.0 else 1e-8

        spot_gap = abs(loc_spot - ex_spot) * price if price > 0.0 else 0.0
        perp_gap = abs(loc_perp - ex_perp) * price if price > 0.0 else 0.0

        s_st, s_act, s_qty, s_rsn = _classify_leg(loc_spot, ex_spot, "spot", price, step, min_notional, tolerance_usd)
        p_st, p_act, p_qty, p_rsn = _classify_leg(loc_perp, ex_perp, "perp", price, step, min_notional, tolerance_usd)

        if s_st == ORPHAN_ON_EXCHANGE or p_st == ORPHAN_ON_EXCHANGE:
            item_st = ORPHAN_ON_EXCHANGE
            item_act = s_act if s_st == ORPHAN_ON_EXCHANGE else p_act
        elif s_st == PHANTOM_IN_STATE or p_st == PHANTOM_IN_STATE:
            item_st = PHANTOM_IN_STATE
            item_act = "PURGE_PHANTOM_STATE"
        elif s_st == QTY_DRIFT or p_st == QTY_DRIFT:
            item_st = QTY_DRIFT
            item_act = "RESYNC_QTY"
        else:
            item_st = SYNCED
            item_act = None

        if s_act is not None:
            repair_actions.append({"symbol": sym, "action": s_act, "side": "spot", "qty": float(s_qty), "reason": s_rsn})
        if p_act is not None:
            repair_actions.append({"symbol": sym, "action": p_act, "side": "perp", "qty": float(p_qty), "reason": p_rsn})

        items.append({
            "symbol": sym,
            "local_spot_qty": loc_spot,
            "exchange_spot_qty": ex_spot,
            "local_perp_qty": loc_perp,
            "exchange_perp_qty": ex_perp,
            "spot_gap_usd": spot_gap,
            "perp_gap_usd": perp_gap,
            "status": item_st,
            "repair_action": item_act,
        })

    orphans = [it["symbol"] for it in items if it["status"] == ORPHAN_ON_EXCHANGE]
    phantoms = [it["symbol"] for it in items if it["status"] == PHANTOM_IN_STATE]
    drifting = [it["symbol"] for it in items if it["status"] == QTY_DRIFT]

    loc_eq = _to_float(_get_field(local_state, "current_equity", 0.0))
    real_ex_eq = _to_float(_get_field(exchange_snapshot, "exchange_equity_usd", 0.0))
    raw_recon_eq = _get_field(exchange_snapshot, "reconcile_equity_usd", None)
    if raw_recon_eq is not None:
        recon_eq = _to_float(raw_recon_eq)
    else:
        recon_eq = real_ex_eq
    eq_gap_usd = recon_eq - loc_eq
    eq_gap_pct = (abs(eq_gap_usd) / loc_eq * 100.0) if loc_eq > 0.0 else (0.0 if eq_gap_usd == 0.0 else 100.0)

    # Phan tich suc khoe ky quy va attribution
    health_result = analyze_equity_and_margin_health(
        local_state=local_state,
        exchange_snapshot=exchange_snapshot,
        prices=prices,
        lot_specs=lot_specs,
        tolerance_usd=tolerance_usd,
        equity_tolerance_pct=equity_tolerance_pct,
        max_maint_margin_ratio=max_maint_margin_ratio,
        min_liq_buffer_pct=min_liq_buffer_pct,
    )
    repair_actions.extend(health_result["repair_actions"])

    # Thu tu uu tien status:
    # BLOCKED > EQUITY_MISMATCH > MARGIN_DANGER > ORPHAN_ON_EXCHANGE > PHANTOM_IN_STATE
    # > QTY_DRIFT > SYNCED
    if blocked_by_untracked_perp:
        status = BLOCKED
    elif eq_gap_pct > equity_tolerance_pct:
        status = EQUITY_MISMATCH
    elif health_result["margin_danger"]:
        status = MARGIN_DANGER
    elif orphans:
        status = ORPHAN_ON_EXCHANGE
    elif phantoms:
        status = PHANTOM_IN_STATE
    elif drifting:
        status = QTY_DRIFT
    else:
        status = SYNCED

    has_margin_data = (
        _has_field(exchange_snapshot, "margin_balance_usd")
        or _has_field(exchange_snapshot, "maint_margin_usd")
        or health_result["margin_health"]["maint_margin_ratio"] > 0.0
    )
    mr = health_result["margin_health"]["maint_margin_ratio"]
    mr_str = f"{mr:.2f}" if (has_margin_data and mr > 0.0) else ("0.00" if has_margin_data else "N/A")

    ld = health_result["margin_health"]["min_liq_distance_pct"]
    ld_str = f"{ld:.1f}%" if ld is not None else "N/A"

    untracked_str = ""
    if env == "mainnet":
        untracked_str = (
            f" | untracked_spot={len(untracked_spot_flagged)}"
            f" untracked_perp={len(untracked_perp_positions)}"
        )

    summary = (
        f"{status} | items={len(items)} | equity_gap={eq_gap_usd:.2f} USD ({eq_gap_pct:.2f}%)"
        f" | margin_ratio={mr_str} | liq_dist={ld_str}{untracked_str}"
    )

    return {
        "status": status,
        "trading_allowed": (status == SYNCED),
        "repair_actions": repair_actions,
        "items": items,
        "orphans": orphans,
        "phantoms": phantoms,
        "drifting": drifting,
        "equity_gap_usd": eq_gap_usd,
        "equity_gap_pct": eq_gap_pct,
        "reconcile_equity_usd": recon_eq,
        "exchange_equity_usd": real_ex_eq,
        "tolerance_usd": float(tolerance_usd),
        "equity_tolerance_pct": float(equity_tolerance_pct),
        "summary": summary,
        "equity_drift_attribution": health_result["equity_drift_attribution"],
        "margin_health": health_result["margin_health"],
        "margin_danger": health_result["margin_danger"],
        # --- Sprint 21: doi soat toan ro + canh bao tai san la Mainnet ---
        "environment": env,
        "universe_symbols": sorted(universe_set),
        "ignored_spot_assets": sorted(ignored_assets),
        "untracked_spot_assets": dict(sorted(untracked_spot_assets.items())),
        "untracked_spot_assets_flagged": dict(sorted(untracked_spot_flagged.items())),
        "untracked_spot_count": len(untracked_spot_assets),
        "untracked_perp_positions": dict(sorted(untracked_perp_positions.items())),
        "untracked_perp_count": len(untracked_perp_positions),
        "untracked_mainnet_asset_warning": mainnet_untracked_warning,
        "untracked_testnet_perp_warning": bool(env == "testnet" and untracked_perp_positions),
        "warnings": warnings,
        "blocked": status == BLOCKED,
    }



def load_or_create_testnet_baseline(
    snapshot: dict[str, Any],
    prices: dict[str, Any],
    api_key_hash: str,
    universe_symbols: Sequence[str] | None = None,
    path: Path | str = DEFAULT_TESTNET_BASELINE_PATH,
    reset_baseline: bool = False,
) -> dict[str, Any]:
    """Sprint 22: Quan ly file baseline testnet ben vung, atomic, fail-closed."""
    p = Path(path)
    clean_key_hash = str(api_key_hash or "").strip()
    if not clean_key_hash:
        raise RuntimeError("BASELINE_KEY_HASH_EMPTY: api_key_hash khong duoc de trong. Abort fail-closed.")

    if not reset_baseline and p.is_file():
        try:
            raw_text = p.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except Exception as exc:
            raise RuntimeError(f"BASELINE_CORRUPTED: File {p} bi hong ({exc}). Abort fail-closed.") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"BASELINE_CORRUPTED: File {p} khong phai JSON object hop le. Abort fail-closed.")

        for req_k in REQUIRED_BASELINE_KEYS:
            if req_k not in data:
                raise RuntimeError(
                    f"BASELINE_SCHEMA_INVALID: File {p} thieu truong bat buoc '{req_k}'. Abort fail-closed."
                )

        saved_hash = str(data.get("credentials_sha256") or "").strip()
        if not saved_hash:
            raise RuntimeError(
                f"BASELINE_SCHEMA_INVALID: File {p} co credentials_sha256 bi rong. Abort fail-closed."
            )

        if saved_hash != clean_key_hash:
            raise RuntimeError(
                f"BASELINE_KEY_MISMATCH: API key hien tai khong khop baseline cu. "
                f"Dung co --reset-testnet-baseline de tao lai baseline."
            )
        return data

    # Khi tao baseline moi: bat buoc moi vi the perp trong universe phai phang (abs(qty) < 1e-6)
    symbols_to_check = list(universe_symbols or default_universe_symbols())
    perp_pos = dict(_get_field(snapshot, "perp_positions", {}) or {})
    for sym in symbols_to_check:
        canon = normalize_symbol(str(sym))
        qty = float(_to_float(perp_pos.get(canon, 0.0)))
        if abs(qty) >= 1e-6:
            raise RuntimeError(
                f"BASELINE_PERP_NOT_FLAT: Khong the tao baseline khi vi the perp {canon} con mo ({qty}). "
                f"Dong vi the truoc khi tao baseline."
            )

    # Kiem tra ca perp_positions_detail neu co
    perp_details = _get_field(snapshot, "perp_positions_detail", []) or []
    if isinstance(perp_details, list):
        for item in perp_details:
            if isinstance(item, dict):
                canon = normalize_symbol(str(item.get("symbol", "")))
                if canon in [normalize_symbol(str(s)) for s in symbols_to_check]:
                    qty = float(_to_float(item.get("qty", item.get("positionAmt", 0.0))))
                    if abs(qty) >= 1e-6:
                        raise RuntimeError(
                            f"BASELINE_PERP_NOT_FLAT: Khong the tao baseline khi vi the perp {canon} con mo ({qty}). "
                            f"Dong vi the truoc khi tao baseline."
                        )

    spot_bal = dict(_get_field(snapshot, "spot_balances", {}) or {})
    pure_cash = float(sum(_to_float(v) for a, v in spot_bal.items() if _is_cash_asset_local(a)))
    if pure_cash == 0.0:
        # Sprint 23: uu tien pure_cash_usd (chi stablecoin) de baseline khong bi lan
        # bien dong gia coin faucet testnet.
        _pure_field = float(_to_float(_get_field(snapshot, "pure_cash_usd", 0.0)))
        if _pure_field > 0.0:
            pure_cash = _pure_field
        elif _get_field(snapshot, "spot_cash_usd", 0.0):
            pure_cash = float(_to_float(_get_field(snapshot, "spot_cash_usd", 0.0)))
    fut_wallet = float(_to_float(_get_field(snapshot, "perp_wallet_balance_usd", 0.0)))
    fut_upnl = float(_to_float(_get_field(snapshot, "perp_unrealized_pnl_usd", 0.0)))
    recon_equity_0 = pure_cash + fut_wallet + fut_upnl

    baseline_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "credentials_sha256": str(api_key_hash),
        "initial_recon_equity": float(recon_equity_0),
        "spot_balances": {str(k).upper(): float(_to_float(v)) for k, v in spot_bal.items()},
        "perp_positions": {normalize_symbol(str(k)): float(_to_float(v)) for k, v in perp_pos.items()},
        "prices": {normalize_symbol(str(k)): float(_to_float(v)) for k, v in (prices or {}).items()},
    }

    # Ghi atomic qua file tam cung thu muc
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    os.replace(tmp_path, p)

    logger.warning(
        "\n"
        "====================================================================\n"
        "[TESTNET BASELINE INITIALIZED]\n"
        "Timestamp        : %s\n"
        "Spot Assets Count: %d\n"
        "Perp Positions   : ALL FLAT (0.0)\n"
        "Initial Equity   : %.2f USD\n"
        "Baseline File    : %s\n"
        "====================================================================",
        baseline_data["created_at"],
        len(baseline_data["spot_balances"]),
        baseline_data["initial_recon_equity"],
        str(p),
    )
    baseline_data["_is_created"] = True
    return baseline_data


def compute_testnet_delta_snapshot(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
    prices: dict[str, Any],
    capital: float,
    local_equity: float,
    dust_notional_usd: float = DEFAULT_DUST_NOTIONAL_USD,
) -> dict[str, Any]:
    """Sprint 22: Tinh toan delta snapshot co dau va loai bo hoan toan bien dong gia coin faucet."""
    out = dict(snapshot)
    base_spot = dict(baseline.get("spot_balances") or {})
    curr_spot = dict(_get_field(snapshot, "spot_balances", {}) or {})
    base_perp = dict(baseline.get("perp_positions") or {})
    curr_perp = dict(_get_field(snapshot, "perp_positions", {}) or {})

    # 1. Delta Spot co dau (khong dung max(0, ...))
    delta_spot: dict[str, float] = {}
    delta_spot_market_value = 0.0
    all_spot_keys = set(curr_spot.keys()) | set(base_spot.keys())

    for asset in all_spot_keys:
        qt = float(_to_float(curr_spot.get(asset, 0.0)))
        q0 = float(_to_float(base_spot.get(asset, 0.0)))
        dq = qt - q0

        if _is_cash_asset_local(asset):
            delta_spot[asset] = dq
        else:
            px = _lookup_price_testnet_delta(prices, asset, asset)
            notional = abs(dq) * px if px > 0.0 else 0.0
            if px > 0.0:
                is_dust = (abs(dq) < 1e-6) or (notional < dust_notional_usd)
            else:
                is_dust = abs(dq) < 1e-6

            if is_dust:
                delta_spot[asset] = 0.0
            else:
                # Sprint 23: lam tron 12 chu so thap phan de khu sai so float cua phep tru.
                delta_spot[asset] = round(dq, 12)
                if px > 0.0:
                    delta_spot_market_value += dq * px

    out["spot_balances"] = delta_spot

    # 2. Delta Perp co dau
    delta_perp: dict[str, float] = {}
    all_perp_keys = set(curr_perp.keys()) | set(base_perp.keys())
    for sym in all_perp_keys:
        pt = float(_to_float(curr_perp.get(sym, 0.0)))
        p0 = float(_to_float(base_perp.get(sym, 0.0)))
        delta_perp[sym] = pt - p0
    out["perp_positions"] = delta_perp

    # Dong bo perp_positions_detail
    perp_details = out.get("perp_positions_detail")
    if isinstance(perp_details, list):
        updated_details = []
        for item in perp_details:
            if isinstance(item, dict):
                c_item = dict(item)
                sym_canon = normalize_symbol(str(c_item.get("symbol", "")))
                if sym_canon in delta_perp:
                    new_q = delta_perp[sym_canon]
                    c_item["qty"] = new_q
                    c_item["positionAmt"] = new_q
                    if abs(new_q) < 1e-6:
                        c_item["maint_margin_usd"] = 0.0
                updated_details.append(c_item)
        out["perp_positions_detail"] = updated_details
    elif isinstance(perp_details, dict):
        updated_details_dict = {}
        for k, item in perp_details.items():
            if isinstance(item, dict):
                c_item = dict(item)
                sym_canon = normalize_symbol(str(k))
                if sym_canon in delta_perp:
                    new_q = delta_perp[sym_canon]
                    c_item["qty"] = new_q
                    c_item["positionAmt"] = new_q
                    if abs(new_q) < 1e-6:
                        c_item["maint_margin_usd"] = 0.0
                updated_details_dict[k] = c_item
            else:
                updated_details_dict[k] = item
        out["perp_positions_detail"] = updated_details_dict

    # 3. Tinh reconcile_equity_usd loai bo MTM coin faucet ban dau
    pure_cash = float(sum(_to_float(v) for a, v in curr_spot.items() if _is_cash_asset_local(a)))
    cash_is_total_valuation = False
    if pure_cash == 0.0:
        # Sprint 23: uu tien pure_cash_usd (chi stablecoin, khong gom coin faucet).
        _pure_field = float(_to_float(_get_field(snapshot, "pure_cash_usd", 0.0)))
        if _pure_field > 0.0:
            pure_cash = _pure_field
        elif _get_field(snapshot, "spot_cash_usd", 0.0):
            # Sprint 23 FIX (double-count): spot_cash_usd la TONG dinh gia spot (cash + coin da
            # priced), khong phai rieng tien mat. Cong them delta_spot_market_value nua se dem
            # hai lan gia tri coin vua mua -> EQUITY_MISMATCH gia tao khi mo carry that.
            pure_cash = float(_to_float(_get_field(snapshot, "spot_cash_usd", 0.0)))
            cash_is_total_valuation = True
    fut_wallet = float(_to_float(_get_field(snapshot, "perp_wallet_balance_usd", 0.0)))
    fut_upnl = float(_to_float(_get_field(snapshot, "perp_unrealized_pnl_usd", 0.0)))
    if cash_is_total_valuation:
        actual_recon_equity = pure_cash + fut_wallet + fut_upnl
    else:
        actual_recon_equity = pure_cash + fut_wallet + fut_upnl + delta_spot_market_value

    init_recon_equity = float(baseline.get("initial_recon_equity", 0.0))
    expected_recon_equity = init_recon_equity + (float(local_equity) - float(capital))
    net_drift = actual_recon_equity - expected_recon_equity

    out["reconcile_equity_usd"] = float(local_equity) + net_drift
    return out
