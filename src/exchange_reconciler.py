"""
Module doi soat trang thai danh muc giua local bot state va real exchange snapshot.
Dung cho pre-flight reconciliation truoc khi bat dat lenh testnet that.
Thiet ke hoan toan deterministic, khong side effects, khong mang/network I/O.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SYNCED = "SYNCED"
ORPHAN_ON_EXCHANGE = "ORPHAN_ON_EXCHANGE"
PHANTOM_IN_STATE = "PHANTOM_IN_STATE"
QTY_DRIFT = "QTY_DRIFT"
EQUITY_MISMATCH = "EQUITY_MISMATCH"

RECONCILE_STATUSES: tuple[str, ...] = (
    SYNCED,
    ORPHAN_ON_EXCHANGE,
    PHANTOM_IN_STATE,
    QTY_DRIFT,
    EQUITY_MISMATCH,
)


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
    """Doc file JSON exchange snapshot. Yeu cau root phai la dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
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
    loc_nz = abs(loc_qty) >= step
    ex_nz = abs(ex_qty) >= step
    if not loc_nz and not ex_nz:
        return SYNCED, None, 0.0, ""
    if not loc_nz and ex_nz:
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


def reconcile_portfolio_state(
    local_state: Any,
    exchange_snapshot: Any,
    prices: dict[str, Any],
    lot_specs: dict[str, Any],
    tolerance_usd: float = 10.0,
    equity_tolerance_pct: float = 1.5,
) -> dict[str, Any]:
    carry_positions = _get_field(local_state, "carry_positions") or {}
    overlay_positions = _get_field(local_state, "overlay_positions") or {}
    spot_balances = _get_field(exchange_snapshot, "spot_balances") or {}
    perp_positions = _get_field(exchange_snapshot, "perp_positions") or {}

    canonical_symbols: set[str] = set()
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
                canonical_symbols.add(normalize_symbol(b_str) if "/" in b_str else f"{b_str.upper()}/USDT:USDT")
    if isinstance(perp_positions, dict):
        for k in perp_positions:
            s = normalize_symbol(str(k))
            if s:
                canonical_symbols.add(s)

    norm_carry = {normalize_symbol(str(k)): v for k, v in carry_positions.items()} if isinstance(carry_positions, dict) else {}
    norm_overlay = {normalize_symbol(str(k)): v for k, v in overlay_positions.items()} if isinstance(overlay_positions, dict) else {}

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
    ex_eq = _to_float(_get_field(exchange_snapshot, "exchange_equity_usd", 0.0))
    eq_gap_usd = ex_eq - loc_eq
    eq_gap_pct = (abs(eq_gap_usd) / loc_eq * 100.0) if loc_eq > 0.0 else (0.0 if eq_gap_usd == 0.0 else 100.0)

    if eq_gap_pct > equity_tolerance_pct:
        status = EQUITY_MISMATCH
    elif orphans:
        status = ORPHAN_ON_EXCHANGE
    elif phantoms:
        status = PHANTOM_IN_STATE
    elif drifting:
        status = QTY_DRIFT
    else:
        status = SYNCED

    summary = f"{status} | items={len(items)} | equity_gap={eq_gap_usd:.2f} USD ({eq_gap_pct:.2f}%)"

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
        "tolerance_usd": float(tolerance_usd),
        "equity_tolerance_pct": float(equity_tolerance_pct),
        "summary": summary,
    }
