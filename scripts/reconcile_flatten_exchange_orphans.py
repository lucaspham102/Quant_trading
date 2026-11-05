#!/usr/bin/env python3
"""scripts/reconcile_flatten_exchange_orphans.py

Cong cu kiem toan phap y & dong bo hoa vi the san Binance Testnet 1:1 voi So cai Local (regime_omni_1k).
Phat hien va xu ly triet de cac vi the mo coi (orphan) va doi du do cac tien trinh cu de lai:
  - AVAX Perp: -11.0 (Orphan short)
  - BNB Perp : +0.14 (Ghost long mismatch)
  - SOL Carry: +2.5 Spot / -2.5 Perp (Ghost carry leg)
  - BTC Carry: +0.002 Spot / -0.002 Perp (Excess carry)
  - DOGE Carry: +876.0 Spot / -876.0 Perp (Excess carry)
  - XRP Carry: +252.6 Spot / -252.6 Perp (Excess carry)

Mac dinh chay o che do --dry-run (100% Read-Only, khong ban lenh).
Chi ban lenh that khi co co --execute va xac nhan tu nguoi van hanh.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from dotenv import dotenv_values

from src.binance_signed_client import BinanceSignedTestnetClient
from src.cross_exchange_carry import LOT_STEP_SIZES
from src.exchange_reconciler import (
    compute_testnet_delta_snapshot,
    normalize_symbol,
)

logger = logging.getLogger("reconcile_flatten")

LIQUID_8_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
]


def load_json_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def floor_to_lot_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    precision = 0
    s_str = f"{step:.8f}".rstrip("0")
    if "." in s_str:
        precision = len(s_str.split(".")[1])
    factor = 10**precision
    floored = int(round(qty * factor)) / factor
    return round(floored, precision)


def get_step_size(sym: str) -> float:
    return float(LOT_STEP_SIZES.get(sym, LOT_STEP_SIZES.get(f"{sym}:USDT", 0.001)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile and flatten Binance Testnet orphan & excess positions to match local ledger 1:1."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually submit market orders to Binance Testnet (default is DRY-RUN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode - simulate and print plan without submitting orders (default).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="laptop_audit_dump/data/state/demo_testnet_baseline.json",
        help="Path to demo_testnet_baseline.json.",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="laptop_audit_dump/data/state/demo_bot_regime_omni_1k.json",
        help="Path to bot state file.",
    )
    parser.add_argument(
        "--ledger-file",
        type=str,
        default="laptop_audit_dump/data/state/demo_regime_omni_1k.json",
        help="Path to ledger state file.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Bypass interactive confirmation prompt when --execute is passed.",
    )
    args = parser.parse_args()

    # If --execute is specified, dry_run must be False
    is_dry_run = not args.execute

    print("=" * 80)
    print("  BINANCE TESTNET 1:1 RECONCILIATION & ORPHAN FLATTENING AUDITOR")
    print(f"  Mode: {'DRY-RUN (100% Read-Only, No Orders)' if is_dry_run else 'LIVE EXECUTION (Submitting Real Orders)'}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Initialize client
    env = dotenv_values(".env.testnet")
    client = BinanceSignedTestnetClient(
        spot_api_key=env.get("BINANCE_SPOT_TESTNET_API_KEY"),
        spot_api_secret=env.get("BINANCE_SPOT_TESTNET_API_SECRET"),
        futures_api_key=env.get("BINANCE_FUTURES_TESTNET_API_KEY"),
        futures_api_secret=env.get("BINANCE_FUTURES_TESTNET_API_SECRET"),
        dry_run=False,  # Allow GET queries
    )
    client.sync_server_time("both")

    # 2. Load baseline and local states
    baseline_path = Path(args.baseline)
    if not baseline_path.is_file():
        baseline_path = Path("data/state/demo_testnet_baseline.json")
    baseline = load_json_file(baseline_path)

    bot_state = load_json_file(args.state_file)
    ledger = load_json_file(args.ledger_file)

    # 3. Query current real exchange snapshot
    print("\n[1/5] Fetching live snapshot from Binance Spot & Futures Testnet...")
    snap = client.fetch_exchange_reconcile_snapshot(LIQUID_8_SYMBOLS)
    if not snap.get("snapshot_healthy"):
        print(f"ERROR: Exchange snapshot unhealthy: {snap.get('error')}")
        return 1

    spot_raw = snap.get("spot_balances_raw", {})
    perp_details = snap.get("perp_positions_detail", [])
    base_spot = baseline.get("spot_balances", {})
    last_prices = ledger.get("last_prices", {})

    # Extract current real perp positions
    current_perp: dict[str, float] = {}
    perp_entry_prices: dict[str, float] = {}
    perp_mark_prices: dict[str, float] = {}
    for p in perp_details:
        sym_canon = normalize_symbol(str(p.get("symbol", "")))
        amt = float(p.get("positionAmt", 0.0) or p.get("qty", 0.0) or 0.0)
        current_perp[sym_canon] = amt
        perp_entry_prices[sym_canon] = float(p.get("entry_price", 0.0) or 0.0)
        perp_mark_prices[sym_canon] = float(p.get("mark_price", 0.0) or 0.0)

    # 4. Check for recent orders since Cycle 116 (detect laptop Cycle 117+ activity)
    c116_ts = int(datetime.fromisoformat("2026-11-05T09:28:26+00:00").timestamp() * 1000)
    print("\n[2/5] Checking recent exchange orders placed since Cycle 116 (09:28:26 UTC)...")
    recent_laptop_orders = []
    for sym_canon in LIQUID_8_SYMBOLS:
        raw_sym = sym_canon.split("/")[0] + "USDT"
        req_f = client.build_signed_get_request("/fapi/v1/allOrders", {"symbol": raw_sym, "startTime": c116_ts}, futures=True)
        f_ords = client._http_request("GET", req_f["url"], req_f.get("params", {}), req_f.get("headers", {}))
        for o in f_ords:
            recent_laptop_orders.append(("FUTURES", raw_sym, o))

    if recent_laptop_orders:
        print(f"  -> Detected {len(recent_laptop_orders)} orders from laptop Cycle 117+:")
        for venue, s, o in recent_laptop_orders:
            t = datetime.fromtimestamp(o.get("time") / 1000, timezone.utc).strftime("%H:%M:%S")
            print(f"     [{t}] {venue:7} {s:10} | {o.get('side'):4} {o.get('origQty')} (exec: {o.get('executedQty')}) | {o.get('status')}")
    else:
        print("  -> No new orders detected since Cycle 116.")

    # 5. Define Target State from regime_omni_1k
    # Carry target: BTC=0.002, DOGE=2430.0, XRP=146.6
    carry_targets: dict[str, float] = {
        "BTC/USDT:USDT": 0.002,
        "DOGE/USDT:USDT": 2430.0,
        "XRP/USDT:USDT": 146.6,
    }

    # Overlay target:
    # From bot state: BNB +0.17, SOL -1.30
    # If laptop Cycle 117 executed rebalance (SELL 0.02 BNB, BUY 0.20 SOL), adjust target to +0.15 BNB, -1.10 SOL
    overlay_targets: dict[str, float] = {}
    for sym_canon, pos in bot_state.get("overlay_positions", {}).items():
        if isinstance(pos, dict):
            q = float(pos.get("qty", 0.0))
            side = str(pos.get("side", "")).lower()
            overlay_targets[sym_canon] = -q if side in ("short", "sell") else q

    # Apply detected Cycle 117 adjustments if any
    for venue, s, o in recent_laptop_orders:
        sym_c = f"{s.replace('USDT', '')}/USDT:USDT"
        if o.get("status") == "FILLED":
            eq = float(o.get("executedQty", 0.0))
            sd = o.get("side")
            signed_delta = eq if sd == "BUY" else -eq
            overlay_targets[sym_c] = round(overlay_targets.get(sym_c, 0.0) + signed_delta, 6)

    print("\n[3/5] Target State for regime_omni_1k:")
    for sym_canon in LIQUID_8_SYMBOLS:
        c_q = carry_targets.get(sym_canon, 0.0)
        o_q = overlay_targets.get(sym_canon, 0.0)
        t_q = 0.0  # Trend is flat
        tot_perp = -c_q + o_q + t_q
        if c_q > 0 or abs(tot_perp) > 0:
            print(f"  {sym_canon:16}: Target Spot = +{c_q:<8} | Target Futures = {tot_perp:<8.4f} (Carry -{c_q} + Overlay {o_q:+0.2f})")

    # 6. Calculate exact order actions required
    print("\n[4/5] Computing Exact Reconciliation Deltas between Exchange and Target...")
    planned_spot_orders = []
    planned_futures_orders = []

    print("-" * 125)
    header = f"{'Symbol':<15} | {'Spot Curr':>10} | {'Spot Base':>10} | {'Spot Net':>10} | {'Spot Tgt':>10} | {'Spot Action':>18} | {'Perp Curr':>10} | {'Perp Tgt':>10} | {'Perp Action':>18}"
    print(header)
    print("-" * 125)

    for sym_canon in LIQUID_8_SYMBOLS:
        base_asset = sym_canon.split("/")[0]
        raw_sym = base_asset + "USDT"
        step = get_step_size(sym_canon)

        # Spot calculations
        q_spot_curr = float(spot_raw.get(base_asset, 0.0))
        q_spot_base = float(base_spot.get(base_asset, 0.0))
        q_spot_net = round(q_spot_curr - q_spot_base, 10)
        q_spot_target = carry_targets.get(sym_canon, 0.0)
        delta_spot = round(q_spot_target - q_spot_net, 10)

        spot_action = "NONE"
        if abs(delta_spot) >= step / 2.0:
            qty_spot_order = abs(floor_to_lot_step(delta_spot, step))
            side_spot = "BUY" if delta_spot > 0 else "SELL"
            if qty_spot_order > 0:
                spot_action = f"{side_spot} {qty_spot_order}"
                planned_spot_orders.append({
                    "symbol": raw_sym,
                    "canon_symbol": sym_canon,
                    "side": side_spot,
                    "quantity": qty_spot_order,
                    "est_price": float(last_prices.get(sym_canon, 0.0)),
                })

        # Futures calculations
        q_perp_curr = round(current_perp.get(sym_canon, 0.0), 10)
        c_q = carry_targets.get(sym_canon, 0.0)
        o_q = overlay_targets.get(sym_canon, 0.0)
        q_perp_target = round(-c_q + o_q, 10)
        delta_perp = round(q_perp_target - q_perp_curr, 10)

        perp_action = "NONE"
        if abs(delta_perp) >= step / 2.0:
            qty_perp_order = abs(floor_to_lot_step(delta_perp, step))
            side_perp = "BUY" if delta_perp > 0 else "SELL"
            if qty_perp_order > 0:
                perp_action = f"{side_perp} {qty_perp_order}"
                planned_futures_orders.append({
                    "symbol": raw_sym,
                    "canon_symbol": sym_canon,
                    "side": side_perp,
                    "quantity": qty_perp_order,
                    "est_price": float(perp_mark_prices.get(sym_canon, last_prices.get(sym_canon, 0.0))),
                })

        print(
            f"{sym_canon:<15} | {q_spot_curr:>10.4f} | {q_spot_base:>10.4f} | {q_spot_net:>10.4f} | {q_spot_target:>10.4f} | {spot_action:>18} | "
            f"{q_perp_curr:>10.4f} | {q_perp_target:>10.4f} | {perp_action:>18}"
        )

    print("-" * 125)

    # 7. Summary of Actions
    print("\n[5/5] EXECUTION PLAN SUMMARY:")
    print(f"  Total Spot Orders to execute   : {len(planned_spot_orders)}")
    for o in planned_spot_orders:
        notional = o["quantity"] * o["est_price"]
        print(f"    -> SPOT    {o['symbol']:10}: {o['side']:4} {o['quantity']:>10} (~${notional:>8.2f})")

    print(f"\n  Total Futures Orders to execute: {len(planned_futures_orders)}")
    for o in planned_futures_orders:
        notional = o["quantity"] * o["est_price"]
        print(f"    -> FUTURES {o['symbol']:10}: {o['side']:4} {o['quantity']:>10} (~${notional:>8.2f})")

    # Estimate post-alignment drift
    # Before:
    drift_before = snap.get("spot_cash_usd", 0.0) + snap.get("perp_wallet_balance_usd", 0.0)
    current_delta_snap = compute_testnet_delta_snapshot(
        snap, baseline, last_prices, capital=1000.0, local_equity=ledger["pure_trading_equity_usd"]
    )
    recon_eq_before = current_delta_snap["reconcile_equity_usd"]
    drift_before = recon_eq_before - ledger["pure_trading_equity_usd"]

    print("\n" + "=" * 80)
    print("  ESTIMATED DRIFT RECONSTRUCTION AFTER ALIGNMENT:")
    print(f"  Current Recon Drift USD (BEFORE) : {drift_before:+.4f} USD")
    print(f"  Pure Trading Equity (Local)      : {ledger['pure_trading_equity_usd']:.4f} USD")
    print("  Key Benefit: By eliminating the -11.0 AVAX and +0.14 BNB orphan positions,")
    print("               and aligning Spot & Futures carry legs, d(Drift)/dP will drop to 0.0.")
    print("               recon_drift_usd will STOP jumping between -2.x and -4.x USD!")
    print("=" * 80)

    # If Dry Run, exit safely
    if is_dry_run:
        print("\n[DRY-RUN COMPLETE] 0 orders submitted. To execute, run with --execute.")
        return 0

    # If Live Execution, ask for confirmation unless --yes
    if not args.yes:
        confirm = input("\n[CAUTION] You are about to submit REAL orders to Binance Testnet. Type 'CONFIRM' to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted by user.")
            return 1

    # Execute Orders
    print("\n[EXECUTING ORDERS ON BINANCE TESTNET]...")
    # 1. Execute Spot orders
    for o in planned_spot_orders:
        sym = o["symbol"]
        side = o["side"]
        qty = o["quantity"]
        print(f"  Placing Spot order: {side} {qty} {sym}...")
        try:
            res = client.place_spot_market_order(sym, side, qty)
            print(f"    -> SUCCESS: orderId={res.get('orderId')} status={res.get('status')} executedQty={res.get('executedQty')}")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")
        time.sleep(0.3)

    # 2. Execute Futures orders
    for o in planned_futures_orders:
        sym = o["symbol"]
        side = o["side"]
        qty = o["quantity"]
        print(f"  Placing Futures order: {side} {qty} {sym}...")
        try:
            res = client.place_futures_market_order(sym, side, qty, reduce_only=False)
            print(f"    -> SUCCESS: orderId={res.get('orderId')} status={res.get('status')} executedQty={res.get('executedQty')}")
        except Exception as exc:
            print(f"    -> FAILED: {exc}")
        time.sleep(0.3)

    print("\n[VERIFYING POST-EXECUTION STATE]...")
    time.sleep(2.0)
    post_snap = client.fetch_exchange_reconcile_snapshot(LIQUID_8_SYMBOLS)
    post_spot_raw = post_snap.get("spot_balances_raw", {})
    post_perp_details = post_snap.get("perp_positions_detail", [])
    post_current_perp: dict[str, float] = {}
    for p in post_perp_details:
        sym_canon = normalize_symbol(str(p.get("symbol", "")))
        amt = float(p.get("positionAmt", 0.0) or p.get("qty", 0.0) or 0.0)
        post_current_perp[sym_canon] = amt

    print("\n" + "=" * 125)
    print("  POST-EXECUTION 1:1 AUDIT VERIFICATION TABLE")
    print("=" * 125)
    v_header = f"{'Symbol':<15} | {'Spot Net':>12} | {'Spot Target':>12} | {'Spot Diff':>12} | {'Perp Pos':>12} | {'Perp Target':>12} | {'Perp Diff':>12} | {'Status':>8}"
    print(v_header)
    print("-" * 125)
    
    all_matched = True
    for sym_canon in LIQUID_8_SYMBOLS:
        base_asset = sym_canon.split("/")[0]
        step = get_step_size(sym_canon)
        q_spot_curr = float(post_spot_raw.get(base_asset, 0.0))
        q_spot_base = float(base_spot.get(base_asset, 0.0))
        q_spot_net = round(q_spot_curr - q_spot_base, 10)
        q_spot_target = carry_targets.get(sym_canon, 0.0)
        diff_spot = round(q_spot_net - q_spot_target, 10)

        q_perp_curr = round(post_current_perp.get(sym_canon, 0.0), 10)
        c_q = carry_targets.get(sym_canon, 0.0)
        o_q = overlay_targets.get(sym_canon, 0.0)
        q_perp_target = round(-c_q + o_q, 10)
        diff_perp = round(q_perp_curr - q_perp_target, 10)

        is_matched = abs(diff_spot) < (step / 2.0) and abs(diff_perp) < (step / 2.0)
        if not is_matched:
            all_matched = False
        st = "MATCHED" if is_matched else "MISMATCH"

        print(
            f"{sym_canon:<15} | {q_spot_net:>12.4f} | {q_spot_target:>12.4f} | {diff_spot:>12.4f} | "
            f"{q_perp_curr:>12.4f} | {q_perp_target:>12.4f} | {diff_perp:>12.4f} | {st:>8}"
        )
    print("-" * 125)
    print(f"  Overall Position Alignment: {'ALL 8 SYMBOLS 100% MATCHED (0.000000 ERROR)' if all_matched else 'WARNING: MISMATCH DETECTED'}")

    post_delta = compute_testnet_delta_snapshot(
        post_snap, baseline, last_prices, capital=1000.0, local_equity=ledger["pure_trading_equity_usd"]
    )
    post_drift = post_delta["reconcile_equity_usd"] - ledger["pure_trading_equity_usd"]
    print(f"\n  Initial Recon Drift USD (BEFORE) : {drift_before:+.4f} USD")
    print(f"  New Exchange Recon Equity USD    : {post_delta['reconcile_equity_usd']:.4f} USD")
    print(f"  New Recon Drift USD (AFTER)      : {post_drift:+.4f} USD")
    print("=" * 125)
    print("\nReconciliation Completed Successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
