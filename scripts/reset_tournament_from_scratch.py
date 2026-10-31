"""Reset the 4-profile tournament to Cycle 0 and flatten all Testnet positions.

Steps performed:
1. Connect to Binance Spot & USD-M Futures Testnet.
2. Close all open Perpetual Futures positions across the canonical universe.
3. Sell any Spot Carry excess above the initial faucet baseline back to USDT.
4. Remove all demo ledger files, bot state files, `.bak` backups, heartbeat, and scoreboard logs.
5. Snapshot a fresh `data/state/demo_testnet_baseline.json` with 0 open positions.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cross_exchange_carry import quantize_order_qty
from scripts.run_parallel_live_demo import (
    LIQUID_8_SYMBOLS,
    build_signed_client_or_none,
    demo_testnet_baseline_path,
    initialize_demo_testnet_baseline,
)


def _clean_symbol_pair(sym: str) -> str:
    return sym.split(":")[0].replace("/", "").replace("_", "").upper()


def reset_tournament(state_dir: str = "data/state", log_dir: str = "logs") -> int:
    print("=== [1/4] KET NOI BINANCE TESTNET & KIEM TRA VI THE HIEN TAI ===")
    client, note = build_signed_client_or_none()
    if client is None:
        print(f"ERROR: Khong the khoi tao signed client ({note}). Kiem tra file .env!")
        return 1

    baseline_path = demo_testnet_baseline_path(state_dir)
    old_spot_baseline: dict[str, float] = {}
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_spot_baseline = {
                str(k): float(v)
                for k, v in (old_data.get("spot_balances") or {}).items()
                if v is not None
            }
        except Exception as exc:
            print(f"WARNING: Khong doc duoc baseline cu ({exc})")

    # Also read live profile ledger spot carry positions if present
    ledger_spot_carry: dict[str, float] = {}
    live_ledger_file = os.path.join(state_dir, "demo_regime_omni_1k.json")
    if os.path.exists(live_ledger_file):
        try:
            with open(live_ledger_file, "r", encoding="utf-8") as f:
                lg_data = json.load(f)
            for pos in (lg_data.get("positions_snapshot") or {}).values():
                if isinstance(pos, dict) and str(pos.get("kind", "carry")) == "carry":
                    sym = str(pos.get("symbol") or "")
                    sq = float(pos.get("spot_qty") or pos.get("qty") or 0.0)
                    if sym and sq > 0:
                        base_asset = sym.split("/")[0].upper()
                        ledger_spot_carry[base_asset] = ledger_spot_carry.get(base_asset, 0.0) + sq
        except Exception:
            pass

    snap = client.fetch_exchange_reconcile_snapshot(list(LIQUID_8_SYMBOLS), {})
    if not snap.get("snapshot_healthy"):
        print("ERROR: Snapshot san Testnet khong healthy.")
        return 1

    perp_positions: dict[str, float] = snap.get("perp_positions") or {}
    spot_balances: dict[str, float] = snap.get("spot_balances") or {}

    print(f"-> Vi the Perp dang mo: {perp_positions}")

    print("\n=== [2/4] DONG TOAN BO VI THE FUTURES PERP & BAN SPOT CARRY DU THUA ===")
    # 1. Close all open Perp positions
    for sym, qty in list(perp_positions.items()):
        q = float(qty or 0.0)
        if abs(q) <= 1e-12:
            continue
        side = "BUY" if q < 0 else "SELL"
        close_qty, _ = quantize_order_qty(sym, abs(q), price=100000.0)
        if close_qty <= 0:
            close_qty = abs(q)
        pair = _clean_symbol_pair(sym)
        print(f"  [PERP CLOSE] {side} {close_qty} {pair} (current={q})...")
        try:
            res = client.submit_signed_order(
                venue="futures",
                params={
                    "symbol": pair,
                    "side": side,
                    "type": "MARKET",
                    "quantity": close_qty,
                },
            )
            print(f"    -> OK: status={res.get('status')} executedQty={res.get('executedQty')}")
        except Exception as exc:
            print(f"    -> ERROR closing perp {pair}: {exc}")

    # 2. Sell any Spot Carry excess above old_spot_baseline (or recorded in ledger_spot_carry)
    for sym in LIQUID_8_SYMBOLS:
        base_asset = sym.split("/")[0].upper()
        cur_bal = float(spot_balances.get(base_asset) or 0.0)
        base_bal = float(old_spot_baseline.get(base_asset) or cur_bal)
        excess = max(cur_bal - base_bal, float(ledger_spot_carry.get(base_asset) or 0.0))
        sell_qty, _ = quantize_order_qty(sym, max(0.0, excess), price=100000.0)
        if sell_qty > 0:
            pair = _clean_symbol_pair(sym)
            print(
                f"  [SPOT SELL EXCESS] SELL {sell_qty} {pair} "
                f"(current={cur_bal}, baseline={base_bal})..."
            )
            try:
                res = client.submit_signed_order(
                    venue="spot",
                    params={
                        "symbol": pair,
                        "side": "SELL",
                        "type": "MARKET",
                        "quantity": sell_qty,
                    },
                )
                print(f"    -> OK: status={res.get('status')} executedQty={res.get('executedQty')}")
            except Exception as exc:
                print(f"    -> WARNING selling spot excess {pair}: {exc}")

    # Verify post-flatten snapshot
    snap_after = client.fetch_exchange_reconcile_snapshot(list(LIQUID_8_SYMBOLS), {})
    remaining_perp = snap_after.get("perp_positions") or {}
    print(f"-> Vi the Perp sau khi dong: {remaining_perp}")
    if remaining_perp:
        print("ERROR: Van con vi the Perp chua dong het!")
        return 1

    print("\n=== [3/4] XOA TOAN BO FILE STATE CU & LOG GIAI DAU CU ===")
    patterns = [
        os.path.join(state_dir, "demo_*.json"),
        os.path.join(state_dir, "demo_*.json.bak"),
        os.path.join(state_dir, "demo_*.json.tmp"),
        os.path.join(state_dir, "testnet_baseline.json"),
        os.path.join(state_dir, "testnet_baseline.json.bak"),
        os.path.join(state_dir, "tournament_heartbeat.json"),
        os.path.join(state_dir, "tournament_heartbeat.json.bak"),
        os.path.join(log_dir, "demo_scoreboard.jsonl"),
        os.path.join(log_dir, "demo_Scoreboard.jsonl"),
        os.path.join(log_dir, "demo_summary.md"),
    ]
    removed_count = 0
    for pat in patterns:
        for fpath in glob.glob(pat):
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    removed_count += 1
                    print(f"  [DELETED] {fpath}")
                except OSError as exc:
                    print(f"  [WARN] Could not remove {fpath}: {exc}")
    print(f"-> Da xoa {removed_count} file state/log cu.")

    print("\n=== [4/4] KHOI TAO BASELINE MOI TINH (CYCLE 0 CLEAN STATE) ===")
    ok, base_note = initialize_demo_testnet_baseline(
        client,
        baseline_path,
        symbols=list(LIQUID_8_SYMBOLS),
    )
    print(f"-> Baseline moi: ok={ok}, status={base_note}, path={baseline_path}")
    if not ok:
        print("ERROR: Khong the tao baseline moi.")
        return 1

    print("\n=====================================================================")
    print("HOAN TAT RESET 100%! Tat ca vi the da duoc dong va State da ve Cycle 0.")
    print("Khi chay giai dau se bat dau moi tinh tu Cycle #1 voi $1,000.00!")
    print("=====================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(reset_tournament())
