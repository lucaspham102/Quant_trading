"""Sprint 24: preflight audit READ-ONLY cho Canary split-wallet (khong dat lenh)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("canary_preflight_audit")

SCHEMA = "sprint24.canary_preflight_audit.v1"
DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT,"
    "XRP/USDT:USDT,LINK/USDT:USDT,DOGE/USDT:USDT,AVAX/USDT:USDT"
)


def _emit(path: str, record: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "schema": SCHEMA}
    rec.update(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _build_client(args: argparse.Namespace, log_path: str) -> Optional[Any]:
    from src.binance_signed_client import (
        DEFAULT_LIVE_RECV_WINDOW_MS,
        BinanceSignedTestnetClient,
        load_dotenv_testnet,
    )

    if args.mode == "testnet":
        try:
            load_dotenv_testnet()
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_dotenv_testnet loi (%s).", exc)
        client = BinanceSignedTestnetClient(dry_run=False, recv_window=DEFAULT_LIVE_RECV_WINDOW_MS)
    else:
        spot_key = os.environ.get("BINANCE_SPOT_API_KEY", "")
        fut_key = os.environ.get("BINANCE_FUTURES_API_KEY", "")
        if not spot_key or not fut_key:
            _emit(log_path, {
                "phase": "CREDENTIALS", "status": "FAIL", "mode": args.mode,
                "error": "thieu BINANCE_SPOT_API_KEY / BINANCE_FUTURES_API_KEY",
            })
            return None
        # Mode live: TUYET DOI khong nap .env.testnet.
        client = BinanceSignedTestnetClient(
            dry_run=False,
            recv_window=DEFAULT_LIVE_RECV_WINDOW_MS,
            spot_base_url=os.environ.get("BINANCE_MAINNET_SPOT_BASE_URL", "https://api.binance.com"),
            perp_base_url=os.environ.get("BINANCE_MAINNET_PERP_BASE_URL", "https://fapi.binance.com"),
            spot_api_key=spot_key,
            spot_api_secret=os.environ.get("BINANCE_SPOT_API_SECRET", ""),
            futures_api_key=fut_key,
            futures_api_secret=os.environ.get("BINANCE_FUTURES_API_SECRET", ""),
            autoload_dotenv=False,
        )
    try:
        client.sync_server_time("both")
    except Exception as exc:  # noqa: BLE001
        logger.warning("sync_server_time loi (%s).", exc)
    return client


def _assert_endpoint_safety(client: Any, mode: str) -> None:
    """Sprint 24: chot chan bao mat - mode testnet phai tro testnet, mode live KHONG duoc tro testnet."""
    spot_url = str(getattr(client, "spot_base_url", "") or "").lower()
    perp_url = str(getattr(client, "perp_base_url", "") or "").lower()
    if mode == "testnet":
        if "testnet" not in spot_url or "testnet" not in perp_url:
            raise RuntimeError(
                f"CRITICAL_SECURITY_ABORT: mode testnet nhung endpoint khong phai testnet ({spot_url} | {perp_url})"
            )
    else:
        if "testnet" in spot_url or "testnet" in perp_url:
            raise RuntimeError(
                f"CRITICAL_SECURITY_ABORT: mode live nhung endpoint tro testnet ({spot_url} | {perp_url})"
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 24 canary preflight audit (read-only)")
    parser.add_argument("--mode", choices=["testnet", "live"], default="testnet")
    parser.add_argument("--capital", type=float, default=400.0)
    parser.add_argument("--carry-eta", type=float, default=0.65)
    parser.add_argument("--log-jsonl", default="logs/sprint24_canary_preflight_audit.jsonl")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.capital <= 0:
        parser.error("--capital phai > 0")
    if not (0.0 < args.carry_eta < 1.0):
        parser.error("--carry-eta phai trong (0, 1)")
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols rong")
    log_path = str(args.log_jsonl)

    from src.quant_invariants import compute_split_wallet_canary_plan, validate_wallet_split_readiness

    try:
        client = _build_client(args, log_path)
    except Exception as exc:  # noqa: BLE001
        _emit(log_path, {"phase": "CLIENT", "status": "FAIL", "mode": args.mode, "error": str(exc)})
        return 2
    if client is None:
        return 2
    try:
        _assert_endpoint_safety(client, args.mode)
    except Exception as exc:  # noqa: BLE001
        logger.critical("Endpoint safety FAIL: %s", exc)
        _emit(log_path, {"phase": "ENDPOINT_SAFETY", "status": "FAIL", "mode": args.mode, "error": str(exc)})
        return 2

    try:
        snapshot = client.fetch_exchange_reconcile_snapshot(symbols, {})
    except Exception as exc:  # noqa: BLE001
        _emit(log_path, {"phase": "SNAPSHOT", "status": "FAIL", "mode": args.mode, "error": str(exc)})
        return 2
    snapshot_healthy = bool(snapshot.get("snapshot_healthy", False))

    pure_cash = float(snapshot.get("pure_cash_usd") or snapshot.get("spot_cash_usd") or 0.0)
    # Luu y: day la wallet balance cua Futures, KHONG phai availableMargin.
    futures_wallet = float(snapshot.get("perp_wallet_balance_usd") or 0.0)

    bnb_qty = float((snapshot.get("spot_balances") or {}).get("BNB", 0.0) or 0.0)
    try:
        # Spot ticker can cap base+quote (BNBUSDT), khong phai "BNB".
        bnb_price = float((client.fetch_book_ticker("BNBUSDT", venue="spot") or {}).get("mid") or 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_book_ticker BNB loi (%s).", exc)
        bnb_price = 0.0
    bnb_usd = round(bnb_qty * bnb_price, 2)
    bnb_low = bnb_usd < 5.0

    perp_positions = snapshot.get("perp_positions") or {}
    nonflat = {s: q for s, q in perp_positions.items() if abs(float(q or 0.0)) > 0.0}
    perp_flat = len(nonflat) == 0

    plan = compute_split_wallet_canary_plan(args.capital, args.carry_eta)
    readiness = validate_wallet_split_readiness(pure_cash, futures_wallet, plan)

    # PASS: healthy + perp flat + du so du + du BNB buffer. WARN: chi thieu BNB hoac perp chua flat.
    if snapshot_healthy and perp_flat and readiness["ready"] and not bnb_low:
        status = "PASS"
    elif snapshot_healthy and readiness["ready"] and (bnb_low or not perp_flat):
        status = "WARN"
    else:
        status = "FAIL"

    _emit(log_path, {
        "mode": args.mode,
        "capital": args.capital,
        "status": status,
        "snapshot_healthy": snapshot_healthy,
        "pure_cash_usd": pure_cash,
        "futures_wallet_usd": futures_wallet,
        "bnb_usd": bnb_usd,
        "bnb_fee_buffer_low": bnb_low,
        "perp_positions_flat": perp_flat,
        "nonflat_positions": nonflat,
        "plan": plan,
        "readiness": readiness,
    })

    rows = [
        ("MODE", args.mode),
        ("CAPITAL", f"{args.capital:.2f} USDT"),
        ("SPOT CASH", f"{pure_cash:.2f} USDT (need {readiness['spot_required_usdt']:.2f})"),
        ("FUTURES WALLET", f"{futures_wallet:.2f} USDT (wallet balance, need "
                           f"{readiness['futures_required_usdt']:.2f})"),
        ("BNB", f"{bnb_usd:.2f} USD{' (LOW)' if bnb_low else ''}"),
        ("PERP FLAT", str(perp_flat)),
        ("LEVERAGE", f"{plan['effective_futures_leverage']:.2f}x safe={plan['is_safe_leverage']}"),
        ("SPOT SHORTFALL", f"{readiness['spot_shortfall_usd']:.2f} USDT"),
        ("FUTURES SHORTFALL", f"{readiness['futures_shortfall_usd']:.2f} USDT"),
        ("STATUS", status),
    ]
    print("=" * 60)
    for key, val in rows:
        print(f"{key:<18}: {val}"[:100])
    print("=" * 60)
    return 0 if status in ("PASS", "WARN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
