"""Sprint 23: burn-in 3 chu ky persistent tren Binance TESTNET (Canary 400 USDT) ghi bang chung JSONL."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Sprint 23: them thu muc goc repo TRUOC khi import src.* va run_live.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("canary_persistent_burnin")

SCHEMA = "sprint23.canary_persistent_burnin.v1"
DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT,"
    "DOGE/USDT:USDT,XRP/USDT:USDT,AVAX/USDT:USDT,LINK/USDT:USDT"
)


def _emit(path: str, record: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "schema": SCHEMA}
    rec.update(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _client(dry_run: bool = False) -> Any:
    from src.binance_signed_client import (
        DEFAULT_LIVE_RECV_WINDOW_MS,
        BinanceSignedTestnetClient,
        load_dotenv_testnet,
    )

    load_dotenv_testnet()
    client = BinanceSignedTestnetClient(dry_run=dry_run, recv_window=DEFAULT_LIVE_RECV_WINDOW_MS)
    try:
        client.sync_server_time("both")
    except Exception as exc:  # noqa: BLE001
        logger.warning("sync_server_time loi (%s).", exc)
    return client


def _flat_check(client: Any, symbols: list[str]) -> dict:
    snap = client.fetch_exchange_reconcile_snapshot(symbols, {})
    bases = {str(s).split("/")[0].upper() for s in symbols if "/" in str(s)}
    raw_spot = dict(snap.get("spot_balances") or snap.get("spot_positions") or {})
    # Sprint 23: chi log tai san thuoc universe (testnet faucet cap ~500 dust asset).
    spot = {k: v for k, v in raw_spot.items() if str(k).upper() in bases}
    return {
        "healthy": bool(snap.get("snapshot_healthy", False)),
        "perp": dict(snap.get("perp_positions") or {}),
        "spot": spot,
        "spot_backfilled": list(snap.get("backfilled_price_symbols") or []),
        "equity": float(snap.get("exchange_equity_usd", snap.get("total_equity_usd", 0.0)) or 0.0),
        "error": snap.get("error"),
    }


def _read_state(state_path: str) -> dict:
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _state_positions_count(state: dict) -> int:
    n = 0
    for key in ("carry_positions", "overlay_positions"):
        val = state.get(key)
        if isinstance(val, dict):
            n += len(val)
    return n


def _run_one_cycle(args: argparse.Namespace, symbols: list[str], reset_baseline: bool = False) -> int:
    import run_live
    import src.carry_live_executor as cle

    argv = [
        "--mode", "testnet",
        "--strategy", "barbell",
        "--capital", str(args.capital),
        "--once",
        "--persistent-positions",
        "--state-file", str(args.state_file),
        "--audit-log", str(args.cycle_audit_log),
    ]
    if reset_baseline:
        argv.append("--reset-testnet-baseline")
        # Sprint 23: bootstrap canary -> state sach (equity = capital canary), tranh
        # state cu (3000 USDT) lam lech doi soat baseline-delta.
        argv.append("--reset-state")
    if args.baseline_file:
        argv += ["--testnet-baseline-file", str(args.baseline_file)]

    original = cle.select_portfolio
    if args.force_entry_signal:

        def _forced(*a: Any, **kw: Any) -> Any:
            decision = original(*a, **kw)
            ranked = list(getattr(decision, "ranked_symbols", []) or [])
            top = [str(item[0]) for item in ranked[:2] if isinstance(item, (tuple, list)) and item]
            actions = dict(getattr(decision, "actions", {}) or {})
            reasons = dict(getattr(decision, "reasons", {}) or {})
            for sym in top:
                actions[sym] = "OPEN"
                reasons[sym] = "SPRINT23_FORCE_ENTRY"
            if dataclasses.is_dataclass(decision):
                return dataclasses.replace(decision, actions=actions, reasons=reasons)
            decision.actions = actions
            decision.reasons = reasons
            return decision

        cle.select_portfolio = _forced
    try:
        return int(run_live.main(argv))
    finally:
        cle.select_portfolio = original


def _reconcile_report(
    client: Any, symbols: list[str], state_path: str, capital: float, baseline_path: Optional[str]
) -> dict:
    from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
    from src.exchange_reconciler import (
        DEFAULT_TESTNET_BASELINE_PATH,
        compute_testnet_delta_snapshot,
        load_or_create_testnet_baseline,
        normalize_symbol,
        reconcile_portfolio_state,
    )

    # Sprint 23: co y truyen prices rong de chung minh SpotValuationGuard tu backfill.
    snapshot = client.fetch_exchange_reconcile_snapshot(symbols, {})
    if not bool(snapshot.get("snapshot_healthy", False)):
        raise RuntimeError(f"SNAPSHOT_UNHEALTHY: {snapshot.get('error', 'unknown')}")
    prices: dict[str, float] = {}
    for item in snapshot.get("perp_positions_detail") or []:
        if isinstance(item, dict):
            px = float(item.get("mark_price", item.get("markPrice", 0.0)) or 0.0)
            sym = normalize_symbol(str(item.get("symbol", "")))
            if px > 0.0 and sym:
                prices[sym] = px
    for s in symbols:
        canon = normalize_symbol(str(s))
        if not canon or prices.get(canon, 0.0) > 0.0:
            continue
        tk = client.fetch_book_ticker(canon.split(":")[0].replace("/", "").upper(), venue="spot")
        mid = float((tk or {}).get("mid") or (tk or {}).get("bid") or (tk or {}).get("ask") or 0.0)
        if mid > 0.0:
            prices[canon] = mid

    key_hash = hashlib.sha256(
        f"{os.environ.get('BINANCE_SPOT_TESTNET_API_KEY', '')}:"
        f"{os.environ.get('BINANCE_FUTURES_TESTNET_API_KEY', '')}".encode()
    ).hexdigest()
    baseline = load_or_create_testnet_baseline(
        snapshot=snapshot,
        prices=prices,
        api_key_hash=key_hash,
        universe_symbols=symbols,
        path=baseline_path or DEFAULT_TESTNET_BASELINE_PATH,
    )
    state = _read_state(state_path)
    local_equity = float(state.get("current_equity", capital) or capital)
    delta = compute_testnet_delta_snapshot(snapshot, baseline, prices, capital, local_equity)
    report = dict(reconcile_portfolio_state(state, delta, prices, EXCHANGE_LOT_SPECS))
    report["_snapshot"] = snapshot
    report["_delta"] = delta
    return report


def _status(report: dict) -> str:
    return str(report.get("reconcile_status") or report.get("status") or "")


def _fail(path: str, phase: str, exc: Exception) -> int:
    _emit(path, {"phase": phase, "status": "ERROR", "error": str(exc)})
    logger.error("%s loi: %s", phase, exc)
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 23 canary persistent burn-in (testnet)")
    parser.add_argument("--capital", type=float, default=400.0)
    parser.add_argument(
        "--log-jsonl", "--audit-log", dest="log_jsonl",
        default="logs/sprint23_canary_persistent_burnin.jsonl",
    )
    parser.add_argument("--state-file", default="data/state/bot_state.json")
    parser.add_argument("--baseline-file", default=None)
    parser.add_argument("--reset-testnet-baseline", action="store_true")
    parser.add_argument("--force-entry-signal", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = parser.parse_args(argv)

    if args.capital <= 0:
        parser.error("--capital phai > 0")
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols rong")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_path = str(args.log_jsonl)
    # Sprint 23: audit cua run_live ghi file rieng de bang chung burn-in sach.
    args.cycle_audit_log = str(Path(log_path).with_name(Path(log_path).stem + "_cycle_audit.jsonl"))
    phases: dict[str, bool] = {}

    if args.preflight_only:
        try:
            client = _client()
            flat = _flat_check(client, symbols)
        except Exception as exc:  # noqa: BLE001
            return _fail(log_path, "PREFLIGHT", exc)
        _emit(log_path, {"phase": "PREFLIGHT", **flat})
        print(f"PREFLIGHT equity={flat['equity']:.2f} healthy={flat['healthy']}")
        return 0

    # Phase 1
    phase = "BOOTSTRAP_ENTRY"
    try:
        client = _client()
        flat = _flat_check(client, symbols)
        if any(abs(float(v or 0.0)) > 0.0 for v in flat["perp"].values()):
            _emit(log_path, {"phase": "PRECHECK_FLAT", "status": "FAIL", "perp": flat["perp"]})
            return 2
        rc = _run_one_cycle(args, symbols, reset_baseline=True)
        rep = _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)
        gap_pct = float(rep.get("equity_gap_pct", 0.0) or 0.0)
        pos_count = _state_positions_count(_read_state(args.state_file))
        # Sprint 23: khi ep tin hieu, BAT BUOC co vi the that duoc mo (khong duoc "PASS 0 lenh").
        if bool(args.force_entry_signal) and pos_count <= 0:
            _emit(log_path, {
                "phase": phase, "passed": False, "rc": rc, "status": "NO_POSITION_OPENED",
                "reconcile_status": _status(rep), "state_positions_count": pos_count,
            })
            return 3
        ok = rc == 0 and _status(rep) == "SYNCED" and gap_pct <= 1.5
        phases[phase] = ok
        _emit(log_path, {
            "phase": phase, "passed": ok, "rc": rc, "reconcile_status": _status(rep),
            "equity_gap_usd": rep.get("equity_gap_usd"), "equity_gap_pct": gap_pct,
            "lot_neutrality_passed": rep.get("lot_neutrality_passed"),
            "phantom_symbols": list(rep.get("phantoms", []) or []),
            "drift_items": [
                {"symbol": it.get("symbol"), "status": it.get("status"),
                 "loc_spot": it.get("local_spot_qty"), "ex_spot": it.get("exchange_spot_qty"),
                 "loc_perp": it.get("local_perp_qty"), "ex_perp": it.get("exchange_perp_qty"),
                 "repair": it.get("repair_action")}
                for it in (rep.get("items") or [])
                if str(it.get("status")) not in ("SYNCED", "")
            ][:8],
            "state_positions_count": pos_count,
        })
    except Exception as exc:  # noqa: BLE001
        return _fail(log_path, phase, exc)

    # Phase 2
    phase = "RESTART_HOLD_SYNCED"
    try:
        from src.carry_live_executor import CarryLiveExecutor
        from src.dynamic_carry_basis import DynamicCarryConfig
        from src.state_store import StateStore

        executor = CarryLiveExecutor(
            carry_config=DynamicCarryConfig(
                initial_capital=args.capital, ema_span=42, top_k=4, entry_threshold=0.000045,
                exit_threshold=0.0, min_hold_periods=63, buffer_rank=3, w_basis=0.0,
                basis_ema_hours=168, expected_hold_periods=63,
                switch_rule="tier1b_buffer_rank", eta_override=0.80,
            ),
            mode="testnet",
            state_store=StateStore(args.state_file),
            capital=args.capital,
            initial_capital=args.capital,
            runtime_mode="testnet",
            testnet_dry_state=False,
            execute_real_orders=True,
            signed_client=client,
        )
        rep = _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)
        # Sprint 23: ap report doi soat vao executor MOI (cold restart) truoc khi chu ky tiep tuc.
        try:
            executor.apply_reconcile_report(rep)
        except Exception as _ap_exc:  # noqa: BLE001
            logger.warning("apply_reconcile_report loi (%s).", _ap_exc)
        phantoms = len(rep.get("phantoms", []) or [])
        ok = _status(rep) == "SYNCED" and phantoms == 0
        phases[phase] = ok
        _emit(log_path, {
            "phase": phase, "passed": ok, "reconcile_status": _status(rep),
            "executor_reconcile_status": getattr(executor, "reconcile_status", None),
            "trading_allowed": bool(getattr(executor, "trading_allowed", False)),
            "backfilled_price_symbols": list(rep["_snapshot"].get("backfilled_price_symbols") or []),
            "phantom_count": phantoms,
            "phantom_symbols": list(rep.get("phantoms", []) or []),
            "drift_items": [
                {"symbol": it.get("symbol"), "status": it.get("status"),
                 "loc_spot": it.get("local_spot_qty"), "ex_spot": it.get("exchange_spot_qty"),
                 "loc_perp": it.get("local_perp_qty"), "ex_perp": it.get("exchange_perp_qty"),
                 "repair": it.get("repair_action")}
                for it in (rep.get("items") or [])
                if str(it.get("status")) not in ("SYNCED", "")
            ][:8],
            "state_positions_count": _state_positions_count(_read_state(args.state_file)),
        })
    except Exception as exc:  # noqa: BLE001
        return _fail(log_path, phase, exc)

    # Phase 3
    phase = "EXIT_FLAT_SYNCED"
    try:
        res = executor.close_all_persistent_positions({}) or {}
        rep = _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)
        drift = abs(float(rep.get("equity_gap_usd", 0.0) or 0.0))
        zero = bool(res.get("zero_residual", False)) if isinstance(res, dict) else False
        ok = zero and _status(rep) == "SYNCED" and drift <= 2.0
        phases[phase] = ok
        _emit(log_path, {
            "phase": phase, "passed": ok,
            "closed_carry": res.get("closed_carry") if isinstance(res, dict) else None,
            "closed_overlay": res.get("closed_overlay") if isinstance(res, dict) else None,
            "zero_residual": zero, "reconcile_status": _status(rep),
            "abs_equity_drift_vs_initial_usd": drift,
        })
    except Exception as exc:  # noqa: BLE001
        return _fail(log_path, phase, exc)

    passed = all(phases.values()) and len(phases) == 3
    _emit(log_path, {"phase": "SUMMARY", "passed": passed, "phases": phases})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
