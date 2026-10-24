"""Sprint 24: burn-in 3 phase overlay persistent tren Binance TESTNET (Canary 400 USDT)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_canary_persistent_burnin import (  # noqa: E402
    _client,
    _flat_check,
    _read_state,
    _reconcile_report,
    _status,
)

logger = logging.getLogger("overlay_persistent_burnin")

SCHEMA = "sprint24.overlay_persistent_burnin.v1"
DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT,"
    "XRP/USDT:USDT,LINK/USDT:USDT,DOGE/USDT:USDT,AVAX/USDT:USDT"
)
# Tin hieu tat dinh: 4 symbol (tong |w| = 1.0) + BTC raw de chung minh filter loai BTC.
OVERLAY_WEIGHTS = {
    "ETH/USDT:USDT": 0.35,
    "SOL/USDT:USDT": 0.15,
    "DOGE/USDT:USDT": -0.35,
    "XRP/USDT:USDT": -0.15,
}
BTC_SYMBOL = "BTC/USDT:USDT"
BTC_RAW_WEIGHT = 0.10


def _emit(path: str, record: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "schema": SCHEMA}
    rec.update(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _fail(path: str, phase: str, exc: Exception) -> int:
    _emit(path, {"phase": phase, "status": "ERROR", "error": str(exc)})
    logger.error("%s loi: %s", phase, exc)
    return 1


def _fetch_prices(client: Any, symbols: list[str]) -> dict[str, float]:
    """Gia tu mark_price trong snapshot, thieu thi lay mid book ticker spot."""
    from src.exchange_reconciler import normalize_symbol

    snap = client.fetch_exchange_reconcile_snapshot(symbols, {})
    prices: dict[str, float] = {}
    for item in snap.get("perp_positions_detail") or []:
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
    return prices


def _lot_steps(specs: dict) -> dict[str, float]:
    """Chuan hoa EXCHANGE_LOT_SPECS (so hoac dict) ve {symbol: step} cho filter."""
    out: dict[str, float] = {}
    for sym, spec in specs.items():
        if isinstance(spec, dict):
            spec = spec.get("step_size", spec.get("step", spec.get("qty_step", 0.0)))
        try:
            step = float(spec)
        except (TypeError, ValueError):
            continue
        if step > 0.0:
            out[sym] = step
    return out


def _side_notional(targets: dict[str, float], prices: dict[str, float], sign: int) -> float:
    """Tong notional TUYET DOI cua mot phia (long sign=1, short sign=-1)."""
    return float(
        abs(sum(q * prices[s] for s, q in targets.items() if (q > 0.0) == (sign > 0)))
    )


def _balance_targets(
    targets: dict[str, float], prices: dict[str, float], lot_steps: dict[str, float], tol_usd: float = 2.0
) -> dict[str, float]:
    """Sprint 24: sau quantize, bu sai so lot step de |Long - Short| <= tol_usd.

    Uu tien dieu chinh chan co granularity (step*price) nho nhat; neu khong the tang
    chan thieu thi giam chan thua. Deterministic, khong dung random.
    """
    import math

    out = dict(targets)
    prev_abs_diff = float("inf")
    for _ in range(6):
        long_t = _side_notional(out, prices, 1)
        short_t = _side_notional(out, prices, -1)
        diff = long_t - short_t
        if abs(diff) <= tol_usd:
            break
        if abs(diff) >= prev_abs_diff:
            break  # khong cai thien duoc nua -> dung (tranh chay loan)
        prev_abs_diff = abs(diff)
        short_deficient = diff > 0.0  # long nhieu hon -> thieu Short
        cands = [
            (sym, q) for sym, q in out.items() if (q < 0.0) == short_deficient
        ]
        cands.sort(key=lambda sq: lot_steps.get(sq[0], 1e-8) * float(prices.get(sq[0], 0.0)))
        adjusted = False
        for sym, q in cands:
            px = float(prices.get(sym, 0.0) or 0.0)
            step = float(lot_steps.get(sym, 0.0) or 0.0)
            if px <= 0.0 or step <= 0.0:
                continue
            add_steps = math.floor((abs(diff) / px) / step)
            if add_steps <= 0:
                continue
            add = add_steps * step
            out[sym] = q - add if q < 0.0 else q + add
            adjusted = True
            break
        if adjusted:
            continue
        # Khong tang duoc chan thieu -> giam chan thua (giu qty > 0).
        excess_short = not short_deficient
        ex_cands = [(sym, q) for sym, q in out.items() if (q < 0.0) == excess_short]
        ex_cands.sort(key=lambda sq: lot_steps.get(sq[0], 1e-8) * float(prices.get(sq[0], 0.0)))
        for sym, q in ex_cands:
            px = float(prices.get(sym, 0.0) or 0.0)
            step = float(lot_steps.get(sym, 0.0) or 0.0)
            if px <= 0.0 or step <= 0.0:
                continue
            cut_steps = math.floor((abs(diff) / px) / step)
            if cut_steps <= 0:
                continue
            cut = cut_steps * step
            new_q = q + cut if q < 0.0 else q - cut
            if abs(new_q) <= 0.0:
                continue
            out[sym] = new_q
            adjusted = True
            break
        if not adjusted:
            break
    return out


def _new_executor(args: argparse.Namespace, client: Any, reset: bool) -> Any:
    from src.carry_live_executor import CarryLiveExecutor
    from src.dynamic_carry_basis import DynamicCarryConfig
    from src.state_store import StateStore

    # reset=True: constructor goi state_store.load(reset=True) roi save -> state sach (equity = capital).
    return CarryLiveExecutor(
        carry_config=DynamicCarryConfig(initial_capital=args.capital, top_k=4, eta_override=0.80),
        mode="testnet",
        capital=args.capital,
        initial_capital=args.capital,
        state_store=StateStore(Path(args.state_file)),
        runtime_mode="testnet",
        testnet_dry_state=False,
        execute_real_orders=True,
        signed_client=client,
        persistent_positions=True,
        reset_state=reset,
    )


def _overlay_sides(state: dict) -> tuple[bool, bool]:
    """Tra ve (co_long, co_short) tu overlay_positions trong state tren dia."""
    has_long = has_short = False
    pos = state.get("overlay_positions")
    for val in (pos.values() if isinstance(pos, dict) else []):
        if isinstance(val, dict):
            side = str(val.get("side", "")).lower()
            qty = float(val.get("qty", val.get("quantity", val.get("perp_qty", 0.0))) or 0.0)
            is_long = side == "long" or (not side and qty > 0.0)
            is_short = side == "short" or (not side and qty < 0.0)
        else:
            qty = float(val or 0.0)
            is_long, is_short = qty > 0.0, qty < 0.0
        has_long = has_long or is_long
        has_short = has_short or is_short
    return has_long, has_short


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 24 overlay persistent burn-in (testnet)")
    parser.add_argument("--capital", type=float, default=400.0)
    # LUU Y: 0.30 la muc STRESS chi dung cho burn-in co che tren von 400 USDT (de 4 leg
    # deu > min_notional 10 USD). Con so TUAN THU leverage la 0.15 (xem
    # compute_split_wallet_canary_plan -> 2.29x). Khong dung 0.30 cho canary that.
    parser.add_argument("--overlay-gross-pct", type=float, default=0.30)
    parser.add_argument("--log-jsonl", default="logs/sprint24_overlay_persistent_burnin.jsonl")
    parser.add_argument("--state-file", default="data/state/bot_state.json")
    parser.add_argument("--baseline-file", default=None)
    parser.add_argument(
        "--reset-testnet-baseline", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    args = parser.parse_args(argv)
    if args.capital <= 0:
        parser.error("--capital phai > 0")
    if not (0.0 < args.overlay_gross_pct <= 1.0):
        parser.error("--overlay-gross-pct phai trong (0, 1]")
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols rong")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log_path = str(args.log_jsonl)
    phases: dict[str, bool] = {}

    from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
    from src.exchange_reconciler import DEFAULT_TESTNET_BASELINE_PATH
    from src.quant_invariants import filter_canary_lot_unviable_overlay_symbols

    # Phase 1
    phase = "OVERLAY_BOOTSTRAP_ENTRY"
    try:
        client = _client()
        flat = _flat_check(client, symbols)
        if any(abs(float(v or 0.0)) > 0.0 for v in flat["perp"].values()):
            _emit(log_path, {"phase": "PRECHECK_FLAT", "status": "FAIL", "perp": flat["perp"]})
            return 2
        if args.reset_testnet_baseline:
            Path(args.baseline_file or DEFAULT_TESTNET_BASELINE_PATH).unlink(missing_ok=True)
        executor = _new_executor(args, client, reset=True)
        # Tao baseline luc con flat (TRUOC khi gui lenh) de doi soat delta ve sau.
        _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)

        prices = _fetch_prices(client, symbols)
        gross_usd = args.capital * args.overlay_gross_pct
        weights = dict(OVERLAY_WEIGHTS)
        weights[BTC_SYMBOL] = BTC_RAW_WEIGHT
        filtered = filter_canary_lot_unviable_overlay_symbols(
            weights, gross_usd / 2.0, prices, _lot_steps(EXCHANGE_LOT_SPECS)
        )
        # Khong bao gio gui lenh BTC; chuan hoa lai moi phia ve 0.5 de giu trung tinh USD.
        filtered.pop(BTC_SYMBOL, None)
        long_sum = sum(w for w in filtered.values() if w > 0.0)
        short_sum = sum(-w for w in filtered.values() if w < 0.0)
        targets: dict[str, float] = {}
        for sym, w in filtered.items():
            side_sum = long_sum if w > 0.0 else short_sum
            if side_sum <= 0.0 or w == 0.0 or prices.get(sym, 0.0) <= 0.0:
                continue
            notional = abs(w) / side_sum * 0.5 * gross_usd
            qty, _ = executor.quantize_leg(sym, notional, prices[sym])
            if qty <= 0.0:
                continue
            targets[sym] = qty if w > 0.0 else -qty
        if not targets:
            raise RuntimeError("Khong co target overlay hop le sau filter/quantize")
        targets = _balance_targets(targets, prices, _lot_steps(EXCHANGE_LOT_SPECS))

        res = executor._execute_overlay_deltas_real(
            targets, dict(executor.state.overlay_positions), prices
        )
        res = res if isinstance(res, dict) else {}
        res_status = str(res.get("status", ""))
        executor.state_store.save(executor.state)
        live_pos = dict(getattr(executor.state, "overlay_positions", {}) or {})
        net_delta = 0.0
        for _sym, _pos in live_pos.items():
            _q = float(_pos.get("qty", 0.0)) if isinstance(_pos, dict) else float(_pos or 0.0)
            _side = str(_pos.get("side", "")).lower() if isinstance(_pos, dict) else ""
            _signed = -abs(_q) if _side == "short" else abs(_q)
            net_delta += _signed * float(prices.get(_sym, 0.0) or 0.0)
        rep = _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)
        state = _read_state(args.state_file)
        has_long, has_short = _overlay_sides(state)
        btc_in_state = BTC_SYMBOL in (state.get("overlay_positions") or {})
        ok = (
            res_status == "OK"
            and abs(net_delta) <= 10.0
            and _status(rep) == "SYNCED"
            and has_long
            and has_short
            and not btc_in_state
        )
        phases[phase] = ok
        _emit(log_path, {
            "phase": phase, "passed": ok, "exec_status": res_status,
            "overlay_gross_pct": float(getattr(args, "overlay_gross_pct", 0.0) or 0.0),
            "targets": targets, "rollback_qty": res.get("rollback_qty"),
            "net_delta_usd": net_delta,
            "btc_filtered": BTC_SYMBOL not in targets, "btc_in_state": btc_in_state,
            "has_long": has_long, "has_short": has_short,
            "reconcile_status": _status(rep),
            "equity_gap_usd": rep.get("equity_gap_usd"),
        })
    except Exception as exc:  # noqa: BLE001
        return _fail(log_path, phase, exc)

    # Phase 2
    phase = "OVERLAY_RESTART_HOLD_SYNCED"
    try:
        executor = _new_executor(args, client, reset=False)
        rep = _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)
        try:
            executor.apply_reconcile_report(rep)
        except Exception as ap_exc:  # noqa: BLE001
            logger.warning("apply_reconcile_report loi (%s).", ap_exc)
        phantoms = list(rep.get("phantoms") or [])
        state = _read_state(args.state_file)
        ok = (
            _status(rep) == "SYNCED"
            and len(phantoms) == 0
            and bool(getattr(executor, "trading_allowed", False))
            and bool(state.get("invariants_passed", True))
        )
        phases[phase] = ok
        _emit(log_path, {
            "phase": phase, "passed": ok, "reconcile_status": _status(rep),
            "trading_allowed": bool(getattr(executor, "trading_allowed", False)),
            "phantom_symbols": phantoms,
            "invariants_passed": bool(state.get("invariants_passed", True)),
        })
    except Exception as exc:  # noqa: BLE001
        return _fail(log_path, phase, exc)

    # Phase 3
    phase = "OVERLAY_EXIT_FLAT_SYNCED"
    try:
        prices = _fetch_prices(client, symbols)
        res = executor.close_all_persistent_positions(prices) or {}
        flat = _flat_check(client, symbols)
        rep = _reconcile_report(client, symbols, args.state_file, args.capital, args.baseline_file)
        perp_flat = all(abs(float(q or 0.0)) <= 0.0 for q in flat["perp"].values())
        state = _read_state(args.state_file)
        equity = float(
            rep.get("reconcile_equity", rep.get("local_equity", state.get("current_equity", args.capital)))
            or args.capital
        )
        drift = abs(equity - args.capital)
        zero = bool(res.get("zero_residual", False)) if isinstance(res, dict) else False
        ok = zero and perp_flat and _status(rep) == "SYNCED" and drift <= 2.0
        phases[phase] = ok
        _emit(log_path, {
            "phase": phase, "passed": ok, "zero_residual": zero, "perp_flat": perp_flat,
            "reconcile_status": _status(rep), "equity_drift_usd": drift,
        })
    except Exception as exc:  # noqa: BLE001
        return _fail(log_path, phase, exc)

    passed = all(phases.values()) and len(phases) == 3
    _emit(log_path, {"phase": "SUMMARY", "passed": passed, "phases": phases})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
