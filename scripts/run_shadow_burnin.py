"""Shadow-live burn-in Sprint 15 cho barbell Tier 1B + overlay smooth_conviction.

Moi chu ky:
1. Lay snapshot orderbook (Binance public, read-only) - ``--offline-fixture`` dung
   ``data/shadow_fixture_example.json`` (khong goi mang).
2. Kiem tra feed: du symbol, ``best_bid < best_ask``, spread <= 10 bps.
3. Chay 1 vong ``CarryLiveExecutor.reconcile_once`` (paper, tier1b_gate=True).
4. Kiem tra state: hedge khop (leg2_hedged_qty == leg1_filled_qty), khong am so du,
   dust duoc danh dau.
5. Kiem tra risk: drawdown < 6%, kill-switch khong bat.
6. Ghi PASS/FAIL tung chu ky; exit 1 neu co FAIL va ``--strict``.

Chay:
  python scripts/run_shadow_burnin.py --cycles 5 --interval-sec 2 --offline-fixture
  python scripts/run_shadow_burnin.py --cycles 3            (mang that, read-only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.barbell_portfolio import BarbellConfig  # noqa: E402
from src.binance_public_feed import fetch_binance_shadow_snapshot  # noqa: E402
from src.carry_live_executor import CarryLiveExecutor  # noqa: E402
from src.config import TradingConfig  # noqa: E402
from src.cross_exchange_carry import LIQUID_8_SYMBOLS  # noqa: E402
from src.dynamic_carry_basis import DynamicCarryConfig  # noqa: E402
from src.state_store import StateStore  # noqa: E402

FIXTURE_PATH = _ROOT / "data" / "shadow_fixture_example.json"
DEFAULT_STATE = _ROOT / "data" / "shadow_burnin_state.json"
MAX_SPREAD_BPS = 10.0
MAX_DRAWDOWN_PCT = 6.0
DUST_NOTIONAL_USD = 10.0


def tier1b_carry_config(capital: float) -> DynamicCarryConfig:
    """Cau hinh carry y het Frozen Alpha Sprint 11 (khop run_live.py nhanh barbell)."""
    return DynamicCarryConfig(
        initial_capital=float(capital),
        ema_span=42,
        top_k=4,
        entry_threshold=0.000045,
        exit_threshold=0.0,
        min_hold_periods=63,
        buffer_rank=3,
        w_basis=0.0,
        basis_ema_hours=168,
        expected_hold_periods=63,
        switch_rule="tier1b_buffer_rank",
        eta_override=0.80,
    )


def build_burnin_executor(
    capital: float,
    state_path: Path,
    raw_dir: Path,
    symbols: Sequence[str] = LIQUID_8_SYMBOLS,
) -> CarryLiveExecutor:
    """Executor paper barbell Tier 1B + overlay smooth_conviction, khong goi mang."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    config = TradingConfig(
        timeframe="1h",
        raw_data_dir=raw_dir,
        state_file_path=state_path,
    )
    return CarryLiveExecutor(
        config=config,
        carry_config=tier1b_carry_config(capital),
        mode="paper",
        symbols=list(symbols),
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
        capital=float(capital),
        initial_capital=float(capital),
        overlay_config=BarbellConfig(
            initial_capital=float(capital),
            carry_variant="tier1b",
            gating_mode="smooth_conviction",
        ),
        strategy="barbell",
        tier1b_gate=True,
    )


def check_feed(snapshot: dict[str, Any], symbols: Sequence[str]) -> list[str]:
    """Tra ve danh sach loi feed; rong nghia la PASS."""
    problems: list[str] = []
    if not isinstance(snapshot, dict):
        return ["FEED_UNAVAILABLE"]
    if not snapshot.get("healthy"):
        problems.append("FEED_UNHEALTHY")
    books = snapshot.get("symbols") or {}
    missing = [s for s in symbols if s not in books]
    if missing:
        problems.append(f"FEED_MISSING_SYMBOLS:{len(missing)}")
    for sym, book in books.items():
        try:
            bid = float(book.get("bid", 0.0))
            ask = float(book.get("ask", 0.0))
            spread_bps = float(book.get("spread_bps", 0.0))
        except (TypeError, ValueError):
            problems.append(f"BOOK_INVALID:{sym}")
            continue
        if bid <= 0.0 or ask <= 0.0 or bid >= ask:
            problems.append(f"BOOK_CROSSED:{sym}")
            continue
        if spread_bps > MAX_SPREAD_BPS:
            problems.append(f"SPREAD_TOO_WIDE:{sym}={spread_bps:.2f}bps")
    return problems


def check_state(executor: CarryLiveExecutor, capital: float) -> list[str]:
    """Kiem tra sanity state sau moi chu ky."""
    problems: list[str] = []
    positions = getattr(executor.state, "carry_positions", {}) or {}
    for sym, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        leg1 = float(pos.get("leg1_filled_qty", pos.get("qty", 0.0)) or 0.0)
        leg2 = float(pos.get("leg2_hedged_qty", -1.0) or 0.0)
        if pos.get("leg1_status") == "FILLED" and abs(leg1 - leg2) > max(1e-9, 1e-9 * max(leg1, 1.0)):
            problems.append(f"HEDGE_MISMATCH:{sym}")
        margin = float(pos.get("margin_allocated_usdt", 0.0) or 0.0)
        notional = float(pos.get("notional_usdt", 0.0) or 0.0)
        if margin < 0.0 or notional < 0.0:
            problems.append(f"NEGATIVE_BALANCE:{sym}")
        if notional < DUST_NOTIONAL_USD and notional > 0.0:
            pos["dust_flagged"] = True

    for name in ("current_equity", "peak_equity"):
        value = getattr(executor.state, name, None)
        if value is not None and float(value) < 0.0:
            problems.append(f"NEGATIVE_BALANCE:{name}")

    if getattr(executor.state, "kill_switch_triggered", False):
        problems.append("KILL_SWITCH_TRIGGERED")
    pending = getattr(executor.state, "kill_switch_pending_close", False)
    if pending:
        problems.append("KILL_SWITCH_PENDING_CLOSE")

    equity = float(getattr(executor.state, "current_equity", capital) or capital)
    peak = float(getattr(executor.state, "peak_equity", equity) or equity)
    if peak > 0.0:
        dd_pct = (peak - equity) / peak * 100.0
        if dd_pct >= MAX_DRAWDOWN_PCT:
            problems.append(f"DRAWDOWN_EXCEEDED:{dd_pct:.2f}%")
    return problems


def run_cycle(
    executor: CarryLiveExecutor,
    symbols: Sequence[str],
    fixture_data: dict[str, Any] | None,
    timeout_sec: float,
    capital: float,
    expected_symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Chay mot chu ky burn-in va tra ve ban ghi ket qua."""
    snapshot = fetch_binance_shadow_snapshot(
        list(symbols), timeout_sec=timeout_sec, fixture_data=fixture_data
    )
    problems = check_feed(snapshot, expected_symbols if expected_symbols is not None else symbols)

    plan: dict[str, Any] = {}
    try:
        plan = executor.reconcile_once()
    except Exception as exc:  # khong de 1 chu ky loi lam hong burn-in
        problems.append(f"RECONCILE_ERROR:{type(exc).__name__}")
        plan = {}

    diagnostics = plan.get("diagnostics") or getattr(executor, "last_decision_diagnostics", {}) or {}
    if not diagnostics:
        problems.append("DIAGNOSTICS_MISSING")
    problems.extend(check_state(executor, capital))

    books = snapshot.get("symbols") or {}
    max_spread = max((float(b.get("spread_bps", 0.0)) for b in books.values()), default=0.0)
    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "feed_source": snapshot.get("source", "unknown"),
        "feed_symbols": len(books),
        "feed_errors": list(snapshot.get("errors") or []),
        "max_spread_bps": max_spread,
        "equity": float(getattr(executor.state, "current_equity", capital) or capital),
        "carry_positions": len(getattr(executor.state, "carry_positions", {}) or {}),
        "overlay_positions": len(getattr(executor.state, "overlay_positions", {}) or {}),
        "carry_gate_summary": str(diagnostics.get("summary", "")),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow-live burn-in Sprint 15 (paper, read-only)")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument("--capital", type=float, default=3000.0)
    parser.add_argument("--state-file", type=str, default=str(DEFAULT_STATE))
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--shadow-timeout-sec", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-out", type=str, default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = list(LIQUID_8_SYMBOLS)
    state_path = Path(args.state_file)
    fixture_data: dict[str, Any] | None = None
    if args.offline_fixture:
        fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    executor = build_burnin_executor(
        capital=float(args.capital),
        state_path=state_path,
        raw_dir=_ROOT / "data" / "raw",
        symbols=symbols,
    )

    records: list[dict[str, Any]] = []
    cycles = max(1, int(args.cycles))
    # Offline fixture chi chua 4 symbol -> chi ky vong dung tap symbol co trong fixture.
    expected_feed_symbols: Sequence[str] = (
        sorted(fixture_data.keys()) if fixture_data is not None else symbols
    )
    for i in range(1, cycles + 1):
        record = run_cycle(
            executor,
            symbols,
            fixture_data,
            float(args.shadow_timeout_sec),
            float(args.capital),
            expected_symbols=expected_feed_symbols,
        )
        record["cycle"] = i
        records.append(record)
        print(
            f"[cycle {i}/{cycles}] {record['status']} | feed={record['feed_source']} "
            f"symbols={record['feed_symbols']} max_spread={record['max_spread_bps']:.2f}bps "
            f"equity={record['equity']:.2f} carry_pos={record['carry_positions']} "
            f"overlay_pos={record['overlay_positions']} problems={record['problems']}"
        )
        if i < cycles:
            time.sleep(max(0.0, float(args.interval_sec)))

    n_pass = sum(1 for r in records if r["status"] == "PASS")
    summary = {
        "cycles": cycles,
        "passed": n_pass,
        "failed": cycles - n_pass,
        "offline_fixture": bool(args.offline_fixture),
        "capital": float(args.capital),
        "records": records,
    }
    print(f"SUMMARY: {n_pass}/{cycles} PASS")
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Da xuat bao cao: {out}")

    if args.strict and n_pass != cycles:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
