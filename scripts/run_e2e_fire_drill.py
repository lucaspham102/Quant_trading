"""Kich ban dien tap 5 chu ky offline chung minh guardrail hoat dong end-to-end.

Su dung CarryLiveExecutor va StateStore voi FakeExchange de kiem tra:
1. Cycle 1: Mo vi the maker-taker va overlay, luu state, chuan lot step.
2. Cycle 2: Phat hien delta drift, danh dau sub-lot va tu sua khi du lon.
3. Cycle 3: Chan spread qua rong va crossed book ngay truoc khi vao lenh.
4. Cycle 4: Drawdown kill-switch dong sach vi the va luu co khoa.
5. Cycle 5: Co sticky chan chu ky tiep theo, go bang --reset-state.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.barbell_portfolio import BarbellConfig
from src.carry_live_executor import CarryLiveExecutor
from src.config import TradingConfig
from src.cross_exchange_carry import LOT_STEP_SIZES
from src.dynamic_carry_basis import DynamicCarryConfig
from src.state_store import StateStore

PERP_BTC = "BTC/USDT:USDT"
PERP_SOL = "SOL/USDT:USDT"
STEP_BTC = LOT_STEP_SIZES.get(PERP_BTC, 0.001)


def _price_frame(price: float) -> pd.DataFrame:
    """DataFrame 1 dong gia cho audit va overlay."""
    idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"close": float(price), "spot_close": float(price)}, index=idx
    ).astype("float64")


class FakeExchange:
    """San gia ghi nhan lenh va order book tinh trong bo nho."""

    def __init__(
        self,
        bids: Sequence[Sequence[float]] | None = None,
        asks: Sequence[Sequence[float]] | None = None,
        with_order_book: bool = True,
        status: str = "closed",
        symbol_books: dict[str, dict[str, list[list[float]]]] | None = None,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self.bids = [list(level) for level in (bids or [[99.9, 10]])]
        self.asks = [list(level) for level in (asks or [[100.1, 10]])]
        self.status = status
        self.symbol_books = symbol_books or {}
        if with_order_book:
            self.fetch_order_book = self._fetch_order_book

    def _fetch_order_book(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        if symbol in self.symbol_books:
            return self.symbol_books[symbol]
        return {
            "bids": [list(level) for level in self.bids],
            "asks": [list(level) for level in self.asks],
        }

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        order_dict = {
            "id": f"ord_{len(self.orders) + 1}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": float(amount),
            "price": None if price is None else float(price),
            "params": dict(params) if params is not None else {},
        }
        self.orders.append(order_dict)
        if type == "limit":
            return {"id": order_dict["id"], "status": "open", "filled": 0.0}
        return {"id": order_dict["id"], "status": self.status, "filled": float(amount)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        self.cancelled.append({"id": order_id, "symbol": symbol})
        return {"id": order_id, "status": "canceled"}

    def fetch_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        for o in self.orders:
            if o.get("id") == order_id:
                return {"id": order_id, "status": "closed", "filled": o.get("amount", 0.0)}
        return {"id": order_id, "status": "closed", "filled": 0.2}


def _make_executor(
    state_path: Path,
    *,
    mode: str = "live",
    capital: float = 3000.0,
    perp: Any = None,
    spot: Any = None,
    max_allowed_spread_bps: float = 10.0,
    max_live_drawdown_pct: float = 0.06,
    strategy: str | None = None,
) -> CarryLiveExecutor:
    """Khoi tao executor offline voi state luu tai state_path."""
    raw_dir = state_path.parent / "raw"
    config = dataclasses.replace(
        TradingConfig(),
        timeframe="1h",
        raw_data_dir=raw_dir,
        state_file_path=state_path,
    )
    exchange_map = {"binance": perp, "binance_spot": spot}
    return CarryLiveExecutor(
        config=config,
        carry_config=DynamicCarryConfig(),
        mode=mode,  # type: ignore[arg-type]
        symbols=[PERP_BTC, PERP_SOL],
        capital=capital,
        perp_exchange=perp,
        spot_exchange=spot,
        exchange_map=exchange_map,
        state_store=StateStore(state_path),
        raw_data_dir=raw_dir,
        max_allowed_spread_bps=max_allowed_spread_bps,
        max_live_drawdown_pct=max_live_drawdown_pct,
        strategy=strategy,
        overlay_config=BarbellConfig(),
    )


def run_five_cycle_fire_drill(tmp_state_path: Path, capital: float = 3000.0) -> dict[str, Any]:
    """Chay kich ban dien tap 5 chu ky offline kiem tra guardrail."""
    tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_state_path.exists():
        tmp_state_path.unlink()

    checks: dict[str, bool] = {}
    cycles: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Cycle 1: Mo vi the maker-taker va overlay rebalance
    # -------------------------------------------------------------------------
    perp1 = FakeExchange(bids=[[59994.0, 10.0]], asks=[[60006.0, 10.0]])
    spot1 = FakeExchange(bids=[[59994.0, 10.0]], asks=[[60006.0, 10.0]])
    ex1 = _make_executor(tmp_state_path, mode="live", capital=capital, perp=perp1, spot=spot1)

    ex1.execute_maker_then_taker_pair(
        PERP_BTC, "buy", 0.2, 60000.0, sleep_fn=lambda _s: None
    )
    ex1.execute_overlay_rebalance(
        {PERP_BTC: {"weight": 0.1}}, {PERP_BTC: _price_frame(60000.0)}
    )
    ex1.state_store.save(ex1.state)

    c1_has_pos = (len(ex1.state.carry_positions) >= 1) or (len(ex1.state.overlay_positions) >= 1)
    checks["c1_position_opened"] = bool(c1_has_pos)
    checks["c1_state_persisted"] = bool(tmp_state_path.exists())

    reloaded_c1 = StateStore(tmp_state_path).load(default_equity=capital)
    checks["c1_kill_switch_false"] = (reloaded_c1.kill_switch_triggered is False)

    step_ok = True
    for sym, pos in ex1.state.carry_positions.items():
        step = LOT_STEP_SIZES.get(sym, 0.001)
        for k in ("qty", "spot_qty"):
            val = float(pos.get(k, 0.0) or 0.0)
            if val > 0:
                units = round(val / step)
                if abs(val - units * step) > 1e-6:
                    step_ok = False
    for sym, pos in ex1.state.overlay_positions.items():
        step = LOT_STEP_SIZES.get(sym, 0.001)
        val = float(pos.get("qty", 0.0) or 0.0)
        if val > 0:
            units = round(val / step)
            if abs(val - units * step) > 1e-6:
                step_ok = False
    checks["c1_lot_step_multiple"] = bool(step_ok)

    cycles.append({
        "cycle": 1,
        "action": "Mo vi the Maker-Taker pair va Overlay rebalance",
        "expected": "Vi the mo hop le, state duoc ghi len dia, qty theo lot-step",
        "actual": f"carry={len(ex1.state.carry_positions)}, overlay={len(ex1.state.overlay_positions)}, file={tmp_state_path.exists()}",
        "pass": all(checks[k] for k in ["c1_position_opened", "c1_state_persisted", "c1_kill_switch_false", "c1_lot_step_multiple"]),
    })

    # -------------------------------------------------------------------------
    # Cycle 2: Drift audit & repair
    # -------------------------------------------------------------------------
    btc_pos = ex1.state.carry_positions.setdefault(
        PERP_BTC,
        {"symbol": PERP_BTC, "spot_symbol": "BTC/USDT", "qty": 0.2, "spot_qty": 0.2, "entry_spot_price": 60000.0},
    )
    base_btc_qty = float(btc_pos.get("qty", 0.2))
    btc_pos["spot_qty"] = base_btc_qty + 0.0004

    ex1.state.carry_positions[PERP_SOL] = {
        "symbol": PERP_SOL,
        "spot_symbol": "SOL/USDT",
        "qty": 1.0,
        "spot_qty": 1.25,
        "entry_spot_price": 200.0,
        "entry_perp_price": 200.0,
        "spot_price": 200.0,
        "perp_price": 200.0,
    }

    price_map = {PERP_BTC: _price_frame(60000.0), PERP_SOL: _price_frame(200.0)}
    btc_orders_a_before = sum(1 for o in perp1.orders + spot1.orders if "BTC" in o.get("symbol", ""))
    total_orders_a_before = len(perp1.orders) + len(spot1.orders)

    items_a = ex1.audit_delta_drift(price_map)
    for it in items_a:
        ex1.repair_delta_drift(it)

    btc_flag_set = (
        ex1.state.carry_positions[PERP_BTC].get("drift_unrepairable") is True
        or PERP_BTC in getattr(ex1.state, "drift_unrepairable_symbols", {})
    )
    checks["c2a_btc_unrepairable_flag"] = bool(btc_flag_set)

    btc_orders_a_after = sum(1 for o in perp1.orders + spot1.orders if "BTC" in o.get("symbol", ""))
    checks["c2a_btc_no_order"] = (btc_orders_a_after == btc_orders_a_before)

    sol_repaired = any(it.get("symbol") == PERP_SOL for it in items_a) and (
        len(perp1.orders) + len(spot1.orders) > total_orders_a_before
    )
    checks["c2a_sol_repaired"] = bool(sol_repaired)

    # Tang lech BTC len 0.002
    btc_orders_b_before = sum(1 for o in perp1.orders + spot1.orders if "BTC" in o.get("symbol", ""))
    ex1.state.carry_positions[PERP_BTC]["spot_qty"] = base_btc_qty + 0.002
    items_b = ex1.audit_delta_drift(price_map)
    for it in items_b:
        ex1.repair_delta_drift(it)

    btc_flag_cleared = (
        "drift_unrepairable" not in ex1.state.carry_positions[PERP_BTC]
        and PERP_BTC not in getattr(ex1.state, "drift_unrepairable_symbols", {})
    )
    checks["c2b_btc_flag_cleared"] = bool(btc_flag_cleared)

    btc_orders_b_after = sum(1 for o in perp1.orders + spot1.orders if "BTC" in o.get("symbol", ""))
    checks["c2b_btc_repair_order"] = (btc_orders_b_after > btc_orders_b_before)
    ex1.state_store.save(ex1.state)

    cycles.append({
        "cycle": 2,
        "action": "Delta drift audit (sub-lot -> unrepairable flag; >1 lot -> repair)",
        "expected": "BTC lech nho bi danh dau, SOL sua thanh cong, BTC lech lon tu go co va sua",
        "actual": f"flag_a={btc_flag_set}, sol_rep={sol_repaired}, flag_cleared={btc_flag_cleared}",
        "pass": all(checks[k] for k in ["c2a_btc_unrepairable_flag", "c2a_btc_no_order", "c2a_sol_repaired", "c2b_btc_flag_cleared", "c2b_btc_repair_order"]),
    })

    # -------------------------------------------------------------------------
    # Cycle 3: Spread guard (wide spread & crossed book)
    # -------------------------------------------------------------------------
    books = {
        PERP_BTC: {"bids": [[59994.0, 10.0]], "asks": [[60105.0, 10.0]]},  # ~18.5 bps
        PERP_SOL: {"bids": [[200.5, 10.0]], "asks": [[199.5, 10.0]]},     # crossed
    }
    perp3 = FakeExchange(symbol_books=books)
    spot3 = FakeExchange()
    ex3 = _make_executor(tmp_state_path.parent / "temp_guard.json", mode="live", perp=perp3, spot=spot3)

    res_wide = ex3.place_maker_leg1(PERP_BTC, "buy", 0.2, 60000.0)
    res_crossed = ex3.place_maker_leg1(PERP_SOL, "buy", 1.0, 200.0)

    checks["c3_wide_spread_skipped"] = (
        res_wide.get("status") == "skipped" and res_wide.get("reason") == "SPREAD_TOO_WIDE"
    )
    checks["c3_crossed_book_skipped"] = (
        res_crossed.get("status") == "skipped" and res_crossed.get("reason") == "SPREAD_CROSSED_BOOK"
    )
    checks["c3_no_orders_placed"] = (len(perp3.orders) == 0 and len(spot3.orders) == 0)

    cycles.append({
        "cycle": 3,
        "action": "Kiem tra spread guard voi wide spread va crossed book",
        "expected": "Ca hai bi tu choi (SPREAD_TOO_WIDE, SPREAD_CROSSED_BOOK), 0 lenh moi",
        "actual": f"wide={res_wide.get('reason')}, crossed={res_crossed.get('reason')}, orders={len(perp3.orders)}",
        "pass": all(checks[k] for k in ["c3_wide_spread_skipped", "c3_crossed_book_skipped", "c3_no_orders_placed"]),
    })

    # -------------------------------------------------------------------------
    # Cycle 4: Kill-switch emergency shutdown
    # -------------------------------------------------------------------------
    ex1.state.peak_equity = 3000.0
    ex1.state.kill_switch_triggered = False
    ex1.kill_switch_triggered = False
    ex1.state.carry_positions = {
        PERP_BTC: {
            "symbol": PERP_BTC,
            "spot_symbol": "BTC/USDT",
            "qty": 0.2,
            "spot_qty": 0.2,
            "entry_spot_price": 60000.0,
            "entry_perp_price": 60000.0,
            "spot_price": 60000.0,
            "perp_price": 60000.0,
        }
    }
    ex1.state.overlay_positions = {
        PERP_BTC: {"qty": 0.2, "side": "short", "entry_price": 60000.0, "ts": "2024-01-01T00:00:00+00:00"}
    }
    orders_c4_before = len(perp1.orders) + len(spot1.orders)

    triggered = ex1.check_kill_switch(2812.5)  # -6.25%
    ex1.state_store.save(ex1.state)

    checks["c4_kill_switch_returned_true"] = bool(triggered)
    checks["c4_kill_switch_flag_active"] = bool(ex1.state.kill_switch_triggered)
    checks["c4_positions_flushed"] = (ex1.state.carry_positions == {} and ex1.state.overlay_positions == {})

    reloaded_c4 = StateStore(tmp_state_path).load(default_equity=capital)
    checks["c4_disk_state_persisted"] = bool(reloaded_c4.kill_switch_triggered)

    new_orders_c4 = (perp1.orders + spot1.orders)[orders_c4_before:]
    checks["c4_only_market_orders"] = (
        len(new_orders_c4) > 0 and all(o.get("type") == "market" for o in new_orders_c4)
    )

    cycles.append({
        "cycle": 4,
        "action": "Kich hoat Drawdown Kill-switch tai equity=2812.5 (-6.25%)",
        "expected": "Kill-switch kich hoat, dong toan bo vi the, chi dung lenh market close, luu co",
        "actual": f"triggered={triggered}, carry_pos={len(ex1.state.carry_positions)}, new_orders={len(new_orders_c4)}",
        "pass": all(checks[k] for k in ["c4_kill_switch_returned_true", "c4_kill_switch_flag_active", "c4_positions_flushed", "c4_disk_state_persisted", "c4_only_market_orders"]),
    })

    # -------------------------------------------------------------------------
    # Cycle 5: Sticky kill-switch & state reset
    # -------------------------------------------------------------------------
    perp5 = FakeExchange()
    spot5 = FakeExchange()
    ex5 = _make_executor(tmp_state_path, mode="live", capital=5000.0, perp=perp5, spot=spot5, strategy="barbell")
    plan5 = ex5.reconcile_once({})

    checks["c5a_reconcile_kill_switch_sticky"] = (
        ex5.state.kill_switch_triggered is True
        and plan5.get("kill_switch") is True
        and plan5.get("actions") == []
    )

    store5 = StateStore(tmp_state_path)
    state_reset = store5.load(default_equity=3000.0, reset=True, initial_capital=3000.0)
    # Ghi lai xuong dia giong hanh vi that cua --reset-state (reset phai ben vung).
    store5.save(state_reset)
    checks["c5b_reset_state_cleared"] = (
        state_reset.kill_switch_triggered is False
        and state_reset.current_equity == 3000.0
        and state_reset.kill_switch_pending_close is False
    )
    on_disk = StateStore(tmp_state_path).load(default_equity=3000.0, initial_capital=3000.0)
    checks["c5b_reset_persisted"] = (
        on_disk.kill_switch_triggered is False and abs(float(on_disk.current_equity) - 3000.0) < 1e-9
    )

    cycles.append({
        "cycle": 5,
        "action": "Kiem tra co sticky tren bot moi va go bo bang reset=True",
        "expected": "Khoa moi van giu co kill-switch, actions=[]; reset=True go sach co ve 3000.0",
        "actual": f"sticky={checks['c5a_reconcile_kill_switch_sticky']}, reset={checks['c5b_reset_state_cleared']}",
        "pass": checks["c5a_reconcile_kill_switch_sticky"]
        and checks["c5b_reset_state_cleared"]
        and checks["c5b_reset_persisted"],
    })

    return {
        "cycles": cycles,
        "checks": checks,
        "passed": all(checks.values()),
        "state_path": str(tmp_state_path),
    }


def main() -> None:
    """Entry point chay drill, xuat bang ASCII va ghi bang chung Markdown."""
    state_file = Path("data/state/fire_drill_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)

    result = run_five_cycle_fire_drill(state_file, capital=3000.0)
    checks = result["checks"]
    total = len(checks)
    passed_count = sum(1 for v in checks.values() if v)

    print("\n+------------------------------------+--------+")
    print("| Check Name                         | Status |")
    print("+------------------------------------+--------+")
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"| {name:<34} | {status:<6} |")
    print("+------------------------------------+--------+")
    print(f"Tong: {passed_count}/{total} checks PASS\n")

    evidence_path = Path("docs/sprint13_fire_drill_evidence.md")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Sprint 13 — Fire Drill Evidence Report",
        "",
        "Kich ban dien tap 5 chu ky offline tren o dia, su dung `CarryLiveExecutor` + `StateStore` that voi `FakeExchange`.",
        "",
        "| Chu ky | Hanh dong | Ket qua mong doi | Ket qua thuc te | PASS/FAIL |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for c in result["cycles"]:
        st = "PASS" if c["pass"] else "FAIL"
        md_lines.append(f"| Chu ky {c['cycle']} | {c['action']} | {c['expected']} | {c['actual']} | {st} |")

    md_lines.extend([
        "",
        f"**Tong: {passed_count}/{total} checks PASS**",
        "",
    ])

    evidence_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Da ghi bao cao bang chung vao: {evidence_path}")

    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
