"""Sprint 17: Test suite cho module doi soat trang thai danh muc (exchange reconciler).

Bao gom:
- Chuan hoa symbol
- Kiem tra cac trang thai doi soat (SYNCED, ORPHAN, PHANTOM, QTY_DRIFT, EQUITY_MISMATCH)
- Co che tolerance USD va equity mismatch
- Fail-closed khi snapshot unreadable hoac khong hop le
- Tich hop preflight reconciliation va chan mo vi the moi trong CarryLiveExecutor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from run_live import _run_reconcile_preflight
from src.carry_live_executor import CarryLiveExecutor
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.exchange_reconciler import (
    EQUITY_MISMATCH,
    ORPHAN_ON_EXCHANGE,
    PHANTOM_IN_STATE,
    QTY_DRIFT,
    SYNCED,
    load_snapshot,
    normalize_symbol,
    reconcile_portfolio_state,
)

PRICES: dict[str, float] = {
    "BTC/USDT:USDT": 96558.0,
    "ETH/USDT:USDT": 4200.0,
}


def test_normalize_symbol_variants() -> None:
    assert normalize_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert normalize_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert normalize_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
    assert normalize_symbol(" ethusdt ") == "ETH/USDT:USDT"
    assert normalize_symbol("") == ""


def test_synced_state_allows_trading() -> None:
    local_state = {
        "carry_positions": {"BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01}},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {"BTCUSDT": -0.01},
        "exchange_equity_usd": 3000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True
    assert report["repair_actions"] == []


def test_orphan_spot_on_exchange_blocks_trading() -> None:
    local_state = {
        "carry_positions": {},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {},
        "exchange_equity_usd": 3000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "ORPHAN_ON_EXCHANGE"
    assert report["trading_allowed"] is False
    assert "BTC/USDT:USDT" in report["orphans"]
    assert any(a["action"] == "CLOSE_ORPHAN_SPOT" for a in report["repair_actions"])


def test_phantom_position_in_state_blocks_trading() -> None:
    local_state = {
        "carry_positions": {"BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01}},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {},
        "exchange_equity_usd": 3000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] in ("PHANTOM_IN_STATE",)
    assert report["trading_allowed"] is False
    assert any(a["action"] == "PURGE_PHANTOM_STATE" for a in report["repair_actions"])


def test_qty_drift_detected() -> None:
    local_state = {
        "carry_positions": {"BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01}},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {"BTC": 0.015},
        "perp_positions": {"BTCUSDT": -0.015},
        "exchange_equity_usd": 3000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "QTY_DRIFT"
    assert report["trading_allowed"] is False
    assert any(a["action"] == "RESYNC_QTY" for a in report["repair_actions"])


def test_equity_mismatch_blocks_even_when_positions_synced() -> None:
    local_state = {
        "carry_positions": {"BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01}},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {"BTCUSDT": -0.01},
        "exchange_equity_usd": 3050.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "EQUITY_MISMATCH"
    assert report["trading_allowed"] is False
    assert report["equity_gap_usd"] == pytest.approx(50.0)
    assert report["equity_gap_pct"] == pytest.approx(50.0 / 3000.0 * 100.0)


def test_small_spot_gap_within_tolerance_is_synced() -> None:
    local_state = {
        "carry_positions": {"BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01}},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01002},
        "perp_positions": {"BTCUSDT": -0.01},
        "exchange_equity_usd": 3000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True


def test_overlay_short_hedge_is_matched() -> None:
    local_state = {
        "carry_positions": {},
        "overlay_positions": {"ETH/USDT:USDT": {"qty": 0.1, "side": "short"}},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {"ETHUSDT": -0.1},
        "exchange_equity_usd": 3000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "SYNCED"
    eth_item = next(it for it in report["items"] if "ETH" in it["symbol"])
    assert eth_item["local_perp_qty"] == pytest.approx(-0.1)


def test_unknown_exchange_position_is_never_synced() -> None:
    local_state = {
        "carry_positions": {},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {"SOLUSDT": -2.0},
        "exchange_equity_usd": 3000.0,
    }
    prices = {**PRICES, "SOL/USDT:USDT": 210.0}
    report = reconcile_portfolio_state(local_state, snapshot, prices, EXCHANGE_LOT_SPECS)
    assert report["trading_allowed"] is False
    assert report["status"] != "SYNCED"


def test_load_snapshot_roundtrip_and_validation(tmp_path: Path) -> None:
    import json

    valid_payload = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {},
        "exchange_equity_usd": 3000.0,
    }
    valid_file = tmp_path / "valid_snapshot.json"
    with valid_file.open("w", encoding="utf-8") as f:
        json.dump(valid_payload, f)

    loaded = load_snapshot(valid_file)
    assert loaded == valid_payload

    invalid_file = tmp_path / "invalid_snapshot.json"
    with invalid_file.open("w", encoding="utf-8") as f:
        json.dump([1, 2], f)

    with pytest.raises(ValueError, match="JSON root must be a dict"):
        load_snapshot(invalid_file)


def test_reconcile_preflight_blocks_unreadable_snapshot(tmp_path: Path) -> None:
    class _FakeExecutor:
        def __init__(self) -> None:
            self.state: dict = {}
            self.trading_allowed: bool = True
            self.reconcile_report: dict | None = None
            self.reconcile_status: str | None = None

        def apply_reconcile_report(self, report: dict) -> bool:
            return CarryLiveExecutor.apply_reconcile_report(self, report)

    args = argparse.Namespace(reconcile_snapshot=str(tmp_path / "missing.json"))
    fake = _FakeExecutor()
    report = _run_reconcile_preflight(args, fake)

    assert report is not None
    assert report["trading_allowed"] is False
    assert report["status"] == "SNAPSHOT_UNREADABLE"
    assert fake.trading_allowed is False


def test_reconcile_preflight_returns_none_when_flag_absent() -> None:
    args = argparse.Namespace(reconcile_snapshot=None)
    assert _run_reconcile_preflight(args, object()) is None


def test_executor_apply_reconcile_report_gates_new_entries() -> None:
    ex = CarryLiveExecutor.__new__(CarryLiveExecutor)
    ex.reconcile_report = None
    ex.reconcile_status = None
    ex.trading_allowed = True

    report_payload = {
        "status": "ORPHAN_ON_EXCHANGE",
        "trading_allowed": False,
        "summary": "x",
        "repair_actions": [],
    }
    result = CarryLiveExecutor.apply_reconcile_report(ex, report_payload)
    assert result is False
    assert ex.trading_allowed is False
    assert ex.reconcile_status == "ORPHAN_ON_EXCHANGE"

    try:
        from src.carry_live_executor import build_barbell_allocations
        alloc = build_barbell_allocations(ex, {})
    except (ImportError, TypeError):
        alloc = CarryLiveExecutor.build_barbell_allocations(ex, {})

    assert alloc.get("reconcile_blocked") is True
    assert alloc["carry_sleeve"]["allocations"] == []

def test_load_snapshot_accepts_utf8_bom(tmp_path) -> None:
    """Windows/PowerShell hay ghi BOM - snapshot van phai doc duoc."""
    p = tmp_path / "snap_bom.json"
    p.write_text('\ufeff{"spot_balances": {}, "perp_positions": {}, "exchange_equity_usd": 3000.0}', encoding="utf-8")
    snap = load_snapshot(p)
    assert snap["exchange_equity_usd"] == 3000.0


def test_executor_reconcile_dashboard_label() -> None:
    ex = CarryLiveExecutor.__new__(CarryLiveExecutor)
    ex.reconcile_status = None
    ex.trading_allowed = True
    assert CarryLiveExecutor._reconcile_dashboard_label(ex) == "OFF"

    ex.reconcile_status = "SYNCED"
    assert CarryLiveExecutor._reconcile_dashboard_label(ex) == "SYNCED"

    ex.reconcile_status = "PHANTOM_IN_STATE"
    ex.trading_allowed = False
    assert CarryLiveExecutor._reconcile_dashboard_label(ex) == "BLOCKED (PHANTOM_IN_STATE)"


def test_executor_apply_none_report_restores_trading() -> None:
    ex = CarryLiveExecutor.__new__(CarryLiveExecutor)
    ex.reconcile_report = {"status": "QTY_DRIFT"}
    ex.reconcile_status = "QTY_DRIFT"
    ex.trading_allowed = False
    assert CarryLiveExecutor.apply_reconcile_report(ex, None) is True
    assert ex.trading_allowed is True
    assert ex.reconcile_status is None


def test_audit_record_carries_reconcile_status() -> None:
    from src.carry_live_executor import build_cycle_audit_record

    rec = build_cycle_audit_record(
        cycle=1,
        mode="paper",
        execution_style="maker_taker_hedge",
        capital=3000.0,
        equity=3000.0,
        peak_equity=3000.0,
        drawdown_pct=0.0,
        carry_spot_usd=0.0,
        carry_perp_usd=0.0,
        overlay_long_usd=0.0,
        overlay_short_usd=0.0,
        reconcile_status="ORPHAN_ON_EXCHANGE",
    )
    assert rec["reconcile_status"] == "ORPHAN_ON_EXCHANGE"
    rec_off = build_cycle_audit_record(
        cycle=1,
        mode="paper",
        execution_style="maker_taker_hedge",
        capital=3000.0,
        equity=3000.0,
        peak_equity=3000.0,
        drawdown_pct=0.0,
        carry_spot_usd=0.0,
        carry_perp_usd=0.0,
        overlay_long_usd=0.0,
        overlay_short_usd=0.0,
    )
    assert rec_off["reconcile_status"] is None
