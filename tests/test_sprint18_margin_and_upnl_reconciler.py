"""Sprint 18: Test suite cho doi soat ky quy Perp, unrealized PnL va khoang cach thanh ly.

Bao gom 12 test:
- Lech tien mat do nhan funding (CASH_OR_FUNDING_DRIFT)
- Lech do mark price perp (UNREALIZED_PNL_DRIFT)
- Truong hop khong giai thich duoc (UNEXPLAINED_EQUITY_DRIFT)
- Snapshot toi gian tuong thich nguoc (khong margin fields)
- maint_margin_ratio vuot nguong -> MARGIN_DANGER + EMERGENCY_DELEVERAGE_PERP
- liq_distance_pct vi pham buffer an toan -> MARGIN_DANGER + fail-closed
- Bien an toan margin & liquidation distance -> SYNCED
- Vi the flat hoac liquidation_price = 0 khong bao dong
- perp_positions_detail dang dict duoc chuan hoa dung
- analyze_equity_and_margin_health voi local_state dang SimpleNamespace
- Thu tu uu tien status EQUITY_MISMATCH > MARGIN_DANGER
- Dinh dang summary chua margin_ratio va liq_dist
"""

from __future__ import annotations

import types

import pytest

from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.exchange_reconciler import (
    EQUITY_MISMATCH,
    MARGIN_DANGER,
    RECONCILE_STATUSES,
    SYNCED,
    analyze_equity_and_margin_health,
    reconcile_portfolio_state,
)

PRICES: dict[str, float] = {
    "BTC/USDT:USDT": 96558.0,
    "ETH/USDT:USDT": 4200.0,
}


def test_margin_danger_constant_and_statuses() -> None:
    assert MARGIN_DANGER == "MARGIN_DANGER"
    assert MARGIN_DANGER in RECONCILE_STATUSES


def test_drift_attribution_cash_and_funding() -> None:
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {
            "BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01, "entry_perp_price": 96558.0}
        },
        "overlay_positions": {},
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {"BTCUSDT": -0.01},
        "spot_cash_usd": 1000.0,
        "perp_wallet_balance_usd": 2050.0,
        "perp_unrealized_pnl_usd": 0.0,
        "exchange_equity_usd": 3050.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS, equity_tolerance_pct=5.0)
    attr = report["equity_drift_attribution"]
    assert attr is not None
    assert attr["primary_mismatch_cause"] == "CASH_OR_FUNDING_DRIFT"
    assert attr["cash_and_funding_drift_usd"] > 0.0
    assert attr["cash_and_funding_drift_usd"] == pytest.approx(50.0)
    assert attr["unexplained_residual_drift_usd"] == pytest.approx(0.0, abs=1e-6)
    assert attr["unrealized_pnl_drift_usd"] == pytest.approx(0.0)


def test_drift_attribution_unrealized_pnl() -> None:
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {
            "BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01, "entry_perp_price": 96558.0}
        },
        "overlay_positions": {},
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {"BTCUSDT": -0.01},
        "spot_cash_usd": 1000.0,
        "perp_wallet_balance_usd": 2000.0,
        "perp_unrealized_pnl_usd": 80.0,
        "exchange_equity_usd": 3080.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS, equity_tolerance_pct=5.0)
    attr = report["equity_drift_attribution"]
    assert attr is not None
    assert attr["primary_mismatch_cause"] == "UNREALIZED_PNL_DRIFT"
    assert attr["unrealized_pnl_drift_usd"] > 0.0
    assert attr["unrealized_pnl_drift_usd"] == pytest.approx(80.0)
    assert attr["cash_and_funding_drift_usd"] == pytest.approx(0.0)
    assert attr["unexplained_residual_drift_usd"] == pytest.approx(0.0, abs=1e-6)


def test_drift_attribution_unexplained_drift() -> None:
    local_state = {
        "current_equity": 3000.0,
        "carry_positions": {},
        "overlay_positions": {},
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {},
        "spot_cash_usd": 1500.0,
        "perp_wallet_balance_usd": 1500.0,
        "perp_unrealized_pnl_usd": 0.0,
        "exchange_equity_usd": 3120.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS, equity_tolerance_pct=5.0)
    attr = report["equity_drift_attribution"]
    assert attr is not None
    assert attr["primary_mismatch_cause"] == "UNEXPLAINED_EQUITY_DRIFT"
    assert attr["unexplained_residual_drift_usd"] == pytest.approx(120.0)
    assert attr["cash_and_funding_drift_usd"] == pytest.approx(0.0)
    assert attr["unrealized_pnl_drift_usd"] == pytest.approx(0.0)


def test_minimal_snapshot_backward_compatibility() -> None:
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
    assert report["equity_drift_attribution"] is None
    assert report["margin_health"]["margin_danger"] is False
    assert report["repair_actions"] == []


def test_maintenance_margin_ratio_danger_blocks_trading() -> None:
    local_state = {
        "carry_positions": {"BTC/USDT:USDT": {"qty": 0.01, "leg2_hedged_qty": 0.01}},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {"BTC": 0.01},
        "perp_positions": {"BTCUSDT": -0.01},
        "exchange_equity_usd": 3000.0,
        "maint_margin_usd": 550.0,
        "margin_balance_usd": 1000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == "MARGIN_DANGER"
    assert report["trading_allowed"] is False
    assert report["margin_health"]["maint_margin_ratio"] == pytest.approx(0.55)
    assert any(a["action"] == "EMERGENCY_DELEVERAGE_PERP" for a in report["repair_actions"])


def test_liquidation_distance_buffer_danger_fail_closed() -> None:
    prices = {**PRICES, "XYZ/USDT:USDT": 100.0}
    local_state = {
        "carry_positions": {},
        "overlay_positions": {"XYZ/USDT:USDT": {"qty": 10.0, "side": "long", "entry_price": 95.0}},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {"XYZUSDT": 10.0},
        "exchange_equity_usd": 3000.0,
        "perp_positions_detail": [
            {
                "symbol": "XYZUSDT",
                "qty": 10.0,
                "mark_price": 100.0,
                "liquidation_price": 82.0,
            }
        ],
    }
    report = reconcile_portfolio_state(local_state, snapshot, prices, EXCHANGE_LOT_SPECS)
    assert report["status"] == "MARGIN_DANGER"
    assert report["trading_allowed"] is False
    assert report["margin_health"]["min_liq_distance_pct"] == pytest.approx(18.0)
    assert any(a["action"] == "EMERGENCY_DELEVERAGE_PERP" for a in report["repair_actions"])


def test_safe_margin_and_liquidation_distance_not_danger() -> None:
    prices = {**PRICES, "XYZ/USDT:USDT": 100.0}
    local_state = {
        "carry_positions": {},
        "overlay_positions": {"XYZ/USDT:USDT": {"qty": 10.0, "side": "long", "entry_price": 100.0}},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {"XYZUSDT": 10.0},
        "exchange_equity_usd": 3000.0,
        "maint_margin_usd": 300.0,
        "margin_balance_usd": 1000.0,
        "perp_positions_detail": [
            {
                "symbol": "XYZUSDT",
                "qty": 10.0,
                "mark_price": 100.0,
                "liquidation_price": 60.0,
            }
        ],
    }
    report = reconcile_portfolio_state(local_state, snapshot, prices, EXCHANGE_LOT_SPECS)
    assert report["margin_health"]["margin_danger"] is False
    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True
    assert report["margin_health"]["maint_margin_ratio"] == pytest.approx(0.30)
    assert report["margin_health"]["min_liq_distance_pct"] == pytest.approx(40.0)


def test_flat_position_or_zero_liquidation_price_no_alarm() -> None:
    local_state = {
        "carry_positions": {},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {},
        "exchange_equity_usd": 3000.0,
        "perp_positions_detail": [
            {
                "symbol": "BTCUSDT",
                "qty": 0.0,
                "mark_price": 96558.0,
                "liquidation_price": 50000.0,
            },
            {
                "symbol": "ETHUSDT",
                "qty": 1.0,
                "mark_price": 4200.0,
                "liquidation_price": 0.0,
            },
        ],
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["margin_health"]["min_liq_distance_pct"] is None
    assert report["margin_health"]["margin_danger"] is False


def test_perp_positions_detail_as_dict_normalized() -> None:
    prices = {**PRICES, "SOL/USDT:USDT": 200.0}
    snapshot = {
        "spot_balances": {},
        "perp_positions": {},
        "exchange_equity_usd": 3000.0,
        "perp_positions_detail": {
            "SOLUSDT": {
                "qty": 5.0,
                "mark_price": 200.0,
                "entry_price": 190.0,
                "liquidation_price": 160.0,
            }
        },
    }
    health = analyze_equity_and_margin_health(
        local_state={"current_equity": 3000.0},
        exchange_snapshot=snapshot,
        prices=prices,
    )
    details = health["margin_health"]["perp_positions_detail"]
    assert isinstance(details, list)
    assert len(details) == 1
    assert details[0]["symbol"] == "SOL/USDT:USDT"
    assert details[0]["qty"] == pytest.approx(5.0)
    assert health["margin_health"]["min_liq_distance_pct"] == pytest.approx(20.0)


def test_analyze_equity_and_margin_health_with_object_state() -> None:
    local_state = types.SimpleNamespace(
        current_equity=3000.0,
        carry_positions={
            "BTC/USDT:USDT": types.SimpleNamespace(
                qty=0.01,
                leg2_hedged_qty=0.01,
                entry_perp_price=96558.0,
            )
        },
        overlay_positions={
            "ETH/USDT:USDT": types.SimpleNamespace(
                qty=0.5,
                side="short",
                entry_price=4200.0,
            )
        },
    )
    snapshot = types.SimpleNamespace(
        spot_cash_usd=1000.0,
        perp_wallet_balance_usd=2000.0,
        perp_unrealized_pnl_usd=0.0,
        exchange_equity_usd=3000.0,
        maint_margin_usd=100.0,
        margin_balance_usd=2000.0,
        perp_positions_detail=[
            types.SimpleNamespace(
                symbol="BTCUSDT",
                qty=-0.01,
                mark_price=96558.0,
                entry_price=96558.0,
                liquidation_price=50000.0,
            )
        ],
    )
    result = analyze_equity_and_margin_health(local_state, snapshot, PRICES)
    assert isinstance(result, dict)
    assert "margin_health" in result
    assert "equity_drift_attribution" in result
    assert result["margin_health"]["margin_danger"] is False
    assert result["equity_drift_attribution"]["primary_mismatch_cause"] is None


def test_status_priority_equity_mismatch_over_margin_danger() -> None:
    local_state = {
        "carry_positions": {},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {},
        "exchange_equity_usd": 3100.0,
        "maint_margin_usd": 600.0,
        "margin_balance_usd": 1000.0,
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert report["status"] == EQUITY_MISMATCH
    assert report["trading_allowed"] is False
    assert report["margin_danger"] is True
    assert any(a["action"] == "EMERGENCY_DELEVERAGE_PERP" for a in report["repair_actions"])


def test_summary_format_includes_margin_and_liq_distance() -> None:
    local_state = {
        "carry_positions": {},
        "overlay_positions": {},
        "current_equity": 3000.0,
    }
    snapshot = {
        "spot_balances": {},
        "perp_positions": {},
        "exchange_equity_usd": 3000.0,
        "maint_margin_usd": 300.0,
        "margin_balance_usd": 1000.0,
        "perp_positions_detail": [
            {
                "symbol": "BTCUSDT",
                "qty": 0.01,
                "mark_price": 100.0,
                "liquidation_price": 70.0,
            }
        ],
    }
    report = reconcile_portfolio_state(local_state, snapshot, PRICES, EXCHANGE_LOT_SPECS)
    assert "margin_ratio=0.30" in report["summary"]
    assert "liq_dist=30.0%" in report["summary"]
