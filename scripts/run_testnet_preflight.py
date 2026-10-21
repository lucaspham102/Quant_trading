"""Script chay kiem tra an toan Testnet truoc khi bat giao dich (Pre-flight Verification).

Bao gom 6 buoc kiem tra offline:
[1/6] Ky HMAC-SHA256 cho lenh doc va lenh dat cap Delta-Neutral
[2/6] Chuyen doi mock payload chuan Binance thanh exchange_snapshot
[3/6] Doi soat trang thai danh muc (reconcile) dat SYNCED va trading_allowed
[4/6] Doi soat 2 kich ban xau (MARGIN_DANGER va ORPHAN_ON_EXCHANGE) fail-closed
[5/6] Cong bao hoa warmup 720h va 5 Dinh luat Bat bien Dinh luong
[6/6] Thuc thi 1 chu ky Barbell (Carry + Overlay) va xuat JSONL audit log
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.barbell_portfolio import BarbellConfig  # noqa: E402
from src.binance_signed_client import (
    SPOT_ACCOUNT_PATH,
    BinanceSignedTestnetClient,
    build_query_string,
)
from src.carry_live_executor import (
    CarryLiveExecutor,
    append_audit_record,
    build_cycle_audit_record,
)
from src.config import TradingConfig
from src.cross_exchange_carry import EXCHANGE_LOT_SPECS
from src.dynamic_carry_basis import DynamicCarryConfig
from src.exchange_reconciler import reconcile_portfolio_state
from src.quant_invariants import (
    verify_accounting_identity_invariant,
    verify_clock_shift_invariance,
    verify_lot_neutrality_invariant,
    verify_warmup_saturation_invariant,
    verify_zero_compounding_leak_invariant,
)
from src.state_store import StateStore

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"
BNB = "BNB/USDT:USDT"
CORE4 = (BTC, ETH, SOL, BNB)


def _step_1_hmac_signing() -> None:
    api_key = "preflight-test-key"
    api_secret = "preflight-test-secret"
    client = BinanceSignedTestnetClient(
        api_key=api_key,
        api_secret=api_secret,
        dry_run=True,
    )

    # 1. build_signed_get_request
    get_req = client.build_signed_get_request(SPOT_ACCOUNT_PATH, timestamp_ms=1700000000000)
    assert get_req["method"] == "GET"
    assert get_req["headers"].get("X-MBX-APIKEY") == api_key
    assert "signature" in get_req["params"]
    assert get_req["params"]["timestamp"] == 1700000000000

    unsigned_get = {k: v for k, v in get_req["params"].items() if k != "signature"}
    expected_get_sig = hmac.new(
        api_secret.encode("utf-8"),
        build_query_string(unsigned_get).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert get_req["params"]["signature"] == expected_get_sig

    # 2. build_delta_neutral_order_pair
    pair = client.build_delta_neutral_order_pair(
        symbol=BTC,
        spot_qty=0.001,
        perp_qty=0.001,
        spot_price=96558.0,
        timestamp_ms=1700000000000,
    )
    assert pair["spot"]["headers"].get("X-MBX-APIKEY") == api_key
    assert pair["perp"]["headers"].get("X-MBX-APIKEY") == api_key
    assert "signature" in pair["spot"]["params"]
    assert "signature" in pair["perp"]["params"]

    print("[1/6] HMAC-SHA256 signing (read & order pair) ... PASS")


def _step_2_mock_payload_conversion() -> dict:
    client = BinanceSignedTestnetClient(
        api_key="preflight-key",
        api_secret="preflight-secret",
        dry_run=True,
    )
    mock_responses = {
        "spot_account": {
            "balances": [
                {"asset": "USDT", "free": "1000.0", "locked": "0.0"},
                {"asset": "BTC", "free": "0.01", "locked": "0.0"},
            ]
        },
        "futures_account": {
            "totalWalletBalance": "2000.0",
            "totalUnrealizedProfit": "50.0",
            "totalMaintMargin": "100.0",
            "totalMarginBalance": "2050.0",
        },
        "position_risk": [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "-0.01",
                "entryPrice": "95000.0",
                "markPrice": "96000.0",
                "liquidationPrice": "150000.0",
                "maintMargin": "100.0",
                "marginBalance": "2050.0",
            }
        ],
    }
    prices = {BTC: 96000.0}
    snap = client.fetch_exchange_reconcile_snapshot([BTC], prices, mock_responses=mock_responses)

    assert snap["snapshot_healthy"] is True
    assert snap["snapshot_source"] == "mock"
    assert snap["spot_balances"].get("BTC") == 0.01
    # Tien mat quote (USDT) duoc tong hop trong spot_cash_usd, khong nam trong spot_balances.
    assert "USDT" not in snap["spot_balances"]
    assert snap["perp_positions"].get(BTC) == -0.01
    assert len(snap["perp_positions_detail"]) == 1
    assert snap["perp_positions_detail"][0]["symbol"] == BTC
    assert snap["perp_wallet_balance_usd"] == 2000.0
    assert snap["perp_unrealized_pnl_usd"] == 50.0
    assert snap["maint_margin_usd"] == 100.0
    assert snap["margin_balance_usd"] == 2050.0
    # spot_cash_usd = 1000.0 (USDT) + 0.01 * 96000.0 (BTC) = 1960.0
    # exchange_equity_usd = 1960.0 + 2000.0 + 50.0 = 4010.0
    assert snap["spot_cash_usd"] == 1960.0
    assert snap["exchange_equity_usd"] == 4010.0

    print("[2/6] Binance mock payload conversion ... PASS")
    return snap


def _step_3_reconcile_synced(snapshot: dict) -> None:
    prices = {BTC: 96000.0}
    local_state = {
        "current_equity": snapshot["exchange_equity_usd"],
        "carry_positions": {
            BTC: {
                "qty": 0.01,
                "leg2_hedged_qty": 0.01,
                "entry_perp_price": 95000.0,
            }
        },
        "overlay_positions": {},
    }
    report = reconcile_portfolio_state(local_state, snapshot, prices, EXCHANGE_LOT_SPECS)
    assert report["status"] == "SYNCED"
    assert report["trading_allowed"] is True
    assert report["margin_health"]["maint_margin_ratio"] is not None
    assert report["margin_health"]["min_liq_distance_pct"] is not None
    assert report["margin_danger"] is False

    print("[3/6] Portfolio state reconciliation (SYNCED) ... PASS")


def _step_4_bad_scenarios(snapshot: dict) -> None:
    prices = {BTC: 96000.0, ETH: 3000.0}

    # Kich ban 1: maint_margin_ratio = 0.55 (> 0.50) -> MARGIN_DANGER
    danger_snap = dict(snapshot)
    danger_snap["maint_margin_usd"] = 1100.0
    danger_snap["margin_balance_usd"] = 2000.0
    # Cap nhat detail de dong bo
    danger_snap["perp_positions_detail"] = [
        dict(snapshot["perp_positions_detail"][0], maint_margin_usd=1100.0, margin_balance_usd=2000.0)
    ]
    local_state_danger = {
        "current_equity": danger_snap["exchange_equity_usd"],
        "carry_positions": {
            BTC: {
                "qty": 0.01,
                "leg2_hedged_qty": 0.01,
                "entry_perp_price": 95000.0,
            }
        },
        "overlay_positions": {},
    }
    rep_danger = reconcile_portfolio_state(local_state_danger, danger_snap, prices, EXCHANGE_LOT_SPECS)
    assert rep_danger["status"] == "MARGIN_DANGER"
    assert rep_danger["trading_allowed"] is False
    assert len(rep_danger["repair_actions"]) > 0
    assert any(a["action"] == "EMERGENCY_DELEVERAGE_PERP" for a in rep_danger["repair_actions"])

    # Kich ban 2: vi the mo tren san khong co trong local state -> ORPHAN_ON_EXCHANGE
    orphan_snap = dict(snapshot)
    orphan_snap["perp_positions"] = dict(snapshot["perp_positions"])
    orphan_snap["perp_positions"][ETH] = -1.0
    orphan_snap["perp_positions_detail"] = list(snapshot["perp_positions_detail"]) + [
        {
            "symbol": ETH,
            "qty": -1.0,
            "mark_price": 3000.0,
            "entry_price": 3000.0,
            "liquidation_price": 5000.0,
            "maint_margin_usd": 50.0,
            "margin_balance_usd": 2000.0,
        }
    ]
    orphan_snap["exchange_equity_usd"] = 4010.0
    local_state_orphan = {
        "current_equity": 4010.0,
        "carry_positions": {
            BTC: {
                "qty": 0.01,
                "leg2_hedged_qty": 0.01,
                "entry_perp_price": 95000.0,
            }
        },
        "overlay_positions": {},
    }
    rep_orphan = reconcile_portfolio_state(local_state_orphan, orphan_snap, prices, EXCHANGE_LOT_SPECS)
    assert rep_orphan["status"] == "ORPHAN_ON_EXCHANGE"
    assert rep_orphan["trading_allowed"] is False
    assert len(rep_orphan["repair_actions"]) > 0
    assert any(a["action"] == "CLOSE_ORPHAN_PERP" for a in rep_orphan["repair_actions"])

    print("[4/6] Bad scenarios reconciliation (MARGIN_DANGER & ORPHAN) ... PASS")


def _step_5_warmup_and_invariants() -> None:
    # 1. Warmup readiness 720h
    idx = pd.date_range("2025-01-01", periods=720, freq="1h", tz="UTC")
    df_warmup = pd.DataFrame(
        {
            "close": 96000.0,
            "spot_close": 96000.0,
            "perp_close": 96000.0,
            "basis_pct": 0.0,
            "funding_rate_event": 0.0001,
            "last_known_funding_rate": 0.0001,
        },
        index=idx,
    )
    universe_720 = {sym: df_warmup for sym in CORE4}

    dummy_executor = CarryLiveExecutor(mode="paper")
    w_res = dummy_executor.compute_warmup_readiness(universe_720)

    warmup_ready = w_res.get("warmup_ready") if isinstance(w_res, dict) else getattr(w_res, "warmup_ready")
    threshold_hours = w_res.get("warmup_threshold_hours") if isinstance(w_res, dict) else getattr(w_res, "warmup_threshold_hours")
    assert warmup_ready is True
    assert threshold_hours == 720

    # 2. 5 Dinh luat Bat bien Dinh luong
    # Invariant 1: Warmup Saturation
    r1 = verify_warmup_saturation_invariant(
        {720: {BTC: 0.5, ETH: 0.5}, 1440: {BTC: 0.5, ETH: 0.5}},
        raise_on_fail=True,
    )
    assert r1["passed"] is True

    # Invariant 2: Zero Compounding Leak
    r2 = verify_zero_compounding_leak_invariant(
        [10.0, 15.0, 12.0],
        [10.0, 15.0, 12.0],
        raise_on_fail=True,
    )
    assert r2["passed"] is True

    # Invariant 3: Clock Shift Invariance
    ts_list = [100, 200, 300, 400]
    r3 = verify_clock_shift_invariance(
        ts_list,
        ts_list,
        eval_start_ts=100,
        raise_on_fail=True,
    )
    assert r3["passed"] is True

    # Invariant 4: Accounting Identity
    r4 = verify_accounting_identity_invariant(
        net_pnl=95.0,
        sleeve_gross_pnls=[100.0],
        fee_components=[5.0],
        raise_on_fail=True,
    )
    assert r4["passed"] is True

    # Invariant 5: Lot Neutrality
    r5 = verify_lot_neutrality_invariant(
        leg_a_usd=100.0,
        leg_b_usd=102.5,
        max_imbalance_usd=10.0,
        raise_on_fail=True,
    )
    assert r5["passed"] is True

    print("[5/6] Warmup readiness & 5 Quant Invariants ... PASS")


def _step_6_barbell_execution_and_audit() -> None:
    bars = 950
    idx = pd.date_range("2025-01-01", periods=bars, freq="1h", tz="UTC")
    base = {BTC: 96000.0, ETH: 4200.0, SOL: 150.0, BNB: 600.0}
    slopes = {BTC: 1.0, ETH: 2.5, SOL: -1.8, BNB: -2.0}

    universe: dict[str, pd.DataFrame] = {}
    for sym in CORE4:
        close = base[sym] * (1.0 + slopes[sym] * np.arange(bars) / 1000.0)
        universe[sym] = pd.DataFrame(
            {
                "close": close,
                "spot_close": close * 0.9998,
                "perp_close": close,
                "basis_pct": 0.0,
                "funding_rate_event": 0.0001,
                "last_known_funding_rate": 0.0001,
            },
            index=idx,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        state_path = tmp_path / "state.json"

        config = dataclasses.replace(
            TradingConfig(),
            timeframe="1h",
            raw_data_dir=raw_dir,
            state_file_path=state_path,
        )
        overlay_cfg = BarbellConfig(
            initial_capital=3000.0,
            overlay_min_snr_dispersion=0.5,
            overlay_lookback_hours=336,
        )
        executor = CarryLiveExecutor(
            config=config,
            carry_config=DynamicCarryConfig(initial_capital=3000.0),
            mode="paper",
            symbols=list(CORE4),
            state_store=StateStore(state_path),
            raw_data_dir=raw_dir,
            capital=3000.0,
            initial_capital=3000.0,
            execution_style="maker_taker_hedge",
            overlay_config=overlay_cfg,
            tier1b_gate=True,
        )

        scores = {sym: 0.0 for sym in CORE4}
        alloc = executor.build_barbell_allocations(universe, scores)

        delta = getattr(executor, "last_overlay_delta", {})
        if "net_delta_usd" in delta:
            net_delta_usd = abs(float(delta["net_delta_usd"]))
        else:
            overlay = alloc.get("momentum_overlay", {})
            allocs = overlay.get("allocations", [])
            long_usd = sum(
                float(a["notional"]) for a in allocs if "long" in str(a.get("side", "")).lower()
            )
            short_usd = sum(
                float(a["notional"]) for a in allocs if "short" in str(a.get("side", "")).lower()
            )
            net_delta_usd = abs(long_usd - short_usd)

        assert net_delta_usd <= 10.0, f"Overlay net delta {net_delta_usd} exceeds 10.0 USD"

        # Xuat 1 dong JSONL audit
        audit_file = tmp_path / "preflight_audit.jsonl"
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
            overlay_neutrality={"applied": True},
            kill_switch=False,
            diagnostics_summary="preflight_test",
            actions={"carry": [], "overlay": []},
        )
        written = append_audit_record(audit_file, rec)
        assert written is True
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["schema"] == "sprint16.cycle.v1"
        assert parsed["cycle"] == 1

    print("[6/6] Barbell execution & JSONL cycle audit ... PASS")


def main() -> int:
    try:
        _step_1_hmac_signing()
        snap = _step_2_mock_payload_conversion()
        _step_3_reconcile_synced(snap)
        _step_4_bad_scenarios(snap)
        _step_5_warmup_and_invariants()
        _step_6_barbell_execution_and_audit()
        print("PREFLIGHT: 6/6 PASS")
        return 0
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
