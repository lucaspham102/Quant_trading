"""Test Sprint 16 Chang 2 - phan ra PnL theo nhanh, ty le giai ngan va cua so 30d/151d.

Bao phu:
1. `compute_sleeve_attribution` tach dung carry_net / overlay_net / net.
2. Dang thuc cua so: first_30d + last_151d == net.
3. `avg_carry_slots_held` va `carry_utilization_pct` tinh dung tren holdings_log.
4. Thieu attribution_series -> available=False, moi field = 0.
5. Holdings log ngoai cua so holdout bi loai.
6. Bridge gan sleeve cho tung phuong phap va khop 2 dang thuc < 1e-6.
7. Bridge tinh chenh lech B - D' theo nhanh va cua so (carry/overlay/30d/151d).
8. Phuong phap thieu du lieu sleeve -> available=False, khong lam vo dang thuc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sprint11_eval import (
    SPRINT11_HOLDOUT_END,
    SPRINT11_INITIAL_CAPITAL,
    SPRINT11_SPLIT_DATE,
    compute_holdout_accounting_bridge,
    compute_sleeve_attribution,
)


def _attr_frame(hours: int = 4344) -> pd.DataFrame:
    """Attribution 1h bat dau dung tai split_date, tong PnL kiem soat duoc."""
    idx = pd.date_range(SPRINT11_SPLIT_DATE, periods=hours, freq="1h", tz="UTC")
    n = len(idx)
    funding = pd.Series(np.full(n, 0.02), index=idx)
    basis = pd.Series(np.full(n, 0.01), index=idx)
    overlay_gross = pd.Series(np.full(n, 0.004), index=idx)
    overlay_fees = pd.Series(np.full(n, 0.001), index=idx)
    carry_fees = pd.Series(np.full(n, 0.002), index=idx)
    overlay_pnl = overlay_gross - overlay_fees
    net = funding + basis + overlay_pnl - carry_fees
    # Giong engine that: nen dau tien khong co PnL nao (moi cot = 0).
    for series in (funding, basis, overlay_gross, overlay_fees, carry_fees, overlay_pnl, net):
        series.iloc[0] = 0.0
    return pd.DataFrame(
        {
            "funding_pnl": funding,
            "basis_pnl": basis,
            "overlay_gross_pnl": overlay_gross,
            "overlay_pnl": overlay_pnl,
            "carry_fees": carry_fees,
            "overlay_fees": overlay_fees,
            "fees": carry_fees + overlay_fees,
            "net_pnl": net,
        },
        index=idx,
    )


def _holdings_log(n_active_values: list[int]) -> pd.DataFrame:
    idx = pd.date_range(SPRINT11_SPLIT_DATE, periods=len(n_active_values), freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "time": idx,
            "symbols": [["A"]] * len(idx),
            "modes": ["x"] * len(idx),
            "n_active": n_active_values,
            "total_notional": [3000.0] * len(idx),
            "equity": [3000.0] * len(idx),
        }
    )


# ---------------------------------------------------------------------------
# 1-3. compute_sleeve_attribution
# ---------------------------------------------------------------------------


def test_sleeve_attribution_splits_carry_and_overlay() -> None:
    attr = _attr_frame()
    res = compute_sleeve_attribution(attr, None)

    hours = int(res["holdout_hours"])
    expected_carry = (0.02 + 0.01 - 0.002) * (hours - 1)
    expected_overlay = (0.004 - 0.001) * (hours - 1)
    assert res["available"] is True
    assert res["carry_net_pnl_usd"] == pytest.approx(expected_carry)
    assert res["overlay_net_pnl_usd"] == pytest.approx(expected_overlay)
    assert res["carry_net_pnl_usd"] + res["overlay_net_pnl_usd"] == pytest.approx(
        res["net_pnl_usd"]
    )


def test_sleeve_attribution_window_identity() -> None:
    attr = _attr_frame()
    res = compute_sleeve_attribution(attr, None)
    assert res["holdout_hours"] >= len(attr) - 24
    assert res["first_30d_net_pnl_usd"] + res["last_151d_net_pnl_usd"] == pytest.approx(
        res["net_pnl_usd"]
    )


def test_sleeve_attribution_utilization_from_holdings_log() -> None:
    attr = _attr_frame()
    log = _holdings_log([4, 4, 3, 2, 4])
    res = compute_sleeve_attribution(attr, log, n_carry_slots=4)
    assert res["avg_carry_slots_held"] == pytest.approx(3.4)
    assert res["carry_utilization_pct"] == pytest.approx(85.0)


# ---------------------------------------------------------------------------
# 4-5. Fallback va loc cua so
# ---------------------------------------------------------------------------


def test_missing_attribution_returns_unavailable() -> None:
    res = compute_sleeve_attribution(None, None)
    assert res["available"] is False
    assert res["carry_net_pnl_usd"] == 0.0
    assert res["first_30d_net_pnl_usd"] == 0.0


def test_holdings_outside_holdout_window_are_ignored() -> None:
    attr = _attr_frame()
    log = _holdings_log([4, 4, 4])
    outside = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]),
            "symbols": [["A"], ["A"]],
            "modes": ["x", "x"],
            "n_active": [0, 0],
            "total_notional": [0.0, 0.0],
            "equity": [3000.0, 3000.0],
        }
    )
    combined = pd.concat([outside, log], ignore_index=True)
    res = compute_sleeve_attribution(attr, combined, n_carry_slots=4)
    assert res["avg_carry_slots_held"] == pytest.approx(4.0)
    assert res["carry_utilization_pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 6-8. Bridge
# ---------------------------------------------------------------------------


def _payload(net_holdout: float, sleeve: dict | None, resim_net: float = 97.07) -> dict:
    row = {
        "combo": "tier1b/smooth_conviction",
        "holdout_equity_start": 3000.0,
        "holdout_net_pnl": net_holdout,
        "holdout_net_pnl_resim": resim_net,
        "holdout_resim_entry_fee_usd": 2.02,
        "holdout_days": 181.0,
    }
    if sleeve is not None:
        row["holdout_sleeve_attribution"] = sleeve
    return {"meta": {"capital": 3000.0}, "tiers": {"capital": 3000.0, "rows": [row]}}


def _sleeve(carry: float, overlay: float, first_30d: float, last: float) -> dict:
    return {
        "available": True,
        "carry_net_pnl_usd": carry,
        "overlay_net_pnl_usd": overlay,
        "avg_carry_slots_held": 3.63,
        "carry_utilization_pct": 90.7,
        "first_30d_net_pnl_usd": first_30d,
        "last_151d_net_pnl_usd": last,
    }


def test_bridge_attaches_sleeves_and_keeps_identities() -> None:
    compound = _payload(159.15, _sleeve(55.62, 103.53, 1.72, 157.43))
    fixed = _payload(118.50, _sleeve(46.16, 72.34, 1.93, 116.57))
    bridge = compute_holdout_accounting_bridge(compound, fixed)

    block = bridge["sleeve_attribution"]
    a = block["methods"]["A"]
    b = block["methods"]["B"]
    assert a["available"] is True and b["available"] is True
    assert a["carry_net_pnl_usd"] == pytest.approx(55.62)
    assert a["carry_net_pnl_usd"] + a["overlay_net_pnl_usd"] == pytest.approx(159.15)
    assert b["first_30d_net_pnl_usd"] + b["last_151d_net_pnl_usd"] == pytest.approx(118.50)
    for name in ("A", "B"):
        assert abs(block["identity"][name]["sleeve_gap_usd"]) < 1e-6
        assert abs(block["identity"][name]["window_gap_usd"]) < 1e-6


def test_bridge_delta_b_vs_d_aligned_components() -> None:
    compound = _payload(159.15, _sleeve(55.62, 103.53, 1.72, 157.43))
    fixed = _payload(118.50, _sleeve(46.16, 72.34, 1.93, 116.57))
    warm_cash = {
        "net_pnl_usd": 81.73,
        "entry_fee_usd": 2.02,
        "warmup_hours": 504,
        "trade_start_idx": 504,
        "start_equity_usd": 3000.0,
        "end_equity_usd": 3081.73,
        "return_on_active_capital_pct": 2.72,
        "sleeve_attribution": _sleeve(41.01, 40.72, -16.09, 97.82),
    }
    warm_cash_aligned = {
        "net_pnl_usd": 76.09,
        "entry_fee_usd": 2.02,
        "warmup_hours": 504,
        "trade_start_idx": 504,
        "rebalance_phase_offset": 22,
        "start_equity_usd": 3000.0,
        "end_equity_usd": 3076.09,
        "return_on_active_capital_pct": 2.54,
        "sleeve_attribution": _sleeve(41.01, 35.08, -15.41, 91.51),
    }
    bridge = compute_holdout_accounting_bridge(
        compound, fixed, warm_cash=warm_cash, warm_cash_aligned=warm_cash_aligned
    )
    delta = bridge["sleeve_attribution"]["delta_b_vs_d_aligned"]

    assert delta["available"] is True
    assert delta["total_usd"] == pytest.approx(118.50 - 76.09)
    assert delta["carry_component_usd"] == pytest.approx(46.16 - 41.01)
    assert delta["overlay_component_usd"] == pytest.approx(72.34 - 35.08)
    assert delta["first_30d_component_usd"] == pytest.approx(1.93 + 15.41)
    assert abs(delta["residual_usd"]) < 1e-6


def test_bridge_without_sleeve_payload_is_safe() -> None:
    compound = _payload(159.15, None)
    fixed = _payload(118.50, None)
    bridge = compute_holdout_accounting_bridge(compound, fixed)
    block = bridge["sleeve_attribution"]
    assert block["methods"]["A"]["available"] is False
    assert block["delta_b_vs_d_aligned"]["available"] is False
    assert block["delta_b_vs_d_aligned"]["total_usd"] == pytest.approx(0.0)
