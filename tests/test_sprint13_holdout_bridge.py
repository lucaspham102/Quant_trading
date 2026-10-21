"""Unit test Sprint 13 — cầu nối kế toán Holdout (warm-state vs cold-start).

Kiểm tra hàm thuần ``compute_holdout_accounting_bridge`` trên payload tổng hợp:
- đọc đúng dòng primary ``tier1b/smooth_conviction``;
- tính 3 phương pháp đúng công thức (A compound, B fixed-base, C cold-start);
- đẳng thức bảo toàn ``A = C + (A-B) + (B-C)`` với sai số < 1e-6;
- phân rã phí mở danh mục / warmup;
- chịu được payload thiếu dòng fixed-base (không crash).
"""
from __future__ import annotations

import pytest

from src.sprint11_eval import compute_holdout_accounting_bridge


def _payload(
    *,
    net_slice: float,
    start_eq: float,
    net_resim: float,
    entry_fee: float,
    capital: float = 3000.0,
    combo: str = "tier1b/smooth_conviction",
    days: float = 181.0,
) -> dict:
    row = {
        "combo": combo,
        "error": None,
        "holdout_net_pnl": net_slice,
        "holdout_equity_start": start_eq,
        "holdout_net_pnl_resim": net_resim,
        "holdout_resim_entry_fee_usd": entry_fee,
        "holdout_days": days,
        "reinvest_profits": True,
    }
    other = dict(row, combo="tier1a/hard_binary", holdout_net_pnl=999.0)
    return {"tiers": {"rows": [other, row], "capital": capital}, "meta": {"capital": capital}}


def test_bridge_reads_primary_row_not_first_row() -> None:
    """Phải chọn dòng primary, không lấy dòng đầu tiên trong danh sách."""
    payload = _payload(net_slice=159.15, start_eq=4091.29, net_resim=97.07, entry_fee=2.02)
    bridge = compute_holdout_accounting_bridge(payload, None)
    assert bridge["continuous_compound"]["net_pnl_usd"] == pytest.approx(159.15)


def test_bridge_method_a_costs_and_pct() -> None:
    """Method A: tỷ suất = net / equity đầu cửa sổ."""
    payload = _payload(net_slice=159.15, start_eq=4091.29, net_resim=97.07, entry_fee=2.02)
    a = compute_holdout_accounting_bridge(payload, None)["continuous_compound"]
    assert a["start_equity_usd"] == pytest.approx(4091.29)
    assert a["end_equity_usd"] == pytest.approx(4091.29 + 159.15)
    assert a["return_on_start_equity_pct"] == pytest.approx(159.15 / 4091.29 * 100.0)


def test_bridge_method_b_uses_active_capital_not_equity() -> None:
    """Method B: tỷ suất tính trên vốn hoạt động 3.000, không phải equity 3.946."""
    comp = _payload(net_slice=159.15, start_eq=4091.29, net_resim=97.07, entry_fee=2.02)
    fixed = _payload(net_slice=118.50, start_eq=3946.72, net_resim=85.0, entry_fee=1.9)
    b = compute_holdout_accounting_bridge(comp, fixed)["continuous_fixed_base"]
    assert b["active_capital_usd"] == pytest.approx(3000.0)
    assert b["return_on_active_capital_pct"] == pytest.approx(118.50 / 3000.0 * 100.0)
    assert b["available"] is True


def test_bridge_identity_holds_within_1e6() -> None:
    """Đẳng thức A = C + (A-B) + (B-C) đúng tuyệt đối theo cấu trúc."""
    comp = _payload(net_slice=159.15, start_eq=4091.29, net_resim=97.07, entry_fee=2.02)
    fixed = _payload(net_slice=118.50, start_eq=3946.72, net_resim=85.0, entry_fee=1.9)
    bridge = compute_holdout_accounting_bridge(comp, fixed)
    a = bridge["continuous_compound"]["net_pnl_usd"]
    c = bridge["standalone_cold_start"]["net_pnl_usd"]
    d = bridge["bridge_decomposition"]
    residual = a - (c + d["compounding_scale_effect_usd"] + d["cold_start_total_drag_usd"])
    assert abs(residual) < 1e-6


def test_bridge_decomposition_splits_fee_and_warmup() -> None:
    """Phân rã: tổng drag = phí mở danh mục + lệch đường đi warmup."""
    comp = _payload(net_slice=159.15, start_eq=4091.29, net_resim=97.07, entry_fee=2.02)
    fixed = _payload(net_slice=118.50, start_eq=3946.72, net_resim=85.0, entry_fee=1.9)
    d = compute_holdout_accounting_bridge(comp, fixed)["bridge_decomposition"]
    # C luon lay tu payload compound (cold-start re-sim), khong phai tu payload fixed.
    assert d["cold_start_total_drag_usd"] == pytest.approx(118.50 - 97.07)
    assert d["warmup_and_min_hold_path_diff_usd"] == pytest.approx(
        d["cold_start_total_drag_usd"] - d["cold_start_initial_entry_fee_usd"]
    )


def test_bridge_without_fixed_base_payload_is_safe() -> None:
    """Thiếu payload fixed-base -> available False, các số vẫn khép kín."""
    comp = _payload(net_slice=159.15, start_eq=4091.29, net_resim=97.07, entry_fee=2.02)
    bridge = compute_holdout_accounting_bridge(comp, None)
    b = bridge["continuous_fixed_base"]
    assert b["available"] is False
    d = bridge["bridge_decomposition"]
    assert d["cold_start_total_drag_usd"] == pytest.approx(-97.07)
    residual = bridge["continuous_compound"]["net_pnl_usd"] - (
        bridge["standalone_cold_start"]["net_pnl_usd"]
        + d["compounding_scale_effect_usd"]
        + d["cold_start_total_drag_usd"]
    )
    assert abs(residual) < 1e-6


def test_bridge_raises_when_payload_empty() -> None:
    """Payload rỗng phải raise ValueError rõ ràng thay vì trả số 0 im lặng."""
    with pytest.raises(ValueError):
        compute_holdout_accounting_bridge({"tiers": {"rows": []}}, None)
