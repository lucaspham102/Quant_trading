"""Test Suite: test_quant_metamorphic_invariants.py
Kiem dinh 5 Dinh luat Bat bien Dinh luong (5 Golden Quant Invariants)
Bao gom ca 2 chieu:
- Chieu PASS: Du lieu chuan Frozen Alpha (Method B vs E, Warmup >= 720h, Holdout Bridge Sprint 17)
- Chieu CATCH BUG (Mutant Injection): 5 loi lich su duoc bom vao de xac nhan bi bat song lap tuc.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.quant_invariants import (
    verify_warmup_saturation_invariant,
    verify_zero_compounding_leak_invariant,
    verify_clock_shift_invariance,
    verify_accounting_identity_invariant,
    verify_lot_neutrality_invariant,
)


# ===========================================================================
# 1. INVARIANT 1: WARMUP & SLICING SATURATION
# ===========================================================================

def test_warmup_saturation_pass_method_b_vs_e() -> None:
    """Chieu PASS: So sanh trong so giua Full Run (Method B) va Warm Slice 720h (Method E).
    Trong so hoi tu cao, Jaccard sign dat 1.0 (>= 0.98), L1 diff rat nho (0.008 <= 0.02).
    """
    weights_b = {
        "BTCUSDT": 0.250,
        "ETHUSDT": -0.250,
        "SOLUSDT": 0.150,
        "AVAXUSDT": -0.150,
        "DOGEUSDT": 0.100,
        "ADAUSDT": -0.100,
    }
    # Method E (720h warmup) rat sat Method B
    weights_e = {
        "BTCUSDT": 0.248,
        "ETHUSDT": -0.251,
        "SOLUSDT": 0.152,
        "AVAXUSDT": -0.149,
        "DOGEUSDT": 0.099,
        "ADAUSDT": -0.101,
    }
    data = {720: weights_e, 1440: weights_b}
    res = verify_warmup_saturation_invariant(data, min_jaccard=0.98, max_l1_diff=0.02, raise_on_fail=True)

    assert res["passed"] is True
    assert res["invariant_name"] == "WARMUP_SATURATION"
    assert res["jaccard"] >= 0.98
    assert res["l1_diff"] <= 0.02
    assert "[PASS]" in res["details"]


def test_warmup_saturation_mutant_undershoot_504h() -> None:
    """Chieu CATCH BUG: Bom loi warmup thieu 504h (Method D') khien huong vi the bi lech pha,
    Jaccard sign tut xuong 0.559 (< 0.98) va L1 lech 0.14 (> 0.02).
    Bat song loi WARMUP_UNDERSHOOT!
    """
    weights_full = {
        "BTCUSDT": 0.25,
        "ETHUSDT": -0.25,
        "SOLUSDT": 0.15,
        "AVAXUSDT": -0.15,
        "LINKUSDT": 0.10,
        "DOTUSDT": -0.10,
    }
    # Warmup 504h thieu hoi tu, dao nguoc dau LINK/DOT va mat SOL/AVAX
    weights_undershoot_504h = {
        "BTCUSDT": 0.20,
        "ETHUSDT": -0.20,
        "SOLUSDT": -0.12,  # Dao chieu
        "AVAXUSDT": 0.12,   # Dao chieu
        "LINKUSDT": -0.10,  # Dao chieu
        "DOTUSDT": 0.10,   # Dao chieu
    }
    data = {504: weights_undershoot_504h, 1440: weights_full}

    # 1. Kiem tra tra ve passed == False khi raise_on_fail=False
    res = verify_warmup_saturation_invariant(data, min_jaccard=0.98, max_l1_diff=0.02, raise_on_fail=False)
    assert res["passed"] is False
    assert "WARMUP_UNDERSHOOT" in res["details"]
    assert res["max_error"] > 0.0

    # 2. Kiem tra raise AssertionError khi raise_on_fail=True
    with pytest.raises(AssertionError, match="WARMUP_UNDERSHOOT"):
        verify_warmup_saturation_invariant(data, min_jaccard=0.98, max_l1_diff=0.02, raise_on_fail=True)


def test_warmup_saturation_empty_weights_edge_case() -> None:
    """Edge case: ca 2 cua so deu khong co vi the (weights rong) -> saturation dat chuan."""
    data = {720: {}, 1440: {}}
    res = verify_warmup_saturation_invariant(data, raise_on_fail=True)
    assert res["passed"] is True
    assert res["jaccard"] == 1.0


# ===========================================================================
# 2. INVARIANT 2: ZERO COMPOUNDING LEAK
# ===========================================================================

def test_zero_compounding_leak_pass_frozen_alpha() -> None:
    """Chieu PASS: Khi reinvest_profits=False va current_equity >= initial_capital (3000 USD),
    notional base bi clamp o min(initial_capital, current_equity) = 3000 USD.
    Hai danh muc co current_equity 3000 va 5000 USD phai sinh chuoi PnL USD hoan toan trung nhau.
    """
    # PnL series cho 6 ky danh gia
    pnl_series_low = [14.25, -6.80, 22.10, -3.50, 8.90, 11.20]
    pnl_series_high = [14.25, -6.80, 22.10, -3.50, 8.90, 11.20]

    res = verify_zero_compounding_leak_invariant(pnl_series_low, pnl_series_high, tol=1e-6, raise_on_fail=True)
    assert res["passed"] is True
    assert res["invariant_name"] == "ZERO_COMPOUNDING_LEAK"
    assert res["max_error"] <= 1e-6
    assert "[PASS]" in res["details"]


def test_zero_compounding_leak_mutant_equity_4091() -> None:
    """Chieu CATCH BUG: Bom loi ro ri lai kep lich su khi size vi the vo tinh nhan voi
    equity = 4091 USD thay vi initial_capital = 3000 USD.
    Ti le khuech dai: 4091 / 3000 = 1.3637x khien PnL USD bi lech lon.
    Bat song loi COMPOUNDING_CAPITAL_LEAK!
    """
    pnl_base = [10.0, -5.0, 15.0, 20.0, -8.0]
    # Nhan theo equity = 4091.0 / 3000.0 = 1.363667
    scale_factor = 4091.0 / 3000.0
    pnl_leaked = [p * scale_factor for p in pnl_base]

    res = verify_zero_compounding_leak_invariant(pnl_base, pnl_leaked, tol=1e-6, raise_on_fail=False)
    assert res["passed"] is False
    assert "COMPOUNDING_CAPITAL_LEAK" in res["details"]
    assert res["max_error"] > 1.0  # Chenh lech toi da 20 * 0.3637 ~ 7.27 USD

    with pytest.raises(AssertionError, match="COMPOUNDING_CAPITAL_LEAK"):
        verify_zero_compounding_leak_invariant(pnl_base, pnl_leaked, tol=1e-6, raise_on_fail=True)


def test_zero_compounding_leak_mismatched_lengths() -> None:
    """Edge case: chuoi PnL lech chieu dai -> bat loi ngay lap tuc."""
    res = verify_zero_compounding_leak_invariant([1.0, 2.0], [1.0], raise_on_fail=False)
    assert res["passed"] is False
    assert "Series lengths mismatch" in res["details"]


# ===========================================================================
# 3. INVARIANT 3: CLOCK SHIFT INVARIANCE
# ===========================================================================

def test_clock_shift_invariance_pass_absolute_timestamps() -> None:
    """Chieu PASS: Hai lan chay co do dai lich su khac nhau (Run A: 720h, Run B: 1440h)
    nhung deu neo vao moc UTC tuyet doi (rebalance moi 48h ke tu 2026-04-21 00:00:00 UTC).
    Cac moc tai can bang trong khoang danh gia [2026-04-21 -> 2026-04-29] trung khop 100%.
    """
    eval_start = "2026-04-21 00:00:00"
    # Run A bat dau tu 2026-03-22 (720h warmup)
    run_a_ts = pd.date_range("2026-03-22 00:00:00", "2026-04-30 00:00:00", freq="48h", tz="UTC")
    # Run B bat dau tu 2026-02-20 (1440h warmup)
    run_b_ts = pd.date_range("2026-02-20 00:00:00", "2026-04-30 00:00:00", freq="48h", tz="UTC")

    res = verify_clock_shift_invariance(run_a_ts, run_b_ts, eval_start_ts=eval_start, raise_on_fail=True)
    assert res["passed"] is True
    assert res["invariant_name"] == "CLOCK_SHIFT_INVARIANCE"
    assert res["jaccard"] == 1.0000
    assert res["diff_count"] == 0
    assert "[PASS]" in res["details"]


def test_clock_shift_mutant_local_idx_modulo_48() -> None:
    """Chieu CATCH BUG: Bom loi cat lat DataFrame roi dung `local_idx % 48 == 0`.
    Vi Run A va Run B co do dai warmup lech nhau 7 tieng (hoac 22 tieng),
    local modulo 48 se lam lech pha toan bo gio tai can bang trong holdout!
    Bat song loi REBALANCE_CLOCK_PHASE_SHIFT!
    """
    eval_start = "2026-04-21 00:00:00"
    # Run A: rebalance vao 00:00 moi 48h
    run_a_ts = pd.date_range("2026-04-21 00:00:00", "2026-05-15 00:00:00", freq="48h", tz="UTC")
    # Run B bi lech pha 22 tieng do cat lat local_idx % 48: rebalance vao 22:00
    run_b_shifted_ts = pd.date_range("2026-04-21 22:00:00", "2026-05-15 22:00:00", freq="48h", tz="UTC")

    res = verify_clock_shift_invariance(run_a_ts, run_b_shifted_ts, eval_start_ts=eval_start, raise_on_fail=False)
    assert res["passed"] is False
    assert "REBALANCE_CLOCK_PHASE_SHIFT" in res["details"]
    assert res["jaccard"] == 0.0  # Khong trung bat ky moc nao do lech gio

    with pytest.raises(AssertionError, match="REBALANCE_CLOCK_PHASE_SHIFT"):
        verify_clock_shift_invariance(run_a_ts, run_b_shifted_ts, eval_start_ts=eval_start, raise_on_fail=True)


# ===========================================================================
# 4. INVARIANT 4: ACCOUNTING IDENTITY (NET == GROSS - FEES)
# ===========================================================================

def test_accounting_identity_pass_holdout_bridge() -> None:
    """Chieu PASS: Dang thuc ke toan tu Sprint 17 Holdout Bridge:
    Net_PnL == Carry_Gross + Overlay_Gross - Carry_Fee - Overlay_Fee (sai so < 1e-9).
    So lieu Sprint 17: Carry_Gross = 58.42, Overlay_Gross = 76.55, Carry_Fee = 9.80, Overlay_Fee = 10.90 -> Net = 114.27.
    """
    carry_gross = 58.42
    overlay_gross = 76.55
    carry_fee = 9.80
    overlay_fee = 10.90
    expected_net = carry_gross + overlay_gross - carry_fee - overlay_fee  # 114.27

    res = verify_accounting_identity_invariant(
        net_pnl=expected_net,
        sleeve_gross_pnls=[carry_gross, overlay_gross],
        fee_components=[carry_fee, overlay_fee],
        tol=1e-9,
        raise_on_fail=True,
    )
    assert res["passed"] is True
    assert res["invariant_name"] == "ACCOUNTING_IDENTITY"
    assert res["max_error"] < 1e-9
    assert "[PASS]" in res["details"]


def test_accounting_identity_mutant_missing_carry_fee_15_70() -> None:
    """Chieu CATCH BUG: Bom loi quen tru 15.70 USD phi carry (hoac bo quen phi funding settlement).
    Net PnL cao hon thuc te 15.70 USD.
    Bat song loi ACCOUNTING_IDENTITY_VIOLATION!
    """
    sleeve_gross = [100.0, 50.0]  # Tong gross = 150.0
    actual_fees = [15.70, 8.50]   # Tong phi thuc te = 24.20 -> Net dung phai la 125.80
    # Net bi khai gian (quen tru 15.70 USD phi carry)
    fraudulent_net = 150.0 - 8.50  # 141.50

    res = verify_accounting_identity_invariant(
        net_pnl=fraudulent_net,
        sleeve_gross_pnls=sleeve_gross,
        fee_components=actual_fees,
        tol=1e-9,
        raise_on_fail=False,
    )
    assert res["passed"] is False
    assert "ACCOUNTING_IDENTITY_VIOLATION" in res["details"]
    assert pytest.approx(res["max_error"], abs=1e-6) == 15.70

    with pytest.raises(AssertionError, match="ACCOUNTING_IDENTITY_VIOLATION"):
        verify_accounting_identity_invariant(
            net_pnl=fraudulent_net,
            sleeve_gross_pnls=sleeve_gross,
            fee_components=actual_fees,
            tol=1e-9,
            raise_on_fail=True,
        )


def test_accounting_identity_mutant_double_fee_charge() -> None:
    """Chieu CATCH BUG: Bom loi tru phi mo vi the 2 lan (vua o signal gate vua o live executor).
    Discrepancy = 12.50 USD. Bat song ngay lap tuc!
    """
    sleeve_gross = [80.0, 40.0]
    fee_single = [6.25, 4.00]
    # Net bi tru phi 6.25 hai lan: 120 - 6.25*2 - 4.0 = 103.50
    double_fee_net = 80.0 + 40.0 - (6.25 * 2) - 4.00

    res = verify_accounting_identity_invariant(
        net_pnl=double_fee_net,
        sleeve_gross_pnls=sleeve_gross,
        fee_components=fee_single,
        tol=1e-9,
        raise_on_fail=False,
    )
    assert res["passed"] is False
    assert pytest.approx(res["max_error"], abs=1e-6) == 6.25


# ===========================================================================
# 5. INVARIANT 5: LOT NEUTRALITY & NAKED LEG PREVENTION
# ===========================================================================

def test_lot_neutrality_pass_balanced_overlay() -> None:
    """Chieu PASS: Sau khi lam tron lo theo EXCHANGE_LOT_SPECS,
    chenh lech giua chan Long va Short la 4.50 USDT (<= 10.0 USDT).
    """
    leg_long = 1498.20
    leg_short = 1502.70
    res = verify_lot_neutrality_invariant(leg_long, leg_short, max_imbalance_usd=10.0, raise_on_fail=True)
    assert res["passed"] is True
    assert res["invariant_name"] == "LOT_NEUTRALITY"
    assert res["imbalance_usd"] == pytest.approx(4.50, abs=1e-6)
    assert "[PASS]" in res["details"]


def test_lot_neutrality_mutant_imbalance_76_62_usdt() -> None:
    """Chieu CATCH BUG: Bom loi lech lo Overlay -76.62 USDT (Leg Long = 1500 USD, Leg Short = 1423.38 USD).
    Vuot qua nguong cho phep 10 USDT.
    Bat song loi LOT_QUANTIZATION_DELTA_LEAK!
    """
    leg_long = 1500.00
    leg_short = 1423.38  # Chenh lech 76.62 USDT
    res = verify_lot_neutrality_invariant(leg_long, leg_short, max_imbalance_usd=10.0, raise_on_fail=False)

    assert res["passed"] is False
    assert "LOT_QUANTIZATION_DELTA_LEAK" in res["details"]
    assert pytest.approx(res["imbalance_usd"], abs=1e-6) == 76.62

    with pytest.raises(AssertionError, match="LOT_QUANTIZATION_DELTA_LEAK"):
        verify_lot_neutrality_invariant(leg_long, leg_short, max_imbalance_usd=10.0, raise_on_fail=True)


def test_lot_neutrality_mutant_naked_leg() -> None:
    """Chieu CATCH BUG: Bom loi Naked Leg tham khoc (Leg Long = 1500 USD nhung Leg Short = 0 USD do bi cat lot step).
    Bat song loi NAKED_LEG!
    """
    leg_long = 1500.00
    leg_short = 0.00
    res = verify_lot_neutrality_invariant(leg_long, leg_short, max_imbalance_usd=10.0, raise_on_fail=False)

    assert res["passed"] is False
    assert "NAKED_LEG" in res["details"]
    assert res["max_error"] == 1500.00

    with pytest.raises(AssertionError, match="NAKED_LEG"):
        verify_lot_neutrality_invariant(leg_long, leg_short, max_imbalance_usd=10.0, raise_on_fail=True)


def test_lot_neutrality_pass_flat_position() -> None:
    """Edge case: Ca 2 chan deu bang 0 (flat position) -> PASS."""
    res = verify_lot_neutrality_invariant(0.0, 0.0, max_imbalance_usd=10.0, raise_on_fail=True)
    assert res["passed"] is True
    assert res["imbalance_usd"] == 0.0
