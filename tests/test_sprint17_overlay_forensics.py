"""Test Sprint 17 Chang 1 - phap y khoang cach PnL nhanh Overlay (B vs C vs D' vs E).

Bao phu:
1. Dang thuc phan ra: weight_effect + capital_effect + fee_effect == net gap (residual ~ 0).
2. Chi lech trong so (cung von) -> toan bo vao weight_effect.
3. Chi lech von co so (cung trong so) -> toan bo vao capital_effect.
4. Phi tang o phuong an alt -> fee_effect am (dung dau quy uoc).
5. Thieu du lieu -> available=False va khong crash.
6. Metric moc tai can bang 48h (Jaccard) tinh dung.
7. `_simulate_barbell_equity` xuat `notional_base` + `rebal_flag` (du lieu phap y).
8. Bridge gan `e_true_fixed_warm_cash` + `overlay_divergence_forensics` va khong pha vo test cu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import _simulate_barbell_equity
from src.sprint11_eval import (
    compute_holdout_accounting_bridge,
    decompose_overlay_pnl_divergence,
)

START = "2026-04-21"
END = "2026-04-27"


def _frame(values: list[list[float]], symbols: list[str], start: str = START, freq: str = "1h"):
    idx = pd.date_range(f"{start}", periods=len(values), freq=freq, tz="UTC")
    return pd.DataFrame(np.array(values, dtype=float), index=idx, columns=symbols)


def _series(values: list[float], start: str = START) -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="1h", tz="UTC")
    return pd.Series(np.array(values, dtype=float), index=idx)


# ---------------------------------------------------------------------------
# 1-4. Dang thuc phan ra
# ---------------------------------------------------------------------------


def test_decomposition_identity_closes() -> None:
    symbols = ["AAA", "BBB"]
    ret = _frame([[0.0, 0.0], [0.01, -0.02], [-0.03, 0.04], [0.02, 0.01], [0.0, 0.0]], symbols)
    w_ref = _frame([[0.5, -0.5]] * 5, symbols)
    w_alt = _frame([[0.3, -0.3]] * 5, symbols)
    n_ref = _series([3000.0] * 5)
    n_alt = _series([3000.0] * 5)
    fees = _series([0.0, 1.0, 0.0, 0.5, 0.0])

    out = decompose_overlay_pnl_divergence(
        w_ref, w_alt, ret, notional_ref=n_ref, notional_alt=n_alt, fees_ref=fees, fees_alt=fees
    )
    assert out["available"] is True
    total = out["weight_effect_usd"] + out["capital_effect_usd"] + out["fee_effect_usd"]
    assert out["residual_usd"] == pytest.approx(0.0, abs=1e-9)
    assert total == pytest.approx(out["net_gap_usd"], abs=1e-9)


def test_weight_effect_only_when_weights_differ() -> None:
    symbols = ["AAA", "BBB"]
    ret = _frame([[0.0, 0.0], [0.02, -0.01], [0.01, 0.03], [0.0, 0.0]], symbols)
    w_ref = _frame([[1.0, -1.0]] * 4, symbols)
    w_alt = _frame([[0.5, -0.5]] * 4, symbols)
    n = _series([3000.0] * 4)

    out = decompose_overlay_pnl_divergence(w_ref, w_alt, ret, notional_ref=n, notional_alt=n)
    assert abs(out["capital_effect_usd"]) < 1e-9
    assert abs(out["weight_effect_usd"]) > 0.0
    # Vi du: gio 1 ret = (+2%, -1%) -> ref 3% vs alt 1.5% => chenh 1.5% * 3000 = 45
    # Gio 1: ref 3% vs alt 1.5% -> -45 ; gio 2: ref -2% vs alt -1% -> +30
    assert out["weight_effect_usd"] == pytest.approx(-15.0, abs=1e-6)


def test_capital_effect_only_when_capital_differs() -> None:
    symbols = ["AAA"]
    ret = _frame([[0.0], [0.01], [0.0], [0.0]], symbols)
    w = _frame([[1.0]] * 4, symbols)
    n_ref = _series([3000.0] * 4)
    n_alt = _series([1500.0] * 4)

    out = decompose_overlay_pnl_divergence(w, w, ret, notional_ref=n_ref, notional_alt=n_alt)
    assert abs(out["weight_effect_usd"]) < 1e-9
    # 1% cua 1500 - 1% cua 3000 = -15
    assert out["capital_effect_usd"] == pytest.approx(-15.0, abs=1e-9)
    assert out["net_gap_usd"] == pytest.approx(-15.0, abs=1e-9)


def test_fee_effect_sign() -> None:
    symbols = ["AAA"]
    ret = _frame([[0.0], [0.0], [0.0]], symbols)
    w = _frame([[1.0]] * 3, symbols)
    n = _series([3000.0] * 3)
    fees_ref = _series([1.0, 1.0, 1.0])
    fees_alt = _series([5.0, 5.0, 5.0])

    out = decompose_overlay_pnl_divergence(
        w, w, ret, notional_ref=n, notional_alt=n, fees_ref=fees_ref, fees_alt=fees_alt
    )
    assert out["fee_effect_usd"] == pytest.approx(-8.0, abs=1e-9)
    assert out["net_gap_usd"] == pytest.approx(-8.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5-6. Fallback + metric tai can bang
# ---------------------------------------------------------------------------


def test_missing_inputs_return_unavailable() -> None:
    out = decompose_overlay_pnl_divergence(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert out["available"] is False
    assert out["net_gap_usd"] == 0.0
    assert out["residual_usd"] == 0.0


def test_rebalance_timing_jaccard() -> None:
    symbols = ["AAA"]
    ret = _frame([[0.0]] * 6, symbols)
    # Alt doi trong so o 2 moc khac voi ref -> mismatch 2/3 moc
    w_ref = _frame([[1.0], [1.0], [0.5], [0.5], [0.2], [0.2]], symbols)
    w_alt = _frame([[1.0], [0.6], [0.6], [0.2], [0.2], [0.2]], symbols)
    n = _series([3000.0] * 6)

    out = decompose_overlay_pnl_divergence(w_ref, w_alt, ret, notional_ref=n, notional_alt=n)
    assert out["rebal_hours_ref"] == 2
    assert out["rebal_hours_alt"] == 2
    assert out["rebal_timing_mismatch_hours"] == 4
    assert out["rebal_timing_match_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Du lieu phap y tu engine
# ---------------------------------------------------------------------------


def test_simulator_exposes_notional_base_and_rebal_flags() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    symbols = ["AAA", "BBB"]
    kwargs = {
        "raw_weights": pd.DataFrame(0.5, index=idx, columns=symbols),
        "close_matrix": pd.DataFrame(100.0, index=idx, columns=symbols),
        "funding_matrix": pd.DataFrame(0.0, index=idx, columns=symbols),
        "target_leverage": np.ones(len(idx)),
        "carry_returns": np.zeros(len(idx)),
        "initial_capital": 3000.0,
        "rebalance_hours": 4,
        "one_way_cost": 0.0,
        "quantize": False,
        "min_weight_change": 0.0,
        "dd_cut": 1.0,
        "dd_off": 1.0,
        "enable_dd_cushion": False,
        "reinvest_profits": False,
    }
    out = _simulate_barbell_equity(**kwargs)
    assert "notional_base" in out and "rebal_flag" in out
    assert len(out["notional_base"]) == len(idx)
    assert len(out["rebal_flag"]) == len(idx)
    # Fixed-base: von co so khong vuot initial_capital
    assert float(np.max(out["notional_base"])) <= 3000.0 + 1e-9
    rebal_hours = int(np.sum(out["rebal_flag"]))
    assert rebal_hours >= 1  # moc tai can bang dinh ky (t % 4 == 0, t > 0)


# ---------------------------------------------------------------------------
# 8. Bridge Sprint 17
# ---------------------------------------------------------------------------


def _payload(net_holdout: float, resim: float = 97.07) -> dict:
    row = {
        "combo": "tier1b/smooth_conviction",
        "holdout_equity_start": 3000.0,
        "holdout_net_pnl": net_holdout,
        "holdout_net_pnl_resim": resim,
        "holdout_resim_entry_fee_usd": 2.02,
        "holdout_days": 181.0,
    }
    return {"meta": {"capital": 3000.0}, "tiers": {"capital": 3000.0, "rows": [row]}}


def test_bridge_attaches_e_measure_and_forensics() -> None:
    compound = _payload(159.15)
    fixed = _payload(118.50)
    warm_cash = {"net_pnl_usd": 81.73, "entry_fee_usd": 2.02, "warmup_hours": 504}
    warm_cash_aligned = {"net_pnl_usd": 76.09, "entry_fee_usd": 2.02, "warmup_hours": 504,
                         "rebalance_phase_offset": 22}
    warm_cash_converged = {
        "net_pnl_usd": 114.27,
        "entry_fee_usd": 2.02,
        "warmup_hours": 720,
        "rebalance_phase_offset": 46,
        "overlay_net_pnl_usd": 73.26,
    }
    forensics = {
        "available": True,
        "root_cause": "warmup_ema_non_convergence",
        "overlay_pnl_usd": {"B": 72.34, "Dp": 35.09, "E": 73.26},
    }
    bridge = compute_holdout_accounting_bridge(
        compound,
        fixed,
        warm_cash=warm_cash,
        warm_cash_aligned=warm_cash_aligned,
        warm_cash_converged=warm_cash_converged,
        overlay_forensics=forensics,
    )

    e_block = bridge["e_true_fixed_warm_cash"]
    assert e_block["available"] is True
    assert e_block["label"] == "E_true_fixed_warm_cash"
    assert e_block["net_pnl_usd"] == pytest.approx(114.27)
    assert e_block["gap_b_minus_e_usd"] == pytest.approx(118.50 - 114.27)
    assert e_block["revised_pure_carryover_usd"] == pytest.approx(118.50 - 114.27 - 2.02)
    assert bridge["overlay_divergence_forensics"]["root_cause"] == "warmup_ema_non_convergence"
    # Bridge cu van nguyen: D' va decomposition Sprint 15 khong doi
    assert bridge["indicator_warm_cash_start_clock_aligned"]["net_pnl_usd"] == pytest.approx(76.09)
    assert bridge["warm_cash_decomposition"]["pure_inventory_carryover_usd"] == pytest.approx(
        118.50 - 76.09 - 2.02
    )


def test_bridge_without_e_measure_is_safe() -> None:
    bridge = compute_holdout_accounting_bridge(_payload(159.15), _payload(118.50))
    assert bridge["e_true_fixed_warm_cash"]["available"] is False
    assert bridge["e_true_fixed_warm_cash"]["gap_b_minus_e_usd"] == 0.0
    assert bridge["overlay_divergence_forensics"] == {"available": False}

def test_revised_carryover_uses_e_own_entry_fee() -> None:
    """Sprint 17: add-back phai dung phi vao lenh cua E, khong dung phi cua A/B."""
    compound = _payload(159.15)
    fixed = _payload(118.50)
    warm_cash = {"net_pnl_usd": 81.73, "entry_fee_usd": 2.02, "label": "D"}
    warm_cash_aligned = {"net_pnl_usd": 76.09, "entry_fee_usd": 2.02, "label": "Dp"}
    warm_cash_converged = {
        "net_pnl_usd": 114.27,
        "entry_fee_usd": 0.27,
        "warmup_hours": 720,
        "rebalance_phase_offset": 46,
        "overlay_net_pnl_usd": 73.26,
    }
    bridge = compute_holdout_accounting_bridge(
        compound,
        fixed,
        warm_cash=warm_cash,
        warm_cash_aligned=warm_cash_aligned,
        warm_cash_converged=warm_cash_converged,
    )
    e_block = bridge["e_true_fixed_warm_cash"]
    assert e_block["entry_fee_usd"] == pytest.approx(0.27)
    assert e_block["revised_pure_carryover_usd"] == pytest.approx(118.50 - 114.27 - 0.27)
