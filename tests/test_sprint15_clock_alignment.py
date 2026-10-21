"""Test Sprint 15 Chang 2 - dong ho tai can bang 48h cua Method D.

Bao phu:
1. `build_rebalance_mask` neo moc 48h theo `phase_offset` va luon bat nen 0.
2. `phase_offset=0` giu nguyen hanh vi cu (mask = n % 48 == 0).
3. `weights_changed` hop nhat dung vao mask (khong mat moc tai can bang do trong so doi).
4. `run_indicator_warm_cash_start(phase_aligned=True)` tinh offset = global_start_idx % 48.
5. `compute_holdout_accounting_bridge` tach duoc clock_phase_drift va giu dang thuc 4 thanh phan.
6. Khong truyen `warm_cash_aligned` -> clock_phase_drift = 0 (tuong thich nguoc Sprint 14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import build_rebalance_mask
from src.sprint11_eval import compute_holdout_accounting_bridge, run_indicator_warm_cash_start


# ---------------------------------------------------------------------------
# 1-3. build_rebalance_mask
# ---------------------------------------------------------------------------


def test_rebalance_mask_uses_phase_offset() -> None:
    mask_0 = build_rebalance_mask(96, 48, phase_offset=0)
    mask_22 = build_rebalance_mask(96, 48, phase_offset=22)

    hours_0 = set(np.flatnonzero(mask_0.to_numpy()))
    hours_22 = set(np.flatnonzero(mask_22.to_numpy()))
    # 0 luon bat (nen dau tien); moc 48h dich theo offset
    assert 0 in hours_0 and 0 in hours_22
    assert 48 in hours_0
    assert 26 in hours_22  # (26 + 22) % 48 == 0
    assert 48 not in hours_22


def test_rebalance_mask_default_matches_legacy_behaviour() -> None:
    legacy = np.arange(200) % 48 == 0
    legacy[0] = True
    mask = build_rebalance_mask(200, 48)
    assert np.array_equal(mask.to_numpy(), legacy)


def test_rebalance_mask_unions_weights_changed() -> None:
    idx = pd.RangeIndex(100)
    changed = pd.Series(False, index=idx)
    changed.iloc[37] = True
    mask = build_rebalance_mask(100, 48, weights_changed=changed, index=idx)
    assert bool(mask.iloc[37]) is True
    assert int(mask.sum()) >= 4  # 0, 48, 96 va 37


# ---------------------------------------------------------------------------
# 4. run_indicator_warm_cash_start phase alignment
# ---------------------------------------------------------------------------


def _synthetic_data(bars: int = 900) -> dict[str, pd.DataFrame]:
    """Du lieu tong hop 3 symbol du de chay Method D (khong mang, khong parquet)."""
    idx = pd.date_range("2024-01-01", periods=bars, freq="1h", tz="UTC")
    slopes = {"AAA/USDT:USDT": 0.0012, "BBB/USDT:USDT": -0.0008, "CCC/USDT:USDT": 0.0004}
    out: dict[str, pd.DataFrame] = {}
    for sym, slope in slopes.items():
        close = 100.0 + np.arange(bars) * slope
        out[sym] = pd.DataFrame(
            {
                "perp_close": close,
                "spot_close": close * 0.9998,
                "close": close,
                "funding_rate_event": np.full(bars, 0.00008),
            },
            index=idx,
        )
    return out


def test_phase_aligned_offset_matches_global_start_index() -> None:
    data = _synthetic_data(900)
    split = "2024-02-01"
    ext_start = pd.Timestamp(split, tz="UTC") - pd.Timedelta(hours=504)
    global_start_idx = int(data["AAA/USDT:USDT"].index.searchsorted(ext_start, side="left"))
    expected_offset = global_start_idx % 48

    warm = run_indicator_warm_cash_start(
        data=data, split_date=split, holdout_end="2024-02-20", phase_aligned=True
    )
    assert warm["phase_aligned"] is True
    assert warm["rebalance_phase_offset"] == expected_offset

    legacy = run_indicator_warm_cash_start(
        data=data, split_date=split, holdout_end="2024-02-20"
    )
    assert legacy["phase_aligned"] is False
    assert legacy["rebalance_phase_offset"] == 0


# ---------------------------------------------------------------------------
# 5-6. bridge accounting
# ---------------------------------------------------------------------------


def _payload(holdout_net: float, resim_net: float, entry_fee: float) -> dict:
    return {
        "meta": {"capital": 3000.0},
        "rows": [
            {
                "combo": "tier1b_smooth_conviction",
                "holdout_equity_start": 3000.0,
                "holdout_net_pnl": holdout_net,
                "holdout_net_pnl_resim": resim_net,
                "holdout_resim_entry_fee_usd": entry_fee,
                "holdout_days": 181.0,
            }
        ],
    }


def test_bridge_splits_clock_phase_drift_and_keeps_identity() -> None:
    compound = _payload(159.15, 97.07, 2.02)
    fixed = _payload(118.50, 97.07, 2.02)
    warm_cash = {"net_pnl_usd": 81.73, "entry_fee_usd": 2.02, "warmup_hours": 504,
                 "trade_start_idx": 504, "start_equity_usd": 3000.0,
                 "end_equity_usd": 3081.73, "return_on_active_capital_pct": 2.72}
    warm_cash_aligned = {"net_pnl_usd": 76.09, "entry_fee_usd": 2.02, "warmup_hours": 504,
                         "trade_start_idx": 504, "rebalance_phase_offset": 22,
                         "start_equity_usd": 3000.0, "end_equity_usd": 3076.09,
                         "return_on_active_capital_pct": 2.54}

    bridge = compute_holdout_accounting_bridge(
        compound, fixed, warm_cash=warm_cash, warm_cash_aligned=warm_cash_aligned
    )
    dec = bridge["warm_cash_decomposition"]
    clock = bridge["clock_phase_bridge"]

    assert dec["clock_phase_drift_usd"] == pytest.approx(76.09 - 81.73)
    assert dec["pure_inventory_carryover_usd"] == pytest.approx(118.50 - 76.09 - 2.02)
    assert dec["explained_total_usd"] == pytest.approx(21.44, abs=0.02)
    assert dec["unexplained_total_usd"] == pytest.approx(0.0, abs=1e-6)
    assert clock["identity_gap_usd"] == pytest.approx(0.0, abs=1e-6)
    assert clock["D_clock_aligned_usd"] == pytest.approx(76.09)


def test_bridge_without_aligned_payload_is_backward_compatible() -> None:
    compound = _payload(159.15, 97.07, 2.02)
    fixed = _payload(118.50, 97.07, 2.02)
    warm_cash = {"net_pnl_usd": 81.73, "entry_fee_usd": 2.02, "warmup_hours": 504,
                 "trade_start_idx": 504, "start_equity_usd": 3000.0,
                 "end_equity_usd": 3081.73, "return_on_active_capital_pct": 2.72}

    bridge = compute_holdout_accounting_bridge(compound, fixed, warm_cash=warm_cash)
    dec = bridge["warm_cash_decomposition"]

    assert bridge["indicator_warm_cash_start_clock_aligned"]["available"] is False
    assert dec["clock_phase_drift_usd"] == pytest.approx(0.0)
    assert dec["pure_inventory_carryover_usd"] == pytest.approx(dec["inventory_carryover_diff_usd"])
    assert dec["unexplained_total_usd"] == pytest.approx(0.0, abs=1e-6)
