"""Module: quant_invariants.py
Kiem dinh 5 Dinh luat Bat bien Dinh luong (5 Golden Quant Invariants)
Bao ve he thong khoi:
1. WARMUP & SLICING UNDERSHOOT (Thieu hoi tu EMA/Rolling khi cat lat)
2. ZERO COMPOUNDING LEAK (Ro ri lai kep khi reinvest_profits=False)
3. REBALANCE CLOCK PHASE SHIFT (Lech pha dong ho tai can bang 48h)
4. ACCOUNTING IDENTITY VIOLATION (Dang thuc ke toan Net vs Gross vs Fees)
5. LOT QUANTIZATION & DELTA LEAK (Lech Delta sau lam tron lo hoac Naked Leg)
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. WARMUP SATURATION INVARIANT
# ---------------------------------------------------------------------------
def verify_warmup_saturation_invariant(
    weights_by_warmup: dict[int, dict[str, float]],
    min_jaccard: float = 0.98,
    max_l1_diff: float = 0.02,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """So sanh trong so danh muc tai moc bat dau danh gia t0 giua cac cua so warmup.

    Thuong so sanh giua cua so chuan (vd 720h) va cua so gap doi (1440h),
    hoac cap cua so warmup thap nhat va cao nhat co trong dictionary.
    Neu Jaccard sign < min_jaccard hoac L1 diff > max_l1_diff -> bao loi WARMUP_UNDERSHOOT.
    """
    if len(weights_by_warmup) < 2:
        res = {
            "passed": False,
            "invariant_name": "WARMUP_SATURATION",
            "max_error": 1.0,
            "jaccard": 0.0,
            "l1_diff": 1.0,
            "details": "BLOCKER: QUANT_ARTIFACT: weights_by_warmup must contain at least 2 warmup levels",
        }
        if raise_on_fail:
            raise AssertionError(res["details"])
        return res

    sorted_keys = sorted(weights_by_warmup.keys())
    w1_key = sorted_keys[0]
    w2_key = sorted_keys[-1]
    w1 = weights_by_warmup[w1_key]
    w2 = weights_by_warmup[w2_key]

    all_symbols = sorted(set(w1.keys()) | set(w2.keys()))
    if not all_symbols:
        res = {
            "passed": True,
            "invariant_name": "WARMUP_SATURATION",
            "max_error": 0.0,
            "jaccard": 1.0,
            "l1_diff": 0.0,
            "details": "[PASS] Both weight dictionaries are empty; trivial saturation.",
        }
        return res

    # 1. Sign Jaccard similarity: compare active direction
    # A position is active if |weight| > 1e-6
    s1 = {(sym, 1 if w1.get(sym, 0.0) > 1e-6 else (-1 if w1.get(sym, 0.0) < -1e-6 else 0)) for sym in all_symbols}
    s2 = {(sym, 1 if w2.get(sym, 0.0) > 1e-6 else (-1 if w2.get(sym, 0.0) < -1e-6 else 0)) for sym in all_symbols}
    active1 = {k for k in s1 if k[1] != 0}
    active2 = {k for k in s2 if k[1] != 0}

    union_active = active1 | active2
    inter_active = active1 & active2
    if not union_active:
        jaccard = 1.0
    else:
        jaccard = len(inter_active) / len(union_active)

    # 2. L1 norm weight difference
    l1_diff = sum(abs(w1.get(sym, 0.0) - w2.get(sym, 0.0)) for sym in all_symbols)

    passed = (jaccard >= min_jaccard) and (l1_diff <= max_l1_diff)
    max_error = max(max(0.0, min_jaccard - jaccard), max(0.0, l1_diff - max_l1_diff))

    if passed:
        details = f"[PASS] Saturation achieved between {w1_key}h and {w2_key}h: Jaccard={jaccard:.4f} >= {min_jaccard}, L1={l1_diff:.6f} <= {max_l1_diff}"
    else:
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: WARMUP_UNDERSHOOT] Weights between {w1_key}h and {w2_key}h diverged: "
            f"Jaccard={jaccard:.4f} (req >= {min_jaccard}), L1={l1_diff:.6f} (req <= {max_l1_diff})"
        )

    res = {
        "passed": passed,
        "invariant_name": "WARMUP_SATURATION",
        "max_error": float(max_error),
        "jaccard": float(jaccard),
        "l1_diff": float(l1_diff),
        "details": details,
    }

    if not passed and raise_on_fail:
        raise AssertionError(details)
    return res


# ---------------------------------------------------------------------------
# 2. ZERO COMPOUNDING LEAK INVARIANT
# ---------------------------------------------------------------------------
def verify_zero_compounding_leak_invariant(
    pnl_series_low_eq: Sequence[float],
    pnl_series_high_eq: Sequence[float],
    tol: float = 1e-6,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Khi reinvest_profits=False va current_equity >= initial_capital,
    chuoi PnL USD moi ky cua danh muc co von dau ky khac nhau (cung initial_capital)
    phai trung khop tuyet doi (max_abs_diff <= tol).
    """
    if len(pnl_series_low_eq) != len(pnl_series_high_eq):
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: COMPOUNDING_CAPITAL_LEAK] Series lengths mismatch: "
            f"{len(pnl_series_low_eq)} vs {len(pnl_series_high_eq)}"
        )
        res = {
            "passed": False,
            "invariant_name": "ZERO_COMPOUNDING_LEAK",
            "max_error": float("inf"),
            "details": details,
        }
        if raise_on_fail:
            raise AssertionError(details)
        return res

    if len(pnl_series_low_eq) == 0:
        return {
            "passed": True,
            "invariant_name": "ZERO_COMPOUNDING_LEAK",
            "max_error": 0.0,
            "details": "[PASS] Empty series; no compounding leak.",
        }

    diffs = [abs(a - b) for a, b in zip(pnl_series_low_eq, pnl_series_high_eq)]
    max_abs_diff = max(diffs)
    passed = max_abs_diff <= tol

    if passed:
        details = f"[PASS] Zero compounding leak verified: max_abs_diff={max_abs_diff:.6e} <= {tol:.6e}"
    else:
        max_idx = int(np.argmax(diffs))
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: COMPOUNDING_CAPITAL_LEAK] PnL scaled with accumulated equity! "
            f"max_abs_diff={max_abs_diff:.6e} > tol={tol:.6e} at step {max_idx} "
            f"(low={pnl_series_low_eq[max_idx]:.4f}, high={pnl_series_high_eq[max_idx]:.4f})"
        )

    res = {
        "passed": passed,
        "invariant_name": "ZERO_COMPOUNDING_LEAK",
        "max_error": float(max_abs_diff),
        "details": details,
    }

    if not passed and raise_on_fail:
        raise AssertionError(details)
    return res


# ---------------------------------------------------------------------------
# 3. CLOCK SHIFT INVARIANCE
# ---------------------------------------------------------------------------
def verify_clock_shift_invariance(
    rebalance_timestamps_run_a: Iterable[Any],
    rebalance_timestamps_run_b: Iterable[Any],
    eval_start_ts: Any,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Tap hop cac moc timestamp >= eval_start_ts ma he thong thuc hien tai can bang
    giua 2 lan chay co do dai lich su moi khac nhau (vd lech nhau 7h hoac 216h o dau chuoi)
    phai trung khop 100%.
    """
    ts_eval = pd.to_datetime(eval_start_ts, utc=True) if not isinstance(eval_start_ts, (int, float)) else eval_start_ts

    def _normalize_set(ts_iter: Iterable[Any]) -> set[Any]:
        norm = set()
        for t in ts_iter:
            if isinstance(ts_eval, (int, float)):
                if t >= ts_eval:
                    norm.add(t)
            else:
                t_dt = pd.to_datetime(t, utc=True)
                if t_dt >= ts_eval:
                    norm.add(t_dt)
        return norm

    set_a = _normalize_set(rebalance_timestamps_run_a)
    set_b = _normalize_set(rebalance_timestamps_run_b)

    union_set = set_a | set_b
    inter_set = set_a & set_b

    if not union_set:
        jaccard = 1.0
    else:
        jaccard = len(inter_set) / len(union_set)

    diff_a_only = set_a - set_b
    diff_b_only = set_b - set_a
    passed = (len(diff_a_only) == 0) and (len(diff_b_only) == 0)
    max_error = 1.0 - jaccard

    if passed:
        details = (
            f"[PASS] Clock phase invariance verified: {len(inter_set)} rebalance points >= {eval_start_ts} "
            f"matched identically (Jaccard=1.0000)."
        )
    else:
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: REBALANCE_CLOCK_PHASE_SHIFT] Rebalance clocks desynchronized after slicing! "
            f"Jaccard={jaccard:.4f} < 1.0000. Run A only: {len(diff_a_only)} timestamps, "
            f"Run B only: {len(diff_b_only)} timestamps (sample: {sorted(list(diff_a_only | diff_b_only))[:3]}). "
            f"Do not use local_idx % 48; anchor to absolute timestamp or global_bar_idx."
        )

    res = {
        "passed": passed,
        "invariant_name": "CLOCK_SHIFT_INVARIANCE",
        "max_error": float(max_error),
        "jaccard": float(jaccard),
        "diff_count": len(diff_a_only) + len(diff_b_only),
        "details": details,
    }

    if not passed and raise_on_fail:
        raise AssertionError(details)
    return res


# ---------------------------------------------------------------------------
# 4. ACCOUNTING IDENTITY INVARIANT
# ---------------------------------------------------------------------------
def verify_accounting_identity_invariant(
    net_pnl: float,
    sleeve_gross_pnls: Sequence[float],
    fee_components: Sequence[float],
    tol: float = 1e-9,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Kiem tra dang thuc ke toan tuyet doi:
    Net_PnL == sum(sleeve_gross_pnls) - sum(fee_components) voi sai so <= tol.
    Phat hien double fee, missing fee hoac misattribution.
    """
    gross_total = float(sum(sleeve_gross_pnls))
    fee_total = float(sum(fee_components))
    expected_net = gross_total - fee_total
    diff = abs(net_pnl - expected_net)

    passed = diff <= tol
    if passed:
        details = (
            f"[PASS] Accounting identity closed: Net_PnL ({net_pnl:.6f}) == "
            f"Gross ({gross_total:.6f}) - Fees ({fee_total:.6f}), diff={diff:.2e} <= {tol:.2e}"
        )
    else:
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: ACCOUNTING_IDENTITY_VIOLATION] Accounting identity breached! "
            f"Net={net_pnl:.6f}, Gross={gross_total:.6f}, Fees={fee_total:.6f}, Expected_Net={expected_net:.6f}, "
            f"Discrepancy={diff:.6e} > tol={tol:.6e}. Check for double-fee charging or missing carryover fee."
        )

    res = {
        "passed": passed,
        "invariant_name": "ACCOUNTING_IDENTITY",
        "max_error": float(diff),
        "expected_net": float(expected_net),
        "gross_total": float(gross_total),
        "fee_total": float(fee_total),
        "details": details,
    }

    if not passed and raise_on_fail:
        raise AssertionError(details)
    return res


# ---------------------------------------------------------------------------
# 5. LOT NEUTRALITY INVARIANT
# ---------------------------------------------------------------------------
def verify_lot_neutrality_invariant(
    leg_a_usd: float,
    leg_b_usd: float,
    max_imbalance_usd: float = 10.0,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Kiem tra sau khi lam tron lo:
    Neu mo vi the (leg_a_usd > 0 hoac leg_b_usd > 0), khong duoc co chan nao bang 0
    (chong Naked Leg gay delta leak nguy hiem) va abs(leg_a_usd - leg_b_usd) <= max_imbalance_usd.
    """
    val_a = abs(float(leg_a_usd))
    val_b = abs(float(leg_b_usd))

    # Neu ca 2 chan deu ~ 0 -> khong co vi the (flat)
    if val_a < 1e-6 and val_b < 1e-6:
        return {
            "passed": True,
            "invariant_name": "LOT_NEUTRALITY",
            "max_error": 0.0,
            "imbalance_usd": 0.0,
            "details": "[PASS] Both legs flat; perfect zero delta.",
        }

    # Neu 1 trong 2 chan bang 0 trong khi chan kia > 0 -> NAKED LEG
    if val_a < 1e-6 or val_b < 1e-6:
        imbalance = max(val_a, val_b)
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: NAKED_LEG] Catastrophic lot quantization: One leg is zero while "
            f"other leg is active! (Leg A: {val_a:.2f} USD, Leg B: {val_b:.2f} USD). Delta unhedged: {imbalance:.2f} USD."
        )
        res = {
            "passed": False,
            "invariant_name": "LOT_NEUTRALITY",
            "max_error": float(imbalance),
            "imbalance_usd": float(imbalance),
            "details": details,
        }
        if raise_on_fail:
            raise AssertionError(details)
        return res

    # Ca 2 chan deu co gia tri: kiem tra do lech dollar
    imbalance = abs(val_a - val_b)
    passed = imbalance <= max_imbalance_usd

    if passed:
        details = (
            f"[PASS] Lot neutrality verified: Leg A={val_a:.2f} USD, Leg B={val_b:.2f} USD, "
            f"Imbalance={imbalance:.2f} USD <= {max_imbalance_usd:.2f} USD."
        )
    else:
        details = (
            f"[BLOCKER: QUANT_ARTIFACT: LOT_QUANTIZATION_DELTA_LEAK] Lot rounding imbalance exceeds threshold! "
            f"Leg A={val_a:.2f} USD, Leg B={val_b:.2f} USD, Imbalance={imbalance:.2f} USD > max={max_imbalance_usd:.2f} USD. "
            f"Check lot step size or minimum notional enforcement."
        )

    res = {
        "passed": passed,
        "invariant_name": "LOT_NEUTRALITY",
        "max_error": float(imbalance),
        "imbalance_usd": float(imbalance),
        "details": details,
    }

    if not passed and raise_on_fail:
        raise AssertionError(details)
    return res



def resolve_effective_carry_top_k(
    capital_usd: float,
    base_top_k: int = 4,
    eta: float = 0.80,
    btc_price: float = 85000.0,
    btc_min_lot: float = 0.001,
    min_lots_safety_factor: float = 1.75,
) -> int:
    """Sprint 23: top_k hieu dung theo von (>=1500 USDT giu nguyen base_top_k)."""
    def _num(v: Any, default: float) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return f if math.isfinite(f) else default

    capital = _num(capital_usd, 0.0)
    if capital <= 0.0:
        return 1
    try:
        base_k = int(base_top_k)
    except (TypeError, ValueError):
        base_k = 4
    base_k = max(1, base_k)
    if capital >= 1500.0:
        return base_k
    min_slot_notional_usd = max(
        120.0,
        _num(btc_price, 85000.0) * _num(btc_min_lot, 0.001) * _num(min_lots_safety_factor, 1.75),
    )
    deployable_usd = capital * max(0.0, _num(eta, 0.80))
    max_viable_k = max(1, int(deployable_usd // min_slot_notional_usd))
    return min(base_k, max_viable_k)


def filter_canary_lot_unviable_overlay_symbols(
    target_weights: dict[str, float],
    overlay_budget_usd: float,
    prices: dict[str, float],
    lot_steps: dict[str, float],
) -> dict[str, float]:
    """Sprint 23: loai symbol overlay co min lot USD > 45% budget, re-normalize moi phia."""
    if not isinstance(target_weights, Mapping):
        return {}
    prices = prices if isinstance(prices, Mapping) else {}
    lot_steps = lot_steps if isinstance(lot_steps, Mapping) else {}

    def _num(v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f if math.isfinite(f) else 0.0

    threshold = _num(overlay_budget_usd) * 0.45
    weights: dict[str, float] = {sym: _num(w) for sym, w in target_weights.items()}
    kept: dict[str, bool] = {}
    for sym, w in weights.items():
        keep = True
        if w != 0.0:
            px = _num(prices.get(sym))
            step = _num(lot_steps.get(sym, 0.001))
            if px > 0.0 and step > 0.0 and px * step > threshold:
                keep = False
        kept[sym] = keep

    orig_long = sum(w for w in weights.values() if w > 0.0)
    orig_short = sum(-w for w in weights.values() if w < 0.0)
    kept_long = sum(w for s, w in weights.items() if w > 0.0 and kept[s])
    kept_short = sum(-w for s, w in weights.items() if w < 0.0 and kept[s])
    long_scale = orig_long / kept_long if kept_long > 0.0 else 0.0
    short_scale = orig_short / kept_short if kept_short > 0.0 else 0.0

    out: dict[str, float] = {}
    for sym, w in weights.items():
        if w == 0.0:
            out[sym] = w
        elif not kept[sym]:
            continue
        elif w > 0.0:
            out[sym] = w * long_scale
        else:
            out[sym] = w * short_scale
    return out
