"""Module đánh giá và kiểm định chất lượng Sprint 10 (Production Barbell).

Cung cấp:
- run_sprint10_tiers: Mô phỏng 4 Tier Sprint 10 trên cùng chuỗi dữ liệu.
- compute_period_contributions: Bóc tách PnL các thành phần theo từng giai đoạn.
- check_executor_parity: Kiểm tra tính đồng bộ và máy trạng thái thực thi executor.
- evaluate_sprint10_go_no_go: Kiểm định 5 tiêu chí GO/NO-GO theo docs/go_no_go_sprint10.md.
- build_sprint10_summary_rows: Tạo các dòng tổng hợp chỉ số 4 Tier phục vụ xuất CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.cross_exchange_carry import CrossExchangeCarryConfig, run_cross_exchange_carry
from src.sprint9_eval import (
    build_period_table_for_tier,
    build_tier_row,
    compute_go_inputs,
    compute_ols_beta,
    detect_executor_state_machine,
)

__all__ = [
    "run_sprint10_tiers",
    "compute_period_contributions",
    "check_executor_parity",
    "evaluate_sprint10_go_no_go",
    "build_sprint10_summary_rows",
    "compute_ols_beta",
    "compute_go_inputs",
    "build_tier_row",
    "build_period_table_for_tier",
    "detect_executor_state_machine",
]


def run_sprint10_tiers(
    multi_aligned_data: dict[str, pd.DataFrame],
    btc_benchmark_returns: pd.Series,
    initial_capital: float = 3000.0,
    n_periods: int = 7,
    period_days: int = 180,
) -> dict[str, dict[str, Any]]:
    """Mô phỏng 4 Tier Sprint 10 với True Basis MTM, lượng hóa lot-size và phí maker-taker.

    Tham số:
        multi_aligned_data: Dữ liệu OHLCV và funding đã đồng bộ đa sàn.
        btc_benchmark_returns: Chuỗi tỷ suất lợi nhuận BTC benchmark.
        initial_capital: Vốn ban đầu (mặc định 3000.0 USDT).
        n_periods: Số giai đoạn chia chuỗi (mặc định 7).
        period_days: Số ngày mỗi giai đoạn (mặc định 180).

    Trả về:
        Dict lồng chứa kết quả 4 Tier: tier1_single_binance, tier2_best_venue,
        tier3_hybrid_perp_perp, tier4_production_barbell.
    """
    # Tier 1: Single-Exchange Binance Static Carry
    cfg_t1 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="single_binance",
        execution="maker_taker_hedge",
        lot_size_quantization=True,
    )
    res_t1 = run_cross_exchange_carry(multi_aligned_data, cfg_t1)

    # Tier 2: Multi-Exchange Best-Venue Carry
    cfg_t2 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="best_venue_carry",
        execution="maker_taker_hedge",
        lot_size_quantization=True,
    )
    res_t2 = run_cross_exchange_carry(multi_aligned_data, cfg_t2)

    # Tier 3: Hybrid Cross-Exchange Carry
    cfg_t3 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="hybrid",
        execution="maker_taker_hedge",
        lot_size_quantization=True,
    )
    res_t3 = run_cross_exchange_carry(multi_aligned_data, cfg_t3)

    # Tier 4: Production Barbell (SNR-Gated Vol-Targeted Overlay)
    cfg_hybrid_for_t4 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="hybrid",
        execution="maker_taker_hedge",
        lot_size_quantization=True,
    )
    barbell_cfg = BarbellConfig(
        initial_capital=initial_capital,
        carry_config=cfg_hybrid_for_t4,
        enable_snr_gate=True,
        overlay_min_snr_dispersion=1.35,
        overlay_exit_snr_dispersion=0.95,
        enable_vol_targeting=True,
        overlay_target_vol_ann=0.065,
        overlay_leverage_min=0.10,
        overlay_leverage_max=0.45,
        enable_drawdown_cushion=True,
        overlay_dd_cut_threshold=0.04,
        overlay_dd_off_threshold=0.06,
    )
    res_t4 = run_barbell_portfolio(multi_aligned_data, barbell_cfg)

    tiers_meta = [
        ("tier1_single_binance", "Tier 1 — Binance Static Carry (Long-Horizon + True Basis MTM)", res_t1),
        ("tier2_best_venue", "Tier 2 — Multi-Exchange Best-Venue Carry (True Basis MTM)", res_t2),
        ("tier3_hybrid_perp_perp", "Tier 3 — Hybrid Cross-Exchange Carry (0.5bps entry + True Basis MTM)", res_t3),
        ("tier4_production_barbell", "Tier 4 — Production Barbell (SNR-Gated Vol-Targeted Overlay)", res_t4),
    ]

    out: dict[str, dict[str, Any]] = {}
    for tier_key, tier_name, res in tiers_meta:
        row = build_tier_row(tier_name, res, btc_benchmark_returns, initial_capital)
        returns_s = res.get("returns") if isinstance(res, dict) else None
        periods_df = build_period_table_for_tier(
            returns=returns_s,
            benchmark_returns=btc_benchmark_returns,
            n_periods=n_periods,
            period_days=period_days,
            initial_capital=initial_capital,
        )

        if tier_key == "tier4_production_barbell":
            metrics_dict = res.get("metrics", {}) if isinstance(res.get("metrics"), dict) else {}
            row["overlay_active_ratio"] = float(
                res.get("overlay_active_ratio", metrics_dict.get("overlay_active_ratio", 0.0))
            )
            carry_res = res.get("carry_result") if isinstance(res.get("carry_result"), dict) else {}
            row["basis_mtm_std"] = float(
                carry_res.get("basis_mtm_std", res.get("basis_mtm_std", 0.0))
            )
            row["overlay_mean_leverage"] = float(metrics_dict.get("overlay_mean_leverage", 0.0))
            row["snr_dispersion_mean"] = float(metrics_dict.get("snr_dispersion_mean", 0.0))
        else:
            row["overlay_active_ratio"] = 0.0
            row["basis_mtm_std"] = float(res.get("basis_mtm_std", 0.0)) if isinstance(res, dict) else 0.0
            row["overlay_mean_leverage"] = 0.0
            row["snr_dispersion_mean"] = 0.0

        pos_count = int(sum(1 for p in periods_df["positive"] if p)) if not periods_df.empty else 0
        total_p = int(len(periods_df))
        row["pos_periods"] = pos_count
        row["total_periods"] = total_p
        row["positive_periods"] = f"{pos_count}/{total_p}"

        out[tier_key] = {
            "result": res,
            "row": row,
            "periods": periods_df,
            "tier_name": tier_name,
            "equity_curve": res.get("equity_curve") if isinstance(res, dict) else None,
            "returns": returns_s,
        }

    return out


def compute_period_contributions(
    result: dict[str, Any],
    start: Any = None,
    end: Any = None,
) -> dict[str, float]:
    """Tổng hợp đóng góp PnL (funding, basis, overlay, fees, net) trong kỳ [start, end).

    Tham số:
        result: Dict kết quả từ chiến lược carry hoặc barbell.
        start: Mốc thời gian bắt đầu (inclusive).
        end: Mốc thời gian kết thúc (exclusive).

    Trả về:
        Dict gồm các trường float: funding_pnl, basis_pnl, overlay_pnl, fees, net_pnl.
    """
    default_out = {
        "funding_pnl": 0.0,
        "basis_pnl": 0.0,
        "overlay_pnl": 0.0,
        "fees": 0.0,
        "net_pnl": 0.0,
    }
    if not isinstance(result, dict):
        return default_out

    attr = result.get("attribution_series")
    if attr is None or not isinstance(attr, pd.DataFrame) or attr.empty:
        return default_out

    sub_df = attr
    if isinstance(attr.index, pd.DatetimeIndex) and len(attr) > 0:
        idx_tz = attr.index.tz
        s_ts = None
        e_ts = None
        if start is not None:
            try:
                s_ts = pd.Timestamp(start)
                if idx_tz is not None and s_ts.tzinfo is None:
                    s_ts = s_ts.tz_localize(idx_tz)
                elif idx_tz is None and s_ts.tzinfo is not None:
                    s_ts = s_ts.tz_localize(None)
                elif idx_tz is not None and s_ts.tzinfo is not None:
                    s_ts = s_ts.tz_convert(idx_tz)
            except Exception:
                s_ts = None

        if end is not None:
            try:
                e_ts = pd.Timestamp(end)
                if idx_tz is not None and e_ts.tzinfo is None:
                    e_ts = e_ts.tz_localize(idx_tz)
                elif idx_tz is None and e_ts.tzinfo is not None:
                    e_ts = e_ts.tz_localize(None)
                elif idx_tz is not None and e_ts.tzinfo is not None:
                    e_ts = e_ts.tz_convert(idx_tz)
            except Exception:
                e_ts = None

        if s_ts is not None and e_ts is not None:
            sub_df = attr.loc[(attr.index >= s_ts) & (attr.index < e_ts)]
        elif s_ts is not None:
            sub_df = attr.loc[attr.index >= s_ts]
        elif e_ts is not None:
            sub_df = attr.loc[attr.index < e_ts]

    if sub_df.empty:
        return default_out

    return {
        "funding_pnl": float(sub_df["funding_pnl"].sum()) if "funding_pnl" in sub_df.columns else 0.0,
        "basis_pnl": float(sub_df["basis_pnl"].sum()) if "basis_pnl" in sub_df.columns else 0.0,
        "overlay_pnl": float(sub_df["overlay_pnl"].sum()) if "overlay_pnl" in sub_df.columns else 0.0,
        "fees": float(sub_df["fees"].sum()) if "fees" in sub_df.columns else 0.0,
        "net_pnl": float(sub_df["net_pnl"].sum()) if "net_pnl" in sub_df.columns else 0.0,
    }


def check_executor_parity() -> dict[str, Any]:
    """Kiểm tra các thành phần và cấu trúc thực thi theo tiêu chí GO-5.

    Đọc trực tiếp các file thực thi sản xuất và xác thực máy trạng thái.

    Trả về:
        Dict gồm: passed (bool), checks (dict[str, bool]), missing (list[str]).
    """
    files = {
        "carry_live_executor": Path("src/carry_live_executor.py"),
        "run_live": Path("run_live.py"),
        "state_store": Path("src/state_store.py"),
    }
    texts: dict[str, str] = {}
    for k, p in files.items():
        if p.exists():
            texts[k] = p.read_text(encoding="utf-8")
        else:
            texts[k] = ""

    carry_src = texts["carry_live_executor"]
    run_src = texts["run_live"]
    store_src = texts["state_store"]

    checks = {
        "quantize_leg": "quantize_leg" in carry_src,
        "cancel_maker_leg1": "cancel_maker_leg1" in carry_src,
        "build_barbell_allocations": "build_barbell_allocations" in carry_src,
        "carry_sleeve": ("carry_sleeve" in carry_src) and ("carry_sleeve" in store_src),
        "momentum_overlay": ("momentum_overlay" in carry_src) and ("momentum_overlay" in store_src),
        "state_machine": bool(detect_executor_state_machine()),
    }

    missing = [name for name, ok in checks.items() if not ok]
    passed = bool(len(missing) == 0)

    return {
        "passed": passed,
        "checks": checks,
        "missing": missing,
    }


def evaluate_sprint10_go_no_go(
    tier_payload: dict[str, Any],
    is_barbell: bool = False,
) -> dict[str, Any]:
    """Phán quyết GO / NO-GO Sprint 10 theo tiêu chí docs/go_no_go_sprint10.md.

    Tham số:
        tier_payload: Dict dữ liệu của Tier (chứa row, result, periods).
        is_barbell: True nếu kiểm định Tier 4 Barbell Portfolio.

    Trả về:
        Dict gồm: verdict ('GO' hoặc 'NO-GO'), failed (list tên tiêu chí fail),
        conditions (dict chi tiết), holdout_contributions (dict).
    """
    if not isinstance(tier_payload, dict):
        tier_payload = {}

    row = tier_payload.get("row")
    if not isinstance(row, dict):
        row = tier_payload

    res = tier_payload.get("result")
    if not isinstance(res, dict):
        res = tier_payload

    periods = tier_payload.get("periods")
    if isinstance(periods, pd.DataFrame):
        records = periods.to_dict(orient="records") if not periods.empty else []
    elif isinstance(periods, (list, tuple)):
        records = list(periods)
    else:
        records = []

    total_periods = len(records)
    if total_periods > 0:
        rets = [float(r.get("net_return_pct", 0.0)) for r in records]
        positive_periods = sum(1 for r in rets if r > 0.0)
        worst_period_pct = float(min(rets))
        holdout = records[-1]
        holdout_ann = float(holdout.get("annualized_return_pct", 0.0))
        holdout_start = holdout.get("start")
        holdout_end = holdout.get("end")
    else:
        positive_periods = 0
        worst_period_pct = 0.0
        holdout_ann = 0.0
        holdout_start = None
        holdout_end = None

    holdout_contrib = compute_period_contributions(res, holdout_start, holdout_end)
    carry_holdout_pnl = float(
        holdout_contrib.get("funding_pnl", 0.0) + holdout_contrib.get("basis_pnl", 0.0)
    )
    overlay_holdout_pnl = float(holdout_contrib.get("overlay_pnl", 0.0))

    # GO-1: Annualized Net Return >= 10.0%
    ann_ret = float(row.get("annualized_return_pct", 0.0))
    cond1_passed = bool(ann_ret >= 10.0)

    # GO-2: Multi-Period Consistency
    barbell_holdout_ok = True
    if is_barbell:
        barbell_holdout_ok = (carry_holdout_pnl > 0.0) and (overlay_holdout_pnl > 0.0)

    cond2_passed = bool(
        total_periods >= 6
        and positive_periods >= 6
        and holdout_ann >= 8.0
        and worst_period_pct >= -3.0
        and barbell_holdout_ok
    )

    # GO-3: Risk & Sharpe Control
    sharpe = float(row.get("sharpe", 0.0))
    maxdd_pct = float(abs(row.get("maxdd_pct", 0.0)))
    beta_btc = float(row.get("beta_btc", 0.0))
    margin_calls = int(row.get("margin_calls", res.get("metrics", {}).get("Margin Calls", 0)))
    cond3_passed = bool(
        sharpe >= 1.80
        and maxdd_pct <= 8.0
        and abs(beta_btc) <= 0.15
        and margin_calls == 0
    )

    # GO-4: True Regime Gating & True MTM Verification
    basis_mtm_std = float(
        row.get(
            "basis_mtm_std",
            res.get("basis_mtm_std", res.get("carry_result", {}).get("basis_mtm_std", 0.0)),
        )
    )
    overlay_active_ratio = float(
        row.get(
            "overlay_active_ratio",
            res.get("overlay_active_ratio", res.get("metrics", {}).get("overlay_active_ratio", 0.0)),
        )
    )

    regime_ok = True
    if is_barbell:
        regime_ok = (0.25 <= overlay_active_ratio <= 0.75)

    cond4_passed = bool((basis_mtm_std > 1e-12) and regime_ok)

    # GO-5: Production Executor Parity
    parity_res = check_executor_parity()
    cond5_passed = bool(parity_res.get("passed", False))

    conditions = {
        "cond1_return": {
            "passed": cond1_passed,
            "detail": f"Annualized return: {ann_ret:.2f}% (>= 10.0%)",
        },
        "cond2_consistency": {
            "passed": cond2_passed,
            "detail": (
                f"Positive periods: {positive_periods}/{total_periods} (>= 6), "
                f"Holdout ann: {holdout_ann:.2f}% (>= 8.0%), "
                f"Worst period: {worst_period_pct:.2f}% (>= -3.0%)"
                + (
                    f", Holdout carry: {carry_holdout_pnl:.2f}, overlay: {overlay_holdout_pnl:.2f}"
                    if is_barbell
                    else ""
                )
            ),
        },
        "cond3_risk": {
            "passed": cond3_passed,
            "detail": (
                f"Sharpe: {sharpe:.2f} (>= 1.80), MaxDD: {maxdd_pct:.2f}% (<= 8.0%), "
                f"Beta BTC: {beta_btc:.4f} (|beta| <= 0.15), Margin calls: {margin_calls} (== 0)"
            ),
        },
        "cond4_regime_and_mtm": {
            "passed": cond4_passed,
            "detail": (
                f"Basis MTM std: {basis_mtm_std:.6f} (> 1e-12)"
                + (
                    f", Overlay active ratio: {overlay_active_ratio:.2%} ([25%, 75%])"
                    if is_barbell
                    else ""
                )
            ),
        },
        "cond5_executor_parity": {
            "passed": cond5_passed,
            "detail": f"Executor parity passed: {cond5_passed} (missing: {parity_res.get('missing', [])})",
        },
    }

    failed = [k for k, v in conditions.items() if not v["passed"]]
    verdict = "GO" if len(failed) == 0 else "NO-GO"

    return {
        "verdict": verdict,
        "failed": failed,
        "conditions": conditions,
        "holdout_contributions": holdout_contrib,
    }


def build_sprint10_summary_rows(
    tiers: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tạo bảng dòng phẳng phục vụ xuất báo cáo CSV và so sánh 4 Tier Sprint 10.

    Tham số:
        tiers: Dict kết quả 4 Tier từ run_sprint10_tiers.

    Trả về:
        List các dict chứa đúng 21 trường số liệu chuẩn.
    """
    if not isinstance(tiers, dict):
        return []

    rows: list[dict[str, Any]] = []

    for tier_key, tier_payload in tiers.items():
        if not isinstance(tier_payload, dict):
            continue

        tier_name = str(tier_payload.get("tier_name", tier_key))
        res = tier_payload.get("result", {})
        if not isinstance(res, dict):
            res = {}

        row = tier_payload.get("row", {})
        if not isinstance(row, dict):
            row = {}

        periods = tier_payload.get("periods")
        if isinstance(periods, pd.DataFrame):
            p_recs = periods.to_dict(orient="records") if not periods.empty else []
        elif isinstance(periods, (list, tuple)):
            p_recs = list(periods)
        else:
            p_recs = []

        total_periods = len(p_recs)
        if total_periods > 0:
            rets = [float(p.get("net_return_pct", 0.0)) for p in p_recs]
            pos_periods = sum(1 for r in rets if r > 0.0)
            worst_pct = float(min(rets))
            holdout = p_recs[-1]
            holdout_ann = float(holdout.get("annualized_return_pct", 0.0))
        else:
            pos_periods = 0
            worst_pct = 0.0
            holdout_ann = 0.0

        is_barbell = bool(tier_key == "tier4_production_barbell" or "barbell" in tier_key.lower())
        eval_res = evaluate_sprint10_go_no_go(tier_payload, is_barbell=is_barbell)

        metrics_dict = res.get("metrics", {}) if isinstance(res.get("metrics"), dict) else {}
        carry_res = res.get("carry_result", {}) if isinstance(res.get("carry_result"), dict) else {}

        final_eq = float(row.get("final_equity", res.get("final_equity", 0.0)))
        ann_ret = float(row.get("annualized_return_pct", 0.0))
        sharpe = float(row.get("sharpe", 0.0))
        maxdd = float(abs(row.get("maxdd_pct", 0.0)))
        beta_btc = float(row.get("beta_btc", 0.0))
        net_pnl = float(row.get("net_pnl", 0.0))
        fees = float(row.get("fees", res.get("fees", 0.0)))
        funding_pnl = float(row.get("funding_pnl", res.get("funding_pnl", 0.0)))
        basis_pnl = float(row.get("basis_pnl", res.get("basis_pnl", 0.0)))
        basis_mtm_std = float(
            row.get("basis_mtm_std", res.get("basis_mtm_std", carry_res.get("basis_mtm_std", 0.0)))
        )
        overlay_active_ratio = float(
            row.get("overlay_active_ratio", res.get("overlay_active_ratio", metrics_dict.get("overlay_active_ratio", 0.0)))
        )
        overlay_mean_leverage = float(
            row.get("overlay_mean_leverage", metrics_dict.get("overlay_mean_leverage", 0.0))
        )
        snr_dispersion_mean = float(
            row.get("snr_dispersion_mean", metrics_dict.get("snr_dispersion_mean", 0.0))
        )

        verdict = str(eval_res.get("verdict", "NO-GO"))
        failed_conditions = ", ".join(eval_res.get("failed", []))

        out_row = {
            "tier": tier_key,
            "tier_name": tier_name,
            "final_equity": final_eq,
            "annualized_return_pct": ann_ret,
            "sharpe": sharpe,
            "maxdd_pct": maxdd,
            "beta_btc": beta_btc,
            "net_pnl": net_pnl,
            "positive_periods": pos_periods,
            "total_periods": total_periods,
            "holdout_annualized_pct": holdout_ann,
            "worst_period_pct": worst_pct,
            "fees": fees,
            "funding_pnl": funding_pnl,
            "basis_pnl": basis_pnl,
            "basis_mtm_std": basis_mtm_std,
            "overlay_active_ratio": overlay_active_ratio,
            "overlay_mean_leverage": overlay_mean_leverage,
            "snr_dispersion_mean": snr_dispersion_mean,
            "verdict": verdict,
            "failed_conditions": failed_conditions,
        }
        rows.append(out_row)

    return rows
