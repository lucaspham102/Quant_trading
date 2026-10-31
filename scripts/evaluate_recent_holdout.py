"""Multi-Cycle Holdout & Pruning Backtest Evaluation.

Evaluates 1k strategy profiles across:
1. Window 3M:  2026-08-01 -> 2026-10-30 (~90 days)
2. Window 6M:  2026-05-01 -> 2026-10-30 (~182 days)
3. Window 12M: 2025-11-01 -> 2026-10-30 (~364 days)
4. Window 24M: 2024-11-01 -> 2026-10-30 (~729 days, Full Multi-Year)

Provides:
- Direct Before vs After Pruning comparison
- Full PnL Attribution breakdown
- Monthly Return Distribution (Win rate, Best/Worst month, Profit Factor)
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_sprint11 import load_multi_aligned_data
from src.cross_exchange_carry import LIQUID_8_SYMBOLS
from src.strategy_profiles import PROFILE_REGISTRY, StrategyProfileConfig
from scripts.run_1k_profile_tournament_backtest import (
    build_market_matrices,
    evaluate_window,
    simulate_profile,
)


def build_evaluation_profiles() -> dict[str, StrategyProfileConfig]:
    """Build profiles including both Pruned and Pre-Pruning baseline for rigorous comparison."""
    # Current pruned profiles from registry
    omni_pruned = PROFILE_REGISTRY["regime_omni_1k"]
    flex_pruned = PROFILE_REGISTRY["flexible_regime_1k"]
    base = PROFILE_REGISTRY["baseline_1k"]
    carry_eta = PROFILE_REGISTRY["carry_max_eta_1k"]
    cons_trend = PROFILE_REGISTRY["hybrid_trend_conservative_1k"]

    # Pre-pruning configurations for before/after comparison
    omni_pre_pruning = dataclasses.replace(
        omni_pruned,
        name="regime_omni_pre_pruning",
        overlay_target_vol_ann=0.11,
        overlay_gross_pct=0.25,
        enable_reversal_sleeve=True,
        reversal_gross_pct=0.35,
    )
    flex_pre_pruning = dataclasses.replace(
        flex_pruned,
        name="flexible_regime_pre_pruning",
        enable_reversal_sleeve=True,
        reversal_gross_pct=0.25,
    )

    return {
        "regime_omni_1k (Pruned)": omni_pruned,
        "regime_omni_1k (Pre-Pruning)": omni_pre_pruning,
        "flexible_regime_1k (Pruned)": flex_pruned,
        "flexible_regime_1k (Pre-Pruning)": flex_pre_pruning,
        "baseline_1k": base,
        "carry_max_eta_1k": carry_eta,
        "hybrid_trend_conservative_1k": cons_trend,
    }


def run_multi_cycle_evaluation(
    raw_dir: str = "data/raw",
    warmup_hours: int = 1500,
    out_json: str = "logs/multi_cycle_pruning_evaluation.json",
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    profiles = build_evaluation_profiles()

    print("=" * 110)
    print("MULTI-CYCLE STRUCTURAL PRUNING & HOLDOUT EVALUATION")
    print(f"Data root: {raw_path}")
    print(f"Profiles ({len(profiles)}): {list(profiles.keys())}")
    print(f"Warmup Saturation: {warmup_hours} hours")
    print("=" * 110)

    # 1. Load aligned dataset
    print("\n[1/4] Loading aligned 1h market history for Liquid 8 symbols...")
    aligned = load_multi_aligned_data(raw_path, list(LIQUID_8_SYMBOLS), "2023-05-07", "2026-10-30")
    market = build_market_matrices(aligned)
    market["spot_cost_bps"] = 7.5
    market["perp_cost_bps"] = 5.0
    market["carry_cost_bps"] = 12.5

    index: pd.DatetimeIndex = market["index"]
    n_bars = len(index)
    print(f"[DATA] Aligned index: {index[0]} to {index[-1]} (Total {n_bars} bars)")

    # 2. Define evaluation windows
    windows = {
        "3M": {
            "label": "Recent 3 Months (90 Days)",
            "start_dt": pd.Timestamp("2026-08-01 00:00:00", tz="UTC"),
        },
        "6M": {
            "label": "Recent 6 Months (182 Days)",
            "start_dt": pd.Timestamp("2026-05-01 00:00:00", tz="UTC"),
        },
        "12M": {
            "label": "Recent 12 Months (1 Year)",
            "start_dt": pd.Timestamp("2025-11-01 00:00:00", tz="UTC"),
        },
        "24M": {
            "label": "Recent 24 Months (Full Multi-Year)",
            "start_dt": pd.Timestamp("2024-11-01 00:00:00", tz="UTC"),
        },
    }

    window_configs: dict[str, Any] = {}
    for w_key, w_cfg in windows.items():
        trade_idx = int(index.searchsorted(w_cfg["start_dt"], side="left"))
        warmup_idx = max(0, trade_idx - warmup_hours)
        window_configs[w_key] = {
            "label": w_cfg["label"],
            "start_dt": str(index[trade_idx]),
            "end_dt": str(index[-1]),
            "trade_start_idx": trade_idx,
            "warmup_start_idx": warmup_idx,
            "end_idx": n_bars,
            "n_traded_bars": n_bars - trade_idx,
            "n_warmup_bars": trade_idx - warmup_idx,
        }

    # 3. Simulate all profiles across windows
    print("\n[2/4] Running multi-window evaluations...")
    results: dict[str, dict[str, Any]] = {}
    simulations_raw: dict[str, dict[str, Any]] = {}

    for p_name, p_cfg in profiles.items():
        print(f"  -> Simulating: {p_name}...")
        results[p_name] = {}
        simulations_raw[p_name] = {}
        for w_key, w_info in window_configs.items():
            ev = evaluate_window(
                profile=p_cfg,
                market=market,
                start_idx=w_info["warmup_start_idx"],
                end_idx=w_info["end_idx"],
                trade_start_idx=w_info["trade_start_idx"],
            )
            cap = p_cfg.capital_usd
            ev["initial_capital_usd"] = cap
            ev["pure_trading_final_equity_usd"] = cap + ev["pure_trading_net_pnl_usd"]
            ev["yield_stacked_final_equity_usd"] = cap + ev["yield_stacked_net_pnl_usd"]
            results[p_name][w_key] = ev

            # Store raw sim for monthly breakdown on the 24M window
            if w_key == "24M":
                raw_sim = simulate_profile(
                    p_cfg, market,
                    start_idx=w_info["warmup_start_idx"],
                    end_idx=w_info["end_idx"],
                    trade_start_idx=w_info["trade_start_idx"],
                )
                simulations_raw[p_name][w_key] = raw_sim

    # 4. Compute Monthly Return Distribution for 24M window
    print("\n[3/4] Computing Monthly Return Distribution on 24M Multi-Year window...")
    monthly_distributions: dict[str, Any] = {}
    target_monthly_profiles = [
        "regime_omni_1k (Pruned)",
        "flexible_regime_1k (Pruned)",
        "baseline_1k",
    ]

    for p_name in target_monthly_profiles:
        raw_sim = simulations_raw[p_name]["24M"]
        w_info = window_configs["24M"]
        off = w_info["trade_start_idx"] - w_info["warmup_start_idx"]
        pure_eq_series = raw_sim["equity_pure"].iloc[off:]

        # Resample to monthly returns
        # Group by Year-Month
        df_eq = pd.DataFrame({"equity": pure_eq_series})
        df_eq["year_month"] = df_eq.index.strftime("%Y-%m")

        monthly_records: list[dict[str, Any]] = []
        for ym, group in df_eq.groupby("year_month"):
            m_start_eq = group["equity"].iloc[0]
            m_end_eq = group["equity"].iloc[-1]
            m_ret_pct = ((m_end_eq - m_start_eq) / m_start_eq) * 100.0
            m_pnl_usd = m_end_eq - m_start_eq
            monthly_records.append({
                "month": ym,
                "start_equity": float(m_start_eq),
                "end_equity": float(m_end_eq),
                "return_pct": float(m_ret_pct),
                "pnl_usd": float(m_pnl_usd),
                "is_positive": bool(m_ret_pct >= 0.0),
            })

        pos_months = [m for m in monthly_records if m["return_pct"] >= 0.0]
        neg_months = [m for m in monthly_records if m["return_pct"] < 0.0]
        n_total = len(monthly_records)
        n_pos = len(pos_months)
        n_neg = len(neg_months)
        win_rate = (n_pos / n_total * 100.0) if n_total > 0 else 0.0

        pos_sum = sum(m["pnl_usd"] for m in pos_months)
        neg_sum = abs(sum(m["pnl_usd"] for m in neg_months))
        profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else 999.0

        best_m = max(monthly_records, key=lambda x: x["return_pct"]) if monthly_records else {}
        worst_m = min(monthly_records, key=lambda x: x["return_pct"]) if monthly_records else {}
        avg_m_ret = float(np.mean([m["return_pct"] for m in monthly_records])) if monthly_records else 0.0

        monthly_distributions[p_name] = {
            "records": monthly_records,
            "summary": {
                "total_months": n_total,
                "positive_months": n_pos,
                "negative_months": n_neg,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "best_month": best_m.get("month"),
                "best_month_return_pct": best_m.get("return_pct"),
                "worst_month": worst_m.get("month"),
                "worst_month_return_pct": worst_m.get("return_pct"),
                "avg_monthly_return_pct": avg_m_ret,
                "total_pnl_usd": sum(m["pnl_usd"] for m in monthly_records),
            },
        }

    # 5. Save json output
    print("\n[4/4] Serializing payload to JSON...")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "windows": window_configs,
        "results": results,
        "monthly_distributions": monthly_distributions,
    }

    def _clean_obj(o: Any) -> Any:
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, (np.floating, float)):
            return float(o) if math.isfinite(o) else 0.0
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, dict):
            return {k: _clean_obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean_obj(v) for v in o]
        return o

    out_file = Path(out_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_clean_obj(payload), f, indent=2)
    print(f"Results saved to {out_file}")

    return payload


def display_ascii_reports(payload: dict[str, Any]) -> None:
    results = payload["results"]
    windows = payload["windows"]
    monthly = payload.get("monthly_distributions", {})

    print("\n" + "=" * 145)
    print("PHAN I: SO SANH TRUOC VA SAU KHI TINH GON (BEFORE VS AFTER STRUCTURAL PRUNING)")
    print("=" * 145)

    for w_key in ["3M", "6M", "12M", "24M"]:
        w_info = windows[w_key]
        print(f"\n>>> CUA SO: {w_key} - {w_info['label']} ({w_info['start_dt'][:10]} -> {w_info['end_dt'][:10]})")

        headers1 = [
            "Profile", "Pure Eq ($)", "Net Ret (%)", "Yield Eq ($)", "Tot Ret (%)",
            "Sharpe", "Sortino", "Calmar", "MaxDD (%)", "Passive Yield"
        ]
        col_w1 = [32, 12, 12, 13, 12, 8, 8, 8, 11, 15]
        sep1 = "+-" + "-+-".join("-" * w for w in col_w1) + "-+"

        def format_row(vals: list[str], widths: list[int]) -> str:
            return "| " + " | ".join(v.ljust(widths[i]) if i == 0 else v.rjust(widths[i]) for i, v in enumerate(vals)) + " |"

        print(sep1)
        print(format_row(headers1, col_w1))
        print(sep1)

        for p_name, p_data in results.items():
            r = p_data[w_key]
            row_vals = [
                p_name,
                f"{r['pure_trading_final_equity_usd']:.2f}",
                f"{r['pure_trading_net_return_pct']:+.2f}%",
                f"{r['yield_stacked_final_equity_usd']:.2f}",
                f"{r['yield_stacked_net_return_pct']:+.2f}%",
                f"{r['pure_trading_sharpe']:.2f}",
                f"{r['pure_trading_sortino']:.2f}",
                f"{r['pure_trading_calmar']:.2f}",
                f"{r['pure_trading_max_dd_pct']:.2f}%",
                f"+${r['passive_yield_usd']:.2f}",
            ]
            print(format_row(row_vals, col_w1))
        print(sep1)

        # Attribution Table
        headers2 = [
            "Profile", "Funding ($)", "Basis ($)", "Overlay ($)", "Trend ($)",
            "L-Trend ($)", "S-Trend ($)", "Rev ($)", "Fees ($)", "Net PnL ($)"
        ]
        col_w2 = [32, 12, 11, 12, 11, 12, 12, 10, 10, 12]
        sep2 = "+-" + "-+-".join("-" * w for w in col_w2) + "-+"

        print(sep2)
        print(format_row(headers2, col_w2))
        print(sep2)

        for p_name, p_data in results.items():
            r = p_data[w_key]
            row_vals = [
                p_name,
                f"{r['funding_pnl_usd']:+.2f}",
                f"{r['basis_pnl_usd']:+.2f}",
                f"{r['overlay_pnl_usd']:+.2f}",
                f"{r['trend_pnl_usd']:+.2f}",
                f"{r.get('long_trend_pnl_usd', 0.0):+.2f}",
                f"{r.get('short_trend_pnl_usd', 0.0):+.2f}",
                f"{r['reversal_pnl_usd']:+.2f}",
                f"-${r['fees_usd']:.2f}",
                f"{r['pure_trading_net_pnl_usd']:+.2f}",
            ]
            print(format_row(row_vals, col_w2))
        print(sep2)

    # Monthly Return Breakdown
    print("\n" + "=" * 145)
    print("PHAN II: PHAN PHOI LOI NHUAN THEO TUNG THANG (MONTHLY RETURN DISTRIBUTION - 24M MULTI-YEAR)")
    print("=" * 145)

    for p_name, p_m in monthly.items():
        summary = p_m["summary"]
        print(f"\n>>> Profile: {p_name}")
        print(
            f"    Thang Duong / Thang Am: {summary['positive_months']}/{summary['total_months']} "
            f"(Win Rate: {summary['win_rate_pct']:.1f}%) | Profit Factor: {summary['profit_factor']:.2f} | "
            f"Avg Monthly Ret: {summary['avg_monthly_return_pct']:+.2f}%\n"
            f"    Thang tot nhat: {summary['best_month']} ({summary['best_month_return_pct']:+.2f}%) | "
            f"Thang te nhat: {summary['worst_month']} ({summary['worst_month_return_pct']:+.2f}%) | "
            f"Tong Net PnL: ${summary['total_pnl_usd']:+.2f}"
        )

        headers_m = ["Thang (Year-Month)", "Start Eq ($)", "End Eq ($)", "Return (%)", "PnL ($)", "Trang thai"]
        col_wm = [20, 16, 16, 14, 14, 14]
        sepm = "+-" + "-+-".join("-" * w for w in col_wm) + "-+"

        print(sepm)
        print(format_row(headers_m, col_wm))
        print(sepm)

        for rec in p_m["records"]:
            status = "WIN (+)" if rec["is_positive"] else "LOSS (-)"
            row_m = [
                rec["month"],
                f"{rec['start_equity']:.2f}",
                f"{rec['end_equity']:.2f}",
                f"{rec['return_pct']:+.2f}%",
                f"{rec['pnl_usd']:+.2f}",
                status,
            ]
            print(format_row(row_m, col_wm))
        print(sepm)


if __name__ == "__main__":
    payload = run_multi_cycle_evaluation()
    display_ascii_reports(payload)
