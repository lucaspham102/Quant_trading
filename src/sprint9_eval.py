"""Module đánh giá và kiểm định chất lượng Sprint 9.

Cung cấp:
- compute_ols_beta: Tính OLS beta giữa returns 1h của danh mục so với BTC.
- compute_go_inputs: Tính toán các chỉ số đầu vào cho bộ tiêu chí GO / NO-GO.
- build_tier_row: Dựng dòng tổng hợp chỉ số cho từng Tier.
- evaluate_sprint9_go_no_go: Kiểm định 4 tiêu chí GO và các điều kiện NO-GO theo docs/go_no_go_sprint9.md.
- run_four_tiers_comparison: So sánh trực tiếp 4 cấp độ tiến hóa trên chuỗi 3.45 năm.
- build_period_table_for_tier: Bảng chi tiết 7 giai đoạn nối tiếp (Stitched Equity Curve).
- detect_executor_state_machine: Kiểm tra CarryLiveExecutor có đúng máy trạng thái 2 chân.
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.cross_exchange_carry import CrossExchangeCarryConfig, run_cross_exchange_carry
from src.sprint7_eval import build_benchmark_returns, equity_from_returns, segment_metrics
from src.sprint8_eval import period_metrics, split_into_periods, summarize_periods, to_markdown_table

__all__ = [
    "compute_ols_beta",
    "compute_go_inputs",
    "build_tier_row",
    "evaluate_sprint9_go_no_go",
    "run_four_tiers_comparison",
    "build_period_table_for_tier",
    "detect_executor_state_machine",
    "split_into_periods",
    "period_metrics",
    "summarize_periods",
    "to_markdown_table",
    "build_benchmark_returns",
    "equity_from_returns",
]


def compute_ols_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Tính OLS beta của chuỗi lợi nhuận danh mục so với chuỗi lợi nhuận benchmark.

    Quy tắc:
    - Align 2 series trên cùng DatetimeIndex, loại bỏ NaN.
    - OLS slope = cov(ret, bench) / var(bench) (close-to-close, không lag).
    - Trả về 0.0 nếu var(bench) xấp xỉ 0 (<= 1e-12) hoặc số điểm dữ liệu chung < 10.

    Tham số:
        returns: Chuỗi tỷ suất lợi nhuận 1h của danh mục/equity curve.
        benchmark_returns: Chuỗi tỷ suất lợi nhuận 1h của benchmark (BTC Buy-and-Hold).

    Trả về:
        Hệ số beta (float).
    """
    if returns is None or benchmark_returns is None:
        return 0.0

    s_ret = pd.Series(returns).dropna().astype("float64")
    s_bench = pd.Series(benchmark_returns).dropna().astype("float64")

    if len(s_ret) == 0 or len(s_bench) == 0:
        return 0.0

    aligned = pd.concat([s_ret.rename("ret"), s_bench.rename("bench")], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return 0.0

    ret_arr = aligned["ret"].to_numpy(dtype=np.float64)
    bench_arr = aligned["bench"].to_numpy(dtype=np.float64)

    var_b = float(np.var(bench_arr, ddof=1))
    if var_b <= 1e-12 or not math.isfinite(var_b):
        return 0.0

    cov_mat = np.cov(ret_arr, bench_arr, ddof=1)
    cov_val = float(cov_mat[0, 1])

    beta = cov_val / var_b
    if not math.isfinite(beta):
        return 0.0
    return float(beta)


def compute_go_inputs(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float = 3000.0,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """Tính toán toàn bộ các chỉ số đầu vào phục vụ kiểm định GO / NO-GO Sprint 9.

    Công thức:
    - Annualized Return = ((final_equity / initial_capital) ** (365.25 / n_days) - 1) * 100
      với n_days = (end - start).total_seconds() / 86400.
    - Sharpe = (mean(r) / std(r)) * sqrt(periods_per_year); std=0 -> 0.0.
    - MaxDD MTM = abs(min(equity / cummax(equity) - 1) * 100) (quy ước số DƯƠNG).
    - Net PnL = final_equity - initial_capital (USDT).

    Tham số:
        returns: Chuỗi tỷ suất lợi nhuận 1h của danh mục.
        equity_curve: Chuỗi đường cong vốn MTM theo giờ (khởi đầu = initial_capital).
        benchmark_returns: Chuỗi tỷ suất lợi nhuận benchmark (BTC).
        initial_capital: Vốn ban đầu (mặc định 3000.0 USDT).
        periods_per_year: Số chu kỳ mỗi năm (8760 cho nến 1h).

    Trả về:
        Dict tối thiểu chứa: annualized_return_pct, total_return_pct, net_pnl, final_equity,
        sharpe, maxdd_pct, beta_btc, n_bars, start, end, time_in_market_pct.
    """
    clean_ret = (
        pd.Series(returns).dropna().astype("float64")
        if returns is not None
        else pd.Series([], dtype="float64")
    )
    n_bars = int(len(clean_ret))

    if equity_curve is not None and len(equity_curve) > 0:
        eq_s = pd.Series(equity_curve).dropna().astype("float64")
    elif n_bars > 0:
        eq_s = equity_from_returns(clean_ret, initial_capital=initial_capital)
    else:
        eq_s = pd.Series([initial_capital], dtype="float64")

    final_equity = float(eq_s.iloc[-1]) if len(eq_s) > 0 else float(initial_capital)
    net_pnl = float(final_equity - initial_capital)
    total_return_pct = (
        float((final_equity / initial_capital - 1.0) * 100.0) if initial_capital > 0 else 0.0
    )

    # Xác định start và end timestamp
    start_ts = None
    end_ts = None
    if isinstance(eq_s.index, pd.DatetimeIndex) and len(eq_s) > 0:
        start_ts = eq_s.index[0]
        end_ts = eq_s.index[-1]
    elif isinstance(clean_ret.index, pd.DatetimeIndex) and len(clean_ret) > 0:
        start_ts = clean_ret.index[0]
        end_ts = clean_ret.index[-1]

    if start_ts is not None and end_ts is not None:
        n_days = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / 86400.0
    else:
        n_days = n_bars / 24.0 if n_bars > 0 else 0.0

    if initial_capital > 0 and final_equity > 0 and n_days > 0:
        annualized_return_pct = float(
            ((final_equity / initial_capital) ** (365.25 / n_days) - 1.0) * 100.0
        )
    elif final_equity <= 0:
        annualized_return_pct = -100.0
    else:
        annualized_return_pct = 0.0

    # Sharpe ratio
    if n_bars > 1:
        ret_std = float(clean_ret.std(ddof=1))
        if ret_std > 1e-12:
            sharpe = float((clean_ret.mean() / ret_std) * math.sqrt(periods_per_year))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max Drawdown MTM (quy ước số DƯƠNG)
    if len(eq_s) > 0:
        peak = eq_s.cummax()
        valid = peak > 0
        if np.any(valid):
            dd = (eq_s[valid] - peak[valid]) / peak[valid]
            min_dd = float(dd.min()) * 100.0
            maxdd_pct = float(abs(min_dd))
        else:
            maxdd_pct = 0.0
    else:
        maxdd_pct = 0.0

    # OLS Beta so với BTC benchmark
    beta_btc = compute_ols_beta(clean_ret, benchmark_returns)

    return {
        "annualized_return_pct": annualized_return_pct,
        "total_return_pct": total_return_pct,
        "net_pnl": net_pnl,
        "final_equity": final_equity,
        "sharpe": sharpe,
        "maxdd_pct": maxdd_pct,
        "beta_btc": beta_btc,
        "n_bars": n_bars,
        "start": start_ts,
        "end": end_ts,
        "time_in_market_pct": 100.0,
    }


def build_tier_row(
    tier_name: str,
    result: dict[str, Any],
    benchmark_returns: pd.Series,
    initial_capital: float = 3000.0,
) -> dict[str, Any]:
    """Tạo dòng tổng hợp chỉ số cho một Tier để xuất bảng và so sánh.

    Tham số:
        tier_name: Tên định danh của Tier.
        result: Dict kết quả từ run_cross_exchange_carry hoặc run_barbell_portfolio.
        benchmark_returns: Chuỗi tỷ suất lợi nhuận benchmark (BTC).
        initial_capital: Vốn ban đầu (mặc định 3000.0 USDT).

    Trả về:
        Dict gộp các trường từ compute_go_inputs kèm fees, funding_pnl, basis_pnl,
        turnover, time_in_market_pct, num_trades và key 'tier'.
    """
    if result is None:
        result = {}

    returns = result.get("returns")
    equity_curve = result.get("equity_curve")
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}

    go_inputs = compute_go_inputs(
        returns=returns,
        equity_curve=equity_curve,
        benchmark_returns=benchmark_returns,
        initial_capital=initial_capital,
    )

    row = dict(go_inputs)
    row["tier"] = tier_name
    row["fees"] = float(result.get("fees", metrics.get("Total Fees Paid", 0.0)))
    row["funding_pnl"] = float(
        result.get("funding_pnl", metrics.get("Total Funding Collected", 0.0))
    )
    row["basis_pnl"] = float(result.get("basis_pnl", metrics.get("Total Basis PnL", 0.0)))
    row["turnover"] = float(result.get("turnover", metrics.get("Turnover", 0.0)))

    if "Time in Market %" in metrics:
        row["time_in_market_pct"] = float(metrics["Time in Market %"])

    trades_df = result.get("trades_df")
    if isinstance(trades_df, pd.DataFrame):
        row["num_trades"] = int(len(trades_df))
    else:
        row["num_trades"] = int(metrics.get("Num Entries", 0))

    return row


def evaluate_sprint9_go_no_go(
    full_metrics: dict[str, Any],
    period_rows: Sequence[dict[str, Any]] | pd.DataFrame,
    execution_mode: str = "maker_taker_hedge",
    margin_call_count: int = 0,
    executor_state_machine_verified: bool = True,
    has_lookahead_bias: bool = False,
) -> dict[str, Any]:
    """Kiểm định 4 tiêu chí GO và các điều kiện NO-GO theo docs/go_no_go_sprint9.md.

    Quy tắc kiểm định khóa cứng:
    - GO-1 (cond1_return): annualized_return_pct >= 10.0% (sau toàn bộ phí maker_taker_hedge và rebalance).
    - GO-2 (cond2_consistency):
        + pos_count >= 6 trên tổng số kỳ (>= 7 kỳ);
        + kỳ cuối (Holdout 2026-04-21 -> 2026-10-18) bắt buộc net_return_pct > 0 VÀ annualized_return_pct >= 8.0%;
        + worst_period_return_pct >= -3.0% (không có kỳ 6 tháng nào lỗ quá 3.0%).
        + Nếu period_rows rỗng -> cond2_consistency = False.
    - GO-3 (cond3_risk):
        + sharpe >= 1.80
        + maxdd_pct <= 8.0% (quy ước số dương)
        + abs(beta_btc) <= 0.15
        + margin_call_count == 0
    - GO-4 (cond4_execution_sync):
        + execution_mode == 'maker_taker_hedge'
        + executor_state_machine_verified == True
        + not has_lookahead_bias

    Tham số:
        full_metrics: Dict chỉ số toàn chuỗi (từ compute_go_inputs hoặc build_tier_row).
        period_rows: Bảng hoặc danh sách kết quả 7 kỳ (từ build_period_table_for_tier).
        execution_mode: Chế độ khớp lệnh thực tế (bắt buộc 'maker_taker_hedge').
        margin_call_count: Số lần vi phạm ký quỹ / thanh lý (bắt buộc 0).
        executor_state_machine_verified: Cờ xác thực máy trạng thái thực thi Post-Only -> Market Hedge.
        has_lookahead_bias: Cờ kiểm tra look-ahead bias (bắt buộc False).

    Trả về:
        Dict gồm: cond1_return, cond2_consistency, cond3_risk, cond4_execution_sync,
        verdict ('GO' / 'NO-GO'), failed (list[str]), pos_periods, total_periods,
        holdout_annualized_return_pct, holdout_net_return_pct, details (dict).
    """
    if full_metrics is None:
        full_metrics = {}

    ann_ret = float(
        full_metrics.get("annualized_return_pct", full_metrics.get("Annualized Return %", 0.0))
    )
    cond1_return = bool(ann_ret >= 10.0)

    # Xử lý period_rows
    if isinstance(period_rows, pd.DataFrame):
        records = period_rows.to_dict(orient="records") if not period_rows.empty else []
    elif period_rows is not None:
        records = list(period_rows)
    else:
        records = []

    total_periods = len(records)
    if total_periods == 0:
        cond2_consistency = False
        pos_count = 0
        holdout_ann_ret = 0.0
        holdout_net_ret = 0.0
        worst_period_ret = 0.0
    else:
        rets = [float(r.get("net_return_pct", 0.0)) for r in records]
        pos_count = sum(1 for r in rets if r > 0.0)
        holdout = records[-1]
        holdout_net_ret = float(holdout.get("net_return_pct", 0.0))
        holdout_ann_ret = float(holdout.get("annualized_return_pct", 0.0))
        worst_period_ret = float(min(rets))

        cond2_consistency = bool(
            total_periods >= 7
            and pos_count >= 6
            and holdout_net_ret > 0.0
            and holdout_ann_ret >= 8.0
            and worst_period_ret >= -3.0
        )

    sharpe = float(full_metrics.get("sharpe", full_metrics.get("Sharpe", 0.0)))
    raw_maxdd = full_metrics.get("maxdd_pct", full_metrics.get("MaxDD %", 0.0))
    maxdd = float(abs(raw_maxdd))
    beta = float(full_metrics.get("beta_btc", full_metrics.get("beta", 0.0)))
    mc_count = int(
        full_metrics.get(
            "Margin Calls",
            full_metrics.get("margin_call_count", margin_call_count),
        )
    )

    cond3_risk = bool(
        sharpe >= 1.80
        and maxdd <= 8.0
        and abs(beta) <= 0.15
        and mc_count == 0
    )

    cond4_execution_sync = bool(
        execution_mode == "maker_taker_hedge"
        and executor_state_machine_verified
        and not has_lookahead_bias
    )

    verdict = (
        "GO"
        if (cond1_return and cond2_consistency and cond3_risk and cond4_execution_sync)
        else "NO-GO"
    )

    failed: list[str] = []
    if not cond1_return:
        failed.append("cond1_return")
    if not cond2_consistency:
        failed.append("cond2_consistency")
    if not cond3_risk:
        failed.append("cond3_risk")
    if not cond4_execution_sync:
        failed.append("cond4_execution_sync")

    return {
        "cond1_return": cond1_return,
        "cond2_consistency": cond2_consistency,
        "cond3_risk": cond3_risk,
        "cond4_execution_sync": cond4_execution_sync,
        "verdict": verdict,
        "failed": failed,
        "pos_periods": pos_count,
        "total_periods": total_periods,
        "holdout_annualized_return_pct": holdout_ann_ret,
        "holdout_net_return_pct": holdout_net_ret,
        "details": {
            "annualized_return_pct": ann_ret,
            "sharpe": sharpe,
            "maxdd_pct": maxdd,
            "beta_btc": beta,
            "margin_call_count": mc_count,
            "worst_period_return_pct": worst_period_ret,
            "execution_mode": execution_mode,
            "executor_state_machine_verified": bool(executor_state_machine_verified),
            "has_lookahead_bias": bool(has_lookahead_bias),
        },
    }


def build_period_table_for_tier(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    n_periods: int = 7,
    period_days: int = 180,
    initial_capital: float = 3000.0,
) -> pd.DataFrame:
    """Xây dựng bảng tổng hợp chi tiết theo từng giai đoạn trên một đường cong vốn nối tiếp (Stitched).

    Net return của từng kỳ được tính trên vốn đầu kỳ = vốn cuối kỳ trước (lãi kép liên tục,
    không reset về 3000 USDT ở mỗi kỳ).

    Tham số:
        returns: Chuỗi tỷ suất lợi nhuận 1h đầy đủ của Tier.
        benchmark_returns: Chuỗi tỷ suất lợi nhuận benchmark (BTC).
        n_periods: Số giai đoạn (mặc định 7).
        period_days: Số ngày mỗi giai đoạn (mặc định 180).
        initial_capital: Vốn ban đầu (mặc định 3000.0 USDT).

    Trả về:
        DataFrame chứa các cột: period, period_id, start, end, net_return_pct,
        annualized_return_pct, sharpe, maxdd_pct, positive, start_capital, final_equity, net_pnl, n_bars.
    """
    columns = [
        "period",
        "period_id",
        "start",
        "end",
        "net_return_pct",
        "annualized_return_pct",
        "sharpe",
        "maxdd_pct",
        "positive",
        "start_capital",
        "final_equity",
        "net_pnl",
        "n_bars",
    ]
    if returns is None or len(returns) == 0:
        return pd.DataFrame(columns=columns)

    clean_ret = pd.Series(returns).dropna().sort_index()
    if len(clean_ret) == 0:
        return pd.DataFrame(columns=columns)

    clean_bench = (
        pd.Series(benchmark_returns).dropna().sort_index()
        if benchmark_returns is not None and len(benchmark_returns) > 0
        else None
    )

    periods = split_into_periods(clean_ret.index, n_periods=n_periods, days=period_days)
    curr_capital = float(initial_capital)
    rows: list[dict[str, Any]] = []

    for i, (start_ts, end_ts) in enumerate(periods):
        p_name = f"P{i + 1}"
        sub_ret = clean_ret[(clean_ret.index >= start_ts) & (clean_ret.index < end_ts)]
        sub_bench = (
            clean_bench[(clean_bench.index >= start_ts) & (clean_bench.index < end_ts)]
            if clean_bench is not None
            else None
        )

        m = period_metrics(
            returns=sub_ret,
            benchmark_returns=sub_bench,
            periods_per_year=8760,
            initial_capital=curr_capital,
            start=start_ts,
            end=end_ts,
        )

        net_ret_pct = float(m.get("net_return_pct", 0.0))
        ann_ret_pct = float(m.get("annualized_return_pct", 0.0))
        sharpe = float(m.get("sharpe", 0.0))
        maxdd = float(abs(m.get("maxdd_pct", 0.0)))
        pos = bool(net_ret_pct > 0.0)

        final_eq = float(m.get("final_equity", curr_capital * (1.0 + net_ret_pct / 100.0)))
        net_pnl = float(final_eq - curr_capital)
        n_bars = int(m.get("n_bars", len(sub_ret)))

        row = {
            "period": p_name,
            "period_id": p_name,
            "start": start_ts,
            "end": end_ts,
            "net_return_pct": net_ret_pct,
            "annualized_return_pct": ann_ret_pct,
            "sharpe": sharpe,
            "maxdd_pct": maxdd,
            "positive": pos,
            "start_capital": curr_capital,
            "final_equity": final_eq,
            "net_pnl": net_pnl,
            "n_bars": n_bars,
        }
        rows.append(row)
        curr_capital = final_eq

    return pd.DataFrame(rows)


def run_four_tiers_comparison(
    multi_aligned_data: dict[str, pd.DataFrame],
    btc_benchmark_returns: pd.Series,
    initial_capital: float = 3000.0,
    n_periods: int = 7,
    period_days: int = 180,
) -> dict[str, dict[str, Any]]:
    """Mô phỏng và so sánh 4 cấp độ tiến hóa (Tier 1 đến Tier 4) trên cùng chuỗi thời gian 3.45 năm.

    Tất cả 4 Tier đều dùng vốn chuẩn 3000.0 USDT và cơ chế phí thực tế maker_taker_hedge:
    - Tier 1: Single-Exchange Binance Static Carry (Baseline Sprint 8)
    - Tier 2: Multi-Exchange Best-Venue Carry (chọn sàn funding cao nhất Binance/Bybit/OKX)
    - Tier 3: Hybrid Cross-Exchange Perp-Perp Carry (chênh lệch giữa 3 sàn + eta 100%-125%)
    - Tier 4: Full Barbell Portfolio (Hybrid Carry + Core-4 Buffered Momentum Overlay)

    Tham số:
        multi_aligned_data: Dict map {symbol: DataFrame đã đồng bộ đa sàn}.
        btc_benchmark_returns: Chuỗi tỷ suất lợi nhuận benchmark (BTC).
        initial_capital: Vốn ban đầu (mặc định 3000.0 USDT).
        n_periods: Số giai đoạn cắt chuỗi (mặc định 7).
        period_days: Số ngày mỗi giai đoạn (mặc định 180).

    Trả về:
        Dict lồng chứa kết quả 4 Tier:
        {'tier1_single_binance': {...}, 'tier2_best_venue': {...}, 'tier3_hybrid_perp_perp': {...}, 'tier4_full_barbell': {...}}
        Mỗi Tier gồm: result, row, periods, equity_curve, returns.
    """
    # Tier 1: Single-Exchange Binance Static Carry
    cfg_t1 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="single_binance",
        structure_b="spot_perp_unified",
        execution="maker_taker_hedge",
    )
    res_t1 = run_cross_exchange_carry(multi_aligned_data, cfg_t1)

    # Tier 2: Multi-Exchange Best-Venue Carry
    cfg_t2 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="best_venue_carry",
        structure_b="spot_perp_unified",
        execution="maker_taker_hedge",
    )
    res_t2 = run_cross_exchange_carry(multi_aligned_data, cfg_t2)

    # Tier 3: Hybrid Cross-Exchange Perp-Perp Carry
    cfg_t3 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="hybrid",
        structure_b="spot_perp_unified",
        execution="maker_taker_hedge",
    )
    res_t3 = run_cross_exchange_carry(multi_aligned_data, cfg_t3)

    # Tier 4: Full Barbell Portfolio
    carry_cfg_t4 = CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        mode="hybrid",
        structure_b="spot_perp_unified",
        execution="maker_taker_hedge",
    )
    barbell_cfg = BarbellConfig(
        initial_capital=initial_capital,
        carry_config=carry_cfg_t4,
        enable_overlay=True,
        overlay_gross_leverage=0.40,
        overlay_lookback_hours=336,
        overlay_rebalance_hours=48,
        overlay_buffer_rank=1,
        overlay_min_spread=0.75,
        overlay_top_k=1,
        overlay_execution="maker_taker_hedge",
    )
    res_t4 = run_barbell_portfolio(multi_aligned_data, barbell_cfg)

    tiers_meta = [
        ("tier1_single_binance", "Tier 1 — Single-Exchange Binance Static Carry", res_t1),
        ("tier2_best_venue", "Tier 2 — Multi-Exchange Best-Venue Carry", res_t2),
        ("tier3_hybrid_perp_perp", "Tier 3 — Hybrid Cross-Exchange Perp-Perp Carry", res_t3),
        ("tier4_full_barbell", "Tier 4 — Full Barbell Portfolio", res_t4),
    ]

    out: dict[str, dict[str, Any]] = {}
    for tier_key, tier_name, res in tiers_meta:
        row = build_tier_row(
            tier_name=tier_name,
            result=res,
            benchmark_returns=btc_benchmark_returns,
            initial_capital=initial_capital,
        )
        periods_df = build_period_table_for_tier(
            returns=res.get("returns"),
            benchmark_returns=btc_benchmark_returns,
            n_periods=n_periods,
            period_days=period_days,
            initial_capital=initial_capital,
        )

        pos_count = int(sum(1 for p in periods_df["positive"] if p)) if not periods_df.empty else 0
        total_p = int(len(periods_df))
        row["pos_periods"] = pos_count
        row["total_periods"] = total_p
        row["positive_periods"] = f"{pos_count}/{total_p}"

        out[tier_key] = {
            "result": res,
            "row": row,
            "periods": periods_df,
            "equity_curve": res.get("equity_curve"),
            "returns": res.get("returns"),
        }

    return out


def detect_executor_state_machine() -> bool:
    """Tự động kiểm tra CarryLiveExecutor có được cấu hình đúng máy trạng thái thực thi hai chân hay không.

    Tiêu chí kiểm định:
    - inspect.signature(CarryLiveExecutor.__init__) chứa tham số execution_style với default 'maker_taker_hedge'.
    - Lớp CarryLiveExecutor sở hữu 2 phương thức:
        + place_maker_leg1
        + verify_maker_fill_and_hedge_leg2

    Trả về:
        True nếu thỏa mãn toàn bộ cấu hình máy trạng thái, False nếu có lỗi hoặc thiếu trường.
    """
    try:
        from src.carry_live_executor import CarryLiveExecutor

        sig = inspect.signature(CarryLiveExecutor.__init__)
        if "execution_style" not in sig.parameters:
            return False
        param = sig.parameters["execution_style"]
        if param.default != "maker_taker_hedge":
            return False

        if not hasattr(CarryLiveExecutor, "place_maker_leg1") or not callable(
            getattr(CarryLiveExecutor, "place_maker_leg1")
        ):
            return False

        if not hasattr(CarryLiveExecutor, "verify_maker_fill_and_hedge_leg2") or not callable(
            getattr(CarryLiveExecutor, "verify_maker_fill_and_hedge_leg2")
        ):
            return False

        return True
    except Exception:
        return False
