"""Module đánh giá và kiểm định nghiệm thu Sprint 8.

Cung cấp các hàm phân tích hiệu suất chuyên biệt cho chiến lược Dynamic Carry
và hội tụ Basis Spot-Perp trên đường cong vốn nối tiếp duy nhất (Stitched Equity Curve):
- split_into_periods: Chia chuỗi thời gian thành các cửa sổ 180 ngày nối tiếp.
- period_metrics: Tính toán chỉ số hiệu suất trên từng cửa sổ thời gian.
- build_period_table: Xây dựng bảng thống kê chi tiết từng giai đoạn kèm PnL Attribution.
- build_pnl_attribution: Phân rã rạch ròi các nguồn lợi nhuận (Funding, Basis, Fees).
- evaluate_sprint8_go_no_go: Kiểm định 4 điều kiện GO / NO-GO khóa cứng theo docs/go_no_go_sprint8.md.
- summarize_periods: Tóm tắt tính ổn định qua các giai đoạn.
- to_markdown_table: Xuất bảng Markdown tái sử dụng từ sprint7_eval.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.sprint7_eval import segment_metrics, to_markdown_table

__all__ = [
    "split_into_periods",
    "period_metrics",
    "build_period_table",
    "build_pnl_attribution",
    "evaluate_sprint8_go_no_go",
    "summarize_periods",
    "to_markdown_table",
]


def split_into_periods(
    index: pd.DatetimeIndex,
    n_periods: int = 7,
    days: int = 180,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Chia trục thời gian thành n_periods cửa sổ days ngày nối tiếp nhau.

    Cửa sổ cuối cùng nhận toàn bộ phần thời gian còn lại đến cuối index để phủ trọn vẹn
    chuỗi dữ liệu mà không bỏ sót nến dư.

    Tham số:
        index: DatetimeIndex của dữ liệu nến (kỳ vọng UTC).
        n_periods: Số giai đoạn cần chia (mặc định 7 giai đoạn 6 tháng).
        days: Số ngày mỗi giai đoạn (mặc định 180 ngày).

    Trả về:
        Danh sách các tuple (start_ts, end_ts) có tz UTC, liên tục không chồng lấp.
    """
    if index is None or len(index) == 0:
        return []

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.sort_values()

    if n_periods <= 1:
        start_ts = idx[0]
        end_ts = idx[-1] + pd.Timedelta(hours=1)
        return [(start_ts, end_ts)]

    t0 = idx[0]
    t_end = idx[-1]
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for k in range(n_periods - 1):
        start_k = t0 + pd.Timedelta(days=k * days)
        end_k = t0 + pd.Timedelta(days=(k + 1) * days)
        periods.append((start_k, end_k))

    # Cửa sổ cuối cùng nhận toàn bộ phần còn lại
    start_last = t0 + pd.Timedelta(days=(n_periods - 1) * days)
    end_last = t_end + pd.Timedelta(hours=1)
    if end_last <= start_last:
        end_last = start_last + pd.Timedelta(days=days)
    periods.append((start_last, end_last))

    return periods


def period_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 8760,
    num_trials: int = 12,
    initial_capital: float = 400.0,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Tính toàn bộ chỉ số hiệu suất trên một giai đoạn lợi nhuận định kỳ.

    Tái sử dụng segment_metrics từ sprint7_eval và bổ sung các trường thời gian
    start, end, n_bars.

    Tham số:
        returns: Chuỗi tỷ suất lợi nhuận pct_change của danh mục.
        benchmark_returns: Chuỗi lợi nhuận benchmark (BTC Buy-and-Hold) hoặc None.
        periods_per_year: Số chu kỳ mỗi năm (8760 cho nến 1h).
        num_trials: Số thử nghiệm K dùng để tính Deflated Sharpe Ratio (mặc định 12).
        initial_capital: Vốn ban đầu (USDT).
        start: Mốc thời gian bắt đầu giai đoạn (tùy chọn).
        end: Mốc thời gian kết thúc giai đoạn (tùy chọn).

    Trả về:
        Dict chỉ số chuẩn từ segment_metrics kèm start, end, n_bars.
    """
    seg = segment_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        num_trials=num_trials,
        initial_capital=initial_capital,
    )

    clean_ret = pd.Series(returns).dropna() if returns is not None else pd.Series([], dtype="float64")
    n_bars = int(len(clean_ret))

    if start is not None:
        start_val = pd.Timestamp(start)
    elif n_bars > 0 and isinstance(clean_ret.index, pd.DatetimeIndex):
        start_val = pd.Timestamp(clean_ret.index[0])
    else:
        start_val = None

    if end is not None:
        end_val = pd.Timestamp(end)
    elif n_bars > 0 and isinstance(clean_ret.index, pd.DatetimeIndex):
        end_val = pd.Timestamp(clean_ret.index[-1])
    else:
        end_val = None

    seg["start"] = start_val
    seg["end"] = end_val
    seg["n_bars"] = n_bars

    return seg


def build_period_table(
    returns: pd.Series,
    benchmark: pd.Series | None,
    periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    attribution_series: pd.DataFrame | None = None,
    initial_capital: float = 400.0,
    num_trials: int = 12,
    periods_per_year: int = 8760,
) -> pd.DataFrame:
    """Xây dựng bảng tổng hợp chi tiết theo từng giai đoạn thời gian.

    Lưu ý: funding_pnl, basis_pnl, fees được cắt chính xác theo cửa sổ thời gian
    từ attribution_series của động cơ mô phỏng (không chia đều).

    Tham số:
        returns: Chuỗi tỷ suất lợi nhuận đầy đủ của danh mục.
        benchmark: Chuỗi lợi nhuận benchmark đầy đủ hoặc None.
        periods: Danh sách các khoảng (start_ts, end_ts) từ split_into_periods.
        attribution_series: DataFrame attribution theo giờ gồm funding_pnl, basis_pnl, fees, notional_traded.
        initial_capital: Vốn khởi tạo (USDT).
        num_trials: Số lần thử nghiệm K (mặc định 12).
        periods_per_year: Số chu kỳ mỗi năm (8760).

    Trả về:
        DataFrame với mỗi dòng là 1 giai đoạn gồm:
        period_index, period_id, start, end, net_return_pct, annualized_return_pct,
        sharpe, maxdd_pct, dsr_prob, turnover, funding_pnl, basis_pnl, fees, positive, n_bars.
    """
    columns = [
        "period_index",
        "period_id",
        "start",
        "end",
        "net_return_pct",
        "annualized_return_pct",
        "sharpe",
        "maxdd_pct",
        "dsr_prob",
        "turnover",
        "funding_pnl",
        "basis_pnl",
        "fees",
        "positive",
        "n_bars",
    ]

    if returns is None or len(returns) == 0 or not periods:
        return pd.DataFrame(columns=columns)

    ret_clean = pd.Series(returns).dropna().sort_index()
    bench_clean = pd.Series(benchmark).dropna().sort_index() if benchmark is not None else None

    rows: list[dict[str, Any]] = []

    for i, (start_ts, end_ts) in enumerate(periods):
        p_idx = i + 1
        p_id = f"P{p_idx}"

        sub_ret = ret_clean[(ret_clean.index >= start_ts) & (ret_clean.index < end_ts)]
        sub_bench = (
            bench_clean[(bench_clean.index >= start_ts) & (bench_clean.index < end_ts)]
            if bench_clean is not None
            else None
        )

        m = period_metrics(
            returns=sub_ret,
            benchmark_returns=sub_bench,
            periods_per_year=periods_per_year,
            num_trials=num_trials,
            initial_capital=initial_capital,
            start=start_ts,
            end=end_ts,
        )

        funding_pnl = 0.0
        basis_pnl = 0.0
        fees = 0.0
        turnover = 0.0

        if attribution_series is not None and not attribution_series.empty:
            sub_attr = attribution_series[
                (attribution_series.index >= start_ts) & (attribution_series.index < end_ts)
            ]
            if not sub_attr.empty:
                if "funding_pnl" in sub_attr.columns:
                    funding_pnl = float(sub_attr["funding_pnl"].sum())
                if "basis_pnl" in sub_attr.columns:
                    basis_pnl = float(sub_attr["basis_pnl"].sum())
                if "fees" in sub_attr.columns:
                    fees = float(sub_attr["fees"].sum())
                if "notional_traded" in sub_attr.columns and initial_capital > 0:
                    turnover = float(sub_attr["notional_traded"].sum() / initial_capital)

        net_ret = float(m["net_return_pct"])
        positive = bool(net_ret > 0.0)

        rows.append({
            "period_index": p_idx,
            "period_id": p_id,
            "start": start_ts,
            "end": end_ts,
            "net_return_pct": net_ret,
            "annualized_return_pct": float(m["annualized_return_pct"]),
            "sharpe": float(m["sharpe"]),
            "maxdd_pct": float(m["maxdd_pct"]),
            "dsr_prob": float(m["dsr_prob"]),
            "turnover": turnover,
            "funding_pnl": funding_pnl,
            "basis_pnl": basis_pnl,
            "fees": fees,
            "positive": positive,
            "n_bars": int(m["n_bars"]),
        })

    df = pd.DataFrame(rows)
    return df


def build_pnl_attribution(
    attribution_data: pd.DataFrame | dict[str, Any],
    periods: Sequence[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    initial_capital: float = 400.0,
) -> dict[str, Any]:
    """Phân rã nguồn lợi nhuận PnL Attribution thành Funding, Basis và Phí giao dịch.

    Tham số:
        attribution_data: DataFrame attribution hoặc dict kết quả trả về từ run_dynamic_carry_portfolio.
        periods: Danh sách các khoảng (start, end) để phân rã theo từng kỳ (tùy chọn).
        initial_capital: Vốn ban đầu (USDT).

    Trả về:
        Dict chứa các chỉ số PnL tổng và chi tiết từng giai đoạn nếu periods được cung cấp.
    """
    if isinstance(attribution_data, dict):
        attr_df = attribution_data.get("attribution_series")
    else:
        attr_df = attribution_data

    empty_res: dict[str, Any] = {
        "funding_pnl": 0.0,
        "basis_pnl": 0.0,
        "fees": 0.0,
        "net_pnl": 0.0,
        "notional_traded": 0.0,
        "funding_pnl_usdt": 0.0,
        "basis_pnl_usdt": 0.0,
        "fee_drag_usdt": 0.0,
        "net_pnl_usdt": 0.0,
        "funding_return_pct": 0.0,
        "basis_return_pct": 0.0,
        "fee_drag_pct": 0.0,
        "net_return_pct": 0.0,
        "funding_share_pct": 0.0,
        "basis_share_pct": 0.0,
        "fee_share_pct": 0.0,
        "periods": [],
        "periods_df": pd.DataFrame(),
    }

    if attr_df is None or attr_df.empty:
        return empty_res

    funding_pnl = float(attr_df["funding_pnl"].sum()) if "funding_pnl" in attr_df.columns else 0.0
    basis_pnl = float(attr_df["basis_pnl"].sum()) if "basis_pnl" in attr_df.columns else 0.0
    fees = float(attr_df["fees"].sum()) if "fees" in attr_df.columns else 0.0
    net_pnl = funding_pnl + basis_pnl - fees
    notional_traded = (
        float(attr_df["notional_traded"].sum()) if "notional_traded" in attr_df.columns else 0.0
    )

    cap = float(initial_capital) if initial_capital > 0 else 400.0
    funding_return_pct = (funding_pnl / cap) * 100.0
    basis_return_pct = (basis_pnl / cap) * 100.0
    fee_drag_pct = (fees / cap) * 100.0
    net_return_pct = (net_pnl / cap) * 100.0

    gross_components = abs(funding_pnl) + abs(basis_pnl) + abs(fees)
    if gross_components > 0:
        funding_share_pct = (abs(funding_pnl) / gross_components) * 100.0
        basis_share_pct = (abs(basis_pnl) / gross_components) * 100.0
        fee_share_pct = (abs(fees) / gross_components) * 100.0
    else:
        funding_share_pct = 0.0
        basis_share_pct = 0.0
        fee_share_pct = 0.0

    period_rows: list[dict[str, Any]] = []
    if periods:
        for i, (s_ts, e_ts) in enumerate(periods):
            sub = attr_df[(attr_df.index >= s_ts) & (attr_df.index < e_ts)]
            p_funding = float(sub["funding_pnl"].sum()) if "funding_pnl" in sub.columns else 0.0
            p_basis = float(sub["basis_pnl"].sum()) if "basis_pnl" in sub.columns else 0.0
            p_fees = float(sub["fees"].sum()) if "fees" in sub.columns else 0.0
            p_net = p_funding + p_basis - p_fees
            p_notional = float(sub["notional_traded"].sum()) if "notional_traded" in sub.columns else 0.0

            period_rows.append({
                "period_index": i + 1,
                "period_id": f"P{i+1}",
                "start": s_ts,
                "end": e_ts,
                "funding_pnl": p_funding,
                "basis_pnl": p_basis,
                "fees": p_fees,
                "net_pnl": p_net,
                "notional_traded": p_notional,
                "funding_return_pct": (p_funding / cap) * 100.0,
                "basis_return_pct": (p_basis / cap) * 100.0,
                "fee_drag_pct": (p_fees / cap) * 100.0,
                "net_return_pct": (p_net / cap) * 100.0,
            })

    periods_df = pd.DataFrame(period_rows)

    return {
        "funding_pnl": funding_pnl,
        "basis_pnl": basis_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "notional_traded": notional_traded,
        "funding_pnl_usdt": funding_pnl,
        "basis_pnl_usdt": basis_pnl,
        "fee_drag_usdt": fees,
        "net_pnl_usdt": net_pnl,
        "funding_return_pct": funding_return_pct,
        "basis_return_pct": basis_return_pct,
        "fee_drag_pct": fee_drag_pct,
        "net_return_pct": net_return_pct,
        "funding_share_pct": funding_share_pct,
        "basis_share_pct": basis_share_pct,
        "fee_share_pct": fee_share_pct,
        "periods": period_rows,
        "periods_df": periods_df,
    }


def evaluate_sprint8_go_no_go(
    full_12: dict[str, Any],
    full_8: dict[str, Any],
    period_rows_12: Sequence[dict[str, Any]] | pd.DataFrame,
    live_executor_ready: bool = True,
    num_trials: int = 12,
) -> dict[str, Any]:
    """Đánh giá 4 điều kiện GO / NO-GO khóa cứng theo docs/go_no_go_sprint8.md.

    4 điều kiện bắt buộc:
    1. cond1_return: annualized_return_pct(12) >= 8.0 VÀ annualized_return_pct(8) >= 6.0.
    2. cond2_consistency: số kỳ net_return_pct > 0 >= 6/7 (trên rổ 12 coin).
    3. cond3_risk: sharpe >= 2.00 VÀ dsr_prob >= 0.95 (với K=12) VÀ maxdd_pct <= 5.0.
    4. cond4_production: live_executor_ready == True.

    Tham số:
        full_12: Dict chỉ số toàn chuỗi 3.45 năm trên rổ 12 coin.
        full_8: Dict chỉ số toàn chuỗi 3.45 năm trên rổ 8 coin mở rộng (Out-of-Universe).
        period_rows_12: Bảng kết quả từng giai đoạn của rổ 12 coin.
        live_executor_ready: Cờ xác nhận CarryLiveExecutor sẵn sàng vận hành.
        num_trials: Số lần thử nghiệm K (12).

    Trả về:
        Dict gồm các cờ điều kiện, số kỳ dương, verdict ('GO' / 'NO-GO'), danh sách failed và details.
    """
    ann_ret_12 = float(full_12.get("annualized_return_pct", full_12.get("Annualized Return %", 0.0)))
    ann_ret_8 = float(full_8.get("annualized_return_pct", full_8.get("Annualized Return %", 0.0)))

    sharpe_12 = float(full_12.get("sharpe", full_12.get("Sharpe", 0.0)))
    dsr_prob_12 = float(full_12.get("dsr_prob", full_12.get("dsr", 0.0)))
    maxdd_12 = float(full_12.get("maxdd_pct", full_12.get("MaxDD %", 999.0)))

    if isinstance(period_rows_12, pd.DataFrame):
        if "positive" in period_rows_12.columns:
            pos_count = int(period_rows_12["positive"].sum())
        elif "net_return_pct" in period_rows_12.columns:
            pos_count = int((period_rows_12["net_return_pct"] > 0.0).sum())
        else:
            pos_count = 0
        total_periods = len(period_rows_12)
    else:
        pos_count = sum(1 for r in period_rows_12 if float(r.get("net_return_pct", 0.0)) > 0.0)
        total_periods = len(period_rows_12)

    cond1 = bool(ann_ret_12 >= 8.0 and ann_ret_8 >= 6.0)
    cond2 = bool(pos_count >= 6 and total_periods >= 6)
    cond3 = bool(sharpe_12 >= 2.00 and dsr_prob_12 >= 0.95 and maxdd_12 <= 5.0)
    cond4 = bool(live_executor_ready)

    failed: list[str] = []
    if not cond1:
        failed.append("cond1_return")
    if not cond2:
        failed.append("cond2_consistency")
    if not cond3:
        failed.append("cond3_risk")
    if not cond4:
        failed.append("cond4_production")

    verdict = "GO" if not failed else "NO-GO"

    return {
        "cond1_return": cond1,
        "cond1_annualized_return": cond1,
        "cond2_consistency": cond2,
        "cond2_subperiod_consistency": cond2,
        "cond3_risk": cond3,
        "cond3_risk_quality": cond3,
        "cond4_production": cond4,
        "cond4_production_readiness": cond4,
        "positive_periods_count": int(pos_count),
        "total_periods": int(total_periods),
        "verdict": verdict,
        "failed": failed,
        "details": {
            "annualized_return_12": ann_ret_12,
            "annualized_return_8": ann_ret_8,
            "sharpe_12": sharpe_12,
            "dsr_prob_12": dsr_prob_12,
            "maxdd_pct_12": maxdd_12,
            "positive_periods": pos_count,
            "total_periods": total_periods,
            "live_executor_ready": bool(live_executor_ready),
            "num_trials": int(num_trials),
        },
    }


def summarize_periods(rows: Sequence[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    """Tóm tắt phân phối lợi nhuận qua các giai đoạn.

    Tham số:
        rows: Danh sách dict hoặc DataFrame từ build_period_table.

    Trả về:
        Dict: n_periods, n_positive, positive_ratio, worst_period_return, best_period_return.
    """
    empty = {
        "n_periods": 0,
        "n_positive": 0,
        "positive_ratio": 0.0,
        "worst_period_return": 0.0,
        "best_period_return": 0.0,
    }
    if rows is None:
        return empty

    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return empty
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)

    if not records:
        return empty

    rets = [float(r.get("net_return_pct", 0.0)) for r in records]
    n_periods = len(rets)
    n_positive = sum(1 for ret in rets if ret > 0.0)
    pos_ratio = float(n_positive / n_periods) if n_periods > 0 else 0.0
    worst = float(min(rets)) if rets else 0.0
    best = float(max(rets)) if rets else 0.0

    return {
        "n_periods": n_periods,
        "n_positive": n_positive,
        "positive_ratio": pos_ratio,
        "worst_period_return": worst,
        "best_period_return": best,
    }
