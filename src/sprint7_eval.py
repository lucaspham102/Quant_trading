"""Module đánh giá và kiểm định nghiệm thu Sprint 7.

Cung cấp:
- segment_metrics: tính toàn bộ chỉ số hiệu suất trên một đoạn (DEV / HOLDOUT) của
  đường cong vốn nối tiếp, bao gồm Newey-West Alpha/Beta so với BTC Buy-and-Hold,
  Sharpe năm hóa, Max Drawdown và Deflated Sharpe Ratio (DSR).
- evaluate_cross_sectional_go_no_go / evaluate_carry_go_no_go: áp dụng bộ tiêu chí
  GO / NO-GO đã khóa trong docs/go_no_go_sprint7.md.
- Các tiện ích dựng đường cong vốn, benchmark, bảng Markdown và tổng hợp phán quyết.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.sprint6_eval import (
    annualized_return_pct,
    annualized_sharpe,
    compute_deflated_sharpe_ratio,
    compute_newey_west_alpha_beta,
    max_drawdown_pct,
)

__all__ = [
    "segment_metrics",
    "build_benchmark_returns",
    "equity_from_returns",
    "evaluate_cross_sectional_go_no_go",
    "evaluate_carry_go_no_go",
    "compute_turnover_reduction",
    "summarize_verdicts",
    "to_markdown_table",
]


def _clean_returns(returns: pd.Series) -> pd.Series:
    """Lọc NaN, sắp xếp index và đảm bảo kiểu float64."""
    if returns is None:
        raise ValueError("returns không được là None")
    clean = pd.Series(returns).dropna().astype("float64").sort_index()
    return clean


def segment_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 8760,
    num_trials: int = 1,
    initial_capital: float = 400.0,
) -> dict:
    """Tính toàn bộ chỉ số hiệu suất trên một đoạn lợi nhuận.

    Tham số:
        returns: Chuỗi tỷ suất lợi nhuận định kỳ của chiến lược.
        benchmark_returns: Chuỗi lợi nhuận benchmark (BTC Buy-and-Hold) hoặc None.
        periods_per_year: Số kỳ mỗi năm (8760 cho nến 1h).
        num_trials: Số cấu hình đã thử K dùng cho DSR.
        initial_capital: Vốn khởi tạo để dựng equity curve phục vụ Max Drawdown.

    Trả về:
        Dict chỉ số: net_return_pct, annualized_return_pct, sharpe, maxdd_pct,
        volatility_ann_pct, n_bars, n_days, dsr_prob, sr_star, skew, kurt, num_trials,
        alpha, alpha_t, beta, beta_t, alpha_annualized_pct, nw_lags.
    """
    clean = _clean_returns(returns)
    empty = {
        "net_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "sharpe": 0.0,
        "maxdd_pct": 0.0,
        "volatility_ann_pct": 0.0,
        "n_bars": 0,
        "n_days": 0.0,
        "dsr_prob": 0.0,
        "sr_star": 0.0,
        "skew": 0.0,
        "kurt": 3.0,
        "num_trials": int(num_trials),
        "alpha": 0.0,
        "alpha_t": 0.0,
        "beta": 0.0,
        "beta_t": 0.0,
        "alpha_annualized_pct": 0.0,
        "nw_lags": 0,
    }
    if len(clean) == 0:
        return empty

    equity = equity_from_returns(clean, initial_capital=initial_capital)
    cum = float(np.prod(1.0 + clean.to_numpy(dtype="float64")))
    net_return_pct = float((cum - 1.0) * 100.0) if math.isfinite(cum) else 0.0

    std_val = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    vol_ann = float(std_val * math.sqrt(periods_per_year) * 100.0) if std_val > 0 else 0.0

    dsr = compute_deflated_sharpe_ratio(clean, num_trials=num_trials, periods_per_year=periods_per_year)

    if isinstance(clean.index, pd.DatetimeIndex) and len(clean) > 1:
        n_days = float((clean.index[-1] - clean.index[0]).total_seconds() / 86_400.0)
    else:
        n_days = 0.0

    out: dict[str, Any] = {
        "net_return_pct": net_return_pct,
        "annualized_return_pct": float(annualized_return_pct(clean, periods_per_year)),
        "sharpe": float(annualized_sharpe(clean, periods_per_year)),
        "maxdd_pct": float(max_drawdown_pct(equity)),
        "volatility_ann_pct": vol_ann,
        "n_bars": int(len(clean)),
        "n_days": n_days,
        "dsr_prob": float(dsr["dsr_prob"]),
        "sr_star": float(dsr["sr_star"]),
        "skew": float(dsr["skew"]),
        "kurt": float(dsr["kurt"]),
        "num_trials": int(num_trials),
        "alpha": 0.0,
        "alpha_t": 0.0,
        "beta": 0.0,
        "beta_t": 0.0,
        "alpha_annualized_pct": 0.0,
        "nw_lags": 0,
    }

    if benchmark_returns is not None:
        bench = _clean_returns(benchmark_returns)
        common = clean.index.intersection(bench.index)
        if len(common) >= 2:
            nw = compute_newey_west_alpha_beta(
                clean.loc[common], bench.loc[common], periods_per_year=periods_per_year
            )
            out["alpha"] = float(nw["alpha"])
            out["alpha_t"] = float(nw["alpha_t"])
            out["beta"] = float(nw["beta"])
            out["beta_t"] = float(nw["beta_t"])
            out["alpha_annualized_pct"] = float(nw["alpha_annualized_pct"])
            out["nw_lags"] = int(nw["lags"])

    return out


def build_benchmark_returns(btc_close: pd.Series) -> pd.Series:
    """Chuyển chuỗi giá BTC thành chuỗi lợi nhuận Buy-and-Hold (pct_change, NaN -> 0)."""
    if btc_close is None:
        raise ValueError("btc_close không được là None")
    close = pd.Series(btc_close).astype("float64").sort_index()
    return close.pct_change().fillna(0.0).rename("btc_bh")


def equity_from_returns(returns: pd.Series, initial_capital: float = 400.0) -> pd.Series:
    """Dựng đường cong vốn từ chuỗi lợi nhuận (không reset vốn ở biên)."""
    clean = _clean_returns(returns)
    if len(clean) == 0:
        return pd.Series([], dtype="float64", name="equity")
    values = float(initial_capital) * np.cumprod(1.0 + clean.to_numpy(dtype="float64"))
    return pd.Series(values, index=clean.index, name="equity", dtype="float64")


def evaluate_cross_sectional_go_no_go(
    dev: dict,
    holdout: dict,
    universe: str,
    num_trials: int = 16,
    alpha_t_min: float | None = None,
) -> dict:
    """Đánh giá GO / NO-GO cho hướng Cross-Sectional theo docs/go_no_go_sprint7.md.

    Tham số:
        dev: Dict chỉ số trên tập DEV (từ segment_metrics).
        holdout: Dict chỉ số trên tập HOLDOUT.
        universe: "xs8" (rổ 8 coin mới) hoặc "xs12" (rổ 12 coin).
        num_trials: Số cấu hình K dùng cho DSR (mặc định 16).
        alpha_t_min: Ngưỡng alpha t-stat; mặc định 2.00 (xs8) / 2.20 (xs12).

    Trả về:
        Dict gồm 3 cờ điều kiện, "verdict" ('GO'/'NO-GO'), danh sách điều kiện trượt.
    """
    if universe not in ("xs8", "xs12"):
        raise ValueError(f"universe phải là 'xs8' hoặc 'xs12', nhận: '{universe}'")

    if alpha_t_min is None:
        alpha_t_min = 2.00 if universe == "xs8" else 2.20

    alpha_t = float(dev.get("alpha_t", 0.0))
    beta = float(dev.get("beta", 1.0))
    sharpe_dev = float(dev.get("sharpe", 0.0))
    dsr = float(dev.get("dsr_prob", 0.0))

    ho_ret = float(holdout.get("net_return_pct", 0.0))
    ho_sharpe = float(holdout.get("sharpe", 0.0))
    ho_maxdd = float(holdout.get("maxdd_pct", 999.0))

    cond1 = bool(alpha_t >= float(alpha_t_min) and abs(beta) <= 0.15)
    cond2 = bool(sharpe_dev >= 1.10 and dsr >= 0.90)
    cond3 = bool(ho_ret > 8.0 and ho_sharpe >= 1.20 and ho_maxdd <= 15.0)

    failed: list[str] = []
    if not cond1:
        failed.append("cond1_alpha_beta")
    if not cond2:
        failed.append("cond2_sharpe_dsr")
    if not cond3:
        failed.append("cond3_holdout")

    return {
        "universe": universe,
        "cond1_alpha_beta": cond1,
        "cond2_sharpe_dsr": cond2,
        "cond3_holdout": cond3,
        "verdict": "GO" if not failed else "NO-GO",
        "alpha_t_min": float(alpha_t_min),
        "num_trials": int(num_trials),
        "failed": failed,
    }


def evaluate_carry_go_no_go(dev: dict, holdout: dict) -> dict:
    """Đánh giá GO / NO-GO cho True MTM Carry theo docs/go_no_go_sprint7.md.

    Điều kiện:
        1. MaxDD MTM thực tế > 0% (bác bỏ mô hình phẳng) và <= 10.0% trên cả DEV & HOLDOUT.
        2. Annualized Net Return >= 6.0%/năm trên cả DEV & HOLDOUT và Sharpe MTM >= 1.50.
    """
    dev_dd = float(dev.get("maxdd_pct", 0.0))
    ho_dd = float(holdout.get("maxdd_pct", 0.0))
    dev_ret = float(dev.get("annualized_return_pct", 0.0))
    ho_ret = float(holdout.get("annualized_return_pct", 0.0))
    dev_sharpe = float(dev.get("sharpe", 0.0))
    ho_sharpe = float(holdout.get("sharpe", 0.0))

    cond1 = bool(0.0 < dev_dd <= 10.0 and 0.0 < ho_dd <= 10.0)
    cond2 = bool(
        dev_ret >= 6.0
        and ho_ret >= 6.0
        and dev_sharpe >= 1.50
        and ho_sharpe >= 1.50
    )

    failed: list[str] = []
    if not cond1:
        failed.append("cond1_maxdd")
    if not cond2:
        failed.append("cond2_return_sharpe")

    return {
        "cond1_maxdd": cond1,
        "cond2_return_sharpe": cond2,
        "verdict": "GO" if not failed else "NO-GO",
        "failed": failed,
    }


def compute_turnover_reduction(turnover_no_buffer: float, turnover_buffer: float) -> float:
    """Tính % giảm turnover khi bật buffer band (dương = buffer tốt hơn)."""
    base = float(turnover_no_buffer)
    buffered = float(turnover_buffer)
    if base <= 0.0 or not math.isfinite(base):
        return 0.0
    return float((1.0 - buffered / base) * 100.0)


def summarize_verdicts(rows: Sequence[dict]) -> dict:
    """Đếm số phán quyết GO / NO-GO trong danh sách kết quả đã được đánh giá."""
    evaluated = [r for r in rows if str(r.get("verdict", "")).upper() in ("GO", "NO-GO")]
    go = sum(1 for r in evaluated if str(r.get("verdict", "")).upper() == "GO")
    total = len(evaluated)
    return {"go": int(go), "no_go": int(total - go), "total": int(total)}


def _format_cell(value: Any) -> str:
    """Định dạng một ô cho bảng Markdown."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "✔" if bool(value) else "✘"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4f}"
    text = str(value)
    return text.replace("|", "\\|")


def to_markdown_table(rows: Sequence[dict], columns: list[str] | None = None) -> str:
    """Xuất danh sách dict thành bảng Markdown đơn giản."""
    if not rows:
        return "(không có dữ liệu)"
    cols = list(columns) if columns is not None else list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(_format_cell(row.get(c)) for c in cols) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])
