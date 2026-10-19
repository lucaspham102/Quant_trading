"""Module đánh giá hiệu suất nâng cao và kiểm định nghiệm thu Sprint 6.

Cung cấp các phương pháp thống kê và tiêu chí nghiệm thu chặt chẽ:
- Deflated Sharpe Ratio (DSR) trừng phạt skewness, kurtosis và số lần thử K (Bailey & López de Prado).
- Hồi quy Newey-West HAC tính Alpha/Beta chuẩn hóa theo năm hóa so với Benchmark.
- Chuẩn hóa Stitched Walk-Forward Returns không reset vốn ở biên.
- Bộ tiêu chí kiểm định GO / NO-GO khóa cứng theo docs/go_no_go_sprint6.md.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.deep_analysis import newey_west_ols

__all__ = [
    "annualized_sharpe",
    "max_drawdown_pct",
    "annualized_return_pct",
    "compute_deflated_sharpe_ratio",
    "compute_newey_west_alpha_beta",
    "stitch_returns",
    "evaluate_go_no_go",
]


class GoNoGoResult(dict):
    """Dict chứa kết quả đánh giá GO / NO-GO với khả năng truy cập qua alias."""

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "cond1": "cond1_market_neutral",
            "condition_1": "cond1_market_neutral",
            "market_neutral": "cond1_market_neutral",
            "cond2": "cond2_sharpe",
            "condition_2": "cond2_sharpe",
            "sharpe": "cond2_sharpe",
            "cond3": "cond3_dsr",
            "condition_3": "cond3_dsr",
            "dsr": "cond3_dsr",
            "cond4": "cond4_holdout",
            "condition_4": "cond4_holdout",
            "holdout": "cond4_holdout",
        }
        if key in aliases and key not in self:
            return super().__getitem__(aliases[key])
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        aliases = {
            "cond1": "cond1_market_neutral",
            "condition_1": "cond1_market_neutral",
            "market_neutral": "cond1_market_neutral",
            "cond2": "cond2_sharpe",
            "condition_2": "cond2_sharpe",
            "sharpe": "cond2_sharpe",
            "cond3": "cond3_dsr",
            "condition_3": "cond3_dsr",
            "dsr": "cond3_dsr",
            "cond4": "cond4_holdout",
            "condition_4": "cond4_holdout",
            "holdout": "cond4_holdout",
        }
        target = aliases.get(key, key)
        return super().get(target, default)


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Tính Sharpe ratio năm hóa từ chuỗi tỷ suất lợi nhuận mỗi kỳ.

    Công thức: mean / std(ddof=1) * sqrt(periods_per_year).
    Nếu len < 2 hoặc std == 0 -> trả về 0.0.

    Parameters:
        returns: Chuỗi tỷ suất lợi nhuận định kỳ (pd.Series).
        periods_per_year: Số chu kỳ nến mỗi năm (mặc định 8760 cho nến 1h).

    Returns:
        Sharpe ratio năm hóa (float).
    """
    if returns is None or len(returns) < 2:
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    std_val = float(clean.std(ddof=1))
    if std_val <= 1e-12 or math.isnan(std_val):
        return 0.0
    mean_val = float(clean.mean())
    return float((mean_val / std_val) * math.sqrt(periods_per_year))


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    """Tính mức sụt giảm tối đa lớn nhất (Max Drawdown, %, dương) trên đường cong vốn.

    Tính từ đỉnh lũy tiến (peak-to-trough) trên chuỗi vốn (equity curve), không phải returns.
    Nếu chuỗi rỗng hoặc không sụt giảm -> trả về 0.0.

    Parameters:
        equity_curve: Chuỗi giá trị vốn (equity) qua từng kỳ.

    Returns:
        Max Drawdown theo phần trăm (float >= 0.0).
    """
    if equity_curve is None or len(equity_curve) == 0:
        return 0.0
    clean = equity_curve.dropna()
    if len(clean) == 0:
        return 0.0
    vals = np.asarray(clean, dtype=float)
    peak = np.maximum.accumulate(vals)
    valid = peak > 0.0
    if not np.any(valid):
        return 0.0
    dd = np.zeros_like(vals)
    dd[valid] = (peak[valid] - vals[valid]) / peak[valid] * 100.0
    max_dd = float(np.nanmax(dd))
    return max(max_dd, 0.0)


def annualized_return_pct(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """Tính tỷ suất sinh lời năm hóa theo công thức ghép lãi kép lũy tiến.

    Công thức: (tổng lũy tiến cuối)^(periods_per_year / n) - 1, * 100.
    Xử lý an toàn khi n <= 0 hoặc tổng tích lũy <= 0.

    Parameters:
        returns: Chuỗi tỷ suất lợi nhuận định kỳ (pd.Series).
        periods_per_year: Số chu kỳ nến mỗi năm (mặc định 8760 cho nến 1h).

    Returns:
        Tỷ suất sinh lời năm hóa (%) (float).
    """
    if returns is None or len(returns) == 0:
        return 0.0
    clean = returns.dropna()
    n = len(clean)
    if n <= 0:
        return 0.0
    cum = float((1.0 + clean).prod())
    if cum <= 0.0 or math.isnan(cum) or math.isinf(cum):
        return 0.0
    exponent = float(periods_per_year) / float(n)
    try:
        ann_ret = (cum ** exponent - 1.0) * 100.0
        if math.isnan(ann_ret) or math.isinf(ann_ret):
            return 0.0
        return float(ann_ret)
    except (OverflowError, ValueError):
        return 0.0


def compute_deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    periods_per_year: int = 8760,
    sr_variance: float | None = None,
) -> dict[str, float]:
    """Tính Deflated Sharpe Ratio (DSR) theo Bailey & López de Prado (2014).

    Parameters:
        returns: Chuỗi tỷ suất lợi nhuận định kỳ (pd.Series).
        num_trials: Tổng số cấu hình thử nghiệm K (để phạt multiple testing).
        periods_per_year: Số chu kỳ nến mỗi năm (mặc định 8760).
        sr_variance: Phương sai V[SR_k] nếu đã có; nếu None sẽ dùng ước lượng mẫu.

    Returns:
        Dict gồm các chỉ số:
        - sharpe_per_period: Sharpe mỗi kỳ (không năm hóa).
        - sharpe_annualized: Sharpe năm hóa.
        - sr_star: Ngưỡng kỳ vọng cực đại của Sharpe dưới giả thuyết H0 (SR*).
        - dsr_prob: Xác suất DSR = Phi((SR - SR*) / sqrt(V[SR])).
        - skew: Độ lệch gamma3.
        - kurt: Độ nhọn Pearson gamma4 = kurtosis Fisher + 3.0.
        - n: Số lượng quan sát thực tế.
        - num_trials: Số lần thử nghiệm K.
    """
    if returns is None or len(returns) < 2:
        return {
            "sharpe_per_period": 0.0,
            "sharpe_annualized": 0.0,
            "sr_star": 0.0,
            "dsr_prob": 0.0,
            "skew": 0.0,
            "kurt": 3.0,
            "n": 0 if returns is None else len(returns),
            "num_trials": int(num_trials),
        }

    clean = returns.dropna()
    n = len(clean)
    if n < 2:
        return {
            "sharpe_per_period": 0.0,
            "sharpe_annualized": 0.0,
            "sr_star": 0.0,
            "dsr_prob": 0.0,
            "skew": 0.0,
            "kurt": 3.0,
            "n": n,
            "num_trials": int(num_trials),
        }

    std_val = float(clean.std(ddof=1))
    if std_val <= 1e-12 or math.isnan(std_val):
        return {
            "sharpe_per_period": 0.0,
            "sharpe_annualized": 0.0,
            "sr_star": 0.0,
            "dsr_prob": 0.0,
            "skew": 0.0,
            "kurt": 3.0,
            "n": n,
            "num_trials": int(num_trials),
        }

    mean_val = float(clean.mean())
    sr = mean_val / std_val
    sharpe_annualized = sr * math.sqrt(periods_per_year)

    gamma3 = float(clean.skew())
    if math.isnan(gamma3):
        gamma3 = 0.0

    gamma4 = float(clean.kurt()) + 3.0
    if math.isnan(gamma4):
        gamma4 = 3.0

    denom_expr = 1.0 - gamma3 * sr + ((gamma4 - 1.0) / 4.0) * (sr ** 2)
    denom_val = max(denom_expr, 1e-12)

    if sr_variance is not None:
        var_sr = max(float(sr_variance), 1e-12)
    else:
        var_sr = max(denom_val / (n - 1.0), 1e-12)

    k_val = max(1, int(num_trials))
    if k_val < 2:
        sr_star = 0.0
    else:
        euler_mascheroni = 0.5772156649
        e_const = 2.718281828
        term1 = (1.0 - euler_mascheroni) * norm.ppf(1.0 - 1.0 / k_val)
        term2 = euler_mascheroni * norm.ppf(1.0 - 1.0 / (k_val * e_const))
        sr_star = float(math.sqrt(var_sr) * (term1 + term2))

    z = (sr - sr_star) * math.sqrt(n - 1.0) / math.sqrt(denom_val)
    dsr_prob = float(norm.cdf(z))
    dsr_prob = min(max(dsr_prob, 0.0), 1.0)

    return {
        "sharpe_per_period": float(sr),
        "sharpe_annualized": float(sharpe_annualized),
        "sr_star": float(sr_star),
        "dsr_prob": float(dsr_prob),
        "skew": float(gamma3),
        "kurt": float(gamma4),
        "n": int(n),
        "num_trials": int(num_trials),
    }


def compute_newey_west_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    lags: int | None = None,
    periods_per_year: int = 8760,
) -> dict:
    """Căn chỉnh 2 chuỗi lợi nhuận và hồi quy Newey-West HAC tính alpha/beta kèm alpha năm hóa.

    Căn chỉnh theo index giao nhau (intersection, sorted), dropna, sau đó gọi newey_west_ols.

    Parameters:
        strategy_returns: Chuỗi tỷ suất lợi nhuận của chiến lược (y).
        benchmark_returns: Chuỗi tỷ suất lợi nhuận của tài sản chuẩn, ví dụ BTC (x).
        lags: Số bậc trễ Bartlett kernel (mặc định để newey_west_ols tự chọn).
        periods_per_year: Số chu kỳ nến mỗi năm (mặc định 8760).

    Returns:
        Dict gồm: alpha, beta, alpha_se, beta_se, alpha_t, beta_t, n, lags, alpha_annualized_pct.
    """
    if strategy_returns is None or benchmark_returns is None:
        raise ValueError("strategy_returns và benchmark_returns không được là None")

    common_idx = strategy_returns.index.intersection(benchmark_returns.index).sort_values()
    s_aligned = strategy_returns.loc[common_idx]
    b_aligned = benchmark_returns.loc[common_idx]

    df = pd.DataFrame({"y": s_aligned, "x": b_aligned}).dropna()
    if len(df) < 2:
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "alpha_se": 0.0,
            "beta_se": 0.0,
            "alpha_t": 0.0,
            "beta_t": 0.0,
            "n": len(df),
            "lags": 0,
            "alpha_annualized_pct": 0.0,
        }

    y_arr = df["y"].to_numpy(dtype=float)
    x_arr = df["x"].to_numpy(dtype=float)
    res = newey_west_ols(y=y_arr, x=x_arr, lags=lags)
    res["alpha_annualized_pct"] = float(res["alpha"] * periods_per_year * 100.0)
    return res


def stitch_returns(return_series_list: list[pd.Series]) -> pd.Series:
    """Nối các chuỗi lợi nhuận OOS thành một chuỗi liên tục (không reset vốn ở biên).

    pd.concat, sort_index tăng dần, loại bỏ index trùng lặp (giữ bản cuối cùng).
    Đảm bảo index là DatetimeIndex UTC đơn vị ns.

    Parameters:
        return_series_list: Danh sách các pd.Series tỷ suất lợi nhuận.

    Returns:
        pd.Series được ghép nối liên tục.
    """
    valid_list = [s for s in return_series_list if s is not None and len(s) > 0]
    if not valid_list:
        empty_idx = pd.DatetimeIndex([], tz="UTC").as_unit("ns")
        return pd.Series([], dtype="float64", index=empty_idx)

    combined = pd.concat(valid_list)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    if isinstance(combined.index, pd.DatetimeIndex):
        if combined.index.tz is None:
            combined.index = combined.index.tz_localize("UTC")
        else:
            combined.index = combined.index.tz_convert("UTC")
        combined.index = combined.index.as_unit("ns")

    return combined


def evaluate_go_no_go(metrics: dict, direction: str) -> dict:
    """Đánh giá 4 điều kiện nghiệm thu GO / NO-GO Sprint 6 cho hướng Carry hoặc Cross-Sectional.

    Tiêu chí khóa ở docs/go_no_go_sprint6.md:
    1. Market-neutral: alpha_t >= 2.50 VÀ |beta| <= 0.20.
    2. Sharpe năm hóa: >= 1.50 (Carry) hoặc >= 1.00 (Cross-Sectional).
    3. Deflated Sharpe Ratio (DSR): >= 0.95.
    4. Tập Holdout 180 ngày: holdout_return_pct > 0 VÀ holdout_sharpe > 0.80
       VÀ holdout_max_dd <= 12.0 (Carry) hoặc <= 20.0 (Cross-Sectional).
    verdict = 'GO' nếu cả 4 điều kiện đạt, ngược lại 'NO-GO'.

    Parameters:
        metrics: Dict chứa các chỉ số hiệu suất.
        direction: Hướng nghiên cứu ('carry' hoặc 'xs').

    Returns:
        Dict gồm 4 cờ boolean và 'verdict' ('GO'/'NO-GO').
    """
    if direction not in ("carry", "xs"):
        raise ValueError(f"direction phải là 'carry' hoặc 'xs', nhận: '{direction}'")

    alpha_t = float(metrics.get("alpha_t", 0.0))
    beta = float(metrics.get("beta", 1.0))
    sharpe = float(metrics.get("sharpe", metrics.get("Sharpe", 0.0)))
    dsr_prob = float(metrics.get("dsr_prob", 0.0))

    holdout_return_pct = float(
        metrics.get("holdout_return_pct", metrics.get("holdout_return", 0.0))
    )
    holdout_sharpe = float(metrics.get("holdout_sharpe", 0.0))
    holdout_max_dd = float(
        metrics.get("holdout_max_dd", metrics.get("holdout_max_dd_pct", 999.0))
    )

    cond1 = bool(alpha_t >= 2.50 and abs(beta) <= 0.20)
    sharpe_thresh = 1.50 if direction == "carry" else 1.00
    cond2 = bool(sharpe >= sharpe_thresh)
    cond3 = bool(dsr_prob >= 0.95)
    max_dd_thresh = 12.0 if direction == "carry" else 20.0
    cond4 = bool(
        holdout_return_pct > 0.0
        and holdout_sharpe > 0.80
        and holdout_max_dd <= max_dd_thresh
    )

    verdict = "GO" if (cond1 and cond2 and cond3 and cond4) else "NO-GO"

    return GoNoGoResult({
        "cond1_market_neutral": cond1,
        "cond2_sharpe": cond2,
        "cond3_dsr": cond3,
        "cond4_holdout": cond4,
        "verdict": verdict,
    })
