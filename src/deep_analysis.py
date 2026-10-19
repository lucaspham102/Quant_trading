"""Deep Analysis Engine: Chẩn đoán chiến lược baseline 1h so với Buy-and-Hold.

Module này cung cấp các phân tích định lượng chuyên sâu cho Sprint 5 - Phần A:
- A1: Phơi nhiễm thị trường (Exposure Metrics).
- A2: Mức sụt giảm tối đa theo từng cửa sổ Walk-Forward (Drawdown Analysis).
- A3: Alpha và Beta theo mô hình hồi quy Newey-West HAC trên toàn bộ dữ liệu.
- A4: Block Bootstrap theo ngày kiểm định chênh lệch Sharpe ratio (CI 95%).
- A5: Exposure-Matched Null Hypothesis Test (1000 lần lặp hoán vị).
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.backtester import Backtester, TradeRecord
from src.config import TradingConfig, candles_per_year_for_timeframe
from src.features import add_all_features
from src.research import DEFAULT_SYMBOLS, load_or_fetch_symbol_data
from src.walkforward import (
    Candidate,
    _slice_window,
    apply_fee_scenario,
    generate_walkforward_windows,
    make_candidate_config,
)

REAL_TAKER_FEE = 0.0005
REAL_SLIPPAGE = 0.0002
REAL_FEE_PER_SIDE = 0.0007
DEFAULT_BASELINE = Candidate("breakout", "market", True)


def newey_west_ols(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """Ước lượng OLS y = alpha + beta * x kèm sai số chuẩn vững Newey-West HAC.

    Parameters:
        y: Mảng lợi suất phụ thuộc (ví dụ: strategy return).
        x: Mảng lợi suất độc lập (ví dụ: benchmark return).
        lags: Số bậc trễ Bartlett kernel (mặc định theo công thức Newey-West).

    Returns:
        Dict chứa alpha, beta, alpha_se, beta_se, alpha_t, beta_t, n, lags.
    """
    y_arr = np.asarray(y, dtype=float).flatten()
    x_arr = np.asarray(x, dtype=float).flatten()
    n = len(y_arr)
    if n != len(x_arr):
        raise ValueError(f"Độ dài y ({n}) và x ({len(x_arr)}) phải bằng nhau.")
    if n < 2:
        raise ValueError(f"Số lượng quan sát phải >= 2, nhận: {n}")

    x_mat = np.column_stack([np.ones(n, dtype=float), x_arr])
    xtx = x_mat.T @ x_mat
    try:
        inv_xtx = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        inv_xtx = np.linalg.pinv(xtx)

    beta_hat = inv_xtx @ (x_mat.T @ y_arr)
    alpha = float(beta_hat[0])
    beta = float(beta_hat[1])

    residuals = y_arr - (x_mat @ beta_hat)

    # Lựa chọn số lags mặc định theo công thức Newey-West (1994)
    if lags is None:
        l_lag = max(1, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    else:
        l_lag = max(0, int(lags))
    l_lag = min(l_lag, n - 1)

    # Ma trận hiệp phương sai sai số HAC với trọng số Bartlett
    u_mat = x_mat * residuals[:, np.newaxis]
    gamma_0 = u_mat.T @ u_mat
    s_mat = gamma_0.copy()

    for l_idx in range(1, l_lag + 1):
        gamma_l = u_mat[l_idx:].T @ u_mat[:-l_idx]
        weight = 1.0 - (l_idx / (l_lag + 1.0))
        s_mat += weight * (gamma_l + gamma_l.T)

    var_cov = inv_xtx @ s_mat @ inv_xtx
    diag_var = np.diag(var_cov)
    se = np.sqrt(np.maximum(diag_var, 0.0))

    alpha_se = float(se[0])
    beta_se = float(se[1])
    alpha_t = float(alpha / alpha_se) if alpha_se > 1e-12 else 0.0
    beta_t = float(beta / beta_se) if beta_se > 1e-12 else 0.0

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_se": alpha_se,
        "beta_se": beta_se,
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "n": int(n),
        "lags": int(l_lag),
    }


def max_drawdown_pct(equity: pd.Series | Sequence[float]) -> float:
    """Tính mức sụt giảm vốn tối đa (Max Drawdown) theo phần trăm (số dương >= 0)."""
    if len(equity) == 0:
        return 0.0
    vals = np.asarray(equity, dtype=float)
    running_max = np.maximum.accumulate(vals)
    valid = running_max > 0
    if not np.any(valid):
        return 0.0
    dd = np.zeros_like(vals)
    dd[valid] = (running_max[valid] - vals[valid]) / running_max[valid] * 100.0
    return float(np.nanmax(dd))


def exposure_metrics(
    trades: Sequence,
    index: pd.DatetimeIndex,
    equity: pd.Series,
    initial_capital: float,
) -> dict:
    """Tính toán các chỉ số phơi nhiễm thị trường của chiến lược.

    Returns:
        Dict gồm time_in_market_pct, avg_notional_to_equity, long_ratio,
        avg_holding_bars, n_trades.
    """
    n_trades = len(trades)
    if len(index) == 0 or n_trades == 0:
        return {
            "time_in_market_pct": 0.0,
            "avg_notional_to_equity": 0.0,
            "long_ratio": 0.0,
            "avg_holding_bars": 0.0,
            "n_trades": int(n_trades),
        }

    in_market = np.zeros(len(index), dtype=bool)
    notional_ratios: list[float] = []
    holding_hours: list[float] = []
    long_count = 0

    for tr in trades:
        if getattr(tr, "side", 1) == 1:
            long_count += 1

        t0 = pd.Timestamp(tr.entry_time)
        t1 = pd.Timestamp(tr.exit_time)

        if index.tz is not None and t0.tzinfo is None:
            t0 = t0.tz_localize(index.tz)
        elif index.tz is None and t0.tzinfo is not None:
            t0 = t0.tz_localize(None)

        if index.tz is not None and t1.tzinfo is None:
            t1 = t1.tz_localize(index.tz)
        elif index.tz is None and t1.tzinfo is not None:
            t1 = t1.tz_localize(None)

        i0 = int(index.searchsorted(t0))
        i1 = int(index.searchsorted(t1))
        if i0 < i1:
            in_market[i0:min(i1, len(index))] = True

        hours = (t1 - t0).total_seconds() / 3600.0
        holding_hours.append(hours)

        notional = float(tr.qty * tr.entry_price)
        eq_entry = float(initial_capital)
        if equity is not None and not equity.empty:
            eq_idx = t0
            if equity.index.tz is not None and eq_idx.tzinfo is None:
                eq_idx = eq_idx.tz_localize(equity.index.tz)
            elif equity.index.tz is None and eq_idx.tzinfo is not None:
                eq_idx = eq_idx.tz_localize(None)

            if eq_idx in equity.index:
                eq_val = float(equity.loc[eq_idx])
                if np.isfinite(eq_val) and eq_val > 0:
                    eq_entry = eq_val

        notional_ratios.append(notional / eq_entry)

    time_in_market_pct = float(np.sum(in_market) / len(index) * 100.0)
    avg_notional = float(np.mean(notional_ratios)) if notional_ratios else 0.0
    long_ratio = float(long_count / n_trades) if n_trades > 0 else 0.0
    avg_holding = float(np.mean(holding_hours)) if holding_hours else 0.0

    return {
        "time_in_market_pct": time_in_market_pct,
        "avg_notional_to_equity": avg_notional,
        "long_ratio": long_ratio,
        "avg_holding_bars": avg_holding,
        "n_trades": int(n_trades),
    }


def block_bootstrap_sharpe_diff(
    r_base: pd.Series,
    r_bh: pd.Series,
    candles_per_year: int,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Kiểm định bootstrap theo block ngày cho chênh lệch Sharpe giữa Baseline và Buy-and-Hold.

    Returns:
        Dict gồm mean_diff, ci_low (p2.5), ci_high (p97.5), p_value, n_days.
    """
    common_idx = r_base.index.intersection(r_bh.index)
    if len(common_idx) < 2:
        return {
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 0.0,
            "n_days": 0,
        }

    rb = r_base.loc[common_idx]
    rh = r_bh.loc[common_idx]

    days_norm = common_idx.normalize()
    _, day_inverse = np.unique(days_norm, return_inverse=True)
    n_days = int(len(np.unique(days_norm)))

    if n_days < 2:
        std_b = float(rb.std(ddof=1))
        std_h = float(rh.std(ddof=1))
        sh_b = float(rb.mean() / std_b * math.sqrt(candles_per_year)) if std_b > 1e-12 else 0.0
        sh_h = float(rh.mean() / std_h * math.sqrt(candles_per_year)) if std_h > 1e-12 else 0.0
        diff = sh_b - sh_h
        return {
            "mean_diff": diff,
            "ci_low": diff,
            "ci_high": diff,
            "p_value": 1.0 if diff <= 0.0 else 0.0,
            "n_days": n_days,
        }

    b_vals = rb.to_numpy(dtype=float)
    h_vals = rh.to_numpy(dtype=float)

    day_blocks_b: list[np.ndarray] = []
    day_blocks_h: list[np.ndarray] = []
    for d_idx in range(n_days):
        mask = day_inverse == d_idx
        day_blocks_b.append(b_vals[mask])
        day_blocks_h.append(h_vals[mask])

    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_iter, dtype=float)
    sqrt_cpy = math.sqrt(candles_per_year)

    for it in range(n_iter):
        sampled = rng.integers(0, n_days, size=n_days)
        sample_b = np.concatenate([day_blocks_b[i] for i in sampled])
        sample_h = np.concatenate([day_blocks_h[i] for i in sampled])

        std_b = float(np.std(sample_b, ddof=1))
        if np.isfinite(std_b) and std_b > 1e-12:
            sh_b = float(np.mean(sample_b) / std_b * sqrt_cpy)
        else:
            sh_b = 0.0

        std_h = float(np.std(sample_h, ddof=1))
        if np.isfinite(std_h) and std_h > 1e-12:
            sh_h = float(np.mean(sample_h) / std_h * sqrt_cpy)
        else:
            sh_h = 0.0

        diffs[it] = sh_b - sh_h

    mean_diff = float(np.mean(diffs))
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    p_value = float(np.mean(diffs <= 0.0))

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_days": n_days,
    }


def exposure_matched_null(
    close: pd.Series,
    trades: Sequence,
    fee_per_side: float,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Kiểm định Null Hypothesis ngẫu nhiên hóa khớp phân phối thời gian giữ lệnh và phía giao dịch.

    Returns:
        Dict gồm baseline_metric, null_mean, null_p2_5, null_p97_5, percentile, n_iter.
    """
    m_trades = len(trades)
    if m_trades == 0:
        return {
            "baseline_metric": 0.0,
            "null_mean": 0.0,
            "null_p2_5": 0.0,
            "null_p97_5": 0.0,
            "percentile": 0.0,
            "n_iter": int(n_iter),
        }

    close_vals = np.asarray(close, dtype=float)
    n_bars = len(close_vals)
    if n_bars < 2:
        return {
            "baseline_metric": 0.0,
            "null_mean": 0.0,
            "null_p2_5": 0.0,
            "null_p97_5": 0.0,
            "percentile": 0.0,
            "n_iter": int(n_iter),
        }

    holding_bars_list: list[int] = []
    sides_list: list[int] = []
    real_returns: list[float] = []

    for tr in trades:
        side = int(tr.side)
        sides_list.append(side)
        ret_i = side * (float(tr.exit_price) / float(tr.entry_price) - 1.0) - 2.0 * fee_per_side
        real_returns.append(ret_i)

        t0 = tr.entry_time
        t1 = tr.exit_time
        if isinstance(t0, (pd.Timestamp, str)) or hasattr(t0, "total_seconds"):
            t0 = pd.Timestamp(t0)
            t1 = pd.Timestamp(t1)
            delta_sec = (t1 - t0).total_seconds()
            bar_sec = 3600.0
            if isinstance(close.index, pd.DatetimeIndex) and len(close.index) > 1:
                diff = (close.index[1] - close.index[0]).total_seconds()
                if diff > 0:
                    bar_sec = diff
            bars = max(1, int(round(delta_sec / bar_sec)))
        else:
            try:
                bars = max(1, int(t1 - t0))
            except Exception:
                bars = 1
        holding_bars_list.append(bars)

    baseline_metric = float(np.mean(real_returns))
    holding_bars = np.array(holding_bars_list, dtype=int)
    sides = np.array(sides_list, dtype=int)

    rng = np.random.default_rng(seed)

    entry_cols = []
    for i in range(m_trades):
        h = min(int(holding_bars[i]), n_bars - 1)
        max_entry = max(1, n_bars - h)
        entry_cols.append(rng.integers(0, max_entry, size=n_iter))

    entries = np.column_stack(entry_cols)
    exits = np.minimum(entries + holding_bars[np.newaxis, :], n_bars - 1)
    sampled_sides = rng.choice(sides, size=(n_iter, m_trades))

    p_entry = close_vals[entries]
    p_exit = close_vals[exits]
    safe_p_entry = np.where(p_entry > 0, p_entry, 1.0)
    sim_returns = sampled_sides * (p_exit / safe_p_entry - 1.0) - 2.0 * fee_per_side
    null_metrics = np.mean(sim_returns, axis=1)

    null_mean = float(np.mean(null_metrics))
    null_p2_5 = float(np.percentile(null_metrics, 2.5))
    null_p97_5 = float(np.percentile(null_metrics, 97.5))
    percentile = float(100.0 * np.mean(null_metrics <= baseline_metric))

    return {
        "baseline_metric": baseline_metric,
        "null_mean": null_mean,
        "null_p2_5": null_p2_5,
        "null_p97_5": null_p97_5,
        "percentile": percentile,
        "n_iter": int(n_iter),
    }


def run_deep_analysis(
    symbol: str,
    timeframe: str = "1h",
    days: int = 1260,
    train_days: int = 120,
    test_days: int = 60,
    step_days: int = 60,
    buffer_days: int = 15,
    baseline: Candidate | None = None,
    n_boot: int = 1000,
    n_null: int = 1000,
    seed: int = 42,
    raw_data_dir: Path = Path("data/raw"),
    df: pd.DataFrame | None = None,
    exchange=None,
) -> dict[str, pd.DataFrame]:
    """Thực thi toàn bộ bộ chẩn đoán chuyên sâu Sprint 5 Phần A cho một symbol.

    Returns:
        Dict gồm 5 DataFrame: exposure, drawdown, alpha_beta, bootstrap, null.
    """
    if baseline is None:
        baseline = DEFAULT_BASELINE

    if df is None:
        df = load_or_fetch_symbol_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days + buffer_days,
            raw_data_dir=raw_data_dir,
            exchange=exchange,
        )

    base_common = dataclasses.replace(
        TradingConfig(),
        use_fee_aware_sizing=True,
        min_stop_distance_pct=0.008,
        symbol=symbol,
        timeframe=timeframe,
    )
    feat = add_all_features(df, base_common, drop_warmup=True)
    if len(feat) == 0:
        return {
            "exposure": pd.DataFrame(
                columns=[
                    "symbol",
                    "timeframe",
                    "n_trades",
                    "time_in_market_pct",
                    "avg_notional_to_equity",
                    "long_ratio",
                    "avg_holding_bars",
                ]
            ),
            "drawdown": pd.DataFrame(
                columns=[
                    "symbol",
                    "window_index",
                    "test_start",
                    "test_end",
                    "base_zero_dd",
                    "base_real_dd",
                    "bh_dd",
                ]
            ),
            "alpha_beta": pd.DataFrame(
                columns=[
                    "symbol",
                    "timeframe",
                    "n_obs",
                    "lags",
                    "alpha_per_bar",
                    "alpha_annualized_pct",
                    "beta",
                    "alpha_t",
                    "beta_t",
                ]
            ),
            "bootstrap": pd.DataFrame(
                columns=[
                    "symbol",
                    "timeframe",
                    "mean_sharpe_diff",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "n_days",
                ]
            ),
            "null": pd.DataFrame(
                columns=[
                    "symbol",
                    "timeframe",
                    "baseline_metric",
                    "null_mean",
                    "null_p2_5",
                    "null_p97_5",
                    "percentile",
                    "n_iter",
                ]
            ),
        }

    cpy = candles_per_year_for_timeframe(timeframe)
    initial = TradingConfig().initial_capital

    cfg_zero = make_candidate_config(apply_fee_scenario(base_common, "zero"), baseline, timeframe)
    rep_zero = Backtester(cfg_zero).run(feat)

    cfg_real = make_candidate_config(apply_fee_scenario(base_common, "real"), baseline, timeframe)
    rep_real = Backtester(cfg_real).run(feat)

    eq_bh = initial * feat["close"] / feat["open"].iloc[0]
    r_bh = eq_bh.pct_change().fillna(0.0)

    # 1. Exposure DataFrame
    exp_res = exposure_metrics(rep_real.trades, feat.index, rep_real.equity_curve, initial)
    exposure_df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "n_trades": rep_real.total_trades,
                "time_in_market_pct": exp_res["time_in_market_pct"],
                "avg_notional_to_equity": exp_res["avg_notional_to_equity"],
                "long_ratio": exp_res["long_ratio"],
                "avg_holding_bars": exp_res["avg_holding_bars"],
            }
        ]
    )

    # 2. Drawdown DataFrame
    windows = generate_walkforward_windows(
        feat.index, train_days=train_days, test_days=test_days, step_days=step_days
    )
    dd_rows = []
    for w in windows:
        test_frame = _slice_window(feat, w.test_start, w.test_end)
        w_rep_zero = Backtester(cfg_zero).run(test_frame)
        w_rep_real = Backtester(cfg_real).run(test_frame)
        w_eq_bh = eq_bh.loc[test_frame.index]
        dd_rows.append(
            {
                "symbol": symbol,
                "window_index": w.window_index,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "base_zero_dd": max_drawdown_pct(w_rep_zero.equity_curve),
                "base_real_dd": max_drawdown_pct(w_rep_real.equity_curve),
                "bh_dd": max_drawdown_pct(w_eq_bh),
            }
        )

    dd_rows.append(
        {
            "symbol": symbol,
            "window_index": -1,
            "test_start": feat.index[0],
            "test_end": feat.index[-1],
            "base_zero_dd": float(rep_zero.max_drawdown_pct),
            "base_real_dd": float(rep_real.max_drawdown_pct),
            "bh_dd": max_drawdown_pct(eq_bh),
        }
    )
    drawdown_df = pd.DataFrame(dd_rows)

    # 3. Alpha & Beta DataFrame
    r_base = rep_real.equity_curve.pct_change().fillna(0.0)
    common_idx = r_base.index.intersection(r_bh.index)
    rb_aligned = r_base.loc[common_idx].iloc[1:]
    rh_aligned = r_bh.loc[common_idx].iloc[1:]

    if len(rb_aligned) >= 2:
        ols_res = newey_west_ols(rb_aligned.values, rh_aligned.values)
        alpha_per_bar = ols_res["alpha"]
        alpha_ann = alpha_per_bar * cpy * 100.0
        alpha_beta_df = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "n_obs": ols_res["n"],
                    "lags": ols_res["lags"],
                    "alpha_per_bar": alpha_per_bar,
                    "alpha_annualized_pct": alpha_ann,
                    "beta": ols_res["beta"],
                    "alpha_t": ols_res["alpha_t"],
                    "beta_t": ols_res["beta_t"],
                }
            ]
        )
    else:
        alpha_beta_df = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "n_obs": len(rb_aligned),
                    "lags": 0,
                    "alpha_per_bar": 0.0,
                    "alpha_annualized_pct": 0.0,
                    "beta": 0.0,
                    "alpha_t": 0.0,
                    "beta_t": 0.0,
                }
            ]
        )

    # 4. Block Bootstrap DataFrame
    boot_res = block_bootstrap_sharpe_diff(r_base, r_bh, cpy, n_iter=n_boot, seed=seed)
    bootstrap_df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "mean_sharpe_diff": boot_res["mean_diff"],
                "ci_low": boot_res["ci_low"],
                "ci_high": boot_res["ci_high"],
                "p_value": boot_res["p_value"],
                "n_days": boot_res["n_days"],
            }
        ]
    )

    # 5. Null Hypothesis DataFrame
    null_res = exposure_matched_null(
        feat["close"], rep_real.trades, REAL_FEE_PER_SIDE, n_iter=n_null, seed=seed
    )
    null_df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "baseline_metric": null_res["baseline_metric"],
                "null_mean": null_res["null_mean"],
                "null_p2_5": null_res["null_p2_5"],
                "null_p97_5": null_res["null_p97_5"],
                "percentile": null_res["percentile"],
                "n_iter": null_res["n_iter"],
            }
        ]
    )

    return {
        "exposure": exposure_df,
        "drawdown": drawdown_df,
        "alpha_beta": alpha_beta_df,
        "bootstrap": bootstrap_df,
        "null": null_df,
    }


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample dữ liệu OHLCV 1h sang khung thời gian lớn hơn (chỉ hỗ trợ '4h' và '1d').

    Parameters:
        df: DataFrame OHLCV gốc (khung 1h) có DatetimeIndex.
        timeframe: Chuỗi khung đích, '4h' hoặc '1d'.

    Returns:
        DataFrame OHLCV đã resample, loại bỏ các bucket rỗng, index DatetimeIndex nano-second.
    """
    if timeframe not in ("4h", "1d"):
        raise ValueError(
            f"Khung thời gian không hợp lệ: {timeframe!r}. Chỉ hỗ trợ '4h' và '1d'."
        )
    if len(df) == 0:
        out = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz=df.index.tz, name=df.index.name).as_unit("ns"),
        )
        if hasattr(df, "attrs") and df.attrs:
            out.attrs = df.attrs.copy()
        return out

    rule = "4h" if timeframe == "4h" else "1D"
    out = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index = out.index.as_unit("ns")
    if hasattr(df, "attrs") and df.attrs:
        out.attrs = df.attrs.copy()
    return out


def sharpe_of(equity: pd.Series, candles_per_year: int) -> float:
    """Tính tỷ số Sharpe hàng năm hóa từ chuỗi vốn equity.

    Parameters:
        equity: Chuỗi giá trị vốn qua từng nến.
        candles_per_year: Số lượng nến trong 1 năm theo timeframe tương ứng.

    Returns:
        Tỷ số Sharpe dạng float (0.0 nếu chuỗi vốn < 2 phần tử hoặc độ lệch chuẩn bằng 0).
    """
    if len(equity) < 2:
        return 0.0
    r = equity.pct_change().dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(r.mean() / std * math.sqrt(candles_per_year))


def analyze_timeframe(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    train_days: int = 120,
    test_days: int = 60,
    step_days: int = 60,
    baseline: Candidate | None = None,
    n_boot: int = 1000,
    n_null: int = 1000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Phân tích chuyên sâu chiến lược baseline cho một khung thời gian cụ thể kèm so sánh Sharpe với Buy-and-Hold.

    Parameters:
        symbol: Tên cặp giao dịch (ví dụ 'BTC/USDT:USDT').
        timeframe: Khung thời gian ('4h' hoặc '1d').
        df: DataFrame OHLCV của khung thời gian tương ứng.
        train_days: Số ngày trong tập train walk-forward.
        test_days: Số ngày trong tập test walk-forward.
        step_days: Số ngày bước trượt walk-forward.
        baseline: Cấu hình chiến lược baseline (mặc định DEFAULT_BASELINE).
        n_boot: Số lần lặp Bootstrap.
        n_null: Số lần lặp kiểm định Null.
        seed: Seed sinh số ngẫu nhiên.

    Returns:
        Dict gồm 6 DataFrame: exposure, drawdown, alpha_beta, bootstrap, null, sharpe.
    """
    if baseline is None:
        baseline = DEFAULT_BASELINE

    out = run_deep_analysis(
        symbol=symbol,
        timeframe=timeframe,
        df=df,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        baseline=baseline,
        n_boot=n_boot,
        n_null=n_null,
        seed=seed,
    )

    base_common = dataclasses.replace(
        TradingConfig(),
        use_fee_aware_sizing=True,
        min_stop_distance_pct=0.008,
        symbol=symbol,
        timeframe=timeframe,
    )
    feat = add_all_features(df, base_common, drop_warmup=True)
    if len(feat) == 0:
        sharpe_df = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "n_bars": len(df),
                    "base_sharpe": 0.0,
                    "bh_sharpe": 0.0,
                    "base_minus_bh_sharpe": 0.0,
                    "base_return_pct": 0.0,
                    "bh_return_pct": 0.0,
                    "base_max_dd": 0.0,
                    "bh_max_dd": 0.0,
                }
            ]
        )
    else:
        cfg_real = make_candidate_config(apply_fee_scenario(base_common, "real"), baseline, timeframe)
        rep_real = Backtester(cfg_real).run(feat)
        cpy = candles_per_year_for_timeframe(timeframe)
        initial = base_common.initial_capital
        open_0 = float(feat["open"].iloc[0])
        eq_bh = initial * feat["close"] / open_0 if open_0 > 0 else pd.Series(initial, index=feat.index)
        bh_sharpe = sharpe_of(eq_bh, cpy)
        base_sharpe = float(rep_real.sharpe_ratio)
        base_minus_bh_sharpe = float(base_sharpe - bh_sharpe)
        base_return_pct = float(rep_real.total_return_pct)
        close_end = float(feat["close"].iloc[-1])
        bh_return_pct = float((close_end / open_0 - 1.0) * 100.0) if open_0 > 0 else 0.0
        base_max_dd = float(rep_real.max_drawdown_pct)
        bh_max_dd = float(max_drawdown_pct(eq_bh))

        sharpe_df = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "n_bars": len(df),
                    "base_sharpe": base_sharpe,
                    "bh_sharpe": bh_sharpe,
                    "base_minus_bh_sharpe": base_minus_bh_sharpe,
                    "base_return_pct": base_return_pct,
                    "bh_return_pct": bh_return_pct,
                    "base_max_dd": base_max_dd,
                    "bh_max_dd": bh_max_dd,
                }
            ]
        )

    out["sharpe"] = sharpe_df
    return out


def holdout_return(
    symbol: str,
    timeframe: str,
    context_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    baseline: Candidate | None = None,
    fee_scenario: str = "real",
) -> dict:
    """Đánh giá hiệu suất trên tập dữ liệu Holdout biệt lập (180 ngày cuối).

    Hàm nhận TOÀN BỘ chuỗi ``context_df`` để tính chỉ báo (warm-up) rồi CHỈ
    giao dịch trên phần từ ``cutoff`` trở đi, tránh warm-up ăn mất đoạn holdout
    (đặc biệt quan trọng với khung 1d nơi warm-up ~200 nến).

    Parameters:
        symbol: Tên cặp giao dịch (ví dụ 'BTC/USDT:USDT').
        timeframe: Khung thời gian ('4h' hoặc '1d').
        context_df: DataFrame OHLCV ĐẦY ĐỦ (đã resample) để tính chỉ báo.
        cutoff: Mốc thời gian bắt đầu holdout; chỉ giao dịch từ cutoff trở đi.
        baseline: Cấu hình baseline (mặc định DEFAULT_BASELINE).
        fee_scenario: Kịch bản phí ('real', 'zero', 'double').

    Returns:
        Dict gồm các chỉ số holdout của baseline và buy-and-hold.
    """
    if baseline is None:
        baseline = DEFAULT_BASELINE

    base_common = dataclasses.replace(
        TradingConfig(),
        use_fee_aware_sizing=True,
        min_stop_distance_pct=0.008,
        symbol=symbol,
        timeframe=timeframe,
    )
    feat_full = add_all_features(context_df, base_common, drop_warmup=True)
    feat = feat_full.loc[feat_full.index >= cutoff]
    if len(feat) == 0:
        h_end = context_df.index[-1] if len(context_df) > 0 else pd.Timestamp(0, tz="UTC")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "n_bars": 0,
            "holdout_start": cutoff,
            "holdout_end": h_end,
            "baseline_return_pct": 0.0,
            "baseline_sharpe": 0.0,
            "baseline_max_dd": 0.0,
            "bh_return_pct": 0.0,
            "bh_max_dd": 0.0,
        }

    cfg = make_candidate_config(apply_fee_scenario(base_common, fee_scenario), baseline, timeframe)
    rep = Backtester(cfg).run(feat)
    initial = base_common.initial_capital
    open_0 = float(feat["open"].iloc[0])
    eq_bh = initial * feat["close"] / open_0 if open_0 > 0 else pd.Series(initial, index=feat.index)
    close_end = float(feat["close"].iloc[-1])
    bh_ret = float((close_end / open_0 - 1.0) * 100.0) if open_0 > 0 else 0.0

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "n_bars": len(feat),
        "holdout_start": feat.index[0],
        "holdout_end": feat.index[-1],
        "baseline_return_pct": float(rep.total_return_pct),
        "baseline_sharpe": float(rep.sharpe_ratio),
        "baseline_max_dd": float(rep.max_drawdown_pct),
        "bh_return_pct": bh_ret,
        "bh_max_dd": float(max_drawdown_pct(eq_bh)),
    }


__all__ = [
    "REAL_TAKER_FEE",
    "REAL_SLIPPAGE",
    "REAL_FEE_PER_SIDE",
    "DEFAULT_BASELINE",
    "newey_west_ols",
    "max_drawdown_pct",
    "exposure_metrics",
    "block_bootstrap_sharpe_diff",
    "exposure_matched_null",
    "run_deep_analysis",
    "resample_ohlcv",
    "sharpe_of",
    "analyze_timeframe",
    "holdout_return",
]

