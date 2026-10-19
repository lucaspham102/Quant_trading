"""Động cơ chiến lược Cross-Sectional Market-Neutral đa tài sản.

Bao gồm:
- CrossSectionalConfig: Cấu hình tham số chiến lược, Top-K, buffer band, carry tilt, vol targeting và mô hình chi phí.
- run_cross_sectional_portfolio: Mô phỏng danh mục Cross-Sectional Market-Neutral (momentum/reversal)
  hoàn toàn không có look-ahead bias, tính phí trên turnover, hạch toán funding rate, hỗ trợ Top-K/Bottom-K
  có buffer band chống đảo lệnh và vol targeting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CrossSectionalConfig:
    """Cấu hình tham số cho chiến lược Cross-Sectional Market-Neutral."""

    mode: str = "momentum"
    lookback_hours: int = 168
    rebalance_hours: int = 24
    min_spread: float = 0.5
    gross_leverage: float = 1.0
    perp_fee: float = 0.0005
    slippage: float = 0.0002
    initial_capital: float = 400.0
    top_k: int = 1  # số mã mỗi bên (mặc định 1 để tương thích ngược)
    buffer_rank: int = 0
    carry_tilt_weight: float = 0.0  # lambda >= 0
    vol_target_ann: float | None = None
    carry_lookback_periods: int = 21  # số kỳ funding trung bình (7 ngày * 3 kỳ)
    funding_col: str = "last_known_funding_rate"  # fallback sang funding_rate_event nếu thiếu cột


def run_cross_sectional_portfolio(
    aligned_data: dict[str, pd.DataFrame],
    config: CrossSectionalConfig,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """Mô phỏng danh mục Cross-Sectional Market-Neutral trên rổ coin.

    Tham số:
        aligned_data: Dict map {symbol: DataFrame có tối thiểu cột close và funding_rate_event}.
        config: CrossSectionalConfig chứa tham số chiến lược, Top-K, buffer band, carry tilt, vol targeting và chi phí.
        periods_per_year: Số chu kỳ nến mỗi năm dùng để năm hóa Sharpe và Return (mặc định 8760 cho nến 1h).

    Trả về:
        Dict gồm:
        - equity_curve: pd.Series đường cong vốn trên trục thời gian chung.
        - returns: pd.Series tỷ suất lợi nhuận R_port,t tại từng nến.
        - weights_df: pd.DataFrame trọng số đã thiết lập tại từng nến (index: DatetimeIndex, columns: symbol).
        - trades_df: pd.DataFrame nhật ký đổi trọng số (time, symbol, w_old, w_new, turnover).
        - metrics: Dict các chỉ số hiệu suất tổng hợp.
    """
    if config.mode not in ("momentum", "reversal"):
        raise ValueError(
            f"Mode '{config.mode}' không hợp lệ. Chỉ chấp nhận 'momentum' hoặc 'reversal'."
        )
    if config.lookback_hours < 1:
        raise ValueError(f"lookback_hours phải >= 1, nhận: {config.lookback_hours}")
    if config.rebalance_hours < 1:
        raise ValueError(f"rebalance_hours phải >= 1, nhận: {config.rebalance_hours}")
    if config.gross_leverage <= 0:
        raise ValueError(f"gross_leverage phải > 0, nhận: {config.gross_leverage}")
    if config.initial_capital <= 0:
        raise ValueError(f"initial_capital phải > 0, nhận: {config.initial_capital}")
    if config.top_k < 1:
        raise ValueError(f"top_k phải >= 1, nhận: {config.top_k}")
    if config.buffer_rank < 0:
        raise ValueError(f"buffer_rank phải >= 0, nhận: {config.buffer_rank}")
    if config.carry_tilt_weight < 0:
        raise ValueError(
            f"carry_tilt_weight phải >= 0, nhận: {config.carry_tilt_weight}"
        )
    if config.vol_target_ann is not None and config.vol_target_ann <= 0:
        raise ValueError(
            f"vol_target_ann nếu khác None phải > 0, nhận: {config.vol_target_ann}"
        )

    empty_idx = pd.DatetimeIndex([], tz="UTC", name="datetime").as_unit("ns")
    empty_metrics = {
        "Total Return %": 0.0,
        "Total Return % ": 0.0,
        "Annualized Return %": 0.0,
        "Sharpe": 0.0,
        "MaxDD %": 0.0,
        "Turnover": 0.0,
        "Time in Market %": 0.0,
        "Total Funding Paid": 0.0,
        "Total Funding Collected": 0.0,
        "Total Fees Paid": 0.0,
        "Avg Top-K Turnover": 0.0,
        "Num Rebalances": 0,
        "Vol Target Scale Mean": 0.0,
    }

    if not aligned_data:
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=empty_idx, dtype="float64", name="returns"),
            "weights_df": pd.DataFrame(index=empty_idx),
            "trades_df": pd.DataFrame(
                columns=["time", "symbol", "w_old", "w_new", "turnover"]
            ),
            "metrics": empty_metrics,
        }

    # Chuẩn hóa DatetimeIndex UTC ns tên 'datetime' cho từng symbol
    clean_aligned: dict[str, pd.DataFrame] = {}
    for sym, df in aligned_data.items():
        if "close" not in df.columns:
            raise ValueError(f"Symbol '{sym}' thiếu cột bắt buộc 'close'.")
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            if "datetime" in d.columns:
                d = d.set_index("datetime")
        idx = pd.DatetimeIndex(d.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        idx = idx.as_unit("ns")
        idx.name = "datetime"
        d.index = idx
        clean_aligned[sym] = d

    symbols = sorted(clean_aligned.keys())
    n_symbols = len(symbols)
    if n_symbols < 2:
        raise ValueError(
            f"Chiến lược Cross-Sectional yêu cầu tối thiểu 2 symbol, nhận: {n_symbols}"
        )
    if config.top_k * 2 > n_symbols:
        raise ValueError(
            f"top_k * 2 ({config.top_k * 2}) vượt quá số lượng symbol ({n_symbols})."
        )

    # Trục thời gian chung: union index của các symbol, sort tăng dần
    all_ts = sorted(set().union(*(d.index for d in clean_aligned.values())))
    common_index = pd.DatetimeIndex(all_ts, tz="UTC").as_unit("ns")
    common_index.name = "datetime"

    n_steps = len(common_index)
    if n_steps == 0:
        empty_eq = pd.Series([], index=common_index, dtype="float64", name="equity")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=common_index, dtype="float64", name="returns"),
            "weights_df": pd.DataFrame(index=common_index, columns=symbols),
            "trades_df": pd.DataFrame(
                columns=["time", "symbol", "w_old", "w_new", "turnover"]
            ),
            "metrics": empty_metrics,
        }

    lookback = config.lookback_hours

    # Trích xuất ma trận giá đóng cửa và funding rate event
    # close_matrix: reindex common_index và ffill tối đa
    close_matrix = np.empty((n_steps, n_symbols), dtype=np.float64)
    funding_matrix = np.zeros((n_steps, n_symbols), dtype=np.float64)
    score_matrix = np.full((n_steps, n_symbols), np.nan, dtype=np.float64)
    f_matrix = np.zeros((n_steps, n_symbols), dtype=np.float64)

    for s_idx, sym in enumerate(symbols):
        d = clean_aligned[sym]
        c_series = d["close"].reindex(common_index).ffill()
        close_matrix[:, s_idx] = c_series.to_numpy(dtype=np.float64)

        if "funding_rate_event" in d.columns:
            fr_series = d["funding_rate_event"].reindex(common_index).fillna(0.0)
            funding_matrix[:, s_idx] = fr_series.to_numpy(dtype=np.float64)

        # Tính điểm số M tại nến t CHỈ dùng dữ liệu tới t-1:
        # M = (Close.shift(1)/Close.shift(1+L) - 1) / (rolling_std(ret.shift(1), L) + 1e-8)
        # Sử dụng shift(1) để ngăn chặn tuyệt đối look-ahead bias
        cum_ret = (c_series.shift(1) / c_series.shift(1 + lookback)) - 1.0
        ret_s = c_series.pct_change()
        ret_lag1 = ret_s.shift(1)
        rolling_std = ret_lag1.rolling(window=lookback, min_periods=lookback).std(ddof=1)
        s_series = cum_ret / (rolling_std + 1e-8)
        s_series = s_series.replace([np.inf, -np.inf], np.nan)
        score_matrix[:, s_idx] = s_series.to_numpy(dtype=np.float64)

        # Trích xuất chuỗi funding F cho carry-aware score nếu carry_tilt_weight > 0
        if config.carry_tilt_weight > 0:
            if config.funding_col in d.columns:
                f_raw = d[config.funding_col].reindex(common_index).ffill().fillna(0.0)
            elif "funding_rate_event" in d.columns:
                f_raw = d["funding_rate_event"].reindex(common_index).fillna(0.0)
            else:
                f_raw = pd.Series(0.0, index=common_index)
            f_rolling = (
                f_raw.rolling(window=config.carry_lookback_periods, min_periods=1)
                .mean()
                .shift(1)
            )
            f_matrix[:, s_idx] = f_rolling.to_numpy(dtype=np.float64)

    # Tìm nến đầu tiên đủ lookback (tất cả symbols đều có score không phải NaN)
    valid_score_mask = ~np.isnan(score_matrix).any(axis=1)
    valid_indices = np.where(valid_score_mask)[0]
    first_valid_idx: int | None = int(valid_indices[0]) if len(valid_indices) > 0 else None

    # Khởi tạo mô phỏng
    equity = float(config.initial_capital)
    equity_vals = np.empty(n_steps, dtype=np.float64)
    returns_vals = np.zeros(n_steps, dtype=np.float64)
    weights_matrix = np.zeros((n_steps, n_symbols), dtype=np.float64)

    current_weights = np.zeros(n_symbols, dtype=np.float64)
    total_fees_paid = 0.0
    total_funding_paid = 0.0
    total_turnover = 0.0
    num_rebalances = 0
    rebalance_scales: list[float] = []
    trades_list: list[dict[str, Any]] = []

    # Nến đầu tiên (t = 0): khởi đầu = initial_capital, R = 0
    equity_vals[0] = equity
    returns_vals[0] = 0.0
    weights_matrix[0] = current_weights

    one_way_cost = config.perp_fee + config.slippage

    # Vòng lặp mô phỏng trên trục thời gian chung từ nến t = 1
    for k in range(1, n_steps):
        equity_prev = equity_vals[k - 1]

        # 1. Xác định trọng số mới tại nến k nếu thuộc chu kỳ tái cân bằng
        is_rebalance_candle = (
            first_valid_idx is not None
            and k >= first_valid_idx
            and (k - first_valid_idx) % config.rebalance_hours == 0
        )

        new_weights = current_weights.copy()
        if is_rebalance_candle:
            num_rebalances += 1
            m_k = score_matrix[k]
            # CHỈ khi không có NaN ở bất kỳ symbol nào
            if not np.isnan(m_k).any():
                mean_m = float(np.mean(m_k))
                std_m = float(np.std(m_k, ddof=0))
                z_mom = (m_k - mean_m) / (std_m + 1e-8)

                sign_mode = 1.0 if config.mode == "momentum" else -1.0
                if config.carry_tilt_weight > 0:
                    f_k = f_matrix[k]
                    valid_f = ~np.isnan(f_k)
                    z_fund = np.zeros(n_symbols, dtype=np.float64)
                    if np.any(valid_f):
                        mean_f = float(np.mean(f_k[valid_f]))
                        std_f = float(np.std(f_k[valid_f], ddof=0))
                        z_fund[valid_f] = (f_k[valid_f] - mean_f) / (std_f + 1e-8)
                    scores_k = sign_mode * z_mom - config.carry_tilt_weight * z_fund
                else:
                    scores_k = sign_mode * z_mom

                s_max = float(np.max(scores_k))
                s_min = float(np.min(scores_k))
                spread = s_max - s_min

                target_w = np.zeros(n_symbols, dtype=np.float64)
                if spread >= config.min_spread:
                    # Sắp xếp các symbol theo score giảm dần
                    # Sắp xếp ổn định: ưu tiên điểm số, phụ thuộc s_idx nếu hòa điểm
                    sorted_indices = sorted(
                        range(n_symbols),
                        key=lambda idx: (float(scores_k[idx]), -idx),
                        reverse=True,
                    )
                    rank_of = [0] * n_symbols
                    for r, s_idx in enumerate(sorted_indices, start=1):
                        rank_of[s_idx] = r

                    # Long: giữ các mã thuộc Long kỳ trước nếu rank <= top_k + buffer_rank
                    long_candidates = [
                        s
                        for s in range(n_symbols)
                        if current_weights[s] > 1e-9
                        and rank_of[s] <= (config.top_k + config.buffer_rank)
                    ]
                    long_candidates.sort(key=lambda s: rank_of[s])
                    long_basket = long_candidates[: config.top_k]

                    # Nếu chưa đủ top_k thì lấy thêm từ đầu bảng xếp hạng (bỏ qua mã đã có)
                    if len(long_basket) < config.top_k:
                        for s in sorted_indices:
                            if s not in long_basket:
                                long_basket.append(s)
                                if len(long_basket) == config.top_k:
                                    break

                    # Short: giữ các mã thuộc Short kỳ trước nếu rank_từ_dưới <= top_k + buffer_rank
                    # Không cho 1 mã nằm cả 2 rổ (loại khỏi Short nếu đã nằm trong Long)
                    short_candidates = [
                        s
                        for s in range(n_symbols)
                        if current_weights[s] < -1e-9
                        and (n_symbols - rank_of[s] + 1)
                        <= (config.top_k + config.buffer_rank)
                        and s not in long_basket
                    ]
                    short_candidates.sort(
                        key=lambda s: n_symbols - rank_of[s] + 1
                    )
                    short_basket = short_candidates[: config.top_k]

                    # Nếu chưa đủ thì lấy thêm từ cuối bảng lên (bỏ qua mã đã có / đã nằm trong Long)
                    if len(short_basket) < config.top_k:
                        for s in reversed(sorted_indices):
                            if s not in short_basket and s not in long_basket:
                                short_basket.append(s)
                                if len(short_basket) == config.top_k:
                                    break

                    w_long = config.gross_leverage / (2.0 * config.top_k)
                    w_short = -config.gross_leverage / (2.0 * config.top_k)
                    for s in long_basket:
                        target_w[s] = w_long
                    for s in short_basket:
                        target_w[s] = w_short

                # 2. Vol targeting (chỉ khi vol_target_ann khác None)
                if config.vol_target_ann is not None:
                    min_periods = max(10, config.lookback_hours // 4)
                    hist_rets = returns_vals[
                        max(1, k - config.lookback_hours) : k
                    ]
                    if len(hist_rets) >= min_periods:
                        vol_hist = float(np.std(hist_rets, ddof=1))
                        vol_ann = vol_hist * math.sqrt(periods_per_year)
                        scale = float(
                            np.clip(
                                config.vol_target_ann / (vol_ann + 1e-12),
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        scale = 1.0
                    rebalance_scales.append(scale)
                    target_w = target_w * scale

                new_weights = target_w

        # 3. Tính Turnover và chi phí thực thi tại nến k
        delta_w = new_weights - current_weights
        turnover_k = float(np.sum(np.abs(delta_w)))

        if turnover_k > 1e-12:
            current_time = common_index[k]
            for s_idx, sym in enumerate(symbols):
                d_w = float(abs(delta_w[s_idx]))
                if d_w > 1e-12:
                    trades_list.append({
                        "time": current_time,
                        "symbol": sym,
                        "w_old": float(current_weights[s_idx]),
                        "w_new": float(new_weights[s_idx]),
                        "turnover": d_w,
                    })

        fee_pct_k = turnover_k * one_way_cost

        # 4. Cập nhật lợi nhuận nến k dùng trọng số đã thiết lập ở nến k-1 (current_weights)
        # R_port,t = sum_i w_{i,t-1} * (Close_{i,t}/Close_{i,t-1} - 1 - funding_rate_event_{i,t}) - FeePct_t
        gross_ret_k = 0.0
        funding_paid_usd_k = 0.0

        for s_idx in range(n_symbols):
            w_prev = current_weights[s_idx]
            if abs(w_prev) > 1e-12:
                p_now = close_matrix[k, s_idx]
                p_prev = close_matrix[k - 1, s_idx]

                if np.isnan(p_now) or np.isnan(p_prev) or p_prev <= 0.0:
                    ret_price = 0.0
                else:
                    ret_price = (p_now / p_prev) - 1.0

                fr_event = funding_matrix[k, s_idx]
                if np.isnan(fr_event):
                    fr_event = 0.0

                gross_ret_k += w_prev * (ret_price - fr_event)
                funding_paid_usd_k += w_prev * fr_event * equity_prev

        port_ret_k = gross_ret_k - fee_pct_k
        returns_vals[k] = port_ret_k

        # Cập nhật vốn equity
        equity = equity_prev * (1.0 + port_ret_k)
        if equity < 0.0:
            equity = 0.0
        equity_vals[k] = equity

        # Tích lũy số liệu tài chính
        fee_usd_k = fee_pct_k * equity_prev
        total_fees_paid += fee_usd_k
        total_turnover += turnover_k
        total_funding_paid += funding_paid_usd_k

        # Ghi nhận trọng số đã thiết lập tại nến k
        current_weights = new_weights
        weights_matrix[k] = current_weights

    equity_curve = pd.Series(
        equity_vals, index=common_index, name="equity", dtype="float64"
    )
    returns = pd.Series(
        returns_vals, index=common_index, name="returns", dtype="float64"
    )
    weights_df = pd.DataFrame(
        weights_matrix, index=common_index, columns=symbols, dtype="float64"
    )
    trades_df = pd.DataFrame(
        trades_list,
        columns=["time", "symbol", "w_old", "w_new", "turnover"],
    )

    # 5. Tính toán các chỉ số hiệu suất tổng hợp
    total_return_pct = (
        float((equity - config.initial_capital) / config.initial_capital * 100.0)
        if config.initial_capital > 0
        else 0.0
    )

    if n_steps > 0 and config.initial_capital > 0 and equity > 0:
        years = n_steps / float(periods_per_year)
        if years > 0:
            annualized_return_pct = float(
                ((equity / config.initial_capital) ** (1.0 / years) - 1.0) * 100.0
            )
        else:
            annualized_return_pct = 0.0
    else:
        annualized_return_pct = 0.0

    ret_std = float(returns.std(ddof=1))
    if ret_std > 1e-12 and len(returns) > 1:
        sharpe = float(returns.mean() / ret_std * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_vals)
    valid = peak > 0
    if np.any(valid):
        dd = (peak[valid] - equity_vals[valid]) / peak[valid]
        max_dd_pct = float(np.nanmax(dd) * 100.0)
    else:
        max_dd_pct = 0.0

    in_market_mask = np.sum(np.abs(weights_matrix), axis=1) > 1e-9
    time_in_market_pct = (
        float(np.mean(in_market_mask) * 100.0) if n_steps > 0 else 0.0
    )

    avg_topk_turnover = (
        float(total_turnover / num_rebalances) if num_rebalances > 0 else 0.0
    )
    vol_target_scale_mean = (
        float(np.mean(rebalance_scales))
        if (config.vol_target_ann is not None and len(rebalance_scales) > 0)
        else 0.0
    )

    metrics = {
        "Total Return %": total_return_pct,
        "Total Return % ": total_return_pct,
        "Annualized Return %": annualized_return_pct,
        "Sharpe": sharpe,
        "MaxDD %": max_dd_pct,
        "Turnover": float(total_turnover),
        "Time in Market %": time_in_market_pct,
        "Total Funding Paid": float(total_funding_paid),
        "Total Funding Collected": float(-total_funding_paid),
        "Total Fees Paid": float(total_fees_paid),
        "Avg Top-K Turnover": avg_topk_turnover,
        "Num Rebalances": int(num_rebalances),
        "Vol Target Scale Mean": vol_target_scale_mean,
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "weights_df": weights_df,
        "trades_df": trades_df,
        "metrics": metrics,
    }
