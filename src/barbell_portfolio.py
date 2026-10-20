"""Module danh mục đầu tư Barbell (Core Yield Sleeve + Tactical Alpha Overlay).

Sprint 9 - Chặng 2b & Sprint 10 - Chặng 2:
Cung cấp:
- BarbellConfig: Cấu hình tham số danh mục Barbell (hỗ trợ SNR gate, Vol Targeting, Drawdown Cushion).
- compute_snr_dispersion: Tính toán SNR dispersion không look-ahead cho vũ trụ coin.
- build_overlay_data: Chuyển đổi dữ liệu multi_aligned_data sang định dạng chuẩn cho Cross-Sectional.
- run_barbell_portfolio: Mô phỏng danh mục Barbell thống nhất trên tài khoản Unified Margin,
  không double-count vốn, tự động quản trị rủi ro đa tầng (SNR Regime Gate, Vol Targeting, DD Cushion).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.cross_exchange_carry import (
    CrossExchangeCarryConfig,
    run_binance_tier1_plus_carry,
    run_cross_exchange_carry,
)
from src.cross_sectional import CrossSectionalConfig, run_cross_sectional_portfolio


@dataclass(frozen=True)
class BarbellConfig:
    """Cấu hình tham số cho danh mục đầu tư Barbell Portfolio (Sprint 9 & Sprint 10).

    Kết hợp 2 nguồn alpha:
    - Core Yield Sleeve: 100% NAV chạy Cross-Exchange Carry.
    - Tactical Alpha Overlay: 30% - 50% NAV chạy Core-4 Buffered XS Momentum
      trên phần dư ký quỹ (Unified Margin).
    - Quản trị rủi ro đa tầng: SNR Gate, Pair Volatility Targeting, Drawdown Cushion.
    """

    initial_capital: float = 3000.0
    carry_config: CrossExchangeCarryConfig = CrossExchangeCarryConfig(initial_capital=3000.0)
    enable_overlay: bool = True
    overlay_gross_leverage: float = 0.40
    overlay_lookback_hours: int = 336
    overlay_rebalance_hours: int = 48
    overlay_buffer_rank: int = 1
    overlay_min_spread: float = 0.75
    overlay_top_k: int = 1
    overlay_execution: str = "maker_taker_hedge"
    max_total_gross_leverage: float = 2.50

    # Cấu hình mới Sprint 10: SNR Gate
    enable_snr_gate: bool = True
    overlay_min_snr_dispersion: float = 1.35
    overlay_exit_snr_dispersion: float = 0.95

    # Cấu hình mới Sprint 10: Volatility Targeting
    enable_vol_targeting: bool = True
    overlay_target_vol_ann: float = 0.065
    overlay_vol_lookback: int = 168
    overlay_leverage_min: float = 0.10
    overlay_leverage_max: float = 0.45

    # Cấu hình mới Sprint 10: Drawdown Cushion
    enable_drawdown_cushion: bool = True
    overlay_dd_cut_threshold: float = 0.04
    overlay_dd_off_threshold: float = 0.06

    # Cấu hình mới Sprint 11
    carry_variant: str = "tier1b"            # "tier1a" | "tier1b" | "legacy"
    gating_mode: str = "hard_binary"         # "hard_binary" | "smooth_conviction"
    carry_tilt_weight: float = 0.20
    smooth_target_vol_ann: float = 0.075
    smooth_leverage_min: float = 0.15
    smooth_leverage_max: float = 0.50
    min_weight_change_to_trade: float = 0.08
    overlay_quantize_legs: bool = True
    reinvest_profits: bool = True


def compute_snr_dispersion(
    close_df: pd.DataFrame,
    lookback: int = 336,
) -> pd.DataFrame:
    """Tính toán SNR (Signal-to-Noise Ratio) dispersion cho vũ trụ tài sản.

    SNR là t-stat của trend 14 ngày (lookback nến 1h), KHÔNG chuẩn hóa ngang giữa các coin.
    Được tính toán hoàn toàn không có look-ahead bias (đã shift(1)).

    Tham số:
        close_df: DataFrame giá đóng cửa (index 1h UTC, các cột là symbol).
        lookback: Số nến lookback tính trend và biến động (mặc định 336 = 14 ngày).

    Trả về:
        DataFrame gồm các cột SNR của từng symbol và cột 'raw_snr_dispersion'.
    """
    if close_df.empty:
        cols = list(close_df.columns) + ["raw_snr_dispersion"]
        return pd.DataFrame(index=close_df.index, columns=cols, dtype="float64")

    if lookback < 1:
        raise ValueError(f"lookback phải >= 1, nhận: {lookback}")

    r_l = close_df.pct_change(lookback).shift(1)
    ret_1h = close_df.pct_change()
    sigma_1h = ret_1h.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)

    denom = sigma_1h * math.sqrt(lookback)
    denom = denom.replace(0.0, np.nan)
    snr = r_l / denom
    snr = snr.replace([np.inf, -np.inf], np.nan)

    raw_snr_dispersion = snr.max(axis=1) - snr.min(axis=1)

    out_df = snr.copy()
    out_df["raw_snr_dispersion"] = raw_snr_dispersion
    return out_df


def build_overlay_data(
    multi_aligned_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Chuyển đổi dữ liệu multi_aligned_data sang định dạng chuẩn cho Cross-Sectional Momentum.

    Quy tắc ánh xạ cho từng symbol:
    - 'close': ưu tiên giá perp ('perp_close' nếu có và hợp lệ, fallback 'close' hoặc 'spot_close').
    - 'funding_rate_event': lấy funding_event_{best_short_ex} theo từng nến (fallback funding_event_binance).
    - Giữ nguyên DatetimeIndex UTC ns tên 'datetime'.

    Tham số:
        multi_aligned_data: Dict map {symbol: DataFrame đồng bộ từ align_multi_exchange_funding}.

    Trả về:
        Dict map {symbol: DataFrame chứa tối thiểu cột 'close' và 'funding_rate_event'}.
    """
    overlay_data: dict[str, pd.DataFrame] = {}

    for sym, df in multi_aligned_data.items():
        if df is None:
            continue
        if df.empty:
            overlay_data[sym] = df.copy()
            continue

        df_out = df.copy()

        # 1. Đảm bảo DatetimeIndex UTC ns mang tên 'datetime'
        if isinstance(df_out.index, pd.DatetimeIndex):
            if df_out.index.tz is None:
                df_out.index = df_out.index.tz_localize("UTC")
            else:
                df_out.index = df_out.index.tz_convert("UTC")
            df_out.index = df_out.index.as_unit("ns")
            df_out.index.name = "datetime"

        # 2. Xác định giá close perp
        if "perp_close" in df_out.columns and df_out["perp_close"].notna().any():
            close_s = df_out["perp_close"].astype("float64")
            if "close" in df_out.columns:
                close_s = close_s.fillna(df_out["close"].astype("float64"))
        elif "close" in df_out.columns:
            close_s = df_out["close"].astype("float64")
        elif "spot_close" in df_out.columns:
            close_s = df_out["spot_close"].astype("float64")
        else:
            raise ValueError(
                f"DataFrame của symbol '{sym}' thiếu cột giá 'close' hoặc 'perp_close'."
            )
        df_out["close"] = close_s

        # 3. Xác định funding_rate_event tương ứng với sàn short tốt nhất
        n_rows = len(df_out)
        fr_event = np.zeros(n_rows, dtype=np.float64)

        if "funding_event_binance" in df_out.columns:
            binance_fr = df_out["funding_event_binance"].to_numpy(dtype=np.float64)
            fr_event = np.nan_to_num(binance_fr, nan=0.0)
        elif "funding_rate_event" in df_out.columns:
            curr_fr = df_out["funding_rate_event"].to_numpy(dtype=np.float64)
            fr_event = np.nan_to_num(curr_fr, nan=0.0)

        if "best_short_ex" in df_out.columns:
            best_short_arr = df_out["best_short_ex"].to_numpy()
            for ex in ("binance", "bybit", "okx"):
                col = f"funding_event_{ex}"
                if col in df_out.columns:
                    val_arr = df_out[col].to_numpy(dtype=np.float64)
                    mask = (best_short_arr == ex) & np.isfinite(val_arr)
                    fr_event[mask] = val_arr[mask]

        df_out["funding_rate_event"] = fr_event
        overlay_data[sym] = df_out

    return overlay_data


from src.cross_exchange_carry import quantize_order_qty


def _simulate_barbell_equity(
    raw_weights: pd.DataFrame,      # trọng số unit-gross theo nến (đã reindex common_index, fillna 0)
    close_matrix: pd.DataFrame,     # giá đóng cửa theo nến, cùng cột
    funding_matrix: pd.DataFrame,   # funding_rate_event theo nến, cùng cột
    target_leverage: np.ndarray,    # L_t mỗi nến (trước DD cushion)
    carry_returns: np.ndarray,      # lợi nhuận carry 1h
    initial_capital: float,
    rebalance_hours: int,
    one_way_cost: float,
    quantize: bool,
    min_weight_change: float,
    dd_cut: float,
    dd_off: float,
    enable_dd_cushion: bool,
    reinvest_profits: bool = True,
    carry_dollar_pnl: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Mô phỏng động cơ vốn Barbell với lượng hóa lot-size và quản trị drawdown.

    reinvest_profits=False: mo phong neo von tinh tai initial_capital (cash drag), khong dung de tai lap so Sprint 11.
    """
    bars = len(raw_weights)
    symbols = list(raw_weights.columns)
    n_sym = len(symbols)

    equity = np.zeros(bars, dtype=np.float64)
    returns = np.zeros(bars, dtype=np.float64)
    gross_usd = np.zeros(bars, dtype=np.float64)
    fees_usd = np.zeros(bars, dtype=np.float64)
    net_usd = np.zeros(bars, dtype=np.float64)
    weights = np.zeros((bars, n_sym), dtype=np.float64)
    leverage = np.zeros(bars, dtype=np.float64)

    quantization_events = 0
    skipped_rebalances = 0
    turnover_weight = 0.0
    pending_initial_fee = 0.0

    if bars == 0:
        return {
            "equity": equity,
            "returns": returns,
            "gross_usd": gross_usd,
            "fees_usd": fees_usd,
            "net_usd": net_usd,
            "weights": weights,
            "leverage": leverage,
            "quantization_events": quantization_events,
            "skipped_rebalances": skipped_rebalances,
            "turnover_weight": turnover_weight,
        }

    raw_w_arr = raw_weights.to_numpy(dtype=np.float64)
    close_arr = close_matrix.to_numpy(dtype=np.float64)
    funding_arr = funding_matrix.to_numpy(dtype=np.float64)

    ret_arr = np.zeros_like(close_arr)
    if bars > 1:
        prev_close = close_arr[:-1]
        diff_close = close_arr[1:] - prev_close
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_step = np.where(prev_close > 0.0, diff_close / prev_close, 0.0)
            ret_arr[1:] = np.nan_to_num(ret_step, nan=0.0)

    is_funding_bar = np.zeros(bars, dtype=bool)
    if isinstance(raw_weights.index, pd.DatetimeIndex):
        hours = raw_weights.index.hour
        minutes = raw_weights.index.minute
        is_funding_bar = np.isin(hours, [0, 8, 16]) & (minutes == 0)

    w_changed = np.zeros(bars, dtype=bool)
    if bars > 1:
        diff_w = np.abs(raw_w_arr[1:] - raw_w_arr[:-1])
        w_changed[1:] = np.max(diff_w, axis=1) > 1e-12

    equity[0] = initial_capital
    peak = initial_capital
    w_cur = np.zeros(n_sym, dtype=np.float64)
    current_leverage = 0.0

    for t in range(bars):
        if t == 0:
            gross_usd[0] = 0.0
            fees_usd[0] = 0.0
            net_usd[0] = 0.0
            returns[0] = 0.0
            equity[0] = initial_capital
            peak = initial_capital

            L_eff = float(target_leverage[0]) if len(target_leverage) > 0 else 0.0
            current_leverage = L_eff
            target = raw_w_arr[0] * L_eff
            w_q = target.copy()

            if quantize and initial_capital > 0.0:
                for i in range(n_sym):
                    if abs(target[i]) > 1e-12:
                        sym = symbols[i]
                        px = close_arr[0, i]
                        target_val = abs(target[i]) * initial_capital
                        q, actual = quantize_order_qty(sym, target_val, px)
                        w_q_i = float(np.sign(target[i]) * (actual / initial_capital)) if q > 0 else 0.0
                        if abs(w_q_i - target[i]) > 1e-12:
                            quantization_events += 1
                        w_q[i] = w_q_i
                    else:
                        w_q[i] = 0.0

            delta_w = np.abs(w_q - w_cur)
            if np.any(delta_w > 1e-12):
                turnover_weight += float(np.sum(delta_w))
                w_cur = w_q.copy()

            pending_initial_fee = float(np.sum(np.abs(w_q)) * initial_capital * one_way_cost)

            leverage[0] = current_leverage
            weights[0] = w_cur.copy()
            continue

        prev_eq = equity[t - 1]
        dd_t = 1.0 - (prev_eq / peak) if peak > 0.0 else 0.0
        if enable_dd_cushion:
            if dd_t > dd_off:
                mult = 0.0
            elif dd_t > dd_cut:
                mult = 0.5
            else:
                mult = 1.0
        else:
            mult = 1.0

        gross_ret = float(np.sum(w_cur * ret_arr[t]))
        if is_funding_bar[t]:
            gross_ret -= float(np.sum(w_cur * funding_arr[t]))

        g_base = prev_eq if reinvest_profits else float(initial_capital)
        g_usd = gross_ret * g_base
        gross_usd[t] = g_usd

        is_rebal = (t % rebalance_hours == 0) or w_changed[t]
        fee = 0.0

        if is_rebal:
            L_eff = float(target_leverage[t] * mult)
            current_leverage = L_eff
            target = raw_w_arr[t] * L_eff
            w_q = target.copy()

            if reinvest_profits:
                eq_mid = prev_eq + carry_returns[t] * prev_eq + g_usd
            else:
                eq_mid = float(initial_capital)
            if eq_mid <= 0.0:
                eq_mid = 0.0

            if quantize and eq_mid > 0.0:
                for i in range(n_sym):
                    if abs(target[i]) > 1e-12:
                        sym = symbols[i]
                        px = close_arr[t, i]
                        target_val = abs(target[i]) * eq_mid
                        q, actual = quantize_order_qty(sym, target_val, px)
                        w_q_i = float(np.sign(target[i]) * (actual / eq_mid)) if q > 0 else 0.0
                        if abs(w_q_i - target[i]) > 1e-12:
                            quantization_events += 1
                        w_q[i] = w_q_i
                    else:
                        w_q[i] = 0.0

            sign_q = np.where(w_q > 1e-12, 1, np.where(w_q < -1e-12, -1, 0))
            sign_cur = np.where(w_cur > 1e-12, 1, np.where(w_cur < -1e-12, -1, 0))
            max_diff = float(np.max(np.abs(w_q - w_cur))) if n_sym > 0 else 0.0

            if np.array_equal(sign_q, sign_cur) and (max_diff < min_weight_change):
                fee = 0.0
                skipped_rebalances += 1
            else:
                trade_w = np.abs(w_q - w_cur)
                fee = float(np.sum(trade_w) * eq_mid * one_way_cost)
                turnover_weight += float(np.sum(trade_w))
                w_cur = w_q.copy()

        fees_usd[t] = fee
        n_usd = g_usd - fee
        net_usd[t] = n_usd

        if reinvest_profits:
            eq_t = prev_eq + carry_returns[t] * prev_eq + n_usd
        else:
            if carry_dollar_pnl is not None and t < len(carry_dollar_pnl):
                carry_usd = float(carry_dollar_pnl[t])
            else:
                carry_usd = float(carry_returns[t]) * float(initial_capital)
            eq_t = prev_eq + carry_usd + n_usd
        equity[t] = eq_t
        if eq_t > peak:
            peak = eq_t

        returns[t] = (eq_t / prev_eq - 1.0) if prev_eq > 0.0 else 0.0
        leverage[t] = current_leverage
        weights[t] = w_cur.copy()

        if t == 1 and bars > 1:
            fees_usd[1] += pending_initial_fee
            net_usd[1] -= pending_initial_fee
            equity[1] -= pending_initial_fee
            returns[1] = (equity[1] / equity[0] - 1.0) if equity[0] > 0.0 else 0.0
            peak = max(equity[0], equity[1])

    return {
        "equity": equity,
        "returns": returns,
        "gross_usd": gross_usd,
        "fees_usd": fees_usd,
        "net_usd": net_usd,
        "weights": weights,
        "leverage": leverage,
        "quantization_events": quantization_events,
        "skipped_rebalances": skipped_rebalances,
        "turnover_weight": turnover_weight,
    }


def run_barbell_portfolio(
    multi_aligned_data: dict[str, pd.DataFrame],
    config: BarbellConfig,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """Mô phỏng danh mục Barbell (Core Yield Sleeve + Tactical Alpha Overlay).

    Quy tắc tài chính & ký quỹ:
    - Core Yield Sleeve: 100% vốn nền (initial_capital) chạy Hybrid Cross-Exchange Carry.
    - Tactical Alpha Overlay: dùng phần dư ký quỹ trên tài khoản Perp chung, tỷ lệ
      gross leverage được điều chỉnh động bởi SNR Regime Gate, Volatility Targeting và DD Cushion.
    - SNR Regime Gate: Hysteresis trên raw_snr_dispersion (bật khi >= min_snr, tắt khi <= exit_snr).
    - Volatility Targeting: Nghịch đảo biến động pair 168 giờ, giới hạn trong [leverage_min, leverage_max].
    - Drawdown Cushion: Giảm 50% leverage khi DD > 4%, tắt hoàn toàn khi DD > 6%.
    - Nếu overlay không hoạt động: equity_curve Barbell trùng khít 100% với carry_result["equity_curve"] (sai số < 1e-9).

    Tham số:
        multi_aligned_data: Dict map {symbol: DataFrame đã đồng bộ đa sàn}.
        config: Cấu hình BarbellConfig.
        periods_per_year: Số chu kỳ mỗi năm (8760 cho nến 1h).

    Trả về:
        Dict chứa đầy đủ:
        - equity_curve: pd.Series đường cong vốn 1h (khởi đầu = initial_capital).
        - returns: pd.Series tỷ suất lợi nhuận 1h Barbell.
        - carry_result: Dict gốc từ run_cross_exchange_carry hoặc run_binance_tier1_plus_carry.
        - overlay_result: Dict gốc từ run_cross_sectional_portfolio.
        - overlay_weights: pd.DataFrame trọng số overlay đã scale theo leverage_series.
        - attribution_series: pd.DataFrame phân rã PnL 1h.
        - funding_pnl, basis_pnl, overlay_pnl, fees, turnover: Các giá trị tổng hợp lũy kế.
        - metrics: Dict các chỉ số hiệu suất chuẩn Sprint 9 và Sprint 10.
        - carry_variant, gating_mode: Tên cấu hình sleeve và chế độ gating.
        - go_no_go_inputs: Dict chứa annualized_return_pct, sharpe, maxdd_pct, net_pnl, final_equity.
        - overlay_active_ratio: Tỷ lệ nến overlay tham gia thị trường.
        - overlay_leverage_series: Chuỗi đòn bẩy overlay theo giờ.
        - snr_dispersion: Chuỗi raw_snr_dispersion theo giờ.
    """
    # 1. Kiểm tra tính hợp lệ của tham số cấu hình
    if not (0.0 <= config.overlay_gross_leverage <= 0.50):
        raise ValueError(
            f"overlay_gross_leverage phải nằm trong đoạn [0.0, 0.50], nhận: {config.overlay_gross_leverage}"
        )
    if config.overlay_lookback_hours < 1:
        raise ValueError(
            f"overlay_lookback_hours phải >= 1, nhận: {config.overlay_lookback_hours}"
        )
    if config.overlay_rebalance_hours < 1:
        raise ValueError(
            f"overlay_rebalance_hours phải >= 1, nhận: {config.overlay_rebalance_hours}"
        )
    if config.initial_capital <= 0.0:
        raise ValueError(
            f"initial_capital phải > 0, nhận: {config.initial_capital}"
        )
    if config.overlay_top_k < 1:
        raise ValueError(
            f"overlay_top_k phải >= 1, nhận: {config.overlay_top_k}"
        )
    if config.overlay_buffer_rank < 0:
        raise ValueError(
            f"overlay_buffer_rank phải >= 0, nhận: {config.overlay_buffer_rank}"
        )
    if config.overlay_min_spread < 0.0:
        raise ValueError(
            f"overlay_min_spread phải >= 0, nhận: {config.overlay_min_spread}"
        )
    if config.max_total_gross_leverage <= 0.0:
        raise ValueError(
            f"max_total_gross_leverage phải > 0, nhận: {config.max_total_gross_leverage}"
        )
    if config.overlay_vol_lookback < 1:
        raise ValueError(
            f"overlay_vol_lookback phải >= 1, nhận: {config.overlay_vol_lookback}"
        )
    if config.overlay_target_vol_ann <= 0.0:
        raise ValueError(
            f"overlay_target_vol_ann phải > 0, nhận: {config.overlay_target_vol_ann}"
        )
    if not (0.0 <= config.overlay_leverage_min <= config.overlay_leverage_max):
        raise ValueError(
            f"overlay_leverage_min và overlay_leverage_max không hợp lệ: [{config.overlay_leverage_min}, {config.overlay_leverage_max}]"
        )
    if not (0.0 <= config.overlay_dd_cut_threshold <= config.overlay_dd_off_threshold):
        raise ValueError(
            f"Ngưỡng drawdown cushion không hợp lệ: cut={config.overlay_dd_cut_threshold}, off={config.overlay_dd_off_threshold}"
        )
    if config.overlay_min_snr_dispersion < 0.0 or config.overlay_exit_snr_dispersion < 0.0:
        raise ValueError("Ngưỡng SNR dispersion phải >= 0")
    if config.carry_variant not in {"tier1a", "tier1b", "legacy"}:
        raise ValueError(
            f"carry_variant phải thuộc {{'tier1a', 'tier1b', 'legacy'}}, nhận: {config.carry_variant}"
        )
    if config.gating_mode not in {"hard_binary", "smooth_conviction"}:
        raise ValueError(
            f"gating_mode phải thuộc {{'hard_binary', 'smooth_conviction'}}, nhận: {config.gating_mode}"
        )
    if config.carry_tilt_weight < 0.0:
        raise ValueError(
            f"carry_tilt_weight phải >= 0, nhận: {config.carry_tilt_weight}"
        )
    if not (0.0 < config.smooth_leverage_min <= config.smooth_leverage_max <= 1.0):
        raise ValueError(
            f"smooth_leverage_min và smooth_leverage_max không hợp lệ (yêu cầu 0 < min <= max <= 1.0), nhận: min={config.smooth_leverage_min}, max={config.smooth_leverage_max}"
        )
    if config.min_weight_change_to_trade < 0.0:
        raise ValueError(
            f"min_weight_change_to_trade phải >= 0, nhận: {config.min_weight_change_to_trade}"
        )

    # 2. Xử lý trường hợp dữ liệu rỗng
    if not multi_aligned_data:
        empty_idx = pd.DatetimeIndex([], tz="UTC", name="datetime").as_unit("ns")
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        empty_ret = pd.Series([], index=empty_idx, dtype="float64", name="returns")
        empty_attr = pd.DataFrame(
            columns=[
                "funding_pnl",
                "basis_pnl",
                "overlay_pnl",
                "carry_fees",
                "overlay_fees",
                "fees",
                "net_pnl",
            ],
            index=empty_idx,
            dtype="float64",
        )
        empty_metrics = {
            "Total Return %": 0.0,
            "Total Return % ": 0.0,
            "Annualized Return %": 0.0,
            "Sharpe": 0.0,
            "MaxDD %": 0.0,
            "Total Funding Collected": 0.0,
            "Total Fees Paid": 0.0,
            "Overlay PnL": 0.0,
            "Overlay Time in Market %": 0.0,
            "Carry Weight": 1.0,
            "Margin Calls": 0,
            "Num Rebalances": 0,
            "Overlay Rebalances": 0,
            "Carry Rebalances": 0,
            "Price Rebalances": 0,
            "Net PnL": 0.0,
            "Turnover": 0.0,
            "Time in Market %": 0.0,
            "overlay_active_ratio": 0.0,
            "overlay_gate_on_ratio": 0.0,
            "overlay_mean_leverage": 0.0,
            "snr_dispersion_mean": 0.0,
            "drawdown_cushion_events": 0,
            "overlay_gross_pnl": 0.0,
            "overlay_net_pnl": 0.0,
            "overlay_fees": 0.0,
            "overlay_quantization_events": 0,
            "overlay_skipped_rebalances": 0,
            "carry_variant": config.carry_variant,
            "gating_mode": config.gating_mode,
            "carry_fallback": False,
            "reinvest_profits": bool(config.reinvest_profits),
        }
        return {
            "equity_curve": empty_eq,
            "returns": empty_ret,
            "carry_result": {},
            "overlay_result": {},
            "overlay_weights": pd.DataFrame(index=empty_idx),
            "attribution_series": empty_attr,
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "overlay_pnl": 0.0,
            "overlay_fees": 0.0,
            "overlay_turnover_scaled": 0.0,
            "overlay_quantization_events": 0,
            "overlay_skipped_rebalances": 0,
            "fees": 0.0,
            "turnover": 0.0,
            "metrics": empty_metrics,
            "carry_variant": config.carry_variant,
            "gating_mode": config.gating_mode,
            "carry_fallback": False,
            "reinvest_profits": bool(config.reinvest_profits),
            "go_no_go_inputs": {
                "annualized_return_pct": 0.0,
                "sharpe": 0.0,
                "maxdd_pct": 0.0,
                "net_pnl": 0.0,
                "final_equity": config.initial_capital,
            },
            "overlay_active_ratio": 0.0,
            "overlay_leverage_series": pd.Series([], index=empty_idx, dtype="float64"),
            "snr_dispersion": pd.Series([], index=empty_idx, dtype="float64"),
        }

    # 3. Chạy Core Yield Sleeve (100% Vốn nền)
    carry_cfg = config.carry_config
    if carry_cfg.initial_capital != config.initial_capital:
        carry_cfg = replace(carry_cfg, initial_capital=config.initial_capital)

    carry_fallback = False
    if config.carry_variant in ("tier1a", "tier1b"):
        carry_result = run_binance_tier1_plus_carry(
            multi_aligned_data,
            config=carry_cfg,
            variant=config.carry_variant,
            periods_per_year=periods_per_year,
        )
        if len(carry_result["equity_curve"]) == 0:
            carry_result = run_cross_exchange_carry(
                multi_aligned_data,
                config=carry_cfg,
                periods_per_year=periods_per_year,
            )
            carry_fallback = True
    else:
        carry_result = run_cross_exchange_carry(
            multi_aligned_data,
            config=carry_cfg,
            periods_per_year=periods_per_year,
        )

    common_index = carry_result["equity_curve"].index
    symbols = list(multi_aligned_data.keys())
    n_bars = len(common_index)

    # 4. Chạy hoặc bỏ qua Tactical Alpha Overlay
    ov_perp_fee = 0.0003
    ov_slippage = 0.0002
    if config.overlay_execution == "real_taker":
        ov_perp_fee = 0.0005
        ov_slippage = 0.0002

    overlay_data = build_overlay_data(multi_aligned_data)

    if config.enable_overlay and config.overlay_gross_leverage > 0.0:
        xs_config = CrossSectionalConfig(
            mode="momentum",
            lookback_hours=config.overlay_lookback_hours,
            rebalance_hours=config.overlay_rebalance_hours,
            min_spread=config.overlay_min_spread,
            gross_leverage=1.0,
            perp_fee=ov_perp_fee,
            slippage=ov_slippage,
            initial_capital=config.initial_capital,
            top_k=config.overlay_top_k,
            buffer_rank=max(1, config.overlay_buffer_rank),
            carry_tilt_weight=config.carry_tilt_weight,
            vol_target_ann=None,
        )
        overlay_result = run_cross_sectional_portfolio(
            overlay_data,
            config=xs_config,
            periods_per_year=periods_per_year,
        )
    else:
        overlay_result = {
            "equity_curve": pd.Series(
                config.initial_capital, index=common_index, dtype="float64", name="equity"
            ),
            "returns": pd.Series(0.0, index=common_index, dtype="float64", name="returns"),
            "weights_df": pd.DataFrame(
                0.0, index=common_index, columns=symbols, dtype="float64"
            ),
            "trades_df": pd.DataFrame(
                columns=["time", "symbol", "w_old", "w_new", "turnover"]
            ),
            "metrics": {
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
            },
        }

    overlay_returns = overlay_result["returns"].reindex(common_index).fillna(0.0)
    raw_weights = overlay_result.get("weights_df", pd.DataFrame())
    if not raw_weights.empty:
        raw_weights_aligned = raw_weights.reindex(common_index).fillna(0.0)
    else:
        raw_weights_aligned = pd.DataFrame(
            0.0, index=common_index, columns=symbols, dtype="float64"
        )

    # 5. Xây dựng close_matrix và tính toán SNR dispersion
    close_dict: dict[str, pd.Series] = {}
    for sym in symbols:
        if sym in overlay_data and "close" in overlay_data[sym].columns:
            close_dict[sym] = overlay_data[sym]["close"]
    close_matrix = pd.DataFrame(close_dict, index=common_index, dtype="float64").ffill()

    snr_df = compute_snr_dispersion(close_matrix, config.overlay_lookback_hours)
    raw_snr_dispersion = snr_df["raw_snr_dispersion"]

    # 6. Xác định mốc tái cân bằng, SNR Gate và Vol Targeting
    diff_s = raw_weights_aligned.diff().abs().sum(axis=1)
    weights_changed_mask = ~diff_s.le(1e-12)
    hour_rebal_mask = (np.arange(n_bars) % config.overlay_rebalance_hours == 0)
    rebal_mask = hour_rebal_mask | weights_changed_mask
    if n_bars > 0:
        rebal_mask.iloc[0] = True

    gate_arr = np.zeros(n_bars, dtype=np.float64)
    l_raw_arr = np.zeros(n_bars, dtype=np.float64)
    ret_matrix = close_matrix.pct_change()

    gate_state = False
    current_l = 0.0 if config.enable_vol_targeting else config.overlay_gross_leverage

    for t in range(n_bars):
        if rebal_mask.iloc[t]:
            w_row = raw_weights_aligned.iloc[t]
            pos_syms = w_row[w_row > 1e-12].index.tolist()
            neg_syms = w_row[w_row < -1e-12].index.tolist()

            if config.gating_mode == "smooth_conviction":
                if len(pos_syms) == 0 or len(neg_syms) == 0:
                    L_t = 0.0
                    gate_val = 0.0
                    l_val = L_t
                else:
                    d_val = raw_snr_dispersion.iloc[t]
                    if pd.isna(d_val) or d_val < 0.25:
                        c_t = 0.0
                    else:
                        c_t = float(np.clip(d_val / 1.00, 0.50, 1.00))

                    w_pos = w_row[pos_syms]
                    w_neg = w_row[neg_syms].abs()
                    r_long = (ret_matrix[pos_syms] * (w_pos / w_pos.sum())).sum(axis=1)
                    r_short = (ret_matrix[neg_syms] * (w_neg / w_neg.sum())).sum(axis=1)
                    r_diff = r_long - r_short

                    start_idx = max(0, t - config.overlay_vol_lookback)
                    slice_diff = r_diff.iloc[start_idx:t]

                    pair_std = float(slice_diff.std(ddof=1)) if len(slice_diff) > 1 else 0.0
                    sigma_pair_ann = pair_std * math.sqrt(periods_per_year)
                    if np.isnan(sigma_pair_ann) or sigma_pair_ann < 0.08:
                        sigma_pair = 0.08
                    else:
                        sigma_pair = max(sigma_pair_ann, 0.08)

                    L_t = c_t * float(
                        np.clip(
                            config.smooth_target_vol_ann / sigma_pair,
                            config.smooth_leverage_min,
                            config.smooth_leverage_max,
                        )
                    )
                    gate_val = 1.0 if c_t > 0.0 else 0.0
                    l_val = L_t
            else:
                # Đánh giá SNR Regime Gate (Hysteresis)
                if config.enable_snr_gate:
                    d_val = raw_snr_dispersion.iloc[t]
                    if pd.notna(d_val):
                        if not gate_state and d_val >= config.overlay_min_snr_dispersion:
                            gate_state = True
                        elif gate_state and d_val <= config.overlay_exit_snr_dispersion:
                            gate_state = False
                    gate_val = 1.0 if gate_state else 0.0
                else:
                    gate_val = 1.0

                if len(pos_syms) == 0 or len(neg_syms) == 0:
                    l_val = 0.0 if config.enable_vol_targeting else config.overlay_gross_leverage
                elif not config.enable_vol_targeting:
                    l_val = config.overlay_gross_leverage
                else:
                    w_pos = w_row[pos_syms]
                    w_neg = w_row[neg_syms].abs()
                    r_long = (ret_matrix[pos_syms] * (w_pos / w_pos.sum())).sum(axis=1)
                    r_short = (ret_matrix[neg_syms] * (w_neg / w_neg.sum())).sum(axis=1)
                    r_diff = r_long - r_short

                    start_idx = max(0, t - config.overlay_vol_lookback)
                    slice_diff = r_diff.iloc[start_idx:t]

                    pair_std = float(slice_diff.std(ddof=1)) if len(slice_diff) > 1 else 0.0
                    sigma_pair = pair_std * math.sqrt(periods_per_year)
                    if np.isnan(sigma_pair) or sigma_pair < 0.05:
                        sigma_pair = 0.05

                    l_raw = config.overlay_target_vol_ann / sigma_pair
                    l_val = float(np.clip(l_raw, config.overlay_leverage_min, config.overlay_leverage_max))

            gate_arr[t] = gate_val
            l_raw_arr[t] = l_val
            current_l = l_val
        else:
            gate_arr[t] = gate_arr[t - 1]
            l_raw_arr[t] = current_l

    # 7. Mô phỏng động cơ Barbell Equity với lượng hóa lot-size và DD cushion
    funding_dict: dict[str, pd.Series] = {}
    for sym in close_matrix.columns:
        if sym in overlay_data and "funding_rate_event" in overlay_data[sym].columns:
            funding_dict[sym] = overlay_data[sym]["funding_rate_event"]
        else:
            funding_dict[sym] = pd.Series(0.0, index=common_index)
    funding_matrix = pd.DataFrame(funding_dict, index=common_index, dtype="float64").fillna(0.0)

    target_leverage = l_raw_arr * gate_arr
    r_carry_arr = carry_result["returns"].reindex(common_index).fillna(0.0).to_numpy(dtype=np.float64)
    one_way_cost = ov_perp_fee + ov_slippage

    is_overlay_inactive = (
        not config.enable_overlay
        or config.overlay_gross_leverage <= 0.0
        or (np.nanmax(target_leverage) < 1e-12 if len(target_leverage) > 0 else True)
        or (np.abs(raw_weights_aligned.to_numpy()) < 1e-12).all()
    )

    if is_overlay_inactive:
        equity_curve = carry_result["equity_curve"].copy()
        returns = carry_result["returns"].copy()
        overlay_weights = pd.DataFrame(
            0.0, index=common_index, columns=close_matrix.columns, dtype="float64"
        )
        overlay_weights.index.name = "datetime"
        leverage_series = pd.Series(0.0, index=common_index, dtype="float64")
        gate_series = pd.Series(gate_arr, index=common_index, dtype="float64")
        dd_series = pd.Series(1.0, index=common_index, dtype="float64")

        gross_usd = pd.Series(0.0, index=common_index, dtype="float64")
        fees_usd = pd.Series(0.0, index=common_index, dtype="float64")
        net_usd = pd.Series(0.0, index=common_index, dtype="float64")
        r_overlay_scaled = pd.Series(0.0, index=common_index, dtype="float64")

        overlay_pnl_val = 0.0
        overlay_gross_pnl_val = 0.0
        overlay_net_pnl_val = 0.0
        overlay_time_in_market_pct = 0.0
        overlay_active_ratio = 0.0
        overlay_turnover_scaled = 0.0
        overlay_rebalances = 0
        quantization_events = 0
        skipped_rebalances = 0
    else:
        try:
            carry_attr_src = carry_result["attribution_series"]
            carry_dollar_pnl_arr = (
                carry_attr_src["funding_pnl"].reindex(common_index).fillna(0.0)
                + carry_attr_src["basis_pnl"].reindex(common_index).fillna(0.0)
                - carry_attr_src["fees"].reindex(common_index).fillna(0.0)
            ).to_numpy(dtype=np.float64)
        except Exception:
            carry_dollar_pnl_arr = np.zeros(len(common_index), dtype=np.float64)

        sim_res = _simulate_barbell_equity(
            raw_weights=raw_weights_aligned[close_matrix.columns],
            close_matrix=close_matrix,
            funding_matrix=funding_matrix,
            target_leverage=target_leverage,
            carry_returns=r_carry_arr,
            initial_capital=config.initial_capital,
            rebalance_hours=config.overlay_rebalance_hours,
            one_way_cost=one_way_cost,
            quantize=bool(config.overlay_quantize_legs),
            min_weight_change=config.min_weight_change_to_trade,
            dd_cut=config.overlay_dd_cut_threshold,
            dd_off=config.overlay_dd_off_threshold,
            enable_dd_cushion=config.enable_drawdown_cushion,
            reinvest_profits=bool(config.reinvest_profits),
            carry_dollar_pnl=carry_dollar_pnl_arr,
        )

        equity_curve = pd.Series(
            sim_res["equity"], index=common_index, dtype="float64", name="equity"
        )
        returns = pd.Series(
            sim_res["returns"], index=common_index, dtype="float64", name="returns"
        )
        overlay_weights = pd.DataFrame(
            sim_res["weights"], index=common_index, columns=close_matrix.columns, dtype="float64"
        )
        overlay_weights.index.name = "datetime"
        leverage_series = pd.Series(
            sim_res["leverage"], index=common_index, dtype="float64"
        )
        gate_series = pd.Series(gate_arr, index=common_index, dtype="float64")

        peak_arr = np.maximum.accumulate(sim_res["equity"])
        dd_vals = np.where(peak_arr > 0.0, 1.0 - sim_res["equity"] / peak_arr, 0.0)
        if config.enable_drawdown_cushion:
            dd_mult_vals = np.where(
                dd_vals > config.overlay_dd_off_threshold,
                0.0,
                np.where(dd_vals > config.overlay_dd_cut_threshold, 0.5, 1.0),
            )
        else:
            dd_mult_vals = np.ones(n_bars, dtype=np.float64)
        dd_series = pd.Series(dd_mult_vals, index=common_index, dtype="float64")

        gross_usd = pd.Series(sim_res["gross_usd"], index=common_index, dtype="float64")
        fees_usd = pd.Series(sim_res["fees_usd"], index=common_index, dtype="float64")
        net_usd = pd.Series(sim_res["net_usd"], index=common_index, dtype="float64")

        equity_shift = equity_curve.shift(1).fillna(config.initial_capital)
        r_overlay_scaled = np.where(equity_shift > 0.0, net_usd / equity_shift, 0.0)
        r_overlay_scaled = pd.Series(r_overlay_scaled, index=common_index, dtype="float64")

        overlay_gross_pnl_val = float(gross_usd.sum())
        overlay_pnl_val = float(gross_usd.sum() - fees_usd.sum())
        overlay_net_pnl_val = overlay_pnl_val
        overlay_turnover_scaled = float(sim_res["turnover_weight"])
        overlay_active_ratio = (
            float((overlay_weights.abs().sum(axis=1) > 1e-12).mean())
            if len(overlay_weights) > 0
            else 0.0
        )
        overlay_time_in_market_pct = overlay_active_ratio * 100.0
        overlay_rebalances = int((rebal_mask & (leverage_series > 1e-12)).sum())
        quantization_events = int(sim_res["quantization_events"])
        skipped_rebalances = int(sim_res["skipped_rebalances"])

    hourly_overlay_pnl = net_usd
    hourly_overlay_fees = fees_usd

    # 9. Xây dựng attribution_series phân rã 7 cột
    carry_attr = carry_result["attribution_series"]
    if is_overlay_inactive:
        carry_funding = carry_attr["funding_pnl"].reindex(common_index).fillna(0.0)
        carry_basis = carry_attr["basis_pnl"].reindex(common_index).fillna(0.0)
        carry_fees = carry_attr["fees"].reindex(common_index).fillna(0.0)
    else:
        equity_prev = equity_curve.shift(1).fillna(config.initial_capital)
        carry_eq_prev = carry_result["equity_curve"].shift(1).fillna(config.initial_capital)
        if bool(getattr(config, "reinvest_profits", True)):
            scale_ratio = np.where(carry_eq_prev > 0.0, equity_prev / carry_eq_prev, 1.0)
        else:
            scale_ratio = np.ones(len(common_index), dtype=np.float64)
        carry_funding = carry_attr["funding_pnl"].reindex(common_index).fillna(0.0) * scale_ratio
        carry_basis = carry_attr["basis_pnl"].reindex(common_index).fillna(0.0) * scale_ratio
        carry_fees = carry_attr["fees"].reindex(common_index).fillna(0.0) * scale_ratio

    total_fees_series = carry_fees + fees_usd
    hourly_net_pnl = carry_funding + carry_basis + gross_usd - total_fees_series
    if len(hourly_net_pnl) > 0:
        hourly_net_pnl.iloc[0] = 0.0

    attribution_series = pd.DataFrame(
        {
            "funding_pnl": carry_funding,
            "basis_pnl": carry_basis,
            "overlay_gross_pnl": gross_usd,
            "overlay_pnl": gross_usd - fees_usd,
            "carry_fees": carry_fees,
            "overlay_fees": fees_usd,
            "fees": total_fees_series,
            "net_pnl": hourly_net_pnl,
        },
        index=common_index,
        dtype="float64",
    )
    attribution_series.index.name = "datetime"

    # 10. Tính toán các chỉ số hiệu suất Barbell & Metrics mới
    total_hours = len(common_index)
    years = total_hours / float(periods_per_year) if periods_per_year > 0 else 0.0
    final_equity = (
        float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else config.initial_capital
    )

    if config.initial_capital > 0.0:
        total_return_pct = float(
            ((final_equity - config.initial_capital) / config.initial_capital) * 100.0
        )
        if years > 0.0:
            if final_equity > 0.0:
                annualized_return_pct = float(
                    ((final_equity / config.initial_capital) ** (1.0 / years) - 1.0)
                    * 100.0
                )
            else:
                annualized_return_pct = -100.0
        else:
            annualized_return_pct = 0.0
    else:
        total_return_pct = 0.0
        annualized_return_pct = 0.0

    ret_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if ret_std > 1e-12 and len(returns) > 1:
        sharpe = float((returns.mean() / ret_std) * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    equity_arr_full = equity_curve.to_numpy(dtype=np.float64)
    peak_curve = np.maximum.accumulate(equity_arr_full)
    valid_peak = peak_curve > 0
    if np.any(valid_peak):
        dd_curve = (peak_curve[valid_peak] - equity_arr_full[valid_peak]) / peak_curve[valid_peak]
        max_dd_pct = float(np.nanmax(dd_curve) * 100.0)
    else:
        max_dd_pct = 0.0

    funding_collected = float(carry_funding.sum())
    total_fees_paid = float(total_fees_series.sum())
    carry_turnover = float(carry_result.get("turnover", 0.0))
    total_turnover = carry_turnover + overlay_turnover_scaled
    carry_rebalances = int(carry_result.get("metrics", {}).get("Price Rebalances", 0))

    overlay_gate_on_ratio = (
        float(gate_series.mean()) if len(gate_series) > 0 else 0.0
    )
    overlay_mean_leverage = (
        float(leverage_series.mean()) if len(leverage_series) > 0 else 0.0
    )
    valid_snr = raw_snr_dispersion.dropna()
    snr_dispersion_mean = (
        float(valid_snr.mean()) if not valid_snr.empty else 0.0
    )
    drawdown_cushion_events = int((dd_series < 1.0 - 1e-9).sum())

    metrics = {
        "Total Return %": total_return_pct,
        "Total Return % ": total_return_pct,
        "Annualized Return %": annualized_return_pct,
        "Sharpe": sharpe,
        "MaxDD %": max_dd_pct,
        "Total Funding Collected": funding_collected,
        "Total Fees Paid": total_fees_paid,
        "Overlay PnL": float(overlay_pnl_val),
        "Overlay Gross PnL": float(overlay_gross_pnl_val),
        "Overlay Net PnL": float(overlay_net_pnl_val),
        "Overlay Time in Market %": float(overlay_time_in_market_pct),
        "Carry Weight": 1.0,
        "Margin Calls": 0,
        "Num Rebalances": (
            int(overlay_rebalances) if config.enable_overlay else int(carry_rebalances)
        ),
        "Overlay Rebalances": int(overlay_rebalances),
        "Carry Rebalances": int(carry_rebalances),
        "Price Rebalances": int(carry_rebalances),
        "Net PnL": float(final_equity - config.initial_capital),
        "Turnover": float(total_turnover),
        "Time in Market %": float(
            carry_result.get("metrics", {}).get("Time in Market %", 100.0)
        ),
        "overlay_active_ratio": overlay_active_ratio,
        "overlay_gate_on_ratio": overlay_gate_on_ratio,
        "overlay_mean_leverage": overlay_mean_leverage,
        "snr_dispersion_mean": snr_dispersion_mean,
        "drawdown_cushion_events": drawdown_cushion_events,
        "quantization_events": quantization_events,
        "skipped_rebalances": skipped_rebalances,
        "overlay_gross_pnl": float(overlay_gross_pnl_val),
        "overlay_net_pnl": float(overlay_net_pnl_val),
        "overlay_fees": float(fees_usd.sum()),
        "overlay_quantization_events": int(quantization_events),
        "overlay_skipped_rebalances": int(skipped_rebalances),
        "carry_variant": config.carry_variant,
        "gating_mode": config.gating_mode,
        "carry_fallback": carry_fallback,
        "reinvest_profits": bool(config.reinvest_profits),
    }

    go_no_go_inputs = {
        "annualized_return_pct": annualized_return_pct,
        "sharpe": sharpe,
        "maxdd_pct": max_dd_pct,
        "net_pnl": float(final_equity - config.initial_capital),
        "final_equity": float(final_equity),
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "carry_result": carry_result,
        "overlay_result": overlay_result,
        "overlay_weights": overlay_weights,
        "attribution_series": attribution_series,
        "funding_pnl": funding_collected,
        "basis_pnl": float(carry_basis.sum()),
        "overlay_pnl": float(overlay_pnl_val),
        "overlay_gross_pnl": float(overlay_gross_pnl_val),
        "overlay_net_pnl": float(overlay_net_pnl_val),
        "overlay_fees": float(fees_usd.sum()),
        "overlay_turnover_scaled": float(overlay_turnover_scaled),
        "overlay_quantization_events": int(quantization_events),
        "overlay_skipped_rebalances": int(skipped_rebalances),
        "fees": total_fees_paid,
        "turnover": float(total_turnover),
        "metrics": metrics,
        "carry_variant": config.carry_variant,
        "gating_mode": config.gating_mode,
        "carry_fallback": carry_fallback,
        "reinvest_profits": bool(config.reinvest_profits),
        "go_no_go_inputs": go_no_go_inputs,
        "overlay_active_ratio": overlay_active_ratio,
        "overlay_leverage_series": leverage_series,
        "snr_dispersion": raw_snr_dispersion,
    }
