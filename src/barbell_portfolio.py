"""Module danh mục đầu tư Barbell (Core Yield Sleeve + Tactical Alpha Overlay).

Sprint 9 - Chặng 2b:
Cung cấp:
- BarbellConfig: Cấu hình tham số danh mục Barbell kết hợp Lõi Thu Nhập Carry và Lớp Tăng Trưởng Momentum.
- build_overlay_data: Chuyển đổi dữ liệu multi_aligned_data sang định dạng chuẩn cho Cross-Sectional.
- run_barbell_portfolio: Mô phỏng danh mục Barbell thống nhất trên tài khoản Unified Margin,
  không double-count vốn, tự động đóng overlay khi regime gate spread_zscore < min_spread.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.cross_exchange_carry import CrossExchangeCarryConfig, run_cross_exchange_carry
from src.cross_sectional import CrossSectionalConfig, run_cross_sectional_portfolio


@dataclass(frozen=True)
class BarbellConfig:
    """Cấu hình tham số cho danh mục đầu tư Barbell Portfolio (Sprint 9).

    Kết hợp 2 nguồn alpha:
    - Core Yield Sleeve: 100% NAV chạy Cross-Exchange Carry.
    - Tactical Alpha Overlay: 30% - 50% NAV chạy Core-4 Buffered XS Momentum
      trên phần dư ký quỹ (Unified Margin).
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


def run_barbell_portfolio(
    multi_aligned_data: dict[str, pd.DataFrame],
    config: BarbellConfig,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """Mô phỏng danh mục Barbell (Core Yield Sleeve + Tactical Alpha Overlay).

    Quy tắc tài chính & ký quỹ:
    - Core Yield Sleeve: 100% vốn nền (initial_capital) chạy Hybrid Cross-Exchange Carry.
    - Tactical Alpha Overlay: dùng phần dư ký quỹ trên tài khoản Perp chung, tỷ lệ
      gross leverage thực tế = overlay_gross_leverage (mặc định 0.40).
    - Regime Gate: khi spread_zscore < min_spread (0.75), engine cross-sectional tự động
      đưa toàn bộ trọng số về 0.0, overlay đứng ngoài hoàn toàn và không tốn phí.
    - Nếu enable_overlay=False hoặc trọng số overlay toàn 0: equity_curve Barbell phải
      trùng khít 100% với carry_result["equity_curve"] (sai số < 1e-9).

    Tham số:
        multi_aligned_data: Dict map {symbol: DataFrame đã đồng bộ đa sàn}.
        config: Cấu hình BarbellConfig.
        periods_per_year: Số chu kỳ mỗi năm (8760 cho nến 1h).

    Trả về:
        Dict chứa đầy đủ:
        - equity_curve: pd.Series đường cong vốn 1h (khởi đầu = initial_capital).
        - returns: pd.Series tỷ suất lợi nhuận 1h Barbell (r_barbell = r_carry + r_overlay_scaled).
        - carry_result: Dict gốc từ run_cross_exchange_carry.
        - overlay_result: Dict gốc từ run_cross_sectional_portfolio (hoặc mock dict khi tắt).
        - overlay_weights: pd.DataFrame trọng số overlay đã scale theo overlay_gross_leverage.
        - attribution_series: pd.DataFrame phân rã PnL 1h (funding_pnl, basis_pnl, overlay_pnl,
          carry_fees, overlay_fees, fees, net_pnl).
        - funding_pnl, basis_pnl, overlay_pnl, fees, turnover: Các giá trị tổng hợp lũy kế.
        - metrics: Dict các chỉ số hiệu suất chuẩn Sprint 9.
        - go_no_go_inputs: Dict chứa annualized_return_pct, sharpe, maxdd_pct, net_pnl, final_equity.
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
            "fees": 0.0,
            "turnover": 0.0,
            "metrics": empty_metrics,
            "go_no_go_inputs": {
                "annualized_return_pct": 0.0,
                "sharpe": 0.0,
                "maxdd_pct": 0.0,
                "net_pnl": 0.0,
                "final_equity": config.initial_capital,
            },
        }

    # 3. Chạy Core Yield Sleeve (100% Vốn nền)
    carry_cfg = config.carry_config
    if carry_cfg.initial_capital != config.initial_capital:
        carry_cfg = replace(carry_cfg, initial_capital=config.initial_capital)

    carry_result = run_cross_exchange_carry(
        multi_aligned_data,
        config=carry_cfg,
        periods_per_year=periods_per_year,
    )

    common_index = carry_result["equity_curve"].index
    symbols = list(multi_aligned_data.keys())

    # 4. Chạy hoặc bỏ qua Tactical Alpha Overlay
    ov_perp_fee = 0.0003
    ov_slippage = 0.0002
    if config.overlay_execution == "real_taker":
        ov_perp_fee = 0.0005
        ov_slippage = 0.0002

    if config.enable_overlay and config.overlay_gross_leverage > 0.0:
        overlay_data = build_overlay_data(multi_aligned_data)
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
            buffer_rank=config.overlay_buffer_rank,
            carry_tilt_weight=0.0,
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

    # 5. Căn chỉnh và scale lợi suất Overlay
    overlay_returns = overlay_result["returns"].reindex(common_index).fillna(0.0)
    r_overlay_scaled = overlay_returns * config.overlay_gross_leverage

    raw_weights = overlay_result.get("weights_df", pd.DataFrame())
    if not raw_weights.empty:
        raw_weights_aligned = raw_weights.reindex(common_index).fillna(0.0)
    else:
        raw_weights_aligned = pd.DataFrame(
            0.0, index=common_index, columns=symbols, dtype="float64"
        )
    overlay_weights = raw_weights_aligned * config.overlay_gross_leverage
    overlay_weights.index.name = "datetime"

    # Kiểm tra trạng thái đứng ngoài hoàn toàn (regime gate hoặc tắt overlay)
    is_overlay_inactive = (
        not config.enable_overlay
        or config.overlay_gross_leverage <= 0.0
        or (np.abs(overlay_weights.to_numpy()) < 1e-12).all()
        or (r_overlay_scaled.abs().max() < 1e-12)
    )

    r_carry = carry_result["returns"].reindex(common_index).fillna(0.0)

    if is_overlay_inactive:
        # Nếu overlay không hoạt động: equity_curve Barbell trùng khít 100% với Carry
        equity_curve = carry_result["equity_curve"].copy()
        returns = carry_result["returns"].copy()
        overlay_pnl_val = 0.0
        hourly_overlay_pnl = pd.Series(0.0, index=common_index, dtype="float64")
        hourly_overlay_fees = pd.Series(0.0, index=common_index, dtype="float64")
        overlay_time_in_market_pct = 0.0
        overlay_turnover_scaled = 0.0
        overlay_rebalances = 0
    else:
        r_barbell = r_carry + r_overlay_scaled
        equity_vals = config.initial_capital * np.cumprod(1.0 + r_barbell.to_numpy())
        equity_curve = pd.Series(
            equity_vals,
            index=common_index,
            dtype="float64",
            name="equity",
        )
        returns = pd.Series(r_barbell, index=common_index, dtype="float64", name="returns")

        # Đóng góp PnL của Overlay: initial_capital * ((1 + r_overlay_scaled).cumprod() - 1)
        overlay_cum_mult = np.cumprod(1.0 + r_overlay_scaled.to_numpy())
        overlay_pnl_val = float(config.initial_capital * (overlay_cum_mult[-1] - 1.0))

        overlay_dollar_curve = config.initial_capital * overlay_cum_mult
        hourly_overlay_pnl_vals = np.diff(
            np.insert(overlay_dollar_curve, 0, config.initial_capital)
        )
        hourly_overlay_pnl = pd.Series(
            hourly_overlay_pnl_vals, index=common_index, dtype="float64"
        )

        # Tính phí theo giờ của overlay từ nhật ký khớp lệnh trades_df
        one_way_cost = ov_perp_fee + ov_slippage
        trades_df = overlay_result.get("trades_df", pd.DataFrame())
        hourly_overlay_fees = pd.Series(0.0, index=common_index, dtype="float64")
        if (
            not trades_df.empty
            and "turnover" in trades_df.columns
            and "time" in trades_df.columns
        ):
            turnover_by_time = trades_df.groupby("time")["turnover"].sum()
            scaled_fees_by_time = (
                turnover_by_time
                * config.overlay_gross_leverage
                * one_way_cost
                * config.initial_capital
            )
            valid_times = scaled_fees_by_time.index.intersection(common_index)
            hourly_overlay_fees.loc[valid_times] = scaled_fees_by_time.loc[valid_times].to_numpy()

        overlay_time_in_market_pct = float(
            overlay_result.get("metrics", {}).get("Time in Market %", 0.0)
        )
        overlay_turnover_scaled = float(
            overlay_result.get("metrics", {}).get("Turnover", 0.0)
            * config.overlay_gross_leverage
        )
        overlay_rebalances = int(
            overlay_result.get("metrics", {}).get("Num Rebalances", 0)
        )

    # 6. Xây dựng attribution_series phân rã 7 cột
    carry_attr = carry_result["attribution_series"]
    carry_funding = carry_attr["funding_pnl"].reindex(common_index).fillna(0.0)
    carry_basis = carry_attr["basis_pnl"].reindex(common_index).fillna(0.0)
    carry_fees = carry_attr["fees"].reindex(common_index).fillna(0.0)
    total_fees_series = carry_fees + hourly_overlay_fees

    if is_overlay_inactive:
        hourly_net_pnl = carry_attr["net_pnl"].reindex(common_index).fillna(0.0)
    else:
        hourly_net_pnl = (
            carry_attr["net_pnl"].reindex(common_index).fillna(0.0) + hourly_overlay_pnl
        )

    attribution_series = pd.DataFrame(
        {
            "funding_pnl": carry_funding,
            "basis_pnl": carry_basis,
            "overlay_pnl": hourly_overlay_pnl,
            "carry_fees": carry_fees,
            "overlay_fees": hourly_overlay_fees,
            "fees": total_fees_series,
            "net_pnl": hourly_net_pnl,
        },
        index=common_index,
        dtype="float64",
    )
    attribution_series.index.name = "datetime"

    # 7. Tính toán các chỉ số hiệu suất Barbell
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

    equity_arr = equity_curve.to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(equity_arr)
    valid_peak = peak > 0
    if np.any(valid_peak):
        dd = (peak[valid_peak] - equity_arr[valid_peak]) / peak[valid_peak]
        max_dd_pct = float(np.nanmax(dd) * 100.0)
    else:
        max_dd_pct = 0.0

    funding_collected = float(carry_result.get("funding_pnl", 0.0))
    total_fees_paid = float(carry_result.get("fees", 0.0) + hourly_overlay_fees.sum())
    carry_turnover = float(carry_result.get("turnover", 0.0))
    total_turnover = carry_turnover + overlay_turnover_scaled
    carry_rebalances = int(carry_result.get("metrics", {}).get("Price Rebalances", 0))

    metrics = {
        "Total Return %": total_return_pct,
        "Total Return % ": total_return_pct,
        "Annualized Return %": annualized_return_pct,
        "Sharpe": sharpe,
        "MaxDD %": max_dd_pct,
        "Total Funding Collected": funding_collected,
        "Total Fees Paid": total_fees_paid,
        "Overlay PnL": float(overlay_pnl_val),
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
        "basis_pnl": float(carry_result.get("basis_pnl", 0.0)),
        "overlay_pnl": float(overlay_pnl_val),
        "fees": total_fees_paid,
        "turnover": float(total_turnover),
        "metrics": metrics,
        "go_no_go_inputs": go_no_go_inputs,
    }
