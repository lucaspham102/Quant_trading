"""Chiến lược Funding Carry Delta-Neutral đa tài sản.

Bao gồm:
- CarryConfig: Cấu hình tham số chiến lược và mô hình chi phí thực thi.
- run_carry_portfolio: Mô phỏng danh mục carry đa tài sản không có look-ahead bias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CarryConfig:
    """Cấu hình tham số cho chiến lược Funding Carry Delta-Neutral."""

    structure: str = "coinm_1leg"
    ema_span: int = 6
    entry_threshold: float = 0.0001
    exit_threshold: float = 0.0
    min_hold_periods: int = 3
    perp_fee: float = 0.0005
    spot_fee: float = 0.00075
    slippage: float = 0.0002
    initial_capital: float = 400.0
    basis_noise_std: float = 0.0002

    def one_way_cost(self) -> float:
        """Chi phí 1 chiều (phí giao dịch + trượt giá) theo cấu trúc thực thi."""
        if self.structure == "coinm_1leg":
            return self.perp_fee + self.slippage
        elif self.structure == "spot_perp_2leg":
            return self.spot_fee + self.slippage + self.perp_fee + self.slippage
        raise ValueError(f"Cấu trúc thực thi không hợp lệ: {self.structure}")

    def notional_eta(self) -> float:
        """Tỷ lệ vốn hiệu dụng nhận funding (1.0 cho 1 chân Coin-M, 2/3 cho Spot + Perp 2 chân)."""
        if self.structure == "coinm_1leg":
            return 1.0
        elif self.structure == "spot_perp_2leg":
            return 2.0 / 3.0
        raise ValueError(f"Cấu trúc thực thi không hợp lệ: {self.structure}")


def run_carry_portfolio(
    aligned_data: dict[str, pd.DataFrame],
    config: CarryConfig,
    periods_per_year: int = 8760,  # 8760 nến 1h mỗi năm làm mặc định
) -> dict[str, Any]:
    """Mô phỏng danh mục Funding Carry đa tài sản (tối đa 25% NAV mỗi symbol).

    Tham số:
        aligned_data: Dict map {symbol: ohlcv_df_đã_đồng_bộ_funding}.
        config: CarryConfig chứa tham số chiến lược và ma trận chi phí.
        periods_per_year: Số chu kỳ nến mỗi năm dùng để năm hóa Sharpe và Return (mặc định 8760 cho nến 1h).

    Trả về:
        Dict gồm:
        - equity_curve: pd.Series đường cong vốn trên trục thời gian chung.
        - returns: pd.Series tỷ suất lợi nhuận pct_change().fillna(0.0).
        - trades_df: pd.DataFrame nhật ký mở/đóng lệnh.
        - metrics: Dict các chỉ số hiệu suất tổng hợp.
    """
    empty_idx = pd.DatetimeIndex([], tz="UTC", name="datetime").as_unit("ns")
    empty_metrics = {
        "Total Return %": 0.0,
        "Total Return % ": 0.0,
        "Annualized Return %": 0.0,
        "Sharpe": 0.0,
        "MaxDD %": 0.0,
        "Total Funding Collected": 0.0,
        "Total Fees Paid": 0.0,
        "Time in Market %": 0.0,
        "Turnover": 0.0,
    }

    if not aligned_data:
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": pd.DataFrame(columns=["symbol", "time", "action", "notional", "fee", "funding_cum"]),
            "metrics": empty_metrics,
        }

    # Chuẩn hóa DatetimeIndex UTC ns tên 'datetime' cho từng symbol
    clean_aligned: dict[str, pd.DataFrame] = {}
    for sym, df in aligned_data.items():
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

    # Trục thời gian chung: union index của các symbol, sort tăng dần
    all_ts = sorted(set().union(*(d.index for d in clean_aligned.values())))
    common_index = pd.DatetimeIndex(all_ts, tz="UTC").as_unit("ns")
    common_index.name = "datetime"

    if len(common_index) == 0:
        empty_eq = pd.Series([], index=common_index, dtype="float64", name="equity")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=common_index, dtype="float64"),
            "trades_df": pd.DataFrame(columns=["symbol", "time", "action", "notional", "fee", "funding_cum"]),
            "metrics": empty_metrics,
        }

    # Với mỗi symbol i: tính ema_funding = aligned["last_known_funding_rate"].ewm(span=config.ema_span).mean().shift(1)
    ema_funding_map: dict[str, pd.Series] = {}
    funding_event_map: dict[str, pd.Series] = {}
    for sym, d in clean_aligned.items():
        ema = d["last_known_funding_rate"].ewm(span=config.ema_span).mean().shift(1)
        ema_funding_map[sym] = ema.reindex(common_index).ffill().fillna(0.0)
        funding_event_map[sym] = d["funding_rate_event"].reindex(common_index).fillna(0.0)

    # Khởi tạo trạng thái danh mục
    equity = float(config.initial_capital)
    cost_rate = float(config.one_way_cost() if callable(config.one_way_cost) else config.one_way_cost)
    eta = float(config.notional_eta() if callable(config.notional_eta) else config.notional_eta)
    rng = np.random.default_rng(seed=42)

    in_position: dict[str, bool] = {sym: False for sym in clean_aligned}
    periods_held: dict[str, int] = {sym: 0 for sym in clean_aligned}
    notional: dict[str, float] = {sym: 0.0 for sym in clean_aligned}
    trade_funding_cum: dict[str, float] = {sym: 0.0 for sym in clean_aligned}

    total_funding_collected = 0.0
    total_fees_paid = 0.0
    total_notional_traded = 0.0

    trades_list: list[dict[str, Any]] = []
    n_steps = len(common_index)
    equity_vals = np.empty(n_steps, dtype=np.float64)
    in_market_any = np.zeros(n_steps, dtype=bool)

    # Vòng lặp mô phỏng trên trục thời gian chung
    for t_idx, t in enumerate(common_index):
        just_opened: set[str] = set()

        # 1) Nhận Funding PnL nếu đang giữ vị thế và nến t là mốc thanh toán funding
        for sym in clean_aligned:
            if in_position[sym]:
                fr_event = float(funding_event_map[sym].iloc[t_idx])
                if fr_event != 0.0:
                    funding_pnl = notional[sym] * fr_event
                    equity += funding_pnl
                    trade_funding_cum[sym] += funding_pnl
                    total_funding_collected += funding_pnl
                    periods_held[sym] += 1

        # 2) Xét đổi trạng thái mở / đóng vị thế
        for sym in clean_aligned:
            ema_val = float(ema_funding_map[sym].iloc[t_idx])

            if not in_position[sym]:
                # Điều kiện mở vị thế: ema_funding >= entry_threshold
                if ema_val >= config.entry_threshold and equity > 0:
                    pos_notional = 0.25 * equity * eta
                    fee = pos_notional * cost_rate
                    equity -= fee
                    total_fees_paid += fee
                    total_notional_traded += pos_notional

                    in_position[sym] = True
                    notional[sym] = pos_notional
                    periods_held[sym] = 0
                    trade_funding_cum[sym] = 0.0
                    just_opened.add(sym)

                    trades_list.append({
                        "symbol": sym,
                        "time": t,
                        "action": "open",
                        "notional": float(pos_notional),
                        "fee": float(fee),
                        "funding_cum": 0.0,
                    })
            else:
                # Điều kiện đóng vị thế: chỉ xét khi không phải cây nến vừa mở
                if sym not in just_opened:
                    if ema_val <= config.exit_threshold and periods_held[sym] >= config.min_hold_periods:
                        pos_notional = notional[sym]
                        fee = pos_notional * cost_rate
                        equity -= fee
                        total_fees_paid += fee
                        total_notional_traded += pos_notional

                        if config.basis_noise_std > 0.0:
                            noise = pos_notional * config.basis_noise_std * float(rng.normal(0.0, 1.0))
                            equity -= noise

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "close",
                            "notional": float(pos_notional),
                            "fee": float(fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                        })

                        in_position[sym] = False
                        notional[sym] = 0.0
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0

        # 3) Ghi nhận equity_t sau tất cả cập nhật
        equity_vals[t_idx] = equity
        in_market_any[t_idx] = any(in_position.values())

    equity_curve = pd.Series(equity_vals, index=common_index, name="equity", dtype="float64")
    returns = equity_curve.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(
        trades_list,
        columns=["symbol", "time", "action", "notional", "fee", "funding_cum"],
    )

    # Tính toán các chỉ số hiệu suất
    total_return_pct = float((equity - config.initial_capital) / config.initial_capital * 100.0) if config.initial_capital > 0 else 0.0

    if len(equity_curve) > 0 and config.initial_capital > 0 and equity > 0:
        years = len(equity_curve) / float(periods_per_year)
        if years > 0:
            annualized_return_pct = float(((equity / config.initial_capital) ** (1.0 / years) - 1.0) * 100.0)
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

    time_in_market_pct = float(np.mean(in_market_any) * 100.0) if len(in_market_any) > 0 else 0.0
    turnover = float(total_notional_traded / config.initial_capital) if config.initial_capital > 0 else 0.0

    metrics = {
        "Total Return %": total_return_pct,
        "Total Return % ": total_return_pct,
        "Annualized Return %": annualized_return_pct,
        "Sharpe": sharpe,
        "MaxDD %": max_dd_pct,
        "Total Funding Collected": float(total_funding_collected),
        "Total Fees Paid": float(total_fees_paid),
        "Time in Market %": time_in_market_pct,
        "Turnover": turnover,
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades_df": trades_df,
        "metrics": metrics,
    }
