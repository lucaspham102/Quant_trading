"""Chiến lược Funding Carry Delta-Neutral đa tài sản với Bar-by-Bar MTM.

Bao gồm:
- CarryConfig: Cấu hình tham số chiến lược, chế độ giao dịch và mô hình chi phí thực thi.
- run_carry_portfolio: Mô phỏng danh mục carry đa tài sản với tính toán MTM từng nến,
  quản lý ký quỹ và tái cân bằng đệm rủi ro không có look-ahead bias.
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

    carry_mode: str = "hysteresis"  # "hysteresis" | "passive_long_term"
    passive_entry_ema_span: int = 21
    passive_exit_ema_span: int = 42
    passive_entry_threshold: float = 0.00005
    passive_exit_threshold: float = -0.00002
    passive_min_hold_periods: int = 21
    maintenance_margin_ratio: float = 0.05
    margin_rebalance_target_ratio: float = 1.0 / 3.0
    rebalance_price_trigger_pct: float = 0.40
    rebalance_fee_multiplier: float = 2.0

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
        - trades_df: pd.DataFrame nhật ký mở/đóng lệnh và tái cân bằng.
        - metrics: Dict các chỉ số hiệu suất tổng hợp.
    """
    # 0. Xác thực hợp lệ cấu hình chiến lược
    if config.carry_mode not in {"hysteresis", "passive_long_term"}:
        raise ValueError(
            f"carry_mode không hợp lệ: '{config.carry_mode}'. "
            "Chỉ chấp nhận 'hysteresis' hoặc 'passive_long_term'."
        )
    if not (0.0 < config.maintenance_margin_ratio < 1.0):
        raise ValueError(
            f"maintenance_margin_ratio phải nằm trong khoảng (0, 1), nhận được: {config.maintenance_margin_ratio}"
        )
    if config.rebalance_price_trigger_pct <= 0.0:
        raise ValueError(
            f"rebalance_price_trigger_pct phải > 0, nhận được: {config.rebalance_price_trigger_pct}"
        )

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
        "Total MTM PnL": 0.0,
        "Total Basis Noise": 0.0,
        "Num Rebalances": 0,
        "Margin Rebalances": 0,
        "Price Rebalances": 0,
    }

    if not aligned_data:
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": pd.DataFrame(
                columns=["symbol", "time", "action", "notional", "fee", "funding_cum"]
            ),
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
    n_steps = len(common_index)

    if n_steps == 0:
        empty_eq = pd.Series([], index=common_index, dtype="float64", name="equity")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=common_index, dtype="float64"),
            "trades_df": pd.DataFrame(
                columns=["symbol", "time", "action", "notional", "fee", "funding_cum"]
            ),
            "metrics": empty_metrics,
        }

    # Trích xuất dữ liệu giá và tín hiệu per-symbol
    perp_prices: dict[str, np.ndarray] = {}
    spot_prices: dict[str, np.ndarray] = {}
    basis_values: dict[str, np.ndarray] = {}
    ema_entry_map: dict[str, pd.Series] = {}
    ema_exit_map: dict[str, pd.Series] = {}
    funding_event_map: dict[str, pd.Series] = {}

    for sym, d in clean_aligned.items():
        # Giá perp: 'perp_close' nếu có else 'close'
        if "perp_close" in d.columns:
            perp_s = d["perp_close"]
        elif "close" in d.columns:
            perp_s = d["close"]
        else:
            perp_s = pd.Series(np.nan, index=d.index)
        perp_arr = perp_s.reindex(common_index).to_numpy(dtype=np.float64)
        perp_prices[sym] = perp_arr

        # Giá spot: 'spot_close' nếu có else perp
        if "spot_close" in d.columns:
            spot_arr = d["spot_close"].reindex(common_index).to_numpy(dtype=np.float64)
        else:
            spot_arr = perp_arr.copy()
        spot_prices[sym] = spot_arr

        # Giá basis: 'basis_pct' nếu có else 0.0 (không có cột -> MTM = 0)
        if "basis_pct" in d.columns:
            basis_arr = d["basis_pct"].reindex(common_index).to_numpy(dtype=np.float64)
        else:
            basis_arr = np.zeros(n_steps, dtype=np.float64)
        basis_values[sym] = basis_arr

        # Tín hiệu funding rate
        lk_fr = (
            d["last_known_funding_rate"]
            if "last_known_funding_rate" in d.columns
            else pd.Series(0.0, index=d.index)
        )
        if config.carry_mode == "hysteresis":
            ema = lk_fr.ewm(span=config.ema_span).mean().shift(1)
            reindexed_ema = ema.reindex(common_index).ffill().fillna(0.0)
            ema_entry_map[sym] = reindexed_ema
            ema_exit_map[sym] = reindexed_ema
        else:  # passive_long_term
            ema_in = lk_fr.ewm(span=config.passive_entry_ema_span).mean().shift(1)
            ema_out = lk_fr.ewm(span=config.passive_exit_ema_span).mean().shift(1)
            ema_entry_map[sym] = ema_in.reindex(common_index).ffill().fillna(0.0)
            ema_exit_map[sym] = ema_out.reindex(common_index).ffill().fillna(0.0)

        fr_event_s = (
            d["funding_rate_event"]
            if "funding_rate_event" in d.columns
            else pd.Series(0.0, index=d.index)
        )
        funding_event_map[sym] = fr_event_s.reindex(common_index).fillna(0.0)

    # Khởi tạo trạng thái danh mục
    equity = float(config.initial_capital)
    cost_rate = float(
        config.one_way_cost() if callable(config.one_way_cost) else config.one_way_cost
    )
    eta = float(
        config.notional_eta() if callable(config.notional_eta) else config.notional_eta
    )
    rng = np.random.default_rng(seed=42)

    in_position: dict[str, bool] = {sym: False for sym in clean_aligned}
    periods_held: dict[str, int] = {sym: 0 for sym in clean_aligned}
    notional: dict[str, float] = {sym: 0.0 for sym in clean_aligned}
    Q_qty: dict[str, float] = {sym: 0.0 for sym in clean_aligned}
    margin_balance: dict[str, float] = {sym: 0.0 for sym in clean_aligned}
    entry_spot_price: dict[str, float] = {sym: 0.0 for sym in clean_aligned}
    entry_perp_price: dict[str, float] = {sym: 0.0 for sym in clean_aligned}
    trade_funding_cum: dict[str, float] = {sym: 0.0 for sym in clean_aligned}

    total_funding_collected = 0.0
    total_fees_paid = 0.0
    total_notional_traded = 0.0
    total_mtm_pnl = 0.0
    total_basis_noise = 0.0
    num_rebalances = 0
    margin_rebalances = 0
    price_rebalances = 0

    trades_list: list[dict[str, Any]] = []
    equity_vals = np.empty(n_steps, dtype=np.float64)
    in_market_any = np.zeros(n_steps, dtype=bool)

    # Vòng lặp mô phỏng bar-by-bar trên trục thời gian chung
    for t_idx, t in enumerate(common_index):
        just_opened: set[str] = set()

        # 1) MTM mỗi nến t khi đang giữ vị thế
        for sym in clean_aligned:
            if in_position[sym]:
                if t_idx > 0:
                    if config.structure == "spot_perp_2leg":
                        sp_prev = spot_prices[sym][t_idx - 1]
                        sp_curr = spot_prices[sym][t_idx]
                        pe_prev = perp_prices[sym][t_idx - 1]
                        pe_curr = perp_prices[sym][t_idx]
                        if (
                            np.isnan(sp_prev)
                            or np.isnan(sp_curr)
                            or np.isnan(pe_prev)
                            or np.isnan(pe_curr)
                        ):
                            mtm = 0.0
                        else:
                            mtm = Q_qty[sym] * (sp_curr - sp_prev) - Q_qty[sym] * (
                                pe_curr - pe_prev
                            )
                    else:  # coinm_1leg
                        ba_prev = basis_values[sym][t_idx - 1]
                        ba_curr = basis_values[sym][t_idx]
                        if np.isnan(ba_prev) or np.isnan(ba_curr):
                            mtm = 0.0
                        else:
                            mtm = -notional[sym] * (ba_curr - ba_prev)
                else:
                    mtm = 0.0

                equity += mtm
                margin_balance[sym] += mtm
                total_mtm_pnl += mtm

        # 2) Nhận Funding PnL nếu đang giữ vị thế và nến t là mốc thanh toán funding
        for sym in clean_aligned:
            if in_position[sym]:
                fr_event = float(funding_event_map[sym].iloc[t_idx])
                if fr_event != 0.0:
                    funding_pnl = notional[sym] * fr_event
                    equity += funding_pnl
                    margin_balance[sym] += funding_pnl
                    trade_funding_cum[sym] += funding_pnl
                    total_funding_collected += funding_pnl
                    periods_held[sym] += 1

        # 3) Kiểm tra ký quỹ và tái cân bằng (rebalance, tối đa 1 lần/symbol/nến)
        for sym in clean_aligned:
            if in_position[sym]:
                sp_curr = spot_prices[sym][t_idx]
                pe_curr = perp_prices[sym][t_idx]
                curr_notional = notional[sym]
                margin_ratio = margin_balance[sym] / max(curr_notional, 1e-12)

                margin_trig = margin_ratio < config.maintenance_margin_ratio
                price_trig = (
                    config.structure == "spot_perp_2leg"
                    and entry_spot_price[sym] > 0
                    and not np.isnan(sp_curr)
                    and sp_curr
                    > entry_spot_price[sym] * (1.0 + config.rebalance_price_trigger_pct)
                )

                if margin_trig:
                    reason = "margin"
                elif price_trig:
                    reason = "price"
                else:
                    reason = None

                if reason is not None:
                    if config.structure == "spot_perp_2leg":
                        if not np.isnan(sp_curr) and sp_curr > 0:
                            Q_new = min(Q_qty[sym], curr_notional / sp_curr)
                            delta = abs(Q_qty[sym] - Q_new) * sp_curr
                        else:
                            delta = 0.0
                    else:  # coinm_1leg
                        notional_new = (
                            margin_balance[sym] / config.margin_rebalance_target_ratio
                        )
                        if notional_new < curr_notional:
                            delta = curr_notional - notional_new
                        else:
                            delta = 0.0

                    if delta > 1e-12:
                        reb_fee = config.rebalance_fee_multiplier * cost_rate * delta
                        equity -= reb_fee
                        margin_balance[sym] -= reb_fee
                        total_fees_paid += reb_fee
                        total_notional_traded += delta

                        if config.structure == "spot_perp_2leg":
                            Q_qty[sym] = Q_new
                            notional[sym] = Q_new * sp_curr
                        else:
                            notional[sym] = notional_new

                        entry_spot_price[sym] = sp_curr
                        entry_perp_price[sym] = pe_curr
                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "rebalance",
                            "notional": float(delta),
                            "fee": float(reb_fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                            "reason": reason,
                        })
                        num_rebalances += 1
                        if reason == "margin":
                            margin_rebalances += 1
                        else:
                            price_rebalances += 1

        # 4) Xét đổi trạng thái mở / đóng vị thế
        for sym in clean_aligned:
            ema_in_val = float(ema_entry_map[sym].iloc[t_idx])
            ema_out_val = float(ema_exit_map[sym].iloc[t_idx])

            if not in_position[sym]:
                # Điều kiện mở vị thế
                if config.carry_mode == "hysteresis":
                    signal_entry = ema_in_val >= config.entry_threshold
                else:  # passive_long_term
                    signal_entry = ema_in_val > config.passive_entry_threshold

                if signal_entry and equity > 0:
                    allocated = 0.25 * equity
                    leg_notional = allocated * eta
                    sp_t0 = spot_prices[sym][t_idx]
                    pe_t0 = perp_prices[sym][t_idx]

                    if np.isnan(sp_t0) or sp_t0 <= 0:
                        sp_t0 = 1.0
                    if np.isnan(pe_t0) or pe_t0 <= 0:
                        pe_t0 = sp_t0

                    Q_val = leg_notional / sp_t0
                    Q_qty[sym] = Q_val
                    entry_spot_price[sym] = sp_t0
                    entry_perp_price[sym] = pe_t0
                    notional[sym] = leg_notional

                    if config.structure == "spot_perp_2leg":
                        margin_balance[sym] = allocated - leg_notional
                    else:  # coinm_1leg
                        margin_balance[sym] = allocated

                    fee = leg_notional * cost_rate
                    equity -= fee
                    margin_balance[sym] -= fee
                    total_fees_paid += fee
                    total_notional_traded += leg_notional

                    in_position[sym] = True
                    periods_held[sym] = 0
                    trade_funding_cum[sym] = 0.0
                    just_opened.add(sym)

                    trades_list.append({
                        "symbol": sym,
                        "time": t,
                        "action": "open",
                        "notional": float(leg_notional),
                        "fee": float(fee),
                        "funding_cum": 0.0,
                    })
            else:
                # Điều kiện đóng vị thế: chỉ xét khi không phải cây nến vừa mở
                if sym not in just_opened:
                    if config.carry_mode == "hysteresis":
                        signal_exit = (
                            ema_out_val <= config.exit_threshold
                            and periods_held[sym] >= config.min_hold_periods
                        )
                    else:  # passive_long_term
                        req_periods = max(
                            config.min_hold_periods, config.passive_min_hold_periods
                        )
                        signal_exit = (
                            ema_out_val < config.passive_exit_threshold
                            and periods_held[sym] >= req_periods
                        )

                    if signal_exit:
                        pos_notional = notional[sym]
                        fee = pos_notional * cost_rate
                        equity -= fee
                        total_fees_paid += fee
                        total_notional_traded += pos_notional

                        if config.basis_noise_std > 0.0:
                            noise = (
                                pos_notional
                                * config.basis_noise_std
                                * float(rng.normal(0.0, 1.0))
                            )
                            equity -= noise
                            total_basis_noise += noise

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
                        Q_qty[sym] = 0.0
                        margin_balance[sym] = 0.0
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0
                        entry_spot_price[sym] = 0.0
                        entry_perp_price[sym] = 0.0

        # 5) Ghi nhận equity_t sau tất cả cập nhật
        equity_vals[t_idx] = equity
        in_market_any[t_idx] = any(in_position.values())

    equity_curve = pd.Series(
        equity_vals, index=common_index, name="equity", dtype="float64"
    )
    returns = equity_curve.pct_change().fillna(0.0)

    base_cols = ["symbol", "time", "action", "notional", "fee", "funding_cum"]
    if trades_list:
        has_reason = any("reason" in tr for tr in trades_list)
        cols = base_cols + (["reason"] if has_reason else [])
        trades_df = pd.DataFrame(trades_list)
        trades_df = trades_df.reindex(columns=cols)
    else:
        trades_df = pd.DataFrame(columns=base_cols)

    # Tính toán các chỉ số hiệu suất
    total_return_pct = (
        float((equity - config.initial_capital) / config.initial_capital * 100.0)
        if config.initial_capital > 0
        else 0.0
    )

    if len(equity_curve) > 0 and config.initial_capital > 0 and equity > 0:
        years = len(equity_curve) / float(periods_per_year)
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

    time_in_market_pct = (
        float(np.mean(in_market_any) * 100.0) if len(in_market_any) > 0 else 0.0
    )
    turnover = (
        float(total_notional_traded / config.initial_capital)
        if config.initial_capital > 0
        else 0.0
    )

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
        "Total MTM PnL": float(total_mtm_pnl),
        "Total Basis Noise": float(total_basis_noise),
        "Num Rebalances": int(num_rebalances),
        "Margin Rebalances": int(margin_rebalances),
        "Price Rebalances": int(price_rebalances),
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades_df": trades_df,
        "metrics": metrics,
    }
