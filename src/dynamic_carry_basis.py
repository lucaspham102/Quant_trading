"""Động cơ xoay vòng Funding Carry động và khai thác hội tụ Basis Spot-Perp.

Sprint 8 - Chặng 2:
Module cung cấp:
- DynamicCarryConfig: Cấu hình tham số chiến lược carry xoay vòng, mô hình chi phí và ký quỹ.
- build_carry_basis_scores: Tính điểm Carry + Basis chống look-ahead tuyệt đối.
- SelectionDecision & select_portfolio: Hàm thuần chọn danh mục tập trung có đệm thứ hạng,
  ngưỡng trễ hysteresis và yêu cầu lợi thế chuyển đổi tối thiểu.
- run_dynamic_carry_portfolio: Mô phỏng Bar-by-Bar True Spot-Perp MTM, khớp lệnh taker/maker,
  thanh toán funding, quản trị ký quỹ và phân rã PnL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicCarryConfig:
    """Cấu hình tham số cho chiến lược Dynamic Carry và Hội tụ Basis Spot-Perp."""

    # Cấu trúc vốn & thực thi
    structure: Literal["spot_perp_3x", "coinm_1leg"] = "spot_perp_3x"
    execution: Literal["real_taker", "passive_limit_maker"] = "real_taker"
    initial_capital: float = 400.0

    # Tham số tính điểm Carry + Basis (đơn vị tỷ lệ thập phân, 1 bps = 0.0001)
    ema_span: int = 6
    basis_ema_hours: int = 168
    expected_hold_periods: int = 21
    w_basis: float = 0.5

    # Tham số chọn danh mục & Buffer/Hysteresis
    top_k: int = 2
    entry_threshold: float = 0.00010
    exit_threshold: float = 0.00002
    min_hold_periods: int = 9
    buffer_rank: int = 1
    min_switch_advantage_override: float | None = None

    # Mô hình chi phí khóa cứng (đơn vị tỷ lệ thập phân)
    taker_spot_fee: float = 0.00075
    taker_perp_fee: float = 0.00050
    taker_slippage: float = 0.00020
    maker_spot_fee: float = 0.00050
    maker_perp_fee: float = 0.00020
    maker_entry_buffer_bps: float = 0.5

    # Tham số quản trị ký quỹ & tái cân bằng
    maintenance_margin_ratio: float = 0.05
    margin_rebalance_target_ratio: float = 1.0 / 3.0
    margin_rebalance_lower_ratio: float = 0.15
    margin_rebalance_upper_ratio: float = 0.50
    rebalance_price_trigger_pct: float = 0.35
    rebalance_fee_multiplier: float = 2.0

    # Sprint 15: che do cua Tier 1B (giu mac dinh "advantage" cho test Sprint 7-10)
    switch_rule: Literal["advantage", "tier1b_buffer_rank"] = "advantage"
    eta_override: float | None = None

    def notional_eta(self) -> float:
        """Tỷ lệ vốn hiệu dụng nhận funding (0.75 cho spot_perp_3x, 1.0 cho coinm_1leg)."""
        if self.eta_override is not None:
            return float(self.eta_override)
        if self.structure == "spot_perp_3x":
            return 0.75
        elif self.structure == "coinm_1leg":
            return 1.0
        raise ValueError(f"Cau truc vi the khong hop le: {self.structure}")

    def one_way_cost(self) -> float:
        """Chi phí 1 chiều (phí giao dịch + trượt giá) tính trên notional một chân."""
        if self.execution == "real_taker":
            if self.structure == "spot_perp_3x":
                return (
                    (self.taker_spot_fee + self.taker_slippage)
                    + (self.taker_perp_fee + self.taker_slippage)
                )
            elif self.structure == "coinm_1leg":
                return self.taker_perp_fee + self.taker_slippage
            raise ValueError(f"Cau truc vi the khong hop le: {self.structure}")
        elif self.execution == "passive_limit_maker":
            if self.structure == "spot_perp_3x":
                return self.maker_spot_fee + self.maker_perp_fee
            elif self.structure == "coinm_1leg":
                return self.maker_perp_fee
            raise ValueError(f"Cau truc vi the khong hop le: {self.structure}")
        raise ValueError(f"Phuong thuc thuc thi khong hop le: {self.execution}")

    def cost_round_trip(self) -> float:
        """Chi phí khứ hồi (mở + đóng) tính trên notional một chân."""
        return 2.0 * self.one_way_cost()

    def min_switch_advantage(self) -> float:
        """Lợi thế điểm tối thiểu cần thiết để đảo vị thế sang tài sản mới."""
        if self.min_switch_advantage_override is not None:
            return float(self.min_switch_advantage_override)
        return 2.0 * self.cost_round_trip() / float(self.expected_hold_periods)


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa DataFrame có DatetimeIndex UTC ns tên datetime."""
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        if "datetime" in d.columns:
            d = d.set_index("datetime")
        elif "timestamp" in d.columns:
            d["datetime"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
            d = d.set_index("datetime")
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.as_unit("ns")
    idx.name = "datetime"
    d.index = idx
    return d


def build_carry_basis_scores(
    aligned_data: dict[str, pd.DataFrame],
    config: DynamicCarryConfig,
    common_index: pd.DatetimeIndex | None = None,
) -> dict[str, pd.DataFrame]:
    """Tính toán chuỗi điểm Carry + Basis chống look-ahead bias cho từng tài sản.

    Tham số:
        aligned_data: Dict map {symbol: DataFrame} chuẩn hóa từ align_spot_perp_funding.
        config: DynamicCarryConfig cấu hình các tham số trượt và trọng số.
        common_index: Trục thời gian chung DatetimeIndex UTC ns (nếu None sẽ tự union).

    Trả về:
        Dict map {symbol: DataFrame} với index là common_index và các cột float64:
        ['f_ewma', 'basis_ema', 'basis_dev', 'score', 'basis_pct_prev'].
    """
    if not aligned_data:
        return {}

    clean_aligned: dict[str, pd.DataFrame] = {
        sym: _standardize_df(df) for sym, df in aligned_data.items()
    }

    if common_index is None:
        all_ts = sorted(set().union(*(d.index for d in clean_aligned.values())))
        target_index = pd.DatetimeIndex(all_ts, tz="UTC").as_unit("ns")
        target_index.name = "datetime"
    else:
        idx = pd.DatetimeIndex(common_index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        target_index = idx.as_unit("ns")
        target_index.name = "datetime"

    out_dict: dict[str, pd.DataFrame] = {}

    for sym, df in clean_aligned.items():
        if df.empty:
            empty_df = pd.DataFrame(
                np.nan,
                index=target_index,
                columns=["f_ewma", "basis_ema", "basis_dev", "score", "basis_pct_prev"],
                dtype="float64",
            )
            out_dict[sym] = empty_df
            continue

        # 1. Trích xuất hoặc tính toán basis_pct 1h
        if "basis_pct" in df.columns:
            bp_series = df["basis_pct"].astype("float64")
        elif "spot_close" in df.columns and "perp_close" in df.columns:
            sp = df["spot_close"].astype("float64")
            pe = df["perp_close"].astype("float64")
            valid = (sp > 0) & pe.notna()
            bp_series = pd.Series(np.nan, index=df.index, dtype="float64")
            bp_series[valid] = (pe[valid] - sp[valid]) / sp[valid]
        elif "close" in df.columns:
            bp_series = pd.Series(0.0, index=df.index, dtype="float64")
        else:
            bp_series = pd.Series(0.0, index=df.index, dtype="float64")

        bp_series = bp_series.replace([np.inf, -np.inf], np.nan)

        # Anti look-ahead: Tính EMA basis trên dữ liệu 1h và dịch trễ 1 nến (.shift(1))
        basis_ema_raw = bp_series.ewm(span=config.basis_ema_hours, adjust=True).mean()
        basis_ema = basis_ema_raw.shift(1).astype("float64")
        basis_pct_prev = bp_series.shift(1).astype("float64")
        basis_dev = (basis_pct_prev - basis_ema).astype("float64")

        # 2. Thành phần Funding kỳ vọng F_{i,t}
        # Lọc các mốc sự kiện funding 8h (00, 08, 16 UTC) để tính EWMA chuẩn không lặp
        mask_8h = (df.index.hour.isin([0, 8, 16])) & (df.index.minute == 0)
        if mask_8h.any():
            if "funding_rate_event" in df.columns:
                fr_events = df.loc[mask_8h, "funding_rate_event"].astype("float64")
                if (fr_events == 0.0).all() and "last_known_funding_rate" in df.columns:
                    lk = df.loc[mask_8h, "last_known_funding_rate"].astype("float64")
                    if not (lk == 0.0).all():
                        fr_events = lk
            elif "last_known_funding_rate" in df.columns:
                fr_events = df.loc[mask_8h, "last_known_funding_rate"].astype("float64")
            else:
                fr_events = pd.Series(0.0, index=df.index[mask_8h], dtype="float64")

            ewma_8h = fr_events.ewm(span=config.ema_span, adjust=True).mean()
            f_ewma_1h = ewma_8h.reindex(df.index).ffill().fillna(0.0).astype("float64")
        else:
            # Dự phòng khi mẩu dữ liệu nhỏ không có mốc 8h
            if "last_known_funding_rate" in df.columns:
                s = df["last_known_funding_rate"].astype("float64")
            elif "funding_rate_event" in df.columns:
                s = df["funding_rate_event"].astype("float64")
            else:
                s = pd.Series(0.0, index=df.index, dtype="float64")
            f_ewma_1h = s.ewm(span=config.ema_span, adjust=True).mean().astype("float64")

        # Anti look-ahead: Dịch trễ 1 nến trên trục 1h để tại t chỉ biết kết quả <= t-1
        f_ewma = f_ewma_1h.shift(1).fillna(0.0).astype("float64")

        # 3. Tính tổng điểm Score_{i,t}
        expected_periods = max(float(config.expected_hold_periods), 1e-6)
        score = f_ewma + float(config.w_basis) * (basis_dev / expected_periods)

        # Kiểm tra tính toàn vẹn dữ liệu tại t-1: nếu giá tại t-1 NaN hoặc <= 0 thì Score = NaN
        sp_col = "spot_close" if "spot_close" in df.columns else ("close" if "close" in df.columns else None)
        pe_col = "perp_close" if "perp_close" in df.columns else ("close" if "close" in df.columns else None)

        sp_prev = df[sp_col].shift(1).astype("float64") if sp_col else pd.Series(np.nan, index=df.index)
        pe_prev = df[pe_col].shift(1).astype("float64") if pe_col else pd.Series(np.nan, index=df.index)

        invalid_mask = sp_prev.isna() | pe_prev.isna() | (sp_prev <= 0) | (pe_prev <= 0) | basis_pct_prev.isna()
        score = score.mask(invalid_mask, np.nan).astype("float64")

        res_df = pd.DataFrame(
            {
                "f_ewma": f_ewma,
                "basis_ema": basis_ema,
                "basis_dev": basis_dev,
                "score": score,
                "basis_pct_prev": basis_pct_prev,
            },
            index=df.index,
            dtype="float64",
        )

        res_df = res_df.reindex(target_index)
        # Bất kỳ mốc nào không có dữ liệu gốc của symbol đều có score = NaN
        out_dict[sym] = res_df

    return out_dict


@dataclass(frozen=True)
class SelectionDecision:
    """Kết quả quyết định xoay vòng danh mục tại mốc thanh toán funding 8h."""

    target_weights: dict[str, float]
    actions: dict[str, Literal["HOLD", "OPEN", "CLOSE", "SKIP"]]
    reasons: dict[str, str]
    ranked_symbols: list[tuple[str, float, int]]


def select_portfolio(
    scores_at_t: dict[str, float],
    current_holdings: set[str],
    periods_held: dict[str, int],
    config: DynamicCarryConfig,
    switch_rule: Literal["advantage", "tier1b_buffer_rank"] | None = None,
) -> SelectionDecision:
    """Hàm thuần lựa chọn danh mục tập trung có cơ chế Buffer, Hysteresis và Min Switch Advantage.

    Tham số:
        scores_at_t: Dict map {symbol: score_t} tại thời điểm xét mốc 8h.
        current_holdings: Tập hợp các symbol đang nắm giữ từ kỳ trước.
        periods_held: Dict map {symbol: so_ky_funding_8h_da_nam_giu}.
        config: DynamicCarryConfig cấu hình top_k, ngưỡng entry/exit và buffer.

    Trả về:
        SelectionDecision chứa tỷ trọng mục tiêu, hành động cụ thể, lý do và bảng xếp hạng.
    """
    # 1. Lọc symbol có điểm số hợp lệ và sắp xếp giảm dần (tie-break theo tên symbol tăng dần)
    valid_items: list[tuple[str, float]] = []
    for sym, sc in scores_at_t.items():
        if sc is not None and np.isfinite(sc):
            valid_items.append((sym, float(sc)))

    valid_items.sort(key=lambda item: (-item[1], item[0]))

    ranked_symbols: list[tuple[str, float, int]] = [
        (sym, sc, rank) for rank, (sym, sc) in enumerate(valid_items, start=1)
    ]
    rank_map: dict[str, int] = {sym: rank for sym, sc, rank in ranked_symbols}
    score_map: dict[str, float] = {sym: sc for sym, sc, rank in ranked_symbols}

    # 2. Tập hợp ứng viên mở mới đạt ngưỡng vào lệnh
    new_candidates: list[str] = [
        sym
        for sym, sc, rank in ranked_symbols
        if sym not in current_holdings and sc >= config.entry_threshold
    ]

    actions: dict[str, Literal["HOLD", "OPEN", "CLOSE", "SKIP"]] = {}
    reasons: dict[str, str] = {}
    kept_holdings: set[str] = set()
    used_candidates: set[str] = set()
    min_switch = config.min_switch_advantage()
    rule = switch_rule or config.switch_rule
    drop_rank_threshold = int(config.top_k + config.buffer_rank)

    # 3. Xét từng coin đang nắm giữ từ kỳ trước (duyệt theo điểm từ cao xuống thấp)
    sorted_holdings = sorted(
        list(current_holdings),
        key=lambda h: (-score_map.get(h, -float("inf")), h),
    )

    for h in sorted_holdings:
        held_p = periods_held.get(h, 0)
        # Điều kiện giữ bắt buộc theo thời gian nắm giữ tối thiểu
        if held_p < config.min_hold_periods:
            actions[h] = "HOLD"
            reasons[h] = "HOLD_MIN_PERIODS"
            kept_holdings.add(h)
            continue

        h_score = score_map.get(h, -float("inf"))
        h_rank = rank_map.get(h, 999999)

        if rule == "tier1b_buffer_rank":
            # Tier 1B (Frozen Alpha): chi thoat khi het han giu toi thieu VA diem <= exit_threshold.
            if (not np.isfinite(h_score)) or h_score <= config.exit_threshold:
                actions[h] = "CLOSE"
                reasons[h] = "EXIT_BELOW_THRESHOLD"
                continue

            # Thoat/switch khi roi ngoai vung dem (top_k + buffer_rank).
            if h_rank > drop_rank_threshold:
                cand_found = None
                for cand in new_candidates:
                    if cand not in used_candidates:
                        cand_found = cand
                        break
                if cand_found is not None:
                    actions[h] = "CLOSE"
                    reasons[h] = "SWITCH_RANK_DROP"
                    used_candidates.add(cand_found)
                else:
                    actions[h] = "HOLD"
                    reasons[h] = "HOLD_BUFFER_RANK"
                    kept_holdings.add(h)
                continue

            actions[h] = "HOLD"
            reasons[h] = "HOLD_BUFFER_RANK"
            kept_holdings.add(h)
            continue

        # Điều kiện thoát khi điểm rơi dưới ngưỡng thoát
        if h_score < config.exit_threshold:
            actions[h] = "CLOSE"
            reasons[h] = "EXIT_BELOW_THRESHOLD"
            continue

        # Điều kiện thoát khi thứ hạng rơi ngoài vùng đệm
        if h_rank > config.top_k + config.buffer_rank:
            actions[h] = "CLOSE"
            reasons[h] = "EXIT_OUT_OF_BUFFER_RANK"
            continue

        # Kiểm tra xem có ứng viên mới nào vượt trội hơn hẳn min_switch_advantage
        better_candidate = None
        for cand in new_candidates:
            if cand not in used_candidates:
                cand_score = score_map[cand]
                if cand_score - h_score > min_switch:
                    better_candidate = cand
                # Chỉ xét ứng viên tốt nhất hiện có
                break

        if better_candidate is not None:
            actions[h] = "CLOSE"
            reasons[h] = "SWITCH_TO_BETTER_CANDIDATE"
            used_candidates.add(better_candidate)
        else:
            actions[h] = "HOLD"
            reasons[h] = "HOLD_BUFFER_RANK"
            kept_holdings.add(h)

    # 4. Lấp đầy các vị trí còn trống bằng ứng viên mở mới tốt nhất
    available_slots = max(0, config.top_k - len(kept_holdings))
    opened_holdings: set[str] = set()

    for cand in new_candidates:
        if len(opened_holdings) >= available_slots:
            break
        actions[cand] = "OPEN"
        reasons[cand] = "OPEN_TOP_RANK"
        opened_holdings.add(cand)

    # 5. Đánh dấu các symbol còn lại là SKIP
    for sym in scores_at_t:
        if sym not in actions:
            actions[sym] = "SKIP"
            sym_score = scores_at_t.get(sym)
            if sym_score is None or not np.isfinite(sym_score) or sym_score < config.entry_threshold:
                reasons[sym] = "SCORE_BELOW_ENTRY_THRESHOLD"
            elif rank_map.get(sym, 999999) > config.top_k:
                reasons[sym] = "RANK_BELOW_TOP_K"
            else:
                reasons[sym] = "SKIP"

    # 6. Phân bổ tỷ trọng cố định 1/top_k cho mỗi vị thế trong danh mục mục tiêu
    target_set = kept_holdings | opened_holdings
    weight_per_asset = 1.0 / float(config.top_k) if config.top_k > 0 else 0.0
    target_weights = {sym: weight_per_asset for sym in target_set}

    return SelectionDecision(
        target_weights=target_weights,
        actions=actions,
        reasons=reasons,
        ranked_symbols=ranked_symbols,
    )


def run_dynamic_carry_portfolio(
    aligned_data: dict[str, pd.DataFrame],
    config: DynamicCarryConfig,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """Mô phỏng Bar-by-Bar danh mục Dynamic Carry và Hội tụ Basis Spot-Perp.

    Tham số:
        aligned_data: Dict map {symbol: DataFrame} dữ liệu đã đồng bộ Spot, Perp và Funding.
        config: DynamicCarryConfig cấu hình các quy tắc giao dịch, chi phí và ký quỹ.
        periods_per_year: Số chu kỳ nến mỗi năm (mặc định 8760 cho khung 1h).

    Trả về:
        Dict gồm:
        - equity_curve: pd.Series đường cong vốn trên trục thời gian chung.
        - returns: pd.Series tỷ suất lợi nhuận pct_change().fillna(0.0).
        - trades_df: pd.DataFrame chi tiết các giao dịch mở, đóng, tái cân bằng.
        - holdings_log: pd.DataFrame ảnh chụp trạng thái danh mục tại mỗi kỳ 8h.
        - attribution_series: pd.DataFrame phân rã PnL từng giờ.
        - funding_pnl: float tổng funding thu được (USDT).
        - basis_pnl: float tổng lãi/lỗ MTM hội tụ giá Spot-Perp (USDT).
        - fees: float tổng phí giao dịch và trượt giá đã trả (USDT).
        - turnover: float hệ số quay vòng vốn (total_notional_traded / initial_capital).
        - metrics: Dict tổng hợp các chỉ số hiệu suất.
    """
    empty_idx = pd.DatetimeIndex([], tz="UTC", name="datetime").as_unit("ns")
    empty_metrics = {
        "Total Return %": 0.0,
        "Total Return % ": 0.0,
        "Annualized Return %": 0.0,
        "Sharpe": 0.0,
        "MaxDD %": 0.0,
        "Total Funding Collected": 0.0,
        "Total Basis PnL": 0.0,
        "Total MTM PnL": 0.0,
        "Total Fees Paid": 0.0,
        "Net PnL": 0.0,
        "Time in Market %": 0.0,
        "Turnover": 0.0,
        "Num Rebalances": 0,
        "Margin Rebalances": 0,
        "Price Rebalances": 0,
    }

    if not aligned_data:
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        empty_trades = pd.DataFrame(
            columns=["symbol", "time", "action", "notional", "fee", "funding_cum", "reason", "score"]
        )
        empty_holdings = pd.DataFrame(columns=["time", "symbols", "n_active", "total_notional", "equity"])
        empty_attr = pd.DataFrame(
            columns=["funding_pnl", "basis_pnl", "fees", "net_pnl", "notional_traded"],
            index=empty_idx,
            dtype="float64",
        )
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": empty_trades,
            "holdings_log": empty_holdings,
            "attribution_series": empty_attr,
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "fees": 0.0,
            "turnover": 0.0,
            "metrics": empty_metrics,
        }

    clean_aligned: dict[str, pd.DataFrame] = {
        sym: _standardize_df(df) for sym, df in aligned_data.items()
    }

    all_ts = sorted(set().union(*(d.index for d in clean_aligned.values())))
    common_index = pd.DatetimeIndex(all_ts, tz="UTC").as_unit("ns")
    common_index.name = "datetime"
    n_steps = len(common_index)

    if n_steps == 0:
        empty_eq = pd.Series([], index=common_index, dtype="float64", name="equity")
        empty_trades = pd.DataFrame(
            columns=["symbol", "time", "action", "notional", "fee", "funding_cum", "reason", "score"]
        )
        empty_holdings = pd.DataFrame(columns=["time", "symbols", "n_active", "total_notional", "equity"])
        empty_attr = pd.DataFrame(
            columns=["funding_pnl", "basis_pnl", "fees", "net_pnl", "notional_traded"],
            index=common_index,
            dtype="float64",
        )
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=common_index, dtype="float64"),
            "trades_df": empty_trades,
            "holdings_log": empty_holdings,
            "attribution_series": empty_attr,
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "fees": 0.0,
            "turnover": 0.0,
            "metrics": empty_metrics,
        }

    # Tính điểm số cho toàn bộ các symbol trên trục thời gian chung
    scores_dict = build_carry_basis_scores(clean_aligned, config, common_index)

    symbols = list(clean_aligned.keys())

    # Trích xuất dữ liệu giá và sự kiện funding dạng mảng numpy để tối ưu tốc độ duyệt nến
    spot_prices: dict[str, np.ndarray] = {}
    perp_prices: dict[str, np.ndarray] = {}
    perp_high_prices: dict[str, np.ndarray] = {}
    basis_values: dict[str, np.ndarray] = {}
    funding_events: dict[str, np.ndarray] = {}
    score_values: dict[str, np.ndarray] = {}

    for sym in symbols:
        df = clean_aligned[sym]
        sp_s = df["spot_close"] if "spot_close" in df.columns else (df["close"] if "close" in df.columns else pd.Series(np.nan, index=df.index))
        pe_s = df["perp_close"] if "perp_close" in df.columns else (df["close"] if "close" in df.columns else pd.Series(np.nan, index=df.index))
        hi_s = df["high"] if "high" in df.columns else pe_s

        if "basis_pct" in df.columns:
            ba_s = df["basis_pct"]
        else:
            valid = (sp_s > 0) & pe_s.notna()
            ba_s = pd.Series(0.0, index=df.index, dtype="float64")
            ba_s[valid] = (pe_s[valid] - sp_s[valid]) / sp_s[valid]

        fr_s = df["funding_rate_event"] if "funding_rate_event" in df.columns else pd.Series(0.0, index=df.index)

        spot_prices[sym] = sp_s.reindex(common_index).to_numpy(dtype=np.float64)
        perp_prices[sym] = pe_s.reindex(common_index).to_numpy(dtype=np.float64)
        perp_high_prices[sym] = hi_s.reindex(common_index).to_numpy(dtype=np.float64)
        basis_values[sym] = ba_s.reindex(common_index).to_numpy(dtype=np.float64)
        funding_events[sym] = fr_s.reindex(common_index).fillna(0.0).to_numpy(dtype=np.float64)

        sc_s = scores_dict[sym]["score"] if sym in scores_dict else pd.Series(np.nan, index=common_index)
        score_values[sym] = sc_s.reindex(common_index).to_numpy(dtype=np.float64)

    # Khởi tạo trạng thái danh mục
    equity = float(config.initial_capital)
    eta = float(config.notional_eta())
    cost_rate = float(config.one_way_cost())

    in_position: dict[str, bool] = {sym: False for sym in symbols}
    periods_held: dict[str, int] = {sym: 0 for sym in symbols}
    notional: dict[str, float] = {sym: 0.0 for sym in symbols}
    Q_qty: dict[str, float] = {sym: 0.0 for sym in symbols}
    margin_balance: dict[str, float] = {sym: 0.0 for sym in symbols}
    entry_spot_price: dict[str, float] = {sym: 0.0 for sym in symbols}
    entry_perp_price: dict[str, float] = {sym: 0.0 for sym in symbols}
    trade_funding_cum: dict[str, float] = {sym: 0.0 for sym in symbols}

    pending_maker_orders: dict[str, dict[str, Any]] = {}

    total_funding_collected = 0.0
    total_basis_pnl = 0.0
    total_fees_paid = 0.0
    total_notional_traded = 0.0
    num_rebalances = 0
    margin_rebalances = 0
    price_rebalances = 0

    trades_list: list[dict[str, Any]] = []
    holdings_log_list: list[dict[str, Any]] = []

    equity_vals = np.empty(n_steps, dtype=np.float64)
    in_market_any = np.zeros(n_steps, dtype=bool)

    hourly_funding_pnl = np.zeros(n_steps, dtype=np.float64)
    hourly_basis_pnl = np.zeros(n_steps, dtype=np.float64)
    hourly_fees = np.zeros(n_steps, dtype=np.float64)
    hourly_notional_traded = np.zeros(n_steps, dtype=np.float64)

    buffer_threshold = float(config.maker_entry_buffer_bps) * 1e-4

    # Vòng lặp mô phỏng bar-by-bar
    for t_idx, t in enumerate(common_index):
        # -------------------------------------------------------------
        # Bước 1 — Định giá True Bar-by-Bar Spot-Perp MTM
        # -------------------------------------------------------------
        if t_idx > 0:
            for sym in symbols:
                if in_position[sym]:
                    if config.structure == "spot_perp_3x":
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
                            mtm = Q_qty[sym] * (sp_curr - sp_prev) - Q_qty[sym] * (pe_curr - pe_prev)
                    else:  # coinm_1leg
                        ba_prev = basis_values[sym][t_idx - 1]
                        ba_curr = basis_values[sym][t_idx]
                        if np.isnan(ba_prev) or np.isnan(ba_curr):
                            mtm = 0.0
                        else:
                            mtm = -notional[sym] * (ba_curr - ba_prev)

                    equity += mtm
                    margin_balance[sym] += mtm
                    total_basis_pnl += mtm
                    hourly_basis_pnl[t_idx] += mtm

        # -------------------------------------------------------------
        # Bước 2 — Khớp lệnh Chờ passive_limit_maker (nếu có lệnh pending)
        # -------------------------------------------------------------
        if config.execution == "passive_limit_maker" and pending_maker_orders:
            for sym in list(pending_maker_orders.keys()):
                order = pending_maker_orders[sym]
                basis_ref = order["basis_ref"]
                sp_t = spot_prices[sym][t_idx]
                pe_t = perp_prices[sym][t_idx]
                hi_t = perp_high_prices[sym][t_idx]
                ba_t = basis_values[sym][t_idx]

                if np.isnan(sp_t) or sp_t <= 0 or np.isnan(pe_t) or pe_t <= 0:
                    continue

                curr_basis_cand = ba_t
                if not np.isnan(hi_t):
                    high_basis = (hi_t - sp_t) / sp_t
                    if high_basis > curr_basis_cand:
                        curr_basis_cand = high_basis

                if curr_basis_cand >= basis_ref + buffer_threshold and not in_position[sym] and equity > 0:
                    allocated = order["allocated"]
                    leg_notional = allocated * eta
                    Q_val = leg_notional / sp_t
                    Q_qty[sym] = Q_val
                    entry_spot_price[sym] = sp_t
                    entry_perp_price[sym] = pe_t
                    notional[sym] = leg_notional

                    if config.structure == "spot_perp_3x":
                        margin_balance[sym] = allocated - leg_notional
                    else:
                        margin_balance[sym] = allocated

                    fee = leg_notional * cost_rate
                    equity -= fee
                    margin_balance[sym] -= fee
                    total_fees_paid += fee
                    hourly_fees[t_idx] += fee
                    total_notional_traded += leg_notional
                    hourly_notional_traded[t_idx] += leg_notional

                    in_position[sym] = True
                    periods_held[sym] = 0
                    trade_funding_cum[sym] = 0.0

                    trades_list.append({
                        "symbol": sym,
                        "time": t,
                        "action": "open",
                        "notional": float(leg_notional),
                        "fee": float(fee),
                        "funding_cum": 0.0,
                        "reason": "OPEN_MAKER_FILLED",
                        "score": float(order["score"]),
                    })
                    del pending_maker_orders[sym]
                else:
                    order["ttl"] -= 1
                    if order["ttl"] <= 0:
                        del pending_maker_orders[sym]

        # -------------------------------------------------------------
        # Bước 3 — Thanh toán Funding Rate (tại mốc 00:00, 08:00, 16:00 UTC)
        # -------------------------------------------------------------
        is_8h_event = (t.hour in (0, 8, 16)) and (t.minute == 0)

        if is_8h_event:
            for sym in symbols:
                if in_position[sym]:
                    fr_event = float(funding_events[sym][t_idx])
                    if abs(fr_event) > 0.0:
                        f_pnl = notional[sym] * fr_event
                        equity += f_pnl
                        margin_balance[sym] += f_pnl
                        trade_funding_cum[sym] += f_pnl
                        total_funding_collected += f_pnl
                        hourly_funding_pnl[t_idx] += f_pnl
                    periods_held[sym] += 1

        # -------------------------------------------------------------
        # Bước 4 — Kiểm tra An toàn Ký quỹ & Tự động Rebalance
        # -------------------------------------------------------------
        for sym in symbols:
            if in_position[sym]:
                sp_curr = spot_prices[sym][t_idx]
                pe_curr = perp_prices[sym][t_idx]
                curr_perp_notional = Q_qty[sym] * pe_curr if config.structure == "spot_perp_3x" else notional[sym]
                margin_ratio = margin_balance[sym] / max(curr_perp_notional, 1e-12)

                margin_trig = (
                    margin_ratio < config.maintenance_margin_ratio
                    or margin_ratio < config.margin_rebalance_lower_ratio
                )
                price_trig = (
                    config.structure == "spot_perp_3x"
                    and entry_spot_price[sym] > 0
                    and not np.isnan(sp_curr)
                    and (abs(sp_curr - entry_spot_price[sym]) / entry_spot_price[sym] >= config.rebalance_price_trigger_pct)
                )

                if margin_trig or price_trig:
                    reason = "margin" if margin_trig else "price"
                    if config.structure == "spot_perp_3x":
                        if not np.isnan(sp_curr) and sp_curr > 0:
                            v_slot = Q_qty[sym] * sp_curr + margin_balance[sym]
                            if v_slot <= 0.0:
                                # Thanh lý khẩn cấp khi giá trị slot cạn kiệt
                                pos_notional = Q_qty[sym] * sp_curr
                                fee = pos_notional * cost_rate
                                equity -= fee
                                total_fees_paid += fee
                                hourly_fees[t_idx] += fee
                                total_notional_traded += pos_notional
                                hourly_notional_traded[t_idx] += pos_notional
                                in_position[sym] = False
                                Q_qty[sym] = 0.0
                                notional[sym] = 0.0
                                margin_balance[sym] = 0.0
                                periods_held[sym] = 0
                                trades_list.append({
                                    "symbol": sym,
                                    "time": t,
                                    "action": "close",
                                    "notional": float(pos_notional),
                                    "fee": float(fee),
                                    "funding_cum": float(trade_funding_cum[sym]),
                                    "reason": "EMERGENCY_LIQUIDATION",
                                    "score": float(score_values[sym][t_idx]) if not np.isnan(score_values[sym][t_idx]) else 0.0,
                                })
                                continue

                            notional_new = 0.75 * v_slot
                            Q_new = notional_new / sp_curr
                            margin_balance[sym] = 0.25 * v_slot
                            delta_notional = abs(Q_new - Q_qty[sym]) * sp_curr
                            reb_fee = config.rebalance_fee_multiplier * cost_rate * delta_notional

                            equity -= reb_fee
                            margin_balance[sym] -= reb_fee
                            total_fees_paid += reb_fee
                            hourly_fees[t_idx] += reb_fee
                            total_notional_traded += delta_notional
                            hourly_notional_traded[t_idx] += delta_notional

                            Q_qty[sym] = Q_new
                            notional[sym] = notional_new
                            entry_spot_price[sym] = sp_curr
                            entry_perp_price[sym] = pe_curr

                            trades_list.append({
                                "symbol": sym,
                                "time": t,
                                "action": "rebalance",
                                "notional": float(delta_notional),
                                "fee": float(reb_fee),
                                "funding_cum": float(trade_funding_cum[sym]),
                                "reason": reason,
                                "score": float(score_values[sym][t_idx]) if not np.isnan(score_values[sym][t_idx]) else 0.0,
                            })
                            num_rebalances += 1
                            if reason == "margin":
                                margin_rebalances += 1
                            else:
                                price_rebalances += 1
                    else:  # coinm_1leg
                        target_ratio = max(config.margin_rebalance_target_ratio, 1e-4)
                        notional_new = margin_balance[sym] / target_ratio
                        delta_notional = abs(curr_perp_notional - notional_new)
                        reb_fee = config.rebalance_fee_multiplier * cost_rate * delta_notional

                        equity -= reb_fee
                        margin_balance[sym] -= reb_fee
                        total_fees_paid += reb_fee
                        hourly_fees[t_idx] += reb_fee
                        total_notional_traded += delta_notional
                        hourly_notional_traded[t_idx] += delta_notional
                        notional[sym] = notional_new

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "rebalance",
                            "notional": float(delta_notional),
                            "fee": float(reb_fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                            "reason": reason,
                            "score": float(score_values[sym][t_idx]) if not np.isnan(score_values[sym][t_idx]) else 0.0,
                        })
                        num_rebalances += 1
                        if reason == "margin":
                            margin_rebalances += 1
                        else:
                            price_rebalances += 1

        # -------------------------------------------------------------
        # Bước 5 — Xếp Hạng & Xoay Vòng Danh Mục (tại mốc 8h)
        # -------------------------------------------------------------
        if is_8h_event:
            scores_at_t = {sym: float(score_values[sym][t_idx]) for sym in symbols}
            current_holdings = {sym for sym in symbols if in_position[sym]}

            decision = select_portfolio(scores_at_t, current_holdings, periods_held, config)

            # Thực thi CLOSE trước
            for sym in symbols:
                if decision.actions.get(sym) == "CLOSE" and in_position[sym]:
                    sp_curr = spot_prices[sym][t_idx]
                    if np.isnan(sp_curr) or sp_curr <= 0:
                        sp_curr = entry_spot_price[sym] if entry_spot_price[sym] > 0 else 1.0

                    pos_notional = Q_qty[sym] * sp_curr if config.structure == "spot_perp_3x" else notional[sym]
                    fee = pos_notional * cost_rate

                    equity -= fee
                    margin_balance[sym] -= fee
                    total_fees_paid += fee
                    hourly_fees[t_idx] += fee
                    total_notional_traded += pos_notional
                    hourly_notional_traded[t_idx] += pos_notional

                    trades_list.append({
                        "symbol": sym,
                        "time": t,
                        "action": "close",
                        "notional": float(pos_notional),
                        "fee": float(fee),
                        "funding_cum": float(trade_funding_cum[sym]),
                        "reason": decision.reasons.get(sym, "CLOSE"),
                        "score": float(scores_at_t.get(sym, 0.0)),
                    })

                    in_position[sym] = False
                    notional[sym] = 0.0
                    Q_qty[sym] = 0.0
                    margin_balance[sym] = 0.0
                    periods_held[sym] = 0
                    trade_funding_cum[sym] = 0.0

            # Thực thi OPEN sau
            for sym in symbols:
                if decision.actions.get(sym) == "OPEN" and not in_position[sym] and equity > 0:
                    allocated = (1.0 / float(config.top_k)) * equity
                    leg_notional = allocated * eta
                    sp_t0 = spot_prices[sym][t_idx]
                    pe_t0 = perp_prices[sym][t_idx]
                    hi_t0 = perp_high_prices[sym][t_idx]
                    ba_t0 = basis_values[sym][t_idx]

                    if np.isnan(sp_t0) or sp_t0 <= 0:
                        sp_t0 = 1.0
                    if np.isnan(pe_t0) or pe_t0 <= 0:
                        pe_t0 = sp_t0

                    basis_ref = basis_values[sym][t_idx - 1] if t_idx > 0 else ba_t0
                    if np.isnan(basis_ref):
                        basis_ref = 0.0

                    if config.execution == "real_taker":
                        Q_val = leg_notional / sp_t0
                        Q_qty[sym] = Q_val
                        entry_spot_price[sym] = sp_t0
                        entry_perp_price[sym] = pe_t0
                        notional[sym] = leg_notional

                        if config.structure == "spot_perp_3x":
                            margin_balance[sym] = allocated - leg_notional
                        else:
                            margin_balance[sym] = allocated

                        fee = leg_notional * cost_rate
                        equity -= fee
                        margin_balance[sym] -= fee
                        total_fees_paid += fee
                        hourly_fees[t_idx] += fee
                        total_notional_traded += leg_notional
                        hourly_notional_traded[t_idx] += leg_notional

                        in_position[sym] = True
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "open",
                            "notional": float(leg_notional),
                            "fee": float(fee),
                            "funding_cum": 0.0,
                            "reason": decision.reasons.get(sym, "OPEN"),
                            "score": float(scores_at_t.get(sym, 0.0)),
                        })
                    else:  # passive_limit_maker
                        curr_basis_cand = ba_t0
                        if not np.isnan(hi_t0):
                            high_basis = (hi_t0 - sp_t0) / sp_t0
                            if high_basis > curr_basis_cand:
                                curr_basis_cand = high_basis

                        if curr_basis_cand >= basis_ref + buffer_threshold:
                            # Khớp ngay tại nến đánh giá t_0 với phí Maker
                            Q_val = leg_notional / sp_t0
                            Q_qty[sym] = Q_val
                            entry_spot_price[sym] = sp_t0
                            entry_perp_price[sym] = pe_t0
                            notional[sym] = leg_notional

                            if config.structure == "spot_perp_3x":
                                margin_balance[sym] = allocated - leg_notional
                            else:
                                margin_balance[sym] = allocated

                            fee = leg_notional * cost_rate
                            equity -= fee
                            margin_balance[sym] -= fee
                            total_fees_paid += fee
                            hourly_fees[t_idx] += fee
                            total_notional_traded += leg_notional
                            hourly_notional_traded[t_idx] += leg_notional

                            in_position[sym] = True
                            periods_held[sym] = 0
                            trade_funding_cum[sym] = 0.0

                            trades_list.append({
                                "symbol": sym,
                                "time": t,
                                "action": "open",
                                "notional": float(leg_notional),
                                "fee": float(fee),
                                "funding_cum": 0.0,
                                "reason": "OPEN_MAKER_FILLED",
                                "score": float(scores_at_t.get(sym, 0.0)),
                            })
                        else:
                            # Đăng ký lệnh Maker chờ khớp ở nến kế tiếp
                            pending_maker_orders[sym] = {
                                "allocated": allocated,
                                "basis_ref": basis_ref,
                                "score": scores_at_t.get(sym, 0.0),
                                "ttl": 1,
                            }

            # Ghi snapshot danh mục tại mốc 8h
            active_syms = [s for s in symbols if in_position[s]]
            tot_notional = sum(notional[s] for s in active_syms)
            holdings_log_list.append({
                "time": t,
                "symbols": active_syms,
                "n_active": len(active_syms),
                "total_notional": float(tot_notional),
                "equity": float(equity),
            })

        # -------------------------------------------------------------
        # Bước 6 — Ghi nhận vốn từng nến
        # -------------------------------------------------------------
        equity_vals[t_idx] = equity
        in_market_any[t_idx] = any(in_position.values())

    # Xây dựng các DataFrame kết quả
    equity_curve = pd.Series(equity_vals, index=common_index, dtype="float64", name="equity")
    returns = equity_curve.pct_change().fillna(0.0)

    trades_cols = ["symbol", "time", "action", "notional", "fee", "funding_cum", "reason", "score"]
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=trades_cols)
    if not trades_df.empty:
        for col in trades_cols:
            if col not in trades_df.columns:
                trades_df[col] = np.nan

    holdings_cols = ["time", "symbols", "n_active", "total_notional", "equity"]
    holdings_log = pd.DataFrame(holdings_log_list) if holdings_log_list else pd.DataFrame(columns=holdings_cols)

    hourly_net_pnl = hourly_funding_pnl + hourly_basis_pnl - hourly_fees
    attribution_series = pd.DataFrame(
        {
            "funding_pnl": hourly_funding_pnl,
            "basis_pnl": hourly_basis_pnl,
            "fees": hourly_fees,
            "net_pnl": hourly_net_pnl,
            "notional_traded": hourly_notional_traded,
        },
        index=common_index,
        dtype="float64",
    )

    # Tính toán các chỉ số hiệu suất tổng hợp
    total_hours = len(common_index)
    years = total_hours / float(periods_per_year) if periods_per_year > 0 else 0.0

    if config.initial_capital > 0:
        total_return_pct = ((equity - config.initial_capital) / config.initial_capital) * 100.0
        if years > 0:
            if equity > 0:
                annualized_return_pct = (
                    ((equity / config.initial_capital) ** (1.0 / years) - 1.0) * 100.0
                )
            else:
                annualized_return_pct = -100.0
        else:
            annualized_return_pct = 0.0
    else:
        total_return_pct = 0.0
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
        "Total Basis PnL": float(total_basis_pnl),
        "Total MTM PnL": float(total_basis_pnl),
        "Total Fees Paid": float(total_fees_paid),
        "Net PnL": float(equity - config.initial_capital),
        "Time in Market %": time_in_market_pct,
        "Turnover": turnover,
        "Num Rebalances": int(num_rebalances),
        "Margin Rebalances": int(margin_rebalances),
        "Price Rebalances": int(price_rebalances),
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades_df": trades_df,
        "holdings_log": holdings_log,
        "attribution_series": attribution_series,
        "funding_pnl": float(total_funding_collected),
        "basis_pnl": float(total_basis_pnl),
        "fees": float(total_fees_paid),
        "turnover": turnover,
        "metrics": metrics,
    }
