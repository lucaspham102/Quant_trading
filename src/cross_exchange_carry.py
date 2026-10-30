"""Module mô phỏng chiến lược Cross-Exchange Carry đa sàn (Binance, Bybit, OKX).

Sprint 10 - Chặng 1:
Cung cấp:
- EXCHANGE_LOT_SPECS, LOT_STEP_SIZES, DEFAULT_MIN_NOTIONAL, quantize_order_qty: Quy chuẩn lot-size và min-notional từng sàn.
- CrossExchangeCarryConfig: Cấu hình tham số chiến lược carry dài hạn, ngưỡng entry/exit và ma trận chi phí.
- compute_expected_8h_yield: Tính toán kỳ vọng lợi suất mỗi 8h cho Chế độ A và B với ngưỡng entry hiệu dụng.
- run_cross_exchange_carry: Engine mô phỏng bar-by-bar delta-neutral carry đa sàn với Bar-by-Bar Basis MTM,
  lượng hóa lot-size, mô hình maker fill tất định, quản trị ký quỹ liên sàn và cơ chế Hysteresis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from src.multi_exchange_loader import SUPPORTED_EXCHANGES

# -----------------------------------------------------------------------------
# Yêu cầu 1: Quy chuẩn kích thước lệnh (Lot-size & Min Notional) từng cặp sàn
# -----------------------------------------------------------------------------
EXCHANGE_LOT_SPECS: dict[str, dict[str, float]] = {
    "BTC/USDT:USDT": {"step_size": 0.001, "min_notional": 10.0},
    "ETH/USDT:USDT": {"step_size": 0.01,  "min_notional": 10.0},
    "SOL/USDT:USDT": {"step_size": 0.1,   "min_notional": 10.0},
    "BNB/USDT:USDT": {"step_size": 0.01,  "min_notional": 10.0},
    "DOGE/USDT:USDT": {"step_size": 1.0,  "min_notional": 10.0},
    "XRP/USDT:USDT": {"step_size": 0.1,   "min_notional": 10.0},
    "AVAX/USDT:USDT": {"step_size": 1.0,  "min_notional": 10.0},
    "LINK/USDT:USDT": {"step_size": 0.01, "min_notional": 10.0},
}
LOT_STEP_SIZES: dict[str, float] = {k: v["step_size"] for k, v in EXCHANGE_LOT_SPECS.items()}
DEFAULT_MIN_NOTIONAL: float = 10.0

from src.config import CORE_4_SYMBOLS

LIQUID_8_SYMBOLS: list[str] = list(CORE_4_SYMBOLS) + [
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
]


def quantize_order_qty(
    symbol: str,
    target_notional_usd: float,
    price: float,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
) -> tuple[float, float]:
    """Lượng hóa khối lượng đặt lệnh theo step_size và min_notional của sàn.

    Tham số:
        symbol: Mã cặp giao dịch (vd: 'BTC/USDT:USDT' hoặc 'BTC/USDT').
        target_notional_usd: Giá trị danh nghĩa mục tiêu (USD).
        price: Giá vào lệnh (entry price).
        min_notional: Giá trị danh nghĩa tối thiểu của lệnh (mặc định 10.0 USD).

    Trả về:
        Tuple (q, actual): Khối lượng và giá trị danh nghĩa thực tế sau khi làm tròn.
    """
    if price <= 0.0 or target_notional_usd <= 0.0:
        return (0.0, 0.0)

    step = LOT_STEP_SIZES.get(symbol)
    if step is None:
        if f"{symbol}:USDT" in LOT_STEP_SIZES:
            step = LOT_STEP_SIZES[f"{symbol}:USDT"]
        elif symbol.endswith(":USDT") and symbol[:-5] in LOT_STEP_SIZES:
            step = LOT_STEP_SIZES[symbol[:-5]]
        else:
            base = symbol.split("/")[0] if "/" in symbol else symbol
            for k, v in LOT_STEP_SIZES.items():
                if k.startswith(f"{base}/"):
                    step = v
                    break
            if step is None:
                step = 0.001

    q_raw = target_notional_usd / price
    q = math.floor((q_raw + 1e-12) / step) * step
    precision = 8 if step < 0.001 else 6
    q = round(q, precision)
    actual = q * price
    if actual < min_notional:
        return (0.0, 0.0)
    return (float(q), float(actual))


@dataclass(frozen=True)
class CrossExchangeCarryConfig:
    """Cấu hình tham số chiến lược Cross-Exchange Carry và ma trận chi phí thực thi."""

    mode: Literal["perp_perp_spread", "best_venue_carry", "hybrid", "single_binance"] = "hybrid"
    structure_b: Literal["coinm_1leg", "spot_perp_unified"] = "spot_perp_unified"
    execution: Literal["maker_taker_hedge", "real_taker"] = "maker_taker_hedge"
    initial_capital: float = 3000.0
    top_k: int = 2
    ema_span: int = 21

    # Đòn bẩy & hiệu suất sử dụng vốn (eta)
    perp_leg_leverage: float = 2.0
    perp_perp_eta: float = 1.00
    coinm_eta: float = 1.00
    spot_perp_eta: float = 0.80

    # Phí thực thi khóa cứng theo docs/go_no_go_sprint9.md
    maker_perp_fee: float = 0.0002          # 2 bps
    adverse_selection_bps: float = 0.0001   # 1 bp
    taker_perp_fee: float = 0.0005          # 5 bps
    taker_slippage: float = 0.0002          # 2 bps
    maker_spot_fee: float = 0.0005          # 5 bps
    spot_perp_one_way_override: float = 0.001125  # 11.25 bps/chiều -> 22.5 bps khứ hồi

    # Mô hình khớp lệnh Maker Post-Only (deterministic)
    enable_maker_fill_model: bool = True
    maker_fill_buffer_bps: float = 0.5      # 0.5 bps buffer
    maker_order_ttl_bars: int = 2           # Hết 2 nến không khớp -> hủy

    # Hysteresis siêu bền
    max_payback_periods: int = 42           # <= 42 kỳ funding phải thu hồi đủ round-trip cost
    horizon_periods_7d: int = 21            # 7 ngày = 21 kỳ funding 8h
    switch_cost_multiplier: float = 3.0     # Chênh lệch kỳ vọng 21 kỳ > 3 * round_trip_cost mới đóng/chuyển sàn
    min_hold_periods: int = 42              # Giữ tối thiểu 42 kỳ trừ khi âm sâu
    exit_yield_threshold: float = -0.00002  # Ngưỡng thoát khi expected yield âm sâu
    entry_funding_threshold: float = 0.00005  # Ngưỡng entry tối thiểu mỗi 8h (0.50 bps/8h)
    exit_funding_threshold: float = -0.00001  # Ngưỡng thoát khi funding rate âm (-0.10 bps/8h)
    lot_size_quantization: bool = True       # Bật lượng hóa khối lượng đặt lệnh theo lot-size sàn

    # Quản trị ký quỹ liên sàn
    price_rebalance_trigger_pct: float = 0.25  # Biến động giá >= ±25% từ lúc mở/rebalance gần nhất
    rebalance_cost_bps: float = 5.0            # 5.0 bps (0.0005) trừ vào equity
    maintenance_margin_ratio: float = 0.05

    @property
    def spot_perp_one_way_Override(self) -> float:
        """Alias tương thích chữ hoa chữ thường theo đặc tả Sprint 9."""
        return self.spot_perp_one_way_override

    def one_way_cost_perp_perp(self) -> float:
        """Chi phí 1 chiều mở hoặc đóng cho Chế độ Perp-vs-Perp (Perp-Perp)."""
        if self.execution == "maker_taker_hedge":
            # Chân 1 Maker (2 bps + 1 bp adverse selection) + Chân 2 Taker (5 bps + 2 bps slippage) = 10 bps
            leg1_cost = self.maker_perp_fee + self.adverse_selection_bps
            leg2_cost = self.taker_perp_fee + self.taker_slippage
            return leg1_cost + leg2_cost
        elif self.execution == "real_taker":
            # Cả 2 chân Taker: 2 * (5 bps + 2 bps slippage) = 14 bps
            return 2.0 * (self.taker_perp_fee + self.taker_slippage)
        raise ValueError(f"Kiểu thực thi không hợp lệ: {self.execution}")

    def round_trip_cost_perp_perp(self) -> float:
        """Chi phí khứ hồi (mở + đóng) cho Chế độ Perp-vs-Perp (20 bps ở maker_taker_hedge)."""
        return 2.0 * self.one_way_cost_perp_perp()

    def one_way_cost_spot_perp(self) -> float:
        """Chi phí 1 chiều mở hoặc đóng cho Chế độ Spot-Perp (11.25 bps ở maker_taker_hedge)."""
        if self.execution == "maker_taker_hedge":
            if self.spot_perp_one_way_override > 0.0:
                return float(self.spot_perp_one_way_override)
            return self.maker_spot_fee + self.taker_perp_fee + self.taker_slippage + 0.000025
        elif self.execution == "real_taker":
            return 0.00075 + self.taker_perp_fee + (2.0 * self.taker_slippage)
        raise ValueError(f"Kiểu thực thi không hợp lệ: {self.execution}")

    def round_trip_cost_spot_perp(self) -> float:
        """Chi phí khứ hồi (mở + đóng) cho Chế độ Spot-Perp (22.5 bps ở maker_taker_hedge)."""
        return 2.0 * self.one_way_cost_spot_perp()

    def round_trip_cost_for_mode(self, mode_type: str) -> float:
        """Tính chi phí khứ hồi theo từng chế độ giao dịch."""
        m = str(mode_type).lower()
        if "perp_perp" in m or m in ("a", "mode_a"):
            return self.round_trip_cost_perp_perp()
        elif "coinm" in m:
            # 1 chân Coin-M: mở + đóng = 2 x 10 bps = 20 bps khứ hồi
            return self.round_trip_cost_perp_perp()
        else:
            return self.round_trip_cost_spot_perp()

    def min_entry_rate_per_8h(self, mode_type: str) -> float:
        """Lợi suất tối thiểu mỗi 8h cần đạt để thu hồi đủ phí khứ hồi trong <= max_payback_periods."""
        rt_cost = self.round_trip_cost_for_mode(mode_type)
        periods = max(int(self.max_payback_periods), 1)
        return rt_cost / float(periods)


def compute_expected_8h_yield(
    df: pd.DataFrame | pd.Series,
    config: CrossExchangeCarryConfig,
) -> pd.DataFrame:
    """Tính kỳ vọng lợi suất mỗi 8h tại nến t (dùng EWMA t-1) cho Chế độ A và Chế độ B.

    Tham số:
        df: DataFrame hoặc Series chứa các cột đã đồng bộ từ align_multi_exchange_funding.
        config: Cấu hình chiến lược CrossExchangeCarryConfig.

    Trả về:
        DataFrame chứa các cột:
        - expected_yield_a: Lợi suất kỳ vọng mỗi 8h cho Chế độ A (cross_ex_spread * perp_perp_eta).
        - expected_yield_b: Lợi suất kỳ vọng mỗi 8h cho Chế độ B (max_single_funding * eta_b hoặc single binance).
        - selected_mode: Chế độ tối ưu được chọn ('perp_perp_spread', 'best_venue_carry', hoặc None).
        - expected_yield: Lợi suất kỳ vọng của chế độ được chọn.
        - min_entry_rate: Ngưỡng lợi suất hiệu dụng tối thiểu (max giữa min_entry_rate_per_8h và entry_funding_threshold).
        - can_enter: Cờ boolean chỉ ra có đủ điều kiện mở vị thế mới theo ngưỡng hiệu dụng hay không.
        - short_ex: Sàn giao dịch mở vị thế Short.
        - long_ex: Sàn giao dịch mở vị thế Long (hoặc 'spot').
    """
    is_series = isinstance(df, pd.Series)
    frame = df.to_frame().T if is_series else df.copy()

    n_rows = len(frame)
    idx = frame.index

    # 1. Tính lợi suất kỳ vọng Chế độ A (Perp-vs-Perp)
    if "cross_ex_spread" in frame.columns:
        spread_raw = frame["cross_ex_spread"].to_numpy(dtype=np.float64)
    else:
        spread_raw = np.full(n_rows, np.nan, dtype=np.float64)

    best_short_s = frame["best_short_ex"] if "best_short_ex" in frame.columns else pd.Series([None] * n_rows, index=idx)
    best_long_s = frame["best_long_ex"] if "best_long_ex" in frame.columns else pd.Series([None] * n_rows, index=idx)

    short_arr = best_short_s.to_numpy()
    long_arr = best_long_s.to_numpy()

    valid_mode_a = (
        np.isfinite(spread_raw)
        & (spread_raw > 0.0)
        & (short_arr != None)  # noqa: E711
        & (long_arr != None)   # noqa: E711
        & (short_arr != long_arr)
    )
    yield_a = np.where(valid_mode_a, spread_raw * float(config.perp_perp_eta), -np.inf)

    # 2. Tính lợi suất kỳ vọng Chế độ B (Best-Venue Carry hoặc Single Binance)
    eta_b = float(config.coinm_eta if config.structure_b == "coinm_1leg" else config.spot_perp_eta)

    if config.mode == "single_binance":
        col_binance = "ewma_funding_binance"
        if col_binance in frame.columns:
            binance_rate = frame[col_binance].to_numpy(dtype=np.float64)
        else:
            binance_rate = np.full(n_rows, np.nan, dtype=np.float64)
        valid_mode_b = np.isfinite(binance_rate)
        yield_b = np.where(valid_mode_b, binance_rate * eta_b, -np.inf)
        short_b_arr = np.where(valid_mode_b, "binance", None)
    else:
        if "max_single_funding" in frame.columns:
            max_single = frame["max_single_funding"].to_numpy(dtype=np.float64)
        else:
            max_single = np.full(n_rows, np.nan, dtype=np.float64)
        valid_mode_b = np.isfinite(max_single) & (short_arr != None)  # noqa: E711
        yield_b = np.where(valid_mode_b, max_single * eta_b, -np.inf)
        short_b_arr = short_arr

    # 3. Lựa chọn chế độ theo config.mode
    selected_mode_arr = np.full(n_rows, None, dtype=object)
    selected_yield_arr = np.full(n_rows, np.nan, dtype=np.float64)
    selected_short_arr = np.full(n_rows, None, dtype=object)
    selected_long_arr = np.full(n_rows, None, dtype=object)

    if config.mode == "perp_perp_spread":
        mask = valid_mode_a
        selected_mode_arr[mask] = "perp_perp_spread"
        selected_yield_arr[mask] = yield_a[mask]
        selected_short_arr[mask] = short_arr[mask]
        selected_long_arr[mask] = long_arr[mask]

    elif config.mode in ("best_venue_carry", "single_binance"):
        mask = valid_mode_b
        selected_mode_arr[mask] = "best_venue_carry"
        selected_yield_arr[mask] = yield_b[mask]
        selected_short_arr[mask] = short_b_arr[mask]
        selected_long_arr[mask] = "spot"

    elif config.mode == "hybrid":
        choose_a = valid_mode_a & (yield_a > yield_b)
        choose_b = (~choose_a) & valid_mode_b

        selected_mode_arr[choose_a] = "perp_perp_spread"
        selected_yield_arr[choose_a] = yield_a[choose_a]
        selected_short_arr[choose_a] = short_arr[choose_a]
        selected_long_arr[choose_a] = long_arr[choose_a]

        selected_mode_arr[choose_b] = "best_venue_carry"
        selected_yield_arr[choose_b] = yield_b[choose_b]
        selected_short_arr[choose_b] = short_b_arr[choose_b]
        selected_long_arr[choose_b] = "spot"

    # 4. Tính ngưỡng entry hiệu dụng và kiểm tra điều kiện can_enter
    min_entry_arr = np.full(n_rows, np.nan, dtype=np.float64)
    can_enter_arr = np.zeros(n_rows, dtype=bool)

    entry_thr = getattr(config, "entry_funding_threshold", 0.0)

    for i in range(n_rows):
        m = selected_mode_arr[i]
        if m is not None:
            min_rate = config.min_entry_rate_per_8h(str(m))
            effective_min = max(min_rate, float(entry_thr)) if (entry_thr is not None and entry_thr > 0.0) else min_rate
            min_entry_arr[i] = effective_min
            if np.isfinite(selected_yield_arr[i]) and selected_yield_arr[i] >= effective_min:
                can_enter_arr[i] = True

    res = pd.DataFrame(
        {
            "expected_yield_a": pd.Series(yield_a, index=idx, dtype="float64"),
            "expected_yield_b": pd.Series(yield_b, index=idx, dtype="float64"),
            "selected_mode": pd.Series(selected_mode_arr, index=idx, dtype="object"),
            "expected_yield": pd.Series(selected_yield_arr, index=idx, dtype="float64"),
            "min_entry_rate": pd.Series(min_entry_arr, index=idx, dtype="float64"),
            "can_enter": pd.Series(can_enter_arr, index=idx, dtype="bool"),
            "short_ex": pd.Series(selected_short_arr, index=idx, dtype="object"),
            "long_ex": pd.Series(selected_long_arr, index=idx, dtype="object"),
        },
        index=idx,
    )
    return res


def run_cross_exchange_carry(
    multi_aligned_data: dict[str, pd.DataFrame],
    config: CrossExchangeCarryConfig,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """Mô phỏng Bar-by-Bar danh mục Cross-Exchange Carry đa sàn.

    Tham số:
        multi_aligned_data: Dict map {symbol: DataFrame} đã đồng bộ từ align_multi_exchange_funding.
        config: Cấu hình chiến lược CrossExchangeCarryConfig.
        periods_per_year: Số nến 1h mỗi năm (mặc định 8760).

    Trả về:
        Dict gồm:
        - equity_curve: pd.Series (index DatetimeIndex UTC ns 'datetime', bắt đầu = initial_capital).
        - returns: pd.Series (1h pct_change fillna 0.0).
        - trades_df: pd.DataFrame chi tiết các giao dịch mở, đóng, tái cân bằng.
        - holdings_log: pd.DataFrame ảnh chụp trạng thái danh mục tại mỗi kỳ 8h.
        - attribution_series: pd.DataFrame phân rã PnL từng giờ.
        - funding_pnl: float tổng funding thu được (USDT).
        - basis_pnl: float tổng Basis MTM PnL (USDT).
        - basis_mtm_std: float độ lệch chuẩn Basis MTM từng giờ (tiêu chí GO-4).
        - fees: float tổng trading_fees + rebalance_fees (USDT).
        - rebalance_fees: float tổng phí rebalance ký quỹ liên sàn (USDT).
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
        "Total Fees Paid": 0.0,
        "Price Rebalances": 0,
        "Margin Calls": 0,
        "Maker No-Fill Count": 0,
        "Time in Market %": 0.0,
        "Turnover": 0.0,
        "Num Entries": 0,
        "Num Switches": 0,
        "Num Rebalances": 0,
        "Margin Rebalances": 0,
        "Total Basis PnL": 0.0,
        "Total MTM PnL": 0.0,
        "Net PnL": 0.0,
        "Forced Exits": 0,
    }

    trades_cols = [
        "symbol", "time", "action", "mode", "short_ex", "long_ex",
        "notional", "fee", "funding_cum", "reason", "score",
    ]
    holdings_cols = ["time", "symbols", "modes", "n_active", "total_notional", "equity"]
    attr_cols = [
        "funding_pnl", "basis_pnl", "rebalance_fees", "trading_fees",
        "fees", "net_pnl", "notional_traded",
    ]

    if not multi_aligned_data:
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        empty_trades = pd.DataFrame(columns=trades_cols)
        empty_holdings = pd.DataFrame(columns=holdings_cols)
        empty_attr = pd.DataFrame(columns=attr_cols, index=empty_idx, dtype="float64")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": empty_trades,
            "holdings_log": empty_holdings,
            "attribution_series": empty_attr,
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "basis_mtm_std": 0.0,
            "fees": 0.0,
            "rebalance_fees": 0.0,
            "turnover": 0.0,
            "metrics": empty_metrics,
        }

    # 1. Chuẩn hóa DatetimeIndex UTC ns cho toàn bộ DataFrame
    clean_aligned: dict[str, pd.DataFrame] = {}
    for sym, df in multi_aligned_data.items():
        if df is None or df.empty:
            continue
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

    if not clean_aligned:
        empty_eq = pd.Series([], index=empty_idx, dtype="float64", name="equity")
        empty_trades = pd.DataFrame(columns=trades_cols)
        empty_holdings = pd.DataFrame(columns=holdings_cols)
        empty_attr = pd.DataFrame(columns=attr_cols, index=empty_idx, dtype="float64")
        return {
            "equity_curve": empty_eq,
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": empty_trades,
            "holdings_log": empty_holdings,
            "attribution_series": empty_attr,
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "basis_mtm_std": 0.0,
            "fees": 0.0,
            "rebalance_fees": 0.0,
            "turnover": 0.0,
            "metrics": empty_metrics,
        }

    all_ts = sorted(set().union(*(d.index for d in clean_aligned.values())))
    common_index = pd.DatetimeIndex(all_ts, tz="UTC").as_unit("ns")
    common_index.name = "datetime"
    n_steps = len(common_index)

    symbols = list(clean_aligned.keys())

    # 2. Vector hóa tính toán yield/score cho từng symbol
    yield_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df_re = clean_aligned[sym].reindex(common_index)
        yield_dfs[sym] = compute_expected_8h_yield(df_re, config)

    # 3. Trích xuất mảng numpy tối ưu hiệu năng duyệt bar-by-bar
    perp_prices: dict[str, np.ndarray] = {}
    high_prices: dict[str, np.ndarray] = {}
    low_prices: dict[str, np.ndarray] = {}
    funding_events: dict[str, dict[str, np.ndarray]] = {sym: {} for sym in symbols}
    ewma_funding: dict[str, dict[str, np.ndarray]] = {sym: {} for sym in symbols}
    basis_arr: dict[str, np.ndarray] = {}

    cand_yield: dict[str, np.ndarray] = {}
    cand_mode: dict[str, list[str | None]] = {}
    cand_short_ex: dict[str, list[str | None]] = {}
    cand_long_ex: dict[str, list[str | None]] = {}

    for sym in symbols:
        df_raw = clean_aligned[sym].reindex(common_index)
        y_df = yield_dfs[sym]

        # Xác định cột giá perp: ưu tiên perp_close > close
        if "perp_close" in df_raw.columns and df_raw["perp_close"].notna().any():
            pe_s = df_raw["perp_close"].ffill().bfill()
        elif "close" in df_raw.columns:
            pe_s = df_raw["close"].ffill().bfill()
        else:
            pe_s = pd.Series(100.0, index=common_index)

        # Cột high & low cho mô hình maker fill
        hi_s = df_raw["high"].ffill().bfill() if "high" in df_raw.columns else pe_s
        lo_s = df_raw["low"].ffill().bfill() if "low" in df_raw.columns else pe_s

        perp_prices[sym] = pe_s.to_numpy(dtype=np.float64)
        high_prices[sym] = hi_s.to_numpy(dtype=np.float64)
        low_prices[sym] = lo_s.to_numpy(dtype=np.float64)

        for ex in SUPPORTED_EXCHANGES:
            fe_col = f"funding_event_{ex}"
            if fe_col in df_raw.columns:
                funding_events[sym][ex] = df_raw[fe_col].fillna(0.0).to_numpy(dtype=np.float64)
            else:
                funding_events[sym][ex] = np.zeros(n_steps, dtype=np.float64)

            ewma_col = f"ewma_funding_{ex}"
            if ewma_col in df_raw.columns:
                ewma_funding[sym][ex] = df_raw[ewma_col].to_numpy(dtype=np.float64)
            else:
                ewma_funding[sym][ex] = np.full(n_steps, np.nan, dtype=np.float64)

        # Trích mảng basis_pct cho True Basis MTM
        if "basis_pct" in df_raw.columns:
            basis_arr[sym] = df_raw["basis_pct"].fillna(0.0).to_numpy(dtype=np.float64)
        else:
            basis_arr[sym] = np.zeros(n_steps, dtype=np.float64)

        cand_yield[sym] = y_df["expected_yield"].to_numpy(dtype=np.float64)
        cand_mode[sym] = y_df["selected_mode"].tolist()
        cand_short_ex[sym] = y_df["short_ex"].tolist()
        cand_long_ex[sym] = y_df["long_ex"].tolist()

    # 4. Khởi tạo trạng thái danh mục
    equity = float(config.initial_capital)
    buffer_mult = float(config.maker_fill_buffer_bps) * 1e-4

    in_position: dict[str, bool] = {sym: False for sym in symbols}
    periods_held: dict[str, int] = {sym: 0 for sym in symbols}
    notional: dict[str, float] = {sym: 0.0 for sym in symbols}
    ref_perp_price: dict[str, float] = {sym: 0.0 for sym in symbols}
    current_mode: dict[str, str] = {sym: "" for sym in symbols}
    current_short_ex: dict[str, str] = {sym: "" for sym in symbols}
    current_long_ex: dict[str, str] = {sym: "" for sym in symbols}
    trade_funding_cum: dict[str, float] = {sym: 0.0 for sym in symbols}

    pending_maker_orders: dict[str, dict[str, Any]] = {}

    total_funding_collected = 0.0
    total_trading_fees = 0.0
    total_rebalance_fees = 0.0
    total_notional_traded = 0.0
    price_rebalances = 0
    maker_no_fill_count = 0
    num_entries = 0
    num_switches = 0
    forced_exits = 0

    trades_list: list[dict[str, Any]] = []
    holdings_log_list: list[dict[str, Any]] = []

    equity_vals = np.empty(n_steps, dtype=np.float64)
    in_market_any = np.zeros(n_steps, dtype=bool)

    hourly_funding_pnl = np.zeros(n_steps, dtype=np.float64)
    hourly_basis_pnl = np.zeros(n_steps, dtype=np.float64)
    prev_basis: dict[str, float] = {sym: 0.0 for sym in symbols}
    hourly_rebalance_fees = np.zeros(n_steps, dtype=np.float64)
    hourly_trading_fees = np.zeros(n_steps, dtype=np.float64)
    hourly_notional_traded = np.zeros(n_steps, dtype=np.float64)

    hours = common_index.hour
    minutes = common_index.minute

    # 5. Vòng lặp mô phỏng bar-by-bar
    for t_idx, t in enumerate(common_index):
        # Lưu notional tại đầu nến t trước mọi thay đổi trong nến t
        notional_start = {s: notional[s] for s in symbols}

        # -----------------------------------------------------------------
        # Bước 1 — Mô hình khớp lệnh Maker Fill tất định cho pending orders
        # -----------------------------------------------------------------
        if pending_maker_orders:
            for sym in list(pending_maker_orders.keys()):
                order = pending_maker_orders[sym]
                hi_t = high_prices[sym][t_idx]
                limit_p = order["limit_price"]

                # Chân 1 lệnh Bán (Short Perp): Khớp khi high_t >= limit_p
                # Lệnh Mua: Khớp khi low_t <= limit_p
                is_filled = False
                if order["side"] == "sell":
                    if not np.isnan(hi_t) and hi_t >= limit_p:
                        is_filled = True
                else:
                    lo_t = low_prices[sym][t_idx]
                    if not np.isnan(lo_t) and lo_t <= limit_p:
                        is_filled = True

                if is_filled and not in_position[sym] and equity > 0:
                    leg_notional = float(order["notional"])
                    if getattr(config, "lot_size_quantization", False) and leg_notional <= 0.0:
                        maker_no_fill_count += 1
                        del pending_maker_orders[sym]
                        continue

                    fee = leg_notional * float(order["one_way_cost"])

                    equity -= fee
                    total_trading_fees += fee
                    hourly_trading_fees[t_idx] += fee
                    total_notional_traded += leg_notional
                    hourly_notional_traded[t_idx] += leg_notional

                    in_position[sym] = True
                    notional[sym] = leg_notional
                    periods_held[sym] = 0
                    trade_funding_cum[sym] = 0.0
                    ref_perp_price[sym] = perp_prices[sym][t_idx]
                    current_mode[sym] = order["mode"]
                    current_short_ex[sym] = order["short_ex"]
                    current_long_ex[sym] = order["long_ex"]
                    num_entries += 1

                    trades_list.append({
                        "symbol": sym,
                        "time": t,
                        "action": "open",
                        "mode": order["mode"],
                        "short_ex": order["short_ex"],
                        "long_ex": order["long_ex"],
                        "notional": float(leg_notional),
                        "fee": float(fee),
                        "funding_cum": 0.0,
                        "reason": "OPEN_MAKER_FILLED",
                        "score": float(order["score"]),
                    })
                    del pending_maker_orders[sym]

                else:
                    order["ttl_remaining"] -= 1
                    if order["ttl_remaining"] <= 0:
                        maker_no_fill_count += 1
                        del pending_maker_orders[sym]

        # -----------------------------------------------------------------
        # Bước 2 — Quản trị ký quỹ liên sàn & Tự động Rebalance ±25%
        # -----------------------------------------------------------------
        for sym in symbols:
            if in_position[sym]:
                pe_curr = perp_prices[sym][t_idx]
                ref_p = ref_perp_price[sym]
                if ref_p > 0 and not np.isnan(pe_curr):
                    price_move = abs(pe_curr - ref_p) / ref_p
                    if price_move >= config.price_rebalance_trigger_pct:
                        reb_fee = notional[sym] * (config.rebalance_cost_bps * 1e-4)
                        equity -= reb_fee
                        total_rebalance_fees += reb_fee
                        hourly_rebalance_fees[t_idx] += reb_fee
                        total_notional_traded += notional[sym]
                        hourly_notional_traded[t_idx] += notional[sym]
                        price_rebalances += 1
                        ref_perp_price[sym] = pe_curr

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "rebalance",
                            "mode": current_mode[sym],
                            "short_ex": current_short_ex[sym],
                            "long_ex": current_long_ex[sym],
                            "notional": float(notional[sym]),
                            "fee": float(reb_fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                            "reason": "PRICE_REBALANCE_25PCT",
                            "score": float(cand_yield[sym][t_idx]) if not np.isnan(cand_yield[sym][t_idx]) else 0.0,
                        })

        # -----------------------------------------------------------------
        # Bước 3 — Thanh toán Funding Rate (tại mốc 00:00, 08:00, 16:00 UTC)
        # -----------------------------------------------------------------
        is_8h_event = (hours[t_idx] in (0, 8, 16)) and (minutes[t_idx] == 0)

        if is_8h_event:
            for sym in symbols:
                if in_position[sym]:
                    m = current_mode[sym]
                    sh_ex = current_short_ex[sym]
                    lo_ex = current_long_ex[sym]

                    if m == "perp_perp_spread":
                        fe_sh = float(funding_events[sym][sh_ex][t_idx]) if sh_ex in funding_events[sym] else 0.0
                        fe_lo = float(funding_events[sym][lo_ex][t_idx]) if lo_ex in funding_events[sym] else 0.0
                        f_pnl = notional[sym] * (fe_sh - fe_lo)
                    else:  # best_venue_carry / single_binance
                        fe_sh = float(funding_events[sym][sh_ex][t_idx]) if sh_ex in funding_events[sym] else 0.0
                        f_pnl = notional[sym] * fe_sh

                    equity += f_pnl
                    trade_funding_cum[sym] += f_pnl
                    total_funding_collected += f_pnl
                    hourly_funding_pnl[t_idx] += f_pnl
                    periods_held[sym] += 1

        # -----------------------------------------------------------------
        # Bước 3b — Mark-to-Market (MTM) Basis PnL từng nến 1h
        # -----------------------------------------------------------------
        for sym in symbols:
            if in_position[sym] and notional_start[sym] > 0.0:
                delta_basis = -notional_start[sym] * (basis_arr[sym][t_idx] - prev_basis[sym])
                hourly_basis_pnl[t_idx] += delta_basis
                equity += delta_basis
            prev_basis[sym] = float(basis_arr[sym][t_idx])

        # -----------------------------------------------------------------
        # Bước 4 — Hysteresis & Xoay Vòng Danh Mục (tại mốc 8h)
        # -----------------------------------------------------------------
        if is_8h_event:
            # 4.1. Đánh giá vị thế hiện tại: Thoát âm sâu hoặc chuyển sàn (Switch)
            for sym in symbols:
                if in_position[sym]:
                    sh_ex = current_short_ex[sym]
                    lo_ex = current_long_ex[sym]
                    m = current_mode[sym]

                    # Tính lợi suất kỳ vọng hiện tại của vị thế
                    if m == "perp_perp_spread":
                        ew_sh = ewma_funding[sym][sh_ex][t_idx] if sh_ex in ewma_funding[sym] else np.nan
                        ew_lo = ewma_funding[sym][lo_ex][t_idx] if lo_ex in ewma_funding[sym] else np.nan
                        if np.isfinite(ew_sh) and np.isfinite(ew_lo):
                            curr_hold_yield = (ew_sh - ew_lo) * float(config.perp_perp_eta)
                        else:
                            curr_hold_yield = -np.inf
                        one_way_cost_curr = config.one_way_cost_perp_perp()
                    else:
                        ew_sh = ewma_funding[sym][sh_ex][t_idx] if sh_ex in ewma_funding[sym] else np.nan
                        eta_b = float(config.coinm_eta if config.structure_b == "coinm_1leg" else config.spot_perp_eta)
                        curr_hold_yield = ew_sh * eta_b if np.isfinite(ew_sh) else -np.inf
                        one_way_cost_curr = config.one_way_cost_spot_perp()

                    close_reason = ""

                    # Điều kiện A: Thoát vị thế âm sâu khi đã giữ đủ min_hold_periods
                    thr = (
                        config.exit_funding_threshold
                        if config.exit_funding_threshold is not None
                        else config.exit_yield_threshold
                    )
                    if periods_held[sym] >= config.min_hold_periods and curr_hold_yield < thr:
                        close_reason = "EXIT_NEGATIVE_YIELD"
                        forced_exits += 1

                    # Điều kiện B: Switch sang phương án mới tốt hơn vượt trội
                    elif periods_held[sym] >= config.min_hold_periods:
                        best_cand_y = cand_yield[sym][t_idx]
                        best_cand_m = cand_mode[sym][t_idx]
                        best_cand_sh = cand_short_ex[sym][t_idx]
                        best_cand_lo = cand_long_ex[sym][t_idx]

                        # Kiểm tra xem phương án tốt nhất trên symbol này có phải sàn mới không
                        is_new_plan = (
                            best_cand_m is not None
                            and (best_cand_m != m or best_cand_sh != sh_ex or best_cand_lo != lo_ex)
                        )
                        if is_new_plan and np.isfinite(best_cand_y):
                            rt_cost_new = config.round_trip_cost_for_mode(best_cand_m)
                            # Chênh lệch lợi suất kỳ vọng 21 kỳ > 3 * round_trip_cost
                            if (best_cand_y - curr_hold_yield) * config.horizon_periods_7d > config.switch_cost_multiplier * rt_cost_new:
                                close_reason = "SWITCH_BETTER_YIELD"
                                num_switches += 1

                    if close_reason:
                        # Thực thi ĐÓNG vị thế (không cộng thêm basis vì đã MTM bar-by-bar)
                        pos_notional = notional[sym]
                        close_fee = pos_notional * one_way_cost_curr
                        equity -= close_fee
                        total_trading_fees += close_fee
                        hourly_trading_fees[t_idx] += close_fee
                        total_notional_traded += pos_notional
                        hourly_notional_traded[t_idx] += pos_notional

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "close",
                            "mode": m,
                            "short_ex": sh_ex,
                            "long_ex": lo_ex,
                            "notional": float(pos_notional),
                            "fee": float(close_fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                            "reason": close_reason,
                            "score": float(curr_hold_yield) if np.isfinite(curr_hold_yield) else 0.0,
                        })

                        in_position[sym] = False
                        notional[sym] = 0.0
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0
                        ref_perp_price[sym] = 0.0
                        current_mode[sym] = ""
                        current_short_ex[sym] = ""
                        current_long_ex[sym] = ""

            # 4.2. Mở vị thế mới (Top-k ranking)
            active_symbols = [s for s in symbols if in_position[s]]
            n_slots_available = max(0, config.top_k - len(active_symbols))

            if n_slots_available > 0 and equity > 0:
                candidates: list[tuple[float, str, str, str, str]] = []
                entry_thr = getattr(config, "entry_funding_threshold", 0.0)
                for sym in symbols:
                    if not in_position[sym] and sym not in pending_maker_orders:
                        y = cand_yield[sym][t_idx]
                        m = cand_mode[sym][t_idx]
                        sh_ex = cand_short_ex[sym][t_idx]
                        lo_ex = cand_long_ex[sym][t_idx]

                        if m is not None and np.isfinite(y) and sh_ex is not None:
                            min_rate = config.min_entry_rate_per_8h(m)
                            effective_min = max(min_rate, float(entry_thr)) if (entry_thr is not None and entry_thr > 0.0) else min_rate
                            if y >= effective_min:
                                candidates.append((float(y), sym, m, str(sh_ex), str(lo_ex)))

                # Sắp xếp giảm dần theo expected yield
                candidates.sort(key=lambda item: item[0], reverse=True)

                for y_val, sym, m, sh_ex, lo_ex in candidates[:n_slots_available]:
                    allocated = equity / float(config.top_k)
                    eta = float(
                        config.perp_perp_eta if m == "perp_perp_spread"
                        else (config.coinm_eta if config.structure_b == "coinm_1leg" else config.spot_perp_eta)
                    )
                    leg_notional = allocated * eta
                    one_way_cost = float(
                        config.one_way_cost_perp_perp() if m == "perp_perp_spread"
                        else config.one_way_cost_spot_perp()
                    )

                    pe_prev = perp_prices[sym][t_idx - 1] if t_idx > 0 else perp_prices[sym][t_idx]
                    if np.isnan(pe_prev) or pe_prev <= 0:
                        pe_prev = perp_prices[sym][t_idx]

                    # Chân 1 Limit Post-Only: Bán treo cao hơn giá đóng cửa t-1
                    limit_p = pe_prev * (1.0 + buffer_mult)

                    entry_price = float(
                        limit_p
                        if (config.enable_maker_fill_model and config.execution == "maker_taker_hedge")
                        else perp_prices[sym][t_idx]
                    )
                    if np.isnan(entry_price) or entry_price <= 0:
                        entry_price = float(perp_prices[sym][t_idx]) if not np.isnan(perp_prices[sym][t_idx]) and perp_prices[sym][t_idx] > 0 else 1.0

                    # Lượng hóa lot-size theo sàn
                    if getattr(config, "lot_size_quantization", False):
                        q_qty, actual_notional = quantize_order_qty(sym, leg_notional, entry_price)
                        if q_qty <= 0.0 or actual_notional <= 0.0:
                            maker_no_fill_count += 1
                            continue
                        leg_notional = actual_notional

                    if config.enable_maker_fill_model and config.execution == "maker_taker_hedge":
                        pending_maker_orders[sym] = {
                            "mode": m,
                            "short_ex": sh_ex,
                            "long_ex": lo_ex,
                            "notional": float(leg_notional),
                            "one_way_cost": float(one_way_cost),
                            "side": "sell",
                            "limit_price": float(limit_p),
                            "ttl_remaining": int(config.maker_order_ttl_bars),
                            "score": float(y_val),
                        }
                    else:
                        # Khớp ngay lập tức nếu không bật mô hình fill
                        fee = leg_notional * one_way_cost
                        equity -= fee
                        total_trading_fees += fee
                        hourly_trading_fees[t_idx] += fee
                        total_notional_traded += leg_notional
                        hourly_notional_traded[t_idx] += leg_notional

                        in_position[sym] = True
                        notional[sym] = leg_notional
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0
                        ref_perp_price[sym] = perp_prices[sym][t_idx]
                        current_mode[sym] = m
                        current_short_ex[sym] = sh_ex
                        current_long_ex[sym] = lo_ex
                        num_entries += 1

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "open",
                            "mode": m,
                            "short_ex": sh_ex,
                            "long_ex": lo_ex,
                            "notional": float(leg_notional),
                            "fee": float(fee),
                            "funding_cum": 0.0,
                            "reason": "OPEN_IMMEDIATE",
                            "score": float(y_val),
                        })

            # Ghi snapshot danh mục tại mốc 8h
            active_s = [s for s in symbols if in_position[s]]
            active_m = [current_mode[s] for s in active_s]
            tot_not = sum(notional[s] for s in active_s)
            holdings_log_list.append({
                "time": t,
                "symbols": active_s,
                "modes": active_m,
                "n_active": len(active_s),
                "total_notional": float(tot_not),
                "equity": float(equity),
            })

        # -----------------------------------------------------------------
        # Bước 5 — Ghi nhận vốn và trạng thái từng nến
        # -----------------------------------------------------------------
        equity_vals[t_idx] = equity
        in_market_any[t_idx] = any(in_position.values())

    # 6. Xây dựng các DataFrame kết quả
    equity_curve = pd.Series(equity_vals, index=common_index, dtype="float64", name="equity")
    returns = equity_curve.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=trades_cols)
    for col in trades_cols:
        if col not in trades_df.columns:
            trades_df[col] = np.nan

    holdings_log = pd.DataFrame(holdings_log_list) if holdings_log_list else pd.DataFrame(columns=holdings_cols)

    hourly_fees = hourly_trading_fees + hourly_rebalance_fees
    hourly_basis = hourly_basis_pnl
    hourly_net_pnl = hourly_funding_pnl + hourly_basis - hourly_fees

    attribution_series = pd.DataFrame(
        {
            "funding_pnl": hourly_funding_pnl,
            "basis_pnl": hourly_basis,
            "rebalance_fees": hourly_rebalance_fees,
            "trading_fees": hourly_trading_fees,
            "fees": hourly_fees,
            "net_pnl": hourly_net_pnl,
            "notional_traded": hourly_notional_traded,
        },
        index=common_index,
        dtype="float64",
    )

    # 7. Tính toán các chỉ số hiệu suất tổng hợp
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
    valid_peak = peak > 0
    if np.any(valid_peak):
        dd = (peak[valid_peak] - equity_vals[valid_peak]) / peak[valid_peak]
        max_dd_pct = float(np.nanmax(dd) * 100.0)
    else:
        max_dd_pct = 0.0

    time_in_market_pct = float(np.mean(in_market_any) * 100.0) if len(in_market_any) > 0 else 0.0
    turnover = (
        float(total_notional_traded / config.initial_capital)
        if config.initial_capital > 0
        else 0.0
    )

    total_fees_paid = total_trading_fees + total_rebalance_fees
    total_basis_pnl = float(hourly_basis_pnl.sum())
    basis_mtm_std = float(np.std(hourly_basis_pnl)) if len(hourly_basis_pnl) > 0 else 0.0

    metrics = {
        "Total Return %": total_return_pct,
        "Total Return % ": total_return_pct,
        "Annualized Return %": annualized_return_pct,
        "Sharpe": sharpe,
        "MaxDD %": max_dd_pct,
        "Total Funding Collected": float(total_funding_collected),
        "Total Fees Paid": float(total_fees_paid),
        "Price Rebalances": int(price_rebalances),
        "Margin Calls": 0,
        "Maker No-Fill Count": int(maker_no_fill_count),
        "Time in Market %": time_in_market_pct,
        "Turnover": turnover,
        "Num Entries": int(num_entries),
        "Num Switches": int(num_switches),
        "Num Rebalances": int(price_rebalances),
        "Margin Rebalances": 0,
        "Total Basis PnL": total_basis_pnl,
        "Total MTM PnL": total_basis_pnl,
        "Net PnL": float(equity - config.initial_capital),
        "Forced Exits": int(forced_exits),
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades_df": trades_df,
        "holdings_log": holdings_log,
        "attribution_series": attribution_series,
        "funding_pnl": float(total_funding_collected),
        "basis_pnl": total_basis_pnl,
        "basis_mtm_std": basis_mtm_std,
        "fees": float(total_fees_paid),
        "rebalance_fees": float(total_rebalance_fees),
        "turnover": turnover,
        "metrics": metrics,
    }



def run_binance_tier1_plus_carry(
    multi_aligned_data: dict[str, pd.DataFrame],
    config: CrossExchangeCarryConfig | None = None,
    *,
    variant: Literal["tier1a", "tier1b"] = "tier1a",
    top_k: int = 4,
    buffer_rank: int = 3,
    ema_span: int = 42,
    min_hold_periods: int = 63,
    entry_funding_threshold: float = 0.000045,
    exit_funding_threshold: float = -0.00001,
    eta: float = 0.80,
    periods_per_year: int = 8760,
    trade_start_idx: int | None = None,
) -> dict[str, Any]:
    """Mô phỏng Bar-by-bar chiến lược Binance Single-Venue Tier 1+ Carry.

    Hỗ trợ Chế độ Tier 1A (vũ trụ Core 4 tĩnh) và Tier 1B (vũ trụ Liquid 8 động kèm Patient Buffer).
    Hiệu suất sử dụng vốn Unified Margin eta=0.80, True Basis MTM bar-by-bar, lượng hóa lot-size.

    Tham số:
        multi_aligned_data: Dict map {symbol: DataFrame} dữ liệu 1h đồng bộ từ align_multi_exchange_funding.
        config: Cấu hình CrossExchangeCarryConfig hoặc None (dùng giá trị mặc định).
        variant: 'tier1a' (core 4 tĩnh) hoặc 'tier1b' (liquid 8 động kèm patient buffer).
        top_k: Số vị thế tối đa được nắm giữ đồng thời (mặc định 4).
        buffer_rank: Khoảng đệm thứ hạng bảo vệ trước khi đảo vị thế (mặc định 3).
        ema_span: Chu kỳ EMA funding 8h (mặc định 42 kỳ 8h = 14 ngày).
        min_hold_periods: Số kỳ funding tối thiểu phải giữ trước khi thoát/đảo (mặc định 63 kỳ = 21 ngày).
        entry_funding_threshold: Ngưỡng EWMA funding tối thiểu để mở vị thế (mặc định 0.45 bps/8h).
        exit_funding_threshold: Ngưỡng EWMA funding đóng vị thế khi âm/hết hấp dẫn (mặc định -0.10 bps/8h).
        eta: Hệ số hiệu suất sử dụng vốn Unified Margin (mặc định 0.80).
        periods_per_year: Số nến 1h mỗi năm (mặc định 8760).

    Trả về:
        Dict chứa equity_curve, returns, trades_df, holdings_log, attribution_series,
        funding_pnl, basis_pnl, basis_mtm_std, fees, rebalance_fees, turnover,
        variant, symbol_universe, metrics.
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
        "Price Rebalances": 0,
        "Margin Rebalances": 0,
        "Margin Calls": 0,
        "Maker No-Fill Count": 0,
        "Time in Market %": 0.0,
        "Turnover": 0.0,
        "Num Entries": 0,
        "Num Switches": 0,
        "Num Rebalances": 0,
        "Total Basis PnL": 0.0,
        "Total MTM PnL": 0.0,
        "Net PnL": 0.0,
        "Forced Exits": 0,
    }

    trades_cols = [
        "symbol", "time", "action", "mode", "short_ex", "long_ex",
        "notional", "fee", "funding_cum", "reason", "score",
    ]
    holdings_cols = ["time", "symbols", "modes", "n_active", "total_notional", "equity"]
    attr_cols = [
        "funding_pnl", "basis_pnl", "rebalance_fees", "trading_fees",
        "fees", "net_pnl", "notional_traded",
    ]

    target_universe = CORE_4_SYMBOLS if variant == "tier1a" else LIQUID_8_SYMBOLS

    if not multi_aligned_data:
        return {
            "equity_curve": pd.Series([], index=empty_idx, dtype="float64", name="equity"),
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": pd.DataFrame(columns=trades_cols),
            "holdings_log": pd.DataFrame(columns=holdings_cols),
            "attribution_series": pd.DataFrame(columns=attr_cols, index=empty_idx, dtype="float64"),
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "basis_mtm_std": 0.0,
            "fees": 0.0,
            "rebalance_fees": 0.0,
            "turnover": 0.0,
            "variant": variant,
            "symbol_universe": [],
            "metrics": empty_metrics,
        }

    # 1. Chuẩn hóa DatetimeIndex UTC ns
    clean_aligned: dict[str, pd.DataFrame] = {}
    for sym, df in multi_aligned_data.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            if "datetime" in d.columns:
                d = d.set_index("datetime")
        if not isinstance(d.index, pd.DatetimeIndex):
            continue
        idx = pd.DatetimeIndex(d.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        idx = idx.as_unit("ns")
        idx.name = "datetime"
        d.index = idx
        clean_aligned[sym] = d

    # Lọc universe theo cấu hình variant
    sym_universe: list[str] = []
    for s in target_universe:
        if s in clean_aligned:
            sym_universe.append(s)
        elif s.replace(":USDT", "") in clean_aligned:
            sym_universe.append(s.replace(":USDT", ""))
        elif f"{s}:USDT" in clean_aligned:
            sym_universe.append(f"{s}:USDT")

    if not sym_universe:
        return {
            "equity_curve": pd.Series([], index=empty_idx, dtype="float64", name="equity"),
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": pd.DataFrame(columns=trades_cols),
            "holdings_log": pd.DataFrame(columns=holdings_cols),
            "attribution_series": pd.DataFrame(columns=attr_cols, index=empty_idx, dtype="float64"),
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "basis_mtm_std": 0.0,
            "fees": 0.0,
            "rebalance_fees": 0.0,
            "turnover": 0.0,
            "variant": variant,
            "symbol_universe": [],
            "metrics": empty_metrics,
        }

    all_ts = sorted(set().union(*(clean_aligned[s].index for s in sym_universe)))
    if not all_ts:
        return {
            "equity_curve": pd.Series([], index=empty_idx, dtype="float64", name="equity"),
            "returns": pd.Series([], index=empty_idx, dtype="float64"),
            "trades_df": pd.DataFrame(columns=trades_cols),
            "holdings_log": pd.DataFrame(columns=holdings_cols),
            "attribution_series": pd.DataFrame(columns=attr_cols, index=empty_idx, dtype="float64"),
            "funding_pnl": 0.0,
            "basis_pnl": 0.0,
            "basis_mtm_std": 0.0,
            "fees": 0.0,
            "rebalance_fees": 0.0,
            "turnover": 0.0,
            "variant": variant,
            "symbol_universe": sym_universe,
            "metrics": empty_metrics,
        }

    common_index = pd.DatetimeIndex(all_ts, tz="UTC").as_unit("ns")
    common_index.name = "datetime"
    n_steps = len(common_index)

    # 2. Khởi tạo tham số chi phí và hiệu suất vốn
    initial_capital = float(config.initial_capital) if config is not None else 3000.0
    one_way_cost = float(config.one_way_cost_spot_perp()) if config is not None else 0.001125
    lot_quant = bool(getattr(config, "lot_size_quantization", True)) if config is not None else True
    price_rebalance_trigger_pct = 0.15
    rebalance_cost_bps = float(getattr(config, "rebalance_cost_bps", 5.0)) if config is not None else 5.0
    enable_maker_fill = bool(
        getattr(config, "enable_maker_fill_model", False)
        and getattr(config, "execution", "maker_taker_hedge") == "maker_taker_hedge"
    ) if config is not None else False
    buffer_mult = float(getattr(config, "maker_fill_buffer_bps", 0.5)) * 1e-4 if config is not None else 0.5 * 1e-4
    order_ttl = int(getattr(config, "maker_order_ttl_bars", 2)) if config is not None else 2

    # 3. Tiền trích xuất mảng numpy tối ưu hiệu năng duyệt bar-by-bar
    perp_prices: dict[str, np.ndarray] = {}
    high_prices: dict[str, np.ndarray] = {}
    low_prices: dict[str, np.ndarray] = {}
    funding_events: dict[str, np.ndarray] = {}
    ewma_funding_42: dict[str, np.ndarray] = {}
    basis_arr: dict[str, np.ndarray] = {}

    for sym in sym_universe:
        df_sym = clean_aligned[sym].reindex(common_index)

        # Cột giá perp: ưu tiên perp_close > close
        if "perp_close" in df_sym.columns and df_sym["perp_close"].notna().any():
            pe_s = df_sym["perp_close"].ffill().bfill()
        elif "close" in df_sym.columns:
            pe_s = df_sym["close"].ffill().bfill()
        else:
            pe_s = pd.Series(100.0, index=common_index)

        hi_s = df_sym["high"].ffill().bfill() if "high" in df_sym.columns else pe_s
        lo_s = df_sym["low"].ffill().bfill() if "low" in df_sym.columns else pe_s

        perp_prices[sym] = pe_s.to_numpy(dtype=np.float64)
        high_prices[sym] = hi_s.to_numpy(dtype=np.float64)
        low_prices[sym] = lo_s.to_numpy(dtype=np.float64)

        # Cột funding event Binance và EWMA funding 8h reindex 1h shift(1)
        fe_col = None
        for cand in ("funding_event_binance", "funding_rate_event", "funding_event"):
            if cand in df_sym.columns:
                fe_col = cand
                break

        if fe_col is not None:
            funding_events[sym] = df_sym[fe_col].fillna(0.0).to_numpy(dtype=np.float64)
            fe_s = df_sym[fe_col]
            mask_8h = (common_index.hour.isin([0, 8, 16])) & (common_index.minute == 0)
            fe_8h = fe_s.loc[mask_8h].dropna()
            if not fe_8h.empty:
                ewma_8h = fe_8h.ewm(span=ema_span, adjust=True).mean()
                ewma_1h = ewma_8h.reindex(common_index).ffill()
                ewma_funding_42[sym] = ewma_1h.shift(1).astype("float64").to_numpy(dtype=np.float64)
            else:
                ewma_funding_42[sym] = np.full(n_steps, np.nan, dtype=np.float64)
        elif "ewma_funding_binance" in df_sym.columns:
            funding_events[sym] = np.zeros(n_steps, dtype=np.float64)
            ewma_funding_42[sym] = df_sym["ewma_funding_binance"].to_numpy(dtype=np.float64)
        else:
            funding_events[sym] = np.zeros(n_steps, dtype=np.float64)
            ewma_funding_42[sym] = np.full(n_steps, np.nan, dtype=np.float64)

        # Cột basis_pct cho True Basis MTM
        if "basis_pct" in df_sym.columns:
            basis_arr[sym] = df_sym["basis_pct"].fillna(0.0).to_numpy(dtype=np.float64)
        elif "perp_close" in df_sym.columns and "spot_close" in df_sym.columns:
            sc = df_sym["spot_close"].replace(0.0, np.nan)
            basis_arr[sym] = ((df_sym["perp_close"] - sc) / sc).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            basis_arr[sym] = np.zeros(n_steps, dtype=np.float64)

    # 4. Khởi tạo trạng thái danh mục
    equity = float(initial_capital)
    in_position: dict[str, bool] = {s: False for s in sym_universe}
    periods_held: dict[str, int] = {s: 0 for s in sym_universe}
    notional: dict[str, float] = {s: 0.0 for s in sym_universe}
    ref_perp_price: dict[str, float] = {s: 0.0 for s in sym_universe}
    trade_funding_cum: dict[str, float] = {s: 0.0 for s in sym_universe}
    prev_basis: dict[str, float] = {s: float(basis_arr[s][0]) if n_steps > 0 else 0.0 for s in sym_universe}

    pending_maker_orders: dict[str, dict[str, Any]] = {}

    total_funding_collected = 0.0
    total_trading_fees = 0.0
    total_rebalance_fees = 0.0
    total_notional_traded = 0.0
    price_rebalances = 0
    margin_rebalances = 0
    maker_no_fill_count = 0
    num_entries = 0
    num_switches = 0
    forced_exits = 0

    trades_list: list[dict[str, Any]] = []
    holdings_log_list: list[dict[str, Any]] = []

    equity_vals = np.empty(n_steps, dtype=np.float64)
    in_market_any = np.zeros(n_steps, dtype=bool)

    hourly_funding_pnl = np.zeros(n_steps, dtype=np.float64)
    hourly_basis_pnl = np.zeros(n_steps, dtype=np.float64)
    hourly_rebalance_fees = np.zeros(n_steps, dtype=np.float64)
    hourly_trading_fees = np.zeros(n_steps, dtype=np.float64)
    hourly_notional_traded = np.zeros(n_steps, dtype=np.float64)

    hours = common_index.hour
    minutes = common_index.minute

    # 5. Vòng lặp mô phỏng bar-by-bar (1 tầng trên trục thời gian)
    for t_idx, t in enumerate(common_index):
        # Sprint 14: giu 100% tien mat truoc moc trade_start_idx (chi warm indicator).
        if trade_start_idx is not None and t_idx < int(trade_start_idx):
            equity_vals[t_idx] = equity
            continue

        notional_start = {s: notional[s] for s in sym_universe}

        # -------------------------------------------------------------
        # Bước 1: Mô hình Maker Fill cho pending orders (nếu bật)
        # -------------------------------------------------------------
        if pending_maker_orders:
            for sym in list(pending_maker_orders.keys()):
                order = pending_maker_orders[sym]
                hi_t = high_prices[sym][t_idx]
                limit_p = order["limit_price"]
                is_filled = not np.isnan(hi_t) and hi_t >= limit_p

                if is_filled and not in_position[sym] and equity > 0:
                    leg_notional = float(order["notional"])
                    if lot_quant and leg_notional <= 0.0:
                        maker_no_fill_count += 1
                        del pending_maker_orders[sym]
                        continue

                    fee = leg_notional * float(order["one_way_cost"])
                    equity -= fee
                    total_trading_fees += fee
                    hourly_trading_fees[t_idx] += fee
                    total_notional_traded += leg_notional
                    hourly_notional_traded[t_idx] += leg_notional

                    in_position[sym] = True
                    notional[sym] = leg_notional
                    periods_held[sym] = 0
                    trade_funding_cum[sym] = 0.0
                    ref_perp_price[sym] = perp_prices[sym][t_idx]
                    num_entries += 1

                    trades_list.append({
                        "symbol": sym,
                        "time": t,
                        "action": "open",
                        "mode": variant,
                        "short_ex": "binance",
                        "long_ex": "spot",
                        "notional": float(leg_notional),
                        "fee": float(fee),
                        "funding_cum": 0.0,
                        "reason": "OPEN_MAKER_FILLED",
                        "score": float(order["score"]),
                    })
                    del pending_maker_orders[sym]
                else:
                    order["ttl_remaining"] -= 1
                    if order["ttl_remaining"] <= 0:
                        maker_no_fill_count += 1
                        del pending_maker_orders[sym]

        # -------------------------------------------------------------
        # Bước 2: Rebalance ký quỹ khi biến động giá >= ±15%
        # -------------------------------------------------------------
        for sym in sym_universe:
            if in_position[sym]:
                pe_curr = perp_prices[sym][t_idx]
                ref_p = ref_perp_price[sym]
                if ref_p > 0.0 and not np.isnan(pe_curr):
                    price_move = abs(pe_curr - ref_p) / ref_p
                    if price_move >= price_rebalance_trigger_pct:
                        reb_fee = notional[sym] * (rebalance_cost_bps * 1e-4)
                        equity -= reb_fee
                        total_rebalance_fees += reb_fee
                        hourly_rebalance_fees[t_idx] += reb_fee
                        total_notional_traded += notional[sym]
                        hourly_notional_traded[t_idx] += notional[sym]
                        price_rebalances += 1
                        margin_rebalances += 1
                        ref_perp_price[sym] = pe_curr

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "rebalance",
                            "mode": variant,
                            "short_ex": "binance",
                            "long_ex": "spot",
                            "notional": float(notional[sym]),
                            "fee": float(reb_fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                            "reason": "PRICE_REBALANCE_15PCT",
                            "score": float(ewma_funding_42[sym][t_idx]) if np.isfinite(ewma_funding_42[sym][t_idx]) else 0.0,
                        })

        # -------------------------------------------------------------
        # Bước 3: Thu Funding Rate tại các mốc 8h (00, 08, 16 UTC)
        # -------------------------------------------------------------
        is_8h_event = (hours[t_idx] in (0, 8, 16)) and (minutes[t_idx] == 0)
        if is_8h_event:
            for sym in sym_universe:
                if in_position[sym]:
                    fe = float(funding_events[sym][t_idx])
                    f_pnl = notional[sym] * fe
                    equity += f_pnl
                    trade_funding_cum[sym] += f_pnl
                    total_funding_collected += f_pnl
                    hourly_funding_pnl[t_idx] += f_pnl
                    periods_held[sym] += 1

        # -------------------------------------------------------------
        # Bước 3b: True Basis MTM PnL từng nến 1h
        # -------------------------------------------------------------
        for sym in sym_universe:
            if in_position[sym] and notional_start[sym] > 0.0:
                delta_basis = -notional_start[sym] * (basis_arr[sym][t_idx] - prev_basis[sym])
                hourly_basis_pnl[t_idx] += delta_basis
                equity += delta_basis
            prev_basis[sym] = float(basis_arr[sym][t_idx])

        # -------------------------------------------------------------
        # Bước 4: Hysteresis, Patient Buffer & Xoay vòng danh mục (mỗi 8h)
        # -------------------------------------------------------------
        if is_8h_event:
            scores = {s: ewma_funding_42[s][t_idx] for s in sym_universe}
            ranked_symbols = sorted(
                sym_universe,
                key=lambda s: scores[s] if np.isfinite(scores[s]) else -np.inf,
                reverse=True,
            )
            ranks = {s: i + 1 for i, s in enumerate(ranked_symbols)}
            drop_rank_threshold = top_k + buffer_rank

            closed_this_bar: set[str] = set()

            # 4.1 Kiểm tra thoát vị thế âm sâu/hết hấp dẫn
            for sym in sym_universe:
                if in_position[sym]:
                    sc = scores[sym]
                    if periods_held[sym] >= min_hold_periods and (not np.isfinite(sc) or sc <= exit_funding_threshold):
                        pos_notional = notional[sym]
                        close_fee = pos_notional * one_way_cost
                        equity -= close_fee
                        total_trading_fees += close_fee
                        hourly_trading_fees[t_idx] += close_fee
                        total_notional_traded += pos_notional
                        hourly_notional_traded[t_idx] += pos_notional
                        forced_exits += 1

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "close",
                            "mode": variant,
                            "short_ex": "binance",
                            "long_ex": "spot",
                            "notional": float(pos_notional),
                            "fee": float(close_fee),
                            "funding_cum": float(trade_funding_cum[sym]),
                            "reason": "EXIT_NEGATIVE_YIELD",
                            "score": float(sc) if np.isfinite(sc) else 0.0,
                        })

                        in_position[sym] = False
                        notional[sym] = 0.0
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0
                        ref_perp_price[sym] = 0.0
                        closed_this_bar.add(sym)

            # 4.2 Kiểm tra đảo vị thế theo Patient Buffer (chỉ áp dụng cho Tier 1B)
            if variant == "tier1b":
                earmarked_candidates: set[str] = set()
                for sym in sym_universe:
                    if in_position[sym] and sym not in closed_this_bar:
                        sc = scores[sym]
                        if periods_held[sym] >= min_hold_periods and ranks[sym] > drop_rank_threshold:
                            cand_found = None
                            for cand in ranked_symbols:
                                if (
                                    not in_position[cand]
                                    and cand not in pending_maker_orders
                                    and cand not in earmarked_candidates
                                    and np.isfinite(scores[cand])
                                    and scores[cand] >= entry_funding_threshold
                                ):
                                    cand_found = cand
                                    break

                            if cand_found is not None:
                                earmarked_candidates.add(cand_found)
                                pos_notional = notional[sym]
                                close_fee = pos_notional * one_way_cost
                                equity -= close_fee
                                total_trading_fees += close_fee
                                hourly_trading_fees[t_idx] += close_fee
                                total_notional_traded += pos_notional
                                hourly_notional_traded[t_idx] += pos_notional
                                num_switches += 1

                                trades_list.append({
                                    "symbol": sym,
                                    "time": t,
                                    "action": "close",
                                    "mode": variant,
                                    "short_ex": "binance",
                                    "long_ex": "spot",
                                    "notional": float(pos_notional),
                                    "fee": float(close_fee),
                                    "funding_cum": float(trade_funding_cum[sym]),
                                    "reason": "SWITCH_RANK_DROP",
                                    "score": float(sc) if np.isfinite(sc) else 0.0,
                                })

                                in_position[sym] = False
                                notional[sym] = 0.0
                                periods_held[sym] = 0
                                trade_funding_cum[sym] = 0.0
                                ref_perp_price[sym] = 0.0
                                closed_this_bar.add(sym)

            # 4.3 Mở vị thế mới cho các slot còn trống
            active_symbols = [s for s in sym_universe if in_position[s]]
            n_pending = len(pending_maker_orders)
            n_slots_available = max(0, top_k - len(active_symbols) - n_pending)

            if n_slots_available > 0 and equity > 0:
                open_candidates = [
                    s for s in ranked_symbols
                    if not in_position[s]
                    and s not in pending_maker_orders
                    and np.isfinite(scores[s])
                    and scores[s] >= entry_funding_threshold
                ]

                for sym in open_candidates[:n_slots_available]:
                    allocated = equity / float(top_k)
                    leg_notional = allocated * float(eta)
                    sc = scores[sym]

                    pe_prev = perp_prices[sym][t_idx - 1] if t_idx > 0 else perp_prices[sym][t_idx]
                    if np.isnan(pe_prev) or pe_prev <= 0:
                        pe_prev = perp_prices[sym][t_idx]
                    limit_p = pe_prev * (1.0 + buffer_mult)

                    entry_price = float(limit_p if enable_maker_fill else perp_prices[sym][t_idx])
                    if np.isnan(entry_price) or entry_price <= 0:
                        entry_price = float(perp_prices[sym][t_idx]) if not np.isnan(perp_prices[sym][t_idx]) and perp_prices[sym][t_idx] > 0 else 1.0

                    if lot_quant:
                        q_qty, actual_notional = quantize_order_qty(sym, leg_notional, entry_price)
                        if q_qty <= 0.0 or actual_notional <= 0.0:
                            maker_no_fill_count += 1
                            continue
                        leg_notional = actual_notional

                    if enable_maker_fill:
                        pending_maker_orders[sym] = {
                            "notional": float(leg_notional),
                            "one_way_cost": float(one_way_cost),
                            "side": "sell",
                            "limit_price": float(limit_p),
                            "ttl_remaining": int(order_ttl),
                            "score": float(sc),
                        }
                    else:
                        fee = leg_notional * one_way_cost
                        equity -= fee
                        total_trading_fees += fee
                        hourly_trading_fees[t_idx] += fee
                        total_notional_traded += leg_notional
                        hourly_notional_traded[t_idx] += leg_notional

                        in_position[sym] = True
                        notional[sym] = leg_notional
                        periods_held[sym] = 0
                        trade_funding_cum[sym] = 0.0
                        ref_perp_price[sym] = perp_prices[sym][t_idx]
                        num_entries += 1

                        trades_list.append({
                            "symbol": sym,
                            "time": t,
                            "action": "open",
                            "mode": variant,
                            "short_ex": "binance",
                            "long_ex": "spot",
                            "notional": float(leg_notional),
                            "fee": float(fee),
                            "funding_cum": 0.0,
                            "reason": "OPEN_TIER1_CARRY",
                            "score": float(sc),
                        })

            # Snapshot holdings tại mốc 8h
            act_s = [s for s in sym_universe if in_position[s]]
            act_m = [variant for _ in act_s]
            tot_not = sum(notional[s] for s in act_s)
            holdings_log_list.append({
                "time": t,
                "symbols": act_s,
                "modes": act_m,
                "n_active": len(act_s),
                "total_notional": float(tot_not),
                "equity": float(equity),
            })

        # -------------------------------------------------------------
        # Bước 5: Ghi nhận equity và trạng thái từng nến
        # -------------------------------------------------------------
        equity_vals[t_idx] = equity
        in_market_any[t_idx] = any(in_position.values())

    # 6. Xây dựng các DataFrame và Series kết quả
    equity_curve = pd.Series(equity_vals, index=common_index, dtype="float64", name="equity")
    returns = equity_curve.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(columns=trades_cols)
    for col in trades_cols:
        if col not in trades_df.columns:
            trades_df[col] = np.nan

    holdings_log = pd.DataFrame(holdings_log_list) if holdings_log_list else pd.DataFrame(columns=holdings_cols)

    hourly_fees = hourly_trading_fees + hourly_rebalance_fees
    hourly_net_pnl = hourly_funding_pnl + hourly_basis_pnl - hourly_fees

    attribution_series = pd.DataFrame(
        {
            "funding_pnl": hourly_funding_pnl,
            "basis_pnl": hourly_basis_pnl,
            "rebalance_fees": hourly_rebalance_fees,
            "trading_fees": hourly_trading_fees,
            "fees": hourly_fees,
            "net_pnl": hourly_net_pnl,
            "notional_traded": hourly_notional_traded,
        },
        index=common_index,
        dtype="float64",
    )

    total_hours = len(common_index)
    years = total_hours / float(periods_per_year) if periods_per_year > 0 else 0.0

    if initial_capital > 0:
        total_return_pct = ((equity - initial_capital) / initial_capital) * 100.0
        if years > 0 and equity > 0:
            annualized_return_pct = (
                ((equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0
            )
        else:
            annualized_return_pct = -100.0 if equity <= 0 else 0.0
    else:
        total_return_pct = 0.0
        annualized_return_pct = 0.0

    ret_std = float(returns.std(ddof=1))
    if ret_std > 1e-12 and len(returns) > 1:
        sharpe = float(returns.mean() / ret_std * math.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_vals)
    valid_peak = peak > 0
    if np.any(valid_peak):
        dd = (peak[valid_peak] - equity_vals[valid_peak]) / peak[valid_peak]
        max_dd_pct = float(np.nanmax(dd) * 100.0)
    else:
        max_dd_pct = 0.0

    time_in_market_pct = float(np.mean(in_market_any) * 100.0) if len(in_market_any) > 0 else 0.0
    turnover = (
        float(total_notional_traded / initial_capital)
        if initial_capital > 0
        else 0.0
    )

    total_fees_paid = total_trading_fees + total_rebalance_fees
    total_basis_pnl = float(hourly_basis_pnl.sum())
    basis_mtm_std = float(np.std(hourly_basis_pnl)) if len(hourly_basis_pnl) > 0 else 0.0

    metrics = {
        "Total Return %": total_return_pct,
        "Total Return % ": total_return_pct,
        "Annualized Return %": annualized_return_pct,
        "Sharpe": sharpe,
        "MaxDD %": max_dd_pct,
        "Total Funding Collected": float(total_funding_collected),
        "Total Fees Paid": float(total_fees_paid),
        "Price Rebalances": int(price_rebalances),
        "Margin Rebalances": int(margin_rebalances),
        "Margin Calls": 0,
        "Maker No-Fill Count": int(maker_no_fill_count),
        "Time in Market %": time_in_market_pct,
        "Turnover": turnover,
        "Num Entries": int(num_entries),
        "Num Switches": int(num_switches),
        "Num Rebalances": int(price_rebalances),
        "Total Basis PnL": total_basis_pnl,
        "Total MTM PnL": total_basis_pnl,
        "Net PnL": float(equity - initial_capital),
        "Forced Exits": int(forced_exits),
    }

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades_df": trades_df,
        "holdings_log": holdings_log,
        "attribution_series": attribution_series,
        "funding_pnl": float(total_funding_collected),
        "basis_pnl": total_basis_pnl,
        "basis_mtm_std": basis_mtm_std,
        "fees": float(total_fees_paid),
        "rebalance_fees": float(total_rebalance_fees),
        "turnover": turnover,
        "variant": variant,
        "symbol_universe": sym_universe,
        "metrics": metrics,
    }
