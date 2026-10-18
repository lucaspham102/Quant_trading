"""Module chẩn đoán (Diagnostics Engine): bóc tách alpha vs beta và phân tích chi phí giao dịch.

Cung cấp các công cụ kiểm định:
1. Buy-and-hold theo từng cửa sổ test để xác định edge thô là alpha hay beta.
2. Chi phí & turnover của baseline cố định (real vs zero fees).
3. Thống kê mean Sharpe zero, phân phối return và điểm hòa vốn chi phí (breakeven fee).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src import features
from src.backtester import Backtester
from src.config import TradingConfig, candles_per_year_for_timeframe
from src.research import load_or_fetch_symbol_data
from src.walkforward import (
    Candidate,
    _slice_window,
    apply_fee_scenario,
    generate_walkforward_windows,
    make_candidate_config,
)

# Hằng số cấu hình phí thực tế và baseline mặc định
REAL_TAKER_FEE: float = 0.0005
REAL_MAKER_FEE: float = 0.0002
REAL_SLIPPAGE: float = 0.0002
REAL_FEE_PER_SIDE_BH: float = REAL_TAKER_FEE + REAL_SLIPPAGE  # 0.0007 (7.0 bps)
DEFAULT_BASELINE: Candidate = Candidate("breakout", "market", True)

SUMMARY_COLUMNS: list[str] = [
    "symbol",
    "n_windows",
    "bh_mean_return_pct",
    "bh_mean_sharpe",
    "bh_pct_positive",
    "base_zero_mean_return_pct",
    "base_zero_mean_sharpe",
    "base_zero_pct_positive",
    "base_real_mean_return_pct",
    "base_real_mean_sharpe",
    "edge_zero_minus_bh_mean_pct",
    "base_zero_min_return_pct",
    "base_zero_median_return_pct",
    "base_zero_max_return_pct",
    "base_real_min_return_pct",
    "base_real_median_return_pct",
    "base_real_max_return_pct",
    "mean_trades",
    "mean_fees_usdt",
    "mean_fees_pct_equity",
    "mean_fee_per_trade_usdt",
    "mean_gross_return_pct",
    "breakeven_fee_per_side_bps",
]


def buy_and_hold_return_pct(
    open_prices: pd.Series, close_prices: pd.Series, fee_per_side: float
) -> float:
    """Tính tỷ suất sinh lời (%) Buy-and-Hold trong cửa sổ test sau khi trừ phí 2 chiều."""
    if len(open_prices) == 0 or len(close_prices) == 0:
        return 0.0
    entry = float(open_prices.iloc[0])
    exit_ = float(close_prices.iloc[-1])
    if entry <= 0.0:
        return 0.0
    return float(((exit_ / entry - 1.0) - 2.0 * fee_per_side) * 100.0)


def buy_and_hold_sharpe(close_series: pd.Series, candles_per_year: int) -> float:
    """Tính tỷ số Sharpe chuẩn hóa theo năm cho chiến lược Buy-and-Hold."""
    r = close_series.pct_change().dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if not math.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(r.mean() / std * math.sqrt(candles_per_year))


@dataclass(frozen=True)
class DiagWindowRow:
    """Kết quả chẩn đoán chi tiết cho từng cửa sổ Walk-Forward."""

    symbol: str
    window_index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    bh_return_pct: float
    bh_sharpe: float
    base_zero_return_pct: float
    base_zero_sharpe: float
    base_real_return_pct: float
    base_real_sharpe: float
    base_real_trades: int
    base_real_fees_usdt: float
    base_real_fees_pct_equity: float
    base_real_fee_per_trade_usdt: float
    base_real_gross_return_pct: float
    edge_zero_minus_bh_pct: float


def run_diagnostics(
    symbol: str,
    timeframe: str = "1h",
    days: int = 1260,
    train_days: int = 120,
    test_days: int = 60,
    step_days: int = 60,
    buffer_days: int = 15,
    baseline: Optional[Candidate] = None,
    raw_data_dir: Path = Path("data/raw"),
    df: Optional[pd.DataFrame] = None,
    exchange=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy phân tích chẩn đoán alpha/beta và chi phí theo từng cửa sổ Walk-Forward."""
    baseline_cand = baseline if baseline is not None else DEFAULT_BASELINE

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
    feat = features.add_all_features(df, base_common, drop_warmup=True)
    windows = generate_walkforward_windows(feat.index, train_days, test_days, step_days)

    candles_py = candles_per_year_for_timeframe(timeframe)
    base_zero = apply_fee_scenario(base_common, "zero")
    cfg_zero = make_candidate_config(base_zero, baseline_cand, timeframe)
    base_real = apply_fee_scenario(base_common, "real")
    cfg_real = make_candidate_config(base_real, baseline_cand, timeframe)

    initial_capital = float(base_common.initial_capital)
    rows: list[DiagWindowRow] = []

    for w in windows:
        # Chuẩn hóa timezone để cắt chuỗi nến test
        start = w.test_start
        end = w.test_end
        if feat.index.tz is not None and start.tzinfo is None:
            start = start.tz_localize(feat.index.tz)
        if feat.index.tz is not None and end.tzinfo is None:
            end = end.tz_localize(feat.index.tz)
        if feat.index.tz is None and start.tzinfo is not None:
            start = start.tz_localize(None)
        if feat.index.tz is None and end.tzinfo is not None:
            end = end.tz_localize(None)

        test_plain = feat.loc[(feat.index >= start) & (feat.index < end)]
        if len(test_plain) == 0:
            continue

        bh_ret = buy_and_hold_return_pct(
            test_plain["open"], test_plain["close"], REAL_FEE_PER_SIDE_BH
        )
        bh_sh = buy_and_hold_sharpe(test_plain["close"], candles_py)

        frame_ctx = _slice_window(feat, w.test_start, w.test_end)
        rep_zero = Backtester(cfg_zero).run(frame_ctx)
        rep_real = Backtester(cfg_real).run(frame_ctx)

        fees_usdt = float(rep_real.total_fees_paid)
        fees_pct_equity = (
            float(fees_usdt / initial_capital * 100.0) if initial_capital > 0 else 0.0
        )
        trades = int(rep_real.total_trades)
        fee_per_trade_usdt = float(fees_usdt / trades) if trades > 0 else 0.0
        gross_return_pct = float(rep_real.total_return_pct + fees_pct_equity)
        edge_zero_minus_bh_pct = float(rep_zero.total_return_pct - bh_ret)

        rows.append(
            DiagWindowRow(
                symbol=symbol,
                window_index=w.window_index,
                test_start=w.test_start,
                test_end=w.test_end,
                bh_return_pct=bh_ret,
                bh_sharpe=bh_sh,
                base_zero_return_pct=float(rep_zero.total_return_pct),
                base_zero_sharpe=float(rep_zero.sharpe_ratio),
                base_real_return_pct=float(rep_real.total_return_pct),
                base_real_sharpe=float(rep_real.sharpe_ratio),
                base_real_trades=trades,
                base_real_fees_usdt=fees_usdt,
                base_real_fees_pct_equity=fees_pct_equity,
                base_real_fee_per_trade_usdt=fee_per_trade_usdt,
                base_real_gross_return_pct=gross_return_pct,
                edge_zero_minus_bh_pct=edge_zero_minus_bh_pct,
            )
        )

    window_cols = [f.name for f in dataclasses.fields(DiagWindowRow)]
    if rows:
        per_window_df = pd.DataFrame([dataclasses.asdict(r) for r in rows], columns=window_cols)
        mean_gross_return_pct = float(per_window_df["base_real_gross_return_pct"].mean())
        mean_fees_pct_equity = float(per_window_df["base_real_fees_pct_equity"].mean())
        # total_fees_paid CHỈ gồm taker fee (slippage đã nằm trong giá khớp, thuộc gross),
        # nên cơ sở để ngoại suy là mức taker (5.0 bps/lượt một chiều), không phải 7.0.
        c = REAL_TAKER_FEE * 10_000.0  # 5.0 bps
        breakeven_bps = (
            float(c * mean_gross_return_pct / mean_fees_pct_equity)
            if mean_fees_pct_equity > 0
            else 0.0
        )

        summary_data = {
            "symbol": symbol,
            "n_windows": len(rows),
            "bh_mean_return_pct": float(per_window_df["bh_return_pct"].mean()),
            "bh_mean_sharpe": float(per_window_df["bh_sharpe"].mean()),
            "bh_pct_positive": float((per_window_df["bh_return_pct"] > 0).mean() * 100.0),
            "base_zero_mean_return_pct": float(per_window_df["base_zero_return_pct"].mean()),
            "base_zero_mean_sharpe": float(per_window_df["base_zero_sharpe"].mean()),
            "base_zero_pct_positive": float(
                (per_window_df["base_zero_return_pct"] > 0).mean() * 100.0
            ),
            "base_real_mean_return_pct": float(per_window_df["base_real_return_pct"].mean()),
            "base_real_mean_sharpe": float(per_window_df["base_real_sharpe"].mean()),
            "edge_zero_minus_bh_mean_pct": float(per_window_df["edge_zero_minus_bh_pct"].mean()),
            "base_zero_min_return_pct": float(per_window_df["base_zero_return_pct"].min()),
            "base_zero_median_return_pct": float(per_window_df["base_zero_return_pct"].median()),
            "base_zero_max_return_pct": float(per_window_df["base_zero_return_pct"].max()),
            "base_real_min_return_pct": float(per_window_df["base_real_return_pct"].min()),
            "base_real_median_return_pct": float(per_window_df["base_real_return_pct"].median()),
            "base_real_max_return_pct": float(per_window_df["base_real_return_pct"].max()),
            "mean_trades": float(per_window_df["base_real_trades"].mean()),
            "mean_fees_usdt": float(per_window_df["base_real_fees_usdt"].mean()),
            "mean_fees_pct_equity": mean_fees_pct_equity,
            "mean_fee_per_trade_usdt": float(per_window_df["base_real_fee_per_trade_usdt"].mean()),
            "mean_gross_return_pct": mean_gross_return_pct,
            "breakeven_fee_per_side_bps": breakeven_bps,
        }
    else:
        per_window_df = pd.DataFrame(columns=window_cols)
        summary_data = {
            "symbol": symbol,
            "n_windows": 0,
            "bh_mean_return_pct": 0.0,
            "bh_mean_sharpe": 0.0,
            "bh_pct_positive": 0.0,
            "base_zero_mean_return_pct": 0.0,
            "base_zero_mean_sharpe": 0.0,
            "base_zero_pct_positive": 0.0,
            "base_real_mean_return_pct": 0.0,
            "base_real_mean_sharpe": 0.0,
            "edge_zero_minus_bh_mean_pct": 0.0,
            "base_zero_min_return_pct": 0.0,
            "base_zero_median_return_pct": 0.0,
            "base_zero_max_return_pct": 0.0,
            "base_real_min_return_pct": 0.0,
            "base_real_median_return_pct": 0.0,
            "base_real_max_return_pct": 0.0,
            "mean_trades": 0.0,
            "mean_fees_usdt": 0.0,
            "mean_fees_pct_equity": 0.0,
            "mean_fee_per_trade_usdt": 0.0,
            "mean_gross_return_pct": 0.0,
            "breakeven_fee_per_side_bps": 0.0,
        }

    summary_df = pd.DataFrame([summary_data], columns=SUMMARY_COLUMNS)
    return per_window_df, summary_df


__all__ = [
    "REAL_TAKER_FEE",
    "REAL_MAKER_FEE",
    "REAL_SLIPPAGE",
    "REAL_FEE_PER_SIDE_BH",
    "DEFAULT_BASELINE",
    "SUMMARY_COLUMNS",
    "DiagWindowRow",
    "buy_and_hold_return_pct",
    "buy_and_hold_sharpe",
    "run_diagnostics",
]
