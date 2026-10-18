"""Engine Walk-Forward Validation: kiểm định out-of-sample việc lựa chọn cấu hình chiến lược.

Không look-ahead bias, phân tách nghiêm ngặt giữa cửa sổ train và test.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src import features
from src.backtester import Backtester, BacktestReport
from src.config import TradingConfig, candles_per_year_for_timeframe
from src.research import load_or_fetch_symbol_data


@dataclass(frozen=True)
class WalkForwardWindow:
    """Cửa sổ Walk-Forward bao gồm khoảng thời gian train và test."""

    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True, order=True)
class Candidate:
    """Ứng viên cấu hình chiến lược."""

    strategy_mode: str
    entry_mode: str
    use_trailing_stop: bool


DEFAULT_CANDIDATES: tuple[Candidate, ...] = tuple(
    Candidate(mode, entry, trailing)
    for mode in ("breakout", "trend_pullback")
    for entry in ("market", "limit_pullback")
    for trailing in (False, True)
)


@dataclass(frozen=True)
class WindowResult:
    """Kết quả chi tiết cho một cửa sổ Walk-Forward."""

    symbol: str
    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_mode: str
    chosen_entry_mode: str
    chosen_trailing: bool
    train_sharpe: float
    train_total_trades: int
    test_return_pct: float
    test_max_drawdown_pct: float
    test_sharpe_ratio: float
    test_total_trades: int
    baseline_return_pct: float
    baseline_max_drawdown_pct: float
    baseline_sharpe_ratio: float
    baseline_total_trades: int


def generate_walkforward_windows(
    index: pd.DatetimeIndex,
    train_days: int = 120,
    test_days: int = 60,
    step_days: int = 60,
) -> list[WalkForwardWindow]:
    """Tạo danh sách các cửa sổ trượt (train + test) theo thời gian.

    Trả về danh sách rỗng nếu dữ liệu không đủ cho một cửa sổ hoàn chỉnh.
    """
    if len(index) == 0 or train_days <= 0 or test_days <= 0 or step_days <= 0:
        return []

    start = pd.Timestamp(index[0])
    end = pd.Timestamp(index[-1])
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)
    train_delta = pd.Timedelta(days=train_days)

    windows: list[WalkForwardWindow] = []
    window_index = 0
    test_start = start + train_delta

    while test_start + test_delta <= end:
        windows.append(
            WalkForwardWindow(
                window_index=window_index,
                train_start=test_start - train_delta,
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + test_delta,
            )
        )
        test_start += step_delta
        window_index += 1

    return windows


def make_candidate_config(
    base: TradingConfig, cand: Candidate, timeframe: str
) -> TradingConfig:
    """Tạo cấu hình TradingConfig cho ứng viên với candles_per_year phù hợp timeframe."""
    return dataclasses.replace(
        base,
        strategy_mode=cand.strategy_mode,
        entry_mode=cand.entry_mode,
        use_trailing_stop=cand.use_trailing_stop,
        use_ml=False,
        candles_per_year=candles_per_year_for_timeframe(timeframe),
    )


# Bộ (taker_fee, maker_fee, slippage) theo từng kịch bản phí.
FEE_SCENARIOS: dict[str, tuple[float, float, float]] = {
    "zero": (0.0, 0.0, 0.0),
    "real": (0.0005, 0.0002, 0.0002),
    "double": (0.001, 0.0004, 0.0004),
}


def apply_fee_scenario(base: TradingConfig, fee_scenario: str) -> TradingConfig:
    """Trả bản sao config đã áp phí/slippage của kịch bản `zero` / `real` / `double`."""
    if fee_scenario not in FEE_SCENARIOS:
        raise ValueError(
            f"fee_scenario không hợp lệ: {fee_scenario!r}; chọn trong {sorted(FEE_SCENARIOS)}"
        )
    taker, maker, slip = FEE_SCENARIOS[fee_scenario]
    return dataclasses.replace(base, taker_fee=taker, maker_fee=maker, slippage=slip)


def _slice_window(feat: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Cắt cửa sổ nửa mở [start, end) kèm đúng 1 nến ngay trước start để khớp lệnh biên."""
    if len(feat) == 0:
        return feat.copy()

    # Chuẩn hóa timezone nếu cần
    if feat.index.tz is not None and start.tzinfo is None:
        start = start.tz_localize(feat.index.tz)
    if feat.index.tz is not None and end.tzinfo is None:
        end = end.tz_localize(feat.index.tz)
    if feat.index.tz is None and start.tzinfo is not None:
        start = start.tz_localize(None)
    if feat.index.tz is None and end.tzinfo is not None:
        end = end.tz_localize(None)

    i0 = int(feat.index.searchsorted(start))
    i1 = int(feat.index.searchsorted(end))
    frame = feat.iloc[max(0, i0 - 1) : i1].copy()
    if hasattr(feat, "attrs"):
        frame.attrs = feat.attrs.copy()
    return frame


def select_candidate(
    train_reports: dict[Candidate, BacktestReport],
    min_trades: int = 15,
    max_drawdown_pct: float = 8.0,
) -> Optional[Candidate]:
    """Chọn ứng viên tối ưu trên tập train: Sharpe lớn nhất, tie-break qua Max Drawdown và thứ tự Candidate."""
    qualified = [
        cand
        for cand, rep in train_reports.items()
        if rep.total_trades >= min_trades and rep.max_drawdown_pct <= max_drawdown_pct
    ]
    if not qualified:
        return None

    def _sort_key(cand: Candidate):
        rep = train_reports[cand]
        sharpe = float(rep.sharpe_ratio) if math.isfinite(rep.sharpe_ratio) else -1e9
        dd = float(rep.max_drawdown_pct) if math.isfinite(rep.max_drawdown_pct) else 1e9
        return (
            sharpe,
            -dd,
            cand.strategy_mode,
            cand.entry_mode,
            cand.use_trailing_stop,
        )

    return max(qualified, key=_sort_key)


def _format_chosen_mode(r: WindowResult) -> str:
    if r.chosen_mode == "stand_aside":
        return "stand_aside"
    trail_tag = "trail" if r.chosen_trailing else "no_trail"
    return f"{r.chosen_mode}/{r.chosen_entry_mode}/{trail_tag}"


def run_walkforward(
    symbol: str,
    timeframe: str = "1h",
    days: int = 365,
    train_days: int = 120,
    test_days: int = 60,
    step_days: int = 60,
    candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
    base_config: Optional[TradingConfig] = None,
    min_trades: int = 15,
    max_drawdown_pct: float = 8.0,
    buffer_days: int = 15,
    raw_data_dir: Path = Path("data/raw"),
    df: Optional[pd.DataFrame] = None,
    exchange=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thực thi Walk-Forward Validation cho một symbol và trả về (per_window_df, summary_df)."""
    if base_config is not None:
        base = dataclasses.replace(base_config, symbol=symbol, timeframe=timeframe)
    else:
        base = dataclasses.replace(
            TradingConfig(),
            use_fee_aware_sizing=True,
            min_stop_distance_pct=0.008,
            symbol=symbol,
            timeframe=timeframe,
        )

    if df is None:
        df = load_or_fetch_symbol_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days + buffer_days,
            raw_data_dir=raw_data_dir,
            exchange=exchange,
        )

    feat = features.add_all_features(df, base, drop_warmup=True)
    windows = generate_walkforward_windows(
        feat.index, train_days=train_days, test_days=test_days, step_days=step_days
    )

    baseline_cand = Candidate("breakout", "market", True)
    results: list[WindowResult] = []

    for w in windows:
        train_frame = _slice_window(feat, w.train_start, w.train_end)
        train_reports: dict[Candidate, BacktestReport] = {}
        for cand in candidates:
            cfg = make_candidate_config(base, cand, timeframe)
            rep = Backtester(cfg).run(train_frame)
            train_reports[cand] = rep

        chosen = select_candidate(train_reports, min_trades=min_trades, max_drawdown_pct=max_drawdown_pct)
        test_frame = _slice_window(feat, w.test_start, w.test_end)

        base_cfg = make_candidate_config(base, baseline_cand, timeframe)
        rep_base = Backtester(base_cfg).run(test_frame)

        if chosen is None:
            chosen_mode = "stand_aside"
            chosen_entry_mode = ""
            chosen_trailing = False
            train_sharpe = 0.0
            train_total_trades = 0
            test_return_pct = 0.0
            test_max_drawdown_pct = 0.0
            test_sharpe_ratio = 0.0
            test_total_trades = 0
        else:
            chosen_mode = chosen.strategy_mode
            chosen_entry_mode = chosen.entry_mode
            chosen_trailing = chosen.use_trailing_stop
            train_rep = train_reports[chosen]
            train_sharpe = float(train_rep.sharpe_ratio)
            train_total_trades = int(train_rep.total_trades)

            chosen_cfg = make_candidate_config(base, chosen, timeframe)
            rep_chosen = Backtester(chosen_cfg).run(test_frame)
            test_return_pct = float(rep_chosen.total_return_pct)
            test_max_drawdown_pct = float(rep_chosen.max_drawdown_pct)
            test_sharpe_ratio = float(rep_chosen.sharpe_ratio)
            test_total_trades = int(rep_chosen.total_trades)

        res = WindowResult(
            symbol=symbol,
            window_index=w.window_index,
            train_start=w.train_start,
            train_end=w.train_end,
            test_start=w.test_start,
            test_end=w.test_end,
            chosen_mode=chosen_mode,
            chosen_entry_mode=chosen_entry_mode,
            chosen_trailing=chosen_trailing,
            train_sharpe=train_sharpe,
            train_total_trades=train_total_trades,
            test_return_pct=test_return_pct,
            test_max_drawdown_pct=test_max_drawdown_pct,
            test_sharpe_ratio=test_sharpe_ratio,
            test_total_trades=test_total_trades,
            baseline_return_pct=float(rep_base.total_return_pct),
            baseline_max_drawdown_pct=float(rep_base.max_drawdown_pct),
            baseline_sharpe_ratio=float(rep_base.sharpe_ratio),
            baseline_total_trades=int(rep_base.total_trades),
        )
        results.append(res)

    window_cols = [f.name for f in dataclasses.fields(WindowResult)]
    if results:
        per_window_df = pd.DataFrame([dataclasses.asdict(r) for r in results], columns=window_cols)
    else:
        per_window_df = pd.DataFrame(columns=window_cols)

    if results:
        n_windows = len(results)
        pct_positive_windows = float((per_window_df["test_return_pct"] > 0).mean() * 100.0)
        mean_test_return_pct = float(per_window_df["test_return_pct"].mean())
        mean_test_sharpe = float(per_window_df["test_sharpe_ratio"].mean())
        mean_test_max_drawdown_pct = float(per_window_df["test_max_drawdown_pct"].mean())
        mean_baseline_return_pct = float(per_window_df["baseline_return_pct"].mean())
        mean_baseline_sharpe = float(per_window_df["baseline_sharpe_ratio"].mean())
        mean_baseline_max_drawdown_pct = float(per_window_df["baseline_max_drawdown_pct"].mean())

        counts: dict[str, int] = {}
        for r in results:
            key = _format_chosen_mode(r)
            counts[key] = counts.get(key, 0) + 1
        chosen_mode_counts = "; ".join(f"{k}:{v}" for k, v in counts.items())
    else:
        n_windows = 0
        pct_positive_windows = 0.0
        mean_test_return_pct = 0.0
        mean_test_sharpe = 0.0
        mean_test_max_drawdown_pct = 0.0
        mean_baseline_return_pct = 0.0
        mean_baseline_sharpe = 0.0
        mean_baseline_max_drawdown_pct = 0.0
        chosen_mode_counts = ""

    summary_columns = [
        "symbol",
        "n_windows",
        "pct_positive_windows",
        "mean_test_return_pct",
        "mean_test_sharpe",
        "mean_test_max_drawdown_pct",
        "mean_baseline_return_pct",
        "mean_baseline_sharpe",
        "mean_baseline_max_drawdown_pct",
        "chosen_mode_counts",
    ]
    summary_df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "n_windows": n_windows,
                "pct_positive_windows": pct_positive_windows,
                "mean_test_return_pct": mean_test_return_pct,
                "mean_test_sharpe": mean_test_sharpe,
                "mean_test_max_drawdown_pct": mean_test_max_drawdown_pct,
                "mean_baseline_return_pct": mean_baseline_return_pct,
                "mean_baseline_sharpe": mean_baseline_sharpe,
                "mean_baseline_max_drawdown_pct": mean_baseline_max_drawdown_pct,
                "chosen_mode_counts": chosen_mode_counts,
            }
        ],
        columns=summary_columns,
    )

    return per_window_df, summary_df


def run_null_test(
    symbol: str,
    timeframe: str = "1h",
    days: int = 365,
    train_days: int = 120,
    test_days: int = 60,
    step_days: int = 60,
    n_iter: int = 100,
    seed: int = 42,
    candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
    base_config: Optional[TradingConfig] = None,
    fee_scenario: str = "real",
    min_trades: int = 15,
    max_drawdown_pct: float = 8.0,
    buffer_days: int = 15,
    raw_data_dir: Path = Path("data/raw"),
    df: Optional[pd.DataFrame] = None,
    exchange=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm định Null Hypothesis Test cho Walk-Forward Engine.

    Với mỗi cửa sổ trượt, so sánh lợi nhuận out-of-sample của cấu hình được tối ưu
    trên tập train với phân phối null gồm n_iter lần chọn cấu hình ngẫu nhiên.
    """
    if base_config is not None:
        base = dataclasses.replace(base_config, symbol=symbol, timeframe=timeframe)
    else:
        base = dataclasses.replace(
            TradingConfig(),
            use_fee_aware_sizing=True,
            min_stop_distance_pct=0.008,
            symbol=symbol,
            timeframe=timeframe,
        )
    base = apply_fee_scenario(base, fee_scenario)

    if df is None:
        df = load_or_fetch_symbol_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days + buffer_days,
            raw_data_dir=raw_data_dir,
            exchange=exchange,
        )

    feat = features.add_all_features(df, base, drop_warmup=True)
    windows = generate_walkforward_windows(
        feat.index, train_days=train_days, test_days=test_days, step_days=step_days
    )

    rows: list[dict] = []

    for w in windows:
        train_frame = _slice_window(feat, w.train_start, w.train_end)
        train_reports = {
            cand: Backtester(make_candidate_config(base, cand, timeframe)).run(train_frame)
            for cand in candidates
        }
        chosen = select_candidate(
            train_reports, min_trades=min_trades, max_drawdown_pct=max_drawdown_pct
        )

        test_frame = _slice_window(feat, w.test_start, w.test_end)

        cand_test_returns = {
            cand: float(
                Backtester(make_candidate_config(base, cand, timeframe)).run(test_frame).total_return_pct
            )
            for cand in candidates
        }

        if chosen is None:
            chosen_return = 0.0
            chosen_label = "stand_aside"
        else:
            chosen_return = cand_test_returns.get(
                chosen,
                float(
                    Backtester(make_candidate_config(base, chosen, timeframe))
                    .run(test_frame)
                    .total_return_pct
                ),
            )
            trail_tag = "trail" if chosen.use_trailing_stop else "no_trail"
            chosen_label = f"{chosen.strategy_mode}/{chosen.entry_mode}/{trail_tag}"

        rng = np.random.default_rng(seed + w.window_index)
        n_cands = len(candidates)
        null = [
            cand_test_returns[candidates[int(rng.integers(0, n_cands))]]
            for _ in range(n_iter)
        ]

        arr = np.asarray(null, dtype=float)
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        mean = float(arr.mean())
        mx = float(arr.max())
        mn = float(arr.min())
        percentile = float(np.mean(arr <= chosen_return) * 100.0)
        beats_p75 = bool(chosen_return > p75)

        rows.append(
            {
                "symbol": symbol,
                "window_index": w.window_index,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "chosen_label": chosen_label,
                "chosen_test_return_pct": chosen_return,
                "null_mean_return_pct": mean,
                "null_p25_return_pct": p25,
                "null_p75_return_pct": p75,
                "null_max_return_pct": mx,
                "chosen_percentile": percentile,
                "beats_p75": beats_p75,
            }
        )

    per_window_cols = [
        "symbol",
        "window_index",
        "test_start",
        "test_end",
        "chosen_label",
        "chosen_test_return_pct",
        "null_mean_return_pct",
        "null_p25_return_pct",
        "null_p75_return_pct",
        "null_max_return_pct",
        "chosen_percentile",
        "beats_p75",
    ]
    summary_cols = [
        "symbol",
        "n_windows",
        "pct_windows_beating_p75",
        "mean_chosen_return_pct",
        "mean_null_p75_return_pct",
        "mean_null_mean_return_pct",
    ]

    if rows:
        per_window_df = pd.DataFrame(rows, columns=per_window_cols)
        summary_df = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "n_windows": len(rows),
                    "pct_windows_beating_p75": float(per_window_df["beats_p75"].mean() * 100.0),
                    "mean_chosen_return_pct": float(per_window_df["chosen_test_return_pct"].mean()),
                    "mean_null_p75_return_pct": float(per_window_df["null_p75_return_pct"].mean()),
                    "mean_null_mean_return_pct": float(per_window_df["null_mean_return_pct"].mean()),
                }
            ],
            columns=summary_cols,
        )
    else:
        per_window_df = pd.DataFrame(columns=per_window_cols)
        summary_df = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "n_windows": 0,
                    "pct_windows_beating_p75": 0.0,
                    "mean_chosen_return_pct": 0.0,
                    "mean_null_p75_return_pct": 0.0,
                    "mean_null_mean_return_pct": 0.0,
                }
            ],
            columns=summary_cols,
        )

    return per_window_df, summary_df


__all__ = [
    "Candidate",
    "DEFAULT_CANDIDATES",
    "FEE_SCENARIOS",
    "WalkForwardWindow",
    "WindowResult",
    "apply_fee_scenario",
    "generate_walkforward_windows",
    "make_candidate_config",
    "run_null_test",
    "run_walkforward",
    "select_candidate",
]
