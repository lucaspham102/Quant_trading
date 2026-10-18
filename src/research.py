"""Research Engine: chạy ma trận tổ hợp chiến lược đa tài sản / đa khung thời gian.

Thiết kế theo `.agy/architect_417d6c57.md` (mục 2.6):
- Tải/cache dữ liệu OHLCV (Parquet) cho nhiều symbol/timeframe.
- Chạy Backtester trên ma trận: symbol x timeframe x strategy_mode x entry_mode
  x {ML on/off} x {trailing on/off} x {fee-aware}.
- Xếp hạng theo Sharpe / Profit Factor / Total Return, xuất CSV + Markdown.
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src import features
from src.backtester import Backtester
from src.config import TradingConfig, candles_per_year_for_timeframe
from src.data_loader import DataLoader, ExchangeProtocol, merge_and_deduplicate_ohlcv

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
)
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1h", "15m")
DEFAULT_STRATEGY_MODES: tuple[str, ...] = (
    "breakout",
    "trend_pullback",
    "mean_reversion",
    "regime_adaptive",
)
DEFAULT_ENTRY_MODES: tuple[str, ...] = ("market", "limit_pullback")

RESEARCH_COLUMNS: list[str] = [
    "symbol",
    "timeframe",
    "strategy_mode",
    "entry_mode",
    "use_ml",
    "use_trailing_stop",
    "use_fee_aware_sizing",
    "min_stop_distance_pct",
    "total_return_pct",
    "final_equity",
    "max_drawdown_pct",
    "sharpe_ratio",
    "win_rate_pct",
    "profit_factor",
    "total_trades",
    "total_fees_paid",
]

_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


@dataclass(frozen=True)
class ResearchRow:
    symbol: str
    timeframe: str
    strategy_mode: str
    entry_mode: str
    use_ml: bool
    use_trailing_stop: bool
    use_fee_aware_sizing: bool
    min_stop_distance_pct: float
    total_return_pct: float
    final_equity: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    total_fees_paid: float


def generate_synthetic_ohlcv(
    n_candles: int = 2000,
    start_price: float = 60000.0,
    seed: int = 42,
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
) -> pd.DataFrame:
    """Sinh OHLCV giả lập (GBM + pha xu hướng + đột biến volume) để chạy offline/CI."""
    rng = np.random.default_rng(seed)
    minutes = _TF_MINUTES.get(timeframe, 15)
    phase_len = 250
    n_phases = n_candles // phase_len + 1
    phase_drift = rng.choice([-1.0, 1.0, 1.0, 0.0], size=n_phases) * 0.0008
    drift = np.repeat(phase_drift, phase_len)[:n_candles]
    sigma = 0.002
    shocks = rng.normal(0.0, sigma, size=n_candles)
    spike = rng.random(n_candles) < 0.05
    shocks = np.where(spike, shocks * 3.0 + np.sign(drift) * 0.003, shocks)
    log_ret = drift + shocks - 0.5 * sigma**2
    close = start_price * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick_up = np.abs(rng.normal(0.0, sigma * 0.5, size=n_candles))
    wick_dn = np.abs(rng.normal(0.0, sigma * 0.5, size=n_candles))
    high = np.maximum(open_, close) * (1.0 + wick_up)
    low = np.minimum(open_, close) * (1.0 - wick_dn)
    volume = rng.lognormal(mean=4.0, sigma=0.3, size=n_candles)
    volume = np.where(spike, volume * rng.uniform(3.0, 6.0, size=n_candles), volume)
    index = pd.date_range(
        "2024-01-01", periods=n_candles, freq=f"{minutes}min", tz="UTC", name="timestamp"
    ).as_unit("ns")
    df = pd.DataFrame(
        {
            "open": open_.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
        },
        index=index,
    )
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


def load_or_fetch_symbol_data(
    symbol: str,
    timeframe: str,
    days: int = 180,
    raw_data_dir: Path = Path("data/raw"),
    exchange: Optional[ExchangeProtocol] = None,
    use_synthetic: bool = False,
    synthetic_n: int = 2000,
) -> pd.DataFrame:
    """Đọc cache Parquet và BACKFILL phần đầu còn thiếu để phủ đủ `days` ngày gần nhất.

    - Nếu cache chưa phủ tới mốc `now - days` (vượt dung sai 1 nến), tải bù phần thiếu ở phía ĐẦU.
    - Nếu cache đã phủ đủ cả đầu lẫn đuôi, trả cache luôn (không gọi mạng).
    - Lỗi tải dữ liệu -> raise RuntimeError rõ ràng, KHÔNG fallback synthetic.
    """
    if use_synthetic:
        return generate_synthetic_ohlcv(n_candles=synthetic_n, symbol=symbol, timeframe=timeframe)
    cfg = dataclasses.replace(
        TradingConfig(), symbol=symbol, timeframe=timeframe, raw_data_dir=Path(raw_data_dir)
    )
    loader = DataLoader(cfg, exchange=exchange)
    tf_ms = loader.timeframe_ms
    now_ms = int(exchange.milliseconds()) if exchange is not None else int(time.time() * 1000)
    since_ms = now_ms - days * 86400 * 1000
    tol_ms = tf_ms

    path = loader.default_file_path()
    if not path.exists():
        return loader.update_incremental(default_start_ms=since_ms)

    cached = loader.load_parquet()
    if len(cached) == 0:
        return loader.update_incremental(default_start_ms=since_ms)

    first_ms = int(cached.index[0].timestamp() * 1000)
    last_ms = int(cached.index[-1].timestamp() * 1000)
    head_ok = first_ms <= since_ms + tol_ms
    tail_ok = last_ms >= now_ms - 2 * tf_ms
    if head_ok and tail_ok:
        return cached  # cache đã phủ đủ khoảng yêu cầu -> không cần fetch

    merged = cached
    if not tail_ok:
        merged = loader.update_incremental(default_start_ms=since_ms)

    first_ms = int(merged.index[0].timestamp() * 1000)
    if first_ms > since_ms + tol_ms:
        head = loader.fetch_historical_ohlcv(since_ms, until_ms=first_ms - 1000)
        if len(head) == 0:
            since_ts = pd.Timestamp(since_ms, unit="ms", tz="UTC")
            raise RuntimeError(
                f"Không backfill được dữ liệu {symbol} {timeframe}: cache bắt đầu tại "
                f"{merged.index[0]} nhưng yêu cầu {days} ngày (từ {since_ts}). "
                f"Kiểm tra kết nối mạng / phạm vi dữ liệu của sàn."
            )
        merged = merge_and_deduplicate_ohlcv(merged, head)
        merged.attrs["symbol"] = symbol
        merged.attrs["timeframe"] = timeframe
        loader.save_parquet(merged, path)
    return merged


def run_research_matrix(
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    strategy_modes: Sequence[str] = DEFAULT_STRATEGY_MODES,
    ml_options: Sequence[bool] = (False, True),
    trailing_options: Sequence[bool] = (False, True),
    fee_aware_options: Sequence[bool] = (True,),
    entry_modes: Sequence[str] = DEFAULT_ENTRY_MODES,
    days: int = 180,
    base_config: TradingConfig = TradingConfig(),
    data_map: Optional[dict[tuple[str, str], pd.DataFrame]] = None,
    use_synthetic: bool = False,
    raw_data_dir: Path = Path("data/raw"),
    exchange: Optional[ExchangeProtocol] = None,
    synthetic_n: int = 2000,
) -> pd.DataFrame:
    """Chạy ma trận tổ hợp backtest, trả về DataFrame xếp hạng giảm dần theo hiệu năng.

    `data_map` (khóa `(symbol, timeframe)`) cho phép test offline 100% không cần mạng.
    """
    rows: list[ResearchRow] = []
    provided = data_map or {}
    for symbol in symbols:
        for tf in timeframes:
            df = provided.get((symbol, tf))
            if df is None:
                df = load_or_fetch_symbol_data(
                    symbol, tf, days, raw_data_dir, exchange, use_synthetic, synthetic_n
                )
            if df is None or len(df) == 0:
                continue
            cpy = candles_per_year_for_timeframe(tf)
            tf_base = dataclasses.replace(
                base_config, symbol=symbol, timeframe=tf, candles_per_year=cpy
            )
            feat = features.add_all_features(df, tf_base, drop_warmup=True)
            if len(feat) == 0:
                continue
            for mode in strategy_modes:
                for entry in entry_modes:
                    for use_ml in ml_options:
                        for use_tr in trailing_options:
                            for fee_aware in fee_aware_options:
                                min_stop = base_config.min_stop_distance_pct
                                if fee_aware and min_stop == 0.0:
                                    min_stop = 0.008
                                cfg = dataclasses.replace(
                                    tf_base,
                                    strategy_mode=mode,
                                    entry_mode=entry,
                                    use_ml=bool(use_ml),
                                    use_trailing_stop=bool(use_tr),
                                    use_fee_aware_sizing=bool(fee_aware),
                                    min_stop_distance_pct=float(min_stop),
                                )
                                report = Backtester(cfg).run(feat)
                                rows.append(
                                    ResearchRow(
                                        symbol=symbol,
                                        timeframe=tf,
                                        strategy_mode=mode,
                                        entry_mode=entry,
                                        use_ml=bool(use_ml),
                                        use_trailing_stop=bool(use_tr),
                                        use_fee_aware_sizing=bool(fee_aware),
                                        min_stop_distance_pct=float(min_stop),
                                        total_return_pct=float(report.total_return_pct),
                                        final_equity=float(report.final_equity),
                                        max_drawdown_pct=float(report.max_drawdown_pct),
                                        sharpe_ratio=float(report.sharpe_ratio),
                                        win_rate_pct=float(report.win_rate_pct),
                                        profit_factor=float(report.profit_factor),
                                        total_trades=int(report.total_trades),
                                        total_fees_paid=float(report.total_fees_paid),
                                    )
                                )
    results = pd.DataFrame([dataclasses.asdict(r) for r in rows], columns=RESEARCH_COLUMNS)
    if not results.empty:
        results = results.sort_values(
            ["sharpe_ratio", "profit_factor", "total_return_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return results


def _fmt_cell(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.4f}"
    return str(value)


def to_markdown(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
    """Format bảng Markdown kiểu GitHub bằng pure Python (không phụ thuộc tabulate)."""
    cols = list(columns) if columns is not None else list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt_cell(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *body])


def export_research_results(
    results_df: pd.DataFrame,
    out_csv: Optional[Path] = None,
    out_md: Optional[Path] = None,
) -> str:
    """Xuất kết quả ra CSV/Markdown (nếu có) và luôn trả về chuỗi bảng Markdown."""
    if out_csv is not None:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
    md = to_markdown(results_df)
    if out_md is not None:
        md_path = Path(out_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
    return md
