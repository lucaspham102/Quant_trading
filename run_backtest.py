"""CLI chạy backtest: dữ liệu thật (Binance) hoặc dữ liệu giả lập (--synthetic, offline)."""
import argparse
import dataclasses
import sys
import time

import numpy as np
import pandas as pd

from src.backtester import Backtester, BacktestReport
from src.config import TradingConfig

_TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def generate_synthetic_ohlcv(
    n_candles: int = 2000,
    start_price: float = 60000.0,
    seed: int = 42,
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
) -> pd.DataFrame:
    """Sinh OHLCV giả lập: GBM + các pha xu hướng luân phiên + đột biến volume (seed cố định)."""
    rng = np.random.default_rng(seed)
    minutes = _TF_MINUTES.get(timeframe, 15)

    # Pha xu hướng: drift đổi dấu mỗi ~250 nến
    phase_len = 250
    n_phases = n_candles // phase_len + 1
    phase_drift = rng.choice([-1.0, 1.0, 1.0, 0.0], size=n_phases) * 0.0008
    drift = np.repeat(phase_drift, phase_len)[:n_candles]
    sigma = 0.002
    shocks = rng.normal(0.0, sigma, size=n_candles)

    # Đột biến: một số nến có biến động & volume lớn
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

    index = pd.date_range("2024-01-01", periods=n_candles, freq=f"{minutes}min", tz="UTC", name="timestamp")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest chiến lược Donchian Breakout (Binance USDT-M Futures)")
    parser.add_argument("--synthetic", action="store_true", help="Dùng dữ liệu giả lập (offline, không gọi mạng)")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Cặp giao dịch")
    parser.add_argument("--timeframe", default="15m", help="Khung thời gian")
    parser.add_argument("--days", type=int, default=30, help="Số ngày lịch sử khi tải dữ liệu thật")
    return parser.parse_args(argv)


def format_report(report: BacktestReport, config: TradingConfig) -> str:
    pf = "inf" if np.isinf(report.profit_factor) else f"{report.profit_factor:.2f}"
    lines = [
        "=" * 46,
        f" BACKTEST REPORT - {config.symbol} ({config.timeframe})",
        "=" * 46,
        f" Initial Capital     : {config.initial_capital:>14.2f} USDT",
        f" Final Equity        : {report.final_equity:>14.2f} USDT",
        f" Total Return        : {report.total_return_pct:>14.2f} %",
        f" Max Drawdown        : {report.max_drawdown_pct:>14.2f} %",
        f" Sharpe Ratio (ann.) : {report.sharpe_ratio:>14.2f}",
        f" Win Rate            : {report.win_rate_pct:>14.2f} %",
        f" Profit Factor       : {pf:>14}",
        f" Total Trades        : {report.total_trades:>14d}",
        f" Total Fees Paid     : {report.total_fees_paid:>14.2f} USDT",
        "=" * 46,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = dataclasses.replace(TradingConfig(), symbol=args.symbol, timeframe=args.timeframe)

    try:
        if args.synthetic:
            df = generate_synthetic_ohlcv(symbol=args.symbol, timeframe=args.timeframe)
        else:
            from src.data_loader import DataLoader

            loader = DataLoader(config)
            since_ms = int((time.time() - args.days * 86400) * 1000)
            df = loader.fetch_historical_ohlcv(since_ms)
        if df is None or len(df) == 0:
            print("Không có dữ liệu để backtest.", file=sys.stderr)
            return 1
        report = Backtester(config).run(df)
        print(format_report(report, config))
    except Exception as exc:  # noqa: BLE001
        print(f"Lỗi khi chạy backtest: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
