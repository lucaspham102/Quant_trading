"""Download and verify historical 1h Parquet cache for U6 symbols (SUI, NEAR, ADA, DOT).

Downloads Spot 1h, Perp 1h, and Funding rate history from Binance to data/raw.
Also mirrors to data/raw/spot_1h/, data/raw/perp_1h/, and data/raw/funding/ for completeness.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TradingConfig
from src.data_loader import DataLoader, raw_ohlcv_to_dataframe
from src.funding_loader import fetch_funding_history, fetch_spot_ohlcv
from src.raw_data_refresh import cache_paths, _clean_perp, _clean_spot

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("download_u6_cache")

TARGET_SYMBOLS: Sequence[str] = (
    "SUI/USDT:USDT",
    "NEAR/USDT:USDT",
    "ADA/USDT:USDT",
    "DOT/USDT:USDT",
)

START_DATE = "2023-04-01"
END_DATE = "2026-10-31"


def download_perp_1h(
    symbol: str,
    raw_dir: Path,
    perp_exchange: ccxt.Exchange,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> Path:
    out_path = raw_dir / f"{_clean_perp(symbol)}_1h.parquet"
    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
    tf_ms = 3600 * 1000

    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            if not existing.empty:
                first_ts = int(existing.index[0].value // 1_000_000)
                last_ts = int(existing.index[-1].value // 1_000_000)
                # If covers start (or listing) and end, return
                if last_ts >= end_ms - 2 * tf_ms and (first_ts <= since_ms + 24 * tf_ms or "SUI" in symbol):
                    logger.info("Perp cache for %s already sufficient: %d bars (%s -> %s)",
                                symbol, len(existing), existing.index[0], existing.index[-1])
                    return out_path
        except Exception as e:
            logger.warning("Error reading existing perp cache %s: %s", out_path, e)

    logger.info("Downloading Perp 1h for %s from %s to %s...", symbol, start_date, end_date)
    all_rows: list[list[float | int]] = []
    since = since_ms
    limit = 1000

    while since <= end_ms:
        try:
            rows = perp_exchange.fetch_ohlcv(symbol, timeframe="1h", since=since, limit=limit)
        except Exception as exc:
            logger.error("Error fetching perp ohlcv for %s at since=%d: %s", symbol, since, exc)
            time.sleep(2)
            continue

        if not rows:
            break
        for r in rows:
            if int(r[0]) <= end_ms:
                all_rows.append(r)
        last_t = int(rows[-1][0])
        next_since = last_t + tf_ms
        if next_since <= since:
            break
        since = next_since
        time.sleep(0.1)

    if not all_rows:
        logger.error("No perp rows retrieved for %s!", symbol)
        return out_path

    df = raw_ohlcv_to_dataframe(all_rows, symbol, "1h")
    # Deduplicate index
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow")
    logger.info("Saved Perp 1h %s: %d bars (%s -> %s)", symbol, len(df), df.index[0], df.index[-1])
    return out_path


def download_spot_1h(
    symbol: str,
    raw_dir: Path,
    spot_exchange: ccxt.Exchange,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> Path:
    spot_sym = symbol.split(":")[0]
    clean_s = _clean_spot(symbol)
    out_path = raw_dir / f"{clean_s}_SPOT_1h.parquet"

    logger.info("Fetching Spot 1h for %s (%s)...", symbol, spot_sym)
    df = fetch_spot_ohlcv(
        spot_symbol=spot_sym,
        timeframe="1h",
        start_date=start_date,
        end_date=end_date,
        data_dir=str(raw_dir),
        exchange=spot_exchange,
    )
    logger.info("Spot 1h %s complete: %d bars (%s -> %s)", symbol, len(df), df.index[0], df.index[-1])
    return out_path


def download_funding(
    symbol: str,
    raw_dir: Path,
    perp_exchange: ccxt.Exchange,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> Path:
    out_path = raw_dir / f"{_clean_perp(symbol)}_funding.parquet"
    logger.info("Fetching Funding history for %s...", symbol)
    df = fetch_funding_history(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_dir=str(raw_dir),
        exchange=perp_exchange,
    )
    logger.info("Funding %s complete: %d records", symbol, len(df))
    return out_path


def mirror_to_subdirs(raw_dir: Path, symbols: Sequence[str]) -> None:
    """Mirror cache files into spot_1h/, perp_1h/, and funding/ subdirectories."""
    spot_dir = raw_dir / "spot_1h"
    perp_dir = raw_dir / "perp_1h"
    fund_dir = raw_dir / "funding"

    spot_dir.mkdir(parents=True, exist_ok=True)
    perp_dir.mkdir(parents=True, exist_ok=True)
    fund_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        paths = cache_paths(raw_dir, sym, "1h")
        if paths["spot"].exists():
            shutil.copy2(paths["spot"], spot_dir / paths["spot"].name)
        if paths["perp"].exists():
            shutil.copy2(paths["perp"], perp_dir / paths["perp"].name)
        if paths["funding"].exists():
            shutil.copy2(paths["funding"], fund_dir / paths["funding"].name)
    logger.info("Mirrored cache files to subdirectories: spot_1h/, perp_1h/, funding/.")


def main() -> int:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    perp_ex = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 25000})
    spot_ex = ccxt.binance({"enableRateLimit": True, "timeout": 25000})

    for sym in TARGET_SYMBOLS:
        logger.info("=== Processing %s ===", sym)
        download_perp_1h(sym, raw_dir, perp_ex)
        download_spot_1h(sym, raw_dir, spot_ex)
        download_funding(sym, raw_dir, perp_ex)

    # Mirror all 12 symbols (LIQUID_8 + 4 new) to subdirs
    all_12 = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
        "DOGE/USDT:USDT", "XRP/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
        "SUI/USDT:USDT", "NEAR/USDT:USDT", "ADA/USDT:USDT", "DOT/USDT:USDT",
    ]
    mirror_to_subdirs(raw_dir, all_12)
    logger.info("All U6 symbols downloaded and cached successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
