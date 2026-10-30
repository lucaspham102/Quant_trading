"""Incremental refresh of the 1h Spot / Perp / Funding Parquet cache in ``data/raw``.

The live demo loop reads the cache through ``CarryLiveExecutor._load_symbol_from_cache``
and nothing else ever updates it, so the candles silently freeze. This module closes that
gap while keeping every write atomic: each file is first copied to a private staging
directory, updated there by the existing loaders (``DataLoader.update_incremental``,
``fetch_spot_ohlcv``, ``fetch_funding_history``) and only then moved over the original
with ``os.replace``. A reader (or a crash) can therefore never observe a half-written
Parquet file.

The module never raises: every failure is returned in the report so a flaky network
cannot kill a trading cycle.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

STAGING_DIRNAME = ".refresh_staging"
HOUR_MS = 3600 * 1000
FUNDING_PERIOD_MS = 8 * HOUR_MS


def _clean_perp(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _clean_spot(symbol: str) -> str:
    spot = symbol.split(":")[0]
    return spot.replace("/", "").replace("_", "").upper().replace("USDTUSDT", "USDT")


def cache_paths(raw_dir: str | Path, symbol: str, timeframe: str = "1h") -> dict[str, Path]:
    """Return the three cache files (perp / spot / funding) used for ``symbol``."""
    root = Path(raw_dir)
    return {
        "perp": root / f"{_clean_perp(symbol)}_{timeframe}.parquet",
        "spot": root / f"{_clean_spot(symbol)}_SPOT_{timeframe}.parquet",
        "funding": root / f"{_clean_perp(symbol)}_funding.parquet",
    }


def _last_ts_ms(path: Path, kind: str) -> Optional[int]:
    """Last timestamp (epoch ms) stored in a cache file, or None if unreadable."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if kind == "funding":
            return int(df["timestamp"].max())
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        elif "timestamp" in df.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        else:
            return None
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return int(idx.as_unit("ns").asi8.max() // 1_000_000)
    except Exception:  # noqa: BLE001 - freshness probing must never raise
        return None


def _first_ts_ms(path: Path, kind: str) -> Optional[int]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if kind == "funding":
            return int(df["timestamp"].min())
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return int(idx.as_unit("ns").asi8.min() // 1_000_000)
    except Exception:  # noqa: BLE001
        return None


def cache_last_timestamps(
    raw_dir: str | Path, symbols: Sequence[str], timeframe: str = "1h"
) -> dict[str, dict[str, Optional[pd.Timestamp]]]:
    """Per symbol, the last stored timestamp of each leg (perp / spot / funding)."""
    out: dict[str, dict[str, Optional[pd.Timestamp]]] = {}
    for sym in symbols:
        legs: dict[str, Optional[pd.Timestamp]] = {}
        for kind, path in cache_paths(raw_dir, sym, timeframe).items():
            ms = _last_ts_ms(path, kind)
            legs[kind] = None if ms is None else pd.Timestamp(ms, unit="ms", tz="UTC")
        out[sym] = legs
    return out


def needs_refresh(
    raw_dir: str | Path,
    symbols: Sequence[str],
    *,
    now_ms: int,
    max_age_hours: float = 1.0,
    timeframe: str = "1h",
) -> bool:
    """True when any hourly leg's last *open* timestamp is older than the last closed bar.

    A bar opening at ``T`` closes at ``T + 1h``; the newest closed bar on the exchange
    opens at ``floor(now, 1h) - 1h``. We refresh once the cache lags that by more than
    ``max_age_hours`` hours of slack.
    """
    newest_closed_open = (now_ms // HOUR_MS) * HOUR_MS - HOUR_MS
    slack_ms = int(max(0.0, float(max_age_hours) - 1.0) * HOUR_MS)
    for legs in cache_last_timestamps(raw_dir, symbols, timeframe).values():
        for kind in ("perp", "spot"):
            ts = legs.get(kind)
            if ts is None:
                return True
            if int(ts.value // 1_000_000) < newest_closed_open - slack_ms:
                return True
        f_ts = legs.get("funding")
        if f_ts is None or int(f_ts.value // 1_000_000) < now_ms - FUNDING_PERIOD_MS - 2 * HOUR_MS:
            return True
    return False


def _atomic_install(staged: Path, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(staged, tmp)
    os.replace(tmp, target)


def _default_exchanges() -> tuple[Any, Any]:
    import ccxt

    return (
        ccxt.binanceusdm({"enableRateLimit": True}),
        ccxt.binance({"enableRateLimit": True}),
    )


def refresh_raw_data(
    symbols: Sequence[str],
    raw_dir: str | Path = "data/raw",
    *,
    timeframe: str = "1h",
    max_age_hours: float = 1.0,
    force: bool = False,
    perp_exchange: Any = None,
    spot_exchange: Any = None,
    now_ms: Optional[int] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Pull the missing 1h Perp / Spot candles and funding events into ``raw_dir``.

    Returns a report ``{"refreshed": bool, "skipped": bool, "errors": {...}, "last_ts": {...}}``
    and never raises. Each leg is updated on a staged copy and installed atomically.
    """
    from src.config import TradingConfig
    from src.data_loader import DataLoader
    from src.funding_loader import fetch_funding_history, fetch_spot_ohlcv

    raw = Path(raw_dir)
    report: dict[str, Any] = {"refreshed": False, "skipped": False, "errors": {}, "last_ts": {}}
    say = on_progress or (lambda _m: None)

    if now_ms is None:
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    try:
        if not force and not needs_refresh(
            raw, symbols, now_ms=now_ms, max_age_hours=max_age_hours, timeframe=timeframe
        ):
            report["skipped"] = True
            report["last_ts"] = cache_last_timestamps(raw, symbols, timeframe)
            return report
    except Exception as exc:  # noqa: BLE001
        report["errors"]["_probe"] = f"{type(exc).__name__}: {exc}"

    try:
        if perp_exchange is None or spot_exchange is None:
            d_perp, d_spot = _default_exchanges()
            perp_exchange = perp_exchange or d_perp
            spot_exchange = spot_exchange or d_spot
    except Exception as exc:  # noqa: BLE001
        report["errors"]["_exchange"] = f"{type(exc).__name__}: {exc}"
        return report

    raw.mkdir(parents=True, exist_ok=True)
    staging_root = raw / STAGING_DIRNAME
    staging_root.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        paths = cache_paths(raw, sym, timeframe)
        stage = Path(tempfile.mkdtemp(prefix="stage_", dir=staging_root))
        try:
            # ---- Perp OHLCV (DataLoader.update_incremental)
            try:
                staged_perp = stage / paths["perp"].name
                if paths["perp"].exists():
                    shutil.copy2(paths["perp"], staged_perp)
                cfg = TradingConfig(symbol=sym, timeframe=timeframe, raw_data_dir=str(stage))
                loader = DataLoader(cfg, exchange=perp_exchange)
                default_start = now_ms - 1600 * HOUR_MS
                loader.update_incremental(file_path=staged_perp, default_start_ms=default_start)
                _atomic_install(staged_perp, paths["perp"])
                say(f"[REFRESH] {sym} perp ok")
            except Exception as exc:  # noqa: BLE001
                report["errors"][f"{sym}:perp"] = f"{type(exc).__name__}: {exc}"

            # ---- Spot OHLCV (fetch_spot_ohlcv keeps its own cache in data_dir)
            try:
                first = _first_ts_ms(paths["spot"], "spot")
                start_ms = first if first is not None else now_ms - 1600 * HOUR_MS
                if paths["spot"].exists():
                    shutil.copy2(paths["spot"], stage / paths["spot"].name)
                fetch_spot_ohlcv(
                    sym.split(":")[0],
                    timeframe=timeframe,
                    start_date=pd.Timestamp(start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                    data_dir=str(stage),
                    exchange=spot_exchange,
                )
                staged_spot = stage / paths["spot"].name
                if staged_spot.exists():
                    _atomic_install(staged_spot, paths["spot"])
                    say(f"[REFRESH] {sym} spot ok")
            except Exception as exc:  # noqa: BLE001
                report["errors"][f"{sym}:spot"] = f"{type(exc).__name__}: {exc}"

            # ---- Funding history
            try:
                first_f = _first_ts_ms(paths["funding"], "funding")
                start_f = first_f if first_f is not None else now_ms - 1600 * HOUR_MS
                if paths["funding"].exists():
                    shutil.copy2(paths["funding"], stage / paths["funding"].name)
                fetch_funding_history(
                    sym,
                    start_date=pd.Timestamp(start_f, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                    data_dir=str(stage),
                    exchange=perp_exchange,
                )
                staged_fund = stage / paths["funding"].name
                if staged_fund.exists():
                    _atomic_install(staged_fund, paths["funding"])
                    say(f"[REFRESH] {sym} funding ok")
            except Exception as exc:  # noqa: BLE001
                report["errors"][f"{sym}:funding"] = f"{type(exc).__name__}: {exc}"
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    try:
        shutil.rmtree(staging_root, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass
    report["refreshed"] = True
    report["last_ts"] = cache_last_timestamps(raw, symbols, timeframe)
    return report
