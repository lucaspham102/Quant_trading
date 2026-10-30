"""Tải, làm sạch và lưu trữ dữ liệu OHLCV (Binance USDT-M Futures)."""
import time
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from src.config import TradingConfig

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 7 * 86_400_000,
}


class ExchangeProtocol(Protocol):
    """Giao diện tối thiểu cho đối tượng ccxt.binanceusdm hoặc FakeExchange trong Unit Test."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> list[list[float | int]]: ...

    def milliseconds(self) -> int: ...


def timeframe_to_ms(timeframe: str) -> int:
    """Đổi chuỗi khung thời gian ('15m', '1h', '1d'...) sang mili-giây."""
    if len(timeframe) < 2 or timeframe[-1] not in _TIMEFRAME_UNIT_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    try:
        amount = int(timeframe[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}") from exc
    if amount <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return amount * _TIMEFRAME_UNIT_MS[timeframe[-1]]


def _empty_ohlcv_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp").as_unit("ns")
    return pd.DataFrame(
        {col: pd.Series([], dtype="float64") for col in OHLCV_COLUMNS}, index=index
    )


def _index_to_ms(index: pd.DatetimeIndex) -> np.ndarray:
    return index.as_unit("ns").asi8 // 1_000_000


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ép về schema chuẩn: index UTC ns tên 'timestamp', cột float64, sort, unique."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.as_unit("ns")
    idx.name = "timestamp"
    out.index = idx
    out = out[OHLCV_COLUMNS].astype("float64")
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def raw_ohlcv_to_dataframe(
    raw_rows: list[list[float | int]],
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "15m",
) -> pd.DataFrame:
    """
    Chuyển danh sách [timestamp_ms, open, high, low, close, volume] thành DataFrame chuẩn.
    - Ép kiểu cột sang float64, đặt DatetimeIndex tz='UTC' tên 'timestamp'.
    - Lọc trùng lặp timestamp (giữ bản ghi xuất hiện cuối cùng: keep='last').
    - Sắp xếp tăng dần theo thời gian.
    """
    if len(raw_rows) == 0:
        df = _empty_ohlcv_frame()
    else:
        rows = [list(r[:6]) for r in raw_rows]
        raw = pd.DataFrame(rows, columns=["timestamp_ms", *OHLCV_COLUMNS])
        ts = pd.to_datetime(raw["timestamp_ms"].astype("int64"), unit="ms", utc=True)
        idx = pd.DatetimeIndex(ts).as_unit("ns")
        idx.name = "timestamp"
        df = raw[OHLCV_COLUMNS].astype("float64")
        df.index = idx
        df = _normalize_frame(df)
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


def drop_incomplete_last_candle(
    df: pd.DataFrame,
    timeframe: str,
    current_server_time_ms: int,
) -> pd.DataFrame:
    """
    Nếu timestamp_open_ms + timeframe_duration_ms > current_server_time_ms
    (nến cuối chưa đóng hoàn chỉnh), loại bỏ nến cuối đó.
    """
    out = df.copy()
    out.attrs.update(df.attrs)
    if len(out) == 0:
        return out
    last_open_ms = int(_index_to_ms(out.index)[-1])
    if last_open_ms + timeframe_to_ms(timeframe) > current_server_time_ms:
        kept = out.iloc[:-1].copy()
        kept.attrs.update(df.attrs)
        return kept
    return out


def merge_and_deduplicate_ohlcv(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nối dữ liệu mới vào dữ liệu hiện có, lọc duplicate theo index timestamp (keep='last'),
    sắp xếp tăng dần và bảo toàn schema chuẩn.
    """
    parts = [_normalize_frame(d) for d in (existing_df, new_df) if len(d) > 0]
    if not parts:
        merged = _empty_ohlcv_frame()
    else:
        merged = _normalize_frame(pd.concat(parts, axis=0))
    attrs = dict(existing_df.attrs) or dict(new_df.attrs)
    merged.attrs.update(attrs)
    return merged


class DataLoader:
    def __init__(
        self,
        config: TradingConfig = TradingConfig(),
        exchange: Optional[ExchangeProtocol] = None,
    ) -> None:
        """Nếu exchange is None, khởi tạo ccxt.binanceusdm({'enableRateLimit': True})."""
        self.config = config
        if exchange is None:
            import ccxt  # import trễ để unit test offline không cần ccxt

            exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 20000})
        self.exchange = exchange

    @property
    def timeframe_ms(self) -> int:
        return timeframe_to_ms(self.config.timeframe)

    def default_file_path(self) -> Path:
        name = self.config.symbol.replace("/", "_").replace(":", "_")
        return Path(self.config.raw_data_dir) / f"{name}_{self.config.timeframe}.parquet"

    def fetch_historical_ohlcv(
        self,
        since_ms: int,
        until_ms: Optional[int] = None,
        limit_per_call: int = 1000,
    ) -> pd.DataFrame:
        """
        Tải phân trang OHLCV từ since_ms đến until_ms (hoặc hiện tại).
        Con trỏ since = last_timestamp_ms + timeframe_ms sau mỗi trang.
        Dừng khi rỗng, đạt until_ms, hoặc không có nến mới hơn.
        """
        tf_ms = self.timeframe_ms
        server_now = int(self.exchange.milliseconds())
        end_ms = server_now if until_ms is None else int(until_ms)

        all_rows: list[list[float | int]] = []
        since = int(since_ms)
        start_mono = time.monotonic()
        max_elapsed_sec = 60.0
        max_pages = 500
        pages = 0
        while since <= end_ms:
            pages += 1
            if pages > max_pages or (time.monotonic() - start_mono) > max_elapsed_sec:
                break
            rows = self.exchange.fetch_ohlcv(
                self.config.symbol,
                timeframe=self.config.timeframe,
                since=since,
                limit=limit_per_call,
            )
            if not rows:
                break
            last_ts = int(rows[-1][0])
            all_rows.extend(r for r in rows if int(r[0]) <= end_ms)
            next_since = last_ts + tf_ms
            if next_since <= since:  # không có nến mới hơn
                break
            since = next_since

        df = raw_ohlcv_to_dataframe(all_rows, self.config.symbol, self.config.timeframe)
        return drop_incomplete_last_candle(df, self.config.timeframe, server_now)

    def save_parquet(self, df: pd.DataFrame, file_path: Optional[Path] = None) -> Path:
        """Lưu DataFrame chuẩn ra Parquet. Trả về đường dẫn đã lưu."""
        path = Path(file_path) if file_path is not None else self.default_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        out = _normalize_frame(df)
        out.attrs.update(df.attrs)
        out.to_parquet(path, engine="pyarrow")
        return path

    def load_parquet(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Đọc Parquet và khôi phục DatetimeIndex UTC cùng dtypes float64."""
        path = Path(file_path) if file_path is not None else self.default_file_path()
        df = _normalize_frame(pd.read_parquet(path, engine="pyarrow"))
        df.attrs["symbol"] = self.config.symbol
        df.attrs["timeframe"] = self.config.timeframe
        return df

    def update_incremental(
        self,
        file_path: Optional[Path] = None,
        default_start_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Nếu file tồn tại: tải tiếp từ timestamp cuối (nến cuối được tải lại và ghi đè),
        gộp + lọc trùng, lưu đè. Nếu chưa tồn tại: tải mới từ default_start_ms và lưu.
        """
        path = Path(file_path) if file_path is not None else self.default_file_path()
        if path.exists():
            existing = self.load_parquet(path)
            if len(existing) > 0:
                since_ms = int(_index_to_ms(existing.index)[-1])
            elif default_start_ms is not None:
                since_ms = int(default_start_ms)
            else:
                raise ValueError("Parquet file is empty and default_start_ms was not provided")
            new_df = self.fetch_historical_ohlcv(since_ms)
            merged = merge_and_deduplicate_ohlcv(existing, new_df)
        else:
            if default_start_ms is None:
                raise ValueError("default_start_ms is required when the Parquet file does not exist")
            merged = self.fetch_historical_ohlcv(int(default_start_ms))
        merged.attrs["symbol"] = self.config.symbol
        merged.attrs["timeframe"] = self.config.timeframe
        self.save_parquet(merged, path)
        return merged
