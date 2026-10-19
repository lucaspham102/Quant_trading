"""Module tải và đồng bộ dữ liệu Funding Rate (Binance USDT-M / Coin-M).

Cung cấp:
- fetch_funding_history: Tải phân trang và cache dữ liệu Funding Rate.
- align_ohlcv_with_funding: Khớp dữ liệu OHLCV với Funding Rate, loại bỏ hoàn toàn look-ahead bias.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ensure_funding_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo DataFrame tuân thủ schema chuẩn: timestamp, datetime, funding_rate, mark_price."""
    out = df.copy()
    out["timestamp"] = out["timestamp"].astype("int64")
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.as_unit("ns")
    out["funding_rate"] = out["funding_rate"].astype("float64")
    if "mark_price" not in out.columns:
        out["mark_price"] = np.nan
    else:
        out["mark_price"] = out["mark_price"].astype("float64")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values(by="timestamp").reset_index(drop=True)
    return out[["timestamp", "datetime", "funding_rate", "mark_price"]]


def _empty_funding_frame() -> pd.DataFrame:
    """Tạo DataFrame rỗng với schema chuẩn."""
    return pd.DataFrame({
        "timestamp": pd.Series([], dtype="int64"),
        "datetime": pd.Series([], dtype="datetime64[ns, UTC]"),
        "funding_rate": pd.Series([], dtype="float64"),
        "mark_price": pd.Series([], dtype="float64"),
    })


def fetch_funding_history(
    symbol: str,
    start_date: str = "2023-05-07",
    end_date: str | None = None,
    data_dir: str = "./data/raw",
    exchange: Any = None,
) -> pd.DataFrame:
    """Tải lịch sử Funding Rate phân trang từ sàn và lưu trữ cache cục bộ.

    Tham số:
        symbol: Cặp giao dịch Binance (ví dụ 'BTC/USDT:USDT').
        start_date: Ngày bắt đầu (UTC), mặc định '2023-05-07'.
        end_date: Ngày kết thúc (UTC) hoặc None (lấy đến hiện tại).
        data_dir: Thư mục lưu cache Parquet.
        exchange: Đối tượng ccxt exchange hoặc None (tự khởi tạo ccxt.binanceusdm).

    Trả về:
        DataFrame có các cột: timestamp, datetime, funding_rate, mark_price.
    """
    clean = symbol.replace("/", "_").replace(":", "_")
    cache_path = Path(data_dir) / f"{clean}_funding.parquet"

    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000) if end_date is not None else None

    # Khởi tạo exchange nếu chưa được truyền vào
    if exchange is None:
        import ccxt

        exchange = ccxt.binanceusdm({"enableRateLimit": True})

    now_ms = (
        int(exchange.milliseconds())
        if hasattr(exchange, "milliseconds") and callable(exchange.milliseconds)
        else int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    )
    target_end_ms = min(until_ms, now_ms) if until_ms is not None else now_ms
    eight_hours_ms = 8 * 3600 * 1000

    cached_df: pd.DataFrame | None = None
    if cache_path.exists():
        try:
            raw_cached = pd.read_parquet(cache_path)
            if not raw_cached.empty and "timestamp" in raw_cached.columns:
                cached_df = _ensure_funding_schema(raw_cached)
                min_ts = int(cached_df["timestamp"].min())
                max_ts = int(cached_df["timestamp"].max())
                covers_start = min_ts <= (since_ms + eight_hours_ms)
                covers_end = max_ts >= (target_end_ms - eight_hours_ms)
                if covers_start and covers_end:
                    # Cache đã bao phủ trọn vẹn dải thời gian yêu cầu
                    return cached_df
        except Exception:
            cached_df = None

    # Xác định mốc thời gian bắt đầu tải dữ liệu bổ sung
    if cached_df is not None and not cached_df.empty:
        min_ts = int(cached_df["timestamp"].min())
        max_ts = int(cached_df["timestamp"].max())
        if min_ts <= (since_ms + eight_hours_ms):
            fetch_start_ms = max_ts + 1
        else:
            fetch_start_ms = since_ms
    else:
        fetch_start_ms = since_ms

    # Phân trang tải dữ liệu qua exchange API
    limit = 1000
    all_rows: list[dict[str, Any]] = []
    curr_since = fetch_start_ms

    while True:
        records = exchange.fetch_funding_rate_history(symbol, since=curr_since, limit=limit)
        if not records:
            break
        all_rows.extend(records)
        last_ts = int(records[-1]["timestamp"])
        if until_ms is not None and last_ts >= until_ms:
            break
        if len(records) < limit:
            break
        next_since = last_ts + 1
        if next_since <= curr_since:
            break
        curr_since = next_since

    # Chuyển đổi bản ghi tải mới thành DataFrame
    if all_rows:
        parsed_rows: list[dict[str, Any]] = []
        for r in all_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            fr = r.get("fundingRate") if r.get("fundingRate") is not None else r.get("funding_rate")
            mp = r.get("markPrice") if r.get("markPrice") is not None else r.get("mark_price")
            parsed_rows.append({
                "timestamp": int(ts),
                "funding_rate": float(fr) if fr is not None else np.nan,
                "mark_price": float(mp) if mp is not None else np.nan,
            })
        new_df = pd.DataFrame(parsed_rows)
        new_df["timestamp"] = new_df["timestamp"].astype("int64")
        new_df["datetime"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True).dt.as_unit("ns")
        new_df["funding_rate"] = new_df["funding_rate"].astype("float64")
        new_df["mark_price"] = new_df["mark_price"].astype("float64")
        new_df = _ensure_funding_schema(new_df)
    else:
        new_df = _empty_funding_frame()

    # Gộp với cache hiện có (nếu có)
    parts: list[pd.DataFrame] = []
    if cached_df is not None and not cached_df.empty:
        parts.append(cached_df)
    if not new_df.empty:
        parts.append(new_df)

    if parts:
        final_df = pd.concat(parts, axis=0, ignore_index=True)
        final_df = _ensure_funding_schema(final_df)
    else:
        final_df = _empty_funding_frame()

    # Lưu cache nếu có dữ liệu
    if not final_df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(cache_path, index=False)

    return final_df


def align_ohlcv_with_funding(ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """Khớp dữ liệu OHLCV với lịch sử Funding Rate mà không có look-ahead bias.

    Tham số:
        ohlcv_df: DataFrame OHLCV có index là DatetimeIndex (UTC, đơn vị ns).
        funding_df: DataFrame chứa lịch sử funding rate (có cột 'timestamp'/'datetime' và 'funding_rate').

    Trả về:
        Bản sao ohlcv_df có DatetimeIndex tz='UTC' ns tên 'datetime', bổ sung:
        - funding_rate_event: Giá trị funding rate tại đúng cây nến thanh toán (00/08/16 UTC), các nến khác = 0.0.
        - last_known_funding_rate: Funding rate của kỳ gần nhất đã chốt tại thời điểm <= t (backward merge_asof).
    """
    out = ohlcv_df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = idx.as_unit("ns")
    idx.name = "datetime"
    out.index = idx

    if out.empty:
        out["funding_rate_event"] = pd.Series([], dtype="float64", index=idx)
        out["last_known_funding_rate"] = pd.Series([], dtype="float64", index=idx)
        return out

    # Chuẩn bị DataFrame funding rate
    f_df = funding_df.copy()
    if "datetime" not in f_df.columns:
        if isinstance(f_df.index, pd.DatetimeIndex):
            f_df["datetime"] = f_df.index
        elif "timestamp" in f_df.columns:
            f_df["datetime"] = pd.to_datetime(f_df["timestamp"], unit="ms", utc=True)
        else:
            raise ValueError("funding_df phải có cột 'datetime' hoặc 'timestamp'")

    f_df["datetime"] = pd.to_datetime(f_df["datetime"], utc=True).dt.as_unit("ns")
    f_df["funding_rate"] = f_df["funding_rate"].astype("float64")
    f_df = f_df.dropna(subset=["datetime", "funding_rate"])
    f_df = f_df.sort_values(by="datetime").drop_duplicates(subset=["datetime"], keep="last")

    # 1. funding_rate_event:
    # Xây Series funding keyed theo datetime đã làm tròn về nến (giờ UTC thuộc {0, 8, 16})
    # rồi reindex theo index ohlcv.fillna(0.0)
    event_dt = pd.DatetimeIndex(f_df["datetime"].dt.round("1h"), tz="UTC").as_unit("ns")
    f_events = pd.Series(f_df["funding_rate"].to_numpy(dtype="float64"), index=event_dt)
    valid_mask = f_events.index.hour.isin([0, 8, 16]) & (f_events.index.minute == 0)
    f_events = f_events[valid_mask]
    f_events = f_events[~f_events.index.duplicated(keep="last")]

    out["funding_rate_event"] = f_events.reindex(out.index).fillna(0.0).astype("float64")

    # 2. last_known_funding_rate:
    # Dùng pd.merge_asof với direction="backward" sao cho tại nến t chỉ biết funding đã chốt ở <= t.
    if f_df.empty:
        out["last_known_funding_rate"] = 0.0
        return out

    lookup_ohlcv = pd.DataFrame({"_lookup_dt": out.index}, index=out.index)
    lookup_funding = pd.DataFrame({
        "_lookup_dt": pd.DatetimeIndex(f_df["datetime"].dt.round("1h"), tz="UTC").as_unit("ns"),
        "_rate": f_df["funding_rate"].to_numpy(dtype="float64"),
    }).drop_duplicates(subset=["_lookup_dt"], keep="last").sort_values(by="_lookup_dt")

    merged = pd.merge_asof(
        lookup_ohlcv,
        lookup_funding,
        on="_lookup_dt",
        direction="backward",
    )
    out["last_known_funding_rate"] = merged["_rate"].fillna(0.0).to_numpy(dtype="float64")

    return out
