"""Module tải, lưu trữ cache Parquet và đồng bộ Funding Rate đa sàn (Binance, Bybit, OKX).

Sprint 9 - Chặng 1:
Cung cấp:
- clean_symbol_name: Chuẩn hóa ký hiệu cặp giao dịch về dạng liền mạch (ví dụ 'BTCUSDT').
- fetch_exchange_funding_history: Tải phân trang và cache Parquet funding rate từng sàn.
- fetch_all_exchanges_funding: Tải và cache funding rate cho nhiều sàn và nhiều cặp.
- load_funding_by_exchange: Chỉ đọc cache Parquet cục bộ, không gọi mạng.
- align_multi_exchange_funding: Đồng bộ dữ liệu OHLCV 1h với funding đa sàn, chống look-ahead tuyệt đối.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import CORE_4_SYMBOLS
from src.funding_loader import (
    _empty_funding_frame,
    _ensure_funding_schema,
    _to_datetime_indexed,
)

SUPPORTED_EXCHANGES: tuple[str, ...] = ("binance", "bybit", "okx")


def clean_symbol_name(symbol: str) -> str:
    """Chuẩn hóa ký hiệu cặp giao dịch về dạng liền mạch không dấu phân cách.

    Ví dụ: 'BTC/USDT:USDT' -> 'BTCUSDT', 'BTC_USDT_USDT' -> 'BTCUSDT', 'BTC/USDT' -> 'BTCUSDT'.

    Tham số:
        symbol: Ký hiệu cặp giao dịch thô.

    Trả về:
        Chuỗi ký hiệu viết hoa, loại bỏ dấu '/', ':', '_', '-' và khử lặp 'USDTUSDT'.
    """
    clean = (
        symbol.replace("/", "")
        .replace(":", "")
        .replace("_", "")
        .replace("-", "")
        .upper()
    )
    if clean.endswith("USDTUSDT"):
        clean = clean[:-4]
    return clean


# Alias tương thích blueprint kiến trúc Sprint 9
clean_symbol_for_Attachment = clean_symbol_name


def _find_existing_cache_file(
    exchange_name: str,
    symbol: str,
    data_dir: str | Path,
) -> Path | None:
    """Tìm đường dẫn file cache Parquet đã có trong data_dir.

    Hỗ trợ cả quy ước đặt tên chuẩn Sprint 9 ({exchange}_{CLEAN_SYMBOL}_funding.parquet)
    và các quy ước fallback cũ từ Sprint 7/8 (đặc biệt cho sàn Binance).
    """
    dir_path = Path(data_dir)
    clean_sym = clean_symbol_name(symbol)
    ex = exchange_name.lower().strip()

    # 1. Đường dẫn chuẩn Sprint 9: {exchange}_{CLEAN_SYMBOL}_funding.parquet
    candidates = [
        dir_path / f"{ex}_{clean_sym}_funding.parquet",
    ]

    # 2. Fallback dạng {exchange}_{raw_symbol_underscore}_funding.parquet
    raw_under = symbol.replace("/", "_").replace(":", "_")
    candidates.append(dir_path / f"{ex}_{raw_under}_funding.parquet")

    # 3. Fallback đặc biệt cho Binance từ Sprint 7/8
    if ex == "binance":
        candidates.append(dir_path / f"{raw_under}_funding.parquet")
        candidates.append(dir_path / f"{clean_sym}_funding.parquet")

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def fetch_exchange_funding_history(
    exchange_name: str,
    symbol: str,
    start_date: str = "2023-05-07",
    end_date: str | None = "2026-10-18",
    data_dir: str | Path = "./data/raw",
    exchange: Any = None,
) -> pd.DataFrame:
    """Tải lịch sử Funding Rate phân trang từ sàn giao dịch và lưu trữ cache Parquet.

    Tham số:
        exchange_name: Tên sàn ('binance', 'bybit', hoặc 'okx').
        symbol: Ký hiệu cặp giao dịch chuẩn ccxt (ví dụ 'BTC/USDT:USDT').
        start_date: Ngày bắt đầu (UTC), mặc định '2023-05-07'.
        end_date: Ngày kết thúc (UTC), mặc định '2026-10-18' hoặc None.
        data_dir: Thư mục lưu trữ cache Parquet.
        exchange: Đối tượng ccxt exchange hoặc None (tự động khởi tạo ccxt bên trong hàm).

    Trả về:
        DataFrame có schema chuẩn: timestamp, datetime, funding_rate, mark_price.
        0 duplicate timestamp, 0 NaN ở funding_rate, sắp xếp tăng dần theo timestamp.
    """
    ex = exchange_name.lower().strip()
    if ex not in SUPPORTED_EXCHANGES:
        raise ValueError(
            f"Sàn giao dịch '{exchange_name}' không được hỗ trợ. "
            f"Chỉ chấp nhận một trong các sàn: {SUPPORTED_EXCHANGES}"
        )

    dir_path = Path(data_dir)
    clean_sym = clean_symbol_name(symbol)
    primary_cache_path = dir_path / f"{ex}_{clean_sym}_funding.parquet"

    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000) if end_date is not None else None
    eight_hours_ms = 8 * 3600 * 1000

    # 1. Kiểm tra cache cục bộ trước khi gọi mạng
    existing_cache = _find_existing_cache_file(ex, symbol, dir_path)
    cached_df: pd.DataFrame | None = None

    if existing_cache is not None:
        try:
            raw_cached = pd.read_parquet(existing_cache)
            if not raw_cached.empty and "timestamp" in raw_cached.columns:
                cached_cleaned = _ensure_funding_schema(raw_cached).dropna(subset=["funding_rate"])
                if not cached_cleaned.empty:
                    min_ts = int(cached_cleaned["timestamp"].min())
                    max_ts = int(cached_cleaned["timestamp"].max())
                    covers_start = min_ts <= (since_ms + eight_hours_ms)
                    covers_end = (until_ms is None) or (max_ts >= (until_ms - eight_hours_ms))
                    if covers_start and covers_end:
                        # Cache đã bao phủ trọn vẹn dải thời gian yêu cầu -> trả về luôn không gọi mạng
                        return cached_cleaned
                    cached_df = cached_cleaned
        except Exception:
            cached_df = None

    # 2. Khởi tạo sàn giao dịch ccxt khi cần tải bổ sung
    if exchange is None:
        import ccxt

        if ex == "binance":
            exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 20000})
        elif ex == "bybit":
            exchange = ccxt.bybit({"enableRateLimit": True, "timeout": 20000})
        elif ex == "okx":
            exchange = ccxt.okx({"enableRateLimit": True, "timeout": 20000})

    now_ms = (
        int(exchange.milliseconds())
        if hasattr(exchange, "milliseconds") and callable(exchange.milliseconds)
        else int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    )
    target_end_ms = min(until_ms, now_ms) if until_ms is not None else now_ms

    if cached_df is not None and not cached_df.empty:
        min_ts = int(cached_df["timestamp"].min())
        max_ts = int(cached_df["timestamp"].max())
        if min_ts <= (since_ms + eight_hours_ms):
            fetch_start_ms = max_ts + 1
        else:
            fetch_start_ms = since_ms
            cached_df = None
    else:
        fetch_start_ms = since_ms

    # 3. Vòng lặp phân trang theo đặc thù từng sàn
    all_rows: list[dict[str, Any]] = []
    curr_since = fetch_start_ms
    start_mono = time.monotonic()
    max_elapsed_sec = 120.0
    max_pages = 500
    pages = 0

    while True:
        pages += 1
        if pages > max_pages or (time.monotonic() - start_mono) > max_elapsed_sec:
            break
        if curr_since >= target_end_ms:
            break

        if ex == "binance":
            limit = 1000
            records = exchange.fetch_funding_rate_history(symbol, since=curr_since, limit=limit)
        elif ex == "bybit":
            limit = 200
            window_end = min(target_end_ms, curr_since + 200 * eight_hours_ms)
            params = {"startTime": curr_since, "endTime": window_end}
            records = exchange.fetch_funding_rate_history(symbol, since=curr_since, limit=limit, params=params)
        elif ex == "okx":
            limit = 100
            records = exchange.fetch_funding_rate_history(symbol, since=curr_since, limit=limit)
        else:
            limit = 100
            records = exchange.fetch_funding_rate_history(symbol, since=curr_since, limit=limit)

        if not records:
            break

        # Luôn sort records theo timestamp sau mỗi lần gọi API
        valid_records = [
            r for r in records
            if isinstance(r, dict) and r.get("timestamp") is not None
        ]
        if not valid_records:
            break

        valid_records.sort(key=lambda r: int(r["timestamp"]))
        all_rows.extend(valid_records)

        last_ts = int(valid_records[-1]["timestamp"])
        if until_ms is not None and last_ts >= until_ms:
            break
        if ex == "binance" and len(records) < limit:
            break

        next_since = last_ts + 1
        if next_since <= curr_since:
            break
        curr_since = next_since

    # 4. Chuyển đổi và chuẩn hóa bản ghi tải mới
    if all_rows:
        parsed_rows: list[dict[str, Any]] = []
        for r in all_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            fr = r.get("fundingRate") if r.get("fundingRate") is not None else r.get("funding_rate")
            mp = r.get("markPrice") if r.get("markPrice") is not None else r.get("mark_price")
            if fr is None or pd.isna(fr):
                continue
            parsed_rows.append({
                "timestamp": int(ts),
                "funding_rate": float(fr),
                "mark_price": float(mp) if mp is not None and not pd.isna(mp) else np.nan,
            })
        if parsed_rows:
            new_df = pd.DataFrame(parsed_rows)
            new_df = _ensure_funding_schema(new_df).dropna(subset=["funding_rate"])
        else:
            new_df = _empty_funding_frame()
    else:
        new_df = _empty_funding_frame()

    # 5. Gộp với dữ liệu cache cũ
    parts: list[pd.DataFrame] = []
    if cached_df is not None and not cached_df.empty:
        parts.append(cached_df)
    if not new_df.empty:
        parts.append(new_df)

    if parts:
        final_df = pd.concat(parts, axis=0, ignore_index=True)
        final_df = final_df.dropna(subset=["funding_rate"])
        final_df = _ensure_funding_schema(final_df)
    else:
        final_df = _empty_funding_frame()

    # 6. Ghi cache Parquet vào đường dẫn chuẩn mới
    if not final_df.empty:
        try:
            primary_cache_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_parquet(primary_cache_path, index=False)
        except Exception as err:
            raise IOError(
                f"Không thể ghi file cache Parquet tại '{primary_cache_path}': {err}"
            ) from err

    return final_df


def fetch_all_exchanges_funding(
    symbols: Sequence[str] = CORE_4_SYMBOLS,
    exchanges: Sequence[str] = SUPPORTED_EXCHANGES,
    start_date: str = "2023-05-07",
    end_date: str | None = "2026-10-18",
    data_dir: str | Path = "./data/raw",
    exchange_instances: dict[str, Any] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Tải và lưu trữ cache Funding Rate cho danh sách symbol và sàn giao dịch.

    Tham số:
        symbols: Danh sách cặp giao dịch (mặc định CORE_4_SYMBOLS).
        exchanges: Danh sách sàn cần tải (mặc định SUPPORTED_EXCHANGES).
        start_date: Ngày bắt đầu (UTC).
        end_date: Ngày kết thúc (UTC).
        data_dir: Thư mục lưu trữ cache Parquet.
        exchange_instances: Dict map {exchange_name: ccxt_instance} hoặc None.

    Trả về:
        Dict lồng định dạng: {symbol: {exchange_name: DataFrame}}
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        out[sym] = {}
        for ex in exchanges:
            inst = exchange_instances.get(ex) if exchange_instances else None
            df = fetch_exchange_funding_history(
                exchange_name=ex,
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                data_dir=data_dir,
                exchange=inst,
            )
            out[sym][ex] = df
    return out


def load_funding_by_exchange(
    symbol: str,
    exchanges: Sequence[str] = SUPPORTED_EXCHANGES,
    data_dir: str | Path = "./data/raw",
) -> dict[str, pd.DataFrame]:
    """Chỉ đọc cache Parquet cục bộ cho một symbol trên các sàn giao dịch, không gọi mạng.

    Tham số:
        symbol: Ký hiệu cặp giao dịch (ví dụ 'BTC/USDT:USDT').
        exchanges: Danh sách sàn cần nạp (mặc định SUPPORTED_EXCHANGES).
        data_dir: Thư mục chứa file cache Parquet.

    Trả về:
        Dict map {exchange_name: DataFrame}. Sàn nào thiếu cache thì không có key trong dict (hoặc trả {}).
    """
    out: dict[str, pd.DataFrame] = {}
    for ex in exchanges:
        found_path = _find_existing_cache_file(ex, symbol, data_dir)
        if found_path is not None:
            try:
                raw_df = pd.read_parquet(found_path)
                if not raw_df.empty and "timestamp" in raw_df.columns:
                    cleaned_df = _ensure_funding_schema(raw_df).dropna(subset=["funding_rate"])
                    if not cleaned_df.empty:
                        out[ex] = cleaned_df
            except Exception:
                pass
    return out


def align_multi_exchange_funding(
    symbol: str,
    ohlcv_spot_perp_df: pd.DataFrame,
    funding_by_exchange: dict[str, pd.DataFrame],
    ewma_span: int = 6,
) -> pd.DataFrame:
    """Đồng bộ dữ liệu OHLCV 1h với Funding Rate đa sàn, loại bỏ hoàn toàn look-ahead bias.

    Tham số:
        symbol: Ký hiệu cặp giao dịch (ví dụ 'BTC/USDT:USDT').
        ohlcv_spot_perp_df: DataFrame OHLCV có DatetimeIndex UTC.
        funding_by_exchange: Dict map {exchange_name: DataFrame_funding_rate}.
        ewma_span: Số kỳ tính EWMA funding rate (mặc định 6 kỳ 8h = 48h).

    Trả về:
        DataFrame đồng bộ trên DatetimeIndex UTC ns tên 'datetime' với các cột bổ sung:
        - funding_event_{ex}: Funding rate thực thanh toán tại mốc 8h UTC (00, 08, 16), giờ khác = 0.0 (chưa niêm yết = NaN).
        - last_funding_{ex}: Funding rate đã chốt gần nhất tính đến thời điểm <= t (pd.merge_asof backward).
        - ewma_funding_{ex}: EWMA 8h reindex về 1h, ffill và bắt buộc .shift(1) (chưa có dữ liệu = NaN).
        - best_short_ex: Sàn có EWMA cao nhất tại t-1 (tie-break binance > bybit > okx, bỏ NaN).
        - best_long_ex: Sàn có EWMA thấp nhất tại t-1 (tie-break binance > bybit > okx, bỏ NaN).
        - max_single_funding: max_{ex}(ewma_funding_{ex}) tại t-1.
        - min_single_funding: min_{ex}(ewma_funding_{ex}) tại t-1.
        - cross_ex_spread: max_single_funding - min_single_funding tại t-1 (<2 sàn hợp lệ = 0.0).
    """
    out = _to_datetime_indexed(ohlcv_spot_perp_df)
    target_index = out.index
    n_bars = len(target_index)

    # Xử lý trường hợp DataFrame đầu vào rỗng
    if n_bars == 0:
        for ex in SUPPORTED_EXCHANGES:
            out[f"funding_event_{ex}"] = pd.Series([], dtype="float64", index=target_index)
            out[f"last_funding_{ex}"] = pd.Series([], dtype="float64", index=target_index)
            out[f"ewma_funding_{ex}"] = pd.Series([], dtype="float64", index=target_index)
        out["best_short_ex"] = pd.Series([], dtype="object", index=target_index)
        out["best_long_ex"] = pd.Series([], dtype="object", index=target_index)
        out["max_single_funding"] = pd.Series([], dtype="float64", index=target_index)
        out["min_single_funding"] = pd.Series([], dtype="float64", index=target_index)
        out["cross_ex_spread"] = pd.Series([], dtype="float64", index=target_index)
        out.attrs["symbol"] = symbol
        return out

    ewma_arrays: list[np.ndarray] = []

    for ex in SUPPORTED_EXCHANGES:
        f_df = funding_by_exchange.get(ex) if isinstance(funding_by_exchange, dict) else None

        if f_df is None or not isinstance(f_df, pd.DataFrame) or f_df.empty:
            out[f"funding_event_{ex}"] = np.nan
            out[f"last_funding_{ex}"] = np.nan
            out[f"ewma_funding_{ex}"] = np.nan
            ewma_arrays.append(np.full(n_bars, np.nan, dtype=np.float64))
            continue

        fd = f_df.copy()
        if "datetime" not in fd.columns:
            if isinstance(fd.index, pd.DatetimeIndex):
                fd["datetime"] = fd.index
            elif "timestamp" in fd.columns:
                fd["datetime"] = pd.to_datetime(fd["timestamp"], unit="ms", utc=True)
            else:
                out[f"funding_event_{ex}"] = np.nan
                out[f"last_funding_{ex}"] = np.nan
                out[f"ewma_funding_{ex}"] = np.nan
                ewma_arrays.append(np.full(n_bars, np.nan, dtype=np.float64))
                continue

        fd["datetime"] = pd.to_datetime(fd["datetime"], utc=True).dt.as_unit("ns")
        fd["funding_rate"] = fd["funding_rate"].astype("float64")
        fd = fd.dropna(subset=["datetime", "funding_rate"])

        if fd.empty:
            out[f"funding_event_{ex}"] = np.nan
            out[f"last_funding_{ex}"] = np.nan
            out[f"ewma_funding_{ex}"] = np.nan
            ewma_arrays.append(np.full(n_bars, np.nan, dtype=np.float64))
            continue

        # Làm tròn về 1h gần nhất và map các kỳ trung gian (4h/1h) về đúng mốc 8h UTC (00, 08, 16)
        dt_round = fd["datetime"].dt.round("1h")
        h = dt_round.dt.hour
        days_add = np.where(h > 16, 1, 0)
        target_hour = np.where(h == 0, 0, np.where(h <= 8, 8, np.where(h <= 16, 16, 0)))

        base_date = dt_round.dt.floor("1D")
        mapped_dt = base_date + pd.to_timedelta(days_add, unit="D") + pd.to_timedelta(target_hour, unit="h")
        mapped_dt = pd.DatetimeIndex(mapped_dt, tz="UTC").as_unit("ns")

        fd["_target_8h"] = mapped_dt
        grouped_8h = fd.groupby("_target_8h")["funding_rate"].sum().sort_index()
        grouped_8h.index = pd.DatetimeIndex(grouped_8h.index, tz="UTC").as_unit("ns")

        first_funding_ts = grouped_8h.index.min()

        # 1. funding_event_{ex}:
        # Sàn chưa niêm yết (t < first_funding_ts) giữ np.nan.
        # Khi đã niêm yết: mốc 8h lấy giá trị funding thực tế, các giờ khác = 0.0.
        fe_series = pd.Series(np.nan, index=target_index, dtype="float64")
        listed_mask = target_index >= first_funding_ts
        fe_series[listed_mask] = 0.0

        mask_8h = listed_mask & target_index.hour.isin([0, 8, 16]) & (target_index.minute == 0)
        idx_8h = target_index[mask_8h]
        if len(idx_8h) > 0:
            fe_values = grouped_8h.reindex(idx_8h).fillna(0.0).to_numpy(dtype=np.float64)
            fe_series.loc[idx_8h] = fe_values

        out[f"funding_event_{ex}"] = fe_series

        # 2. last_funding_{ex}:
        # Khớp backward merge_asof trên chuỗi sự kiện 8h
        df_left = pd.DataFrame({"_dt": target_index})
        df_right = pd.DataFrame({
            "_dt": grouped_8h.index,
            "_last_fr": grouped_8h.to_numpy(dtype=np.float64),
        }).sort_values(by="_dt")
        merged_lf = pd.merge_asof(df_left, df_right, on="_dt", direction="backward")
        out[f"last_funding_{ex}"] = pd.Series(
            merged_lf["_last_fr"].to_numpy(dtype=np.float64),
            index=target_index,
        )

        # 3. ewma_funding_{ex}:
        # EWMA tính trên chuỗi sự kiện 8h, reindex về 1h, ffill, BẮT BUỘC .shift(1)
        ewma_8h = grouped_8h.ewm(span=ewma_span, adjust=True).mean()
        combined_idx = target_index.union(grouped_8h.index).sort_values()
        ewma_combined = ewma_8h.reindex(combined_idx).ffill()
        ewma_1h = ewma_combined.reindex(target_index)
        ewma_shifted = ewma_1h.shift(1).astype("float64")
        out[f"ewma_funding_{ex}"] = ewma_shifted
        ewma_arrays.append(ewma_shifted.to_numpy(dtype=np.float64))

    # 4. Tính toán các cột tổng hợp đa sàn tại mỗi nến t (dựa hoàn toàn trên EWMA t-1)
    vals = np.column_stack(ewma_arrays)
    is_valid = np.isfinite(vals)
    valid_counts = is_valid.sum(axis=1)

    max_single = np.full(n_bars, np.nan, dtype=np.float64)
    min_single = np.full(n_bars, np.nan, dtype=np.float64)
    cross_spread = np.zeros(n_bars, dtype=np.float64)
    best_short_list: list[str | None] = [None] * n_bars
    best_long_list: list[str | None] = [None] * n_bars

    has_any_valid = valid_counts > 0
    has_multi_valid = valid_counts >= 2

    if np.any(has_any_valid):
        # argmax và argmin trên mảng thay thế -inf/+inf tự động tie-break theo thứ tự cột 0 > 1 > 2
        # khớp chuẩn: binance (0) > bybit (1) > okx (2)
        vals_for_max = np.where(is_valid, vals, -np.inf)
        vals_for_min = np.where(is_valid, vals, np.inf)

        max_indices = np.argmax(vals_for_max, axis=1)
        min_indices = np.argmin(vals_for_min, axis=1)

        row_indices = np.arange(n_bars)
        max_single[has_any_valid] = vals[row_indices[has_any_valid], max_indices[has_any_valid]]
        min_single[has_any_valid] = vals[row_indices[has_any_valid], min_indices[has_any_valid]]

        # cross_ex_spread = max - min; nếu < 2 sàn hợp lệ thì giữ 0.0
        cross_spread[has_multi_valid] = max_single[has_multi_valid] - min_single[has_multi_valid]

        for i in np.where(has_any_valid)[0]:
            best_short_list[i] = SUPPORTED_EXCHANGES[max_indices[i]]
            best_long_list[i] = SUPPORTED_EXCHANGES[min_indices[i]]

    out["best_short_ex"] = pd.Series(best_short_list, index=target_index, dtype="object")
    out["best_long_ex"] = pd.Series(best_long_list, index=target_index, dtype="object")
    out["max_single_funding"] = pd.Series(max_single, index=target_index, dtype="float64")
    out["min_single_funding"] = pd.Series(min_single, index=target_index, dtype="float64")
    out["cross_ex_spread"] = pd.Series(cross_spread, index=target_index, dtype="float64")

    out.attrs["symbol"] = symbol
    return out
