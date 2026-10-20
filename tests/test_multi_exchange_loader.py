"""Unit test offline cho module multi_exchange_loader.

Toàn bộ test đều tất định, không gọi mạng/API thật và không đọc ``data/raw`` thật.
Dữ liệu funding/OHLCV được tổng hợp trong bộ nhớ; cache Parquet chỉ ghi vào
``tmp_path``. Các hành vi được khóa:

- Phân trang funding Bybit (limit 200), chống trùng timestamp và tự sắp xếp.
- Tên file cache chuẩn ``{exchange}_{CLEAN}_funding.parquet`` và cache hit.
- Chống look-ahead tuyệt đối bằng ``ewma_funding_*`` đã ``shift(1)``.
- Sàn chưa niêm yết (OKX rỗng) giữ NaN, không bị điền 0.0 giả tạo.
- Các nhánh biên: symbol chuẩn hóa, sàn không hỗ trợ, OHLCV rỗng.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.funding_loader import _empty_funding_frame
from src.multi_exchange_loader import (
    SUPPORTED_EXCHANGES,
    align_multi_exchange_funding,
    clean_symbol_for_Attachment,
    clean_symbol_name,
    fetch_exchange_funding_history,
)

_EIGHT_HOURS_MS = 8 * 3600 * 1000


def _to_ms(ts: pd.Timestamp) -> int:
    """Đổi Timestamp (UTC) sang Unix milliseconds int64."""
    return int(ts.value // 1_000_000)


class _FakeFundingExchange:
    """Sàn giả thuần Python, không gọi mạng, dùng để kiểm soát phân trang."""

    def __init__(self, records: list[dict[str, Any]], now_ms: int) -> None:
        self.records = [dict(row) for row in records]
        self.now_ms = int(now_ms)
        self.fetch_calls = 0
        self.milliseconds_calls = 0
        self.call_history: list[dict[str, Any]] = []

    def milliseconds(self) -> int:
        """Trả mốc thời gian hiện tại cố định của sàn giả."""
        self.milliseconds_calls += 1
        return self.now_ms

    def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Trả một page funding đã cắt theo ``since``/``limit`` nhưng cố ý đảo thứ tự."""
        self.fetch_calls += 1
        self.call_history.append(
            {
                "symbol": symbol,
                "since": since,
                "limit": limit,
                "params": params,
            }
        )

        rows = [
            row
            for row in self.records
            if since is None or int(row["timestamp"]) >= int(since)
        ]
        rows.sort(key=lambda row: int(row["timestamp"]))
        if limit is not None:
            rows = rows[: int(limit)]

        # Page trả về bị đảo ngược để buộc loader phải tự sort lại.
        return [dict(row) for row in reversed(rows)]


def _funding_records(index: pd.DatetimeIndex, rate: float = 0.0001) -> list[dict[str, Any]]:
    """Sinh danh sách bản ghi funding giả theo từng mốc trong ``index``."""
    return [
        {
            "timestamp": _to_ms(ts),
            "fundingRate": float(rate),
            "markPrice": 100.0,
        }
        for ts in index
    ]


def _hourly_ohlcv(periods: int = 120, start: str = "2024-01-01") -> pd.DataFrame:
    """Tạo OHLCV 1h UTC giả với DatetimeIndex ns tên ``datetime``."""
    index = pd.date_range(start, periods=periods, freq="1h", tz="UTC").as_unit("ns")
    index.name = "datetime"
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        },
        index=index,
    )
    return frame.astype("float64")


def _funding_frame(
    start: str,
    end: str,
    default_rate: float,
    overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Tạo DataFrame funding 8h chuẩn schema, cho phép ghi đè một số mốc."""
    index = pd.date_range(start, end, freq="8h", tz="UTC").as_unit("ns")
    rates = np.full(len(index), float(default_rate), dtype="float64")

    for ts_text, rate in (overrides or {}).items():
        target = pd.Timestamp(ts_text, tz="UTC")
        mask = index == target
        assert mask.any(), f"Mốc funding override {ts_text} không nằm trong dải dữ liệu"
        rates[mask] = float(rate)

    return pd.DataFrame(
        {
            "timestamp": (index.asi8 // 1_000_000).astype("int64"),
            "datetime": index,
            "funding_rate": rates,
            "mark_price": np.full(len(index), 100.0, dtype="float64"),
        }
    )


# ---------------------------------------------------------------------------
# 1. Phân trang + lưu cache Parquet + cache hit
# ---------------------------------------------------------------------------


def test_fetch_exchange_funding_history_pagination_and_cache(tmp_path: Path) -> None:
    """Bybit giả 255 bản ghi (5 trùng, page đảo thứ tự): phân trang 2 lần, cache sạch và hit không gọi mạng."""
    start = pd.Timestamp("2023-01-01", tz="UTC")
    index = pd.date_range(start, periods=250, freq="8h", tz="UTC").as_unit("ns")
    unique_until = index[-1]

    records = _funding_records(index)
    for position in (0, 1, 100, 150, 249):
        records.append(dict(records[position]))

    fake = _FakeFundingExchange(
        records,
        now_ms=_to_ms(unique_until) + 86_400_000,
    )

    frame = fetch_exchange_funding_history(
        exchange_name="bybit",
        symbol="BTC/USDT:USDT",
        start_date="2023-01-01",
        end_date="2023-03-25",
        data_dir=tmp_path,
        exchange=fake,
    )

    cache_path = tmp_path / "bybit_BTCUSDT_funding.parquet"
    assert cache_path.exists()
    assert [path.name for path in tmp_path.iterdir()] == [cache_path.name]

    assert len(frame) == 250
    assert frame["timestamp"].is_unique
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["funding_rate"].notna().all()
    assert str(frame["datetime"].dtype) == "datetime64[ns, UTC]"

    # Bybit limit 200: 255 bản ghi thô phải cần đúng 2 lần gọi.
    assert fake.fetch_calls == 2
    assert all(call["limit"] == 200 for call in fake.call_history)
    assert fake.call_history[0]["params"] is not None
    assert fake.call_history[0]["params"]["startTime"] == _to_ms(start)
    assert fake.call_history[1]["since"] > fake.call_history[0]["since"]

    on_disk = pd.read_parquet(cache_path)
    assert len(on_disk) == 250
    assert on_disk["timestamp"].is_unique
    assert on_disk["timestamp"].is_monotonic_increasing
    assert str(on_disk["datetime"].dtype) == "datetime64[ns, UTC]"

    # Gọi lần 2 cùng dải thời gian: cache phủ trọn nên tuyệt đối không gọi mạng.
    fake_cached = _FakeFundingExchange(records, now_ms=fake.now_ms)
    cached = fetch_exchange_funding_history(
        exchange_name="bybit",
        symbol="BTC/USDT:USDT",
        start_date="2023-01-01",
        end_date="2023-03-25",
        data_dir=tmp_path,
        exchange=fake_cached,
    )

    assert fake_cached.fetch_calls == 0
    assert fake_cached.milliseconds_calls == 0
    pd.testing.assert_frame_equal(cached, frame)


# ---------------------------------------------------------------------------
# 2. Chống look-ahead tuyệt đối với cú sốc funding OKX
# ---------------------------------------------------------------------------


def test_align_multi_exchange_funding_strict_no_lookahead() -> None:
    """Cú sốc OKX tại 00:00 chỉ được dùng từ nến 01:00; EWMA tại 00:00 trùng nến 23:00 hôm trước."""
    # 120 nến (5 ngày) để phủ trọn mốc 2024-01-05 00:00 và các nến sau đó.
    ohlcv = _hourly_ohlcv(periods=120)
    binance = _funding_frame("2024-01-01", "2024-01-06 00:00", 0.0001)
    bybit = _funding_frame("2024-01-01", "2024-01-06 00:00", 0.0001)
    okx = _funding_frame(
        "2024-01-01",
        "2024-01-06 00:00",
        0.0001,
        overrides={"2024-01-05 00:00": 0.05},
    )

    aligned = align_multi_exchange_funding(
        "BTC/USDT:USDT",
        ohlcv,
        {"binance": binance, "bybit": bybit, "okx": okx},
    )

    assert aligned.index.name == "datetime"
    assert str(aligned.index.dtype) == "datetime64[ns, UTC]"

    spike = pd.Timestamp("2024-01-05 00:00", tz="UTC")
    next_hour = spike + pd.Timedelta(hours=1)

    # Funding thực thanh toán tại đúng mốc 00:00 là 0.05.
    assert aligned.loc[spike, "funding_event_okx"] == pytest.approx(0.05)
    assert aligned.loc[spike, "last_funding_okx"] == pytest.approx(0.05)

    # Nhưng quyết định tại 00:00 chỉ dùng EWMA t-1 nên OKX chưa thể thắng.
    assert aligned.loc[spike, "best_short_ex"] == "binance"
    assert aligned.loc[spike, "best_short_ex"] != "okx"
    assert aligned.loc[spike, "max_single_funding"] == pytest.approx(0.0001, abs=1e-12)

    # Từ 01:00, EWMA đã tiếp nhận cú sốc và OKX trở thành best_short.
    assert aligned.loc[next_hour:, "best_short_ex"].eq("okx").all()
    assert aligned.loc[next_hour, "max_single_funding"] > (
        10.0 * aligned.loc[spike, "max_single_funding"]
    )
    assert aligned.loc[next_hour, "cross_ex_spread"] > 0.0

    # Bằng chứng shift 1 giờ: EWMA tại 00:00 trùng đúng giá trị tại 23:00.
    previous_hour = spike - pd.Timedelta(hours=1)
    assert aligned.loc[spike, "ewma_funding_okx"] == pytest.approx(
        aligned.loc[previous_hour, "ewma_funding_okx"]
    )


# ---------------------------------------------------------------------------
# 3. Sàn chưa niêm yết / thiếu dữ liệu
# ---------------------------------------------------------------------------


def test_align_multi_exchange_funding_missing_or_late_listing_exchange() -> None:
    """OKX rỗng: 10 ngày đầu chỉ Binance/Bybit được chọn, EWMA OKX toàn NaN, spread hội tụ 0.0002."""
    ohlcv = _hourly_ohlcv(periods=240)
    binance = _funding_frame("2024-01-01", "2024-01-11 00:00", 0.0003)
    bybit = _funding_frame("2024-01-01", "2024-01-11 00:00", 0.0001)
    okx = _empty_funding_frame()

    aligned = align_multi_exchange_funding(
        "BTC/USDT:USDT",
        ohlcv,
        {"binance": binance, "bybit": bybit, "okx": okx},
    )

    first_ten_days = aligned.loc["2024-01-01 01:00":"2024-01-10 23:00"]
    assert len(first_ten_days) == 240 - 1
    assert first_ten_days["best_short_ex"].eq("binance").all()
    assert first_ten_days["best_long_ex"].eq("bybit").all()

    # OKX chưa niêm yết phải giữ NaN, tuyệt đối không bị điền 0.0 giả tạo.
    assert aligned["ewma_funding_okx"].isna().all()
    assert aligned["funding_event_okx"].isna().all()
    assert aligned["last_funding_okx"].isna().all()
    assert not aligned["ewma_funding_okx"].eq(0.0).any()
    assert "okx" not in set(aligned["best_long_ex"].dropna())
    assert "okx" not in set(aligned["best_short_ex"].dropna())

    # Sau ~5 ngày EWMA hội tụ, spread đa sàn xấp xỉ 0.0003 - 0.0001 = 0.0002.
    converged = aligned.loc["2024-01-06":, "cross_ex_spread"]
    assert len(converged) > 0
    for value in converged:
        assert value == pytest.approx(0.0002, abs=2e-5)

    assert aligned.loc["2024-01-06":, "max_single_funding"].sub(0.0003).abs().max() < 2e-5
    assert aligned.loc["2024-01-06":, "min_single_funding"].sub(0.0001).abs().max() < 2e-5


# ---------------------------------------------------------------------------
# 4. Các nhánh biên
# ---------------------------------------------------------------------------


def test_clean_symbol_name_normalizes_variants() -> None:
    """Các kiểu symbol ccxt/underscore đều về dạng liền mạch; alias Attachment giữ nguyên hành vi."""
    assert clean_symbol_name("BTC/USDT:USDT") == "BTCUSDT"
    assert clean_symbol_name("btc_usdt_usdt") == "BTCUSDT"
    assert clean_symbol_name("ETH-USDT") == "ETHUSDT"
    assert clean_symbol_name("SOL/USDT") == "SOLUSDT"
    assert clean_symbol_for_Attachment is clean_symbol_name


def test_fetch_exchange_funding_history_rejects_unsupported_exchange(tmp_path: Path) -> None:
    """Sàn ngoài SUPPORTED_EXCHANGES phải ném ValueError trước khi chạm tới exchange."""
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        fetch_exchange_funding_history(
            exchange_name="kraken",
            symbol="BTC/USDT:USDT",
            data_dir=tmp_path,
            exchange=object(),
        )


def test_align_multi_exchange_funding_empty_ohlcv_returns_schema() -> None:
    """OHLCV rỗng vẫn trả đủ cột/dtype schema, không ném ngoại lệ và giữ attrs symbol."""
    empty_index = pd.DatetimeIndex([], tz="UTC", name="datetime").as_unit("ns")
    ohlcv = pd.DataFrame(
        {
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
        },
        index=empty_index,
    )

    aligned = align_multi_exchange_funding("BTC/USDT:USDT", ohlcv, {})

    assert aligned.empty
    assert aligned.index.name == "datetime"
    for exchange_name in SUPPORTED_EXCHANGES:
        for prefix in ("funding_event", "last_funding", "ewma_funding"):
            assert f"{prefix}_{exchange_name}" in aligned.columns
    for column in (
        "best_short_ex",
        "best_long_ex",
        "max_single_funding",
        "min_single_funding",
        "cross_ex_spread",
    ):
        assert column in aligned.columns
    assert aligned["best_short_ex"].dtype == object
    assert aligned["max_single_funding"].dtype == "float64"
    assert aligned.attrs["symbol"] == "BTC/USDT:USDT"
