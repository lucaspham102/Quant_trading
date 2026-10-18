import pandas as pd
import pytest

from src.config import TradingConfig
from src.data_loader import (
    DataLoader,
    drop_incomplete_last_candle,
    merge_and_deduplicate_ohlcv,
    raw_ohlcv_to_dataframe,
)

TF_MS = 15 * 60 * 1000
T0 = 1_700_000_100_000 - (1_700_000_100_000 % TF_MS)  # căn theo nến 15m


def make_row(i: int, base: float = 100.0) -> list[float]:
    ts = T0 + i * TF_MS
    p = base + i
    return [ts, p, p + 2.0, p - 2.0, p + 1.0, 10.0 + i]


class FakeExchange:
    """Exchange giả: trả dữ liệu từ danh sách nến có sẵn, không gọi mạng."""

    def __init__(self, rows, now_ms):
        self.rows = rows
        self.now_ms = now_ms
        self.calls = []

    def milliseconds(self):
        return self.now_ms

    def fetch_ohlcv(self, symbol, timeframe="15m", since=None, limit=None, params=None):
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
        out = [r for r in self.rows if since is None or r[0] >= since]
        return [list(r) for r in out[:limit]]


def test_raw_ohlcv_to_dataframe_schema_and_dedup():
    rows = [make_row(2), make_row(0), make_row(1), make_row(1, base=500.0)]
    df = raw_ohlcv_to_dataframe(rows, "BTC/USDT:USDT", "15m")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert all(dt == "float64" for dt in df.dtypes)
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) == 3
    assert df["open"].iloc[1] == 501.0  # keep='last'
    assert df.attrs["symbol"] == "BTC/USDT:USDT" and df.attrs["timeframe"] == "15m"


def test_raw_ohlcv_to_dataframe_empty():
    df = raw_ohlcv_to_dataframe([])
    assert len(df) == 0
    assert str(df.index.tz) == "UTC"
    assert all(dt == "float64" for dt in df.dtypes)


def test_drop_incomplete_last_candle():
    df = raw_ohlcv_to_dataframe([make_row(0), make_row(1), make_row(2)])
    # nến 2 đóng tại T0 + 3*TF; server time trước đó -> chưa đóng
    cut = drop_incomplete_last_candle(df, "15m", T0 + 3 * TF_MS - 1)
    assert len(cut) == 2
    assert cut.index[-1] == df.index[1]
    # đúng bằng thời điểm đóng -> đã đóng, giữ nguyên
    kept = drop_incomplete_last_candle(df, "15m", T0 + 3 * TF_MS)
    assert len(kept) == 3
    assert len(df) == 3  # không sửa df gốc


def test_merge_and_deduplicate():
    a = raw_ohlcv_to_dataframe([make_row(0), make_row(1), make_row(2)])
    b = raw_ohlcv_to_dataframe([make_row(2, base=900.0), make_row(3), make_row(4)])
    merged = merge_and_deduplicate_ohlcv(a, b)
    assert len(merged) == 5
    assert merged.index.is_unique and merged.index.is_monotonic_increasing
    assert merged["open"].iloc[2] == 902.0  # bản mới thắng
    assert str(merged.index.tz) == "UTC"
    assert all(dt == "float64" for dt in merged.dtypes)


def test_fetch_pagination_cursor_advances():
    rows = [make_row(i) for i in range(2500)]
    now = T0 + 2600 * TF_MS
    ex = FakeExchange(rows, now)
    loader = DataLoader(TradingConfig(), exchange=ex)
    df = loader.fetch_historical_ohlcv(T0, limit_per_call=1000)
    assert len(df) == 2500
    assert df.index.is_unique and df.index.is_monotonic_increasing
    sinces = [c["since"] for c in ex.calls]
    assert sinces[:3] == [T0, T0 + 1000 * TF_MS, T0 + 2000 * TF_MS]
    assert all(c["limit"] == 1000 for c in ex.calls)
    assert all(c["symbol"] == "BTC/USDT:USDT" for c in ex.calls)


def test_fetch_drops_incomplete_last_candle_and_respects_until():
    rows = [make_row(i) for i in range(10)]
    # server time nằm giữa nến 9 -> nến 9 chưa đóng
    ex = FakeExchange(rows, T0 + 9 * TF_MS + 1000)
    loader = DataLoader(TradingConfig(), exchange=ex)
    df = loader.fetch_historical_ohlcv(T0)
    assert len(df) == 9
    assert int(df.index[-1].timestamp() * 1000) == T0 + 8 * TF_MS

    ex2 = FakeExchange(rows, T0 + 100 * TF_MS)
    df2 = DataLoader(TradingConfig(), exchange=ex2).fetch_historical_ohlcv(
        T0, until_ms=T0 + 4 * TF_MS
    )
    assert len(df2) == 5


def test_fetch_empty_exchange():
    ex = FakeExchange([], T0)
    df = DataLoader(TradingConfig(), exchange=ex).fetch_historical_ohlcv(T0 - TF_MS)
    assert len(df) == 0
    assert len(ex.calls) == 1


def test_parquet_roundtrip_preserves_utc_and_float64(tmp_path):
    loader = DataLoader(TradingConfig(), exchange=FakeExchange([], T0))
    df = raw_ohlcv_to_dataframe([make_row(i) for i in range(5)])
    path = loader.save_parquet(df, tmp_path / "sub" / "x.parquet")
    assert path.exists()
    back = loader.load_parquet(path)
    assert str(back.index.tz) == "UTC"
    assert back.index.name == "timestamp"
    assert all(dt == "float64" for dt in back.dtypes)
    assert list(back.columns) == list(df.columns)
    pd.testing.assert_frame_equal(back, df, check_dtype=True, check_freq=False)


def test_update_incremental_appends_from_last_timestamp(tmp_path):
    path = tmp_path / "btc.parquet"
    all_rows = [make_row(i) for i in range(20)]
    ex = FakeExchange(all_rows[:10], T0 + 10 * TF_MS)
    loader = DataLoader(TradingConfig(), exchange=ex)
    first = loader.update_incremental(path, default_start_ms=T0)
    assert len(first) == 10
    assert path.exists()

    ex.rows = all_rows
    ex.now_ms = T0 + 20 * TF_MS
    ex.calls.clear()
    second = loader.update_incremental(path, default_start_ms=T0)
    assert ex.calls[0]["since"] == T0 + 9 * TF_MS  # từ timestamp cuối của file
    assert len(second) == 20
    assert second.index.is_unique and second.index.is_monotonic_increasing
    assert str(second.index.tz) == "UTC"
    assert all(dt == "float64" for dt in second.dtypes)
    assert len(loader.load_parquet(path)) == 20


def test_update_incremental_requires_start_when_no_file(tmp_path):
    loader = DataLoader(TradingConfig(), exchange=FakeExchange([], T0))
    with pytest.raises(ValueError):
        loader.update_incremental(tmp_path / "missing.parquet")
