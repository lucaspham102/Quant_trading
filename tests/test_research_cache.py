"""Unit test cho cơ chế cache/backfill của `src.research.load_or_fetch_symbol_data`.

Bối cảnh: hàm này vừa được sửa để BACKFILL phần ĐẦU của cache còn thiếu
(mở rộng lịch sử về quá khứ) thay vì chỉ tải tiếp phần đuôi. Bộ test dưới đây
kiểm tra 3 nhánh hành vi chính bằng một FakeExchange hoàn toàn offline:

1. Cache ngắn hơn `days` -> tự động fetch bù phần đầu và gộp đủ số nến.
2. Cache đã phủ đủ cả đầu lẫn đuôi -> KHÔNG gọi mạng.
3. Không backfill được (sàn không có dữ liệu cũ) -> raise RuntimeError, KHÔNG fallback synthetic.
"""
import pytest
from dataclasses import replace
from pathlib import Path

from src.config import TradingConfig
from src.data_loader import (
    DataLoader,
    raw_ohlcv_to_dataframe,
    timeframe_to_ms,
)
from src.research import load_or_fetch_symbol_data

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"

# Độ dài 1 nến 1h tính bằng mili-giây.
TF_MS = timeframe_to_ms(TIMEFRAME)

# Mốc "now" cố định được căn tròn theo nến 1h để tránh lệch nửa nến.
NOW_MS = (1_700_000_000_000 // TF_MS) * TF_MS


class FakeExchange:
    """Exchange giả: trả dữ liệu từ danh sách nến có sẵn, ghi lại mọi lời gọi mạng."""

    def __init__(self, rows, now_ms):
        self.rows = rows
        self.now_ms = now_ms
        self.calls = []

    def milliseconds(self):
        """Trả về mốc thời gian server giả lập (ms)."""
        return self.now_ms

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=None, params=None):
        """Lọc nến theo `since`, ghi log lời gọi, trả tối đa `limit` bản ghi."""
        self.calls.append({"since": since, "limit": limit})
        out = [r for r in self.rows if since is None or r[0] >= since]
        return [list(r) for r in out[:limit]]


def _make_rows(n, now_ms, base=100.0):
    """Sinh `n` nến 1h liên tục kết thúc ngay trước `now_ms` (nến cuối đã đóng).

    Nến cuối có timestamp mở là `now_aligned - TF_MS`, các nến trước đó lùi dần
    theo bước `TF_MS`, đảm bảo index liên tục và có thể dùng để kiểm tra độ phủ.
    """
    now_aligned = (now_ms // TF_MS) * TF_MS
    last_ts = now_aligned - TF_MS
    start_ts = last_ts - (n - 1) * TF_MS
    return [
        [start_ts + i * TF_MS, base + i, base + i + 2.0, base + i - 2.0, base + i + 1.0, 10.0 + i]
        for i in range(n)
    ]


def _loader(tmp_path, exchange):
    """Tạo DataLoader cấu hình đúng symbol/timeframe và raw_data_dir tạm."""
    cfg = replace(
        TradingConfig(),
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        raw_data_dir=Path(tmp_path),
    )
    return DataLoader(cfg, exchange=exchange)


def test_backfill_when_cache_covers_less_than_requested(tmp_path):
    """Cache chỉ có 60 ngày -> phải fetch bù 120 ngày đầu để phủ đủ 180 ngày.

    Kỳ vọng: kết quả đúng 180*24 nến, index UTC không trùng và tăng dần,
    đồng thời có ít nhất một lời gọi mạng tới exchange.
    """
    rows180 = _make_rows(180 * 24, NOW_MS)
    ex = FakeExchange(rows180, NOW_MS)
    loader = _loader(tmp_path, ex)

    # Ghi cache chỉ gồm 60 ngày cuối (nến cuối vẫn đã đóng sát NOW_MS).
    cache_df = raw_ohlcv_to_dataframe(rows180[-60 * 24 :], SYMBOL, TIMEFRAME)
    loader.save_parquet(cache_df, loader.default_file_path())

    ex.calls.clear()
    out = load_or_fetch_symbol_data(
        SYMBOL, TIMEFRAME, days=180, raw_data_dir=tmp_path, exchange=ex
    )

    assert len(out) == 180 * 24
    assert out.index.is_unique
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == "UTC"
    assert str(out.index.dtype) == "datetime64[ns, UTC]"
    assert len(ex.calls) >= 1  # đã gọi mạng để fetch bù phần đầu


def test_no_fetch_when_cache_already_covers(tmp_path):
    """Cache đã phủ đủ 180 ngày -> trả cache luôn, KHÔNG gọi mạng."""
    rows180 = _make_rows(180 * 24, NOW_MS)
    ex = FakeExchange(rows180, NOW_MS)
    loader = _loader(tmp_path, ex)

    full_df = raw_ohlcv_to_dataframe(rows180, SYMBOL, TIMEFRAME)
    loader.save_parquet(full_df, loader.default_file_path())

    ex.calls.clear()
    out = load_or_fetch_symbol_data(
        SYMBOL, TIMEFRAME, days=180, raw_data_dir=tmp_path, exchange=ex
    )

    assert len(ex.calls) == 0  # cache đủ -> không chạm tới mạng
    assert len(out) == 180 * 24


def test_backfill_failure_raises_runtimeerror(tmp_path):
    """Chỉ có 60 ngày và sàn không có dữ liệu cũ hơn -> raise RuntimeError.

    Đảm bảo hàm KHÔNG âm thầm trả về DataFrame synthetic khi backfill thất bại.
    """
    rows60 = _make_rows(60 * 24, NOW_MS)
    ex = FakeExchange(rows60, NOW_MS)
    loader = _loader(tmp_path, ex)

    cache_df = raw_ohlcv_to_dataframe(rows60, SYMBOL, TIMEFRAME)
    loader.save_parquet(cache_df, loader.default_file_path())

    with pytest.raises(RuntimeError):
        load_or_fetch_symbol_data(
            SYMBOL, TIMEFRAME, days=180, raw_data_dir=tmp_path, exchange=ex
        )
