"""Kiểm thử Null Hypothesis Test cho Walk-Forward Engine (offline, synthetic, deterministic)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import run_null_test as cli_run_null
from src.research import generate_synthetic_ohlcv
from src.walkforward import DEFAULT_CANDIDATES, run_null_test


def test_run_null_test_synthetic_frames():
    """Kiểm tra cấu trúc DataFrame kết quả và các chỉ số null test trên dữ liệu synthetic."""
    df = generate_synthetic_ohlcv(n_candles=365 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    per_window, summaries = run_null_test(
        "BTC/USDT:USDT",
        days=365,
        candidates=DEFAULT_CANDIDATES[:2],
        n_iter=20,
        df=df,
    )

    assert not per_window.empty
    for col in ["beats_p75", "null_p75_return_pct", "chosen_percentile"]:
        assert col in per_window.columns

    assert (per_window["null_p75_return_pct"] >= per_window["null_p25_return_pct"]).all()
    assert len(summaries) == 1
    pct_beating = float(summaries["pct_windows_beating_p75"].iloc[0])
    assert 0.0 <= pct_beating <= 100.0


def test_null_test_deterministic_with_seed():
    """Kiểm tra tính tất định của kiểm định null khi cung cấp cùng seed ngẫu nhiên."""
    df = generate_synthetic_ohlcv(n_candles=365 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    per_window_1, _ = run_null_test(
        "BTC/USDT:USDT",
        days=365,
        candidates=DEFAULT_CANDIDATES[:2],
        n_iter=10,
        seed=7,
        df=df,
    )
    per_window_2, _ = run_null_test(
        "BTC/USDT:USDT",
        days=365,
        candidates=DEFAULT_CANDIDATES[:2],
        n_iter=10,
        seed=7,
        df=df,
    )

    pd.testing.assert_frame_equal(per_window_1, per_window_2)


def test_cli_null_writes_outputs(tmp_path, monkeypatch):
    """Kiểm tra CLI run_null_test xuất đúng định dạng file CSV và Markdown."""
    def mock_load(symbol, timeframe="1h", days=365, *args, **kwargs):
        return generate_synthetic_ohlcv(n_candles=days * 24, symbol=symbol, timeframe=timeframe)

    monkeypatch.setattr("src.walkforward.load_or_fetch_symbol_data", mock_load)

    out_csv = tmp_path / "n.csv"
    out_md = tmp_path / "n.md"

    ret = cli_run_null.main(
        [
            "--symbols",
            "BTC/USDT:USDT",
            "--days",
            "365",
            "--n-iter",
            "10",
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ]
    )

    assert ret == 0
    assert out_csv.exists()
    assert out_md.exists()

    df_csv = pd.read_csv(out_csv)
    assert not df_csv.empty
    assert "symbol" in df_csv.columns
    assert "beats_p75" in df_csv.columns
