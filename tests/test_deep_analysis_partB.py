"""Bộ kiểm thử cho Sprint 5 Phần B (offline, synthetic, deterministic).

Kiểm tra các chức năng mở rộng cho khung thời gian 4h & 1d:
- resample_ohlcv sang 4h và 1d từ 1h.
- sharpe_of cho chuỗi equity tăng đơn điệu và phẳng.
- analyze_timeframe tích hợp đủ 6 bảng phân tích bao gồm khoá sharpe.
- holdout_return trên tập holdout 180 ngày cuối.
- CLI run_sprint5_partB xuất 6 file CSV và 1 file Markdown.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_sprint5_partB
from src.deep_analysis import (
    analyze_timeframe,
    holdout_return,
    resample_ohlcv,
    sharpe_of,
)
from src.research import generate_synthetic_ohlcv


def test_resample_ohlcv_4h():
    """Kiểm tra resample OHLCV 1h sang 4h bảo toàn đúng quy tắc open/high/low/close/volume."""
    index = pd.date_range("2024-01-01 00:00:00", periods=8, freq="1h", tz="UTC").as_unit("ns")
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(8)],
            "high": [105.0 + i for i in range(8)],
            "low": [95.0 + i for i in range(8)],
            "close": [101.0 + i for i in range(8)],
            "volume": [10.0 * (i + 1) for i in range(8)],
        },
        index=index,
    )
    res = resample_ohlcv(df, "4h")
    assert len(res) == 2
    assert res["open"].iloc[0] == df["open"].iloc[0]
    assert res["high"].iloc[0] == df["high"].iloc[:4].max()
    assert res["close"].iloc[0] == df["close"].iloc[3]
    assert res["volume"].iloc[0] == df["volume"].iloc[:4].sum()
    assert res.index.unit == "ns"

    with pytest.raises(ValueError, match="Khung thời gian không hợp lệ"):
        resample_ohlcv(df, "15m")


def test_sharpe_of_flat_and_rising():
    """Kiểm tra tính Sharpe ratio cho chuỗi equity tăng đơn điệu và chuỗi equity phẳng."""
    rising = pd.Series([100.0, 102.0, 105.0, 109.0, 114.0])
    sh_rising = sharpe_of(rising, candles_per_year=2190)
    assert sh_rising > 0.0

    flat = pd.Series([100.0, 100.0, 100.0, 100.0])
    sh_flat = sharpe_of(flat, candles_per_year=2190)
    assert sh_flat == 0.0

    short = pd.Series([100.0])
    assert sharpe_of(short, candles_per_year=2190) == 0.0


def test_analyze_timeframe_has_sharpe_key():
    """Kiểm tra analyze_timeframe trả về đầy đủ 6 bảng phân tích bao gồm khoá sharpe."""
    df = generate_synthetic_ohlcv(n_candles=900 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    d4 = resample_ohlcv(df, "4h")
    out = analyze_timeframe("BTC/USDT:USDT", "4h", d4, n_boot=50, n_null=50)

    assert set(out.keys()) >= {"exposure", "drawdown", "alpha_beta", "bootstrap", "null", "sharpe"}
    for key in {"exposure", "drawdown", "alpha_beta", "bootstrap", "null", "sharpe"}:
        assert isinstance(out[key], pd.DataFrame)
        assert not out[key].empty

    sharpe_cols = set(out["sharpe"].columns)
    assert {"base_sharpe", "bh_sharpe", "base_minus_bh_sharpe"}.issubset(sharpe_cols)


def test_holdout_return_dict():
    """Kiểm tra holdout_return trả về dict chứa đầy đủ các chỉ số hiệu suất tập holdout."""
    df = generate_synthetic_ohlcv(n_candles=420 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    ctx = resample_ohlcv(df, "1d")
    cutoff = ctx.index[-1] - pd.Timedelta(days=180)
    r = holdout_return("BTC/USDT:USDT", "1d", ctx, cutoff)

    assert isinstance(r, dict)
    assert "baseline_return_pct" in r
    assert "bh_return_pct" in r
    assert "baseline_sharpe" in r
    assert r["n_bars"] > 0
    assert r["holdout_start"] < r["holdout_end"]


def test_cli_partB_writes_outputs(tmp_path, monkeypatch):
    """Kiểm tra CLI runner run_sprint5_partB tạo đúng 6 file CSV và 1 file Markdown."""
    def fake_load_symbol_data(symbol, timeframe="1h", days=1260, **kwargs):
        return generate_synthetic_ohlcv(n_candles=1000 * 24, symbol=symbol, timeframe="1h")

    monkeypatch.setattr("run_sprint5_partB.load_or_fetch_symbol_data", fake_load_symbol_data)
    monkeypatch.setattr("src.deep_analysis.load_or_fetch_symbol_data", fake_load_symbol_data)

    out_md = tmp_path / "b.md"
    ret = run_sprint5_partB.main(
        [
            "--symbols",
            "BTC/USDT:USDT",
            "--days",
            "1000",
            "--n-boot",
            "30",
            "--n-null",
            "30",
            "--out-dir",
            str(tmp_path),
            "--out-md",
            str(out_md),
        ]
    )

    assert ret == 0

    expected_csvs = [
        "sprint5_partB_alpha_beta.csv",
        "sprint5_partB_bootstrap.csv",
        "sprint5_partB_null.csv",
        "sprint5_partB_exposure.csv",
        "sprint5_partB_sharpe.csv",
        "sprint5_partB_holdout.csv",
    ]
    for csv_name in expected_csvs:
        csv_file = tmp_path / csv_name
        assert csv_file.exists(), f"Thiếu file CSV: {csv_name}"
        assert csv_file.stat().st_size > 0

    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "# Sprint 5 — Phần B (4h & 1d)" in content
    assert "Số lần thử Phần B = 2 timeframe × 8 config × 4 symbol = 64." in content
