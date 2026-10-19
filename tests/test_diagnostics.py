"""Kiểm thử Diagnostics Engine: bóc tách alpha vs beta, chi phí và CLI."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import run_diagnostics
from src.diagnostics import (
    REAL_FEE_PER_SIDE_BH,
    buy_and_hold_return_pct,
    buy_and_hold_sharpe,
    run_diagnostics as run_diagnostics_fn,
)
from src.research import generate_synthetic_ohlcv


def test_buy_and_hold_return_and_sharpe():
    """Kiểm tra tính toán lợi suất Buy & Hold và tỷ số Sharpe của B&H."""
    open_prices = pd.Series([100.0, 101.0, 102.0])
    close_prices = pd.Series([101.0, 102.0, 103.0])

    ret_zero = buy_and_hold_return_pct(open_prices, close_prices, fee_per_side=0.0)
    ret_fee = buy_and_hold_return_pct(open_prices, close_prices, fee_per_side=REAL_FEE_PER_SIDE_BH)

    assert ret_zero > 0.0
    assert ret_zero > ret_fee

    sh = buy_and_hold_sharpe(close_prices, candles_per_year=8760)
    assert math.isfinite(sh)
    assert sh != 0.0

    const_close = pd.Series([100.0, 100.0, 100.0])
    sh_const = buy_and_hold_sharpe(const_close, candles_per_year=8760)
    assert sh_const == 0.0


def test_run_diagnostics_synthetic_frames():
    """Kiểm tra chạy run_diagnostics với dữ liệu synthetic sinh sẵn."""
    df = generate_synthetic_ohlcv(
        n_candles=365 * 24, symbol="BTC/USDT:USDT", timeframe="1h"
    )
    per_window, summary = run_diagnostics_fn("BTC/USDT:USDT", days=365, df=df)

    assert not per_window.empty
    expected_window_cols = [
        "bh_return_pct",
        "base_zero_return_pct",
        "base_real_fees_pct_equity",
        "base_real_gross_return_pct",
        "edge_zero_minus_bh_pct",
    ]
    for col in expected_window_cols:
        assert col in per_window.columns

    assert len(summary) == 1
    expected_summary_cols = [
        "breakeven_fee_per_side_bps",
        "bh_mean_sharpe",
        "base_zero_mean_sharpe",
    ]
    for col in expected_summary_cols:
        assert col in summary.columns


def test_cli_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Kiểm tra CLI run_diagnostics tạo đúng file CSV và Markdown khi hoàn tất."""
    def _fake_load_or_fetch(symbol: str, timeframe: str, days: int = 180, **kwargs):
        return generate_synthetic_ohlcv(
            n_candles=days * 24, symbol=symbol, timeframe=timeframe
        )

    monkeypatch.setattr("src.diagnostics.load_or_fetch_symbol_data", _fake_load_or_fetch)

    csv_file = tmp_path / "d.csv"
    md_file = tmp_path / "d.md"

    ret = run_diagnostics.main(
        [
            "--symbols",
            "BTC/USDT:USDT",
            "--days",
            "365",
            "--out-csv",
            str(csv_file),
            "--out-md",
            str(md_file),
        ]
    )
    assert ret == 0
    assert csv_file.exists()
    assert md_file.exists()

    df_csv = pd.read_csv(csv_file)
    assert "symbol" in df_csv.columns
    assert len(md_file.read_text(encoding="utf-8")) > 0
