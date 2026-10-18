"""Bộ kiểm thử cho Deep Analysis Engine (offline, synthetic, deterministic).

Kiểm tra toàn bộ các thành phần phân tích của Sprint 5 - Phần A:
- OLS Newey-West HAC covariance.
- Max Drawdown calculation.
- Exposure metrics (time in market, notional ratio, long ratio, holding bars).
- Block Bootstrap Sharpe difference.
- Exposure-matched Null Hypothesis Test.
- Hàm tích hợp run_deep_analysis.
- CLI run_deep_analysis xuất 5 CSV và file Markdown.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_deep_analysis as cli_run_deep
from src.backtester import TradeRecord
from src.deep_analysis import (
    block_bootstrap_sharpe_diff,
    exposure_matched_null,
    exposure_metrics,
    max_drawdown_pct,
    newey_west_ols,
    run_deep_analysis,
)
from src.research import generate_synthetic_ohlcv


def test_newey_west_recovers_slope():
    """Kiểm tra OLS Newey-West khôi phục chính xác hệ số alpha, beta và t-statistic."""
    rng = np.random.default_rng(0)
    x = np.arange(200) / 100.0
    y = 2.0 + 3.0 * x + rng.normal(0, 0.01, 200)
    r = newey_west_ols(y, x)
    assert abs(r["beta"] - 3.0) < 0.1
    assert abs(r["alpha"] - 2.0) < 0.1
    assert r["beta_t"] > 10.0
    assert r["n"] == 200
    assert r["lags"] >= 1


def test_max_drawdown_pct():
    """Kiểm tra tính Max Drawdown: chuỗi có sụt giảm và chuỗi tăng đơn điệu."""
    equity = pd.Series([1.0, 2.0, 1.5, 3.0])
    dd = max_drawdown_pct(equity)
    assert abs(dd - 25.0) < 1e-6

    increasing = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert max_drawdown_pct(increasing) == 0.0


def test_exposure_metrics():
    """Kiểm tra tính toán các chỉ số phơi nhiễm thị trường (exposure metrics)."""
    index = pd.date_range("2024-01-01 00:00:00", freq="1h", periods=10, tz="UTC")
    equity = pd.Series(400.0, index=index)
    sample_trade = TradeRecord(
        entry_time=index[1],
        exit_time=index[4],
        side=1,
        entry_price=50000.0,
        exit_price=51000.0,
        qty=0.01,
        gross_pnl=10.0,
        total_fee=0.5,
        net_pnl=9.5,
        exit_reason="TP",
    )
    trade1 = sample_trade
    trade2 = dataclasses.replace(
        sample_trade,
        entry_time=index[5],
        exit_time=index[8],
        side=-1,
        entry_price=50000.0,
        exit_price=49000.0,
    )
    res = exposure_metrics([trade1, trade2], index, equity, initial_capital=400.0)
    assert 0.0 < res["time_in_market_pct"] <= 100.0
    assert res["long_ratio"] == 0.5
    assert res["avg_holding_bars"] > 0.0
    assert res["n_trades"] == 2


def test_block_bootstrap_returns_ci():
    """Kiểm tra Block Bootstrap trả về khoảng tin cậy 95% và p-value hợp lệ."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", freq="1h", periods=60 * 24, tz="UTC")
    r_base = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    r_bh = pd.Series(rng.normal(0.0002, 0.015, len(idx)), index=idx)

    kq = block_bootstrap_sharpe_diff(r_base, r_bh, candles_per_year=8760, n_iter=200, seed=42)
    assert kq["ci_low"] <= kq["mean_diff"] <= kq["ci_high"]
    assert 0.0 <= kq["p_value"] <= 1.0
    assert kq["n_days"] == 60


def test_exposure_matched_null_percentile_range():
    """Kiểm tra Exposure-Matched Null Test trả về phân vị và khoảng tin cậy hợp lệ."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", freq="1h", periods=500, tz="UTC")
    close = pd.Series(50000.0 * np.exp(np.cumsum(rng.normal(0, 0.002, 500))), index=idx)

    sample_trade = TradeRecord(
        entry_time=idx[10],
        exit_time=idx[20],
        side=1,
        entry_price=float(close.iloc[10]),
        exit_price=float(close.iloc[20]),
        qty=0.01,
        gross_pnl=1.0,
        total_fee=0.1,
        net_pnl=0.9,
        exit_reason="TP",
    )
    trades = [
        dataclasses.replace(
            sample_trade,
            entry_time=idx[i * 50],
            exit_time=idx[i * 50 + 10],
            side=1 if i % 2 == 0 else -1,
        )
        for i in range(5)
    ]
    kq = exposure_matched_null(close, trades, fee_per_side=0.0007, n_iter=200, seed=42)
    assert 0.0 <= kq["percentile"] <= 100.0
    assert kq["null_p2_5"] <= kq["null_p97_5"]
    assert kq["n_iter"] == 200


def test_run_deep_analysis_synthetic_frames():
    """Kiểm tra cấu trúc và tính đầy đủ của 5 DataFrame trong run_deep_analysis."""
    df = generate_synthetic_ohlcv(n_candles=365 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    out = run_deep_analysis("BTC/USDT:USDT", days=365, df=df, n_boot=100, n_null=100)

    assert set(out.keys()) == {"exposure", "drawdown", "alpha_beta", "bootstrap", "null"}
    for k, v in out.items():
        assert isinstance(v, pd.DataFrame)
        assert not v.empty

    assert {
        "symbol",
        "timeframe",
        "n_trades",
        "time_in_market_pct",
        "avg_notional_to_equity",
        "long_ratio",
        "avg_holding_bars",
    }.issubset(out["exposure"].columns)
    assert {
        "symbol",
        "window_index",
        "test_start",
        "test_end",
        "base_zero_dd",
        "base_real_dd",
        "bh_dd",
    }.issubset(out["drawdown"].columns)
    assert {
        "symbol",
        "timeframe",
        "n_obs",
        "lags",
        "alpha_per_bar",
        "alpha_annualized_pct",
        "beta",
        "alpha_t",
        "beta_t",
    }.issubset(out["alpha_beta"].columns)
    assert {
        "symbol",
        "timeframe",
        "mean_sharpe_diff",
        "ci_low",
        "ci_high",
        "p_value",
        "n_days",
    }.issubset(out["bootstrap"].columns)
    assert {
        "symbol",
        "timeframe",
        "baseline_metric",
        "null_mean",
        "null_p2_5",
        "null_p97_5",
        "percentile",
        "n_iter",
    }.issubset(out["null"].columns)

    assert "alpha_t" in out["alpha_beta"].columns
    assert "beta" in out["alpha_beta"].columns


def test_cli_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Kiểm tra CLI run_deep_analysis ghi đủ 5 file CSV và file markdown tổng hợp."""

    def mock_load(symbol, timeframe="1h", days=365, *args, **kwargs):
        return generate_synthetic_ohlcv(n_candles=days * 24, symbol=symbol, timeframe=timeframe)

    monkeypatch.setattr("src.deep_analysis.load_or_fetch_symbol_data", mock_load)

    ret = cli_run_deep.main(
        [
            "--symbols",
            "BTC/USDT:USDT",
            "--days",
            "365",
            "--n-boot",
            "50",
            "--n-null",
            "50",
            "--out-dir",
            str(tmp_path),
            "--out-md",
            str(tmp_path / "a.md"),
        ]
    )
    assert ret == 0
    assert (tmp_path / "sprint5_exposure.csv").exists()
    assert (tmp_path / "sprint5_drawdown.csv").exists()
    assert (tmp_path / "sprint5_alpha_beta.csv").exists()
    assert (tmp_path / "sprint5_bootstrap.csv").exists()
    assert (tmp_path / "sprint5_null.csv").exists()
    assert (tmp_path / "a.md").exists()
