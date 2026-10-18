"""Kiểm thử Walk-Forward Validation Engine (offline, synthetic, deterministic)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from src.backtester import BacktestReport
from src.research import generate_synthetic_ohlcv
from src.walkforward import (
    Candidate,
    DEFAULT_CANDIDATES,
    WalkForwardWindow,
    WindowResult,
    generate_walkforward_windows,
    make_candidate_config,
    run_walkforward,
    select_candidate,
)


def _make_dummy_report(
    trades: int = 20,
    dd: float = 5.0,
    sharpe: float = 1.5,
    ret: float = 10.0,
) -> BacktestReport:
    """Tạo đối tượng BacktestReport giả lập cho mục đích kiểm thử."""
    return BacktestReport(
        total_return_pct=ret,
        final_equity=440.0,
        max_drawdown_pct=dd,
        sharpe_ratio=sharpe,
        win_rate_pct=50.0,
        profit_factor=1.5,
        total_trades=trades,
        total_fees_paid=1.0,
        equity_curve=pd.Series([400.0, 440.0]),
        trades=[],
    )


def test_generate_windows_count_and_no_overlap():
    """Kiểm tra số lượng cửa sổ và tính không chồng lấn giữa các tập test."""
    index = pd.date_range("2024-01-01", freq="1h", periods=365 * 24, tz="UTC")
    windows = generate_walkforward_windows(index, train_days=120, test_days=60, step_days=60)
    assert len(windows) == 4
    for i, w in enumerate(windows):
        assert w.window_index == i
        assert w.train_end == w.test_start
        assert w.train_start < w.train_end
        assert w.test_start < w.test_end
        if i > 0:
            assert w.test_start >= windows[i - 1].test_end


def test_select_candidate_none_when_none_qualify():
    """Kiểm tra trả về None khi không ứng viên nào thỏa mãn tiêu chí lọc."""
    c1 = Candidate("breakout", "market", False)
    c2 = Candidate("trend_pullback", "limit_pullback", True)

    reports = {
        c1: _make_dummy_report(trades=10, dd=5.0, sharpe=2.0),   # vi phạm min_trades (10 < 15)
        c2: _make_dummy_report(trades=20, dd=12.0, sharpe=2.5),  # vi phạm max_drawdown_pct (12.0 > 8.0)
    }
    chosen = select_candidate(reports, min_trades=15, max_drawdown_pct=8.0)
    assert chosen is None


def test_select_candidate_picks_best_sharpe():
    """Kiểm tra chọn ứng viên có Sharpe cao nhất và tie-break chuẩn xác."""
    c1 = Candidate("breakout", "market", False)
    c2 = Candidate("breakout", "market", True)
    c3 = Candidate("trend_pullback", "limit_pullback", False)

    reports = {
        c1: _make_dummy_report(trades=20, dd=5.0, sharpe=1.5),
        c2: _make_dummy_report(trades=20, dd=6.0, sharpe=2.0),
        c3: _make_dummy_report(trades=5, dd=2.0, sharpe=3.0),   # Sharpe cao nhưng không đủ số lệnh
    }
    chosen = select_candidate(reports, min_trades=15, max_drawdown_pct=8.0)
    assert chosen == c2

    # Tie-break khi Sharpe bằng nhau: ưu tiên max_drawdown_pct nhỏ hơn
    reports_tie = {
        c1: _make_dummy_report(trades=20, dd=4.0, sharpe=2.0),
        c2: _make_dummy_report(trades=20, dd=6.0, sharpe=2.0),
    }
    chosen_tie = select_candidate(reports_tie, min_trades=15, max_drawdown_pct=8.0)
    assert chosen_tie == c1


def test_run_walkforward_synthetic_returns_frames():
    """Kiểm tra run_walkforward trả về đúng cấu trúc DataFrame và các chỉ số hợp lệ."""
    df = generate_synthetic_ohlcv(n_candles=365 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    candidates = DEFAULT_CANDIDATES[:2]
    per_window_df, summary_df = run_walkforward(
        "BTC/USDT:USDT",
        timeframe="1h",
        days=365,
        candidates=candidates,
        df=df,
    )
    expected_cols = [f.name for f in dataclasses.fields(WindowResult)]
    assert list(per_window_df.columns) == expected_cols
    assert 0 < len(per_window_df) <= 4

    summary_expected = [
        "symbol",
        "n_windows",
        "pct_positive_windows",
        "mean_test_return_pct",
        "mean_test_sharpe",
        "mean_test_max_drawdown_pct",
        "mean_baseline_return_pct",
        "mean_baseline_sharpe",
        "mean_baseline_max_drawdown_pct",
        "chosen_mode_counts",
    ]
    assert list(summary_df.columns) == summary_expected
    assert len(summary_df) == 1

    for col in (
        "test_return_pct",
        "test_max_drawdown_pct",
        "test_sharpe_ratio",
        "test_total_trades",
    ):
        assert not per_window_df[col].isna().any()

    pct_pos = summary_df["pct_positive_windows"].iloc[0]
    assert 0.0 <= pct_pos <= 100.0


def test_cli_writes_outputs(tmp_path, monkeypatch):
    """Kiểm tra CLI run_walkforward xuất đầy đủ file CSV và Markdown."""
    import run_walkforward

    def _fake_load_or_fetch(symbol, timeframe, days=180, **kwargs):
        return generate_synthetic_ohlcv(
            n_candles=days * 24, symbol=symbol, timeframe=timeframe
        )

    monkeypatch.setattr("src.walkforward.load_or_fetch_symbol_data", _fake_load_or_fetch)

    csv_file = tmp_path / "wf.csv"
    md_file = tmp_path / "wf.md"

    ret = run_walkforward.main(
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
    assert not df_csv.empty
    assert "symbol" in df_csv.columns
    assert len(md_file.read_text(encoding="utf-8")) > 0


def test_stand_aside_when_no_candidate_qualifies_in_walkforward():
    """Kiểm tra trạng thái stand_aside khi không có ứng viên nào đạt ngưỡng trên train."""
    df = generate_synthetic_ohlcv(n_candles=365 * 24, symbol="BTC/USDT:USDT", timeframe="1h")
    per_window_df, summary_df = run_walkforward(
        "BTC/USDT:USDT",
        timeframe="1h",
        days=365,
        min_trades=999999,
        df=df,
    )
    assert not per_window_df.empty
    assert (per_window_df["chosen_mode"] == "stand_aside").all()
    assert (per_window_df["chosen_entry_mode"] == "").all()
    assert (per_window_df["chosen_trailing"] == False).all()
    assert (per_window_df["test_return_pct"] == 0.0).all()
    assert (per_window_df["test_max_drawdown_pct"] == 0.0).all()
    assert (per_window_df["test_sharpe_ratio"] == 0.0).all()
    assert (per_window_df["test_total_trades"] == 0).all()
