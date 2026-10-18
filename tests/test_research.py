"""Test Research Engine (Sprint 3 - Đợt 3): ma trận tổ hợp, sắp xếp và xuất CSV/Markdown offline."""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_research
from src.research import (
    RESEARCH_COLUMNS,
    export_research_results,
    generate_synthetic_ohlcv,
    load_or_fetch_symbol_data,
    run_research_matrix,
    to_markdown,
)

_SYMBOL = "BTC/USDT:USDT"


def _synthetic_map() -> dict[tuple[str, str], pd.DataFrame]:
    """Chỉ dùng 1 symbol với 2 khung -> tránh cross-product tạo key thiếu (không gọi mạng)."""
    return {
        (_SYMBOL, "1h"): generate_synthetic_ohlcv(n_candles=900, timeframe="1h", symbol=_SYMBOL, seed=1),
        (_SYMBOL, "15m"): generate_synthetic_ohlcv(n_candles=900, timeframe="15m", symbol=_SYMBOL, seed=2),
    }


def test_load_or_fetch_symbol_data_synthetic_offline():
    df = load_or_fetch_symbol_data(_SYMBOL, "1h", use_synthetic=True, synthetic_n=300)
    assert len(df) == 300
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"


def test_research_matrix_runs_offline_and_sorted_desc():
    data = _synthetic_map()
    results = run_research_matrix(
        symbols=[_SYMBOL],
        timeframes=["1h", "15m"],
        strategy_modes=["breakout", "mean_reversion"],
        ml_options=(False,),
        trailing_options=(False,),
        fee_aware_options=(True,),
        entry_modes=("market", "limit_pullback"),
        data_map=data,
    )
    assert not results.empty
    assert list(results.columns) == RESEARCH_COLUMNS
    # 2 khung x 2 mode x 2 entry = 8 tổ hợp
    assert len(results) == 8
    assert results["sharpe_ratio"].is_monotonic_decreasing
    assert results["use_fee_aware_sizing"].all()
    assert np.allclose(results["min_stop_distance_pct"].to_numpy(), 0.008)


def test_research_matrix_keeps_legacy_sizing_when_fee_aware_off():
    data = _synthetic_map()
    results = run_research_matrix(
        symbols=[_SYMBOL],
        timeframes=["1h"],
        strategy_modes=["breakout"],
        ml_options=(False,),
        trailing_options=(False,),
        fee_aware_options=(False,),
        entry_modes=("market",),
        data_map=data,
    )
    assert len(results) == 1
    assert results.iloc[0]["min_stop_distance_pct"] == 0.0


def test_export_research_results_writes_csv_and_markdown(tmp_path):
    data = _synthetic_map()
    results = run_research_matrix(
        symbols=[_SYMBOL],
        timeframes=["1h"],
        strategy_modes=["breakout"],
        ml_options=(False,),
        trailing_options=(False,),
        fee_aware_options=(False,),
        entry_modes=("market",),
        data_map=data,
    )
    out_csv = tmp_path / "results.csv"
    out_md = tmp_path / "results.md"
    md = export_research_results(results, out_csv, out_md)
    assert out_csv.exists()
    assert out_md.exists()
    assert out_md.read_text(encoding="utf-8") == md
    assert md.splitlines()[0].startswith("| symbol |")
    assert len(md.splitlines()) == len(results) + 2


def test_to_markdown_formats_inf_and_bool():
    df = pd.DataFrame(
        [{"symbol": _SYMBOL, "use_ml": True, "profit_factor": float("inf"), "sharpe_ratio": 1.2345}]
    )
    lines = to_markdown(df).splitlines()
    assert lines[0] == "| symbol | use_ml | profit_factor | sharpe_ratio |"
    assert "true" in lines[2]
    assert "inf" in lines[2]


def test_run_research_cli_exports_files_and_returns_zero(tmp_path, monkeypatch):
    data = _synthetic_map()
    results = run_research_matrix(
        symbols=[_SYMBOL],
        timeframes=["1h"],
        strategy_modes=["breakout"],
        ml_options=(False,),
        trailing_options=(False,),
        fee_aware_options=(False,),
        entry_modes=("market",),
        data_map=data,
    )
    monkeypatch.setattr(run_research, "run_research_matrix", lambda **kwargs: results.copy())
    out_csv = tmp_path / "cli.csv"
    out_md = tmp_path / "cli.md"
    rc = run_research.main(
        [
            "--symbols",
            _SYMBOL,
            "--timeframes",
            "1h",
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ]
    )
    assert rc == 0
    assert out_csv.exists()
    assert out_md.exists()


def test_research_matrix_smoke_with_ml_and_trailing_offline():
    """Bao phủ các nhánh ML + trailing trong ma trận (offline, data_map)."""
    data = _synthetic_map()
    results = run_research_matrix(
        symbols=[_SYMBOL],
        timeframes=["1h", "15m"],
        strategy_modes=["breakout", "trend_pullback", "mean_reversion", "regime_adaptive"],
        ml_options=(False, True),
        trailing_options=(False, True),
        fee_aware_options=(True,),
        entry_modes=("market",),
        data_map=data,
    )
    # 2 khung x 4 mode x 2 ml x 2 trailing = 32
    assert len(results) == 32
    assert set(results["use_ml"].unique()) == {True, False}
    assert set(results["use_trailing_stop"].unique()) == {True, False}
