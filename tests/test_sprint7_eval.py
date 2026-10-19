"""Unit tests cho module đánh giá Sprint 7 và CLI Runner run_sprint7.py.

Toàn bộ test đều offline, deterministic, không gọi mạng và không đọc `data/raw`.
Các hàm private của runner (`_split_segments`, `_benchmark_from`) được import trực
tiếp theo đúng thiết kế kiểm thử của Sprint 7.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_sprint7
from src.config import EXTRA_8_SYMBOLS, UNIVERSE_12_SYMBOLS
from src.sprint7_eval import (
    build_benchmark_returns,
    compute_turnover_reduction,
    evaluate_carry_go_no_go,
    evaluate_cross_sectional_go_no_go,
    segment_metrics,
    summarize_verdicts,
    to_markdown_table,
)


def _hourly_index(periods: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC theo giờ, dùng chung cho các test tổng hợp."""
    return pd.date_range(start, periods=periods, freq="1h", tz="UTC")


def test_segment_metrics_basic() -> None:
    """Kiểm tra các chỉ số cơ bản của segment_metrics trên chuỗi lợi nhuận dương.

    Lưu ý: chuỗi hằng số tuyệt đối sẽ có độ lệch chuẩn bằng 0 nên Sharpe luôn bằng
    0.0. Để kiểm tra được nhánh Sharpe dương, thêm một nhiễu đối xứng ±1e-10 quanh
    0.001; nhiễu này triệt tiêu khi nhân nên net return không đổi trong sai số 1e-9.
    """
    idx = _hourly_index(100)
    base = pd.Series([0.001] * 100, index=idx, dtype="float64")
    noise = pd.Series(
        np.where(np.arange(100) % 2 == 0, 1e-10, -1e-10), index=idx, dtype="float64"
    )
    returns = base + noise

    metrics = segment_metrics(returns)

    assert metrics["net_return_pct"] == pytest.approx(
        (1.001**100 - 1) * 100, rel=1e-9
    )
    assert metrics["sharpe"] > 0.0
    assert math.isfinite(metrics["sharpe"])
    assert metrics["maxdd_pct"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["n_bars"] == 100
    assert metrics["n_days"] == pytest.approx(99 / 24, rel=1e-6)
    assert 0.0 <= metrics["dsr_prob"] <= 1.0


def test_segment_metrics_drawdown() -> None:
    """Kiểm tra Max Drawdown và net return khi có một cú sụt 20% đột ngột."""
    idx = _hourly_index(52)
    returns = pd.Series([0.001] * 50 + [-0.20] + [0.001], index=idx)

    metrics = segment_metrics(returns)

    # Đỉnh vốn trước cú giảm rồi rơi 20% chỉ một nến => MaxDD xấp xỉ 20%.
    assert metrics["maxdd_pct"] >= 19.0
    assert metrics["net_return_pct"] < 0.0


def test_segment_metrics_benchmark_alpha_beta() -> None:
    """Kiểm tra Alpha/Beta Newey-West với benchmark độc lập và benchmark trùng chiến lược."""
    idx = _hourly_index(500)

    rng_strategy = np.random.default_rng(7)
    rng_benchmark = np.random.default_rng(9)
    strategy = pd.Series(
        0.0005 + rng_strategy.normal(0.0, 0.01, size=500), index=idx
    )
    benchmark = pd.Series(rng_benchmark.normal(0.0, 0.01, size=500), index=idx)

    metrics = segment_metrics(strategy, benchmark_returns=benchmark)

    # Hai chuỗi nhiễu độc lập => beta gần 0.
    assert metrics["beta"] == pytest.approx(0.0, abs=0.3)
    assert isinstance(metrics["alpha_t"], float)
    assert math.isfinite(metrics["alpha_t"])
    assert metrics["nw_lags"] >= 0

    # Chiến lược trùng benchmark => beta xấp xỉ 1.0.
    same = segment_metrics(benchmark, benchmark_returns=benchmark)
    assert same["beta"] == pytest.approx(1.0, abs=0.05)
    assert math.isfinite(same["alpha_t"])


def test_segment_metrics_invalid_input() -> None:
    """Kiểm tra các hàm công khai từ chối đầu vào None bằng ValueError."""
    with pytest.raises(ValueError):
        segment_metrics(None)
    with pytest.raises(ValueError):
        build_benchmark_returns(None)


def test_carry_go_no_go() -> None:
    """Kiểm tra bộ tiêu chí GO / NO-GO cho True MTM Carry và các trường hợp trượt."""
    dev = {"maxdd_pct": 3.0, "annualized_return_pct": 8.0, "sharpe": 2.0}
    holdout = {"maxdd_pct": 4.0, "annualized_return_pct": 7.0, "sharpe": 1.8}

    passed = evaluate_carry_go_no_go(dev, holdout)
    assert passed["verdict"] == "GO"
    assert passed["failed"] == []
    assert passed["cond1_maxdd"] is True
    assert passed["cond2_return_sharpe"] is True

    # (a) MaxDD = 0% trên Holdout bị coi là mô hình phẳng, không hợp lệ.
    flat_holdout = dict(holdout, maxdd_pct=0.0)
    flat = evaluate_carry_go_no_go(dev, flat_holdout)
    assert flat["cond1_maxdd"] is False
    assert flat["verdict"] == "NO-GO"
    assert "cond1_maxdd" in flat["failed"]

    # (b) MaxDD vượt ngưỡng 10%.
    deep_holdout = dict(holdout, maxdd_pct=12.0)
    deep = evaluate_carry_go_no_go(dev, deep_holdout)
    assert deep["cond1_maxdd"] is False
    assert deep["verdict"] == "NO-GO"

    # (c) Return DEV dưới 6.0%/năm.
    low_return_dev = dict(dev, annualized_return_pct=5.99)
    low = evaluate_carry_go_no_go(low_return_dev, holdout)
    assert low["cond2_return_sharpe"] is False
    assert low["verdict"] == "NO-GO"


def test_xs_go_no_go_universe_thresholds() -> None:
    """Kiểm tra tiêu chí Cross-Sectional, ngưỡng alpha theo universe và các biên điều kiện."""
    dev = {"alpha_t": 2.05, "beta": 0.10, "sharpe": 1.20, "dsr_prob": 0.95}
    holdout = {"net_return_pct": 9.0, "sharpe": 1.30, "maxdd_pct": 10.0}

    # Rổ 8 coin: ngưỡng alpha_t >= 2.00 => GO.
    xs8 = evaluate_cross_sectional_go_no_go(dev, holdout, universe="xs8")
    assert xs8["verdict"] == "GO"
    assert xs8["failed"] == []
    assert xs8["alpha_t_min"] == pytest.approx(2.00)

    # Rổ 12 coin: ngưỡng alpha_t >= 2.20 => trượt điều kiện 1.
    xs12 = evaluate_cross_sectional_go_no_go(dev, holdout, universe="xs12")
    assert xs12["verdict"] == "NO-GO"
    assert "cond1_alpha_beta" in xs12["failed"]
    assert xs12["alpha_t_min"] == pytest.approx(2.20)

    # |beta| > 0.15 => trượt điều kiện 1 dù alpha đạt.
    high_beta = evaluate_cross_sectional_go_no_go(
        dict(dev, beta=0.20), holdout, universe="xs8"
    )
    assert high_beta["verdict"] == "NO-GO"
    assert "cond1_alpha_beta" in high_beta["failed"]

    # DSR < 0.90 => trượt điều kiện 2.
    low_dsr = evaluate_cross_sectional_go_no_go(
        dict(dev, dsr_prob=0.89), holdout, universe="xs8"
    )
    assert low_dsr["verdict"] == "NO-GO"
    assert "cond2_sharpe_dsr" in low_dsr["failed"]

    # Holdout net return phải > 8.0% (nghiêm ngặt, 8.0 không đạt) => trượt điều kiện 3.
    low_return = evaluate_cross_sectional_go_no_go(
        dev, dict(holdout, net_return_pct=8.0), universe="xs8"
    )
    assert low_return["verdict"] == "NO-GO"
    assert "cond3_holdout" in low_return["failed"]

    # Holdout MaxDD vượt 15.0% => trượt điều kiện 3.
    high_dd = evaluate_cross_sectional_go_no_go(
        dev, dict(holdout, maxdd_pct=15.0001), universe="xs8"
    )
    assert high_dd["verdict"] == "NO-GO"
    assert "cond3_holdout" in high_dd["failed"]

    # Universe không hợp lệ phải raise ValueError.
    with pytest.raises(ValueError):
        evaluate_cross_sectional_go_no_go(dev, holdout, universe="xs20")


def test_turnover_and_summary() -> None:
    """Kiểm tra % giảm turnover và hàm tổng hợp phán quyết GO / NO-GO."""
    assert compute_turnover_reduction(200, 150) == pytest.approx(25.0)
    assert compute_turnover_reduction(0, 10) == pytest.approx(0.0)

    rows = [
        {"verdict": "GO"},
        {"verdict": "NO-GO"},
        {"verdict": "GO"},
        {},
    ]
    assert summarize_verdicts(rows) == {"go": 2, "no_go": 1, "total": 3}


def test_markdown_table() -> None:
    """Kiểm tra định dạng bảng Markdown: rỗng, boolean, None và làm tròn float."""
    assert to_markdown_table([]) == "(không có dữ liệu)"

    table = to_markdown_table([{"a": 1, "b": True, "c": None, "d": 1.23456}])

    assert "✔" in table
    assert "n/a" in table
    assert "1.2346" in table
    assert table.splitlines()[0] == "| a | b | c | d |"
    assert table.splitlines()[1] == "| --- | --- | --- | --- |"


def test_iter_xs_configs_grid() -> None:
    """Kiểm tra lưới 16 cấu hình Cross-Sectional đóng băng của runner Sprint 7."""
    configs = list(run_sprint7.iter_xs_configs())

    assert len(configs) == 16
    config_ids = [config_id for config_id, _cfg, _params, _symbols in configs]
    assert len(set(config_ids)) == 16

    xs8 = [item for item in configs if item[2]["universe"] == "xs8"]
    xs12 = [item for item in configs if item[2]["universe"] == "xs12"]
    assert len(xs8) == 8
    assert len(xs12) == 8

    # top_k và danh sách symbol phải khớp đúng từng rổ.
    for _config_id, cfg, params, symbols in xs8:
        assert cfg.top_k == 2
        assert params["top_k"] == 2
        assert list(symbols) == EXTRA_8_SYMBOLS
    for _config_id, cfg, params, symbols in xs12:
        assert cfg.top_k == 3
        assert params["top_k"] == 3
        assert list(symbols) == UNIVERSE_12_SYMBOLS

    # Mỗi rổ phủ đủ tích Descartes lookback × buffer × tilt.
    for universe_configs in (xs8, xs12):
        grid = {
            (
                params["lookback_hours"],
                params["buffer_rank"],
                params["carry_tilt_weight"],
            )
            for _config_id, _cfg, params, _symbols in universe_configs
        }
        assert grid == {
            (lookback, buffer_rank, tilt)
            for lookback in (336, 504)
            for buffer_rank in (0, 2)
            for tilt in (0.0, 0.25)
        }

    for _config_id, cfg, _params, _symbols in configs:
        assert cfg.rebalance_hours == 48
        assert cfg.perp_fee == pytest.approx(0.0005)
        assert cfg.slippage == pytest.approx(0.0002)


def test_iter_carry_configs_grid() -> None:
    """Kiểm tra lưới 4 cấu hình Carry cho mỗi kiểu dữ liệu (true_mtm / idealized)."""
    expected_pairs = {
        ("coinm_1leg", "hysteresis"),
        ("coinm_1leg", "passive_long_term"),
        ("spot_perp_2leg", "hysteresis"),
        ("spot_perp_2leg", "passive_long_term"),
    }

    for data_model in ("true_mtm", "idealized"):
        configs = list(run_sprint7.iter_carry_configs(data_model))
        assert len(configs) == 4
        assert all(config_id.startswith(f"{data_model}|") for config_id, _cfg, _params in configs)
        assert {
            (cfg.structure, cfg.carry_mode) for _config_id, cfg, _params in configs
        } == expected_pairs
        assert all(cfg.basis_noise_std == 0.0 for _config_id, cfg, _params in configs)

    true_mtm_ids = {
        config_id for config_id, _cfg, _params in run_sprint7.iter_carry_configs("true_mtm")
    }
    assert all(config_id.startswith("true_mtm|") for config_id in true_mtm_ids)


def test_select_best_from_csv(tmp_path: Path) -> None:
    """Kiểm tra chọn cấu hình tốt nhất trên DEV từ CSV cho Carry và Cross-Sectional."""
    carry_csv = tmp_path / "carry.csv"
    pd.DataFrame(
        [
            {
                "config_id": "true_mtm|coinm_1leg|hysteresis",
                "data_model": "true_mtm",
                "dev_sharpe": 1.0,
                "dev_dsr_prob": 0.90,
            },
            {
                "config_id": "true_mtm|spot_perp_2leg|hysteresis",
                "data_model": "true_mtm",
                "dev_sharpe": 2.0,
                "dev_dsr_prob": 0.95,
            },
            {
                "config_id": "true_mtm|coinm_1leg|passive_long_term",
                "data_model": "true_mtm",
                "dev_sharpe": 1.5,
                "dev_dsr_prob": 0.92,
            },
            {
                "config_id": "idealized|spot_perp_2leg|hysteresis",
                "data_model": "idealized",
                "dev_sharpe": 9.0,
                "dev_dsr_prob": 0.99,
            },
        ]
    ).to_csv(carry_csv, index=False)

    selected_carry = run_sprint7.select_best_from_csv(carry_csv, "carry")
    assert selected_carry == {"true_mtm|spot_perp_2leg|hysteresis"}

    xs_csv = tmp_path / "xs.csv"
    pd.DataFrame(
        [
            {
                "config_id": "xs8|lb336|buf0|tilt0",
                "universe": "xs8",
                "dev_sharpe": 1.0,
                "dev_dsr_prob": 0.90,
            },
            {
                "config_id": "xs8|lb504|buf2|tilt0.25",
                "universe": "xs8",
                "dev_sharpe": 2.0,
                "dev_dsr_prob": 0.95,
            },
            {
                "config_id": "xs12|lb336|buf0|tilt0",
                "universe": "xs12",
                "dev_sharpe": 1.5,
                "dev_dsr_prob": 0.92,
            },
            {
                "config_id": "xs12|lb504|buf2|tilt0.25",
                "universe": "xs12",
                "dev_sharpe": 2.5,
                "dev_dsr_prob": 0.97,
            },
        ]
    ).to_csv(xs_csv, index=False)

    selected_xs = run_sprint7.select_best_from_csv(xs_csv, "xs")
    assert selected_xs == {
        "xs8|lb504|buf2|tilt0.25",
        "xs12|lb504|buf2|tilt0.25",
    }


def test_split_segments_boundary() -> None:
    """Kiểm tra _split_segments cắt đúng biên, không trùng lặp và giữ thứ tự thời gian."""
    idx = _hourly_index(10)
    returns = pd.Series(np.arange(10, dtype="float64") / 1000.0, index=idx)

    dev, holdout = run_sprint7._split_segments(returns, "2024-01-01 05:00")

    assert len(dev) == 5
    assert len(holdout) == 5
    assert dev.index.intersection(holdout.index).empty
    assert dev.index.max() < holdout.index.min()
    assert dev.index.max() == pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
    assert holdout.index.min() == pd.Timestamp("2024-01-01 05:00:00", tz="UTC")


def test_benchmark_from_fallback() -> None:
    """Kiểm tra _benchmark_from dùng fallback_close khi rổ không chứa BTC và raise nếu thiếu."""
    idx = _hourly_index(6)
    frames = {
        "XRP/USDT:USDT": pd.DataFrame({"close": [1.0] * 6}, index=idx),
    }
    fallback_close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=idx)

    result = run_sprint7._benchmark_from(frames, idx, fallback_close=fallback_close)
    expected = build_benchmark_returns(fallback_close)

    pd.testing.assert_series_equal(result, expected)

    # Không có BTC trong rổ và không có fallback => KeyError.
    with pytest.raises(KeyError):
        run_sprint7._benchmark_from(frames, idx)
