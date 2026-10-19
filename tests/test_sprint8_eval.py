"""Unit tests cho module đánh giá Sprint 8 và CLI Runner run_sprint8.py.

Toàn bộ test đều offline, tất định, nhanh, không đọc `data/raw`, không gọi mạng
và không ghi file ra ngoài `tmp_path`.

Phạm vi kiểm thử:
- split_into_periods: chia 7 cửa sổ 180 ngày liên tiếp, phủ trọn dải thời gian.
- period_metrics: lợi suất, Sharpe, MaxDD, DSR và các trường thời gian.
- build_period_table / build_pnl_attribution: bảo toàn PnL khi cắt theo thời gian
  và đẳng thức Net PnL = Funding + Basis - Fees.
- evaluate_sprint8_go_no_go: ánh xạ đúng 4 điều kiện khóa cứng theo
  docs/go_no_go_sprint8.md (8.0% / 6.0% / 6-of-7 / Sharpe 2.0 / DSR 0.95 / MaxDD 5.0).
- summarize_periods: đếm số kỳ dương và biên lợi nhuận.
- run_sprint8: parse_args mặc định, build_universe_frames thiếu cache và lưới K=12.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_sprint8
from src.sprint8_eval import (
    build_period_table,
    build_pnl_attribution,
    evaluate_sprint8_go_no_go,
    period_metrics,
    split_into_periods,
    summarize_periods,
)

N_PERIODS = 7
PERIOD_DAYS = 180


def _hourly_index(periods: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC theo giờ cho dữ liệu tổng hợp."""
    return pd.date_range(start, periods=periods, freq="1h", tz="UTC")


def _full_index() -> pd.DatetimeIndex:
    """Index 1h dài đúng 7 * 180 ngày, tương ứng 2023-05-07 -> 2026-10-18."""
    return pd.date_range(
        "2023-05-07",
        periods=N_PERIODS * PERIOD_DAYS * 24,
        freq="1h",
        tz="UTC",
    )


def _period_rows(n_positive: int = 6, n_total: int = 7) -> list[dict[str, float]]:
    """Sinh danh sách kết quả kỳ với đúng n_positive kỳ có lợi nhuận dương."""
    return [
        {
            "period_id": f"P{i + 1}",
            "net_return_pct": 1.0 if i < n_positive else -1.0,
        }
        for i in range(n_total)
    ]


def _metrics_12(**overrides: float) -> dict[str, float]:
    """Bộ chỉ số toàn chuỗi rổ 12 coin thỏa cả 4 điều kiện GO mặc định."""
    base: dict[str, float] = {
        "annualized_return_pct": 9.0,
        "sharpe": 2.5,
        "dsr_prob": 0.97,
        "maxdd_pct": 3.0,
    }
    base.update(overrides)
    return base


def _metrics_8(**overrides: float) -> dict[str, float]:
    """Bộ chỉ số toàn chuỗi rổ 8 coin thỏa ngưỡng 6.0%/năm mặc định."""
    base: dict[str, float] = {"annualized_return_pct": 6.5}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# A. split_into_periods
# ---------------------------------------------------------------------------


def test_split_into_periods_covers_full_range_7_windows() -> None:
    """7 * 180 ngày phải chia thành đúng 7 cửa sổ liên tiếp, không hở, phủ trọn dải."""
    idx = _full_index()

    periods = split_into_periods(idx, n_periods=N_PERIODS, days=PERIOD_DAYS)

    assert len(periods) == N_PERIODS
    assert periods[0][0] == idx[0]
    assert periods[-1][1] == idx[-1] + pd.Timedelta(hours=1)
    assert periods[-1][1] == pd.Timestamp("2026-10-18", tz="UTC")

    for k in range(len(periods) - 1):
        # Cửa sổ sau bắt đầu đúng nơi cửa sổ trước kết thúc, không chồng lấp.
        assert periods[k][1] == periods[k + 1][0]
        assert periods[k][0] < periods[k][1]

    assert periods[-1][0] < periods[-1][1]
    for start_ts, end_ts in periods:
        assert isinstance(start_ts, pd.Timestamp)
        assert isinstance(end_ts, pd.Timestamp)
        assert str(start_ts.tz) == "UTC"
        assert str(end_ts.tz) == "UTC"

    total_window_hours = sum(
        (end_ts - start_ts).total_seconds() / 3600.0 for start_ts, end_ts in periods
    )
    index_hours = (idx[-1] - idx[0]).total_seconds() / 3600.0 + 1.0
    assert abs(total_window_hours - index_hours) < 48.0


def test_split_into_periods_empty_and_short_index_no_crash() -> None:
    """Index rỗng trả list rỗng; index ngắn không được crash và vẫn tất định.

    Ghi chú: với index ngắn hơn 1 ngày, code thực tế vẫn trả đủ n_periods cửa sổ
    (các cửa sổ rỗng ở giữa). Test xác nhận hành vi không crash và liên tục thay vì
    áp đặt số phần tử, vì đây là contract thực tế của API.
    """
    assert split_into_periods(pd.DatetimeIndex([])) == []
    assert split_into_periods(None) == []

    short = _hourly_index(5, "2024-01-01")
    periods = split_into_periods(short, n_periods=N_PERIODS, days=PERIOD_DAYS)

    assert isinstance(periods, list)
    assert 1 <= len(periods) <= N_PERIODS
    assert periods[0][0] == short[0]
    assert periods[-1][1] >= short[-1]
    for k in range(len(periods) - 1):
        assert periods[k][1] == periods[k + 1][0]


# ---------------------------------------------------------------------------
# B. period_metrics
# ---------------------------------------------------------------------------


def test_period_metrics_constant_positive_returns() -> None:
    """Lợi suất dương nhỏ gần hằng số: ann return > 0, Sharpe lớn hữu hạn, MaxDD = 0."""
    idx = _hourly_index(240)
    base = pd.Series(np.full(240, 0.0005), index=idx, dtype="float64")
    # Nhiễu đối xứng rất nhỏ để std > 0 mà không làm đổi dấu lợi suất.
    noise = pd.Series(
        np.where(np.arange(240) % 2 == 0, 1e-10, -1e-10), index=idx, dtype="float64"
    )
    returns = base + noise

    metrics = period_metrics(returns, periods_per_year=8760, num_trials=12)

    required = {
        "net_return_pct",
        "annualized_return_pct",
        "sharpe",
        "maxdd_pct",
        "dsr_prob",
        "n_bars",
        "start",
        "end",
    }
    assert required.issubset(metrics.keys())

    assert metrics["net_return_pct"] > 0.0
    assert metrics["annualized_return_pct"] > 0.0
    assert not math.isnan(metrics["sharpe"])
    assert math.isfinite(metrics["sharpe"])
    assert metrics["sharpe"] > 1.0
    assert metrics["maxdd_pct"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["n_bars"] == 240
    assert metrics["start"] == idx[0]
    assert metrics["end"] == idx[-1]
    assert 0.0 <= metrics["dsr_prob"] <= 1.0


def test_period_metrics_drawdown_after_large_drop() -> None:
    """Một cú sụt lớn phải tạo Max Drawdown dương và lợi nhuận ròng âm."""
    idx = _hourly_index(60)
    returns = pd.Series([0.001] * 30 + [-0.20] + [0.001] * 29, index=idx)

    metrics = period_metrics(returns)

    assert metrics["maxdd_pct"] > 0.0
    assert metrics["maxdd_pct"] >= 19.0
    assert metrics["net_return_pct"] < 0.0
    assert metrics["n_bars"] == 60


# ---------------------------------------------------------------------------
# C. build_period_table
# ---------------------------------------------------------------------------


def test_build_period_table_attribution_sums_are_preserved() -> None:
    """Cắt attribution theo thời gian không được làm mất hoặc nhân đôi PnL."""
    idx = _full_index()
    periods = split_into_periods(idx, n_periods=N_PERIODS, days=PERIOD_DAYS)
    n = len(idx)

    # Nửa đầu dương, nửa sau âm để bảng có cả kỳ dương và kỳ âm.
    values = np.where(np.arange(n) < n // 2, 0.0005, -0.0002)
    returns = pd.Series(values, index=idx)

    attribution = pd.DataFrame(
        {
            "funding_pnl": np.full(n, 0.0003),
            "basis_pnl": np.full(n, 0.0002),
            "fees": np.full(n, 0.0001),
            "net_pnl": np.full(n, 0.0004),
            "notional_traded": np.full(n, 0.5),
        },
        index=idx,
    )

    table = build_period_table(
        returns,
        None,
        periods,
        attribution_series=attribution,
        initial_capital=400.0,
        num_trials=12,
    )

    assert len(table) == N_PERIODS
    assert list(table["period_id"]) == [f"P{i}" for i in range(1, N_PERIODS + 1)]
    assert int(table["n_bars"].sum()) == n
    assert table["start"].iloc[0] == idx[0]
    assert table["end"].iloc[-1] == idx[-1] + pd.Timedelta(hours=1)

    for col in ("funding_pnl", "basis_pnl", "fees"):
        expected = float(attribution[col].sum())
        actual = float(table[col].sum())
        assert actual == pytest.approx(expected, rel=1e-9)
        assert abs(actual - expected) <= abs(expected) * 0.01

    # Bảng kỳ biểu diễn khối lượng giao dịch qua turnover = notional_traded / vốn.
    assert table["turnover"].sum() == pytest.approx(
        float(attribution["notional_traded"].sum()) / 400.0, rel=1e-9
    )

    expected_positive = (table["net_return_pct"] > 0.0).tolist()
    assert table["positive"].tolist() == expected_positive
    assert table["positive"].any()
    assert not table["positive"].all()


def test_build_period_table_without_attribution_defaults_to_zero() -> None:
    """attribution_series=None không được crash và các cột PnL phải bằng 0."""
    idx = _hourly_index(48)
    periods = split_into_periods(idx, n_periods=2, days=1)
    returns = pd.Series(np.full(48, 0.0001), index=idx)

    table = build_period_table(returns, None, periods, attribution_series=None)

    assert len(table) == 2
    for col in ("funding_pnl", "basis_pnl", "fees", "turnover"):
        assert (table[col] == 0.0).all()


# ---------------------------------------------------------------------------
# D. build_pnl_attribution
# ---------------------------------------------------------------------------


def test_build_pnl_attribution_identity_and_period_sums() -> None:
    """Net PnL = Funding + Basis - Fees và tổng các kỳ xấp xỉ phần tổng."""
    idx = pd.date_range(
        "2024-01-01", periods=N_PERIODS * 2 * 24, freq="1h", tz="UTC"
    )
    n = len(idx)
    attribution = pd.DataFrame(
        {
            "funding_pnl": np.full(n, 0.001),
            "basis_pnl": np.full(n, 0.002),
            "fees": np.full(n, 0.0005),
            "net_pnl": np.full(n, 0.0025),
            "notional_traded": np.full(n, 0.5),
        },
        index=idx,
    )
    periods = split_into_periods(idx, n_periods=N_PERIODS, days=2)

    total = build_pnl_attribution(attribution, initial_capital=400.0)

    assert total["funding_pnl"] == pytest.approx(0.001 * n)
    assert total["basis_pnl"] == pytest.approx(0.002 * n)
    assert total["fees"] == pytest.approx(0.0005 * n)
    assert total["net_pnl"] == pytest.approx(
        total["funding_pnl"] + total["basis_pnl"] - total["fees"]
    )
    assert total["net_pnl_usdt"] == pytest.approx(total["net_pnl"])
    assert total["net_return_pct"] == pytest.approx(total["net_pnl"] / 400.0 * 100.0)
    assert total["periods"] == []
    assert isinstance(total["periods_df"], pd.DataFrame)
    assert total["periods_df"].empty

    with_periods = build_pnl_attribution(
        attribution, periods=periods, initial_capital=400.0
    )

    assert len(with_periods["periods"]) == N_PERIODS
    assert len(with_periods["periods_df"]) == N_PERIODS
    for p in with_periods["periods"]:
        assert p["net_pnl"] == pytest.approx(
            p["funding_pnl"] + p["basis_pnl"] - p["fees"]
        )

    assert sum(p["funding_pnl"] for p in with_periods["periods"]) == pytest.approx(
        with_periods["funding_pnl"], rel=1e-9
    )
    assert sum(p["basis_pnl"] for p in with_periods["periods"]) == pytest.approx(
        with_periods["basis_pnl"], rel=1e-9
    )
    assert sum(p["fees"] for p in with_periods["periods"]) == pytest.approx(
        with_periods["fees"], rel=1e-9
    )
    assert sum(p["net_pnl"] for p in with_periods["periods"]) == pytest.approx(
        with_periods["net_pnl"], rel=1e-9
    )

    # Đầu vào dict (kết quả engine) lấy đúng attribution_series, bỏ qua số tổng cũ.
    from_dict = build_pnl_attribution(
        {"attribution_series": attribution, "funding_pnl": 12345.0},
        periods=periods,
        initial_capital=400.0,
    )
    assert from_dict["funding_pnl"] == pytest.approx(with_periods["funding_pnl"])
    assert from_dict["net_pnl"] == pytest.approx(with_periods["net_pnl"])
    assert len(from_dict["periods"]) == N_PERIODS


# ---------------------------------------------------------------------------
# E. evaluate_sprint8_go_no_go
# ---------------------------------------------------------------------------


def test_evaluate_sprint8_go_no_go_go_case() -> None:
    """Thỏa cả 4 điều kiện khóa cứng => verdict GO và failed rỗng."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(),
        _metrics_8(),
        _period_rows(6),
        live_executor_ready=True,
        num_trials=12,
    )

    assert result["verdict"] == "GO"
    assert result["failed"] == []
    for key in (
        "cond1_return",
        "cond2_consistency",
        "cond3_risk",
        "cond4_production",
    ):
        assert key in result
        assert isinstance(result[key], bool)
        assert result[key] is True
    assert result["positive_periods_count"] == 6
    assert result["total_periods"] == 7
    assert isinstance(result["failed"], list)


def test_evaluate_sprint8_go_no_go_accepts_period_dataframe() -> None:
    """period_rows_12 dạng DataFrame với cột positive phải được đếm đúng."""
    df_rows = pd.DataFrame(_period_rows(6))

    result = evaluate_sprint8_go_no_go(
        _metrics_12(), _metrics_8(), df_rows, live_executor_ready=True
    )

    assert result["positive_periods_count"] == 6
    assert result["total_periods"] == 7
    assert result["verdict"] == "GO"


def test_evaluate_sprint8_cond1_fails_on_12_universe() -> None:
    """Annualized rổ 12 = 7.9% (< 8.0) => NO-GO với cond1_return."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(annualized_return_pct=7.9),
        _metrics_8(),
        _period_rows(6),
        live_executor_ready=True,
    )

    assert result["cond1_return"] is False
    assert result["verdict"] == "NO-GO"
    assert result["failed"] == ["cond1_return"]


def test_evaluate_sprint8_cond1_fails_on_8_universe() -> None:
    """Annualized rổ 8 = 5.9% (< 6.0) => NO-GO với cond1_return."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(),
        _metrics_8(annualized_return_pct=5.9),
        _period_rows(6),
        live_executor_ready=True,
    )

    assert result["cond1_return"] is False
    assert result["verdict"] == "NO-GO"
    assert result["failed"] == ["cond1_return"]


def test_evaluate_sprint8_cond2_fails_below_6_of_7() -> None:
    """Chỉ 5/7 kỳ dương (< 6/7) => NO-GO với cond2_consistency."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(),
        _metrics_8(),
        _period_rows(5),
        live_executor_ready=True,
    )

    assert result["cond2_consistency"] is False
    assert result["positive_periods_count"] == 5
    assert result["total_periods"] == 7
    assert result["verdict"] == "NO-GO"
    assert result["failed"] == ["cond2_consistency"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"sharpe": 1.99},
        {"dsr_prob": 0.94},
        {"maxdd_pct": 5.1},
    ],
)
def test_evaluate_sprint8_cond3_fails_on_each_branch(
    overrides: dict[str, float],
) -> None:
    """Từng nhánh Sharpe / DSR / MaxDD vượt ngưỡng đều phải làm trượt cond3_risk."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(**overrides),
        _metrics_8(),
        _period_rows(6),
        live_executor_ready=True,
    )

    assert result["cond3_risk"] is False
    assert result["verdict"] == "NO-GO"
    assert result["failed"] == ["cond3_risk"]


def test_evaluate_sprint8_cond4_fails_when_executor_not_ready() -> None:
    """live_executor_ready=False => NO-GO với cond4_production."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(),
        _metrics_8(),
        _period_rows(6),
        live_executor_ready=False,
    )

    assert result["cond4_production"] is False
    assert result["verdict"] == "NO-GO"
    assert result["failed"] == ["cond4_production"]


def test_evaluate_sprint8_condition_thresholds_are_inclusive() -> None:
    """Các ngưỡng khóa là so sánh bao gồm: đúng biên vẫn phải GO."""
    result = evaluate_sprint8_go_no_go(
        _metrics_12(
            annualized_return_pct=8.0,
            sharpe=2.0,
            dsr_prob=0.95,
            maxdd_pct=5.0,
        ),
        _metrics_8(annualized_return_pct=6.0),
        _period_rows(6),
        live_executor_ready=True,
    )

    assert result["cond1_return"] is True
    assert result["cond2_consistency"] is True
    assert result["cond3_risk"] is True
    assert result["cond4_production"] is True
    assert result["verdict"] == "GO"
    assert result["failed"] == []


# ---------------------------------------------------------------------------
# F. summarize_periods
# ---------------------------------------------------------------------------


def test_summarize_periods_distribution_and_empty_input() -> None:
    """7 kỳ với 6 dương 1 âm: đếm đúng, ratio 6/7, worst < 0 < best."""
    rows = [
        {"net_return_pct": value}
        for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -2.0]
    ]

    summary = summarize_periods(rows)

    assert summary["n_periods"] == 7
    assert summary["n_positive"] == 6
    assert summary["positive_ratio"] == pytest.approx(6.0 / 7.0)
    assert summary["worst_period_return"] == pytest.approx(-2.0)
    assert summary["best_period_return"] == pytest.approx(6.0)
    assert summary["worst_period_return"] < 0.0
    assert summary["best_period_return"] > 0.0

    from_df = summarize_periods(pd.DataFrame(rows))
    assert from_df == summary

    for empty_input in (None, [], pd.DataFrame()):
        empty = summarize_periods(empty_input)
        assert empty["n_periods"] == 0
        assert empty["n_positive"] == 0
        assert empty["positive_ratio"] == 0.0
        assert empty["worst_period_return"] == 0.0
        assert empty["best_period_return"] == 0.0


# ---------------------------------------------------------------------------
# G. run_sprint8.py
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    """CLI mặc định phải khóa đúng khoảng dữ liệu và cấu hình 7 kỳ 180 ngày."""
    args = run_sprint8.parse_args([])

    assert args.start == "2023-05-07"
    assert args.end == "2026-10-18"
    assert args.timeframe == "1h"
    assert args.n_periods == 7
    assert args.period_days == 180
    assert args.raw_dir == "data/raw"
    assert args.out_dir == "data"
    assert args.download is False
    assert args.skip_baseline is False


def test_build_universe_frames_missing_cache_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """Thiếu parquet cache phải raise FileNotFoundError, không lỗi kiểu khác."""
    with pytest.raises(FileNotFoundError):
        run_sprint8.build_universe_frames(
            ["BTC/USDT:USDT"],
            "1h",
            tmp_path,
            "2023-05-07",
            "2023-05-10",
        )


def test_iter_dynamic_grid_locked_k12_parameters() -> None:
    """Lưới phải đúng 12 cấu hình với các tham số đóng băng theo docs Sprint 8."""
    configs = list(run_sprint8.iter_dynamic_grid())

    assert len(configs) == 12
    assert len({config_id for config_id, _cfg, _params in configs}) == 12

    expected_grid = {
        (top_k, w_basis, execution)
        for top_k in (2, 3)
        for w_basis in (0.0, 0.5, 1.0)
        for execution in ("real_taker", "passive_limit_maker")
    }
    actual_grid = {
        (params["top_k"], params["w_basis"], params["execution"])
        for _config_id, _cfg, params in configs
    }
    assert actual_grid == expected_grid

    for config_id, cfg, params in configs:
        assert cfg.entry_threshold == pytest.approx(0.00010)
        assert cfg.ema_span == 6
        assert cfg.structure == "spot_perp_3x"
        assert cfg.basis_ema_hours == 168
        assert cfg.expected_hold_periods == 21
        assert cfg.exit_threshold == pytest.approx(0.00002)
        assert cfg.min_hold_periods == 9
        assert cfg.buffer_rank == 1

        assert params["top_k"] == cfg.top_k
        assert params["w_basis"] == pytest.approx(cfg.w_basis)
        assert params["execution"] == cfg.execution
        assert params["structure"] == cfg.structure
        assert f"k{cfg.top_k}" in config_id
        assert cfg.execution in config_id
