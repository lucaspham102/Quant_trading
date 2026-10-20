"""Unit tests cho ``src/sprint11_eval.py`` (Sprint 11 GO/NO-GO evaluation).

Phạm vi khóa hành vi
--------------------
- ``compute_beta_to_benchmark``: beta OLS ddof=1 trên phần giao index, trả 0.0
  khi benchmark hằng số / chuỗi rỗng.
- ``split_period_returns``: chia 700 giờ thành 7 đoạn, bảo toàn tích lợi nhuận
  (sai số < 1e-9) và dấu của từng đoạn.
- ``evaluate_sprint11_go_no_go``: chọn primary combo, đủ 10 tiêu chí GO, fallback
  tier1a, không raise với payload rỗng, và cờ executor do caller truyền vào.
- ``build_sprint11_summary_rows``: đầu ra phẳng, cùng số dòng với payload.
- ``smoke_test_executor``: executor paper chạy được trên stub, idempotent và chỉ
  ghi file trong ``tmp_path``.
- ``run_sprint11_tiers``: lưới 4 combo trên dữ liệu tổng hợp 2 symbol x 200 giờ,
  không raise và khép kín kế toán cho các dòng không lỗi.

Nguyên tắc offline
------------------
Không gọi mạng, không sleep, không đọc parquet. Mọi dữ liệu được tổng hợp trong
bộ nhớ; file (nếu có) chỉ nằm trong ``tmp_path``. Mỗi test chạy dưới 3 giây,
riêng test lưới combo 4 cấu hình giữ dưới 20 giây.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.sprint11_eval import (
    build_sprint11_summary_rows,
    compute_beta_to_benchmark,
    evaluate_sprint11_go_no_go,
    run_sprint11_tiers,
    smoke_test_executor,
    split_period_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hourly_index(periods: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC tần suất 1 giờ, tên ``datetime``, độ phân giải ns."""
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC").as_unit("ns")
    index.name = "datetime"
    return index


def _good_row(**overrides: Any) -> dict[str, Any]:
    """Dòng combo đạt *mọi* tiêu chí GO theo docs/go_no_go_sprint11.md.

    Giá trị mặc định bám sát kịch bản Sprint 11: 10.36%/năm, 7/7 kỳ dương,
    holdout +4.0%, Sharpe 1.86, MaxDD 4.44%, |beta| 0.003, basis std 0.4,
    accounting gap 1e-10 và 100 sự kiện lượng hóa lot-size.
    """
    row: dict[str, Any] = {
        "combo": "tier1b/smooth_conviction",
        "annualized_return_pct": 10.36,
        "positive_periods": 7,
        "holdout_return_pct": 4.0,
        "holdout_return_resim_pct": 4.0,
        "holdout_carry_pnl": 50.0,
        "holdout_overlay_net_pnl": 60.0,
        "sharpe": 1.86,
        "maxdd_pct": 4.44,
        "beta_btc": 0.003,
        "basis_mtm_std": 0.4,
        "accounting_gap": 1e-10,
        "overlay_quantization_events": 100,
        "executor_functional": True,
        "error": None,
    }
    row.update(overrides)
    return row


def _synthetic_two_symbol_data(periods: int = 200) -> dict[str, pd.DataFrame]:
    """Dữ liệu 1h tổng hợp cho BTC/ETH, phủ qua mốc split 2026-04-21.

    Chuỗi bắt đầu 2026-04-20 nên cửa sổ holdout (từ 2026-04-21) vẫn còn dữ
    liệu, tránh nhánh xử lý DataFrame rỗng không liên quan tới mục tiêu test.
    """
    index = _hourly_index(periods, start="2026-04-20")
    steps = np.arange(periods, dtype="float64")
    noise = 1e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    specs = (
        ("BTC/USDT:USDT", 0.0004, 60000.0),
        ("ETH/USDT:USDT", -0.0002, 3000.0),
    )

    data: dict[str, pd.DataFrame] = {}
    for i, (sym, drift, start) in enumerate(specs):
        close = start * np.cumprod(1.0 + drift + noise)
        basis = 0.0005 * np.sin(2.0 * np.pi * steps / 24.0 + 0.2 * i)
        data[sym] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.0005,
                "low": close * 0.9995,
                "close": close,
                "perp_close": close,
                "spot_close": close / (1.0 + basis),
                "volume": np.full(periods, 1000.0, dtype="float64"),
                "basis_pct": basis,
                "funding_event_binance": np.full(periods, 0.0005, dtype="float64"),
                "funding_rate_event": np.full(periods, 0.0005, dtype="float64"),
                "ewma_funding_binance": np.full(periods, 0.0005, dtype="float64"),
            },
            index=index.copy(),
        )
    return data


# ---------------------------------------------------------------------------
# 1. compute_beta_to_benchmark
# ---------------------------------------------------------------------------
def test_compute_beta_recovers_known_exposure() -> None:
    """r = 2*b + nhiễu nhỏ -> beta xấp xỉ 2.0 với sai số tuyệt đối < 0.05."""
    rng = np.random.default_rng(11)
    index = _hourly_index(1000)
    benchmark = pd.Series(rng.normal(0.0, 0.01, 1000), index=index)
    noise = pd.Series(rng.normal(0.0, 1e-4, 1000), index=index)
    returns = 2.0 * benchmark + noise

    beta = compute_beta_to_benchmark(returns, benchmark)

    assert math.isfinite(beta)
    assert abs(beta - 2.0) < 0.05


def test_compute_beta_returns_zero_for_constant_benchmark() -> None:
    """Benchmark hằng số có variance bằng 0 -> beta quy ước 0.0, không chia 0."""
    rng = np.random.default_rng(23)
    index = _hourly_index(300)
    returns = pd.Series(rng.normal(0.0, 0.001, 300), index=index)
    benchmark = pd.Series(0.001, index=index, dtype="float64")

    assert compute_beta_to_benchmark(returns, benchmark) == 0.0


def test_compute_beta_uses_only_overlapping_index() -> None:
    """Index lệch nhau: chỉ 5 điểm giao được dùng, đủ khôi phục beta chính xác 2.0."""
    index = _hourly_index(20)
    benchmark = pd.Series(np.arange(1.0, 11.0), index=index[5:15])
    returns = pd.Series(np.arange(-5.0, 5.0), index=index[:10])
    overlap_ts = index[5:10]
    returns.loc[overlap_ts] = 2.0 * benchmark.loc[overlap_ts]

    beta = compute_beta_to_benchmark(returns, benchmark)

    assert abs(beta - 2.0) < 1e-9


def test_compute_beta_returns_zero_for_empty_series() -> None:
    """Chuỗi rỗng (một hoặc cả hai phía) -> 0.0, không raise."""
    empty = pd.Series(dtype="float64")
    non_empty = pd.Series([0.01, -0.02], index=_hourly_index(2))

    assert compute_beta_to_benchmark(empty, empty) == 0.0
    assert compute_beta_to_benchmark(empty, non_empty) == 0.0
    assert compute_beta_to_benchmark(non_empty, empty) == 0.0


# ---------------------------------------------------------------------------
# 2. split_period_returns
# ---------------------------------------------------------------------------
def test_split_period_returns_conserves_compounded_return() -> None:
    """700 giờ -> 7 đoạn 100 giờ; tích các đoạn khớp tích toàn chuỗi < 1e-9.

    Dùng returns dương hoàn toàn để đồng thời khẳng định ``return_pct`` của
    từng đoạn đều dương.
    """
    rng = np.random.default_rng(7)
    index = _hourly_index(700)
    returns = pd.Series(rng.uniform(5e-5, 3.5e-4, 700), index=index)

    periods = split_period_returns(returns, 7)

    assert len(periods) == 7
    for expected_period, row in enumerate(periods, start=1):
        assert set(row) == {"period", "start", "end", "return_pct"}
        assert row["period"] == expected_period
        assert isinstance(row["start"], str) and isinstance(row["end"], str)
        assert row["return_pct"] > 0.0

    assert periods[0]["start"] == index[0].isoformat()
    assert periods[-1]["end"] == index[-1].isoformat()

    total_factor = float(np.prod(1.0 + returns.to_numpy(dtype="float64")))
    split_factor = float(
        np.prod([1.0 + float(row["return_pct"]) / 100.0 for row in periods])
    )

    assert split_factor == pytest.approx(total_factor, rel=0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. evaluate_sprint11_go_no_go
# ---------------------------------------------------------------------------
def test_evaluate_go_when_all_criteria_pass() -> None:
    """Primary tier1b/smooth_conviction đạt đủ 10 tiêu chí -> GO, reasons rỗng."""
    result = evaluate_sprint11_go_no_go({"rows": [_good_row()]})

    assert result["go"] is True
    assert result["reasons"] == []
    assert result["primary_combo"] == "tier1b/smooth_conviction"
    assert len(result["criteria"]) >= 10
    names = {criterion["name"] for criterion in result["criteria"]}
    assert {"holdout_consistency", "holdout_consistency_resim"} <= names
    assert all(criterion["passed"] for criterion in result["criteria"])


def test_evaluate_no_go_when_holdout_return_below_threshold() -> None:
    """Holdout 3.24% < 3.5% -> NO-GO, reason phải nêu rõ tiêu chí holdout."""
    result = evaluate_sprint11_go_no_go(
        {"rows": [_good_row(holdout_return_pct=3.24)]}
    )

    assert result["go"] is False
    assert any("oldout" in reason for reason in result["reasons"])


def test_evaluate_no_go_when_annualized_return_below_threshold() -> None:
    """Annualized 8.54% < 9.0% -> NO-GO."""
    result = evaluate_sprint11_go_no_go(
        {"rows": [_good_row(annualized_return_pct=8.54)]}
    )

    assert result["go"] is False
    assert any("AnnRet" in reason for reason in result["reasons"])


def test_evaluate_empty_payload_does_not_raise_and_is_no_go() -> None:
    """Payload rỗng -> dùng chính payload làm primary, mọi chỉ số 0, NO-GO."""
    result = evaluate_sprint11_go_no_go({"rows": []})

    assert result["go"] is False
    assert result["primary_combo"] == "tier1b/smooth_conviction"
    assert result["reasons"]


def test_evaluate_falls_back_to_tier1a_primary() -> None:
    """Chỉ có row tier1a/smooth_conviction -> chọn fallback làm primary, không raise."""
    row = _good_row(combo="tier1a/smooth_conviction")

    result = evaluate_sprint11_go_no_go({"rows": [row]})

    assert result["primary_combo"] == "tier1a/smooth_conviction"
    assert result["go"] is True


def test_evaluate_executor_flag_override_forces_no_go() -> None:
    """Tham số executor_functional=False thắng cờ trong row -> tiêu chí executor FAIL."""
    result = evaluate_sprint11_go_no_go(
        {"rows": [_good_row()]}, executor_functional=False
    )

    assert result["go"] is False
    executor_criterion = next(
        criterion
        for criterion in result["criteria"]
        if criterion["name"] == "executor_functional"
    )
    assert executor_criterion["passed"] is False
    assert executor_criterion["value"] is False


# ---------------------------------------------------------------------------
# 4. build_sprint11_summary_rows
# ---------------------------------------------------------------------------
def test_build_summary_rows_is_flat_and_same_length() -> None:
    """Mọi dòng đầu ra là dict phẳng, không còn list/dict lồng, giữ nguyên số dòng."""
    nested_periods = [
        {"period": 1, "start": "2024-01-01", "end": "2024-03-01", "return_pct": 1.5}
    ]
    rows = [
        _good_row(periods=nested_periods),
        _good_row(combo="tier1a/hard_binary", periods=nested_periods),
    ]

    summary = build_sprint11_summary_rows({"rows": rows})

    assert len(summary) == len(rows)
    for row in summary:
        assert isinstance(row, dict)
        assert all(not isinstance(value, (list, dict)) for value in row.values())
        assert row["combo"] in {"tier1b/smooth_conviction", "tier1a/hard_binary"}


# ---------------------------------------------------------------------------
# 5. smoke_test_executor
# ---------------------------------------------------------------------------
_SMOKE_KEYS = {
    "methods_present",
    "state_fields_present",
    "carry_routed",
    "overlay_routed",
    "executor_functional",
}


def test_smoke_test_executor_all_green_inside_tmp_path(tmp_path: Path) -> None:
    """Executor paper định tuyến cả carry lẫn overlay, không ghi file ra ngoài tmp_path."""
    cwd_before = {path.name for path in Path.cwd().iterdir()}

    result = smoke_test_executor(tmp_path)

    cwd_after = {path.name for path in Path.cwd().iterdir()}
    assert cwd_after == cwd_before
    assert _SMOKE_KEYS.issubset(result)
    assert all(result[key] is True for key in _SMOKE_KEYS)
    assert "error" not in result

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert tmp_path in path.parents


def test_smoke_test_executor_is_idempotent(tmp_path: Path) -> None:
    """Chạy 2 lần liên tiếp trên cùng thư mục -> cả hai lần đều trả toàn True."""
    first = smoke_test_executor(tmp_path)
    second = smoke_test_executor(tmp_path)

    for result in (first, second):
        assert _SMOKE_KEYS.issubset(result)
        assert all(result[key] is True for key in _SMOKE_KEYS)


# ---------------------------------------------------------------------------
# 6. run_sprint11_tiers (dữ liệu tổng hợp nhỏ)
# ---------------------------------------------------------------------------
def test_run_sprint11_tiers_returns_four_rows_with_closed_accounting() -> None:
    """2 symbol x 200 giờ -> đủ 4 combo, không raise, gap kế toán < 1e-6 khi có kết quả."""
    data = _synthetic_two_symbol_data(periods=200)

    result = run_sprint11_tiers(data)

    rows = result["rows"]
    assert len(rows) == 4
    assert result["n_combos"] == 4
    assert {row["combo"] for row in rows} == {
        "tier1a/hard_binary",
        "tier1a/smooth_conviction",
        "tier1b/hard_binary",
        "tier1b/smooth_conviction",
    }
    for row in rows:
        assert "accounting_gap" in row
        if row.get("error") is None:
            assert float(row["accounting_gap"]) < 1e-6
