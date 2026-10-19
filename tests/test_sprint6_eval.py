"""Unit tests cho module đánh giá Sprint 6 và CLI Runner run_sprint6.py.

Đảm bảo hoàn toàn offline, deterministic, thực thi nhanh (<30s), không gọi mạng,
không look-ahead bias và tuân thủ các ràng buộc Pandas 3.0.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import run_sprint6
from src.research import generate_synthetic_ohlcv
from src.sprint6_eval import (
    annualized_return_pct,
    annualized_sharpe,
    compute_deflated_sharpe_ratio,
    compute_newey_west_alpha_beta,
    evaluate_go_no_go,
    max_drawdown_pct,
    stitch_returns,
)


def test_annualized_sharpe() -> None:
    """Kiểm tra hàm annualized_sharpe xử lý an toàn chuỗi ngắn, std=0 và tính đúng công thức."""
    assert annualized_sharpe(pd.Series([], dtype=float)) == 0.0
    assert annualized_sharpe(pd.Series([0.01])) == 0.0

    # Chuỗi không biến động -> std = 0
    flat = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert annualized_sharpe(flat) == 0.0

    # Chuỗi có biến động
    s = pd.Series([0.01, -0.01, 0.02, -0.02])
    expected = (s.mean() / s.std(ddof=1)) * math.sqrt(8760)
    assert annualized_sharpe(s, periods_per_year=8760) == pytest.approx(expected)


def test_max_drawdown_pct() -> None:
    """Kiểm tra max_drawdown_pct tính chính xác mức sụt giảm vốn peak-to-trough."""
    assert max_drawdown_pct(pd.Series([], dtype=float)) == 0.0

    # Vốn tăng dần liên tục -> MaxDD = 0
    eq_up = pd.Series([100.0, 105.0, 110.0, 120.0])
    assert max_drawdown_pct(eq_up) == 0.0

    # Vốn từ 100 lên 200 rồi giảm xuống 150 -> DD = 25%
    eq_drop = pd.Series([100.0, 200.0, 150.0, 180.0])
    assert max_drawdown_pct(eq_drop) == pytest.approx(25.0)


def test_annualized_return_pct() -> None:
    """Kiểm tra annualized_return_pct tính đúng lãi kép năm hóa và xử lý biên an toàn."""
    assert annualized_return_pct(pd.Series([], dtype=float)) == 0.0

    # 1 nến lãi 10% trong năm 1 nến -> lãi năm hóa 10%
    ret_single = pd.Series([0.10])
    assert annualized_return_pct(ret_single, periods_per_year=1) == pytest.approx(10.0)

    # Tổng vốn âm hoặc bằng 0 -> trả về 0.0 an toàn
    ret_bankrupt = pd.Series([-1.0, 0.05])
    assert annualized_return_pct(ret_bankrupt) == 0.0


def test_dsr_bounds_and_monotonic() -> None:
    """Kiểm tra DSR: xác suất dsr_prob thuộc [0, 1], sr_star tăng và dsr_prob giảm khi K tăng."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=1000))

    trials = [1, 2, 5, 10, 20, 50, 100]
    prev_sr_star = -1.0
    prev_dsr = 2.0

    for k in trials:
        res = compute_deflated_sharpe_ratio(returns, num_trials=k, periods_per_year=8760)
        dsr = res["dsr_prob"]
        sr_star = res["sr_star"]

        # 1. dsr_prob luôn thuộc [0, 1]
        assert 0.0 <= dsr <= 1.0

        # 2. Khi K tăng, sr_star không giảm (đơn điệu tăng)
        assert sr_star >= prev_sr_star - 1e-12

        # 3. Khi K tăng, dsr_prob không tăng (đơn điệu giảm do bị phạt nhiều hơn)
        assert dsr <= prev_dsr + 1e-12

        prev_sr_star = sr_star
        prev_dsr = dsr


def test_dsr_formula_manual() -> None:
    """So khớp kết quả compute_deflated_sharpe_ratio với tính toán công thức độc lập."""
    s = pd.Series([0.012, -0.005, 0.018, 0.007, -0.003, 0.014, 0.002, -0.001])
    num_trials = 6
    res = compute_deflated_sharpe_ratio(s, num_trials=num_trials)

    n = len(s)
    sr = float(s.mean() / s.std(ddof=1))
    gamma3 = float(s.skew())
    gamma4 = float(s.kurt()) + 3.0
    denom = max(1.0 - gamma3 * sr + ((gamma4 - 1.0) / 4.0) * (sr ** 2), 1e-12)
    var_sr = denom / (n - 1.0)

    g = 0.5772156649
    e_const = 2.718281828
    k_val = 6
    term1 = (1.0 - g) * norm.ppf(1.0 - 1.0 / k_val)
    term2 = g * norm.ppf(1.0 - 1.0 / (k_val * e_const))
    expected_sr_star = math.sqrt(var_sr) * (term1 + term2)

    z = (sr - expected_sr_star) * math.sqrt(n - 1.0) / math.sqrt(denom)
    expected_dsr = float(norm.cdf(z))

    assert res["sharpe_per_period"] == pytest.approx(sr)
    assert res["sr_star"] == pytest.approx(expected_sr_star)
    assert res["dsr_prob"] == pytest.approx(expected_dsr)
    assert res["skew"] == pytest.approx(gamma3)
    assert res["kurt"] == pytest.approx(gamma4)
    assert res["n"] == n
    assert res["num_trials"] == num_trials


def test_nw_alpha_beta_recovers_beta() -> None:
    """Kiểm tra hồi quy Newey-West khôi phục chính xác hệ số beta từ dữ liệu tổng hợp."""
    rng = np.random.default_rng(123)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC").as_unit("ns")

    bench = pd.Series(rng.normal(0.0001, 0.01, size=n), index=dates)
    noise = rng.normal(0.0, 0.0001, size=n)
    strat = pd.Series(0.5 * bench.to_numpy() + noise, index=dates)

    res = compute_newey_west_alpha_beta(strat, bench, periods_per_year=8760)

    assert res["beta"] == pytest.approx(0.5, abs=0.02)
    assert np.isfinite(res["alpha_t"])
    assert np.isfinite(res["beta_t"])
    assert "alpha_annualized_pct" in res
    assert res["n"] == n


def test_stitch_no_reset() -> None:
    """Kiểm tra stitch_returns ghép các cửa sổ liên tục, sắp xếp thời gian và giữ bản cuối nếu trùng."""
    dates1 = pd.date_range("2024-01-01 00:00:00", periods=5, freq="1h", tz="UTC").as_unit("ns")
    dates2 = pd.date_range("2024-01-01 05:00:00", periods=5, freq="1h", tz="UTC").as_unit("ns")

    s1 = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates1)
    s2 = pd.Series([0.06, 0.07, 0.08, 0.09, 0.10], index=dates2)

    stitched = stitch_returns([s1, s2])

    assert len(stitched) == 10
    assert stitched.index.is_monotonic_increasing
    assert str(stitched.index.tz) == "UTC"
    assert stitched.iloc[0] == 0.01
    assert stitched.iloc[4] == 0.05
    assert stitched.iloc[5] == 0.06
    assert stitched.iloc[9] == 0.10

    # Kiểm tra xử lý trùng lặp: giữ bản cuối
    dates_dup = pd.date_range("2024-01-01 04:00:00", periods=2, freq="1h", tz="UTC").as_unit("ns")
    s_dup = pd.Series([0.99, 0.88], index=dates_dup)
    stitched_dup = stitch_returns([s1, s_dup])
    assert len(stitched_dup) == 6
    assert stitched_dup.loc[pd.Timestamp("2024-01-01 04:00:00", tz="UTC")] == 0.99


def test_evaluate_go_no_go() -> None:
    """Kiểm tra hàm evaluate_go_no_go theo 4 điều kiện bắt buộc và ngưỡng khác nhau cho carry vs xs."""
    # Bộ chỉ số chuẩn thỏa mãn cả 4 điều kiện của Carry
    pass_metrics = {
        "alpha_t": 2.8,
        "beta": 0.05,
        "sharpe": 1.60,
        "dsr_prob": 0.96,
        "holdout_return_pct": 5.0,
        "holdout_sharpe": 1.10,
        "holdout_max_dd": 10.0,
    }

    res_carry = evaluate_go_no_go(pass_metrics, direction="carry")
    assert res_carry["verdict"] == "GO"
    assert res_carry["cond1_market_neutral"] is True
    assert res_carry["cond2_sharpe"] is True
    assert res_carry["cond3_dsr"] is True
    assert res_carry["cond4_holdout"] is True

    # 1. Trượt điều kiện 1 (alpha_t < 2.50) -> NO-GO
    fail_c1 = dict(pass_metrics, alpha_t=2.1)
    assert evaluate_go_no_go(fail_c1, direction="carry")["verdict"] == "NO-GO"

    # 2. Trượt điều kiện 1 (abs(beta) > 0.20) -> NO-GO
    fail_beta = dict(pass_metrics, beta=0.25)
    assert evaluate_go_no_go(fail_beta, direction="carry")["verdict"] == "NO-GO"

    # 3. Trượt điều kiện 2: Sharpe = 1.20 -> Carry trượt (< 1.50) nhưng XS đạt (>= 1.00)
    metrics_sharpe_12 = dict(pass_metrics, sharpe=1.20)
    assert evaluate_go_no_go(metrics_sharpe_12, direction="carry")["verdict"] == "NO-GO"
    assert evaluate_go_no_go(metrics_sharpe_12, direction="xs")["verdict"] == "GO"

    # 4. Trượt điều kiện 3 (DSR < 0.95) -> NO-GO
    fail_dsr = dict(pass_metrics, dsr_prob=0.92)
    assert evaluate_go_no_go(fail_dsr, direction="carry")["verdict"] == "NO-GO"

    # 5. Ngưỡng MaxDD holdout: 15% -> Carry trượt (> 12%) nhưng XS đạt (<= 20%)
    metrics_dd_15 = dict(pass_metrics, sharpe=1.20, holdout_max_dd=15.0)
    assert evaluate_go_no_go(metrics_dd_15, direction="carry")["verdict"] == "NO-GO"
    assert evaluate_go_no_go(metrics_dd_15, direction="xs")["verdict"] == "GO"


def test_run_sprint6_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kiểm tra tích hợp toàn trình (smoke test) CLI run_sprint6 offline, sinh đủ CSV và Markdown."""
    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]

    # 1. Sinh dữ liệu OHLCV 1h tổng hợp bao phủ 2024-01-01 -> 2024-06-30 (~4400 nến)
    mock_ohlcv: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(symbols):
        df_sym = generate_synthetic_ohlcv(
            n_candles=4500,
            symbol=sym,
            timeframe="1h",
            seed=42 + idx,
        )
        df_sym.index.name = "datetime"
        mock_ohlcv[sym] = df_sym

    def mock_load(symbol: str, timeframe: str = "1h", days: int = 180, **kwargs: object) -> pd.DataFrame:
        return mock_ohlcv.get(symbol, list(mock_ohlcv.values())[0]).copy()

    # 2. Sinh dữ liệu funding rate định kỳ 8h từ 2024-01-01 đến 2024-07-02
    funding_times = pd.date_range("2024-01-01 00:00:00", periods=600, freq="8h", tz="UTC").as_unit("ns")
    mock_funding = pd.DataFrame(
        {
            "timestamp": (funding_times.asi8 // 1_000_000).astype("int64"),
            "datetime": funding_times,
            "funding_rate": 0.0001,
            "mark_price": 50000.0,
        }
    )

    def mock_fetch(symbol: str, start_date: str = "2023-05-07", end_date: str | None = None, **kwargs: object) -> pd.DataFrame:
        return mock_funding.copy()

    # 3. Áp dụng monkeypatch cho hàm tải dữ liệu trong run_sprint6
    monkeypatch.setattr(run_sprint6, "load_or_fetch_symbol_data", mock_load)
    monkeypatch.setattr(run_sprint6, "fetch_funding_history", mock_fetch)

    out_md = tmp_path / "r.md"
    ret = run_sprint6.main(
        [
            "--symbols",
            ",".join(symbols),
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-30",
            "--out-dir",
            str(tmp_path),
            "--out-md",
            str(out_md),
        ]
    )

    assert ret == 0
    assert (tmp_path / "sprint6_carry.csv").exists()
    assert (tmp_path / "sprint6_xs.csv").exists()
    assert out_md.exists()

    md_content = out_md.read_text(encoding="utf-8")
    assert "# Sprint 6 — Báo cáo nghiệm thu" in md_content
    assert "real_taker" in md_content

    # Kiểm tra cấu trúc file CSV
    carry_res = pd.read_csv(tmp_path / "sprint6_carry.csv")
    assert len(carry_res) == 6  # 2 structures * 3 fee scenarios
    assert "sharpe" in carry_res.columns
    assert "dsr_prob" in carry_res.columns

    xs_res = pd.read_csv(tmp_path / "sprint6_xs.csv")
    assert len(xs_res) == 12  # 2 modes * 3 lookbacks * 2 rebalances
    assert "sharpe" in xs_res.columns
    assert "dsr_prob" in xs_res.columns
