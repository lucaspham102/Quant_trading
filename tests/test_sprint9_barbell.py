"""Unit tests cho Cross-Exchange Carry và Barbell Portfolio (Sprint 9).

Phạm vi khóa hành vi:

- Ma trận phí ``maker_taker_hedge``: 10 bps/chiều Perp-Perp (20 bps khứ hồi),
  11.25 bps/chiều Spot-Perp (22.5 bps khứ hồi) và ngưỡng entry tối thiểu mỗi 8h.
- Rebalance ký quỹ liên sàn khi giá biến động +/-25%: có ghi nhận giao dịch
  ``rebalance`` và trừ phí 5 bps vào equity, không phát sinh margin call.
- Mô hình khớp lệnh Maker Post-Only: giá không đi qua mức treo thì Chân 1
  không khớp, không phát sinh phí, equity giữ nguyên vốn gốc.
- Barbell với ``enable_overlay=False`` phải trùng khít 100% equity curve của
  Carry và không tạo PnL overlay.

Toàn bộ dữ liệu được tổng hợp trong bộ nhớ thông qua
``align_multi_exchange_funding`` (đúng schema thật của module loader), không
gọi mạng và không đọc ``data/raw``. Mỗi test tự dựng dữ liệu riêng nên không
phụ thuộc thứ tự thực thi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.cross_exchange_carry import (
    CrossExchangeCarryConfig,
    run_cross_exchange_carry,
)
from src.multi_exchange_loader import align_multi_exchange_funding

SYMBOL = "AAA/USDT:USDT"
START = "2024-01-01 00:00:00"
INITIAL_CAPITAL = 3000.0


# ---------------------------------------------------------------------------
# Helpers dựng dữ liệu tổng hợp (offline, tất định)
# ---------------------------------------------------------------------------
def _hourly_index(periods: int) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC ns tên ``datetime`` với tần suất 1 giờ."""
    index = pd.date_range(START, periods=periods, freq="1h", tz="UTC").as_unit("ns")
    index.name = "datetime"
    return index


def _ohlcv_frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    """Dựng OHLCV giả quanh chuỗi ``close`` với biên high/low +/-5 bps.

    Biên high/low đủ rộng để lệnh Maker Post-Only (treo cao hơn close t-1
    đúng 0.5 bps) có thể khớp khi giá đi ngang, đồng thời vẫn giữ được khả
    năng "không khớp" khi giá giảm liên tục.
    """
    close_arr = np.asarray(close, dtype="float64")
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(len(close_arr), 1000.0, dtype="float64"),
        },
        index=index.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _funding_frame(index: pd.DatetimeIndex, rate: float) -> pd.DataFrame:
    """Dựng DataFrame funding 8h đúng schema ``timestamp/datetime/funding_rate/mark_price``.

    Các mốc được sinh từ 00:00 UTC của ngày đầu tiên, do đó rơi đúng vào
    khung thanh toán 00/08/16 UTC mà ``align_multi_exchange_funding`` kỳ vọng.
    """
    start = index[0].floor("D")
    n_events = int(np.ceil(len(index) / 8.0)) + 2
    event_index = pd.date_range(start, periods=n_events, freq="8h", tz="UTC").as_unit("ns")
    rates = np.full(len(event_index), float(rate), dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": (event_index.asi8 // 1_000_000).astype("int64"),
            "datetime": event_index,
            "funding_rate": rates,
            "mark_price": np.full(len(event_index), 100.0, dtype="float64"),
        }
    )


def _aligned_symbol(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    binance_rate: float,
    bybit_rate: float | None,
) -> dict[str, pd.DataFrame]:
    """Ghép OHLCV giả với funding Binance/Bybit qua loader thật.

    OKX cố ý vắng mặt để mô phỏng sàn chưa niêm yết (``ewma_funding_okx`` và
    ``funding_event_okx`` toàn NaN), đúng kịch bản 2 sàn của Sprint 9.
    """
    funding: dict[str, pd.DataFrame] = {
        "binance": _funding_frame(index, binance_rate),
    }
    if bybit_rate is not None:
        funding["bybit"] = _funding_frame(index, bybit_rate)

    aligned = align_multi_exchange_funding(
        SYMBOL,
        _ohlcv_frame(index, close),
        funding,
    )
    return {SYMBOL: aligned}


def _carry_config(**overrides: float | int | str) -> CrossExchangeCarryConfig:
    """Config Carry tối giản, mặc định ép chế độ Perp-vs-Perp để test tất định."""
    base: dict[str, float | int | str] = {
        "mode": "perp_perp_spread",
        "initial_capital": INITIAL_CAPITAL,
    }
    base.update(overrides)
    return CrossExchangeCarryConfig(**base)


def _rebalance_scenario() -> dict[str, pd.DataFrame]:
    """Kịch bản giá đi ngang 100, nhảy +26% lên 126 rồi +19% lên 150.

    Funding Binance 5 bps và Bybit 1 bps mỗi 8h tạo spread 4 bps -> đủ điều
    kiện mở vị thế Perp-Perp tại mốc 8h đầu tiên có EWMA hợp lệ.
    """
    n = 400
    index = _hourly_index(n)
    close = np.full(n, 100.0, dtype="float64")
    close[100:250] = 126.0  # +26% so với mốc tham chiếu 100 -> rebalance
    close[250:] = 150.0     # +19.05% so với 126 -> chưa đủ ngưỡng 25%
    return _aligned_symbol(index, close, binance_rate=0.0005, bybit_rate=0.0001)


def _no_fill_scenario() -> dict[str, pd.DataFrame]:
    """Kịch bản giá giảm 0.5% mỗi nến: high luôn thấp hơn giá treo lệnh short.

    Giá treo = close(t-1) * 1.00005, trong khi high(t) = close(t) * 1.0005 và
    close(t) = close(t-1) * 0.995, nên high(t) < giá treo tuyệt đối.
    """
    n = 200
    index = _hourly_index(n)
    steps = np.arange(n, dtype="float64")
    close = 100.0 * np.power(0.995, steps)
    return _aligned_symbol(index, close, binance_rate=0.0005, bybit_rate=0.0001)


# ---------------------------------------------------------------------------
# 1. Ma trận phí maker_taker_hedge
# ---------------------------------------------------------------------------
def test_maker_taker_hedge_cost_is_20bps_round_trip() -> None:
    """Chuẩn phí ``maker_taker_hedge`` đúng docs/go_no_go_sprint9.md.

    Chân 1 Maker 2 bps + adverse selection 1 bp, Chân 2 Taker 5 bps + slippage
    2 bps -> 10 bps/chiều, 20 bps khứ hồi. Spot-Perp dùng override 11.25
    bps/chiều -> 22.5 bps khứ hồi. Ngưỡng entry mỗi 8h = round-trip / 12 kỳ.
    """
    config = CrossExchangeCarryConfig()

    # Mặc định phải là maker_taker_hedge (chuẩn so sánh duy nhất khi GO/NO-GO).
    assert config.execution == "maker_taker_hedge"

    assert config.one_way_cost_perp_perp() == pytest.approx(0.0010)
    assert config.round_trip_cost_perp_perp() == pytest.approx(0.0020)
    assert config.one_way_cost_spot_perp() == pytest.approx(0.001125)
    assert config.round_trip_cost_spot_perp() == pytest.approx(0.00225)
    assert config.min_entry_rate_per_8h("perp_perp_spread") == pytest.approx(
        0.0020 / 12.0
    )

    # Đẳng thức kế toán: khứ hồi luôn gấp đôi một chiều.
    assert config.round_trip_cost_perp_perp() == pytest.approx(
        2.0 * config.one_way_cost_perp_perp()
    )
    assert config.round_trip_cost_spot_perp() == pytest.approx(
        2.0 * config.one_way_cost_spot_perp()
    )


# ---------------------------------------------------------------------------
# 2. Rebalance ký quỹ liên sàn khi giá biến động >= 25%
# ---------------------------------------------------------------------------
def test_cross_exchange_price_rebalance_at_25pct_charges_5bps_and_prevents_liquidation() -> None:
    """Giá vượt +/-25% phải kích hoạt rebalance, trừ 5 bps và không margin call.

    Vị thế Perp-Perp được mở khi giá phẳng 100. Sau cú nhảy +26% lên 126,
    engine phải ghi 1 dòng ``action='rebalance'``, cộng dồn ``rebalance_fees``
    dương và giữ ``Margin Calls = 0``. Cú nhảy +19.05% tiếp theo (126 -> 150)
    không đủ ngưỡng nên không tạo thêm rebalance bắt buộc.
    """
    data = _rebalance_scenario()
    result = run_cross_exchange_carry(data, _carry_config())
    metrics = result["metrics"]
    trades = result["trades_df"]

    # Vị thế phải thực sự được mở, nếu không phép thử rebalance vô nghĩa.
    assert metrics["Num Entries"] >= 1

    assert metrics["Price Rebalances"] >= 1
    assert metrics["Margin Calls"] == 0

    rebalance_rows = trades[
        trades["action"].astype(str).str.contains("rebalanc", case=False)
    ]
    assert len(rebalance_rows) >= 1
    assert rebalance_rows["fee"].gt(0.0).all()

    assert result["rebalance_fees"] > 0.0
    assert result["fees"] >= result["rebalance_fees"]
    assert np.isfinite(result["equity_curve"].to_numpy()).all()


# ---------------------------------------------------------------------------
# 3. Maker Post-Only không khớp khi giá không đi qua mức treo
# ---------------------------------------------------------------------------
def test_maker_post_only_no_fill_when_price_does_not_cross_limit() -> None:
    """Giá giảm liên tục: lệnh short Post-Only không khớp, không mất phí.

    Engine đặt lệnh Bán tại ``close(t-1) * (1 + 0.5 bps)`` với TTL 2 nến.
    Vì high(t) luôn thấp hơn mức treo, lệnh bị hủy sau TTL và được đếm vào
    ``Maker No-Fill Count``; không có entry nào và equity đứng yên ở vốn gốc.
    """
    data = _no_fill_scenario()
    result = run_cross_exchange_carry(data, _carry_config())
    metrics = result["metrics"]

    assert metrics["Maker No-Fill Count"] >= 1
    assert metrics["Num Entries"] == 0
    assert result["fees"] == pytest.approx(0.0)
    assert result["rebalance_fees"] == pytest.approx(0.0)
    assert result["equity_curve"].iloc[-1] == pytest.approx(INITIAL_CAPITAL)

    # Không có giao dịch mở/đóng nào được ghi nhận.
    trades = result["trades_df"]
    assert trades.empty or not trades["action"].astype(str).str.contains(
        "open|close", case=False, regex=True
    ).any()


# ---------------------------------------------------------------------------
# 4. Barbell tắt overlay phải trùng khít Carry
# ---------------------------------------------------------------------------
def test_barbell_overlay_off_matches_carry_exactly() -> None:
    """``enable_overlay=False``: equity Barbell trùng khít equity Carry.

    Đây là bất biến kế toán quan trọng: tắt overlay thì không được double-count
    vốn, không tạo PnL overlay và không phát sinh phí overlay.
    """
    data = _rebalance_scenario()
    carry_config = _carry_config()
    barbell = run_barbell_portfolio(
        data,
        BarbellConfig(enable_overlay=False, carry_config=carry_config),
    )

    pd.testing.assert_series_equal(
        barbell["equity_curve"],
        barbell["carry_result"]["equity_curve"],
        check_names=False,
    )
    np.testing.assert_allclose(
        barbell["equity_curve"].to_numpy(dtype="float64"),
        barbell["carry_result"]["equity_curve"].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-9,
    )

    assert barbell["overlay_pnl"] == pytest.approx(0.0)
    assert barbell["fees"] == pytest.approx(barbell["carry_result"]["fees"])

    overlay_pnl_series = barbell["attribution_series"]["overlay_pnl"]
    assert overlay_pnl_series.abs().max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Bộ kiểm định GO / NO-GO Sprint 9 (src/sprint9_eval.py)
# ---------------------------------------------------------------------------
import math

from src.sprint9_eval import (
    build_period_table_for_tier,
    compute_go_inputs,
    compute_ols_beta,
    detect_executor_state_machine,
    evaluate_sprint9_go_no_go,
)
from src.sprint7_eval import equity_from_returns


def _passing_full_metrics(**overrides: float) -> dict[str, float]:
    """Bộ chỉ số toàn chuỗi thỏa cả 4 tiêu chí GO theo docs/go_no_go_sprint9.md.

    Mặc định: annualized 12.5% (>= 10.0), Sharpe 2.1 (>= 1.80), MaxDD 4.5%
    (<= 8.0), |beta_BTC| = 0.05 (<= 0.15). Mỗi lần gọi trả về dict mới nên các
    nhánh vi phạm không thể mutate chéo dữ liệu của nhau.
    """
    base: dict[str, float] = {
        "annualized_return_pct": 12.5,
        "sharpe": 2.1,
        "maxdd_pct": 4.5,
        "beta_btc": 0.05,
    }
    base.update(overrides)
    return base


def _passing_period_rows() -> list[dict[str, float]]:
    """Bảng 7 kỳ 6 tháng: 6 kỳ dương (gồm P7 holdout) và 1 kỳ -1.5% (worst).

    P7 (holdout 2026-04-21 -> 2026-10-18) đạt net_return_pct = 2.0% và
    annualized_return_pct = 9.5% (>= 8.0). Worst period = -1.5% (>= -3.0).
    Trả về list mới mỗi lần gọi để tránh mutate chéo giữa các assert.
    """
    net_returns = [2.0, 1.4, -1.5, 1.9, 2.3, 1.1, 2.0]
    annualized = [10.5, 9.2, -6.0, 10.1, 11.0, 8.6, 9.5]
    return [
        {
            "period": f"P{i + 1}",
            "period_id": f"P{i + 1}",
            "net_return_pct": float(net),
            "annualized_return_pct": float(ann),
        }
        for i, (net, ann) in enumerate(zip(net_returns, annualized))
    ]


def test_evaluate_sprint9_go_no_go_all_conditions_and_holdout_gate() -> None:
    """Ma trận quyết định Sprint 9: GO khi đủ 4 tiêu chí, NO-GO đúng từng nhánh.

    Mỗi nhánh chỉ thay đổi đúng một tham số trên một bản copy độc lập, sau đó
    assert ``verdict == 'NO-GO'`` và ``failed`` chứa đúng tên tiêu chí bị trượt.
    Đặc biệt khóa hành vi holdout gate: kỳ cuối âm, annualized holdout < 8.0%
    hoặc worst period < -3.0% đều phải hạ ``cond2_consistency``.
    """
    baseline = evaluate_sprint9_go_no_go(
        _passing_full_metrics(), _passing_period_rows()
    )
    assert baseline["verdict"] == "GO"
    assert baseline["failed"] == []
    assert baseline["pos_periods"] == 6
    assert baseline["total_periods"] == 7
    assert baseline["holdout_net_return_pct"] == pytest.approx(2.0)
    assert baseline["holdout_annualized_return_pct"] == pytest.approx(9.5)
    assert baseline["details"]["worst_period_return_pct"] == pytest.approx(-1.5)

    def _evaluate(
        metrics_overrides: dict[str, float] | None = None,
        period_rows: list[dict[str, float]] | None = None,
        **kwargs: float | str | bool,
    ) -> dict:
        """Chạy đánh giá trên bản copy tươi của dữ liệu gốc."""
        metrics = _passing_full_metrics(**(metrics_overrides or {}))
        rows = _passing_period_rows() if period_rows is None else period_rows
        return evaluate_sprint9_go_no_go(metrics, rows, **kwargs)

    def _assert_failed(result: dict, expected: str) -> None:
        assert result["verdict"] == "NO-GO"
        assert result["failed"] == [expected]

    # a. Annualized toàn chuỗi 9.9% < 10.0% -> cond1_return.
    _assert_failed(
        _evaluate({"annualized_return_pct": 9.9}), "cond1_return"
    )

    # b. Chỉ còn 5/7 kỳ dương (P3 và P5 cùng âm) -> cond2_consistency.
    two_negative_rows = _passing_period_rows()
    two_negative_rows[4]["net_return_pct"] = -0.8
    result_b = _evaluate(period_rows=two_negative_rows)
    assert result_b["pos_periods"] == 5
    _assert_failed(result_b, "cond2_consistency")

    # c. Kỳ holdout cuối âm (-0.5%) -> cond2_consistency.
    negative_holdout_rows = _passing_period_rows()
    negative_holdout_rows[-1]["net_return_pct"] = -0.5
    _assert_failed(
        _evaluate(period_rows=negative_holdout_rows), "cond2_consistency"
    )

    # d. Holdout annualized 7.9% < 8.0% -> cond2_consistency.
    weak_holdout_rows = _passing_period_rows()
    weak_holdout_rows[-1]["annualized_return_pct"] = 7.9
    _assert_failed(
        _evaluate(period_rows=weak_holdout_rows), "cond2_consistency"
    )

    # e. Worst period -3.5% < -3.0% -> cond2_consistency.
    deep_loss_rows = _passing_period_rows()
    deep_loss_rows[2]["net_return_pct"] = -3.5
    _assert_failed(_evaluate(period_rows=deep_loss_rows), "cond2_consistency")

    # f. Sharpe 1.79 < 1.80 -> cond3_risk.
    _assert_failed(_evaluate({"sharpe": 1.79}), "cond3_risk")

    # g. MaxDD 8.5% > 8.0% -> cond3_risk.
    _assert_failed(_evaluate({"maxdd_pct": 8.5}), "cond3_risk")

    # h. |beta_BTC| 0.18 > 0.15 -> cond3_risk.
    _assert_failed(_evaluate({"beta_btc": -0.18}), "cond3_risk")

    # i. Có margin call -> cond3_risk (truyền qua keyword).
    _assert_failed(_evaluate(margin_call_count=1), "cond3_risk")

    # j. execution_mode không phải maker_taker_hedge -> cond4_execution_sync.
    _assert_failed(_evaluate(execution_mode="real_taker"), "cond4_execution_sync")

    # k. Chưa xác thực máy trạng thái executor -> cond4_execution_sync.
    _assert_failed(
        _evaluate(executor_state_machine_verified=False), "cond4_execution_sync"
    )

    # l. Còn look-ahead bias -> cond4_execution_sync.
    _assert_failed(_evaluate(has_lookahead_bias=True), "cond4_execution_sync")

    # m. Bảng kỳ rỗng -> cond2_consistency (không đủ 7 kỳ).
    result_m = _evaluate(period_rows=[])
    assert result_m["total_periods"] == 0
    _assert_failed(result_m, "cond2_consistency")

    # Dữ liệu gốc không bị mutate bởi bất kỳ nhánh nào ở trên.
    assert _passing_full_metrics()["annualized_return_pct"] == pytest.approx(12.5)
    assert len(_passing_period_rows()) == 7
    assert _passing_period_rows()[-1]["net_return_pct"] == pytest.approx(2.0)


def test_compute_ols_beta_and_go_inputs_math() -> None:
    """OLS beta đúng độ dốc và các chỉ số GO (net PnL, MaxDD, annualized) hợp lý.

    Khóa công thức: beta = cov(ret, bench) / var(bench) với benchmark biến thiên
    hằng số phải trả 0.0 (không NaN); equity tăng đơn điệu phải có MaxDD = 0 và
    Net PnL = final_equity - initial_capital.
    """
    rng = np.random.default_rng(7)
    n = 200
    bench_values = rng.normal(0.0, 0.01, n)
    benchmark = pd.Series(bench_values, dtype="float64")

    noise = pd.Series(rng.normal(0.0, 0.001, n), dtype="float64")
    returns_half = 0.5 * benchmark + noise
    assert compute_ols_beta(returns_half, benchmark) == pytest.approx(0.5, abs=0.05)

    returns_short = -0.3 * benchmark
    assert compute_ols_beta(returns_short, benchmark) == pytest.approx(-0.3, abs=0.05)

    constant_benchmark = pd.Series(np.full(n, 0.001), dtype="float64")
    beta_const = compute_ols_beta(returns_half, constant_benchmark)
    assert beta_const == pytest.approx(0.0)
    assert math.isfinite(beta_const)

    # Equity tăng đều ~1%/ngày (24 nến/ngày) bắt đầu từ 3000 USDT.
    n_bars = 24 * 10
    index = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    per_bar = (1.01 ** (1.0 / 24.0)) - 1.0
    returns = pd.Series(np.full(n_bars, per_bar), index=index, dtype="float64")
    equity = equity_from_returns(returns, initial_capital=INITIAL_CAPITAL)
    benchmark_indexed = pd.Series(
        rng.normal(0.0, 0.001, n_bars), index=index, dtype="float64"
    )

    go = compute_go_inputs(
        returns=returns,
        equity_curve=equity,
        benchmark_returns=benchmark_indexed,
        initial_capital=INITIAL_CAPITAL,
    )

    assert go["final_equity"] > INITIAL_CAPITAL
    assert go["net_pnl"] == pytest.approx(go["final_equity"] - INITIAL_CAPITAL)
    assert go["maxdd_pct"] == pytest.approx(0.0, abs=1e-6)
    assert go["annualized_return_pct"] > 0.0
    assert math.isfinite(go["beta_btc"])
    assert go["n_bars"] == n_bars

    # Cú sụt -50% giữa chuỗi phải cho MaxDD >= 49.0% (quy ước số dương).
    drop_values = np.concatenate(
        [
            np.full(n_bars // 2 - 1, 0.001),
            np.array([-0.5]),
            np.full(n_bars - n_bars // 2, 0.001),
        ]
    )
    drop_returns = pd.Series(drop_values, index=index, dtype="float64")
    drop_equity = equity_from_returns(drop_returns, initial_capital=INITIAL_CAPITAL)
    drop_go = compute_go_inputs(
        returns=drop_returns,
        equity_curve=drop_equity,
        benchmark_returns=benchmark_indexed,
        initial_capital=INITIAL_CAPITAL,
    )
    assert drop_go["maxdd_pct"] >= 49.0


def test_build_period_table_for_tier_is_compounded_and_seven_periods() -> None:
    """Bảng 7 kỳ phải nối tiếp lãi kép: vốn đầu kỳ sau = vốn cuối kỳ trước.

    Chuỗi returns 1h dương đều +0.00001 kéo dài đúng 7 * 180 * 24 nến từ
    2023-05-07 UTC. Kiểm tra số kỳ, nhãn P1..P7, toàn bộ net_return_pct > 0 và
    ``start_capital`` của P2 bằng ``final_equity`` của P1 (sai số 1e-6) để chứng
    minh không reset vốn về 3000 USDT ở mỗi kỳ.
    """
    n_bars = 7 * 180 * 24
    index = pd.date_range("2023-05-07", periods=n_bars, freq="1h", tz="UTC")
    returns = pd.Series(np.full(n_bars, 0.00001), index=index, dtype="float64")
    benchmark = pd.Series(np.zeros(n_bars), index=index, dtype="float64")

    table = build_period_table_for_tier(
        returns=returns,
        benchmark_returns=benchmark,
        n_periods=7,
        period_days=180,
        initial_capital=INITIAL_CAPITAL,
    )

    assert len(table) == 7
    assert list(table["period"]) == [f"P{i}" for i in range(1, 8)]
    assert (table["net_return_pct"] > 0.0).all()
    assert (table["final_equity"] > table["start_capital"]).all()
    assert table["start_capital"].iloc[0] == pytest.approx(INITIAL_CAPITAL)

    for i in range(1, 7):
        assert table["start_capital"].iloc[i] == pytest.approx(
            table["final_equity"].iloc[i - 1], abs=1e-6
        )

    holdout_ann = float(table["annualized_return_pct"].iloc[-1])
    assert math.isfinite(holdout_ann)
    assert holdout_ann > 0.0


def test_detect_executor_state_machine_true() -> None:
    """CarryLiveExecutor phải có execution_style mặc định và đủ 2 bước máy trạng thái.

    ``detect_executor_state_machine`` trả True khi ``__init__`` có tham số
    ``execution_style='maker_taker_hedge'`` và lớp cài đặt cả
    ``place_maker_leg1`` lẫn ``verify_maker_fill_and_hedge_leg2``.
    """
    assert detect_executor_state_machine() is True
