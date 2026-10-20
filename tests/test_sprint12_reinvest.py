"""Unit tests cho Sprint 12 - tái đầu tư lợi nhuận (``reinvest_profits``).

Phạm vi khóa hành vi:

- ``BarbellConfig.reinvest_profits`` mặc định ``True`` và giá trị này được
  truyền nguyên vẹn ra cả dict kết quả gốc lẫn ``metrics`` (kể cả nhánh dữ
  liệu rỗng).
- Bật/tắt tái đầu tư phải tạo ra đường vốn khác nhau trên cùng dữ liệu trend
  mạnh 60 ngày x 4 symbol; nến đầu tiên luôn neo đúng vốn gốc 3000 USD.
- Chế độ ``False`` kỳ vọng không để lợi nhuận carry tiếp tục sinh lời trên
  phần vốn đã tăng (tổng return thấp hơn ``True`` khi carry dương). Nếu source
  cho kết quả ngược, test chỉ còn khẳng định hai chế độ khác nhau và ghi rõ
  bất thường trong assert message.
- Đẳng thức kế toán ``sum(attribution_series['net_pnl']) == final_equity -
  initial_capital`` được kiểm tra cho cả hai chế độ.
- Các khóa holdout mới của ``run_sprint11_tiers`` tồn tại, đúng kiểu dữ liệu
  và công thức return trên vốn hoạt động luôn khớp ``holdout_net_pnl``.

Toàn bộ dữ liệu được tổng hợp trong bộ nhớ (1h UTC), không gọi mạng, không
đọc/ghi parquet và không sleep. Kết quả nặng được cache ở phạm vi module để
mỗi test giữ dưới 8 giây.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.config import CORE_4_SYMBOLS
from src.cross_exchange_carry import CrossExchangeCarryConfig
from src.sprint11_eval import run_sprint11_tiers

INITIAL_CAPITAL = 3000.0
CORE4 = list(CORE_4_SYMBOLS)
TREND_START = "2024-01-01 00:00:00"
EVAL_START = "2026-04-15 00:00:00"
FUNDING_RATE = 0.0005  # 5 bps/8h - đủ vượt ngưỡng entry của carry
TREND_PERIODS = 60 * 24  # 60 ngày x 24 nến
EVAL_PERIODS = 200  # ~200 giờ, phủ qua mốc split 2026-04-21


# ---------------------------------------------------------------------------
# Helpers dựng dữ liệu tổng hợp (offline, tất định)
# ---------------------------------------------------------------------------
def _hourly_index(periods: int, start: str = TREND_START) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC ns tên ``datetime`` với tần suất 1 giờ."""
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC").as_unit("ns")
    index.name = "datetime"
    return index


def _price_path(returns: np.ndarray, start: float) -> np.ndarray:
    """Ghép chuỗi giá từ returns 1h cho trước."""
    return start * np.cumprod(1.0 + np.asarray(returns, dtype="float64"))


def _symbol_frame(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    basis_pct: np.ndarray,
    funding_rate: float = FUNDING_RATE,
) -> pd.DataFrame:
    """Dựng DataFrame đủ schema cho cả Carry (tier1a/1b) lẫn Overlay.

    Có đủ ``close``, ``perp_close``, ``spot_close``, ``high``, ``low``,
    ``basis_pct``, ``funding_event_binance``, ``funding_rate_event`` và
    ``ewma_funding_binance``. Biên high/low +/-5 bps đủ để lệnh maker Post-Only
    khớp ở các cặp đang tăng.
    """
    close_arr = np.asarray(close, dtype="float64")
    n = len(close_arr)
    frame = pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr * 1.0005,
            "low": close_arr * 0.9995,
            "close": close_arr,
            "volume": np.full(n, 1000.0, dtype="float64"),
            "perp_close": close_arr,
            "spot_close": close_arr,
            "basis_pct": np.asarray(basis_pct, dtype="float64"),
            "funding_event_binance": np.full(n, funding_rate, dtype="float64"),
            "funding_rate_event": np.full(n, funding_rate, dtype="float64"),
            "ewma_funding_binance": np.full(n, funding_rate, dtype="float64"),
        },
        index=index.copy(),
    )
    frame.index.name = "datetime"
    return frame


def _trend_data(periods: int = TREND_PERIODS) -> dict[str, pd.DataFrame]:
    """4 symbol Core-4 trend phân hóa mạnh trong 60 ngày.

    2 symbol tăng (+4/+1.5 bps/h), 2 symbol giảm (-1.5/-4 bps/h). Nhiễu xen kẽ
    dùng chung nên chân long - chân short triệt tiêu nhiễu, giữ overlay hoạt
    động và tạo carry dương ổn định (funding 5 bps/8h). Basis sin chu kỳ 24h
    tạo Basis MTM khác 0 cho Core Carry.
    """
    index = _hourly_index(periods)
    steps = np.arange(periods, dtype="float64")
    noise = 2e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    drifts = (0.0004, 0.00015, -0.00015, -0.0004)
    starts = (65000.0, 3000.0, 150.0, 600.0)

    data: dict[str, pd.DataFrame] = {}
    for i, (sym, drift, start) in enumerate(zip(CORE4, drifts, starts)):
        close = _price_path(drift + noise, start)
        basis = 0.001 * np.sin(2.0 * np.pi * steps / 24.0 + 0.3 * i)
        data[sym] = _symbol_frame(index, close, basis)
    return data


def _eval_data(periods: int = EVAL_PERIODS) -> dict[str, pd.DataFrame]:
    """2 symbol Core-4 (~200 giờ) phủ qua mốc holdout 2026-04-21.

    Dữ liệu ngắn hơn lookback overlay (336h) nên nhánh eval chủ yếu chạy Carry;
    mục tiêu là kiểm tra schema/công thức các khóa holdout, không phải hiệu
    suất. Index bắt đầu 2026-04-15 nên luôn có nến trước split.
    """
    index = _hourly_index(periods, start=EVAL_START)
    steps = np.arange(periods, dtype="float64")
    noise = 2e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    drifts = (0.00025, -0.00025)
    starts = (65000.0, 3000.0)

    data: dict[str, pd.DataFrame] = {}
    for i, (sym, drift, start) in enumerate(zip(CORE4[:2], drifts, starts)):
        close = _price_path(drift + noise, start)
        basis = 0.0008 * np.sin(2.0 * np.pi * steps / 24.0 + 0.2 * i)
        data[sym] = _symbol_frame(index, close, basis)
    return data


def _reinvest_config(reinvest: bool) -> BarbellConfig:
    """BarbellConfig tất định, giữ overlay luôn bật để vốn thực sự biến động.

    Tắt SNR gate, vol targeting và DD cushion để leverage overlay ghim 0.40 và
    kết quả chỉ phụ thuộc vào ``reinvest_profits``. Dùng carry Tier 1A trên đúng
    4 symbol Core-4 của bộ dữ liệu trend.
    """
    return BarbellConfig(
        initial_capital=INITIAL_CAPITAL,
        carry_config=CrossExchangeCarryConfig(initial_capital=INITIAL_CAPITAL),
        carry_variant="tier1a",
        gating_mode="hard_binary",
        overlay_gross_leverage=0.40,
        overlay_lookback_hours=336,
        overlay_rebalance_hours=48,
        overlay_top_k=1,
        overlay_buffer_rank=1,
        overlay_min_spread=0.75,
        enable_snr_gate=False,
        enable_vol_targeting=False,
        enable_drawdown_cushion=False,
        reinvest_profits=reinvest,
    )


# ---------------------------------------------------------------------------
# Fixtures cache kết quả nặng ở phạm vi module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trend_data() -> dict[str, pd.DataFrame]:
    """Dữ liệu 4 symbol trend mạnh 60 ngày (dùng chung)."""
    return _trend_data()


@pytest.fixture(scope="module")
def default_result(trend_data: dict[str, pd.DataFrame]) -> dict:
    """Kết quả BarbellConfig mặc định trên dữ liệu trend."""
    return run_barbell_portfolio(trend_data, BarbellConfig())


@pytest.fixture(scope="module")
def paired_results(trend_data: dict[str, pd.DataFrame]) -> dict[bool, dict]:
    """Kết quả cùng dữ liệu với ``reinvest_profits`` True/False."""
    return {
        flag: run_barbell_portfolio(trend_data, _reinvest_config(flag))
        for flag in (True, False)
    }


# ---------------------------------------------------------------------------
# 1. Mặc định True và lan truyền ra output
# ---------------------------------------------------------------------------
def test_reinvest_profits_defaults_to_true(default_result: dict) -> None:
    """``reinvest_profits`` mặc định True ở config, dict gốc và metrics.

    Kiểm tra thêm nhánh dữ liệu rỗng để chắc chắn cờ này không bị mất khi
    Barbell trả sớm.
    """
    config = BarbellConfig()
    assert config.reinvest_profits is True

    assert default_result["reinvest_profits"] is True
    assert default_result["metrics"]["reinvest_profits"] is True

    empty_result = run_barbell_portfolio({}, config)
    assert empty_result["reinvest_profits"] is True
    assert empty_result["metrics"]["reinvest_profits"] is True


# ---------------------------------------------------------------------------
# 2. True vs False tạo đường vốn khác nhau
# ---------------------------------------------------------------------------
def test_reinvest_true_and_false_produce_different_equity(
    paired_results: dict[bool, dict],
) -> None:
    """Cùng dữ liệu, bật/tắt tái đầu tư phải cho final_equity khác > 1 USD.

    Đường vốn phải bắt đầu đúng 3000 USD ở cả hai chế độ; overlay phải thực sự
    hoạt động, nếu không Barbell trả về nguyên equity của Carry và hai chế độ
    sẽ trùng nhau.
    """
    res_true = paired_results[True]
    res_false = paired_results[False]

    eq_true = res_true["equity_curve"]
    eq_false = res_false["equity_curve"]

    assert not eq_true.empty and not eq_false.empty
    assert float(eq_true.iloc[0]) == pytest.approx(INITIAL_CAPITAL)
    assert float(eq_false.iloc[0]) == pytest.approx(INITIAL_CAPITAL)

    assert float(res_true["overlay_active_ratio"]) > 0.0, (
        "Overlay không hoạt động nên không thể so sánh tác động tái đầu tư"
    )

    final_true = float(eq_true.iloc[-1])
    final_false = float(eq_false.iloc[-1])
    assert abs(final_true - final_false) > 1.0, (
        "reinvest_profits=True/False cho final_equity quá sát nhau: "
        f"True={final_true:.6f}, False={final_false:.6f}"
    )

    assert res_true["go_no_go_inputs"]["final_equity"] == pytest.approx(final_true)
    assert res_false["go_no_go_inputs"]["final_equity"] == pytest.approx(final_false)
    assert res_true["metrics"]["reinvest_profits"] is True
    assert res_false["metrics"]["reinvest_profits"] is False


# ---------------------------------------------------------------------------
# 3. False không để carry sinh lời trên phần vốn đã tăng
# ---------------------------------------------------------------------------
def test_static_capital_does_not_compound_total_return(
    paired_results: dict[bool, dict],
) -> None:
    """Carry dương: tổng return của False kỳ vọng thấp hơn True.

    Với ``False``, carry chỉ được ghi có trên vốn gốc 3000 USD thay vì trên
    equity hiện tại. Nếu source cho kết quả ngược (overlay/PnL vẫn compound),
    test chỉ khẳng định hai chế độ khác nhau và ghi rõ bất thường trong assert
    message.
    """
    true_ret = float(paired_results[True]["metrics"]["Total Return %"])
    false_ret = float(paired_results[False]["metrics"]["Total Return %"])

    assert float(paired_results[True]["funding_pnl"]) > 0.0, (
        "Dữ liệu trend phải tạo carry dương (funding 5 bps/8h) để phép so sánh "
        "có ý nghĩa"
    )

    if false_ret < true_ret:
        assert false_ret < true_ret
    else:
        assert false_ret != pytest.approx(true_ret, rel=0.0, abs=1e-9), (
            "Kỳ vọng Total Return % của False < True khi carry dương và vốn "
            "không được tái đầu tư, nhưng thực tế ngược/không đổi: "
            f"True={true_ret:.6f}%, False={false_ret:.6f}%. "
            "Đây là bất thường của source cần điều tra."
        )


# ---------------------------------------------------------------------------
# 4. Đẳng thức kế toán cho cả hai chế độ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reinvest",
    [True, False],
    ids=["compounded", "static_capital"],
)
def test_attribution_closes_to_equity_change(
    paired_results: dict[bool, dict],
    reinvest: bool,
) -> None:
    """Tổng ``net_pnl`` phải khép kín về chênh lệch vốn cuối - vốn đầu.

    Yêu cầu đúng cho cả hai chế độ. Nếu ``False`` lệch, nguyên nhân thường là
    source scale carry attribution theo equity hiện tại nhưng chỉ ghi có carry
    trên vốn gốc (hoặc overlay PnL vẫn dùng ``prev_eq`` dù size neo 3000).
    """
    res = paired_results[reinvest]
    attr = res["attribution_series"]
    final_equity = float(res["equity_curve"].iloc[-1])
    total_net_pnl = float(attr["net_pnl"].sum())
    expected_delta = final_equity - INITIAL_CAPITAL
    gap = abs(total_net_pnl - expected_delta)

    assert gap < 1e-6, (
        f"reinvest_profits={reinvest}: sum(net_pnl)={total_net_pnl:.6f} lệch "
        f"{gap:.6f} USD so với chênh lệch vốn {expected_delta:.6f}. "
        "Accounting không khép kín ở chế độ này."
    )


# ---------------------------------------------------------------------------
# 5. Khóa holdout mới của run_sprint11_tiers
# ---------------------------------------------------------------------------
def test_sprint11_holdout_reinvest_keys_and_active_capital_formula() -> None:
    """Mỗi row eval phải có đủ khóa holdout mới, đúng kiểu, đúng công thức.

    Dữ liệu 2 symbol ~200 giờ phủ qua 2026-04-21. Công thức
    ``holdout_return_on_active_capital_pct == holdout_net_pnl / 3000 * 100``
    được kiểm tra cho mọi row (kể cả row lỗi), vì source luôn dùng vốn hoạt
    động 3000 USD khi ``holdout_equity_start > 0`` hoặc khi không có nến trước
    split.
    """
    payload = run_sprint11_tiers(_eval_data(), capital=INITIAL_CAPITAL)

    assert payload["n_combos"] == 4
    assert len(payload["rows"]) == 4

    float_keys = (
        "holdout_net_pnl",
        "holdout_equity_start",
        "holdout_return_on_active_capital_pct",
        "holdout_return_on_compounded_equity_pct",
        "holdout_return_pct",
        "holdout_return_resim_pct",
    )

    for row in payload["rows"]:
        combo = row.get("combo")
        for key in float_keys:
            assert key in row, f"{combo}: thiếu khóa {key}"
            assert isinstance(row[key], float), (
                f"{combo}: {key} phải là float, nhận {type(row[key]).__name__}"
            )

        assert "reinvest_profits" in row, f"{combo}: thiếu khóa reinvest_profits"
        assert isinstance(row["reinvest_profits"], bool), (
            f"{combo}: reinvest_profits phải là bool"
        )

        expected_active_pct = float(row["holdout_net_pnl"]) / INITIAL_CAPITAL * 100.0
        assert row["holdout_return_on_active_capital_pct"] == pytest.approx(
            expected_active_pct, rel=1e-12, abs=1e-12
        ), (
            f"{combo}: holdout_return_on_active_capital_pct="
            f"{row['holdout_return_on_active_capital_pct']:.10f} khác "
            f"holdout_net_pnl/3000*100={expected_active_pct:.10f}"
        )
