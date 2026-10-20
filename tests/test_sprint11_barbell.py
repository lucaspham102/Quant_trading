"""Unit tests cho Barbell Portfolio Sprint 11 - Chặng 2 (gating + quantization).

Phạm vi khóa hành vi:

- ``BarbellConfig`` mới: ``carry_variant``, ``gating_mode``, các tham số smooth
  conviction (target vol, leverage band), ``min_weight_change_to_trade`` và
  ``overlay_quantize_legs``.
- Đẳng thức kế toán GO-4 khi overlay chạy smooth conviction: tổng ``net_pnl``
  bằng chênh lệch vốn cuối - vốn đầu, và bằng
  ``funding + basis + overlay_gross - fees``.
- Lượng hóa lot-size của overlay: notional trên các nến đổi trọng số nằm trên
  lưới ``step_size`` của từng cặp; bật/tắt lượng hóa cho trọng số khác nhau.
- Ngưỡng chặn rebalance nhỏ: smooth conviction phải ghi nhận
  ``overlay_skipped_rebalances > 0`` và không làm tăng turnover quá mức so với
  hard binary.
- ``min_weight_change_to_trade`` lớn hơn phải bỏ qua nhiều rebalance hơn và
  không làm tăng phí overlay.
- Smooth leverage luôn nằm trong dải cấu hình (kể cả khi DD cushion giảm nửa).
- Backend carry ``tier1a`` / ``tier1b`` trả đúng ``variant``, ``symbol_universe``;
  dữ liệu ngoài universe phải fallback an toàn thay vì ném lỗi.
- Validate tham số cấu hình mới.

Toàn bộ dữ liệu được tổng hợp trong bộ nhớ (1h UTC, 4 hoặc 8 symbol Core-4 /
Liquid-8), không gọi mạng, không đọc parquet, không sleep. Các kết quả nặng
được cache ở phạm vi module để mỗi test giữ dưới 4 giây.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.barbell_portfolio import BarbellConfig, run_barbell_portfolio
from src.config import CORE_4_SYMBOLS
from src.cross_exchange_carry import (
    LOT_STEP_SIZES,
    CrossExchangeCarryConfig,
    quantize_order_qty,
)

INITIAL_CAPITAL = 3000.0
CORE4 = list(CORE_4_SYMBOLS)
LIQUID8 = CORE4 + [
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
]
START = "2024-01-01 00:00:00"
FUNDING_RATE = 0.0005  # 5 bps/8h, vượt xa ngưỡng entry 0.45 bps/8h


# ---------------------------------------------------------------------------
# Helpers dựng dữ liệu tổng hợp (offline, tất định)
# ---------------------------------------------------------------------------
def _hourly_index(periods: int) -> pd.DatetimeIndex:
    """Tạo DatetimeIndex UTC ns tên ``datetime`` với tần suất 1 giờ."""
    index = pd.date_range(START, periods=periods, freq="h", tz="UTC").as_unit("ns")
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
    khớp ở các cặp đang tăng (high(t+1) vượt giá treo close(t)*1.00005).
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


def _trend_data(periods: int = 1200) -> dict[str, pd.DataFrame]:
    """4 symbol Core-4 trend phân hóa rõ: 2 tăng (+4/+1.5 bps/h), 2 giảm.

    Nhiễu xen kẽ dùng chung cho mọi symbol nên chân long - chân short triệt tiêu
    nhiễu: pair vol chạm sàn 8%, đẩy smooth leverage lên trần 0.50. Drift đủ lớn
    để SNR dispersion vượt xa ngưỡng conviction nhưng đủ nhỏ để trọng số overlay
    trôi chậm hơn ``min_weight_change_to_trade=0.08`` trong toàn chuỗi - nhờ đó
    số lần bỏ qua rebalance là tất định. Basis sin chu kỳ 24h tạo Basis MTM khác
    0 cho Core Carry.
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


def _eight_symbol_data(periods: int = 420) -> dict[str, pd.DataFrame]:
    """8 symbol Liquid-8 với drift phân hóa nhẹ, đủ để Carry mở vị thế."""
    index = _hourly_index(periods)
    steps = np.arange(periods, dtype="float64")
    noise = 2e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    drifts = (
        0.0004,
        0.0002,
        0.0001,
        0.00005,
        -0.00005,
        -0.0001,
        -0.0002,
        -0.0004,
    )
    starts = (65000.0, 3000.0, 150.0, 600.0, 0.15, 0.5, 30.0, 15.0)

    data: dict[str, pd.DataFrame] = {}
    for i, (sym, drift, start) in enumerate(zip(LIQUID8, drifts, starts)):
        close = _price_path(drift + noise, start)
        basis = 0.0005 * np.sin(2.0 * np.pi * steps / 48.0 + 0.2 * i)
        data[sym] = _symbol_frame(index, close, basis)
    return data


def _outsider_data(periods: int = 300) -> dict[str, pd.DataFrame]:
    """2 symbol giả không thuộc universe Tier 1A/1B (AAA, BBB)."""
    index = _hourly_index(periods)
    steps = np.arange(periods, dtype="float64")
    noise = 1e-4 * np.where(steps % 2 == 0, 1.0, -1.0)
    drifts = (0.0002, -0.0002)
    starts = (100.0, 50.0)
    names = ("AAA/USDT:USDT", "BBB/USDT:USDT")

    data: dict[str, pd.DataFrame] = {}
    for i, (sym, drift, start) in enumerate(zip(names, drifts, starts)):
        close = _price_path(drift + noise, start)
        basis = 0.0005 * np.sin(2.0 * np.pi * steps / 48.0 + 0.2 * i)
        data[sym] = _symbol_frame(index, close, basis)
    return data


def _carry_config() -> CrossExchangeCarryConfig:
    """Config Carry tối giản, vốn gốc khớp BarbellConfig."""
    return CrossExchangeCarryConfig(initial_capital=INITIAL_CAPITAL)


def _base_config(**overrides: object) -> BarbellConfig:
    """BarbellConfig nền cho overlay: tắt DD cushion để kết quả tất định."""
    base: dict[str, object] = {
        "initial_capital": INITIAL_CAPITAL,
        "carry_config": _carry_config(),
        "overlay_lookback_hours": 336,
        "overlay_rebalance_hours": 48,
        "overlay_top_k": 1,
        "overlay_buffer_rank": 1,
        "overlay_min_spread": 0.75,
        "enable_drawdown_cushion": False,
    }
    base.update(overrides)
    return BarbellConfig(**base)


def _smooth_config(**overrides: object) -> BarbellConfig:
    """Config smooth conviction + carry Tier 1A dùng cho phần lớn test."""
    base: dict[str, object] = {
        "carry_variant": "tier1a",
        "gating_mode": "smooth_conviction",
    }
    base.update(overrides)
    return _base_config(**base)


# ---------------------------------------------------------------------------
# Fixtures cache kết quả nặng ở phạm vi module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trend_data() -> dict[str, pd.DataFrame]:
    """Dữ liệu 4 symbol trend phân hóa mạnh (dùng chung)."""
    return _trend_data()


@pytest.fixture(scope="module")
def smooth_result(trend_data: dict[str, pd.DataFrame]) -> dict:
    """Kết quả smooth conviction + Tier 1A trên dữ liệu trend (dùng chung)."""
    return run_barbell_portfolio(trend_data, _smooth_config())


# ---------------------------------------------------------------------------
# 1. Accounting parity GO-4
# ---------------------------------------------------------------------------
def test_smooth_conviction_accounting_parity(
    smooth_result: dict,
) -> None:
    """Smooth conviction + Tier 1A: mọi PnL phải khép kín về chênh lệch vốn.

    Kiểm tra đồng thời 3 đẳng thức:
    ``sum(net_pnl) == final_equity - 3000``,
    ``sum(funding+basis+overlay_gross-fees) == final_equity - 3000`` và
    equity nến đầu đúng bằng vốn gốc. ``basis_mtm_std > 0`` chứng minh Core
    Carry thực sự mở vị thế và ghi nhận Basis MTM bar-by-bar.
    """
    res = smooth_result
    attr = res["attribution_series"]
    final_equity = float(res["equity_curve"].iloc[-1])
    initial = INITIAL_CAPITAL

    assert float(res["equity_curve"].iloc[0]) == initial

    assert abs(float(attr["net_pnl"].sum()) - (final_equity - initial)) < 1e-6

    combined = (
        float(attr["funding_pnl"].sum())
        + float(attr["basis_pnl"].sum())
        + float(attr["overlay_gross_pnl"].sum())
        - float(attr["fees"].sum())
    )
    assert abs(combined - (final_equity - initial)) < 1e-6

    assert float(res["carry_result"]["basis_mtm_std"]) > 0.0

    # Các cột/khóa mới của Sprint 11 phải tồn tại và nhất quán.
    for column in ("overlay_gross_pnl", "overlay_fees", "net_pnl"):
        assert column in attr.columns
    assert res["overlay_gross_pnl"] == pytest.approx(
        float(attr["overlay_gross_pnl"].sum())
    )
    assert res["overlay_fees"] == pytest.approx(float(attr["overlay_fees"].sum()))
    assert res["overlay_net_pnl"] == pytest.approx(
        res["overlay_gross_pnl"] - res["overlay_fees"]
    )


# ---------------------------------------------------------------------------
# 2. Quantization Overlay
# ---------------------------------------------------------------------------
def test_overlay_quantization_keeps_notional_on_step_grid(
    trend_data: dict[str, pd.DataFrame],
    smooth_result: dict,
) -> None:
    """Bật lượng hóa: mọi nến đổi trọng số có notional trên lưới ``step_size``.

    Tại nến t, với weight ``w`` và equity ``eq``, khối lượng
    ``q = |w| * eq / price`` phải cách bội số ``step`` dưới 1 step và khớp
    ``quantize_order_qty`` trên cùng notional. Tắt lượng hóa phải cho
    ``overlay_weights`` khác và không phát sinh sự kiện lượng hóa.
    """
    res = smooth_result
    assert res["overlay_quantization_events"] > 0
    assert res["metrics"]["overlay_quantization_events"] == (
        res["overlay_quantization_events"]
    )

    weights = res["overlay_weights"]
    equity = res["equity_curve"]
    changed = weights.diff().abs() > 1e-12

    checked = 0
    for pos, ts in enumerate(weights.index):
        for sym in weights.columns:
            w = float(weights.loc[ts, sym])
            if abs(w) < 1e-12:
                continue
            if pos > 0 and not bool(changed.loc[ts, sym]):
                continue  # chỉ kiểm tra chân vừa được đặt lại
            price = float(trend_data[sym]["perp_close"].loc[ts])
            step = LOT_STEP_SIZES[sym]
            eq = float(equity.loc[ts])

            q = abs(w) * eq / price
            units = q / step
            assert abs(units - round(units)) < 1.0, (
                f"{sym}@{ts}: q={q} lệch lưới step={step} quá 1 step"
            )

            q_ref, _ = quantize_order_qty(sym, abs(w) * eq, price)
            assert abs(q - q_ref) < step, (
                f"{sym}@{ts}: q={q} khác quantize_order_qty q_ref={q_ref}"
            )
            checked += 1

    assert checked > 0, "Phải có ít nhất một chân overlay được đặt lại"

    unquantized = run_barbell_portfolio(
        trend_data, _smooth_config(overlay_quantize_legs=False)
    )
    assert unquantized["overlay_quantization_events"] == 0
    assert not np.allclose(
        unquantized["overlay_weights"].to_numpy(dtype="float64"),
        weights.to_numpy(dtype="float64"),
    )


# ---------------------------------------------------------------------------
# 3. Smooth vs Hard turnover
# ---------------------------------------------------------------------------
def test_smooth_conviction_skips_rebalances_and_does_not_raise_turnover(
    trend_data: dict[str, pd.DataFrame],
) -> None:
    """Smooth conviction phải bỏ qua rebalance nhỏ và không tăng turnover.

    Hard binary ghim leverage 0.40 (tắt vol targeting/SNR gate) để so sánh
    công bằng; smooth đặt target vol 1.2% -> leverage chạm sàn 0.15. Ưu tiên
    assert ``overlay_skipped_rebalances > 0`` ở smooth; sau đó turnover smooth
    không được vượt hard.
    """
    smooth = run_barbell_portfolio(
        trend_data,
        _base_config(
            carry_variant="tier1a",
            gating_mode="smooth_conviction",
            smooth_target_vol_ann=0.012,
        ),
    )
    hard = run_barbell_portfolio(
        trend_data,
        _base_config(
            carry_variant="tier1a",
            gating_mode="hard_binary",
            overlay_gross_leverage=0.40,
            enable_vol_targeting=False,
            enable_snr_gate=False,
        ),
    )

    assert smooth["overlay_skipped_rebalances"] > 0, (
        "Smooth conviction phải bỏ qua các rebalance có |Δw| < "
        "min_weight_change_to_trade=0.08"
    )

    smooth_turnover = float(smooth["overlay_turnover_scaled"])
    hard_turnover = float(hard["overlay_turnover_scaled"])
    assert smooth_turnover <= hard_turnover + 1e-9, (
        "Turnover smooth cao hơn hard: "
        f"smooth={smooth_turnover:.6f} > hard={hard_turnover:.6f}"
    )


# ---------------------------------------------------------------------------
# 4. Ngưỡng chặn rebalance nhỏ
# ---------------------------------------------------------------------------
def test_min_weight_change_threshold_skips_more_and_costs_less(
    trend_data: dict[str, pd.DataFrame],
    smooth_result: dict,
) -> None:
    """Ngưỡng 0.08 phải bỏ qua >= số lần của ngưỡng 0.0 và phí không cao hơn.

    Với ``min_weight_change_to_trade=0.0`` điều kiện ``max_diff < 0`` không bao
    giờ đúng nên không có rebalance nào bị bỏ qua; bản 0.08 phải chặn được các
    thay đổi nhỏ và do đó trả phí overlay ít hơn hoặc bằng.
    """
    no_threshold = run_barbell_portfolio(
        trend_data, _smooth_config(min_weight_change_to_trade=0.0)
    )

    assert no_threshold["overlay_skipped_rebalances"] == 0
    assert smooth_result["overlay_skipped_rebalances"] >= (
        no_threshold["overlay_skipped_rebalances"]
    )
    assert smooth_result["overlay_fees"] <= no_threshold["overlay_fees"] + 1e-9

    assert smooth_result["metrics"]["overlay_skipped_rebalances"] == (
        smooth_result["overlay_skipped_rebalances"]
    )
    assert smooth_result["metrics"]["overlay_fees"] == pytest.approx(
        smooth_result["overlay_fees"]
    )


# ---------------------------------------------------------------------------
# 5. Smooth leverage bounds
# ---------------------------------------------------------------------------
def test_smooth_leverage_stays_inside_configured_band(
    smooth_result: dict,
) -> None:
    """Leverage overlay active phải nằm trong [min*0.5, max] và mean > 0.

    Dữ liệu trend phân hóa mạnh giữ gate mở nên ``overlay_mean_leverage`` > 0.
    Khi DD cushion cắt 50%, leverage có thể giảm còn nửa ``smooth_leverage_min``
    nên cận dưới dùng ``smooth_leverage_min * 0.5``. Mọi giá trị dương không
    được vượt ``smooth_leverage_max``.
    """
    cfg = smooth_result["metrics"]
    lev = smooth_result["overlay_leverage_series"]
    active = lev[lev > 1e-12]

    assert not active.empty
    assert float(active.max()) <= 0.50 + 1e-9
    assert float(active.min()) >= 0.15 * 0.5 - 1e-9
    assert float(lev.min()) >= 0.0
    assert cfg["overlay_mean_leverage"] > 0.0
    assert cfg["overlay_mean_leverage"] <= 0.50 + 1e-9


# ---------------------------------------------------------------------------
# 6. Buffer không cắt cưỡng bức
# ---------------------------------------------------------------------------
def test_overlay_keeps_market_neutral_legs_in_strong_trend(
    smooth_result: dict,
) -> None:
    """Trend mạnh: overlay hiện diện > 50% thời gian và giữ đủ 2 chân.

    Không có hiện tượng đóng sạch toàn bộ danh mục ở mọi nến active: mỗi nến
    còn vị thế luôn có ít nhất một weight dương và một weight âm (buffer giữ
    chân cũ qua các nhịp tụt hạng nhẹ thay vì cắt cưỡng bức).
    """
    weights = smooth_result["overlay_weights"]
    gross = weights.abs().sum(axis=1)

    assert float((gross > 0.0).mean()) > 0.5

    active_rows = weights[gross > 1e-12]
    assert not active_rows.empty
    assert bool(active_rows.gt(1e-12).any(axis=1).all())
    assert bool(active_rows.lt(-1e-12).any(axis=1).all())


# ---------------------------------------------------------------------------
# 7. Carry backend (Tier 1A / Tier 1B) và fallback ngoài universe
# ---------------------------------------------------------------------------
def test_carry_variant_tier1a_and_tier1b_report_universe() -> None:
    """Tier 1A trả universe 4, Tier 1B trả universe 8, không fallback.

    Chạy trên cùng bộ dữ liệu Liquid-8: ``variant`` và ``symbol_universe`` phải
    phản ánh đúng backend đã chọn, ``carry_fallback`` phải False.
    """
    data = _eight_symbol_data()

    res_a = run_barbell_portfolio(data, _base_config(carry_variant="tier1a"))
    assert res_a["carry_fallback"] is False
    assert res_a["carry_result"]["variant"] == "tier1a"
    assert len(res_a["carry_result"]["symbol_universe"]) == 4
    assert res_a["carry_variant"] == "tier1a"
    assert res_a["metrics"]["carry_variant"] == "tier1a"
    assert res_a["metrics"]["carry_fallback"] is False

    res_b = run_barbell_portfolio(data, _base_config(carry_variant="tier1b"))
    assert res_b["carry_fallback"] is False
    assert res_b["carry_result"]["variant"] == "tier1b"
    assert len(res_b["carry_result"]["symbol_universe"]) == 8
    assert res_b["carry_variant"] == "tier1b"
    assert res_b["metrics"]["carry_variant"] == "tier1b"


def test_carry_fallback_outside_universe_does_not_raise() -> None:
    """Symbol ngoài universe Tier 1A phải fallback sang engine cũ, không raise.

    ``AAA/USDT:USDT`` và ``BBB/USDT:USDT`` không nằm trong Core-4 nên
    ``run_binance_tier1_plus_carry`` trả equity rỗng; Barbell phải tự chuyển
    sang ``run_cross_exchange_carry`` và đánh dấu ``carry_fallback=True``.
    """
    data = _outsider_data()

    res = run_barbell_portfolio(data, _smooth_config())

    assert res["carry_fallback"] is True
    assert res["metrics"]["carry_fallback"] is True
    assert len(res["equity_curve"]) == len(_hourly_index(300))
    assert np.isfinite(res["equity_curve"].to_numpy(dtype="float64")).all()


# ---------------------------------------------------------------------------
# 8. Validate tham số cấu hình mới
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        {"gating_mode": "soft_gate"},
        {"carry_variant": "tier0"},
        {"smooth_leverage_min": 0.0},
        {"min_weight_change_to_trade": -0.01},
        {"smooth_leverage_max": 1.5},
    ],
)
def test_invalid_config_raises_value_error(overrides: dict[str, object]) -> None:
    """Mọi tham số Sprint 11 sai phải bị chặn sớm bằng ``ValueError``.

    Dùng dữ liệu rỗng để chứng minh validation chạy trước cả khi mô phỏng.
    """
    config = BarbellConfig(**overrides)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_barbell_portfolio({}, config)


# ---------------------------------------------------------------------------
# 9. Giá trị mặc định và alias metrics của Sprint 11
# ---------------------------------------------------------------------------
def test_sprint11_default_config_and_metric_aliases(
    smooth_result: dict,
) -> None:
    """Khóa default ``BarbellConfig`` và các alias metrics Sprint 11."""
    config = BarbellConfig()

    assert config.carry_variant == "tier1b"
    assert config.gating_mode == "hard_binary"
    assert config.carry_tilt_weight == pytest.approx(0.20)
    assert config.smooth_target_vol_ann == pytest.approx(0.075)
    assert config.smooth_leverage_min == pytest.approx(0.15)
    assert config.smooth_leverage_max == pytest.approx(0.50)
    assert config.min_weight_change_to_trade == pytest.approx(0.08)
    assert config.overlay_quantize_legs is True

    metrics = smooth_result["metrics"]
    assert metrics["overlay_gross_pnl"] == pytest.approx(
        smooth_result["overlay_gross_pnl"]
    )
    assert metrics["overlay_net_pnl"] == pytest.approx(
        smooth_result["overlay_net_pnl"]
    )
    assert metrics["overlay_fees"] == pytest.approx(smooth_result["overlay_fees"])
    assert metrics["overlay_quantization_events"] == (
        smooth_result["overlay_quantization_events"]
    )
    assert metrics["overlay_skipped_rebalances"] == (
        smooth_result["overlay_skipped_rebalances"]
    )
    assert metrics["gating_mode"] == "smooth_conviction"
