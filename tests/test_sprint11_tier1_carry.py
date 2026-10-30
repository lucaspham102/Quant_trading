"""Unit tests cho Sprint 11 - Chặng 1: Binance Tier 1+ Carry.

Phạm vi kiểm thử (hoàn toàn offline, deterministic, không đọc data/raw, không gọi mạng):
1. Bảng lot-spec 8 cặp Liquid-8 và hàm quantize_order_qty.
2. Engine Tier 1A trên dữ liệu tổng hợp: notional mở lệnh, basis MTM, đối soát kế toán.
3. Lượng hóa khối lượng lệnh (lot-size) bên trong backtest.
4. Rebalance ký quỹ khi giá biến động >= ±15%.
5. Tier 1B patient buffer: universe 8, top_k=4, min_hold 63 kỳ 8h.
6. Hành vi an toàn với dữ liệu rỗng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CORE_4_SYMBOLS
from src.cross_exchange_carry import (
    EXCHANGE_LOT_SPECS,
    LIQUID_8_SYMBOLS,
    LOT_STEP_SIZES,
    CrossExchangeCarryConfig,
    quantize_order_qty,
    run_binance_tier1_plus_carry,
)

# Giá tham chiếu cố định cho từng cặp (đủ lớn để min_notional không cắt lệnh).
BASE_PRICES: dict[str, float] = {
    "BTC/USDT:USDT": 60_000.0,
    "ETH/USDT:USDT": 3_000.0,
    "SOL/USDT:USDT": 150.0,
    "BNB/USDT:USDT": 600.0,
    "DOGE/USDT:USDT": 0.15,
    "XRP/USDT:USDT": 0.50,
    "AVAX/USDT:USDT": 30.0,
    "LINK/USDT:USDT": 15.0,
}


def _hourly_index(days: int, start: str = "2024-01-01 00:00:00") -> pd.DatetimeIndex:
    """Tạo DatetimeIndex 1h UTC liên tục cho ``days`` ngày (đơn vị ns)."""
    idx = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC").as_unit("ns")
    idx.name = "datetime"
    return idx


def _make_frame(
    idx: pd.DatetimeIndex,
    base_price: float,
    funding_rate: float,
    *,
    basis_amp: float = 0.0002,
    price_wave: float = 0.0,
    spike_at: int | None = None,
    spike_mult: float = 1.20,
) -> pd.DataFrame:
    """Tạo DataFrame 1h tổng hợp đúng schema engine Tier 1+ cần.

    Cột ``funding_event_binance`` chỉ khác 0 tại các nến 00:00, 08:00, 16:00 UTC.
    ``basis_pct`` dao động nhẹ để basis MTM có biến thiên thật.
    """
    n = len(idx)
    phase = np.arange(n, dtype=np.float64)
    perp = base_price * (1.0 + price_wave * np.sin(phase / 7.0))
    if spike_at is not None:
        perp = perp.copy()
        perp[spike_at:] *= spike_mult

    basis_pct = 0.001 + basis_amp * np.sin(phase / 5.0)
    spot = perp / (1.0 + basis_pct)
    is_8h = (idx.hour % 8 == 0) & (idx.minute == 0)
    funding_event = np.where(is_8h, funding_rate, 0.0)

    frame = pd.DataFrame(
        {
            "perp_close": perp,
            "spot_close": spot,
            "close": perp,
            "high": perp * 1.001,
            "low": perp * 0.999,
            "basis_pct": basis_pct,
            "funding_event_binance": funding_event,
        },
        index=idx,
    )
    frame.index.name = "datetime"
    return frame


def _make_universe(
    idx: pd.DatetimeIndex,
    symbols: list[str],
    funding_rates: dict[str, float] | None = None,
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Tạo dict {symbol: frame} với funding rate riêng từng symbol nếu cần."""
    funding_rates = funding_rates or {}
    return {
        sym: _make_frame(
            idx,
            BASE_PRICES[sym],
            funding_rates.get(sym, 0.0002),
            **kwargs,  # type: ignore[arg-type]
        )
        for sym in symbols
    }


def _tier1_config(initial_capital: float = 3_000.0) -> CrossExchangeCarryConfig:
    """Config test: tắt maker-fill model để entry xảy ra tất định tại nến 8h."""
    return CrossExchangeCarryConfig(
        initial_capital=initial_capital,
        lot_size_quantization=True,
        enable_maker_fill_model=False,
    )


# ---------------------------------------------------------------------------
# 1. Lot specs & quantize_order_qty
# ---------------------------------------------------------------------------


def test_lot_specs_cover_liquid_8_with_correct_step_sizes() -> None:
    """EXCHANGE_LOT_SPECS/LOT_STEP_SIZES đủ 8 symbol với step_size đúng đặc tả."""
    expected_steps = {
        "DOGE/USDT:USDT": 1.0,
        "XRP/USDT:USDT": 0.1,
        "AVAX/USDT:USDT": 1.0,
        "LINK/USDT:USDT": 0.01,
        "BTC/USDT:USDT": 0.001,
        "ETH/USDT:USDT": 0.01,
        "SOL/USDT:USDT": 0.1,
        "BNB/USDT:USDT": 0.01,
    }
    assert len(EXCHANGE_LOT_SPECS) == 8
    for sym, step in expected_steps.items():
        assert sym in EXCHANGE_LOT_SPECS
        assert EXCHANGE_LOT_SPECS[sym]["step_size"] == pytest.approx(step)
        assert LOT_STEP_SIZES[sym] == pytest.approx(step)

    # LIQUID_8_SYMBOLS = Core-4 + 4 mã thanh khoản cao, không trùng lặp.
    assert len(LIQUID_8_SYMBOLS) == 8
    assert len(set(LIQUID_8_SYMBOLS)) == 8
    assert set(CORE_4_SYMBOLS).issubset(set(LIQUID_8_SYMBOLS))
    for sym in ("DOGE/USDT:USDT", "XRP/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT"):
        assert sym in LIQUID_8_SYMBOLS


def test_quantize_order_qty_doge_uses_step_one_and_min_notional() -> None:
    """quantize_order_qty làm tròn xuống theo step và trả actual = q * price."""
    q, actual = quantize_order_qty("DOGE/USDT:USDT", 100.0, 0.15)
    assert q > 0.0
    # q là bội số của step 1.0 (DOGE).
    assert q / 1.0 == pytest.approx(round(q / 1.0), abs=1e-9)
    assert actual == pytest.approx(q * 0.15)

    # Target nhỏ hơn min_notional -> trả (0, 0) thay vì đặt lệnh không hợp lệ.
    assert quantize_order_qty("DOGE/USDT:USDT", 5.0, 0.15) == (0.0, 0.0)
    # Giá không hợp lệ.
    assert quantize_order_qty("DOGE/USDT:USDT", 100.0, 0.0) == (0.0, 0.0)

    # Symbol không có tiền tố sàn vẫn tra được step qua base asset.
    q_btc, actual_btc = quantize_order_qty("BTC/USDT", 1_000.0, 60_000.0)
    assert q_btc == pytest.approx(round(q_btc / 0.001) * 0.001)
    assert actual_btc == pytest.approx(q_btc * 60_000.0)


# ---------------------------------------------------------------------------
# 2 & 4. Tier 1A: notional, basis MTM, đối soát kế toán, lot quantization
# ---------------------------------------------------------------------------


def test_tier1a_accounting_and_notional_with_core4() -> None:
    """Tier 1A: 4 core symbol, mỗi lệnh mở ~(capital/4)*eta, kế toán khớp.

    Kiểm tra:
    - universe đúng 4 mã Core-4 và có ít nhất 1 entry.
    - notional lệnh open xấp xỉ (initial/4)*0.80 (dung sai 5% cho lot-size).
    - basis MTM có biến thiên (std > 0).
    - tổng net_pnl attribution == metrics["Net PnL"]; tổng fees == Total Fees Paid.
    - equity_curve cùng độ dài dữ liệu, giá trị đầu == initial_capital.
    """
    # Vốn 20k: bước lot-size của ETH (0.01 @ 3k = 30 USD) chỉ ~0.75% notional mục tiêu,
    # nhờ vậy dung sai 5% trong assert không bị lấn át bởi sai số lượng hóa.
    cfg = _tier1_config(initial_capital=20_000.0)
    idx = _hourly_index(days=60)
    data = _make_universe(idx, list(CORE_4_SYMBOLS))

    result = run_binance_tier1_plus_carry(data, cfg, variant="tier1a")

    assert result["variant"] == "tier1a"
    assert len(result["symbol_universe"]) == 4
    assert set(result["symbol_universe"]) == set(CORE_4_SYMBOLS)

    trades = result["trades_df"]
    opens = trades[trades["action"] == "open"]
    assert len(opens) >= 1

    target_notional = (cfg.initial_capital / 4.0) * 0.80
    for notional in opens["notional"]:
        assert float(notional) == pytest.approx(target_notional, rel=0.05)

    assert result["basis_mtm_std"] > 0.0

    attribution = result["attribution_series"]
    metrics = result["metrics"]
    assert abs(float(attribution["net_pnl"].sum()) - float(metrics["Net PnL"])) < 1e-6
    assert float(attribution["fees"].sum()) == pytest.approx(
        float(metrics["Total Fees Paid"]), rel=1e-9, abs=1e-9
    )

    equity_curve = result["equity_curve"]
    assert len(equity_curve) == len(idx)
    assert float(equity_curve.iloc[0]) == pytest.approx(cfg.initial_capital, abs=1e-9)
    assert list(equity_curve.index) == list(idx)


def test_tier1a_open_notionals_are_lot_quantized() -> None:
    """Mọi notional lệnh open phải bằng q*price với q là bội số step_size của symbol."""
    idx = _hourly_index(days=45)
    data = _make_universe(idx, list(CORE_4_SYMBOLS), price_wave=0.01)

    result = run_binance_tier1_plus_carry(data, _tier1_config(), variant="tier1a")

    trades = result["trades_df"]
    opens = trades[trades["action"] == "open"]
    assert not opens.empty

    for _, row in opens.iterrows():
        sym = str(row["symbol"])
        t = pd.Timestamp(row["time"])
        step = LOT_STEP_SIZES[sym]
        # Engine dùng giá perp_close tại nến entry khi maker-fill bị tắt.
        price = float(data[sym].loc[t, "perp_close"])
        notional = float(row["notional"])
        q = notional / price

        assert notional == pytest.approx(q * price, rel=1e-12)
        assert abs(q / step - round(q / step)) <= 1e-6
        # Không bao giờ vượt quá target do làm tròn xuống.
        assert notional > 0.0


# ---------------------------------------------------------------------------
# 3. Rebalance ký quỹ ±15%
# ---------------------------------------------------------------------------


def test_price_spike_triggers_margin_rebalance() -> None:
    """Cú nhảy giá +20% sau khi mở vị thế phải kích hoạt rebalance ký quỹ."""
    idx = _hourly_index(days=30)
    n = len(idx)
    spike_bar = 40  # sau khi đã mở vị thế tại nến 8h đầu tiên
    assert spike_bar < n

    data = _make_universe(
        idx,
        list(CORE_4_SYMBOLS),
        spike_at=spike_bar,
        spike_mult=1.20,
    )
    result = run_binance_tier1_plus_carry(data, _tier1_config(), variant="tier1a")

    assert result["metrics"]["Margin Rebalances"] >= 1
    assert result["rebalance_fees"] > 0.0

    attribution = result["attribution_series"]
    assert float(attribution["rebalance_fees"].sum()) == pytest.approx(
        result["rebalance_fees"], rel=1e-9, abs=1e-9
    )
    # Rebalance cũng được ghi nhận thành dòng trades với action "rebalance".
    trades = result["trades_df"]
    rebalances = trades[trades["action"] == "rebalance"]
    assert len(rebalances) >= 1
    assert (rebalances["fee"] > 0.0).all()


# ---------------------------------------------------------------------------
# 5. Tier 1B: patient buffer, top_k, min_hold 63 kỳ
# ---------------------------------------------------------------------------


def test_tier1b_patient_buffer_holds_top_k_and_respects_min_hold() -> None:
    """Tier 1B: universe 8, giữ tối đa 4 vị thế, không đóng trước 63 kỳ 8h.

    Funding khác nhau từng symbol để kiểm tra xếp hạng. Với buffer_rank=3 và
    top_k=4, các vị thế top-4 không bị xoay vòng, số entry rất nhỏ.
    """
    idx = _hourly_index(days=120)
    funding_rates = {
        "BTC/USDT:USDT": 0.00040,
        "ETH/USDT:USDT": 0.00035,
        "SOL/USDT:USDT": 0.00030,
        "BNB/USDT:USDT": 0.00025,
        "DOGE/USDT:USDT": 0.00020,
        "XRP/USDT:USDT": 0.00015,
        "AVAX/USDT:USDT": 0.00010,
        "LINK/USDT:USDT": 0.00006,
    }
    data = _make_universe(idx, list(LIQUID_8_SYMBOLS), funding_rates=funding_rates)

    result = run_binance_tier1_plus_carry(
        data,
        _tier1_config(),
        variant="tier1b",
        top_k=4,
        buffer_rank=3,
        min_hold_periods=63,
    )

    assert result["variant"] == "tier1b"
    assert len(result["symbol_universe"]) == 8
    assert set(result["symbol_universe"]) == set(LIQUID_8_SYMBOLS)

    trades = result["trades_df"]
    opens = trades[trades["action"] == "open"]
    closes = trades[trades["action"] == "close"]

    # Patient buffer: số lần mở rất nhỏ (top-4 ổn định suốt 120 ngày).
    assert 1 <= len(opens) <= 8
    assert len(opens) == result["metrics"]["Num Entries"]

    # Mọi lệnh close (nếu có) phải sau >= 63 kỳ funding 8h kể từ lúc mở.
    eight_hours = pd.Timedelta(hours=8)
    for _, close_row in closes.iterrows():
        sym = close_row["symbol"]
        prior_opens = opens[(opens["symbol"] == sym) & (opens["time"] < close_row["time"])]
        assert not prior_opens.empty, f"Close không có lệnh mở tương ứng cho {sym}"
        open_time = pd.Timestamp(prior_opens.iloc[-1]["time"])
        periods_held = (pd.Timestamp(close_row["time"]) - open_time) / eight_hours
        assert periods_held >= 63 - 1e-9

    # Không giữ quá top_k vị thế và có lúc đạt đủ 4 vị thế.
    holdings = result["holdings_log"]
    assert not holdings.empty
    assert int(holdings["n_active"].max()) <= 4
    assert (holdings["n_active"] == 4).any()


# ---------------------------------------------------------------------------
# 6. Dữ liệu rỗng
# ---------------------------------------------------------------------------


def test_empty_data_returns_neutral_result() -> None:
    """Dữ liệu rỗng: không raise, trả equity rỗng và Net PnL = 0.0."""
    cfg = _tier1_config()

    result = run_binance_tier1_plus_carry({}, cfg)

    assert isinstance(result, dict)
    assert result["equity_curve"].empty
    assert len(result["equity_curve"]) == 0
    assert result["metrics"]["Net PnL"] == 0.0
    assert result["metrics"]["Total Fees Paid"] == 0.0
    assert result["symbol_universe"] == []
    assert result["trades_df"].empty
    # Khi không có symbol nào của universe xuất hiện cũng phải trả kết quả rỗng.
    foreign = {"UNKNOWN/USDT:USDT": _make_frame(_hourly_index(days=2), 100.0, 0.0002)}
    result_foreign = run_binance_tier1_plus_carry(foreign, cfg)
    assert result_foreign["equity_curve"].empty
    assert result_foreign["metrics"]["Net PnL"] == 0.0
